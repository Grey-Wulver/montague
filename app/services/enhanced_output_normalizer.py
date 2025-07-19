# app/services/enhanced_output_normalizer.py

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.models.intents import NetworkIntent
from app.services.local_llm_normalizer import local_llm_normalizer

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result from output normalization attempt"""

    success: bool
    method_used: str  # "manual", "llm", "fallback"
    normalized_data: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    raw_output_preview: Optional[str] = None


class EnhancedOutputNormalizer:
    """
    Hybrid output normalizer that uses:
    1. Fast manual parsers for critical intents (version, interfaces)
    2. Local LLM for all other intents and fallback
    3. Caching to avoid repeat LLM processing
    4. Graceful degradation when LLM fails
    """

    def __init__(self):
        self.llm_available = False
        self._normalization_cache = {}  # Cache LLM results
        self.stats = {"manual_normalizations": 0, "llm_normalizations": 0, "cache_hits": 0, "fallback_to_raw": 0}

    async def initialize(self) -> bool:
        """Initialize normalizer and check LLM availability"""
        self.llm_available = await local_llm_normalizer.health_check()
        if self.llm_available:
            logger.info("Local LLM is available - hybrid normalization enabled")
        else:
            logger.warning("Local LLM not available - using manual parsers only")
        return self.llm_available

    async def normalize_output(
        self, raw_output: str, intent: NetworkIntent, vendor: str, command: str, device_name: str
    ) -> NormalizationResult:
        """
        Main normalization entry point with hybrid strategy
        """
        start_time = time.time()

        # Generate cache key for LLM results
        cache_key = self._generate_cache_key(raw_output, intent, vendor)

        # Check cache first for LLM results
        if cache_key in self._normalization_cache:
            self.stats["cache_hits"] += 1
            cached_result = self._normalization_cache[cache_key]
            return NormalizationResult(
                success=True,
                method_used="cache",
                normalized_data=cached_result,
                processing_time=time.time() - start_time,
                confidence_score=0.95,  # High confidence for cached results
            )

        # Strategy 1: Use fast manual parsers for critical intents
        if intent in [NetworkIntent.GET_SYSTEM_VERSION, NetworkIntent.GET_INTERFACE_STATUS]:
            try:
                result = await self._manual_normalize(raw_output, intent, vendor, device_name)
                if result.success:
                    self.stats["manual_normalizations"] += 1
                    return result
            except Exception as e:
                logger.warning(f"Manual parser failed for {intent}: {e}")

        # Strategy 2: Use LLM for other intents or as fallback
        if self.llm_available:
            try:
                llm_result = await local_llm_normalizer.normalize_output(raw_output, intent, vendor, command)

                if llm_result.success:
                    self.stats["llm_normalizations"] += 1

                    # Cache successful LLM results
                    self._normalization_cache[cache_key] = llm_result.normalized_data

                    return NormalizationResult(
                        success=True,
                        method_used="llm",
                        normalized_data=llm_result.normalized_data,
                        processing_time=llm_result.processing_time,
                        confidence_score=llm_result.confidence_score,
                    )
                else:
                    logger.warning(f"LLM normalization failed: {llm_result.error_message}")
            except Exception as e:
                logger.error(f"LLM normalization error: {e}")

        # Strategy 3: Fallback to raw output with basic structure
        self.stats["fallback_to_raw"] += 1
        return self._fallback_normalization(raw_output, intent, device_name, start_time)

    async def _manual_normalize(
        self, raw_output: str, intent: NetworkIntent, vendor: str, device_name: str
    ) -> NormalizationResult:
        """Fast manual parsers for critical intents"""

        start_time = time.time()

        try:
            if intent == NetworkIntent.GET_SYSTEM_VERSION:
                normalized_data = await self._parse_version_manual(raw_output, vendor)
            elif intent == NetworkIntent.GET_INTERFACE_STATUS:
                normalized_data = await self._parse_interfaces_manual(raw_output, vendor)
            else:
                raise ValueError(f"Manual parser not available for {intent}")

            return NormalizationResult(
                success=True,
                method_used="manual",
                normalized_data=normalized_data,
                processing_time=time.time() - start_time,
                confidence_score=0.95,  # High confidence for manual parsers
            )

        except Exception as e:
            return NormalizationResult(
                success=False, method_used="manual", processing_time=time.time() - start_time, error_message=str(e)
            )

    async def _parse_version_manual(self, raw_output: str, vendor: str) -> Dict[str, Any]:
        """Manual parser for version information (your existing logic)"""

        lines = raw_output.strip().split("\n")
        result = {
            "vendor": vendor.replace("_", " ").title(),
            "hostname": None,
            "os_version": None,
            "model": None,
            "serial_number": None,
            "uptime_seconds": None,
            "memory_total_kb": None,
            "memory_free_kb": None,
            "last_reload_reason": None,
        }

        # Arista EOS parsing logic (extend for other vendors)
        if vendor == "arista_eos":
            for line in lines:
                line = line.strip()

                if "Software image version:" in line:
                    result["os_version"] = line.split(":")[-1].strip()
                elif "Serial number:" in line:
                    result["serial_number"] = line.split(":")[-1].strip()
                elif "Hardware version:" in line and not result["model"]:
                    result["model"] = line.split(":")[-1].strip()
                elif "Total memory:" in line:
                    memory_str = line.split(":")[-1].strip().replace(" kB", "")
                    try:
                        result["memory_total_kb"] = int(memory_str)
                    except ValueError:
                        pass
                elif "Free memory:" in line:
                    memory_str = line.split(":")[-1].strip().replace(" kB", "")
                    try:
                        result["memory_free_kb"] = int(memory_str)
                    except ValueError:
                        pass
                elif "Uptime:" in line:
                    # Parse uptime to seconds (basic implementation)
                    uptime_str = line.split(":")[-1].strip()
                    result["uptime_seconds"] = self._parse_uptime_to_seconds(uptime_str)

        return result

    async def _parse_interfaces_manual(self, raw_output: str, vendor: str) -> Dict[str, Any]:
        """Manual parser for interface status (your existing logic)"""

        lines = raw_output.strip().split("\n")
        interfaces = []

        # Skip header lines and parse data
        # data_started = False
        for line in lines:
            line = line.strip()

            # Skip empty lines and headers
            if not line or "Port" in line or "Name" in line or "---" in line:
                continue

            # Arista EOS interface parsing
            if vendor == "arista_eos":
                parts = line.split()
                if len(parts) >= 3:
                    interface = {
                        "interface": parts[0],
                        "status": "up" if "connected" in line else "down",
                        "protocol": "up" if "connected" in line else "down",
                        "description": None,
                        "speed": None,
                        "duplex": None,
                        "mtu": None,
                        "ip_address": None,
                        "subnet_mask": None,
                        "vlan": None,
                    }

                    # Parse additional fields if available
                    if len(parts) >= 4:
                        try:
                            interface["vlan"] = int(parts[3]) if parts[3].isdigit() else None
                        except Exception:
                            pass

                    if len(parts) >= 6:
                        interface["speed"] = parts[5]

                    interfaces.append(interface)

        # Calculate summary statistics
        total_interfaces = len(interfaces)
        up_interfaces = len([i for i in interfaces if i["status"] == "up"])
        down_interfaces = total_interfaces - up_interfaces

        return {
            "interfaces": interfaces,
            "total_interfaces": total_interfaces,
            "up_interfaces": up_interfaces,
            "down_interfaces": down_interfaces,
        }

    def _parse_uptime_to_seconds(self, uptime_str: str) -> Optional[int]:
        """Convert uptime string to seconds"""
        try:
            # Basic parsing for "X hours and Y minutes"
            total_seconds = 0
            if "hour" in uptime_str:
                hours = int(uptime_str.split("hour")[0].strip().split()[-1])
                total_seconds += hours * 3600
            if "minute" in uptime_str:
                minutes = int(uptime_str.split("minute")[0].strip().split()[-1])
                total_seconds += minutes * 60
            return total_seconds if total_seconds > 0 else None
        except Exception:
            return None

    def _fallback_normalization(
        self, raw_output: str, intent: NetworkIntent, device_name: str, start_time: float
    ) -> NormalizationResult:
        """Fallback to basic structure when all else fails"""

        fallback_data = {
            "device": device_name,
            "intent": intent.value,
            "raw_output": raw_output[:500] + "..." if len(raw_output) > 500 else raw_output,
            "normalized": False,
            "timestamp": time.time(),
            "note": "Raw output - normalization not available",
        }

        return NormalizationResult(
            success=True,  # Still "successful" as we return data
            method_used="fallback",
            normalized_data=fallback_data,
            processing_time=time.time() - start_time,
            confidence_score=0.1,  # Low confidence for raw output
            raw_output_preview=raw_output[:200],
        )

    def _generate_cache_key(self, raw_output: str, intent: NetworkIntent, vendor: str) -> str:
        """Generate cache key for LLM results"""
        import hashlib

        content = f"{intent.value}_{vendor}_{raw_output}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics"""
        total_normalizations = sum(self.stats.values())

        result = {
            **self.stats,
            "total_normalizations": total_normalizations,
            "llm_available": self.llm_available,
            "cache_size": len(self._normalization_cache),
        }

        if total_normalizations > 0:
            result["manual_percentage"] = round((self.stats["manual_normalizations"] / total_normalizations) * 100, 2)
            result["llm_percentage"] = round((self.stats["llm_normalizations"] / total_normalizations) * 100, 2)
            result["cache_hit_rate"] = round((self.stats["cache_hits"] / total_normalizations) * 100, 2)

        # Include LLM stats if available
        if self.llm_available:
            result["llm_stats"] = local_llm_normalizer.get_stats()

        return result

    def clear_cache(self) -> int:
        """Clear normalization cache and return number of entries cleared"""
        cache_size = len(self._normalization_cache)
        self._normalization_cache.clear()
        return cache_size


# Global instance
enhanced_normalizer = EnhancedOutputNormalizer()
