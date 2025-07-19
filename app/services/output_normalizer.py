import importlib.util
import logging
import re
from typing import Dict, List, Optional

from app.models.intents import NetworkIntent
from app.models.schemas import (
    StandardARPTable,
    StandardBGPSummary,
    StandardInterfaceStatus,
    StandardizedOutput,
    StandardVersionInfo,
)

logger = logging.getLogger(__name__)

# Check for optional dependencies at module level
GENIE_AVAILABLE = importlib.util.find_spec("genie") is not None
TEXTFSM_AVAILABLE = importlib.util.find_spec("textfsm") is not None


class MockDevice:
    """Mock device for Genie parsing"""

    def __init__(self, platform: str):
        self.os = platform
        self.platform = platform


class OutputNormalizer:
    """Normalize raw command output to structured, vendor-agnostic data"""

    def __init__(self):
        self.genie_available = GENIE_AVAILABLE
        self.textfsm_available = TEXTFSM_AVAILABLE
        logger.info(f"OutputNormalizer initialized - Genie: {self.genie_available}, TextFSM: {self.textfsm_available}")

    def normalize(
        self, intent: NetworkIntent, raw_output: str, device_platform: str, command: str
    ) -> Optional[StandardizedOutput]:
        """Convert raw output to normalized format"""

        if not raw_output or not raw_output.strip():
            logger.warning("Empty or None raw output provided")
            return None

        logger.info(f"Normalizing {intent} output for {device_platform}")

        # Try Genie first (highest quality for supported platforms)
        if self.genie_available:
            genie_result = self._try_genie_parse(raw_output, device_platform, command)
            if genie_result:
                logger.info("Successfully parsed with Genie")
                return self._normalize_genie_output(intent, genie_result, device_platform)

        # Fallback to TextFSM
        if self.textfsm_available:
            textfsm_result = self._try_textfsm_parse(raw_output, device_platform, command)
            if textfsm_result:
                logger.info("Successfully parsed with TextFSM")
                return self._normalize_textfsm_output(intent, textfsm_result, device_platform)

        # Final fallback: regex-based parsing
        logger.info("Falling back to regex-based parsing")
        return self._try_regex_parse(intent, raw_output, device_platform)

    def _try_genie_parse(self, raw_output: str, platform: str, command: str) -> Optional[Dict]:
        """Try parsing with Genie"""
        try:
            from genie.libs.parser.utils import get_parser

            # Map our platform names to Genie platform names
            genie_platform_map = {
                "arista_eos": "eos",
                "cisco_ios": "ios",
                "cisco_nxos": "nxos",
                "cisco_xr": "iosxr",
                "juniper_junos": "junos",
            }

            genie_platform = genie_platform_map.get(platform)
            if not genie_platform:
                logger.debug(f"Platform {platform} not supported by Genie")
                return None

            parser_class = get_parser(command, genie_platform)
            if parser_class:
                mock_device = MockDevice(genie_platform)
                parser = parser_class(device=mock_device)
                result = parser.parse(output=raw_output)
                logger.debug(f"Genie parsed result: {type(result)}")
                return result
            else:
                logger.debug(f"No Genie parser found for command: {command}")
                return None

        except Exception as e:
            logger.debug(f"Genie parsing failed: {e}")
            return None

    def _try_textfsm_parse(self, raw_output: str, platform: str, command: str) -> Optional[List[Dict]]:
        """Try parsing with TextFSM"""
        try:
            from ntc_templates.parse import parse_output

            result = parse_output(platform=platform, command=command, data=raw_output)

            if result:
                logger.debug(f"TextFSM parsed {len(result)} entries")
                return result
            else:
                logger.debug("TextFSM returned empty result")
                return None

        except Exception as e:
            logger.debug(f"TextFSM parsing failed: {e}")
            return None

    def _normalize_genie_output(self, intent: NetworkIntent, genie_data: Dict, platform: str) -> StandardizedOutput:
        """Convert Genie output to standardized format"""

        if intent == NetworkIntent.GET_SYSTEM_VERSION:
            return self._normalize_version_genie(genie_data, platform)
        elif intent == NetworkIntent.GET_INTERFACE_STATUS:
            return self._normalize_interface_genie(genie_data, platform)
        elif intent == NetworkIntent.GET_BGP_SUMMARY:
            return self._normalize_bgp_genie(genie_data, platform)
        elif intent == NetworkIntent.GET_ARP_TABLE:
            return self._normalize_arp_genie(genie_data, platform)

        # Return raw Genie data if no specific normalizer
        logger.info(f"No specific normalizer for intent {intent}, returning raw Genie data")
        return genie_data

    def _normalize_textfsm_output(
        self, intent: NetworkIntent, textfsm_data: List[Dict], platform: str
    ) -> StandardizedOutput:
        """Convert TextFSM output to standardized format"""

        if intent == NetworkIntent.GET_SYSTEM_VERSION:
            return self._normalize_version_textfsm(textfsm_data, platform)
        elif intent == NetworkIntent.GET_INTERFACE_STATUS:
            return self._normalize_interface_textfsm(textfsm_data, platform)
        elif intent == NetworkIntent.GET_BGP_SUMMARY:
            return self._normalize_bgp_textfsm(textfsm_data, platform)
        elif intent == NetworkIntent.GET_ARP_TABLE:
            return self._normalize_arp_textfsm(textfsm_data, platform)

        # Return raw TextFSM data if no specific normalizer
        logger.info(f"No specific normalizer for intent {intent}, returning raw TextFSM data")
        return {"parsed_data": textfsm_data}

    def _normalize_version_genie(self, data: Dict, platform: str) -> StandardVersionInfo:
        """Normalize version output from Genie data"""

        # Genie version data structure varies by platform
        version_data = data.get("version", {})

        # Extract vendor from platform
        vendor = platform.split("_")[0] if "_" in platform else platform

        # Common fields extraction
        hostname = version_data.get("hostname")
        os_version = version_data.get("version") or version_data.get("software_version")
        model = version_data.get("chassis") or version_data.get("platform") or version_data.get("hardware")
        serial = version_data.get("chassis_sn") or version_data.get("serial_number")

        # Parse uptime
        uptime_seconds = None
        uptime_str = version_data.get("uptime")
        if uptime_str:
            uptime_seconds = self._parse_uptime_to_seconds(uptime_str)

        # Memory information
        memory_total = None
        memory_free = None
        memory_info = version_data.get("memory", {})
        if memory_info:
            memory_total = memory_info.get("total")
            memory_free = memory_info.get("free")

        return StandardVersionInfo(
            hostname=hostname,
            vendor=vendor,
            os_version=os_version,
            model=model,
            serial_number=serial,
            uptime_seconds=uptime_seconds,
            memory_total_kb=memory_total,
            memory_free_kb=memory_free,
        )

    def _normalize_version_textfsm(self, data: List[Dict], platform: str) -> StandardVersionInfo:
        """Normalize version output from TextFSM data"""

        if not data:
            return StandardVersionInfo()

        # TextFSM typically returns a list, take first entry
        version_data = data[0] if isinstance(data, list) else data

        vendor = platform.split("_")[0] if "_" in platform else platform

        return StandardVersionInfo(
            hostname=version_data.get("hostname"),
            vendor=vendor,
            os_version=version_data.get("version") or version_data.get("software"),
            model=version_data.get("hardware") or version_data.get("platform"),
            serial_number=version_data.get("serial"),
            uptime_seconds=self._parse_uptime_to_seconds(version_data.get("uptime"))
            if version_data.get("uptime")
            else None,
        )

    def _try_regex_parse(self, intent: NetworkIntent, raw_output: str, platform: str) -> Optional[StandardizedOutput]:
        """Fallback regex-based parsing for when Genie/TextFSM fail"""

        if intent == NetworkIntent.GET_SYSTEM_VERSION:
            return self._regex_parse_version(raw_output, platform)

        logger.info(f"No regex parser available for intent {intent}")
        return None

    def _regex_parse_version(self, raw_output: str, platform: str) -> StandardVersionInfo:
        """Regex-based version parsing as fallback"""

        vendor = platform.split("_")[0] if "_" in platform else platform

        # Common regex patterns
        version_patterns = {
            "arista": r"Software image version:\s+(.+?)(?:\s|\n)",
            "cisco": r"(?:Cisco IOS|IOS-XE|NX-OS).*?Version\s+(.+?)(?:\s|\n|,)",
            "juniper": r"Junos:\s+(.+?)(?:\s|\n)",
        }

        serial_patterns = {
            "arista": r"Serial number:\s+(\S+)",
            "cisco": r"Processor board ID\s+(\S+)",
            "juniper": r"Serial number:\s+(\S+)",
        }

        uptime_patterns = {
            "arista": r"Uptime:\s+(.+?)(?:\n|$)",
            "cisco": r"uptime is\s+(.+?)(?:\n|$)",
            "juniper": r"System booted:\s+(.+?)(?:\n|$)",
        }

        # Memory patterns
        memory_patterns = {
            "arista": {"total": r"Total memory:\s+(\d+)\s+kB", "free": r"Free memory:\s+(\d+)\s+kB"},
            "cisco": {"total": r"with (\d+)K bytes of memory", "free": r"(\d+)K bytes of.*?memory available"},
        }

        # Model patterns
        model_patterns = {
            "arista": r"^(.+?)(?:\n|$)",  # First line for Arista
            "cisco": r"(cisco \S+)",
            "juniper": r"Model:\s+(.+?)(?:\n|$)",
        }

        # Extract version
        os_version = None
        if vendor in version_patterns:
            match = re.search(version_patterns[vendor], raw_output, re.IGNORECASE)
            if match:
                os_version = match.group(1).strip()

        # Extract serial number
        serial_number = None
        if vendor in serial_patterns:
            match = re.search(serial_patterns[vendor], raw_output, re.IGNORECASE)
            if match:
                serial_number = match.group(1).strip()

        # Extract uptime
        uptime_seconds = None
        if vendor in uptime_patterns:
            match = re.search(uptime_patterns[vendor], raw_output, re.IGNORECASE)
            if match:
                uptime_str = match.group(1).strip()
                uptime_seconds = self._parse_uptime_to_seconds(uptime_str)

        # Extract memory information
        memory_total_kb = None
        memory_free_kb = None
        if vendor in memory_patterns:
            # Total memory
            if "total" in memory_patterns[vendor]:
                match = re.search(memory_patterns[vendor]["total"], raw_output, re.IGNORECASE)
                if match:
                    memory_total_kb = int(match.group(1))

            # Free memory
            if "free" in memory_patterns[vendor]:
                match = re.search(memory_patterns[vendor]["free"], raw_output, re.IGNORECASE)
                if match:
                    memory_free_kb = int(match.group(1))

        # Extract model (first line for Arista)
        model = None
        if vendor == "arista":
            first_line = raw_output.split("\n")[0].strip()
            if first_line:
                model = first_line
        elif vendor in model_patterns:
            match = re.search(model_patterns[vendor], raw_output, re.IGNORECASE)
            if match:
                model = match.group(1).strip()

        return StandardVersionInfo(
            vendor=vendor,
            os_version=os_version,
            model=model,
            serial_number=serial_number,
            uptime_seconds=uptime_seconds,
            memory_total_kb=memory_total_kb,
            memory_free_kb=memory_free_kb,
        )

    def _parse_uptime_to_seconds(self, uptime_str: str) -> Optional[int]:
        """Parse various uptime formats to seconds"""
        if not uptime_str:
            return None

        try:
            # Handle different uptime formats
            uptime_str = uptime_str.lower().strip()

            # Format: "1 hour and 58 minutes"
            if "hour" in uptime_str and "minute" in uptime_str:
                hours = re.search(r"(\d+)\s+hour", uptime_str)
                minutes = re.search(r"(\d+)\s+minute", uptime_str)
                total_seconds = 0
                if hours:
                    total_seconds += int(hours.group(1)) * 3600
                if minutes:
                    total_seconds += int(minutes.group(1)) * 60
                return total_seconds

            # Format: "2d4h" or "2 days, 4 hours"
            days = re.search(r"(\d+)\s*d", uptime_str)
            hours = re.search(r"(\d+)\s*h", uptime_str)
            minutes = re.search(r"(\d+)\s*m", uptime_str)

            total_seconds = 0
            if days:
                total_seconds += int(days.group(1)) * 86400
            if hours:
                total_seconds += int(hours.group(1)) * 3600
            if minutes:
                total_seconds += int(minutes.group(1)) * 60

            return total_seconds if total_seconds > 0 else None

        except Exception as e:
            logger.debug(f"Failed to parse uptime '{uptime_str}': {e}")
            return None

    # TODO: Implement other normalizers for interfaces, BGP, ARP
    def _normalize_interface_genie(self, data: Dict, platform: str) -> StandardInterfaceStatus:
        """Normalize interface output from Genie data"""
        # Implementation placeholder
        return StandardInterfaceStatus()

    def _normalize_bgp_genie(self, data: Dict, platform: str) -> StandardBGPSummary:
        """Normalize BGP output from Genie data"""
        # Implementation placeholder
        return StandardBGPSummary()

    def _normalize_arp_genie(self, data: Dict, platform: str) -> StandardARPTable:
        """Normalize ARP output from Genie data"""
        # Implementation placeholder
        return StandardARPTable()

    def _normalize_interface_textfsm(self, data: List[Dict], platform: str) -> StandardInterfaceStatus:
        """Normalize interface output from TextFSM data"""
        # Implementation placeholder
        return StandardInterfaceStatus()

    def _normalize_bgp_textfsm(self, data: List[Dict], platform: str) -> StandardBGPSummary:
        """Normalize BGP output from TextFSM data"""
        # Implementation placeholder
        return StandardBGPSummary()

    def _normalize_arp_textfsm(self, data: List[Dict], platform: str) -> StandardARPTable:
        """Normalize ARP output from TextFSM data"""
        # Implementation placeholder
        return StandardARPTable()


# Global normalizer instance
output_normalizer = OutputNormalizer()
