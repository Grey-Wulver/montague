# app/services/dynamic_command_discovery.py

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import aiohttp

from app.models.devices import Device

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result from dynamic command discovery"""

    success: bool
    intent: Optional[str] = None
    command: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
    safety_approved: bool = False
    processing_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class CommandMapping:
    """Cached command mapping entry"""

    user_intent: str
    platform: str
    command: str
    confidence: float
    usage_count: int
    created_at: float
    last_used: float


class DynamicCommandDiscovery:
    """
    Discovers network commands for unknown intents using LLM intelligence.
    Integrates with existing LLM infrastructure and caches successful mappings.
    """

    def __init__(
        self,
        llm_normalizer,
        ollama_url: str = "http://192.168.1.11:1234",
        model_name: str = "meta-llama-3.1-8b-instruct",
        timeout: int = 30,
    ):
        self.llm_normalizer = llm_normalizer
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout = timeout

        # Cache for discovered command mappings
        self._discovery_cache: Dict[str, CommandMapping] = {}

        # Safety patterns - block dangerous commands
        self.unsafe_patterns = [
            r"reload",
            r"reboot",
            r"shutdown",
            r"reset",
            r"erase",
            r"delete",
            r"remove",
            r"clear config",
            r"factory-reset",
            r"write erase",
            r"format",
            r"copy.*start.*run",
            r"no ",
            r"undo",
            r"rollback",
        ]

        # Known vendor command patterns for validation
        self.vendor_patterns = {
            "arista_eos": {
                "show_commands": [r"^show ", r"^bash ", r"^enable"],
                "config_commands": [r"^configure", r"^interface", r"^router"],
            },
            "cisco_ios": {
                "show_commands": [r"^show ", r"^ping ", r"^traceroute"],
                "config_commands": [r"^configure", r"^interface", r"^router"],
            },
            "juniper_junos": {
                "show_commands": [r"^show ", r"^ping ", r"^traceroute"],
                "config_commands": [r"^configure", r"^set ", r"^edit"],
            },
        }

        # Statistics
        self.stats = {
            "total_discoveries": 0,
            "successful_discoveries": 0,
            "cache_hits": 0,
            "safety_blocks": 0,
            "average_discovery_time": 0.0,
        }

    async def discover_command(
        self, user_intent: str, device: Device, allow_cached: bool = True
    ) -> DiscoveryResult:
        """
        Main entry point for discovering network commands from user intent
        """
        start_time = time.time()
        self.stats["total_discoveries"] += 1

        logger.info(
            f"Starting command discovery for intent: '{user_intent}' on platform: {device.platform}"
        )

        # Check cache first
        if allow_cached:
            cached_result = self._check_discovery_cache(user_intent, device.platform)
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.info(
                    f"Cache hit for intent '{user_intent}' on {device.platform}"
                )

                processing_time = time.time() - start_time
                return DiscoveryResult(
                    success=True,
                    intent=user_intent,
                    command=cached_result.command,
                    confidence=cached_result.confidence,
                    reasoning=f"Cached mapping (used {cached_result.usage_count} times)",
                    safety_approved=True,  # Already validated when cached
                    processing_time=processing_time,
                )

        # Use LLM to discover command
        discovery_result = await self._llm_command_discovery(user_intent, device)

        processing_time = time.time() - start_time
        discovery_result.processing_time = processing_time

        # Cache successful discoveries
        if discovery_result.success and discovery_result.safety_approved:
            self._cache_discovery(user_intent, device.platform, discovery_result)
            self.stats["successful_discoveries"] += 1
            self._update_average_discovery_time(processing_time)

        logger.info(
            f"Command discovery completed in {processing_time:.2f}s - Success: {discovery_result.success}"
        )
        return discovery_result

    async def _llm_command_discovery(
        self, user_intent: str, device: Device
    ) -> DiscoveryResult:
        """Use LLM to discover appropriate network command"""

        # Generate discovery prompt
        prompt = self._generate_discovery_prompt(user_intent, device)

        try:
            # Call LLM for command discovery
            response = await self._call_lm_studio_discovery(prompt)

            if not response:
                return DiscoveryResult(
                    success=False, error_message="LLM did not return valid response"
                )

            # Parse LLM response
            intent = response.get("intent", user_intent)
            command = response.get("command", "").strip()
            confidence = response.get("confidence", 0.0)
            reasoning = response.get("reasoning", "")

            if not command:
                return DiscoveryResult(
                    success=False, error_message="LLM did not suggest a command"
                )

            # Safety validation
            safety_check = self._validate_command_safety(command, device.platform)
            if not safety_check[0]:
                self.stats["safety_blocks"] += 1
                return DiscoveryResult(
                    success=False,
                    intent=intent,
                    command=command,
                    confidence=confidence,
                    reasoning=reasoning,
                    safety_approved=False,
                    error_message=f"Command blocked for safety: {safety_check[1]}",
                )

            # Vendor validation
            vendor_check = self._validate_vendor_command(command, device.platform)
            if not vendor_check[0]:
                return DiscoveryResult(
                    success=False,
                    intent=intent,
                    command=command,
                    confidence=confidence * 0.5,  # Reduce confidence
                    reasoning=reasoning,
                    safety_approved=True,
                    error_message=f"Command may not be valid for {device.platform}: {vendor_check[1]}",
                )

            return DiscoveryResult(
                success=True,
                intent=intent,
                command=command,
                confidence=confidence,
                reasoning=reasoning,
                safety_approved=True,
            )

        except Exception as e:
            logger.error(f"LLM command discovery failed: {e}")
            return DiscoveryResult(
                success=False, error_message=f"Discovery failed: {str(e)}"
            )

    def _generate_discovery_prompt(self, user_intent: str, device: Device) -> str:
        """Generate LLM prompt for command discovery"""

        vendor_examples = {
            "arista_eos": {
                "examples": [
                    "show version - displays system information",
                    "show interfaces status - shows interface status",
                    "show ip bgp summary - displays BGP neighbors",
                    "show mac address-table - shows MAC addresses",
                ]
            },
            "cisco_ios": {
                "examples": [
                    "show version - displays system information",
                    "show ip interface brief - shows interface status",
                    "show ip bgp summary - displays BGP neighbors",
                    "show mac address-table - shows MAC addresses",
                ]
            },
        }

        examples = vendor_examples.get(device.platform, vendor_examples["cisco_ios"])

        return f"""You are a network automation expert. A user wants to "{user_intent}" on a {device.vendor} {device.platform} device.

TASK: Determine the correct CLI command to execute.

PLATFORM: {device.platform}
VENDOR: {device.vendor}

COMMON {device.platform.upper()} COMMANDS:
{chr(10).join(examples["examples"])}

SAFETY RULES:
- ONLY suggest "show" or read-only commands
- NEVER suggest configuration changes
- NEVER suggest reload, reboot, or destructive commands
- If user intent requires configuration, suggest "Not supported - read-only commands only"

USER REQUEST: "{user_intent}"

Analyze what the user wants and suggest the most appropriate CLI command for this platform.

RESPONSE FORMAT (JSON only, no explanations):
{{
    "intent": "normalized_intent_name",
    "command": "exact CLI command",
    "confidence": 0.95,
    "reasoning": "why this command was chosen"
}}

EXAMPLES:
- "show serial number" → "show version" (serial is in version output)
- "check power status" → "show power" or "show environment power"
- "display cpu usage" → "show processes cpu" or "show cpu"
- "get interface counters" → "show interfaces counters"

JSON:"""

    async def _call_lm_studio_discovery(self, prompt: str) -> Optional[Dict]:
        """Call LM Studio API for command discovery"""

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 500,
            "stream": False,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.ollama_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        choices = result.get("choices", [])

                        if choices:
                            response_text = (
                                choices[0].get("message", {}).get("content", "").strip()
                            )
                            logger.debug(f"Discovery LLM response: {response_text}")

                            # Parse JSON response
                            try:
                                cleaned_response = self._clean_discovery_json(
                                    response_text
                                )
                                parsed = json.loads(cleaned_response)
                                logger.info("Successfully parsed discovery response")
                                return parsed
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON in discovery response: {e}")
                                logger.debug(f"Problematic response: {response_text}")
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"LM Studio discovery API error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Discovery LLM call failed: {e}")

        return None

    def _clean_discovery_json(self, response_text: str) -> str:
        """Clean JSON response from discovery LLM call"""
        # Reuse the existing JSON cleaning logic
        return self.llm_normalizer._clean_json_response(response_text)

    def _validate_command_safety(self, command: str, platform: str) -> Tuple[bool, str]:
        """Validate that command is safe to execute"""
        import re

        command_lower = command.lower()

        # Check against unsafe patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, command_lower):
                return False, f"Command contains unsafe pattern: {pattern}"

        # Ensure it's a read-only command
        safe_prefixes = ["show ", "display ", "get ", "ping ", "traceroute ", "bash "]
        if not any(command_lower.startswith(prefix) for prefix in safe_prefixes):
            return False, "Command does not start with safe read-only prefix"

        return True, "Command passed safety validation"

    def _validate_vendor_command(self, command: str, platform: str) -> Tuple[bool, str]:
        """Validate command looks appropriate for the vendor platform"""
        import re

        if platform not in self.vendor_patterns:
            return True, "Platform validation patterns not defined"

        patterns = self.vendor_patterns[platform]
        command_lower = command.lower()

        # Check if it matches expected show command patterns
        show_patterns = patterns.get("show_commands", [])
        for pattern in show_patterns:
            if re.search(pattern, command_lower):
                return True, f"Command matches expected pattern for {platform}"

        return False, f"Command format may not be valid for {platform}"

    def _check_discovery_cache(
        self, user_intent: str, platform: str
    ) -> Optional[CommandMapping]:
        """Check if we have a cached mapping for this intent/platform"""
        cache_key = f"{user_intent.lower()}_{platform}"

        if cache_key in self._discovery_cache:
            mapping = self._discovery_cache[cache_key]
            mapping.usage_count += 1
            mapping.last_used = time.time()
            return mapping

        return None

    def _cache_discovery(
        self, user_intent: str, platform: str, result: DiscoveryResult
    ):
        """Cache successful command discovery"""
        cache_key = f"{user_intent.lower()}_{platform}"

        mapping = CommandMapping(
            user_intent=user_intent,
            platform=platform,
            command=result.command,
            confidence=result.confidence,
            usage_count=1,
            created_at=time.time(),
            last_used=time.time(),
        )

        self._discovery_cache[cache_key] = mapping
        logger.info(
            f"Cached discovery: '{user_intent}' → '{result.command}' for {platform}"
        )

    def _update_average_discovery_time(self, new_time: float):
        """Update running average of discovery times"""
        if self.stats["successful_discoveries"] == 1:
            self.stats["average_discovery_time"] = new_time
        else:
            current_avg = self.stats["average_discovery_time"]
            total_successful = self.stats["successful_discoveries"]
            self.stats["average_discovery_time"] = (
                current_avg * (total_successful - 1) + new_time
            ) / total_successful

    def get_cache_stats(self) -> Dict:
        """Get discovery cache statistics"""
        total_mappings = len(self._discovery_cache)
        total_usage = sum(
            mapping.usage_count for mapping in self._discovery_cache.values()
        )

        return {
            "cached_mappings": total_mappings,
            "total_cache_usage": total_usage,
            "cache_entries": [
                {
                    "intent": mapping.user_intent,
                    "platform": mapping.platform,
                    "command": mapping.command,
                    "confidence": mapping.confidence,
                    "usage_count": mapping.usage_count,
                    "age_hours": (time.time() - mapping.created_at) / 3600,
                }
                for mapping in self._discovery_cache.values()
            ],
        }

    def get_discovery_stats(self) -> Dict:
        """Get overall discovery statistics"""
        success_rate = 0.0
        if self.stats["total_discoveries"] > 0:
            success_rate = (
                self.stats["successful_discoveries"] / self.stats["total_discoveries"]
            ) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "cache_size": len(self._discovery_cache),
        }

    def clear_cache(self):
        """Clear the discovery cache"""
        cleared_count = len(self._discovery_cache)
        self._discovery_cache.clear()
        logger.info(f"Cleared discovery cache ({cleared_count} entries)")
        return cleared_count


# Global instance - will be initialized in main.py with LLM normalizer
dynamic_discovery = None
