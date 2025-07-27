# app/services/pure_llm_intent_parser.py
"""
Pure LLM Intent Parser - No Pattern Matching
Uses LLM to understand user intent and extract devices, then lets your existing
Dynamic Discovery handle command mapping and Universal Formatter handle output
"""

import json
import logging
import re
from typing import Dict, List, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class PureLLMIntentParser:
    """
    Pure LLM-based intent parser that replaces pattern matching entirely
    Understands ANY user request and extracts intent + devices for your existing pipeline
    """

    def __init__(self, llm_url: str = "http://192.168.1.11:1234"):
        self.llm_url = llm_url
        self.timeout = 10.0

        logger.info("🧠 Pure LLM Intent Parser initialized - no pattern matching!")

    async def extract_intent(self, user_input: str) -> str:
        """
        Use LLM to extract the user's intent from ANY natural language input
        Returns a clean intent string for your Dynamic Discovery system
        """
        logger.info(f"🤖 LLM extracting intent from: '{user_input}'")

        try:
            # Build prompt for intent extraction
            prompt = self._build_intent_extraction_prompt(user_input)

            # Get LLM response
            llm_response = await self._call_llm(prompt)

            # Parse intent from response
            intent = self._extract_intent_from_response(llm_response)

            logger.info(f"✅ LLM extracted intent: '{intent}'")
            return intent

        except Exception as e:
            logger.error(f"❌ LLM intent extraction failed: {e}")
            # Simple fallback - just use the user input as intent
            # Your Dynamic Discovery can handle this
            return user_input

    async def extract_devices(
        self, user_input: str, default_devices: List[str] = None
    ) -> List[str]:
        """
        Use LLM to extract device names from ANY natural language input
        """
        logger.info(f"📍 LLM extracting devices from: '{user_input}'")

        try:
            # Build prompt for device extraction
            prompt = self._build_device_extraction_prompt(user_input)

            # Get LLM response - NOW ASYNC!
            llm_response = await self._call_llm(prompt)

            # Parse devices from response
            devices = self._extract_devices_from_response(llm_response)

            if devices:
                logger.info(f"✅ LLM extracted devices: {devices}")
                return devices
            else:
                logger.info("📍 No devices found, using defaults")
                return default_devices or ["all"]

        except Exception as e:
            logger.error(f"❌ LLM device extraction failed: {e}")
            # Fallback to defaults
            return default_devices or ["all"]

    async def parse(
        self, user_input: str, default_devices: List[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Complete parsing: extract both intent and devices using LLM
        """
        intent = await self.extract_intent(user_input)
        devices = await self.extract_devices(user_input, default_devices)
        return intent, devices

    def _build_intent_extraction_prompt(self, user_input: str) -> str:
        """Build prompt for LLM to understand user intent"""

        prompt = f"""You are a network automation assistant. A user has made this request:

"{user_input}"

Your job is to understand what the user wants and describe their intent in simple, clear terms.

EXAMPLES:
- "can you show me the interface statistics for spine1, leaf1 and leaf2?" → "show interface statistics"
- "what's the cpu usage on all devices?" → "show cpu usage"
- "please check bgp status" → "show bgp status"
- "get me the version info" → "show version"
- "how are the interfaces doing?" → "show interface status"
- "check power consumption" → "show power status"
- "what's running on spine1?" → "show version"
- "tell me about memory usage" → "show memory usage"

INSTRUCTIONS:
1. Focus on WHAT the user wants to see/check/get
2. Use simple, clear language
3. Don't worry about specific commands - just the intent
4. If unclear, make your best guess
5. Keep it concise - 2-4 words usually

Respond with just the intent phrase, nothing else:"""

        return prompt

    def _build_device_extraction_prompt(self, user_input: str) -> str:
        """Build prompt for LLM to extract device names"""

        prompt = f"""Extract device names from this network request:

"{user_input}"

DEVICE PATTERNS TO LOOK FOR:
- spine1, spine2, leaf1, leaf2, core1, etc.
- "all devices", "everything", "all spines", "all leafs"
- Multiple devices: "spine1, leaf1 and leaf2"

RULES:
- If user says "all devices" or similar → return ["all"]
- If specific devices mentioned → return the device names
- If no devices mentioned → return ["all"]
- Clean up device names (lowercase, remove extra spaces)

EXAMPLES:
- "show version on spine1" → ["spine1"]
- "spine1, leaf1 and leaf2" → ["spine1", "leaf1", "leaf2"]
- "check all devices" → ["all"]
- "what's the cpu usage?" → ["all"]

Respond with JSON array only:
["device1", "device2"] or ["all"]

JSON:"""

        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM asynchronously"""

        payload = {
            "model": "llama-3.1-8b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise network automation assistant. Follow instructions exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 100,  # Short responses
            "stop": ["\n\n"],
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.llm_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    raise Exception(f"LLM API error: {response.status}")

                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()

    def _extract_intent_from_response(self, llm_response: str) -> str:
        """Extract clean intent from LLM response"""

        # Clean up the response
        intent = llm_response.strip()

        # Remove quotes if present
        intent = intent.strip("\"'")

        # Remove any extra explanations
        if "\n" in intent:
            intent = intent.split("\n")[0]

        # Clean up common prefixes/suffixes
        intent = intent.replace("Intent:", "").replace("User wants to:", "").strip()

        return intent

    def _extract_devices_from_response(self, llm_response: str) -> List[str]:
        """Extract device list from LLM JSON response"""

        try:
            # Try to find JSON array in response
            json_match = re.search(r"\[.*?\]", llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = llm_response

            # Parse JSON
            devices = json.loads(json_str)

            if isinstance(devices, list):
                # Clean device names
                cleaned_devices = []
                for device in devices:
                    if isinstance(device, str):
                        cleaned = device.strip().lower()
                        if cleaned:
                            cleaned_devices.append(cleaned)

                return cleaned_devices if cleaned_devices else ["all"]
            else:
                return ["all"]

        except Exception as e:
            logger.debug(f"Failed to parse device JSON: {e}")
            return ["all"]


# Enhanced Intent Parser that uses LLM instead of patterns
class EnhancedIntentParser:
    """
    Drop-in replacement for your existing intent parser
    Uses pure LLM understanding instead of pattern matching
    """

    def __init__(self):
        # Pure LLM parser only - no pattern-based fallback
        self.llm_parser = PureLLMIntentParser()

        logger.info("✅ Enhanced Intent Parser initialized with pure LLM understanding")

    async def extract_intent(self, user_input: str) -> str:
        """Extract intent using pure LLM understanding"""

        try:
            # Use LLM parser
            intent = await self.llm_parser.extract_intent(user_input)
            if intent and intent.strip():
                return intent
        except Exception as e:
            logger.warning(f"LLM intent extraction failed: {e}")

        # Return the raw input as a fallback
        return user_input

    async def extract_devices(
        self, user_input: str, default_devices: List[str] = None
    ) -> List[str]:
        """Extract devices using LLM first, fallback to base parser"""

        try:
            # Try LLM first - NOW ASYNC!
            devices = await self.llm_parser.extract_devices(user_input, default_devices)
            if devices:
                return devices
        except Exception as e:
            logger.warning(f"LLM device extraction failed: {e}")

        # Return default devices as fallback
        return default_devices or []

    async def parse(
        self, user_input: str, default_devices: List[str] = None
    ) -> Tuple[str, List[str]]:
        """Complete parsing using LLM"""
        intent = await self.extract_intent(user_input)
        devices = await self.extract_devices(user_input, default_devices)
        return intent, devices

    # Delegate other methods to base parser
    def get_supported_operations(self) -> Dict[str, str]:
        return [
            "show version",
            "show interfaces",
            "show running-config",
            "show ip route",
            "show bgp summary",
            "show vlan",
            "and any other network command via LLM understanding",
        ]

    def generate_help_message(self) -> str:
        return (
            "I understand natural language requests for network operations. "
            "Just describe what you want to do, and I'll help you execute it."
        )


# Factory function
def create_enhanced_intent_parser() -> EnhancedIntentParser:
    """Create enhanced intent parser with pure LLM understanding"""
    return EnhancedIntentParser()


# Global instance for compatibility
enhanced_intent_parser = create_enhanced_intent_parser()
