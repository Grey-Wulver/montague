"""
Natural Language Processing for chat commands
Integrates with your existing LLM normalizer for advanced processing
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParsedCommand:
    """Represents a parsed chat command"""

    intent: str
    devices: List[str]
    parameters: Dict
    confidence: float
    method: str  # "pattern" or "llm"
    raw_message: str


class ChatNLP:
    """Natural Language Processing for chat interactions"""

    def __init__(self, llm_normalizer=None):
        self.llm_normalizer = llm_normalizer

        # Pattern-based intent recognition (fast path)
        self.intent_patterns = {
            "get_system_version": [  # CHANGED from 'system_version'
                r"(?:show|get|check).+version",
                r"what.+version",
                r"software.+version",
                r"system.+info",
            ],
            "get_interface_status": [  # CHANGED from 'interfaces_status'
                r"(?:show|get|list|check).+interfaces?",
                r"interface.+(?:status|state)",
                r"port.+(?:status|state)",
                r"link.+status",
            ],
            "get_bgp_summary": [  # CHANGED from 'routing_bgp_summary'
                r"(?:show|get|check).+bgp",
                r"bgp.+(?:summary|neighbors?|status)",
                r"routing.+(?:table|protocol)",
                r"neighbor.+(?:status|state)",
            ],
            "get_arp_table": [  # CHANGED from 'network_arp_table'
                r"(?:show|get|list).+arp",
                r"arp.+(?:table|entries)",
                r"mac.+(?:table|address)",
                r"address.+table",
            ],
            "get_system_logs": [  # CHANGED from 'system_logs'
                r"(?:show|get|check).+logs?",
                r"error.+messages?",
                r"system.+messages?",
                r"syslog",
            ],
            "help": [
                r"^help$",
                r"commands?",
                r"usage",
                r"what.+(?:can|do)",
                r"instructions?",
            ],
        }

        # Device name patterns
        self.device_patterns = [
            r"\b(?:spine|leaf|core|edge|border|access)\d+\b",  # spine1, leaf2, etc.
            r"\b[a-zA-Z]+-[a-zA-Z]+-\d+\b",  # device-role-01
            r"\b[a-zA-Z]+\d+\b",  # switch01, router1
        ]

    async def process_command(self, message: str) -> ParsedCommand:
        """Process natural language message into structured command"""

        # Clean the message
        clean_message = self._clean_message(message)

        # Try pattern matching first (fast path)
        intent, confidence = self._pattern_match_intent(clean_message)

        if confidence > 0.8:
            # Extract devices and parameters using patterns
            devices = self._extract_devices(clean_message)
            parameters = self._extract_parameters(clean_message, intent)

            return ParsedCommand(
                intent=intent,
                devices=devices,
                parameters=parameters,
                confidence=confidence,
                method="pattern",
                raw_message=message,
            )

        # Fall back to LLM processing if available
        if self.llm_normalizer:
            return await self._llm_process_command(clean_message, message)

        # If no LLM available, return unknown intent
        return ParsedCommand(
            intent="unknown",
            devices=["all"],
            parameters={},
            confidence=0.1,
            method="fallback",
            raw_message=message,
        )

    def _clean_message(self, message: str) -> str:
        """Clean and normalize message text"""
        # Remove mentions, URLs, formatting
        clean = re.sub(r"@\w+", "", message)
        clean = re.sub(r"http[s]?://\S+", "", clean)
        clean = re.sub(r"[*_`]", "", clean)
        clean = re.sub(r"\s+", " ", clean)  # Normalize whitespace
        return clean.strip().lower()

    def _pattern_match_intent(self, message: str) -> Tuple[str, float]:
        """Fast pattern matching for common intents"""
        best_intent = "unknown"
        best_score = 0.0

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score = 0.9  # High confidence for pattern matches
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        return best_intent, best_score

    def _extract_devices(self, message: str) -> List[str]:
        """Extract device names from message"""
        devices = []

        # Look for specific device names using patterns
        for pattern in self.device_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            devices.extend(matches)

        # Check for "all" keyword
        if re.search(r"\ball\b", message):
            return ["all"]

        # If no specific devices found, default to "all"
        return list(set(devices)) if devices else ["all"]

    def _extract_parameters(self, message: str, intent: str) -> Dict:
        """Extract intent-specific parameters"""
        parameters = {}

        # Interface-specific parameters
        if "interface" in intent:
            if re.search(r"\bdown\b", message):
                parameters["status_filter"] = "down"
            elif re.search(r"\bup\b", message):
                parameters["status_filter"] = "up"

        # BGP-specific parameters
        if "bgp" in intent:
            if re.search(r"established", message):
                parameters["state_filter"] = "established"
            elif re.search(r"down", message):
                parameters["state_filter"] = "down"

        # Log-specific parameters
        if "logs" in intent or "system" in intent:
            if re.search(r"error", message):
                parameters["level_filter"] = "error"
            elif re.search(r"warning", message):
                parameters["level_filter"] = "warning"

        return parameters

    async def _llm_process_command(
        self, clean_message: str, original_message: str
    ) -> ParsedCommand:
        """Use LLM for complex command processing"""

        prompt = f"""
        Analyze this network operations command and extract the intent and parameters:

        Command: "{clean_message}"

        Available intents (map to these exactly):
        - get_system_version: Get device software version and system info
        - get_interface_status: Display interface/port status and configuration
        - get_bgp_summary: BGP neighbor information and routing status
        - get_arp_table: ARP/MAC address table information
        - get_system_logs: System logs, error messages, syslog
        - help: User needs assistance with commands
        - unknown: Cannot determine intent clearly

        Device targeting:
        - Extract specific device names (spine1, leaf2, core-router-01, etc.)
        - If "all" or no specific devices mentioned, use ["all"]

        Return JSON format:
        {{
            "intent": "intent_name",
            "devices": ["device1", "device2"] or ["all"],
            "parameters": {{}},
            "confidence": 0.95
        }}
        """

        try:
            # Use your existing LLM normalizer
            response = await self.llm_normalizer.normalize_output(
                raw_output=clean_message,
                intent_type="chat_command",
                device_platform="chat",
                custom_prompt=prompt,
            )

            if response.get("success") and "normalized_output" in response:
                result = response["normalized_output"]
                return ParsedCommand(
                    intent=result.get("intent", "unknown"),
                    devices=result.get("devices", ["all"]),
                    parameters=result.get("parameters", {}),
                    confidence=result.get("confidence", 0.7),
                    method="llm",
                    raw_message=original_message,
                )
        except Exception as e:
            logger.error(f"LLM command processing failed: {e}")

        # Fallback if LLM processing fails
        return ParsedCommand(
            intent="unknown",
            devices=["all"],
            parameters={},
            confidence=0.1,
            method="llm_failed",
            raw_message=original_message,
        )

    def generate_help_message(self) -> str:
        """Generate help message for users"""
        return """
🤖 **NetOps Bot - Available Commands**

**Device Operations:**
• `@netops-bot show version on spine1,leaf1` - Get device software versions
• `@netops-bot check interfaces on all devices` - Interface status and configuration
• `@netops-bot bgp summary for spine1` - BGP neighbor information
• `@netops-bot show arp table on leaf2` - ARP/MAC address tables
• `@netops-bot check logs on spine1` - System logs and error messages

**Device Selection:**
• **Specific devices:** `spine1,leaf1,leaf2` or `spine1 and leaf2`
• **All devices:** `all devices` or just `all`
• **Device groups:** `all spine devices` (if configured)

**Example Commands:**
• *"What's the version on spine1?"*
• *"Show me interface status for all leaf switches"*
• *"Check BGP neighbors on spine1 and spine2"*
• *"Any errors in the logs on leaf1?"*
• *"Help"* - Show this message

**💡 Tip:** Just mention @netops-bot and describe what you want to check!
        """

    def is_help_request(self, message: str) -> bool:
        """Check if the message is asking for help"""
        help_keywords = ["help", "usage", "commands", "what can you do", "instructions"]
        clean_msg = message.lower().strip()
        return any(keyword in clean_msg for keyword in help_keywords)
