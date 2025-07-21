# app/services/intent_parser.py
"""
Intent Parser - Enhanced Natural Language Understanding
Replaces chat_nlp.py with a more powerful, universal approach
Extracts intent from ANY natural language input (chat, API, CLI, web)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentParser:
    """
    Universal intent parser that works for ALL interfaces
    Extracts intent and devices from natural language input
    """

    def __init__(self):
        # Common network operation patterns
        self.operation_patterns = {
            # Version/System Info
            "version": r"\b(?:show|get|display|check)\s+(?:system\s+)?version\b",
            "system_info": r"\b(?:show|get|display)\s+(?:system|hardware|platform)\s+(?:info|information)\b",
            "serial": r"\b(?:show|get|display|find)\s+(?:system\s+)?serial\s+(?:number|num)?\b",
            "hostname": r"\b(?:show|get|display)\s+(?:system\s+)?hostname\b",
            "uptime": r"\b(?:show|get|display|check)\s+(?:system\s+)?uptime\b",
            # Interface Operations
            "interfaces": r"\b(?:show|get|display|check|list)\s+(?:interface|interfaces|int|intf)\s*(?:status|state|brief)?\b",
            "interface_status": r"\b(?:show|get|display|check)\s+(?:interface|interfaces)\s+(?:status|state|brief)\b",
            "interface_counters": r"\b(?:show|get|display)\s+(?:interface|interfaces)\s+(?:counters|statistics|stats)\b",
            "interface_errors": r"\b(?:show|get|display|check)\s+(?:interface|interfaces)\s+(?:errors|error)\b",
            # BGP Operations
            "bgp": r"\b(?:show|get|display|check)\s+(?:ip\s+)?bgp\s*(?:summary|neighbors?|status)?\b",
            "bgp_summary": r"\b(?:show|get|display)\s+(?:ip\s+)?bgp\s+summary\b",
            "bgp_neighbors": r"\b(?:show|get|display)\s+(?:ip\s+)?bgp\s+neighbors?\b",
            # Routing
            "routing": r"\b(?:show|get|display)\s+(?:ip\s+)?(?:route|routing)\s*(?:table)?\b",
            "arp": r"\b(?:show|get|display)\s+(?:ip\s+)?arp\s*(?:table)?\b",
            "mac": r"\b(?:show|get|display)\s+mac\s*(?:address)?(?:-table)?\b",
            # Performance/Resources
            "cpu": r"\b(?:show|get|display|check)\s+(?:cpu|processor)\s*(?:usage|utilization)?\b",
            "memory": r"\b(?:show|get|display|check)\s+(?:memory|ram)\s*(?:usage|utilization)?\b",
            "processes": r"\b(?:show|get|display|list)\s+(?:processes|process)\b",
            # Environment
            "temperature": r"\b(?:show|get|display|check)\s+(?:environment\s+)?(?:temperature|temp)\b",
            "power": r"\b(?:show|get|display|check)\s+(?:environment\s+)?(?:power|psu)\s*(?:status)?\b",
            "fans": r"\b(?:show|get|display|check)\s+(?:environment\s+)?(?:fans?|cooling)\s*(?:status)?\b",
            # Logs
            "logs": r"\b(?:show|get|display)\s+(?:log|logs|logging)\b",
            "syslog": r"\b(?:show|get|display)\s+(?:syslog|system\s+log)\b",
        }

        # Device extraction patterns
        self.device_patterns = [
            r"\bon\s+([\w\-,\s]+)(?:\s+and\s+[\w\-,\s]+)*\b",  # "on spine1, leaf1"
            r"\bfor\s+([\w\-,\s]+)(?:\s+and\s+[\w\-,\s]+)*\b",  # "for spine1, leaf1"
            r"\b((?:spine|leaf|core|edge|router|switch|device)\d+(?:[\w\-]*)?(?:,\s*(?:spine|leaf|core|edge|router|switch|device)\d+[\w\-]*)*)\b",  # device naming patterns
            r"\b(all\s+(?:devices|spines?|leafs?|leaves|routers?|switches?))\b",  # "all devices"
            r"\b(everything|all)\b",  # simple "all"
        ]

        # Special keywords
        self.special_keywords = {
            "all": ["all", "everything", "every device", "all devices"],
            "help": ["help", "usage", "commands", "what can you do"],
            "status": ["status", "health", "check"],
        }

    async def extract_intent(self, user_input: str) -> str:
        """
        Extract intent from natural language input
        Returns a normalized intent string for command discovery
        """

        clean_input = self._clean_input(user_input)
        logger.info(f"🧠 Parsing intent from: '{clean_input}'")

        # Check for help requests
        if self._is_help_request(clean_input):
            return "help"

        # Extract operation intent
        intent = self._extract_operation_intent(clean_input)

        logger.info(f"✅ Extracted intent: '{intent}'")
        return intent

    def extract_devices(
        self, user_input: str, default_devices: List[str] = None
    ) -> List[str]:
        """
        Extract device names from natural language input
        Returns list of device names
        """

        clean_input = self._clean_input(user_input)
        devices = []

        # Try each device pattern
        for pattern in self.device_patterns:
            matches = re.finditer(pattern, clean_input, re.IGNORECASE)
            for match in matches:
                device_text = match.group(1).strip()

                # Handle "all" keywords
                if any(
                    keyword in device_text.lower()
                    for keyword in self.special_keywords["all"]
                ):
                    devices.append("all")
                    continue

                # Parse comma-separated devices
                device_names = [
                    d.strip()
                    for d in re.split(r"[,\s]+|(?:\s+and\s+)", device_text)
                    if d.strip()
                ]
                devices.extend(device_names)

        # Remove duplicates while preserving order
        seen = set()
        unique_devices = []
        for device in devices:
            if device.lower() not in seen:
                seen.add(device.lower())
                unique_devices.append(device)

        # Fallback to default devices if none found
        if not unique_devices and default_devices:
            unique_devices = default_devices

        logger.info(f"📍 Extracted devices: {unique_devices}")
        return unique_devices

    async def parse(
        self, user_input: str, default_devices: List[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Parse user input to extract both intent and devices
        Returns (intent, devices) tuple
        """

        intent = await self.extract_intent(user_input)
        devices = self.extract_devices(user_input, default_devices)

        return intent, devices

    def _clean_input(self, user_input: str) -> str:
        """Clean and normalize user input"""

        # Remove bot mentions
        clean = re.sub(r"@[\w\-_]+\s*", "", user_input)

        # Remove extra whitespace
        clean = re.sub(r"\s+", " ", clean).strip()

        # Remove punctuation at the end
        clean = re.sub(r"[.!?]+$", "", clean)

        return clean

    def _is_help_request(self, clean_input: str) -> bool:
        """Check if input is a help request"""

        help_patterns = [
            r"\bhelp\b",
            r"\busage\b",
            r"\bcommands?\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bhow\s+to\s+use\b",
            r"^\?+$",  # Just question marks
        ]

        return any(
            re.search(pattern, clean_input, re.IGNORECASE) for pattern in help_patterns
        )

    def _extract_operation_intent(self, clean_input: str) -> str:
        """Extract the main operation intent"""

        # Try to match known operation patterns
        for intent_name, pattern in self.operation_patterns.items():
            if re.search(pattern, clean_input, re.IGNORECASE):
                logger.debug(f"Matched pattern '{intent_name}': {pattern}")
                return self._normalize_intent(intent_name, clean_input)

        # If no direct match, try to infer from keywords
        inferred_intent = self._infer_intent_from_keywords(clean_input)
        if inferred_intent:
            return inferred_intent

        # Generate dynamic intent from the input
        return self._generate_dynamic_intent(clean_input)

    def _normalize_intent(self, intent_name: str, full_input: str) -> str:
        """Normalize and enhance intent based on context"""

        # Map similar intents to more specific ones
        intent_mappings = {
            "serial": "show_serial_number",
            "version": "show_version",
            "hostname": "show_hostname",
            "uptime": "show_uptime",
            "interfaces": "show_interface_status",
            "interface_status": "show_interface_status",
            "interface_counters": "show_interface_counters",
            "interface_errors": "show_interface_errors",
            "bgp": "show_bgp_summary",
            "bgp_summary": "show_bgp_summary",
            "bgp_neighbors": "show_bgp_neighbors",
            "routing": "show_routing_table",
            "arp": "show_arp_table",
            "mac": "show_mac_table",
            "cpu": "show_cpu_usage",
            "memory": "show_memory_usage",
            "processes": "show_processes",
            "temperature": "show_temperature",
            "power": "show_power_status",
            "fans": "show_fan_status",
            "logs": "show_logs",
            "syslog": "show_syslog",
        }

        return intent_mappings.get(intent_name, intent_name)

    def _infer_intent_from_keywords(self, clean_input: str) -> Optional[str]:
        """Infer intent from individual keywords when patterns don't match"""

        keyword_mappings = {
            "serial": "show_serial_number",
            "version": "show_version",
            "interface": "show_interface_status",
            "bgp": "show_bgp_summary",
            "cpu": "show_cpu_usage",
            "memory": "show_memory_usage",
            "temperature": "show_temperature",
            "power": "show_power_status",
            "fan": "show_fan_status",
            "uptime": "show_uptime",
            "hostname": "show_hostname",
            "route": "show_routing_table",
            "arp": "show_arp_table",
            "mac": "show_mac_table",
            "log": "show_logs",
        }

        # Look for keywords in the input
        words = clean_input.lower().split()
        for word in words:
            if word in keyword_mappings:
                logger.debug(f"Inferred intent from keyword '{word}'")
                return keyword_mappings[word]

        return None

    def _generate_dynamic_intent(self, clean_input: str) -> str:
        """Generate a dynamic intent name from user input"""

        # Remove common action words
        words = clean_input.lower().split()
        action_words = {"show", "get", "display", "check", "list", "find", "view"}

        # Filter out action words and common words
        meaningful_words = [
            word
            for word in words
            if word not in action_words
            and len(word) > 2
            and word not in {"the", "and", "for", "from", "with", "all", "any"}
        ]

        if meaningful_words:
            # Create intent from meaningful words
            intent = "show_" + "_".join(meaningful_words[:3])  # Limit to 3 words
        else:
            # Fallback intent
            intent = "show_system_info"

        logger.debug(f"Generated dynamic intent: '{intent}' from '{clean_input}'")
        return intent

    def get_supported_operations(self) -> Dict[str, str]:
        """Get list of supported operations for help"""

        return {
            "System Information": [
                "show version",
                "show serial number",
                "show hostname",
                "show uptime",
            ],
            "Interface Operations": [
                "show interfaces",
                "show interface status",
                "show interface counters",
            ],
            "BGP Operations": ["show bgp summary", "show bgp neighbors"],
            "Routing Information": [
                "show routing table",
                "show arp table",
                "show mac table",
            ],
            "Performance Monitoring": [
                "show cpu usage",
                "show memory usage",
                "show processes",
            ],
            "Environment": ["show temperature", "show power status", "show fan status"],
            "Logs": ["show logs", "show syslog"],
        }

    def generate_help_message(self) -> str:
        """Generate help message for users"""

        help_sections = []
        help_sections.append("🤖 **NetOps Bot - Available Commands**\n")

        operations = self.get_supported_operations()

        for category, commands in operations.items():
            help_sections.append(f"**{category}:**")
            for command in commands:
                help_sections.append(f"• `@netops-bot {command} on <device>`")
            help_sections.append("")

        help_sections.extend(
            [
                "**Device Examples:**",
                "• `spine1` - specific device",
                "• `spine1,leaf1,leaf2` - multiple devices",
                "• `all devices` - all devices",
                "",
                "**💡 AI Discovery:**",
                "I can learn new commands! Just describe what you want:",
                "• `show cpu utilization on spine1`",
                "• `check disk space on all devices`",
                "• `get power consumption on leaf1`",
                "",
                "**Examples:**",
                "• `@netops-bot show version on spine1`",
                "• `@netops-bot check bgp summary on all devices`",
                "• `@netops-bot show serial number on spine1,leaf1`",
            ]
        )

        return "\n".join(help_sections)


# Global instance
intent_parser = IntentParser()
