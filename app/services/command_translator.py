# app/services/command_translator.py
import logging
from typing import Dict, List, Optional

from app.models.devices import Device
from app.models.intents import NetworkIntent

logger = logging.getLogger(__name__)


class CommandTranslator:
    """Translates network intents to vendor-specific commands"""

    # Command mapping: Intent -> Platform -> Command
    COMMAND_MAP = {
        NetworkIntent.GET_SYSTEM_VERSION: {
            "arista_eos": "show version",
            "cisco_ios": "show version",
            "cisco_nxos": "show version",
            "cisco_xr": "show version",
            "juniper_junos": "show version",
            "hp_comware": "display version",
            "hp_procurve": "show version",
        },
        NetworkIntent.GET_INTERFACE_STATUS: {
            "arista_eos": "show interfaces status",
            "cisco_ios": "show ip interface brief",
            "cisco_nxos": "show interface brief",
            "cisco_xr": "show ipv4 interface brief",
            "juniper_junos": "show interfaces terse",
            "hp_comware": "display interface brief",
            "hp_procurve": "show interfaces brief",
        },
        NetworkIntent.GET_BGP_SUMMARY: {
            "arista_eos": "show ip bgp summary",
            "cisco_ios": "show ip bgp summary",
            "cisco_nxos": "show ip bgp summary",
            "cisco_xr": "show bgp summary",
            "juniper_junos": "show bgp summary",
            "hp_comware": "display bgp peer",
            "hp_procurve": None,  # Not supported
        },
        NetworkIntent.GET_BGP_NEIGHBORS: {
            "arista_eos": "show ip bgp neighbors",
            "cisco_ios": "show ip bgp neighbors",
            "cisco_nxos": "show ip bgp neighbors",
            "cisco_xr": "show bgp neighbors",
            "juniper_junos": "show bgp neighbor",
            "hp_comware": "display bgp peer verbose",
            "hp_procurve": None,
        },
        NetworkIntent.GET_ROUTING_TABLE: {
            "arista_eos": "show ip route",
            "cisco_ios": "show ip route",
            "cisco_nxos": "show ip route",
            "cisco_xr": "show route",
            "juniper_junos": "show route",
            "hp_comware": "display ip routing-table",
            "hp_procurve": "show ip route",
        },
        NetworkIntent.GET_ARP_TABLE: {
            "arista_eos": "show arp",
            "cisco_ios": "show arp",
            "cisco_nxos": "show ip arp",
            "cisco_xr": "show arp",
            "juniper_junos": "show arp",
            "hp_comware": "display arp",
            "hp_procurve": "show arp",
        },
        NetworkIntent.GET_MAC_TABLE: {
            "arista_eos": "show mac address-table",
            "cisco_ios": "show mac address-table",
            "cisco_nxos": "show mac address-table",
            "cisco_xr": None,  # Different architecture
            "juniper_junos": "show ethernet-switching table",
            "hp_comware": "display mac-address",
            "hp_procurve": "show mac-address",
        },
        NetworkIntent.GET_SYSTEM_LOGS: {
            "arista_eos": "show logging",
            "cisco_ios": "show logging",
            "cisco_nxos": "show logging last 100",
            "cisco_xr": "show logging",
            "juniper_junos": "show log messages",
            "hp_comware": "display logbuffer",
            "hp_procurve": "show logging",
        },
    }

    def get_command(self, intent: NetworkIntent, device: Device) -> Optional[str]:
        """Get vendor-specific command for an intent"""
        intent_commands = self.COMMAND_MAP.get(intent)
        if not intent_commands:
            logger.warning(f"Intent {intent} not found in command map")
            return None

        command = intent_commands.get(device.platform)
        if command is None:
            logger.info(f"Intent {intent} not supported on platform {device.platform}")

        return command

    def is_supported(self, intent: NetworkIntent, device: Device) -> bool:
        """Check if intent is supported on this device platform"""
        command = self.get_command(intent, device)
        return command is not None

    def get_supported_intents(self, device: Device) -> List[NetworkIntent]:
        """Get list of supported intents for a device"""
        supported = []
        for intent in NetworkIntent:
            if self.is_supported(intent, device):
                supported.append(intent)
        return supported

    def get_all_platforms(self) -> List[str]:
        """Get list of all supported platforms"""
        platforms = set()
        for intent_commands in self.COMMAND_MAP.values():
            platforms.update(intent_commands.keys())
        return sorted(list(platforms))

    def get_platform_coverage(self, platform: str) -> Dict[str, bool]:
        """Get intent support coverage for a specific platform"""
        coverage = {}
        for intent in NetworkIntent:
            intent_commands = self.COMMAND_MAP.get(intent, {})
            coverage[intent.value] = intent_commands.get(platform) is not None
        return coverage


# Global translator instance
command_translator = CommandTranslator()
