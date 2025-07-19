# app/services/community_mapper.py
import json
import logging
from typing import Dict, Set

import requests

logger = logging.getLogger(__name__)


class CommunityCommandMapper:
    """Extract command mappings from existing open source projects"""

    def __init__(self):
        self.ntc_templates_cache = None
        self.genie_parsers_cache = None
        self.napalm_getters_cache = None

    def harvest_ntc_templates(self) -> Dict[str, Set[str]]:
        """Extract commands from ntc-templates repository"""
        if self.ntc_templates_cache:
            return self.ntc_templates_cache

        logger.info("🔍 Harvesting ntc-templates repository...")

        # GitHub API to get all template files
        api_url = "https://api.github.com/repos/networktocode/ntc-templates/contents/ntc_templates/templates"

        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            files = response.json()

            command_map = {}
            processed_count = 0

            for file_info in files:
                filename = file_info["name"]
                if filename.endswith(".textfsm"):
                    # Parse filename: platform_command.textfsm
                    # Example: arista_eos_show_version.textfsm -> platform: arista_eos, command: show version

                    base_name = filename.replace(".textfsm", "")
                    parts = base_name.split("_")

                    if len(parts) >= 3:
                        # Handle platform names (first 2 parts usually)
                        if parts[0] in ["arista", "cisco", "juniper", "hp", "fortinet", "paloalto"]:
                            platform = "_".join(parts[:2])  # e.g., arista_eos, cisco_ios
                            command_parts = parts[2:]
                        else:
                            platform = parts[0]  # e.g., linux, windows
                            command_parts = parts[1:]

                        # Reconstruct command from parts
                        command = " ".join(command_parts).replace("_", " ")

                        # Clean up common patterns
                        command = self._clean_command(command)

                        if platform not in command_map:
                            command_map[platform] = set()
                        command_map[platform].add(command)
                        processed_count += 1

            logger.info(f"✅ Processed {processed_count} ntc-templates")
            logger.info(f"📊 Found commands for platforms: {list(command_map.keys())}")

            self.ntc_templates_cache = command_map
            return command_map

        except Exception as e:
            logger.error(f"❌ Error harvesting ntc-templates: {e}")
            return {}

    def _clean_command(self, command: str) -> str:
        """Clean up command strings from filename parsing"""
        # Replace underscores with spaces
        command = command.replace("_", " ")

        # Common cleanup patterns
        replacements = {
            "show ip route summary": "show ip route summary",
            "show interfaces status": "show interfaces status",
            "show mac address table": "show mac address-table",
            "show spanning tree": "show spanning-tree",
        }

        for pattern, replacement in replacements.items():
            if pattern in command:
                command = replacement
                break

        return command.strip()

    def harvest_genie_parsers(self) -> Dict[str, Set[str]]:
        """Extract commands from Genie parsers metadata"""
        if self.genie_parsers_cache:
            return self.genie_parsers_cache

        logger.info("🔍 Harvesting Genie parsers...")

        try:
            # Try to import Genie and get parser list
            from genie.libs.parser.utils import get_parser_list

            command_map = {}
            parser_data = get_parser_list()

            for platform, commands in parser_data.items():
                if isinstance(commands, dict):
                    clean_commands = set()
                    for cmd in commands.keys():
                        # Genie commands are already clean
                        clean_commands.add(cmd)
                    command_map[platform] = clean_commands

            logger.info(f"✅ Found Genie parsers for {len(command_map)} platforms")

            self.genie_parsers_cache = command_map
            return command_map

        except ImportError:
            logger.warning("⚠️  Genie not available, skipping Genie parser harvest")
            return {}
        except Exception as e:
            logger.error(f"❌ Error harvesting Genie parsers: {e}")
            return {}

    def get_napalm_getters(self) -> Dict[str, Dict[str, str]]:
        """Get NAPALM standardized operations (already intent-mapped!)"""
        if self.napalm_getters_cache:
            return self.napalm_getters_cache

        logger.info("🔍 Loading NAPALM getter mappings...")

        # NAPALM standardized getters - these are gold standard intent mappings!
        napalm_getters = {
            "get_arp_table": {
                "arista_eos": "show arp",
                "cisco_ios": "show arp",
                "cisco_nxos": "show ip arp",
                "cisco_xr": "show arp",
                "juniper_junos": "show arp",
                "fortinet": "get system arp",
                "paloalto_panos": "show arp all",
            },
            "get_bgp_neighbors": {
                "arista_eos": "show ip bgp neighbors",
                "cisco_ios": "show ip bgp neighbors",
                "cisco_nxos": "show ip bgp neighbors",
                "cisco_xr": "show bgp neighbors",
                "juniper_junos": "show bgp neighbor",
            },
            "get_bgp_neighbors_detail": {
                "arista_eos": "show ip bgp neighbors",
                "cisco_ios": "show ip bgp neighbors",
                "cisco_nxos": "show ip bgp neighbors",
                "cisco_xr": "show bgp neighbors",
                "juniper_junos": "show bgp neighbor",
            },
            "get_environment": {
                "arista_eos": "show environment all",
                "cisco_ios": "show environment",
                "cisco_nxos": "show environment",
                "cisco_xr": "show environment",
                "juniper_junos": "show chassis environment",
            },
            "get_facts": {
                "arista_eos": "show version",
                "cisco_ios": "show version",
                "cisco_nxos": "show version",
                "cisco_xr": "show version",
                "juniper_junos": "show version",
            },
            "get_interfaces": {
                "arista_eos": "show interfaces",
                "cisco_ios": "show interfaces",
                "cisco_nxos": "show interface",
                "cisco_xr": "show interfaces",
                "juniper_junos": "show interfaces",
            },
            "get_interfaces_counters": {
                "arista_eos": "show interfaces counters",
                "cisco_ios": "show interfaces",
                "cisco_nxos": "show interface counters",
                "cisco_xr": "show interfaces",
                "juniper_junos": "show interfaces statistics",
            },
            "get_interfaces_ip": {
                "arista_eos": "show ip interface",
                "cisco_ios": "show ip interface brief",
                "cisco_nxos": "show ip interface brief",
                "cisco_xr": "show ipv4 interface brief",
                "juniper_junos": "show interfaces terse",
            },
            "get_lldp_neighbors": {
                "arista_eos": "show lldp neighbors",
                "cisco_ios": "show lldp neighbors",
                "cisco_nxos": "show lldp neighbors",
                "cisco_xr": "show lldp neighbors",
                "juniper_junos": "show lldp neighbors",
            },
            "get_lldp_neighbors_detail": {
                "arista_eos": "show lldp neighbors detail",
                "cisco_ios": "show lldp neighbors detail",
                "cisco_nxos": "show lldp neighbors detail",
                "cisco_xr": "show lldp neighbors detail",
                "juniper_junos": "show lldp neighbors detail",
            },
            "get_mac_address_table": {
                "arista_eos": "show mac address-table",
                "cisco_ios": "show mac address-table",
                "cisco_nxos": "show mac address-table",
                "juniper_junos": "show ethernet-switching table",
            },
            "get_network_instances": {
                "arista_eos": "show vrf",
                "cisco_ios": "show vrf",
                "cisco_nxos": "show vrf",
                "cisco_xr": "show vrf all",
                "juniper_junos": "show route instance",
            },
            "get_ntp_servers": {
                "arista_eos": "show ntp associations",
                "cisco_ios": "show ntp associations",
                "cisco_nxos": "show ntp peer-status",
                "cisco_xr": "show ntp associations",
                "juniper_junos": "show ntp associations",
            },
            "get_route_to": {
                "arista_eos": "show ip route {destination}",
                "cisco_ios": "show ip route {destination}",
                "cisco_nxos": "show ip route {destination}",
                "cisco_xr": "show route {destination}",
                "juniper_junos": "show route {destination}",
            },
            "get_snmp_information": {
                "arista_eos": "show snmp",
                "cisco_ios": "show snmp",
                "cisco_nxos": "show snmp",
                "cisco_xr": "show snmp",
                "juniper_junos": "show snmp",
            },
            "get_users": {
                "arista_eos": "show users",
                "cisco_ios": "show users",
                "cisco_nxos": "show users",
                "cisco_xr": "show users",
                "juniper_junos": "show system users",
            },
            "get_vlans": {
                "arista_eos": "show vlan",
                "cisco_ios": "show vlan brief",
                "cisco_nxos": "show vlan brief",
                "juniper_junos": "show vlans",
            },
        }

        logger.info(f"✅ Loaded {len(napalm_getters)} NAPALM getters")

        self.napalm_getters_cache = napalm_getters
        return napalm_getters

    def get_combined_mappings(self) -> Dict[str, Set[str]]:
        """Get all commands from all sources combined"""
        logger.info("🔄 Combining all community command sources...")

        # Get all sources
        ntc_commands = self.harvest_ntc_templates()
        genie_commands = self.harvest_genie_parsers()
        napalm_getters = self.get_napalm_getters()

        # Combine all command sets
        all_commands = {}

        # Add ntc-templates commands
        for platform, commands in ntc_commands.items():
            if platform not in all_commands:
                all_commands[platform] = set()
            all_commands[platform].update(commands)

        # Add Genie commands
        for platform, commands in genie_commands.items():
            if platform not in all_commands:
                all_commands[platform] = set()
            all_commands[platform].update(commands)

        # Add NAPALM commands (extract from getters)
        for getter_name, platform_commands in napalm_getters.items():
            for platform, command in platform_commands.items():
                if platform not in all_commands:
                    all_commands[platform] = set()
                all_commands[platform].add(command)

        # Log statistics
        total_commands = sum(len(commands) for commands in all_commands.values())
        logger.info("📊 Combined mapping statistics:")
        logger.info(f"   Total platforms: {len(all_commands)}")
        logger.info(f"   Total commands: {total_commands}")

        for platform, commands in sorted(all_commands.items()):
            logger.info(f"   {platform}: {len(commands)} commands")

        return all_commands

    def export_to_json(self, filename: str = "community_command_mappings.json"):
        """Export all discovered mappings to JSON file"""

        mappings = {
            "ntc_templates": {platform: list(commands) for platform, commands in self.harvest_ntc_templates().items()},
            "genie_parsers": {platform: list(commands) for platform, commands in self.harvest_genie_parsers().items()},
            "napalm_getters": self.get_napalm_getters(),
            "combined": {platform: list(commands) for platform, commands in self.get_combined_mappings().items()},
        }

        with open(filename, "w") as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"📄 Exported community mappings to {filename}")
        return filename


# Global community mapper instance
community_mapper = CommunityCommandMapper()
