# app/services/inventory.py
import json
from pathlib import Path
from typing import Dict, List, Optional

from app.models.devices import Device, DeviceGroup


class InventoryService:
    """Service for managing device inventory"""

    def __init__(self, inventory_file: str = "inventory/devices.json"):
        self.inventory_file = Path(inventory_file)
        self._devices_cache = None
        self._groups_cache = None

    def _load_inventory(self) -> Dict:
        """Load inventory from JSON file"""
        try:
            with open(self.inventory_file) as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Inventory file not found: {self.inventory_file}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in inventory file: {e}") from e

    def _get_devices_data(self) -> Dict:
        """Get devices data with caching"""
        if self._devices_cache is None:
            inventory = self._load_inventory()
            self._devices_cache = inventory.get("devices", {})
        return self._devices_cache

    def _get_groups_data(self) -> Dict:
        """Get groups data with caching"""
        if self._groups_cache is None:
            inventory = self._load_inventory()
            self._groups_cache = inventory.get("groups", {})
        return self._groups_cache

    def refresh_cache(self):
        """Refresh the inventory cache"""
        self._devices_cache = None
        self._groups_cache = None

    def get_all_devices(self) -> List[Device]:
        """Get all devices from inventory"""
        devices_data = self._get_devices_data()
        devices = []

        for name, data in devices_data.items():
            device = Device(name=name, **data)
            devices.append(device)

        return devices

    def get_device(self, device_name: str) -> Optional[Device]:
        """Get a specific device by name"""
        devices_data = self._get_devices_data()

        if device_name in devices_data:
            return Device(name=device_name, **devices_data[device_name])
        return None

    def get_devices_by_group(self, group_name: str) -> List[Device]:
        """Get devices that belong to a specific group"""
        all_devices = self.get_all_devices()
        groups_data = self._get_groups_data()

        if group_name not in groups_data:
            return []

        group_criteria = groups_data[group_name]
        matching_devices = []

        for device in all_devices:
            # Check if device matches group criteria
            matches = True
            for key, value in group_criteria.items():
                if key == "connection_options":  # Skip connection options
                    continue
                device_dict = device.model_dump()
                if device_dict.get(key) != value:
                    matches = False
                    break

            if matches:
                matching_devices.append(device)

        return matching_devices

    def get_devices_by_type(self, device_type: str) -> List[Device]:
        """Get devices by type (spine, leaf, etc.)"""
        all_devices = self.get_all_devices()
        return [d for d in all_devices if d.device_type == device_type]

    def get_devices_by_site(self, site: str) -> List[Device]:
        """Get devices by site"""
        all_devices = self.get_all_devices()
        return [d for d in all_devices if d.site == site]

    def device_exists(self, device_name: str) -> bool:
        """Check if a device exists in inventory"""
        return device_name in self._get_devices_data()

    def get_all_groups(self) -> List[DeviceGroup]:
        """Get all device groups"""
        groups_data = self._get_groups_data()
        groups = []

        for name, data in groups_data.items():
            group = DeviceGroup(
                name=name,
                **{k: v for k, v in data.items() if k != "connection_options"},
            )
            groups.append(group)

        return groups


# Global inventory service instance
inventory_service = InventoryService()
