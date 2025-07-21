# app/models/devices.py
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Device types in the network"""

    SPINE = "spine"
    LEAF = "leaf"
    BORDER = "border"
    FIREWALL = "firewall"
    ROUTER = "router"
    SWITCH = "switch"


class DeviceStatus(str, Enum):
    """Device status"""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class Device(BaseModel):
    """Device model matching your inventory structure with hashability support"""

    name: str = Field(..., description="Device name/hostname")
    hostname: str = Field(..., description="Management IP address")
    platform: str = Field(..., description="Device platform (e.g., arista_eos)")
    username: str = Field(..., description="Login username")
    password: str = Field(..., description="Login password")
    device_type: DeviceType = Field(..., description="Type of device")
    site: str = Field(..., description="Site location")
    vendor: str = Field(..., description="Device vendor")
    role: str = Field(..., description="Device role in network")

    def __hash__(self):
        """Make Device objects hashable using the device name as unique identifier"""
        return hash(self.name)

    def __eq__(self, other):
        """Define equality based on device name"""
        if isinstance(other, Device):
            return self.name == other.name
        return False

    def __repr__(self):
        """String representation for debugging"""
        return f"Device(name='{self.name}', platform='{self.platform}', site='{self.site}')"

    def __str__(self):
        """Human-readable string representation"""
        return f"{self.name} ({self.vendor} {self.device_type.value})"

    class Config:
        json_schema_extra = {
            "example": {
                "name": "spine1",
                "hostname": "172.20.20.100",
                "platform": "arista_eos",
                "username": "admin",
                "password": "admin",
                "device_type": "spine",
                "site": "lab",
                "vendor": "arista",
                "role": "spine",
            }
        }


class DeviceList(BaseModel):
    """Response model for listing devices"""

    devices: List[Device]
    total_count: int


class DeviceGroup(BaseModel):
    """Device group model"""

    name: str
    platform: Optional[str] = None
    site: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
