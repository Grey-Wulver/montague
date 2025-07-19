# app/models/schemas.py
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class StandardVersionInfo(BaseModel):
    """Standardized version information across all vendors"""

    hostname: Optional[str] = Field(None, description="Device hostname")
    vendor: Optional[str] = Field(None, description="Device vendor (arista, cisco, juniper)")
    os_version: Optional[str] = Field(None, description="Operating system version")
    model: Optional[str] = Field(None, description="Hardware model")
    serial_number: Optional[str] = Field(None, description="Serial number")
    uptime_seconds: Optional[int] = Field(None, description="Uptime in seconds")
    memory_total_kb: Optional[int] = Field(None, description="Total memory in KB")
    memory_free_kb: Optional[int] = Field(None, description="Free memory in KB")
    last_reload_reason: Optional[str] = Field(None, description="Reason for last reload")

    class Config:
        json_schema_extra = {
            "example": {
                "hostname": "spine1",
                "vendor": "arista",
                "os_version": "4.33.3.1F",
                "model": "cEOSLab",
                "serial_number": "914BABAF8653DE23744C1F3C19E8E02F",
                "uptime_seconds": 7080,
                "memory_total_kb": 32866904,
                "memory_free_kb": 23728732,
            }
        }


class StandardInterface(BaseModel):
    """Standardized interface information"""

    interface: str = Field(..., description="Interface name")
    status: str = Field(..., description="Interface status (up/down/admin-down)")
    protocol: str = Field(..., description="Protocol status (up/down)")
    description: Optional[str] = Field(None, description="Interface description")
    speed: Optional[str] = Field(None, description="Interface speed")
    duplex: Optional[str] = Field(None, description="Duplex setting")
    mtu: Optional[int] = Field(None, description="MTU size")
    ip_address: Optional[str] = Field(None, description="IP address")
    subnet_mask: Optional[str] = Field(None, description="Subnet mask")
    vlan: Optional[int] = Field(None, description="VLAN ID")


class StandardInterfaceStatus(BaseModel):
    """Collection of interface status information"""

    interfaces: List[StandardInterface] = Field(default_factory=list)
    total_interfaces: int = Field(0, description="Total number of interfaces")
    up_interfaces: int = Field(0, description="Number of up interfaces")
    down_interfaces: int = Field(0, description="Number of down interfaces")


class StandardBGPNeighbor(BaseModel):
    """Standardized BGP neighbor information"""

    neighbor_ip: str = Field(..., description="BGP neighbor IP address")
    remote_as: Optional[int] = Field(None, description="Remote AS number")
    state: str = Field(..., description="BGP session state")
    uptime: Optional[str] = Field(None, description="Session uptime")
    prefixes_received: Optional[int] = Field(None, description="Number of prefixes received")
    prefixes_sent: Optional[int] = Field(None, description="Number of prefixes sent")
    local_interface: Optional[str] = Field(None, description="Local interface")


class StandardBGPSummary(BaseModel):
    """Standardized BGP summary"""

    router_id: Optional[str] = Field(None, description="BGP router ID")
    local_as: Optional[int] = Field(None, description="Local AS number")
    neighbors: List[StandardBGPNeighbor] = Field(default_factory=list)
    total_neighbors: int = Field(0, description="Total number of neighbors")
    established_neighbors: int = Field(0, description="Number of established neighbors")


class StandardARPEntry(BaseModel):
    """Standardized ARP table entry"""

    ip_address: str = Field(..., description="IP address")
    mac_address: str = Field(..., description="MAC address")
    interface: str = Field(..., description="Interface")
    age: Optional[int] = Field(None, description="Entry age in minutes")
    type: Optional[str] = Field(None, description="Entry type (dynamic/static)")


class StandardARPTable(BaseModel):
    """Collection of ARP entries"""

    entries: List[StandardARPEntry] = Field(default_factory=list)
    total_entries: int = Field(0, description="Total number of ARP entries")


# Union type for all possible normalized outputs
StandardizedOutput = Union[
    StandardVersionInfo,
    StandardInterfaceStatus,
    StandardBGPSummary,
    StandardARPTable,
    Dict[str, Any],  # Fallback for unsupported intents
]
