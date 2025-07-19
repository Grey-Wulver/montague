# app/models/intents.py
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NetworkIntent(str, Enum):
    """High-level network operations (vendor-agnostic)"""

    GET_SYSTEM_VERSION = "get_system_version"
    GET_INTERFACE_STATUS = "get_interface_status"
    GET_BGP_SUMMARY = "get_bgp_summary"
    GET_BGP_NEIGHBORS = "get_bgp_neighbors"
    GET_ROUTING_TABLE = "get_routing_table"
    GET_ARP_TABLE = "get_arp_table"
    GET_MAC_TABLE = "get_mac_table"
    GET_SYSTEM_LOGS = "get_system_logs"


class IntentRequest(BaseModel):
    """Request for executing a network intent"""

    devices: List[str] = Field(..., description="List of device names")
    intent: NetworkIntent = Field(..., description="Network intent to execute")
    parameters: Optional[Dict] = Field(default_factory=dict, description="Intent-specific parameters")
    timeout: int = Field(default=30, ge=5, le=300, description="Timeout in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "devices": ["spine1", "leaf1"],
                "intent": "get_system_version",
                "parameters": {},
                "timeout": 30,
            }
        }


class DeviceIntentResult(BaseModel):
    """Result from executing intent on a single device"""

    device: str = Field(..., description="Device name")
    intent: NetworkIntent = Field(..., description="Network intent executed")
    status: str = Field(..., description="Execution status (success/failed)")
    normalized_output: Optional[Dict] = Field(None, description="Standardized output format")
    raw_output: Optional[str] = Field(None, description="Raw command output")
    vendor_command: Optional[str] = Field(None, description="Actual command executed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")


class IntentResponse(BaseModel):
    """Standardized response for any network intent"""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    intent: NetworkIntent = Field(..., description="Network intent executed")
    total_devices: int = Field(..., description="Total number of devices")
    successful: int = Field(..., description="Number of successful executions")
    failed: int = Field(..., description="Number of failed executions")
    results: List[DeviceIntentResult] = Field(..., description="Individual device results")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-01-15T10:30:00Z",
                "intent": "get_system_version",
                "total_devices": 2,
                "successful": 2,
                "failed": 0,
                "results": [
                    {
                        "device": "spine1",
                        "intent": "get_system_version",
                        "status": "success",
                        "vendor_command": "show version",
                        "execution_time": 1.23,
                    }
                ],
            }
        }
