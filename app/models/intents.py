# app/models/intents.py
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Request for executing a network intent (supports both known and dynamic intents)"""

    devices: List[str] = Field(..., description="List of device names")
    intent: str = Field(
        ...,
        description="Network intent to execute (known intent enum or dynamic string)",
    )
    parameters: Optional[Dict] = Field(
        default_factory=dict, description="Intent-specific parameters"
    )
    timeout: int = Field(default=30, ge=5, le=300, description="Timeout in seconds")

    # NEW: Dynamic discovery options
    allow_discovery: bool = Field(
        default=True, description="Allow dynamic command discovery for unknown intents"
    )
    force_fresh_discovery: bool = Field(
        default=False, description="Force fresh discovery (ignore cache)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "devices": ["spine1", "leaf1"],
                "intent": "get_system_version",  # Can also be "show serial number" for discovery
                "parameters": {},
                "timeout": 30,
                "allow_discovery": True,
                "force_fresh_discovery": False,
            }
        }


class DiscoveryInfo(BaseModel):
    """Information about dynamic command discovery"""

    discovered_command: str = Field(..., description="Command discovered by AI")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Discovery confidence score"
    )
    reasoning: Optional[str] = Field(
        None, description="AI reasoning for command choice"
    )
    discovery_time: float = Field(
        ..., description="Time taken for discovery in seconds"
    )
    cached_result: bool = Field(
        default=False, description="Whether result was retrieved from cache"
    )
    safety_approved: bool = Field(
        default=True, description="Whether command passed safety validation"
    )


class DeviceResult(BaseModel):
    """Enhanced result from a device operation with discovery support"""

    device_name: str = Field(..., description="Device name")
    success: bool = Field(..., description="Whether operation succeeded")
    raw_output: Optional[str] = Field(None, description="Raw command output")
    normalized_output: Optional[Dict[str, Any]] = Field(
        None, description="Structured output data"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(default=0, description="Execution time in seconds")
    normalization_method: str = Field(
        default="unknown", description="Parsing method used"
    )
    command_executed: Optional[str] = Field(
        None, description="Actual command that was executed"
    )

    # NEW: Discovery information for unknown intents
    discovery_info: Optional[DiscoveryInfo] = Field(
        None, description="Dynamic discovery details"
    )


class DeviceIntentResult(BaseModel):
    """DEPRECATED: Legacy result format - use DeviceResult instead"""

    device: str = Field(..., description="Device name")
    intent: str = Field(..., description="Network intent executed")
    status: str = Field(..., description="Execution status (success/failed/partial)")
    normalized_output: Optional[Dict] = Field(
        None, description="Standardized output format"
    )
    raw_output: Optional[str] = Field(None, description="Raw command output")
    vendor_command: Optional[str] = Field(None, description="Actual command executed")
    execution_time: Optional[float] = Field(
        None, description="Execution time in seconds"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Execution timestamp"
    )

    # LLM-related fields
    normalization_method: Optional[str] = Field(
        None, description="Method used: manual/llm/cache/fallback"
    )
    confidence_score: Optional[float] = Field(
        None, description="Normalization confidence (0.0-1.0)"
    )

    # NEW: Discovery support
    discovery_info: Optional[DiscoveryInfo] = Field(
        None, description="Dynamic discovery details"
    )


class IntentResponse(BaseModel):
    """Enhanced response from intent execution with discovery support"""

    success: bool = Field(..., description="Overall execution success")
    intent: str = Field(..., description="Network intent executed")
    total_devices: int = Field(..., description="Total number of target devices")
    successful_devices: int = Field(..., description="Number of successful executions")
    failed_devices: int = Field(..., description="Number of failed executions")
    execution_time: float = Field(..., description="Total execution time in seconds")
    results: List[DeviceResult] = Field(..., description="Individual device results")
    summary: str = Field(..., description="Execution summary message")

    # NEW: Discovery tracking
    discovery_used: bool = Field(
        default=False, description="Whether dynamic discovery was used"
    )

    # Optional legacy fields for backward compatibility
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique request ID"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow, description="Request timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "intent": "get_system_version",
                "total_devices": 2,
                "successful_devices": 2,
                "failed_devices": 0,
                "execution_time": 2.45,
                "summary": "Successfully executed on 2/2 devices",
                "discovery_used": False,
                "results": [
                    {
                        "device_name": "spine1",
                        "success": True,
                        "normalized_output": {"vendor": "Arista", "version": "4.27.3F"},
                        "execution_time": 1.23,
                        "normalization_method": "manual",
                        "command_executed": "show version",
                    }
                ],
            }
        }


# NEW: Discovery-specific models
class CommandDiscoveryRequest(BaseModel):
    """Request for testing command discovery"""

    user_intent: str = Field(..., description="Natural language intent from user")
    device_name: str = Field(..., description="Target device name")
    allow_cached: bool = Field(
        default=True, description="Allow cached discovery results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_intent": "show serial number",
                "device_name": "spine1",
                "allow_cached": True,
            }
        }


class CommandDiscoveryResponse(BaseModel):
    """Response from command discovery test"""

    user_intent: str = Field(..., description="Original user intent")
    device_name: str = Field(..., description="Target device name")
    device_platform: str = Field(..., description="Device platform")
    discovery_result: Dict[str, Any] = Field(
        ..., description="Discovery result details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_intent": "show serial number",
                "device_name": "spine1",
                "device_platform": "arista_eos",
                "discovery_result": {
                    "success": True,
                    "command": "show version",
                    "confidence": 0.95,
                    "reasoning": "Serial number is typically found in version output on Arista devices",
                    "safety_approved": True,
                    "processing_time": 1.23,
                },
            }
        }


class DiscoveryStats(BaseModel):
    """Statistics about dynamic discovery usage"""

    total_discoveries: int = Field(..., description="Total discovery attempts")
    successful_discoveries: int = Field(..., description="Successful discoveries")
    cache_hits: int = Field(..., description="Cache hits")
    safety_blocks: int = Field(..., description="Commands blocked for safety")
    success_rate_percent: float = Field(..., description="Success rate percentage")
    average_discovery_time: float = Field(
        ..., description="Average discovery time in seconds"
    )
    cache_size: int = Field(..., description="Number of cached mappings")


class CachedMapping(BaseModel):
    """Cached command mapping entry"""

    intent: str = Field(..., description="User intent")
    platform: str = Field(..., description="Device platform")
    command: str = Field(..., description="Discovered command")
    confidence: float = Field(..., description="Discovery confidence")
    usage_count: int = Field(..., description="Number of times used")
    age_hours: float = Field(..., description="Age in hours since discovery")


class DiscoveryCacheStats(BaseModel):
    """Discovery cache statistics"""

    cached_mappings: int = Field(..., description="Total cached mappings")
    total_cache_usage: int = Field(..., description="Total cache usage count")
    cache_entries: List[CachedMapping] = Field(
        ..., description="Individual cache entries"
    )
