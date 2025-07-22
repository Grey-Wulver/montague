# app/models/universal_requests.py
"""
Universal Request/Response Models
Foundation of the Universal Pipeline - handles ALL interfaces (Chat, API, CLI, Web)
"""

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InterfaceType(str, Enum):
    """Supported interface types"""

    CHAT = "chat"
    API = "api"
    CLI = "cli"
    WEB = "web"
    WEBHOOK = "webhook"


class OutputFormat(str, Enum):
    """Supported output formats"""

    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CSV = "csv"
    TABLE = "table"
    HTML = "html"
    MARKDOWN = "markdown"
    CONVERSATIONAL = "conversational"
    RAW = "raw"


class RequestPriority(str, Enum):
    """Request priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class UniversalRequest(BaseModel):
    """
    Universal request model that works for ALL interfaces
    Chat, API, CLI, Web - all use this same structure
    """

    # Core request data
    user_input: str = Field(..., description="Natural language request")
    devices: List[str] = Field(default=[], description="Target devices")

    # Interface specification
    interface_type: InterfaceType = Field(default=InterfaceType.API)
    output_format: OutputFormat = Field(default=OutputFormat.JSON)

    # Request metadata
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    priority: RequestPriority = Field(default=RequestPriority.NORMAL)

    # Execution parameters
    timeout: int = Field(default=30, ge=1, le=300)
    allow_discovery: bool = Field(default=True)
    force_fresh_discovery: bool = Field(default=False)
    include_raw_output: bool = Field(default=True)

    # Context and memory
    conversation_context: Optional[Dict[str, Any]] = Field(default=None)
    previous_requests: Optional[List[str]] = Field(default=None)

    # Advanced options
    parallel_execution: bool = Field(default=True)
    max_devices: int = Field(default=100, ge=1, le=1000)
    include_discovery_info: bool = Field(default=False)

    # Format-specific options
    format_options: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        json_encoders = {
            # Custom encoders if needed
        }
        schema_extra = {
            "examples": [
                {
                    "user_input": "show serial number spine1",
                    "devices": ["spine1"],
                    "interface_type": "chat",
                    "output_format": "conversational",
                },
                {
                    "user_input": "get bgp summary for all spine devices",
                    "devices": ["all"],
                    "interface_type": "api",
                    "output_format": "json",
                    "include_raw_output": True,
                },
            ]
        }


class DeviceExecutionResult(BaseModel):
    """Result from executing command on a single device"""

    device_name: str
    success: bool
    execution_time: float

    # Command info
    command_executed: Optional[str] = None
    discovery_used: bool = False
    discovery_confidence: Optional[float] = None
    cached_discovery: bool = False

    # Output data
    raw_output: Optional[str] = None
    formatted_output: Optional[Any] = None

    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Metadata
    platform: Optional[str] = None
    vendor: Optional[str] = None
    connection_method: Optional[str] = None


class DiscoveryInfo(BaseModel):
    """Information about command discovery process"""

    intent_detected: str
    discovered_command: str
    confidence: float
    reasoning: Optional[str] = None
    cached_result: bool = False
    discovery_time: float = 0.0
    platform: str
    fallback_used: bool = False


class UniversalResponse(BaseModel):
    """
    Universal response model for ALL interfaces
    Automatically adapts to chat, API, CLI, web, etc.
    """

    # Request correlation
    request_id: str
    success: bool
    execution_time: float
    timestamp: float = Field(default_factory=time.time)

    # Core results
    formatted_response: Any = Field(..., description="Response formatted for interface")
    total_devices: int = 0
    successful_devices: int = 0
    failed_devices: int = 0

    # Device results
    device_results: List[DeviceExecutionResult] = Field(default=[])

    # Discovery information
    discovery_used: bool = False
    discovery_info: List[DiscoveryInfo] = Field(default=[])
    new_discoveries: int = 0  # Count of new command mappings learned

    # Raw data (optional)
    raw_device_outputs: Optional[Dict[str, str]] = Field(default=None)

    # Error handling
    error_message: Optional[str] = None
    partial_success: bool = False

    # Performance metrics
    discovery_time: float = 0.0
    execution_time_per_device: Optional[Dict[str, float]] = None
    formatting_time: float = 0.0

    # Interface-specific metadata
    interface_metadata: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        schema_extra = {
            "examples": [
                {
                    "request_id": "abc123",
                    "success": True,
                    "execution_time": 2.34,
                    "formatted_response": "Serial Number: 5BB785FB6AC8CCAC228195AAA54F5D5D",
                    "total_devices": 1,
                    "successful_devices": 1,
                    "discovery_used": True,
                    "new_discoveries": 1,
                }
            ]
        }


class BatchUniversalRequest(BaseModel):
    """For processing multiple requests in batch"""

    requests: List[UniversalRequest]
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    max_concurrent: int = Field(default=10, ge=1, le=50)
    fail_fast: bool = Field(default=False)


class BatchUniversalResponse(BaseModel):
    """Response for batch requests"""

    batch_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    batch_execution_time: float
    responses: List[UniversalResponse]


# Interface-specific request models (for type safety)


class ChatRequest(UniversalRequest):
    """Chat-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.CHAT
    output_format: OutputFormat = OutputFormat.CONVERSATIONAL
    include_discovery_info: bool = True


class APIRequest(UniversalRequest):
    """API-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.API
    output_format: OutputFormat = OutputFormat.JSON
    include_raw_output: bool = True


class CLIRequest(UniversalRequest):
    """CLI-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.CLI
    output_format: OutputFormat = OutputFormat.TABLE
    include_discovery_info: bool = False


class WebRequest(UniversalRequest):
    """Web UI-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.WEB
    output_format: OutputFormat = OutputFormat.HTML
    include_discovery_info: bool = True


# Helper functions for request creation


def create_chat_request(
    message: str, devices: List[str] = None, user_id: str = None, session_id: str = None
) -> ChatRequest:
    """Helper to create chat requests quickly"""
    return ChatRequest(
        user_input=message,
        devices=devices or [],
        user_id=user_id,
        session_id=session_id,
    )


def create_api_request(
    query: str,
    devices: List[str],
    output_format: OutputFormat = OutputFormat.JSON,
    include_raw: bool = False,
) -> APIRequest:
    """Helper to create API requests quickly"""
    return APIRequest(
        user_input=query,
        devices=devices,
        output_format=output_format,
        include_raw_output=include_raw,
    )


def create_cli_request(
    command: str, devices: List[str], table_format: bool = True
) -> CLIRequest:
    """Helper to create CLI requests quickly"""
    return CLIRequest(
        user_input=command,
        devices=devices,
        output_format=OutputFormat.TABLE if table_format else OutputFormat.JSON,
    )


# Validation helpers


def validate_devices(devices: List[str]) -> List[str]:
    """Validate and normalize device list"""
    if not devices:
        return []

    # Handle "all" keyword
    if "all" in devices:
        # This will be resolved by the inventory service
        return ["all"]

    # Remove duplicates and validate format
    unique_devices = list(set(devices))
    return [device.strip().lower() for device in unique_devices if device.strip()]


def validate_request_size(request: UniversalRequest) -> bool:
    """Validate request doesn't exceed limits"""
    if len(request.devices) > request.max_devices:
        raise ValueError(
            f"Too many devices: {len(request.devices)} > {request.max_devices}"
        )

    if len(request.user_input) > 10000:  # 10KB limit
        raise ValueError("User input too large")

    return True
