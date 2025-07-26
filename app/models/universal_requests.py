# File: ~/net-chatbot/app/models/universal_requests.py
# CORRECTED VERSION - Preserving working chat functionality + adding RAG validation

"""
Universal Request/Response Models
Foundation of the Universal Pipeline - handles ALL interfaces (Chat, API, CLI, Web)
ENHANCED with RAG validation while preserving working chat functionality
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


# NEW: Command validation models for Phase 6 RAG integration
class CommandValidationInfo(BaseModel):
    """Information about command validation using RAG"""

    command: str
    is_valid: bool
    confidence: float

    # Validation metadata
    validation_source: str = "rag"  # "rag", "fallback", or "cache"
    validation_time: float = 0.0

    # For invalid commands
    suggested_command: Optional[str] = None
    explanation: Optional[str] = None
    documentation_reference: Optional[str] = None

    # Error handling
    validation_error: Optional[str] = None
    fallback_used: bool = False


class CommandSuggestion(BaseModel):
    """Command suggestion for user interaction"""

    device_name: str
    original_command: str
    suggested_command: str
    vendor: str
    platform: str

    # Suggestion metadata
    confidence: float
    explanation: str
    documentation_reference: str = "Vendor documentation"

    # User interaction
    requires_confirmation: bool = True
    auto_apply_threshold: float = 0.95
    suggestion_id: str = Field(
        default_factory=lambda: f"suggestion_{int(time.time() * 1000)}"
    )


class UniversalRequest(BaseModel):
    """
    Universal request model that works for ALL interfaces
    Chat, API, CLI, Web - all use this same structure
    ENHANCED with RAG validation options
    """

    # Core request data (UNCHANGED)
    user_input: str = Field(..., description="Natural language request")
    devices: List[str] = Field(default=[], description="Target devices")

    # Interface specification (UNCHANGED)
    interface_type: InterfaceType = Field(default=InterfaceType.API)
    output_format: OutputFormat = Field(default=OutputFormat.JSON)

    # Request metadata (UNCHANGED)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    priority: RequestPriority = Field(default=RequestPriority.NORMAL)

    # Execution parameters (UNCHANGED)
    timeout: int = Field(default=30, ge=1, le=300)
    allow_discovery: bool = Field(default=True)
    force_fresh_discovery: bool = Field(default=False)
    include_raw_output: bool = Field(default=True)

    # Context and memory (UNCHANGED)
    conversation_context: Optional[Dict[str, Any]] = Field(default=None)
    previous_requests: Optional[List[str]] = Field(default=None)

    # Advanced options (UNCHANGED)
    parallel_execution: bool = Field(default=True)
    max_devices: int = Field(default=100, ge=1, le=1000)
    include_discovery_info: bool = Field(default=False)

    # Format-specific options (UNCHANGED)
    format_options: Optional[Dict[str, Any]] = Field(default=None)

    # NEW: RAG validation options (ADDED for Phase 6)
    enable_command_validation: bool = Field(
        default=True, description="Enable RAG-powered command validation"
    )
    auto_apply_suggestions: bool = Field(
        default=False,
        description="Auto-apply high-confidence suggestions without user confirmation",
    )
    suggestion_confidence_threshold: float = Field(
        default=0.95, description="Confidence threshold for auto-applying suggestions"
    )

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
    """Result from executing command on a single device - ENHANCED with validation"""

    device_name: str
    success: bool
    execution_time: float

    # Command info (ENHANCED with validation)
    command_executed: Optional[str] = None
    discovery_used: bool = False
    discovery_confidence: Optional[float] = None
    cached_discovery: bool = False

    # NEW: Command validation info
    validation_info: Optional[CommandValidationInfo] = None
    validation_bypassed: bool = False  # True if validation was skipped

    # Output data (UNCHANGED)
    raw_output: Optional[str] = None
    formatted_output: Optional[Any] = None

    # Error handling (UNCHANGED)
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Metadata (UNCHANGED)
    platform: Optional[str] = None
    vendor: Optional[str] = None
    connection_method: Optional[str] = None


class DiscoveryInfo(BaseModel):
    """Information about command discovery process - UNCHANGED"""

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
    Universal response model for ALL interfaces - ENHANCED with RAG validation
    Automatically adapts to chat, API, CLI, web, etc.
    """

    # Request correlation (UNCHANGED)
    request_id: str
    success: bool
    execution_time: float
    timestamp: float = Field(default_factory=time.time)

    # Core results (UNCHANGED)
    formatted_response: Any = Field(..., description="Response formatted for interface")
    total_devices: int = 0
    successful_devices: int = 0
    failed_devices: int = 0

    # Device results (ENHANCED with validation info)
    device_results: List[DeviceExecutionResult] = Field(default=[])

    # Discovery information (UNCHANGED)
    discovery_used: bool = False
    discovery_info: List[DiscoveryInfo] = Field(default=[])
    new_discoveries: int = 0  # Count of new command mappings learned

    # NEW: Command validation information for Phase 6
    validation_enabled: bool = False
    total_validations: int = 0
    validation_failures: int = 0
    suggestions_generated: List[CommandSuggestion] = Field(default=[])
    validation_bypassed_count: int = 0

    # NEW: Interactive suggestion handling
    requires_user_confirmation: bool = False
    pending_suggestions: List[CommandSuggestion] = Field(default=[])

    # Raw data (UNCHANGED)
    raw_device_outputs: Optional[Dict[str, str]] = Field(default=None)

    # Error handling (UNCHANGED)
    error_message: Optional[str] = None
    partial_success: bool = False

    # Performance metrics (ENHANCED with validation metrics)
    discovery_time: float = 0.0
    validation_time: float = 0.0  # NEW: Total validation time
    execution_time_per_device: Optional[Dict[str, float]] = None
    formatting_time: float = 0.0

    # Interface-specific metadata (UNCHANGED)
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
    """For processing multiple requests in batch - ENHANCED"""

    requests: List[UniversalRequest]
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    max_concurrent: int = Field(default=10, ge=1, le=50)
    fail_fast: bool = Field(default=False)

    # NEW: Batch validation options
    enable_batch_validation: bool = Field(
        default=True, description="Enable validation for all requests in batch"
    )


class BatchUniversalResponse(BaseModel):
    """Response for batch requests - ENHANCED"""

    batch_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    batch_execution_time: float
    responses: List[UniversalResponse]

    # NEW: Batch validation summary
    total_validations: int = 0
    validation_failures: int = 0
    total_suggestions: int = 0
    batch_requires_confirmation: bool = False


# Interface-specific request models (for type safety) - PRESERVED FROM ORIGINAL
class ChatRequest(UniversalRequest):
    """Chat-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.CHAT
    output_format: OutputFormat = OutputFormat.CONVERSATIONAL
    include_discovery_info: bool = True
    enable_command_validation: bool = True  # Enable validation for chat by default


class APIRequest(UniversalRequest):
    """API-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.API
    output_format: OutputFormat = OutputFormat.JSON
    include_raw_output: bool = True
    enable_command_validation: bool = True


class CLIRequest(UniversalRequest):
    """CLI-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.CLI
    output_format: OutputFormat = OutputFormat.TABLE
    include_discovery_info: bool = False
    enable_command_validation: bool = True


class WebRequest(UniversalRequest):
    """Web UI-specific request with defaults"""

    interface_type: InterfaceType = InterfaceType.WEB
    output_format: OutputFormat = OutputFormat.HTML
    include_discovery_info: bool = True
    enable_command_validation: bool = True


# Helper functions for request creation - PRESERVED FROM ORIGINAL
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
    devices: List[str] = None,
    output_format: OutputFormat = OutputFormat.JSON,
    include_raw: bool = True,
) -> APIRequest:
    """Helper to create API requests quickly"""
    return APIRequest(
        user_input=query,
        devices=devices or [],
        output_format=output_format,
        include_raw_output=include_raw,
    )


def create_cli_request(
    command: str,
    devices: List[str] = None,
    table_format: bool = True,
) -> CLIRequest:
    """Helper to create CLI requests quickly"""
    return CLIRequest(
        user_input=command,
        devices=devices or [],
        output_format=OutputFormat.TABLE if table_format else OutputFormat.JSON,
    )


# Validation and utility functions - ENHANCED with validation support
def validate_devices(devices: List[str]) -> List[str]:
    """Validate device list"""
    if not devices:
        return []

    # Remove duplicates and empty strings - FIXED for Ruff
    clean_devices = list({device.strip() for device in devices if device.strip()})

    # Limit to reasonable number
    if len(clean_devices) > 100:
        raise ValueError("Too many devices specified (max 100)")

    return clean_devices


def validate_request_size(request: UniversalRequest) -> None:
    """Validate request size constraints"""
    if len(request.user_input) > 1000:
        raise ValueError("User input too long (max 1000 characters)")

    if request.devices and len(request.devices) > request.max_devices:
        raise ValueError(f"Too many devices specified (max {request.max_devices})")


# NEW: Command validation utility functions
def create_command_suggestion(
    device_name: str,
    original_command: str,
    suggested_command: str,
    vendor: str,
    platform: str,
    confidence: float,
    explanation: str = "Command syntax correction",
    documentation_reference: str = "Vendor documentation",
) -> CommandSuggestion:
    """Helper function to create command suggestions"""

    return CommandSuggestion(
        device_name=device_name,
        original_command=original_command,
        suggested_command=suggested_command,
        vendor=vendor,
        platform=platform,
        confidence=confidence,
        explanation=explanation,
        documentation_reference=documentation_reference,
    )


def should_auto_apply_suggestion(
    suggestion: CommandSuggestion,
    auto_apply_enabled: bool = False,
    confidence_threshold: float = 0.95,
) -> bool:
    """Determine if suggestion should be auto-applied"""

    return (
        auto_apply_enabled
        and suggestion.confidence >= confidence_threshold
        and not suggestion.requires_confirmation
    )


def filter_high_confidence_suggestions(
    suggestions: List[CommandSuggestion],
    confidence_threshold: float = 0.8,
) -> List[CommandSuggestion]:
    """Filter suggestions by confidence threshold"""

    return [
        suggestion
        for suggestion in suggestions
        if suggestion.confidence >= confidence_threshold
    ]
