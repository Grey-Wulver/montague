# File: ~/net-chatbot/app/models/universal_requests.py
# ENHANCED UNIVERSAL MODELS - Add Command Validation Support for Phase 6

"""
Enhanced Universal Request Models - Command Validation Support

Added for Phase 6 Smart Command Suggestions:
- CommandValidationInfo for tracking validation results
- CommandSuggestion for user interaction
- Enhanced UniversalResponse with validation metadata
- Maintains all existing functionality while adding RAG validation support

Maintains VS Code + Ruff + Black standards with 88-character line length.
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Existing enums (UNCHANGED)
class InterfaceType(str, Enum):
    """Interface type for universal requests"""

    CHAT = "chat"
    API = "api"
    CLI = "cli"
    WEB = "web"
    WEBHOOK = "webhook"


class OutputFormat(str, Enum):
    """Output format for universal responses"""

    JSON = "json"
    TABLE = "table"
    CONVERSATIONAL = "conversational"
    RAW = "raw"
    MARKDOWN = "markdown"


# NEW: Command validation models for Phase 6
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


# Existing models (ENHANCED with validation support)
class DeviceExecutionResult(BaseModel):
    """Result from executing command on a single device - ENHANCED"""

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


class UniversalRequest(BaseModel):
    """Universal request model for ALL interfaces - ENHANCED"""

    # Core request data (UNCHANGED)
    user_input: str = Field(..., description="Natural language input from user")
    interface_type: InterfaceType = Field(..., description="Interface requesting this")
    output_format: OutputFormat = Field(default=OutputFormat.CONVERSATIONAL)

    # Device and execution options (UNCHANGED)
    devices: Optional[List[str]] = Field(default=None)
    max_devices: int = Field(default=50, le=100)
    include_raw_output: bool = Field(default=False)

    # NEW: Command validation options for Phase 6
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

    # Request metadata (UNCHANGED)
    request_id: str = Field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    timeout: int = Field(default=30, le=300)

    # Format options (UNCHANGED)
    format_options: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        schema_extra = {
            "examples": [
                {
                    "user_input": "show interfaces status on spine1",
                    "interface_type": "api",
                    "output_format": "json",
                    "devices": ["spine1"],
                    "enable_command_validation": True,
                    "auto_apply_suggestions": False,
                },
                {
                    "user_input": "check bgp neighbors for all spine devices",
                    "devices": ["all"],
                    "interface_type": "api",
                    "output_format": "json",
                    "include_raw_output": True,
                    "enable_command_validation": True,
                },
            ]
        }


class UniversalResponse(BaseModel):
    """Universal response model for ALL interfaces - ENHANCED with validation"""

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
    new_discoveries: int = 0

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
                    "formatted_response": "Interface Status: Ethernet1/1 is up",
                    "total_devices": 1,
                    "successful_devices": 1,
                    "validation_enabled": True,
                    "total_validations": 1,
                    "validation_failures": 0,
                    "discovery_used": True,
                    "new_discoveries": 0,
                },
                {
                    "request_id": "def456",
                    "success": False,
                    "execution_time": 1.23,
                    "formatted_response": "Command validation failed - suggestions available",
                    "total_devices": 1,
                    "successful_devices": 0,
                    "validation_enabled": True,
                    "validation_failures": 1,
                    "requires_user_confirmation": True,
                    "suggestions_generated": [
                        {
                            "device_name": "spine1",
                            "original_command": "show interfaces state",
                            "suggested_command": "show interfaces status",
                            "confidence": 0.98,
                            "explanation": "Correct Arista EOS syntax for interface status",
                        }
                    ],
                },
            ]
        }


class BatchUniversalRequest(BaseModel):
    """For processing multiple requests in batch - ENHANCED"""

    requests: List[UniversalRequest] = Field(..., max_items=10)
    batch_id: str = Field(default_factory=lambda: f"batch_{int(time.time() * 1000)}")

    # NEW: Batch validation options
    enable_batch_validation: bool = Field(
        default=True, description="Enable validation for all requests in batch"
    )
    fail_fast: bool = Field(
        default=False, description="Stop batch processing on first validation failure"
    )


class BatchUniversalResponse(BaseModel):
    """Response from batch processing - ENHANCED"""

    batch_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    execution_time: float

    # Individual responses (ENHANCED with validation info)
    responses: List[UniversalResponse] = Field(default=[])

    # NEW: Batch validation summary
    total_validations: int = 0
    validation_failures: int = 0
    total_suggestions: int = 0
    batch_requires_confirmation: bool = False


# Validation and utility functions (UNCHANGED + NEW)
def validate_devices(devices: List[str]) -> List[str]:
    """Validate device list"""
    if not devices:
        return []

    # Remove duplicates and empty strings
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
