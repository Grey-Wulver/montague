# app/models/commands.py
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CommandStatus(str, Enum):
    """Command execution status"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class OutputFormat(str, Enum):
    """Output format options"""

    RAW = "raw"  # Raw text output
    STRUCTURED = "structured"  # Parsed JSON output
    BOTH = "both"  # Both raw and structured


class CommandRequest(BaseModel):
    """Request model for executing commands"""

    devices: List[str] = Field(..., description="List of device names to execute on")
    command: str = Field(..., description="Command to execute")
    output_format: OutputFormat = Field(
        default=OutputFormat.STRUCTURED, description="Output format"
    )
    timeout: int = Field(default=30, ge=5, le=300, description="Timeout in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "devices": ["spine1", "leaf1"],
                "command": "show version",
                "output_format": "structured",
                "timeout": 30,
            }
        }


class DeviceCommandResult(BaseModel):
    """Result from a single device"""

    device: str = Field(..., description="Device name")
    command: str = Field(..., description="Command executed")
    status: CommandStatus = Field(..., description="Execution status")
    raw_output: Optional[str] = Field(None, description="Raw command output")
    structured_output: Optional[Dict[str, Any]] = Field(
        None, description="Parsed output"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(
        None, description="Execution time in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When command was executed"
    )


class CommandResponse(BaseModel):
    """Response model for command execution"""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique request ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Request timestamp"
    )
    total_devices: int = Field(..., description="Total number of devices")
    successful: int = Field(..., description="Number of successful executions")
    failed: int = Field(..., description="Number of failed executions")
    results: List[DeviceCommandResult] = Field(
        ..., description="Individual device results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-01-15T10:30:00Z",
                "total_devices": 2,
                "successful": 2,
                "failed": 0,
                "results": [
                    {
                        "device": "spine1",
                        "command": "show version",
                        "status": "success",
                        "structured_output": {"version": "4.27.3F"},
                        "execution_time": 1.23,
                    }
                ],
            }
        }
