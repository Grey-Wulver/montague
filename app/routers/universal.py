# app/routers/universal.py
"""
Universal API Router
Replaces specialized routers (intents.py, commands.py) with ONE universal endpoint
Handles ALL request types through the Universal Pipeline
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.universal_requests import (
    BatchUniversalRequest,
    BatchUniversalResponse,
    InterfaceType,
    OutputFormat,
    UniversalRequest,
    UniversalResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Universal Network Operations"])


# Initialize Universal Processor
universal_processor = None


async def get_universal_processor():
    """Get initialized universal processor"""
    global universal_processor
    if not universal_processor:
        from app.services.universal_request_processor import universal_processor as up

        universal_processor = up
        await universal_processor.initialize()
    return universal_processor


@router.post("/universal", response_model=UniversalResponse)
async def execute_universal_request(
    request: UniversalRequest, background_tasks: BackgroundTasks = None
) -> UniversalResponse:
    """
    🚀 THE UNIVERSAL ENDPOINT
    Handles ANY network operation request from ANY interface
    Replaces ALL specialized endpoints with one powerful, flexible API

    Examples:
    - Chat: {"user_input": "show version spine1", "interface_type": "chat"}
    - API: {"user_input": "get bgp summary", "devices": ["spine1"], "output_format": "json"}
    - CLI: {"user_input": "show interfaces", "devices": ["all"], "output_format": "table"}
    """

    try:
        logger.info(
            f"🌐 Universal API request: '{request.user_input}' via {request.interface_type.value}"
        )

        processor = await get_universal_processor()
        response = await processor.process_request(request)

        # Handle background tasks if needed
        if background_tasks and not response.success:
            background_tasks.add_task(log_failed_request, request, response)

        return response

    except Exception as e:
        logger.error(f"Universal API error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Request processing failed: {str(e)}"
        ) from e


@router.post("/universal/batch", response_model=BatchUniversalResponse)
async def execute_batch_requests(
    batch_request: BatchUniversalRequest,
) -> BatchUniversalResponse:
    """
    Execute multiple requests in batch
    Useful for bulk operations or complex workflows
    """

    try:
        logger.info(f"📦 Batch request: {len(batch_request.requests)} requests")

        processor = await get_universal_processor()

        # Process requests with concurrency control
        import asyncio

        semaphore = asyncio.Semaphore(batch_request.max_concurrent)

        async def process_single_request(req: UniversalRequest) -> UniversalResponse:
            async with semaphore:
                return await processor.process_request(req)

        # Execute all requests
        start_time = time.time()

        if batch_request.fail_fast:
            # Stop on first failure
            responses = []
            for req in batch_request.requests:
                response = await process_single_request(req)
                responses.append(response)
                if not response.success:
                    break
        else:
            # Execute all regardless of failures
            tasks = [process_single_request(req) for req in batch_request.requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    # Create error response
                    error_response = UniversalResponse(
                        request_id=batch_request.requests[i].request_id,
                        success=False,
                        execution_time=0.0,
                        formatted_response=f"Error: {str(response)}",
                        error_message=str(response),
                    )
                    processed_responses.append(error_response)
                else:
                    processed_responses.append(response)
            responses = processed_responses

        batch_execution_time = time.time() - start_time

        # Build batch response
        successful_count = sum(1 for r in responses if r.success)
        failed_count = len(responses) - successful_count

        return BatchUniversalResponse(
            batch_id=batch_request.batch_id,
            total_requests=len(batch_request.requests),
            successful_requests=successful_count,
            failed_requests=failed_count,
            batch_execution_time=batch_execution_time,
            responses=responses,
        )

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        ) from e


# Convenience endpoints for different interfaces


@router.post("/chat")
async def chat_interface(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chat interface endpoint
    Simplified endpoint that mimics chat bot interaction
    """

    # Extract parameters from request body
    message = request.get("message")
    devices = request.get("devices", [])
    user_id = request.get("user_id")
    session_id = request.get("session_id")

    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' in request")

    universal_request = UniversalRequest(
        user_input=message,
        devices=devices or [],
        interface_type=InterfaceType.CHAT,
        output_format=OutputFormat.CONVERSATIONAL,
        user_id=user_id,
        session_id=session_id,
        include_discovery_info=True,
    )

    response = await execute_universal_request(universal_request)

    # Return chat-friendly format
    return {
        "success": response.success,
        "message": response.formatted_response,
        "devices_processed": response.total_devices,
        "execution_time": response.execution_time,
        "discovery_used": response.discovery_used,
        "new_discoveries": response.new_discoveries,
    }


@router.get("/query")
async def query_interface(
    q: str,
    devices: str = "",
    format: str = "json",
    timeout: int = 30,
    include_raw: bool = False,
) -> UniversalResponse:
    """
    GET-based query interface
    Simple URL-based queries for quick testing

    Example: /api/v1/query?q=show version&devices=spine1,leaf1&format=json
    """

    device_list = (
        [d.strip() for d in devices.split(",") if d.strip()] if devices else []
    )

    # Validate format
    try:
        output_format = OutputFormat(format)
    except ValueError:
        output_format = OutputFormat.JSON

    # Validate timeout
    if timeout < 1:
        timeout = 1
    elif timeout > 300:
        timeout = 300

    request = UniversalRequest(
        user_input=q,
        devices=device_list,
        interface_type=InterfaceType.API,
        output_format=output_format,
        timeout=timeout,
        include_raw_output=include_raw,
    )

    return await execute_universal_request(request)


@router.post("/cli")
async def cli_interface(request: Dict[str, Any]) -> PlainTextResponse:
    """
    CLI interface endpoint
    Returns plain text suitable for command line tools
    """

    # Extract parameters from request body
    command = request.get("command")
    devices = request.get("devices", [])
    table_format = request.get("table_format", True)
    timeout = request.get("timeout", 30)

    if not command:
        raise HTTPException(status_code=400, detail="Missing 'command' in request")

    if not devices:
        raise HTTPException(status_code=400, detail="Missing 'devices' in request")

    universal_request = UniversalRequest(
        user_input=command,
        devices=devices,
        interface_type=InterfaceType.CLI,
        output_format=OutputFormat.TABLE if table_format else OutputFormat.JSON,
        timeout=timeout,
    )

    response = await execute_universal_request(universal_request)

    # Return plain text response
    if isinstance(response.formatted_response, str):
        content = response.formatted_response
    else:
        content = str(response.formatted_response)

    return PlainTextResponse(content=content)


@router.post("/webhook")
async def webhook_interface(
    request: Dict[str, Any], background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Webhook interface for external system integration
    Handles incoming webhook payloads and converts to network operations
    """

    try:
        # Extract request from webhook payload
        user_input = request.get("query", request.get("message", ""))
        devices = request.get("devices", [])
        format_type = request.get("format", "json")

        if not user_input:
            raise HTTPException(
                status_code=400, detail="Missing 'query' or 'message' in payload"
            )

        universal_request = UniversalRequest(
            user_input=user_input,
            devices=devices,
            interface_type=InterfaceType.WEBHOOK,
            output_format=OutputFormat(format_type),
            timeout=request.get("timeout", 30),
        )

        response = await execute_universal_request(universal_request, background_tasks)

        # Return webhook-friendly response
        return {
            "webhook_id": request.get("id"),
            "success": response.success,
            "result": response.formatted_response,
            "metadata": {
                "execution_time": response.execution_time,
                "devices_processed": response.total_devices,
                "discovery_used": response.discovery_used,
            },
        }

    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Status and monitoring endpoints


@router.get("/status")
async def universal_status() -> Dict[str, Any]:
    """Get universal processor status and statistics"""

    try:
        processor = await get_universal_processor()

        # Get processor health and stats
        health = await processor.health_check()
        stats = processor.get_stats()

        return {
            "status": "healthy"
            if health.get("universal_processor") == "healthy"
            else "degraded",
            "health_check": health,
            "statistics": stats,
            "supported_interfaces": [interface.value for interface in InterfaceType],
            "supported_formats": [format.value for format in OutputFormat],
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Get platform capabilities and supported operations"""

    try:
        from app.services.intent_parser import intent_parser

        return {
            "universal_pipeline": "active",
            "supported_interfaces": {
                "chat": "Natural language chat interface",
                "api": "Programmatic API access",
                "cli": "Command line interface",
                "web": "Web UI interface",
                "webhook": "External system integration",
            },
            "supported_formats": {
                "json": "Structured JSON data",
                "yaml": "YAML format",
                "xml": "XML format",
                "csv": "Comma-separated values",
                "table": "ASCII table format",
                "html": "HTML for web display",
                "markdown": "Markdown format",
                "conversational": "Natural language responses",
                "raw": "Raw device output",
            },
            "discovery_capabilities": {
                "dynamic_discovery": "AI-powered command discovery",
                "multi_vendor": "Universal vendor support",
                "intelligent_caching": "Learning and optimization",
                "safety_validation": "Automatic safety checks",
            },
            "supported_operations": intent_parser.get_supported_operations(),
        }

    except Exception as e:
        return {"error": f"Failed to get capabilities: {str(e)}"}


# Helper functions


def log_failed_request(request: UniversalRequest, response: UniversalResponse):
    """Background task to log failed requests for analysis"""
    logger.warning(
        f"Failed request: {request.user_input} | "
        f"Interface: {request.interface_type.value} | "
        f"Error: {response.error_message}"
    )


# Export router
__all__ = ["router"]
