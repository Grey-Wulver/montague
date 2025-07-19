# app/routers/commands.py
import logging

from fastapi import APIRouter, HTTPException

from app.models.commands import CommandRequest, CommandResponse
from app.services.executor import network_executor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/execute", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """
    Execute a command on multiple network devices
    """
    try:
        logger.info(f"Executing command '{request.command}' on devices: {request.devices}")

        # Validate that we have devices
        if not request.devices:
            raise HTTPException(status_code=400, detail="No devices specified")

        # Execute the command
        result = await network_executor.execute_command(
            device_names=request.devices, command=request.command, timeout=request.timeout
        )

        logger.info(
            f"Command execution completed. Success: {result.successful}, Failed: {result.failed}"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        raise HTTPException(status_code=500, detail=f"Command execution failed: {str(e)}")


@router.get("/commands/common")
async def get_common_commands():
    """
    Get a list of common network commands for quick access
    """
    common_commands = {
        "status_checks": [
            "show version",
            "show interfaces status",
            "show ip interface brief",
            "show running-config",
            "show startup-config",
        ],
        "routing": [
            "show ip route",
            "show ip bgp summary",
            "show ip bgp neighbors",
            "show ip ospf neighbors",
        ],
        "switching": [
            "show mac address-table",
            "show vlan brief",
            "show spanning-tree",
            "show interfaces trunk",
        ],
        "diagnostics": ["show processes top", "show memory", "show logging", "show tech-support"],
    }

    return common_commands


# Convenience endpoints for common commands
@router.post("/show/version", response_model=CommandResponse)
async def show_version(devices: list[str]):
    """Execute 'show version' on specified devices"""
    request = CommandRequest(devices=devices, command="show version")
    return await execute_command(request)


@router.post("/show/interfaces", response_model=CommandResponse)
async def show_interfaces(devices: list[str]):
    """Execute 'show interfaces status' on specified devices"""
    request = CommandRequest(devices=devices, command="show interfaces status")
    return await execute_command(request)


@router.post("/show/bgp", response_model=CommandResponse)
async def show_bgp_summary(devices: list[str]):
    """Execute 'show ip bgp summary' on specified devices"""
    request = CommandRequest(devices=devices, command="show ip bgp summary")
    return await execute_command(request)
