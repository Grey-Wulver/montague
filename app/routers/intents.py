# app/routers/intents.py
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.models.intents import IntentRequest, IntentResponse, NetworkIntent
from app.services.command_translator import command_translator
from app.services.intent_executor import intent_executor
from app.services.inventory import inventory_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/execute-intent", response_model=IntentResponse)
async def execute_intent(request: IntentRequest):
    """Execute a network intent across multiple devices (vendor-agnostic)"""
    try:
        result = await intent_executor.execute_intent(request)
        return result
    except Exception as e:
        logger.error(f"Intent execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Intent execution failed: {str(e)}")


# Convenience endpoints for common intents
@router.post("/system/version", response_model=IntentResponse)
async def get_system_version(devices: List[str]):
    """Get system version from devices (vendor-agnostic)"""
    request = IntentRequest(devices=devices, intent=NetworkIntent.GET_SYSTEM_VERSION)
    return await execute_intent(request)


@router.post("/interfaces/status", response_model=IntentResponse)
async def get_interface_status(devices: List[str]):
    """Get interface status from devices (vendor-agnostic)"""
    request = IntentRequest(devices=devices, intent=NetworkIntent.GET_INTERFACE_STATUS)
    return await execute_intent(request)


@router.post("/routing/bgp-summary", response_model=IntentResponse)
async def get_bgp_summary(devices: List[str]):
    """Get BGP summary from devices (vendor-agnostic)"""
    request = IntentRequest(devices=devices, intent=NetworkIntent.GET_BGP_SUMMARY)
    return await execute_intent(request)


@router.post("/network/arp-table", response_model=IntentResponse)
async def get_arp_table(devices: List[str]):
    """Get ARP table from devices (vendor-agnostic)"""
    request = IntentRequest(devices=devices, intent=NetworkIntent.GET_ARP_TABLE)
    return await execute_intent(request)


@router.post("/switching/mac-table", response_model=IntentResponse)
async def get_mac_table(devices: List[str]):
    """Get MAC address table from devices (vendor-agnostic)"""
    request = IntentRequest(devices=devices, intent=NetworkIntent.GET_MAC_TABLE)
    return await execute_intent(request)


# Discovery and capability endpoints
@router.get("/devices/{device_name}/supported-intents")
async def get_supported_intents(device_name: str):
    """Get list of supported intents for a specific device"""
    device = inventory_service.get_device(device_name)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    supported = command_translator.get_supported_intents(device)
    coverage = command_translator.get_platform_coverage(device.platform)

    return {
        "device": device_name,
        "platform": device.platform,
        "vendor": device.vendor,
        "supported_intents": [intent.value for intent in supported],
        "total_supported": len(supported),
        "coverage": coverage,
    }


@router.get("/intents")
async def list_all_intents():
    """Get list of all available network intents"""
    return {
        "available_intents": [
            {"intent": intent.value, "description": intent.value.replace("_", " ").title()} for intent in NetworkIntent
        ],
        "total_count": len(NetworkIntent),
    }


@router.get("/platforms")
async def list_supported_platforms():
    """Get list of all supported device platforms"""
    platforms = command_translator.get_all_platforms()
    return {"supported_platforms": platforms, "total_count": len(platforms)}


@router.get("/platforms/{platform}/coverage")
async def get_platform_coverage(platform: str):
    """Get intent coverage for a specific platform"""
    coverage = command_translator.get_platform_coverage(platform)
    supported_count = sum(1 for supported in coverage.values() if supported)

    return {
        "platform": platform,
        "coverage": coverage,
        "supported_intents": supported_count,
        "total_intents": len(coverage),
        "coverage_percentage": round((supported_count / len(coverage)) * 100, 1),
    }
