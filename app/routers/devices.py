# app/routers/devices.py
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.devices import Device, DeviceGroup, DeviceList
from app.services.inventory import inventory_service

router = APIRouter()


@router.get("/devices", response_model=DeviceList)
async def get_devices(
    site: Optional[str] = Query(None, description="Filter by site"),
    device_type: Optional[str] = Query(None, description="Filter by device type"),
    vendor: Optional[str] = Query(None, description="Filter by vendor"),
):
    """
    Get all devices from inventory with optional filtering
    """
    try:
        devices = inventory_service.get_all_devices()

        # Apply filters
        filtered_devices = devices  # Start with all devices
        if site:
            filtered_devices = [d for d in filtered_devices if d.site == site]
        if device_type:
            filtered_devices = [d for d in filtered_devices if d.device_type == device_type]
        if vendor:
            filtered_devices = [d for d in filtered_devices if d.vendor == vendor]

        return DeviceList(devices=filtered_devices, total_count=len(filtered_devices))

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Device inventory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading devices: {str(e)}")


@router.get("/devices/{device_name}", response_model=Device)
async def get_device(device_name: str):
    """
    Get a specific device by name
    """
    try:
        device = inventory_service.get_device(device_name)
        if not device:
            raise HTTPException(status_code=404, detail=f"Device '{device_name}' not found")
        return device

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving device: {str(e)}")


@router.get("/devices/groups/{group_name}", response_model=DeviceList)
async def get_devices_by_group(group_name: str):
    """
    Get devices that belong to a specific group
    """
    try:
        devices = inventory_service.get_devices_by_group(group_name)
        return DeviceList(devices=devices, total_count=len(devices))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving group devices: {str(e)}")


@router.get("/groups", response_model=List[DeviceGroup])
async def get_groups():
    """
    Get all device groups
    """
    try:
        groups = inventory_service.get_all_groups()
        return groups

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving groups: {str(e)}")


@router.post("/inventory/refresh")
async def refresh_inventory():
    """
    Refresh the device inventory cache
    """
    try:
        inventory_service.refresh_cache()
        return {"message": "Inventory cache refreshed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing inventory: {str(e)}")
