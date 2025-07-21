# app/routers/discovery.py
"""
Discovery management endpoints for NetOps ChatBot
Provides access to discovery stats and capabilities
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/ntc-templates")
async def get_ntc_templates() -> Dict[str, Any]:
    """Get available ntc-templates parsers"""
    try:
        from app.services.community_mapper import community_mapper

        templates = community_mapper.get_ntc_templates()
        return {
            "total_templates": len(templates),
            "platforms": list({t["platform"] for t in templates}),
            "sample_templates": templates[:10],  # First 10 for preview
        }
    except ImportError:
        return {
            "error": "ntc-templates not available",
            "total_templates": 0,
            "platforms": [],
        }
    except Exception as e:
        logger.error(f"Error getting ntc-templates: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/genie-parsers")
async def get_genie_parsers() -> Dict[str, Any]:
    """Get available Genie parsers"""
    try:
        from app.services.community_mapper import community_mapper

        parsers = community_mapper.get_genie_parsers()
        return {
            "total_parsers": len(parsers),
            "platforms": list({p["platform"] for p in parsers}),
            "sample_parsers": parsers[:10],  # First 10 for preview
        }
    except ImportError:
        return {
            "error": "Genie parsers not available",
            "total_parsers": 0,
            "platforms": [],
        }
    except Exception as e:
        logger.error(f"Error getting Genie parsers: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/napalm-getters")
async def get_napalm_getters() -> Dict[str, Any]:
    """Get available NAPALM getters"""
    try:
        from app.services.community_mapper import community_mapper

        getters = community_mapper.get_napalm_getters()
        return {
            "total_getters": len(getters),
            "platforms": list({g["platform"] for g in getters}),
            "sample_getters": getters[:10],  # First 10 for preview
        }
    except ImportError:
        return {
            "error": "NAPALM getters not available",
            "total_getters": 0,
            "platforms": [],
        }
    except Exception as e:
        logger.error(f"Error getting NAPALM getters: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/combined-mappings")
async def get_combined_mappings() -> Dict[str, Any]:
    """Get combined command mappings from all sources"""
    try:
        from app.services.community_mapper import community_mapper

        mappings = community_mapper.get_combined_mappings()
        return {
            "total_mappings": len(mappings),
            "sources": list({m["source"] for m in mappings}),
            "platforms": list({m["platform"] for m in mappings}),
            "sample_mappings": mappings[:20],  # First 20 for preview
        }
    except Exception as e:
        logger.error(f"Error getting combined mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/export-mappings")
async def export_mappings(format: str = "json") -> Dict[str, Any]:
    """Export command mappings in specified format"""
    try:
        from app.services.community_mapper import community_mapper

        if format not in ["json", "yaml", "csv"]:
            raise HTTPException(
                status_code=400, detail="Format must be json, yaml, or csv"
            )

        mappings = community_mapper.export_mappings(format)
        return {
            "format": format,
            "mappings": mappings,
            "export_time": "2024-01-01T00:00:00Z",  # Add actual timestamp
        }
    except Exception as e:
        logger.error(f"Error exporting mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/platform-commands/{platform}")
async def get_platform_commands(platform: str) -> Dict[str, Any]:
    """Get available commands for a specific platform"""
    try:
        from app.services.community_mapper import community_mapper

        commands = community_mapper.get_platform_commands(platform)
        return {
            "platform": platform,
            "total_commands": len(commands),
            "commands": commands,
        }
    except Exception as e:
        logger.error(f"Error getting platform commands: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats")
async def get_discovery_stats() -> Dict[str, Any]:
    """Get discovery statistics with fallback handling"""
    try:
        # Try to get dynamic discovery stats
        try:
            from app.services.dynamic_command_discovery import dynamic_discovery

            if dynamic_discovery:
                discovery_stats = dynamic_discovery.get_discovery_stats()
                cache_stats = dynamic_discovery.get_cache_stats()
                return {
                    "discovery_statistics": discovery_stats,
                    "cache_statistics": cache_stats,
                    "service_status": "active",
                }
        except ImportError:
            pass

        # Fallback response
        return {
            "discovery_statistics": {
                "total_discoveries": 0,
                "successful_discoveries": 0,
                "cache_hits": 0,
                "success_rate_percent": 0.0,
            },
            "cache_statistics": {
                "cached_mappings": 0,
                "total_cache_usage": 0,
                "cache_entries": [],
            },
            "service_status": "not_initialized",
        }

    except Exception as e:
        logger.error(f"Error getting discovery stats: {e}")
        return {
            "error": str(e),
            "service_status": "error",
        }


@router.get("/test")
async def test_discovery() -> Dict[str, Any]:
    """Test discovery functionality"""
    try:
        from app.services.dynamic_command_discovery import dynamic_discovery
        from app.services.inventory import inventory_service

        if not dynamic_discovery:
            return {
                "test_status": "failed",
                "error": "Dynamic discovery not initialized",
            }

        # Get first device for testing
        devices = inventory_service.get_all_devices()
        if not devices:
            return {
                "test_status": "failed",
                "error": "No devices available for testing",
            }

        test_device = devices[0]

        # Test discovery with a simple intent
        result = await dynamic_discovery.discover_command(
            user_intent="show version", device=test_device, allow_cached=False
        )

        return {
            "test_status": "success" if result.success else "failed",
            "test_device": test_device.name,
            "test_intent": "show version",
            "discovered_command": result.command if result.success else None,
            "confidence": result.confidence if result.success else 0.0,
            "error_message": result.error_message if not result.success else None,
        }

    except Exception as e:
        logger.error(f"Discovery test failed: {e}")
        return {
            "test_status": "error",
            "error": str(e),
        }
