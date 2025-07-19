# app/routers/discovery.py
import logging

from fastapi import APIRouter, HTTPException

from app.services.community_mapper import community_mapper

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/community/ntc-templates")
async def get_ntc_templates():
    """Get all commands discovered from ntc-templates"""
    try:
        mappings = community_mapper.harvest_ntc_templates()

        stats = {
            "total_platforms": len(mappings),
            "total_commands": sum(len(commands) for commands in mappings.values()),
            "platforms": {platform: len(commands) for platform, commands in mappings.items()},
        }

        return {
            "source": "ntc-templates",
            "statistics": stats,
            "mappings": {platform: list(commands) for platform, commands in mappings.items()},
        }

    except Exception as e:
        logger.error(f"Error getting ntc-templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/genie-parsers")
async def get_genie_parsers():
    """Get all commands discovered from Genie parsers"""
    try:
        mappings = community_mapper.harvest_genie_parsers()

        stats = {
            "total_platforms": len(mappings),
            "total_commands": sum(len(commands) for commands in mappings.values()),
            "platforms": {platform: len(commands) for platform, commands in mappings.items()},
        }

        return {
            "source": "genie-parsers",
            "statistics": stats,
            "mappings": {platform: list(commands) for platform, commands in mappings.items()},
        }

    except Exception as e:
        logger.error(f"Error getting Genie parsers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/napalm-getters")
async def get_napalm_getters():
    """Get NAPALM standardized getters (intent-mapped!)"""
    try:
        getters = community_mapper.get_napalm_getters()

        stats = {
            "total_getters": len(getters),
            "total_platforms": len(set().union(*[g.keys() for g in getters.values()])),
            "platforms": list(set().union(*[g.keys() for g in getters.values()])),
        }

        return {"source": "napalm-getters", "statistics": stats, "getters": getters}

    except Exception as e:
        logger.error(f"Error getting NAPALM getters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/combined")
async def get_combined_mappings():
    """Get all community mappings combined"""
    try:
        mappings = community_mapper.get_combined_mappings()

        stats = {
            "total_platforms": len(mappings),
            "total_commands": sum(len(commands) for commands in mappings.values()),
            "platforms": {platform: len(commands) for platform, commands in mappings.items()},
        }

        return {
            "source": "combined-community",
            "statistics": stats,
            "mappings": {platform: list(commands) for platform, commands in mappings.items()},
        }

    except Exception as e:
        logger.error(f"Error getting combined mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/community/export")
async def export_community_mappings():
    """Export all community mappings to JSON file"""
    try:
        filename = community_mapper.export_to_json()

        return {"message": "Community mappings exported successfully", "filename": filename}

    except Exception as e:
        logger.error(f"Error exporting mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/platform/{platform}")
async def get_platform_commands(platform: str):
    """Get all commands for a specific platform"""
    try:
        combined_mappings = community_mapper.get_combined_mappings()

        if platform not in combined_mappings:
            raise HTTPException(status_code=404, detail=f"Platform '{platform}' not found")

        commands = list(combined_mappings[platform])

        return {"platform": platform, "total_commands": len(commands), "commands": sorted(commands)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting platform commands: {e}")
        raise HTTPException(status_code=500, detail=str(e))
