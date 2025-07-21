# app/main.py
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.enhanced_output_normalizer import enhanced_normalizer

load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Specifically enable debug for our LLM service
logging.getLogger("app.services.local_llm_normalizer").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events"""
    # Startup
    print("🚀 Starting NetOps ChatBot API...")

    # Initialize LLM normalizer (your existing code)
    llm_available = await enhanced_normalizer.initialize()
    if llm_available:
        print("✅ Local LLM initialized successfully")
    else:
        print("⚠️  Local LLM not available - using manual parsers only")

    # NEW: Initialize ChatOps handler
    try:
        from app.services.chat_handler import chat_handler

        # Start chat listening in background task
        asyncio.create_task(chat_handler.start_listening())
        print("✅ ChatOps handler started - listening for messages")
    except Exception as e:
        print(f"⚠️  ChatOps handler failed to start: {e}")

    print("🎯 NetOps ChatBot API ready!")

    yield  # Application runs here

    # Shutdown
    print("🔄 Shutting down NetOps ChatBot API...")
    # NEW: Stop chat handler
    try:
        from app.services.chat_handler import chat_handler

        await chat_handler.stop_listening()
    except Exception as e:
        print(f"⚠️  Error stopping chat handler: {e}")


app = FastAPI(
    title="NetOps ChatBot API",
    description="AI-driven network automation platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # Add the lifespan context manager
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basic endpoints
@app.get("/")
async def root():
    return {"message": "NetOps ChatBot API is running!", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "netops-chatbot"}


# LLM status endpoints
@app.get("/api/v1/llm/status", tags=["llm-testing"])
async def llm_status():
    """Get Local LLM status and statistics"""
    return {
        "llm_available": enhanced_normalizer.llm_available,
        "normalization_stats": enhanced_normalizer.get_stats(),
        "service": "netops-chatbot-llm",
    }


@app.post("/api/v1/llm/clear-cache", tags=["llm-testing"])
async def clear_llm_cache():
    """Clear LLM normalization cache"""
    cleared_entries = enhanced_normalizer.clear_cache()
    return {"message": f"Cleared {cleared_entries} cache entries", "cache_size": 0}


# Debug endpoints for LLM troubleshooting
@app.post("/api/v1/debug/json-parse", tags=["llm-testing"])
async def debug_json_parse():
    """Debug the exact JSON parsing issue"""
    from app.services.local_llm_normalizer import local_llm_normalizer

    # Simulate the exact LLM response we got
    test_response = """{
"router_identifier": "192.168.1.1",
"local_asn": 65001,
"neighbors": [{
"ip": "192.168.1.2",
"version": 4,
"remote_asn": 65002,
"messages_received": 125,
"messages_sent": 130,
"table_version": 45,
"input_queue": 0,
"output_queue": 0,
"up_down": "02:05:42",
"state": 5
}]
}"""

    try:
        # Test the cleaning function
        cleaned = local_llm_normalizer._clean_json_response(test_response)

        # Test JSON parsing
        parsed = json.loads(cleaned)

        return {
            "status": "success",
            "original_length": len(test_response),
            "cleaned_length": len(cleaned),
            "original_first_10": repr(test_response[:10]),
            "cleaned_first_10": repr(cleaned[:10]),
            "parsed_successfully": True,
            "parsed_data": parsed,
        }

    except json.JSONDecodeError as e:
        return {
            "status": "json_error",
            "error": str(e),
            "error_position": getattr(e, "pos", None),
            "cleaned_response": repr(cleaned) if "cleaned" in locals() else None,
            "problem_char": repr(cleaned[e.pos : e.pos + 10])
            if "cleaned" in locals() and hasattr(e, "pos")
            else None,
        }
    except Exception as e:
        return {
            "status": "other_error",
            "error": str(e),
            "error_type": type(e).__name__,
        }


@app.post("/api/v1/debug/llm-raw", tags=["llm-testing"])
async def debug_llm_raw():
    """Debug LLM raw response from Ollama API"""
    payload = {
        "model": "codellama:13b",
        "prompt": "Convert this to JSON: BGP router identifier 192.168.1.1. Return only JSON.",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 500},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate", json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    raw_response = result.get("response", "")

                    return {
                        "status": "success",
                        "raw_response": raw_response,
                        "raw_response_repr": repr(raw_response),
                        "response_length": len(raw_response),
                        "first_char": repr(raw_response[0]) if raw_response else None,
                        "contains_newlines": "\n" in raw_response,
                        "starts_with_brace": raw_response.strip().startswith("{")
                        if raw_response
                        else False,
                        "ends_with_brace": raw_response.strip().endswith("}")
                        if raw_response
                        else False,
                    }
                else:
                    return {"status": "error", "code": response.status}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


@app.get("/api/v1/debug/normalizer-state", tags=["llm-testing"])
async def debug_normalizer_state():
    """Debug the current state of the normalizer"""
    try:
        from app.services.enhanced_output_normalizer import enhanced_normalizer
        from app.services.local_llm_normalizer import local_llm_normalizer

        # Test health check
        health = await local_llm_normalizer.health_check()

        return {
            "enhanced_normalizer_available": enhanced_normalizer.llm_available,
            "local_llm_health_check": health,
            "enhanced_normalizer_stats": enhanced_normalizer.get_stats(),
            "local_llm_stats": local_llm_normalizer.get_stats(),
            "imports_successful": True,
        }
    except Exception as e:
        return {
            "imports_successful": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Include routers with error handling
try:
    from app.routers import devices

    app.include_router(devices.router, prefix="/api/v1", tags=["devices"])
    print("✅ Successfully included devices router")
except Exception as e:
    print(f"❌ Failed to import devices router: {e}")

try:
    from app.routers import commands

    app.include_router(commands.router, prefix="/api/v1", tags=["commands"])
    print("✅ Successfully included commands router")
except Exception as e:
    print(f"❌ Failed to import commands router: {e}")

try:
    from app.routers import intents

    app.include_router(intents.router, prefix="/api/v1/intents", tags=["intents"])
    print("✅ Successfully included intents router")
except Exception as e:
    print(f"❌ Failed to import intents router: {e}")

try:
    from app.routers import discovery

    app.include_router(discovery.router, prefix="/api/v1/discovery", tags=["discovery"])
    print("✅ Successfully included discovery router")
except Exception as e:
    print(f"❌ Failed to import discovery router: {e}")

try:
    from app.routers import test_llm

    app.include_router(test_llm.router, prefix="/api/v1/llm", tags=["llm-testing"])
    print("✅ Successfully included LLM testing router")
except Exception as e:
    print(f"❌ Failed to import LLM testing router: {e}")

try:
    from app.routers import test_mattermost

    app.include_router(test_mattermost.router, prefix="/api/v1", tags=["chat"])
    print("✅ Successfully included Mattermost testing router")
except Exception as e:
    print(f"❌ Failed to import Mattermost testing router: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
