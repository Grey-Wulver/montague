# app/main.py
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

app = FastAPI(
    title="NetOps ChatBot API",
    description="AI-driven network automation platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
