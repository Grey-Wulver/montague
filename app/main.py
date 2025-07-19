# app/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# We'll add router imports later
# from app.routers import devices, commands, health

app = FastAPI(
    title="NetOps ChatBot API",
    description="AI-driven network automation platform",
    version="0.1.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc alternative
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basic health check endpoint to test
@app.get("/")
async def root():
    return {"message": "NetOps ChatBot API is running!", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "netops-chatbot"}


# Include routers (we'll uncomment these as we create them)
# app.include_router(health.router, prefix="/api/v1", tags=["health"])
# app.include_router(devices.router, prefix="/api/v1", tags=["devices"])
# app.include_router(commands.router, prefix="/api/v1", tags=["commands"])

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info",
    )
