# app/routers/test_llm.py (new file)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.intents import NetworkIntent
from app.services.enhanced_output_normalizer import enhanced_normalizer

router = APIRouter()


class LLMTestRequest(BaseModel):
    raw_output: str
    intent: NetworkIntent
    vendor: str = "arista_eos"
    command: str = "test command"


@router.post("/test-normalization")
async def test_llm_normalization(request: LLMTestRequest):
    """Test LLM normalization with sample output"""

    if not enhanced_normalizer.llm_available:
        raise HTTPException(status_code=503, detail="Local LLM not available")

    result = await enhanced_normalizer.normalize_output(
        raw_output=request.raw_output,
        intent=request.intent,
        vendor=request.vendor,
        command=request.command,
        device_name="test-device",
    )

    return {
        "success": result.success,
        "method_used": result.method_used,
        "normalized_data": result.normalized_data,
        "processing_time": result.processing_time,
        "confidence_score": result.confidence_score,
        "error_message": result.error_message,
    }
