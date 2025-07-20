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


@router.post("/debug-parsing")
async def debug_parsing():
    """Debug the JSON parsing issue with LM Studio"""
    test_response = """{
"router_identifier": "192.168.1.1",
"local_asn": 65001,
"neighbors": [{
"ip": "192.168.1.2",
"version": 4,
"remote_asn": 65002
}]
}"""

    try:
        import json

        from app.services.local_llm_normalizer import local_llm_normalizer

        # Test cleaning
        cleaned = local_llm_normalizer._clean_json_response(test_response)

        # Test parsing
        parsed = json.loads(cleaned)

        return {
            "status": "success",
            "original": repr(test_response),
            "cleaned": repr(cleaned),
            "parsed": parsed,
            "lm_studio_integration": "working",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


@router.post("/debug-lm-studio")
async def debug_lm_studio():
    """Debug LM Studio response parsing"""
    from app.services.local_llm_normalizer import local_llm_normalizer

    # The exact JSON that LM Studio returned
    test_json = """{\n    "router_id": "192.168.1.1",\n    "local_as": 65001,\n    "neighbors": [\n        {\n            "neighbor_ip": "192.168.1.2",\n            "remote_as": 65002,\n            "state": "up",\n            "uptime": "02:05:42",\n            "prefixes_received": 5,\n            "prefixes_sent": null\n        }\n    ],\n    "total_neighbors": 2,\n    "established_neighbors": 2,\n    "confidence_score": 1.0\n}"""

    try:
        import json

        # Test direct JSON parsing
        direct_parse = json.loads(test_json)

        # Test our cleaning function
        cleaned = local_llm_normalizer._clean_json_response(test_json)
        cleaned_parse = json.loads(cleaned)

        return {
            "status": "success",
            "direct_parse_works": True,
            "cleaning_works": True,
            "original_length": len(test_json),
            "cleaned_length": len(cleaned),
            "data": direct_parse,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_at": "cleaning" if "cleaned" in locals() else "direct_parse",
        }


@router.post("/debug-real-api")
async def debug_real_api():
    """Debug the actual LM Studio API call"""
    from app.models.intents import NetworkIntent
    from app.services.local_llm_normalizer import local_llm_normalizer

    # Test with the same BGP data
    test_output = """BGP router identifier 192.168.1.1, local AS number 65001
BGP table version is 45, main routing table version 45

Neighbor        V           AS MsgRcvd MsgSent   TblVer  InQ OutQ Up/Down  State/PfxRcd
192.168.1.2     4        65002     125     130       45    0    0 02:05:42        5
192.168.1.3     4        65003      89      94       45    0    0 01:45:22        3"""

    try:
        # Call the actual normalize_output method
        result = await local_llm_normalizer.normalize_output(
            raw_output=test_output,
            intent=NetworkIntent.GET_BGP_SUMMARY,
            vendor="arista_eos",
            command="show ip bgp summary",
        )

        return {
            "success": result.success,
            "error_message": result.error_message,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score,
            "normalized_data": result.normalized_data,
            "api_call_successful": True,
        }

    except Exception as e:
        return {
            "success": False,
            "error_message": str(e),
            "api_call_successful": False,
            "error_type": type(e).__name__,
        }


@router.post("/debug-lm-raw-response")
async def debug_lm_raw_response():
    """Debug the raw LM Studio API response"""

    import aiohttp

    payload = {
        "model": "meta-llama-3.1-8b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Convert this BGP output to JSON: BGP router identifier 192.168.1.1. Return only JSON starting with { and ending with }.",
            }
        ],
        "temperature": 0.1,
        "max_tokens": 500,
        "stream": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://192.168.1.11:1234/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    choices = result.get("choices", [])

                    if choices:
                        raw_content = choices[0].get("message", {}).get("content", "")

                        return {
                            "status": "success",
                            "raw_content": raw_content,
                            "raw_content_repr": repr(raw_content),
                            "content_length": len(raw_content),
                            "choices_count": len(choices),
                            "full_response": result,
                        }
                    else:
                        return {"status": "no_choices", "full_response": result}
                else:
                    return {"status": "api_error", "code": response.status}

    except Exception as e:
        return {"status": "exception", "error": str(e)}
