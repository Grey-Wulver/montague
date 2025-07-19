# app/services/local_llm_normalizer.py

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

from app.models.intents import NetworkIntent

logger = logging.getLogger(__name__)


@dataclass
class LLMNormalizationResult:
    """Result from LLM normalization attempt"""

    success: bool
    normalized_data: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    confidence_score: float = 0.0


class LocalLLMNormalizer:
    """
    Local LLM service for network command output normalization.
    Integrates with Ollama running on the same VM.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "codellama:13b",
        timeout: int = 30,
        max_retries: int = 2,
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        # Cache for prompt templates to avoid repeated generation
        self._prompt_cache = {}

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_normalizations": 0,
            "failed_normalizations": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
        }

    async def health_check(self) -> bool:
        """Check if Ollama service is available and model is loaded"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        return self.model_name in models
            return False
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False

    def _get_normalization_prompt(self, intent: NetworkIntent, vendor: str, raw_output: str) -> str:
        """Generate intent-specific normalization prompt"""

        # Cache key for prompt template
        cache_key = f"{intent.value}_{vendor}"

        if cache_key in self._prompt_cache:
            self.stats["cache_hits"] += 1
            base_prompt = self._prompt_cache[cache_key]
        else:
            base_prompt = self._generate_prompt_template(intent, vendor)
            self._prompt_cache[cache_key] = base_prompt

        return base_prompt.format(raw_output=raw_output)

    def _generate_prompt_template(self, intent: NetworkIntent, vendor: str) -> str:
        """Generate intent-specific prompt templates"""

        common_instructions = """
You are a network automation expert. Convert the raw network device output into clean, structured JSON.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no explanations, no markdown, no extra text
2. Use consistent field names across all vendors
3. Handle missing data with null values
4. Ensure all numeric values are properly typed (int/float, not strings)
5. Standardize status values (up/down/admin-down)
6. Include confidence score (0.0-1.0) based on parsing accuracy

"""

        intent_prompts = {
            NetworkIntent.GET_SYSTEM_VERSION: f"""
{common_instructions}

Convert this {vendor} 'show version' output to standardized JSON:

Expected JSON structure:
{{
    "vendor": "string",
    "hostname": "string",
    "os_version": "string",
    "model": "string",
    "serial_number": "string",
    "uptime_seconds": integer,
    "memory_total_kb": integer,
    "memory_free_kb": integer,
    "last_reload_reason": "string",
    "confidence_score": float
}}

Raw Output:
{{raw_output}}

Return only valid JSON:""",
            NetworkIntent.GET_INTERFACE_STATUS: f"""
{common_instructions}

Convert this {vendor} interface status output to standardized JSON:

Expected JSON structure:
{{
    "interfaces": [
        {{
            "interface": "string",
            "status": "up|down|admin-down",
            "protocol": "up|down",
            "description": "string|null",
            "speed": "string|null",
            "duplex": "string|null",
            "mtu": integer|null,
            "ip_address": "string|null",
            "subnet_mask": "string|null",
            "vlan": integer|null
        }}
    ],
    "total_interfaces": integer,
    "up_interfaces": integer,
    "down_interfaces": integer,
    "confidence_score": float
}}

Raw Output:
{{raw_output}}

Return only valid JSON:""",
            NetworkIntent.GET_BGP_SUMMARY: f"""
{common_instructions}

Convert this {vendor} BGP summary output to standardized JSON:

Expected JSON structure:
{{
    "router_id": "string",
    "local_as": integer,
    "neighbors": [
        {{
            "neighbor_ip": "string",
            "remote_as": integer,
            "state": "string",
            "uptime": "string",
            "prefixes_received": integer,
            "prefixes_sent": integer
        }}
    ],
    "total_neighbors": integer,
    "established_neighbors": integer,
    "confidence_score": float
}}

Raw Output:
{{raw_output}}

Return only valid JSON:""",
            NetworkIntent.GET_ARP_TABLE: f"""
{common_instructions}

Convert this {vendor} ARP table output to standardized JSON:

Expected JSON structure:
{{
    "arp_entries": [
        {{
            "ip_address": "string",
            "mac_address": "string",
            "interface": "string",
            "age": "string|null",
            "type": "string|null"
        }}
    ],
    "total_entries": integer,
    "confidence_score": float
}}

Raw Output:
{{raw_output}}

Return only valid JSON:""",
            NetworkIntent.GET_MAC_TABLE: f"""
{common_instructions}

Convert this {vendor} MAC address table output to standardized JSON:

Expected JSON structure:
{{
    "mac_entries": [
        {{
            "mac_address": "string",
            "vlan": integer,
            "interface": "string",
            "type": "string"
        }}
    ],
    "total_entries": integer,
    "confidence_score": float
}}

Raw Output:
{{raw_output}}

Return only valid JSON:""",
        }

        return intent_prompts.get(
            intent,
            f"""
{common_instructions}

Convert this {vendor} network command output to structured JSON format.
Analyze the output and create appropriate field names and structure.

Raw Output:
{{raw_output}}

Return only valid JSON:""",
        )

    async def normalize_output(
        self, raw_output: str, intent: NetworkIntent, vendor: str, command: str
    ) -> LLMNormalizationResult:
        """
        Normalize network command output using local LLM
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        logger.info(f"Starting LLM normalization for {intent} on {vendor}")

        try:
            # Generate normalization prompt
            prompt = self._get_normalization_prompt(intent, vendor, raw_output)
            logger.debug(f"Generated prompt (first 200 chars): {prompt[:200]}...")

            # Call Ollama API
            normalized_data = await self._call_ollama(prompt)

            processing_time = time.time() - start_time

            if normalized_data:
                self.stats["successful_normalizations"] += 1
                self._update_average_processing_time(processing_time)

                logger.info(f"LLM normalization successful in {processing_time:.2f}s")

                return LLMNormalizationResult(
                    success=True,
                    normalized_data=normalized_data,
                    processing_time=processing_time,
                    confidence_score=normalized_data.get("confidence_score", 0.8),
                )
            else:
                self.stats["failed_normalizations"] += 1
                logger.error("LLM normalization failed: No valid JSON returned")
                return LLMNormalizationResult(
                    success=False, processing_time=processing_time, error_message="LLM returned invalid JSON"
                )

        except Exception as e:
            self.stats["failed_normalizations"] += 1
            processing_time = time.time() - start_time

            logger.error(f"LLM normalization failed: {e}")
            return LLMNormalizationResult(success=False, processing_time=processing_time, error_message=str(e))

    async def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make API call to local Ollama instance with improved JSON parsing"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9,
                "num_predict": 2048,  # Max tokens for response
            },
        }

        logger.debug(f"Calling Ollama API with model {self.model_name}")

        for attempt in range(self.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(f"{self.ollama_url}/api/generate", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get("response", "").strip()

                            logger.debug(f"Raw LLM response: {response_text}")

                            # IMPROVED JSON PARSING
                            try:
                                # Clean the response text
                                cleaned_response = self._clean_json_response(response_text)
                                logger.debug(f"Cleaned response: {cleaned_response}")

                                parsed_json = json.loads(cleaned_response)
                                logger.info("Successfully parsed JSON response")
                                return parsed_json

                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON from LLM (attempt {attempt + 1}): {e}")
                                logger.warning(f"Problematic response: {response_text}")

                                # Try to extract JSON from response if it's embedded in text
                                extracted_json = self._extract_json_from_text(response_text)
                                if extracted_json:
                                    logger.info("Successfully extracted JSON from mixed response")
                                    return extracted_json

                                if attempt < self.max_retries:
                                    logger.info(f"Retrying LLM call (attempt {attempt + 2})")
                                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                                    continue

                        else:
                            error_text = await response.text()
                            logger.error(f"Ollama API error {response.status}: {error_text}")

            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue

            except Exception as e:
                logger.error(f"LLM API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

        logger.error("All LLM retry attempts failed")
        return None

    def _clean_json_response(self, response_text: str) -> str:
        """Clean common JSON formatting issues from LLM response"""
        logger.debug(f"Original response: {repr(response_text[:100])}")

        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
            logger.debug("Removed markdown json code blocks")
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            logger.debug("Removed markdown code blocks")

        # Strip whitespace
        response_text = response_text.strip()

        # Remove common prefixes/suffixes
        prefixes_to_remove = ["JSON Response:", "Response:", "JSON:", "Output:", "Here is the JSON:"]
        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix) :].strip()
                logger.debug(f"Removed prefix: {prefix}")

        # CRITICAL FIX: Ensure JSON starts with opening brace
        if not response_text.startswith("{") and not response_text.startswith("["):
            # Look for the first opening brace
            brace_index = response_text.find("{")
            bracket_index = response_text.find("[")

            # Use whichever comes first (or only one that exists)
            if brace_index >= 0 and (bracket_index < 0 or brace_index < bracket_index):
                response_text = response_text[brace_index:]
                logger.debug(f"Trimmed to start from opening brace at position {brace_index}")
            elif bracket_index >= 0:
                response_text = response_text[bracket_index:]
                logger.debug(f"Trimmed to start from opening bracket at position {bracket_index}")

        # Remove trailing explanations (find the last closing brace/bracket)
        if "{" in response_text and "}" in response_text:
            last_brace = response_text.rfind("}")
            if last_brace > 0:
                response_text = response_text[: last_brace + 1]
                logger.debug("Trimmed content after last closing brace")
        elif "[" in response_text and "]" in response_text:
            last_bracket = response_text.rfind("]")
            if last_bracket > 0:
                response_text = response_text[: last_bracket + 1]
                logger.debug("Trimmed content after last closing bracket")

        logger.debug(f"Final cleaned response: {repr(response_text[:100])}")
        return response_text

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON object from mixed text response"""
        import re

        logger.debug("Attempting to extract JSON from mixed text")

        # Look for JSON object patterns - improved regex
        json_pattern = r"\{(?:[^{}]|{[^{}]*})*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                # Try to parse each potential JSON match
                parsed = json.loads(match.strip())
                logger.info(f"Successfully parsed JSON from match {i + 1}")
                return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse match {i + 1}: {e}")
                continue

        # Look for array patterns
        array_pattern = r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]"
        matches = re.findall(array_pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                parsed = json.loads(match.strip())
                # If it's an array, wrap it in an object
                logger.info(f"Successfully parsed array from match {i + 1}")
                return {"data": parsed}
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse array match {i + 1}: {e}")
                continue

        logger.warning("Could not extract valid JSON from response")
        return None

    def _update_average_processing_time(self, new_time: float):
        """Update running average of processing times"""
        if self.stats["successful_normalizations"] == 1:
            self.stats["average_processing_time"] = new_time
        else:
            # Running average calculation
            current_avg = self.stats["average_processing_time"]
            total_successful = self.stats["successful_normalizations"]
            self.stats["average_processing_time"] = (current_avg * (total_successful - 1) + new_time) / total_successful

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics"""
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = (self.stats["successful_normalizations"] / self.stats["total_requests"]) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "model_name": self.model_name,
            "cache_entries": len(self._prompt_cache),
        }


# Global instance
local_llm_normalizer = LocalLLMNormalizer()
