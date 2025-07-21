# app/services/local_llm_normalizer.py

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

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
    Integrates with LM Studio running on Windows host.
    Enhanced to support both NetworkIntent enums and dynamic string intents.
    """

    def __init__(
        self,
        ollama_url: str = "http://192.168.1.11:1234",
        model_name: str = "meta-llama-3.1-8b-instruct",
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
        """Check if LM Studio service is available and model is loaded"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"{self.ollama_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["id"] for model in data.get("data", [])]
                        return self.model_name in models
            return False
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False

    def _get_intent_string(self, intent: Union[NetworkIntent, str]) -> str:
        """Convert intent to string, handling both enum and string inputs"""
        if isinstance(intent, NetworkIntent):
            return intent.value
        return str(intent)  # Handle dynamic discovery intents (strings)

    def _get_normalization_prompt(
        self, intent: Union[NetworkIntent, str], vendor: str, raw_output: str
    ) -> str:
        """Generate intent-specific normalization prompt"""

        # Convert intent to string for consistent processing
        intent_str = self._get_intent_string(intent)

        # Cache key for prompt template
        cache_key = f"{intent_str}_{vendor}"

        if cache_key in self._prompt_cache:
            self.stats["cache_hits"] += 1
            base_prompt = self._prompt_cache[cache_key]
        else:
            base_prompt = self._generate_prompt_template(intent, vendor)
            self._prompt_cache[cache_key] = base_prompt

        return base_prompt.format(raw_output=raw_output)

    def _generate_prompt_template(
        self, intent: Union[NetworkIntent, str], vendor: str
    ) -> str:
        """Generate intent-specific prompt templates"""

        common_instructions = """
You are a network automation expert. Convert network device output to JSON.

STRICT REQUIREMENTS:
- Return ONLY a JSON object
- No explanations, no markdown, no code blocks
- Start response immediately with {{ and end with }}
- No text before or after the JSON

"""

        # Handle both enum and string intents
        if isinstance(intent, NetworkIntent):
            intent_key = intent
        else:
            # For dynamic discovery intents, try to map to known patterns
            intent_str = intent.lower()
            if (
                "version" in intent_str
                or "system" in intent_str
                or "serial" in intent_str
            ):
                intent_key = NetworkIntent.GET_SYSTEM_VERSION
            elif "interface" in intent_str:
                intent_key = NetworkIntent.GET_INTERFACE_STATUS
            elif "bgp" in intent_str:
                intent_key = NetworkIntent.GET_BGP_SUMMARY
            elif "arp" in intent_str:
                intent_key = NetworkIntent.GET_ARP_TABLE
            elif "mac" in intent_str:
                intent_key = NetworkIntent.GET_MAC_TABLE
            else:
                intent_key = None

        intent_prompts = {
            NetworkIntent.GET_SYSTEM_VERSION: f"""
{common_instructions}

Convert this {vendor} version/system output to JSON with these fields:
vendor, hostname, os_version, model, serial_number, uptime_seconds, memory_total_kb, memory_free_kb, last_reload_reason, confidence_score

For serial number extraction, look for patterns like:
- Serial number: ABC123
- Serial Number: XYZ789
- System serial number: DEF456
- Hardware serial: GHI789

Raw Output:
{{raw_output}}

JSON:""",
            NetworkIntent.GET_INTERFACE_STATUS: f"""
{common_instructions}

Convert this {vendor} interface output to JSON with these fields:
interfaces (array), total_interfaces, up_interfaces, down_interfaces, confidence_score

Raw Output:
{{raw_output}}

JSON:""",
            NetworkIntent.GET_BGP_SUMMARY: f"""
{common_instructions}

Convert this {vendor} BGP output to JSON with these exact fields:
router_id, local_as, neighbors (array with neighbor_ip, remote_as, state, uptime, prefixes_received), total_neighbors, established_neighbors, confidence_score

Raw Output:
{{raw_output}}

JSON:""",
            NetworkIntent.GET_ARP_TABLE: f"""
{common_instructions}

Convert this {vendor} ARP output to JSON with these fields:
arp_entries (array with ip_address, mac_address, interface, age, type), total_entries, confidence_score

Raw Output:
{{raw_output}}

JSON:""",
            NetworkIntent.GET_MAC_TABLE: f"""
{common_instructions}

Convert this {vendor} MAC table output to JSON with these fields:
mac_entries (array with mac_address, vlan, interface, type), total_entries, confidence_score

Raw Output:
{{raw_output}}

JSON:""",
        }

        if intent_key in intent_prompts:
            return intent_prompts[intent_key]
        else:
            # Generic prompt for unknown dynamic intents
            intent_display = self._get_intent_string(intent).replace("_", " ").title()
            return f"""
{common_instructions}

Convert this {vendor} network command output to structured JSON format for: {intent_display}

Analyze the output and extract relevant information into a well-structured JSON object.
Include a confidence_score field (0.0-1.0) indicating how confident you are in the extraction.

Raw Output:
{{raw_output}}

JSON:"""

    async def normalize_output(
        self,
        raw_output: str,
        intent: Union[NetworkIntent, str],
        vendor: str,
        command: str,
    ) -> LLMNormalizationResult:
        """
        Normalize network command output using LM Studio LLM
        Enhanced to support both NetworkIntent enums and dynamic string intents
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        intent_str = self._get_intent_string(intent)
        logger.info(f"Starting LLM normalization for {intent_str} on {vendor}")

        try:
            # Generate normalization prompt
            prompt = self._get_normalization_prompt(intent, vendor, raw_output)
            logger.debug(f"Generated prompt (first 200 chars): {prompt[:200]}...")

            # Call LM Studio API
            normalized_data = await self._call_lm_studio(prompt)

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
                    success=False,
                    processing_time=processing_time,
                    error_message="LLM returned invalid JSON",
                )

        except Exception as e:
            self.stats["failed_normalizations"] += 1
            processing_time = time.time() - start_time

            logger.error(f"LLM normalization failed: {e}")
            return LLMNormalizationResult(
                success=False, processing_time=processing_time, error_message=str(e)
            )

    async def _call_lm_studio(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make API call to LM Studio instance using OpenAI-compatible API"""

        # OpenAI-compatible payload format
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False,
        }

        logger.debug(f"Calling LM Studio API with model {self.model_name}")

        for attempt in range(self.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.ollama_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status == 200:
                            result = await response.json()

                            # Extract response from OpenAI format
                            choices = result.get("choices", [])
                            if choices:
                                response_text = (
                                    choices[0]
                                    .get("message", {})
                                    .get("content", "")
                                    .strip()
                                )
                                logger.debug(f"Raw LLM response: {response_text}")

                                # IMPROVED JSON PARSING
                                try:
                                    # Clean the response text
                                    cleaned_response = self._clean_json_response(
                                        response_text
                                    )
                                    logger.debug(
                                        f"Cleaned response: {cleaned_response}"
                                    )

                                    parsed_json = json.loads(cleaned_response)
                                    logger.info("Successfully parsed JSON response")
                                    return parsed_json

                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Invalid JSON from LLM (attempt {attempt + 1}): {e}"
                                    )
                                    logger.warning(
                                        f"Problematic response: {response_text}"
                                    )

                                    # Try to extract JSON from response if it's embedded in text
                                    extracted_json = self._extract_json_from_text(
                                        response_text
                                    )
                                    if extracted_json:
                                        logger.info(
                                            "Successfully extracted JSON from mixed response"
                                        )
                                        return extracted_json

                                    if attempt < self.max_retries:
                                        logger.info(
                                            f"Retrying LLM call (attempt {attempt + 2})"
                                        )
                                        await asyncio.sleep(1 * (attempt + 1))
                                        continue
                            else:
                                logger.error("No choices in LM Studio response")

                        else:
                            error_text = await response.text()
                            logger.error(
                                f"LM Studio API error {response.status}: {error_text}"
                            )

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
        logger.debug(f"Original response: {repr(response_text)}")

        # Remove common prefixes like "Here is the JSON:" etc.
        lines = response_text.split("\n")
        cleaned_lines = []
        json_started = False
        brace_count = 0

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip explanatory text before JSON
            if not json_started and not line.startswith("{"):
                continue

            # Start collecting when we see an opening brace
            if line.startswith("{") or json_started:
                json_started = True
                cleaned_lines.append(line)

                # Count braces to handle nested objects
                brace_count += line.count("{") - line.count("}")

                # Stop when we complete the JSON object
                if brace_count == 0 and json_started:
                    break

        # Join the JSON lines
        cleaned_response = "\n".join(cleaned_lines)

        # Fallback: if we didn't get valid JSON structure, try to extract it
        if not cleaned_response.strip().startswith("{"):
            start_idx = response_text.find("{")
            if start_idx >= 0:
                end_idx = response_text.rfind("}")
                if end_idx > start_idx:
                    cleaned_response = response_text[start_idx : end_idx + 1]

        logger.debug(f"Cleaned response: {repr(cleaned_response)}")
        return cleaned_response

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON object from mixed text response"""
        import re

        logger.debug("Attempting to extract JSON from mixed text")

        # Method 1: Find JSON between braces more aggressively
        start_idx = text.find("{")
        if start_idx >= 0:
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx

            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if brace_count == 0:  # Found matching brace
                json_str = text[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(json_str)
                    logger.info("Successfully extracted JSON using brace matching")
                    return parsed
                except json.JSONDecodeError:
                    pass

        # Method 2: Regex fallback
        json_pattern = r"\{(?:[^{}]|{[^{}]*})*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                parsed = json.loads(match.strip())
                logger.info(f"Successfully parsed JSON from regex match {i + 1}")
                return parsed
            except json.JSONDecodeError:
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
            self.stats["average_processing_time"] = (
                current_avg * (total_successful - 1) + new_time
            ) / total_successful

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics"""
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_normalizations"] / self.stats["total_requests"]
            ) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "model_name": self.model_name,
            "cache_entries": len(self._prompt_cache),
        }


# Global instance
local_llm_normalizer = LocalLLMNormalizer()
