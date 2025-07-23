# app/services/universal_formatter.py
"""
Universal Formatter - Enterprise Scalable Anti-Hallucination Solution
Uses intelligent LLM techniques to prevent fabrication while scaling across ALL vendors/commands
"""

import json
import logging
import re
import time
from typing import Any, Dict, Set

from app.models.universal_requests import (
    DeviceExecutionResult,
    DiscoveryInfo,
    InterfaceType,
    OutputFormat,
)

logger = logging.getLogger(__name__)


class UniversalFormatter:
    """
    Enterprise-grade LLM formatter with scalable anti-hallucination system
    Works for ANY vendor, ANY command, ANY output format
    """

    def __init__(self):
        # Performance tracking
        self.stats = {
            "total_formats": 0,
            "successful_formats": 0,
            "failed_formats": 0,
            "average_format_time": 0.0,
            "format_cache_hits": 0,
            "hallucination_blocks": 0,
            "content_extraction_success": 0,
        }

        # Format cache for performance
        self._format_cache = {}

    async def initialize(self):
        """Initialize direct LLM connection"""
        # Direct LLM configuration
        self.llm_url = "http://192.168.1.11:1234"
        self.model_name = "meta-llama-3.1-8b-instruct"
        self.timeout = 30

        # Test LLM connection
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.llm_url}/v1/models") as response:
                    if response.status == 200:
                        logger.info(
                            "✅ Universal Formatter initialized with direct LLM support"
                        )
                        return True

            logger.warning(
                "⚠️ LLM not available - Universal Formatter will use fallback"
            )
            return False
        except Exception as e:
            logger.warning(f"⚠️ LLM connection failed: {e} - using fallback")
            return False

    async def format_response(
        self,
        user_request: str,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
        interface_type: InterfaceType,
        output_format: OutputFormat,
        format_options: Dict[str, Any] = None,
    ) -> Any:
        """
        Enterprise-grade universal formatter with scalable anti-hallucination
        """

        start_time = time.time()
        self.stats["total_formats"] += 1

        logger.info(
            f"🎨 Formatting response for {interface_type.value} in {output_format.value} format"
        )

        try:
            # STEP 1: Extract factual content from ALL raw outputs
            factual_content = self._extract_factual_content(execution_results)

            # STEP 2: Build enterprise-grade anti-hallucination prompt
            context = self._build_enterprise_context(
                user_request,
                intent,
                execution_results,
                discovery_results,
                factual_content,
            )

            # STEP 3: Generate scalable format prompt with content grounding
            prompt = self._build_grounded_format_prompt(
                context,
                interface_type,
                output_format,
                factual_content,
                format_options or {},
            )

            # STEP 4: LLM formatting with hallucination prevention
            try:
                formatted_response = await self._format_with_content_grounding(
                    prompt, output_format, factual_content
                )

                # STEP 5: Validate response against factual content
                if self._validate_response_against_facts(
                    formatted_response, factual_content
                ):
                    self.stats["content_extraction_success"] += 1
                else:
                    logger.warning(
                        "Response validation failed - using content-safe fallback"
                    )
                    formatted_response = self._generate_content_safe_response(
                        execution_results,
                        factual_content,
                        interface_type,
                        output_format,
                    )

            except Exception as e:
                logger.warning(
                    f"LLM formatting failed: {e} - using content-safe fallback"
                )
                formatted_response = self._generate_content_safe_response(
                    execution_results, factual_content, interface_type, output_format
                )

            self.stats["successful_formats"] += 1
            format_time = time.time() - start_time
            self._update_format_stats(format_time)

            logger.info(f"✅ Response formatted successfully in {format_time:.2f}s")
            return formatted_response

        except Exception as e:
            self.stats["failed_formats"] += 1
            logger.error(f"❌ Response formatting failed: {e}")
            return self._emergency_fallback(execution_results, str(e))

    def _extract_factual_content(
        self, execution_results: Dict[str, DeviceExecutionResult]
    ) -> Dict[str, Any]:
        """
        ENTERPRISE CORE: Extract all factual content from raw outputs
        This creates a "source of truth" that prevents hallucination
        """

        factual_content = {
            "devices": {},
            "has_structured_data": False,
            "has_status_messages": False,
            "extracted_entities": {
                "ip_addresses": set(),
                "interfaces": set(),
                "protocols": set(),
                "as_numbers": set(),
                "mac_addresses": set(),
                "hostnames": set(),
                "numerical_values": set(),
            },
            "content_type": "unknown",
            "total_lines": 0,
            "status_indicators": [],
        }

        for device_name, result in execution_results.items():
            device_content = {
                "raw_output": result.raw_output or "",
                "success": result.success,
                "command": result.command_executed,
                "platform": result.platform,
                "content_analysis": {},
            }

            if result.success and result.raw_output:
                raw = result.raw_output.strip()
                device_content["content_analysis"] = self._analyze_content_structure(
                    raw
                )

                # Extract entities using regex patterns (scalable across vendors)
                self._extract_network_entities(
                    raw, factual_content["extracted_entities"]
                )

                # Determine content type
                if device_content["content_analysis"]["is_status_message"]:
                    factual_content["has_status_messages"] = True
                    factual_content["status_indicators"].extend(
                        device_content["content_analysis"]["status_indicators"]
                    )
                elif device_content["content_analysis"]["has_tabular_data"]:
                    factual_content["has_structured_data"] = True

                factual_content["total_lines"] += device_content["content_analysis"][
                    "line_count"
                ]

            factual_content["devices"][device_name] = device_content

        # Determine overall content type
        if (
            factual_content["has_status_messages"]
            and not factual_content["has_structured_data"]
        ):
            factual_content["content_type"] = "status_only"
        elif factual_content["has_structured_data"]:
            factual_content["content_type"] = "structured_data"
        elif factual_content["total_lines"] > 5:
            factual_content["content_type"] = "unstructured_data"
        else:
            factual_content["content_type"] = "minimal_output"

        return factual_content

    def _analyze_content_structure(self, raw_output: str) -> Dict[str, Any]:
        """
        Analyze content structure to determine data type - VENDOR AGNOSTIC
        """

        lines = raw_output.split("\n")
        analysis = {
            "line_count": len(lines),
            "is_status_message": False,
            "has_tabular_data": False,
            "has_configuration": False,
            "status_indicators": [],
            "data_patterns": [],
        }

        # Check for status messages (vendor agnostic patterns)
        status_patterns = [
            r"^%.*",  # Cisco/Arista % messages
            r"^#.*",  # Some status indicators
            r".*inactive.*",  # Protocol inactive
            r".*not\s+found.*",  # Command not found
            r".*not\s+configured.*",  # Not configured
            r".*disabled.*",  # Protocol disabled
            r".*empty.*",  # Empty table/results
            r".*no\s+entries.*",  # No entries
        ]

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for pattern in status_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    analysis["is_status_message"] = True
                    analysis["status_indicators"].append(line_stripped)
                    break

        # Check for structured data (tables, configs, etc.)
        if analysis["line_count"] > 3:
            # Look for table-like structures (multiple columns)
            column_lines = 0
            for line in lines:
                if len(line.split()) >= 3:  # Likely columnar data
                    column_lines += 1

            if column_lines >= 2:
                analysis["has_tabular_data"] = True
                analysis["data_patterns"].append("tabular")

        # Check for configuration data
        config_indicators = ["!", "interface", "router", "ip route", "access-list"]
        for line in lines:
            for indicator in config_indicators:
                if indicator in line.lower():
                    analysis["has_configuration"] = True
                    analysis["data_patterns"].append("configuration")
                    break

        return analysis

    def _extract_network_entities(
        self, raw_output: str, entities: Dict[str, Set]
    ) -> None:
        """
        Extract network entities using regex - SCALABLE across all vendors
        """

        # IP addresses (IPv4)
        ip_pattern = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        entities["ip_addresses"].update(re.findall(ip_pattern, raw_output))

        # AS numbers
        as_pattern = r"\b(?:AS|as)\s*(\d{1,6})\b"
        entities["as_numbers"].update(re.findall(as_pattern, raw_output, re.IGNORECASE))

        # Interface names (vendor agnostic)
        interface_patterns = [
            r"\b(?:Ethernet|Eth|GigabitEthernet|Gi|TenGigabitEthernet|Te|FortyGigE|Fo)\d+[/\d]*\b",
            r"\b(?:Management|Mgmt|Loopback|Lo)\d*\b",
            r"\b(?:Vlan|VLAN)\d+\b",
            r"\b(?:Port-channel|Po)\d+\b",
        ]

        for pattern in interface_patterns:
            entities["interfaces"].update(
                re.findall(pattern, raw_output, re.IGNORECASE)
            )

        # MAC addresses
        mac_pattern = r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b"
        entities["mac_addresses"].update(re.findall(mac_pattern, raw_output))

        # Numerical values (counters, statistics)
        number_pattern = r"\b\d{3,}\b"  # Numbers with 3+ digits (likely statistics)
        entities["numerical_values"].update(re.findall(number_pattern, raw_output))

    def _build_enterprise_context(
        self,
        user_request: str,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
        factual_content: Dict[str, Any],
    ) -> str:
        """
        Build enterprise-grade context with factual grounding
        """

        context_parts = [
            f"User Request: {user_request}",
            f"Intent: {intent}",
            f"Content Type: {factual_content['content_type']}",
            "",
            "ENTERPRISE ANTI-HALLUCINATION PROTOCOL:",
            "1. You have access to FACTUAL CONTENT extracted from device outputs",
            "2. You MUST use ONLY the factual content provided below",
            "3. If factual content is empty/minimal, state that clearly",
            "4. NEVER create network data not present in factual content",
            "",
        ]

        # Add factual content summary
        if factual_content["extracted_entities"]["ip_addresses"]:
            context_parts.append(
                f"FACTUAL IP ADDRESSES: {list(factual_content['extracted_entities']['ip_addresses'])}"
            )

        if factual_content["extracted_entities"]["interfaces"]:
            context_parts.append(
                f"FACTUAL INTERFACES: {list(factual_content['extracted_entities']['interfaces'])}"
            )

        if factual_content["extracted_entities"]["as_numbers"]:
            context_parts.append(
                f"FACTUAL AS NUMBERS: {list(factual_content['extracted_entities']['as_numbers'])}"
            )

        if factual_content["status_indicators"]:
            context_parts.append(
                f"STATUS MESSAGES: {factual_content['status_indicators']}"
            )

        context_parts.append("")

        # Add device results with analysis
        for device_name, device_data in factual_content["devices"].items():
            context_parts.append(f"=== {device_name} ===")
            context_parts.append(f"Platform: {device_data['platform']}")
            context_parts.append(f"Command: {device_data['command']}")
            context_parts.append(f"Success: {device_data['success']}")

            if device_data["raw_output"]:
                context_parts.append(f"Raw Output: '{device_data['raw_output']}'")
                context_parts.append(
                    f"Content Analysis: {device_data['content_analysis']}"
                )

            context_parts.append("")

        return "\n".join(context_parts)

    def _build_grounded_format_prompt(
        self,
        context: str,
        interface_type: InterfaceType,
        output_format: OutputFormat,
        factual_content: Dict[str, Any],
        format_options: Dict[str, Any],
    ) -> str:
        """
        Build content-grounded format prompt - SCALABLE approach
        """

        base_prompt = f"""You are formatting network automation results for enterprise use.

{context}

CRITICAL FORMATTING RULES:
1. Base your response ONLY on the factual content and raw output provided above
2. If status messages indicate no data is available, explain that clearly
3. If structured data exists, present it organized and readable
4. NEVER add information not present in the raw output
5. Quote exact values from raw output when possible

Content Type: {factual_content["content_type"]}
Response Format: {output_format.value} for {interface_type.value}"""

        # Add format-specific instructions based on content type
        if factual_content["content_type"] == "status_only":
            base_prompt += """

STATUS MESSAGE FORMATTING:
- Explain what the status message means
- Do not create fake data
- Suggest alternative approaches if appropriate
- Be clear about what information is not available"""

        elif factual_content["content_type"] == "structured_data":
            base_prompt += """

STRUCTURED DATA FORMATTING:
- Present data in organized, readable format
- Use actual values from raw output
- Maintain technical accuracy
- Group related information logically"""

        # Add interface-specific formatting
        if (
            interface_type == InterfaceType.CHAT
            and output_format == OutputFormat.CONVERSATIONAL
        ):
            base_prompt += """

CONVERSATIONAL FORMAT:
- Professional network engineer tone
- Plain text response (not JSON)
- Clear explanations of technical concepts
- Direct answers to user questions"""

        return base_prompt

    async def _format_with_content_grounding(
        self, prompt: str, output_format: OutputFormat, factual_content: Dict[str, Any]
    ) -> Any:
        """
        LLM formatting with enterprise content grounding
        """

        try:
            import aiohttp

            # Content-grounded LLM parameters
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low but not zero for natural language
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "max_tokens": 2000,
                "stream": False,
                "stop": ["FACTUAL", "Example:", "For instance:"],  # Prevent examples
            }

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.llm_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        choices = result.get("choices", [])

                        if choices:
                            llm_response = (
                                choices[0].get("message", {}).get("content", "").strip()
                            )

                            # Clean response minimally
                            llm_response = self._clean_response_enterprise(llm_response)

                            if output_format == OutputFormat.CONVERSATIONAL:
                                return llm_response
                            elif output_format == OutputFormat.JSON:
                                try:
                                    return json.loads(llm_response)
                                except json.JSONDecodeError:
                                    return {"formatted_response": llm_response}
                            else:
                                return llm_response

                    error_text = await response.text()
                    logger.error(f"LLM API error {response.status}: {error_text}")
                    raise ValueError(f"LLM API returned {response.status}")

        except Exception as e:
            logger.error(f"Content-grounded LLM call failed: {e}")
            raise ValueError("LLM formatting failed") from e

    def _validate_response_against_facts(
        self, response: str, factual_content: Dict[str, Any]
    ) -> bool:
        """
        Enterprise validation: Ensure response doesn't contain hallucinated content
        """

        if not isinstance(response, str):
            return True  # Non-string responses (JSON, etc.) handled separately

        response_lower = response.lower()

        # Check for hallucinated IP addresses
        response_ips = set(
            re.findall(
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                response,
            )
        )
        factual_ips = factual_content["extracted_entities"]["ip_addresses"]

        hallucinated_ips = response_ips - factual_ips
        if hallucinated_ips:
            logger.warning(f"Hallucinated IP addresses detected: {hallucinated_ips}")
            self.stats["hallucination_blocks"] += 1
            return False

        # Check for hallucinated AS numbers
        response_as = set(re.findall(r"\bas\s*(\d{1,6})\b", response, re.IGNORECASE))
        factual_as = factual_content["extracted_entities"]["as_numbers"]

        hallucinated_as = response_as - factual_as
        if hallucinated_as:
            logger.warning(f"Hallucinated AS numbers detected: {hallucinated_as}")
            self.stats["hallucination_blocks"] += 1
            return False

        # If content type is status_only, ensure no fake structured data
        if factual_content["content_type"] == "status_only":
            # Check for patterns that suggest fake tabular data
            fake_table_indicators = [
                "neighbor",
                "established",
                "idle",
                "192.168",
                "10.0.0",
                "ethernet1/",
            ]

            # If we have status indicators but response contains fake data patterns
            if factual_content["status_indicators"] and any(
                indicator in response_lower for indicator in fake_table_indicators
            ):
                # But those patterns don't exist in actual output
                all_raw = " ".join(
                    [
                        device["raw_output"]
                        for device in factual_content["devices"].values()
                    ]
                ).lower()
                if not any(
                    indicator in all_raw
                    for indicator in fake_table_indicators
                    if indicator in response_lower
                ):
                    logger.warning(
                        "Fake structured data detected in status-only response"
                    )
                    self.stats["hallucination_blocks"] += 1
                    return False

        return True

    def _generate_content_safe_response(
        self,
        execution_results: Dict[str, DeviceExecutionResult],
        factual_content: Dict[str, Any],
        interface_type: InterfaceType,
        output_format: OutputFormat,
    ) -> str:
        """
        Generate content-safe response using only factual content
        """

        # Get first device for simplicity (enterprise extension: handle multiple devices)
        device_name = list(execution_results.keys())[0]
        device_data = factual_content["devices"][device_name]

        if factual_content["content_type"] == "status_only":
            status_msg = (
                factual_content["status_indicators"][0]
                if factual_content["status_indicators"]
                else device_data["raw_output"]
            )

            if "bgp" in status_msg.lower() and "inactive" in status_msg.lower():
                return f"BGP is not currently running on {device_name}. The device returned '{status_msg}', which means the BGP routing process is not configured or enabled. To check BGP configuration, you could try 'show running-config section bgp'."
            else:
                return f"The command on {device_name} returned a status message: '{status_msg}'. This indicates the requested information is not currently available."

        elif factual_content["content_type"] == "structured_data":
            return f"Command executed successfully on {device_name}. Here's the output:\n\n{device_data['raw_output']}"

        else:
            return f"Command output from {device_name}:\n{device_data['raw_output']}"

    def _clean_response_enterprise(self, response: str) -> str:
        """Enterprise-grade response cleaning"""

        # Basic cleaning
        response = response.strip()

        # Remove code blocks if present
        if response.startswith("```") and response.endswith("```"):
            lines = response.split("\n")
            if len(lines) > 2:
                response = "\n".join(lines[1:-1])

        # Remove JSON wrapping for conversational responses
        if response.startswith("{") and response.endswith("}"):
            try:
                data = json.loads(response)
                if isinstance(data, dict) and len(data) == 1:
                    key = list(data.keys())[0]
                    if key in ["response", "message", "content", "formatted_response"]:
                        response = data[key]
            except (json.JSONDecodeError, KeyError):
                pass

        return response

    def _emergency_fallback(
        self, execution_results: Dict[str, DeviceExecutionResult], error: str
    ) -> str:
        """Emergency fallback for critical failures"""

        success_count = sum(1 for r in execution_results.values() if r.success)
        total_count = len(execution_results)

        return f"Command execution completed: {success_count}/{total_count} successful. Formatting error: {error}"

    def _update_format_stats(self, format_time: float):
        """Update formatting performance statistics"""

        current_avg = self.stats["average_format_time"]
        total_successful = self.stats["successful_formats"]

        if total_successful == 1:
            self.stats["average_format_time"] = format_time
        else:
            self.stats["average_format_time"] = (
                current_avg * (total_successful - 1) + format_time
            ) / total_successful

    async def health_check(self) -> bool:
        """Check if direct LLM connection is healthy"""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.llm_url}/v1/models") as response:
                    return response.status == 200
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get formatter statistics including enterprise metrics"""

        success_rate = 0.0
        hallucination_block_rate = 0.0

        if self.stats["total_formats"] > 0:
            success_rate = (
                self.stats["successful_formats"] / self.stats["total_formats"]
            ) * 100
            hallucination_block_rate = (
                self.stats["hallucination_blocks"] / self.stats["total_formats"]
            ) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "hallucination_block_rate_percent": round(hallucination_block_rate, 2),
            "llm_available": hasattr(self, "llm_url"),
            "direct_llm_connection": True,
            "enterprise_grade": True,
        }


# Global instance
universal_formatter = UniversalFormatter()
