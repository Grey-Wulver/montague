# app/services/universal_formatter.py
"""
Universal Formatter - LLM-Powered Response Formatting (FIXED)
Replaces ALL hardcoded normalization and extraction logic
Uses LLM to format responses for ANY interface in ANY format
"""

import json
import logging
import time
from typing import Any, Dict

from app.models.universal_requests import (
    DeviceExecutionResult,
    DiscoveryInfo,
    InterfaceType,
    OutputFormat,
)

logger = logging.getLogger(__name__)


class UniversalFormatter:
    """
    LLM-powered formatter that adapts responses to ANY interface and format
    This is the magic that makes the universal pipeline truly universal
    """

    def __init__(self):
        # Performance tracking
        self.stats = {
            "total_formats": 0,
            "successful_formats": 0,
            "failed_formats": 0,
            "average_format_time": 0.0,
            "format_cache_hits": 0,
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
        THE UNIVERSAL FORMATTER: Formats response for ANY interface/format
        This single method replaces all hardcoded extraction logic
        """

        start_time = time.time()
        self.stats["total_formats"] += 1

        logger.info(
            f"🎨 Formatting response for {interface_type.value} in {output_format.value} format"
        )

        try:
            # Build comprehensive context
            context = self._build_context(
                user_request, intent, execution_results, discovery_results
            )

            # Generate format-specific prompt
            prompt = self._build_format_prompt(
                context, interface_type, output_format, format_options or {}
            )

            # Use direct LLM to format the response
            try:
                formatted_response = await self._format_with_llm_direct(
                    prompt, output_format
                )
            except Exception as e:
                logger.warning(f"LLM formatting failed: {e} - using fallback")
                # Fallback formatting without LLM
                formatted_response = self._fallback_format(
                    execution_results, interface_type, output_format
                )

            self.stats["successful_formats"] += 1
            format_time = time.time() - start_time
            self._update_format_stats(format_time)

            logger.info(f"✅ Response formatted successfully in {format_time:.2f}s")
            return formatted_response

        except Exception as e:
            self.stats["failed_formats"] += 1
            logger.error(f"❌ Response formatting failed: {e}")

            # Return fallback response
            return self._emergency_fallback(execution_results, str(e))

    def _build_context(
        self,
        user_request: str,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
    ) -> str:
        """Build comprehensive context for LLM formatting"""

        context_parts = [
            f"User Request: {user_request}",
            f"Detected Intent: {intent}",
            f"Devices Processed: {len(execution_results)}",
            "",
        ]

        # Add results for each device
        for device_name, result in execution_results.items():
            context_parts.append(f"Device: {device_name}")
            context_parts.append(f"Platform: {result.platform}")
            context_parts.append(f"Command Executed: {result.command_executed}")
            context_parts.append(f"Success: {result.success}")

            if result.success and result.raw_output:
                # Include raw output for LLM processing
                context_parts.append(f"Raw Output:\n{result.raw_output}")
            elif not result.success:
                context_parts.append(f"Error: {result.error_message}")

            context_parts.append("")  # Separator

        # Add discovery information
        if discovery_results:
            context_parts.append("Discovery Information:")
            for device_name, discovery in discovery_results.items():
                context_parts.append(
                    f"- {device_name}: '{discovery.discovered_command}' "
                    f"(confidence: {discovery.confidence:.2f}, "
                    f"cached: {discovery.cached_result})"
                )

        return "\n".join(context_parts)

    def _build_format_prompt(
        self,
        context: str,
        interface_type: InterfaceType,
        output_format: OutputFormat,
        format_options: Dict[str, Any],
    ) -> str:
        """Build format-specific prompts for different interfaces"""

        base_prompt = f"""You are a network automation assistant. Here's the execution context:

{context}

Based on what the user requested, provide the response in the specified format."""

        # Interface-specific formatting instructions
        if interface_type == InterfaceType.CHAT:
            return base_prompt + self._get_chat_format_instructions(
                output_format, format_options
            )

        elif interface_type == InterfaceType.API:
            return base_prompt + self._get_api_format_instructions(
                output_format, format_options
            )

        elif interface_type == InterfaceType.CLI:
            return base_prompt + self._get_cli_format_instructions(
                output_format, format_options
            )

        elif interface_type == InterfaceType.WEB:
            return base_prompt + self._get_web_format_instructions(
                output_format, format_options
            )

        else:
            return base_prompt + f"\n\nFormat: {output_format.value}"

    def _get_chat_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """Chat-specific formatting instructions"""

        if output_format == OutputFormat.CONVERSATIONAL:
            return """

Format: Natural, conversational response (PLAIN TEXT, NOT JSON)
- Be concise but complete
- If user asked for specific data (like serial number), highlight it clearly
- Use friendly, professional tone
- Include key information they requested
- Mention if discovery was used (AI learned new command)
- RETURN ONLY THE CONVERSATIONAL TEXT, NO JSON WRAPPER

Examples:
- "Serial Number: ABC123 (AI discovered this from show version)"
- "BGP Status: 3/4 neighbors established on spine1"
- "Interface Summary: 12 interfaces up, 2 down on leaf1"
"""

        elif output_format == OutputFormat.JSON:
            return """

Format: Clean JSON for chat display
- Return valid JSON object
- Include only requested information
- Add "chat_friendly" field with human-readable summary
- Include discovery info if applicable

Example:
{
  "result": "Serial Number: ABC123",
  "chat_friendly": "Found serial number ABC123 on spine1",
  "discovery_used": true
}
"""

        else:
            return f"\n\nFormat: {output_format.value} suitable for chat display"

    def _get_api_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """API-specific formatting instructions"""

        if output_format == OutputFormat.JSON:
            return """

Format: Structured JSON for API consumption
- Return clean, consistent JSON structure
- Use standard field names (snake_case)
- Include metadata (success, device_count, execution_time)
- Include all relevant extracted data
- Provide arrays for multi-device responses

Example structure:
{
  "success": true,
  "devices": [
    {
      "device": "spine1",
      "result": {...extracted data...},
      "discovery_used": true
    }
  ],
  "summary": "Brief summary",
  "execution_metadata": {...}
}
"""

        elif output_format == OutputFormat.YAML:
            return """

Format: Clean YAML structure
- Use proper YAML formatting
- Include consistent indentation
- Provide clear hierarchy
- Include comments where helpful
"""

        elif output_format == OutputFormat.XML:
            return """

Format: Well-formed XML
- Use descriptive tag names
- Include proper XML declaration
- Nest data logically
- Include attributes for metadata
"""

        else:
            return f"\n\nFormat: {output_format.value} for API consumption"

    def _get_cli_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """CLI-specific formatting instructions"""

        if output_format == OutputFormat.TABLE:
            return """

Format: ASCII table for CLI display
- Create clean, aligned tables with headers
- Use consistent column spacing
- Include only most relevant data
- Limit width to ~80 characters
- Use | for column separators

Example:
| Device | Serial Number      | Status |
|--------|--------------------|--------|
| spine1 | ABC123456789       | Up     |
"""

        elif output_format == OutputFormat.CSV:
            return """

Format: CSV for CLI export
- Include headers
- Use comma separation
- Escape special characters
- One line per device/result
"""

        else:
            return f"\n\nFormat: {output_format.value} for command line display"

    def _get_web_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """Web UI-specific formatting instructions"""

        if output_format == OutputFormat.HTML:
            return """

Format: HTML for web display
- Use semantic HTML elements
- Include CSS classes for styling
- Create tables for structured data
- Use badges/alerts for status indicators
- Make it responsive and accessible

Example:
<div class="device-result">
  <h3>spine1</h3>
  <span class="badge success">Success</span>
  <p>Serial Number: <code>ABC123</code></p>
</div>
"""

        elif output_format == OutputFormat.MARKDOWN:
            return """

Format: Markdown for web/documentation
- Use proper markdown syntax
- Include headers, lists, code blocks
- Create tables where appropriate
- Use emphasis for important data
"""

        else:
            return f"\n\nFormat: {output_format.value} for web display"

    async def _format_with_llm_direct(
        self, prompt: str, output_format: OutputFormat
    ) -> Any:
        """Direct LLM call optimized for each output format"""

        try:
            import aiohttp

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
                "stream": False,
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

                            # Handle response based on expected format
                            if output_format == OutputFormat.CONVERSATIONAL:
                                # Return plain text directly
                                return llm_response

                            elif output_format == OutputFormat.JSON:
                                # Try to parse as JSON, fallback to wrapped response
                                try:
                                    # Clean and parse JSON
                                    cleaned_json = self._clean_json_response(
                                        llm_response
                                    )
                                    return json.loads(cleaned_json)
                                except json.JSONDecodeError:
                                    # Wrap plain text in JSON structure
                                    return {"formatted_response": llm_response}

                            else:
                                # For other formats (table, csv, etc.), return as-is
                                return llm_response

                    # If HTTP request failed
                    error_text = await response.text()
                    logger.error(f"LLM API error {response.status}: {error_text}")
                    raise ValueError(f"LLM API returned {response.status}")

        except Exception as e:
            logger.error(f"Direct LLM call failed: {e}")
            raise ValueError("LLM formatting failed - no response") from e

    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response from LLM (simplified version)"""

        # Find JSON object boundaries
        start_idx = response_text.find("{")
        if start_idx >= 0:
            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx

            for i in range(start_idx, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if brace_count == 0:
                return response_text[start_idx : end_idx + 1]

        # Fallback: return original if no valid JSON structure found
        return response_text

    def _fallback_format(
        self,
        execution_results: Dict[str, DeviceExecutionResult],
        interface_type: InterfaceType,
        output_format: OutputFormat,
    ) -> Any:
        """Fallback formatting when LLM is not available"""

        logger.info("Using fallback formatting (no LLM)")

        # Simple fallback for different formats
        if output_format == OutputFormat.JSON:
            return {
                "success": any(r.success for r in execution_results.values()),
                "devices": [
                    {
                        "device": name,
                        "success": result.success,
                        "command": result.command_executed,
                        "error": result.error_message if not result.success else None,
                    }
                    for name, result in execution_results.items()
                ],
            }

        elif output_format == OutputFormat.TABLE:
            # Simple table format
            lines = ["Device | Status | Command"]
            lines.append("-" * 40)
            for name, result in execution_results.items():
                status = "✅" if result.success else "❌"
                lines.append(f"{name} | {status} | {result.command_executed}")
            return "\n".join(lines)

        else:
            # Simple text format
            results = []
            for name, result in execution_results.items():
                status = (
                    "Success" if result.success else f"Failed: {result.error_message}"
                )
                results.append(f"{name}: {status}")
            return "\n".join(results)

    def _emergency_fallback(
        self, execution_results: Dict[str, DeviceExecutionResult], error: str
    ) -> str:
        """Emergency fallback when all formatting fails"""

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
        """Get formatter statistics"""

        success_rate = 0.0
        if self.stats["total_formats"] > 0:
            success_rate = (
                self.stats["successful_formats"] / self.stats["total_formats"]
            ) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "llm_available": hasattr(self, "llm_url"),
            "direct_llm_connection": True,
        }


# Global instance
universal_formatter = UniversalFormatter()
