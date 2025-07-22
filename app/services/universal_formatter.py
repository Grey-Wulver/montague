# app/services/universal_formatter.py
"""
Universal Formatter - LLM-Powered Response Formatting (ENHANCED FOR SHOWING ACTUAL DATA)
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
            # Build comprehensive context with enhanced preprocessing
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
        """Build comprehensive context for LLM formatting - INTENT-AGNOSTIC APPROACH"""

        context_parts = [
            f"User Request: {user_request}",
            f"Detected Intent: {intent}",
            f"Devices Processed: {len(execution_results)}",
            "",
            "🎯 CRITICAL TASK: Analyze the user's specific request and the actual raw command output to determine what data to extract and present.",
            "",
            "📋 ANALYSIS INSTRUCTIONS:",
            "1. READ the user's exact request - what specifically did they ask for?",
            "2. EXAMINE the raw command output - what actual data is present?",
            "3. MATCH the user request to the available data - extract what they asked for",
            "4. If the user asked for 'configuration' but output shows 'neighbors' - say so clearly",
            "5. If the user asked for 'neighbors' but output shows 'configuration' - say so clearly",
            "",
        ]

        # Add results for each device - NO PREPROCESSING, let LLM decide
        for device_name, result in execution_results.items():
            context_parts.append(f"=== DEVICE: {device_name} ===")
            context_parts.append(f"Platform: {result.platform}")
            context_parts.append(f"Command Executed: {result.command_executed}")
            context_parts.append(f"Success: {result.success}")

            if result.success and result.raw_output:
                # DEBUG: Log what we're sending to LLM
                logger.info(f"DEBUG FORMATTER: raw_output = '{result.raw_output}'")

                # Include raw output WITHOUT preprocessing - let LLM analyze it
                context_parts.append("Raw Command Output:")
                context_parts.append(f"{result.raw_output}")

                # Add general guidance instead of specific extraction hints
                context_parts.append(
                    "⚡ ANALYSIS GUIDANCE: Look at the raw output above and determine what type of data it contains. Extract the data that matches what the user specifically requested."
                )

            elif not result.success:
                context_parts.append(f"❌ ERROR: {result.error_message}")

            context_parts.append("")  # Separator

        # Add discovery information
        if discovery_results:
            context_parts.append("🔍 DISCOVERY INFORMATION:")
            for device_name, discovery in discovery_results.items():
                context_parts.append(
                    f"• {device_name}: '{discovery.discovered_command}' "
                    f"(confidence: {discovery.confidence:.2f}, "
                    f"cached: {discovery.cached_result})"
                )
            context_parts.append("")

        return "\n".join(context_parts)

    def _build_format_prompt(
        self,
        context: str,
        interface_type: InterfaceType,
        output_format: OutputFormat,
        format_options: Dict[str, Any],
    ) -> str:
        """Build format-specific prompts for different interfaces"""

        base_prompt = f"""You are a network automation assistant helping network engineers. Here's the execution context:

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
        """Chat-specific formatting instructions - SCALABLE NETWORK ENGINEERING APPROACH"""

        if output_format == OutputFormat.CONVERSATIONAL:
            return """

Format: Professional conversational response for network engineers (PLAIN TEXT, NOT JSON)

🧠 **YOU ARE AN EXPERT NETWORK ENGINEER** - Apply networking expertise to analyze and respond accurately.

**CORE PRINCIPLES:**
1. **ABSOLUTE GROUND TRUTH RULE**: You MUST ONLY report data that literally exists in the raw device output text
2. **ZERO FABRICATION TOLERANCE**: NEVER create, invent, or suggest any network data (IPs, neighbors, interfaces, routes) that is not explicitly shown in the raw output
3. **STATUS MESSAGE RECOGNITION**: Messages starting with % or # are status/error messages - explain them, don't try to extract data from them
4. **TECHNICAL PRECISION**: Use accurate networking terminology based only on actual device responses
5. **EXPLICIT DATA SOURCE**: Every technical detail must come directly from the raw output

**CRITICAL RULE FOR STATUS MESSAGES:**
If the raw output contains "% BGP inactive", "% Command not found", "% Access denied", or similar:
- This is a STATUS MESSAGE, not data
- Do NOT create fake neighbors, routes, or configurations or any fabrications ever
- Do NOT provide example data
- ONLY explain what the status message means
- Suggest alternative commands if appropriate

**DATA EXTRACTION RULES:**
- Real neighbor data looks like: "Neighbor 192.168.1.1  4  65001    123    456  never    Idle"
- Real interface data looks like: "Ethernet1/1 is up, line protocol is up"
- Real route data looks like: "10.0.0.0/24 is directly connected, Ethernet1/1"
- If you don't see actual data like this, there IS NO DATA to show

**ANALYSIS WORKFLOW:**

**Step 1: Parse User Intent**
- What specific information is the user requesting?
- Are they asking for: status, configuration, statistics, troubleshooting, or analysis?
- What level of technical detail do they expect?

**Step 2: Examine Raw Output**
- What type of data is actually present?
- Is it structured data (tables, configs) or status messages?
- Are there error indicators (%, #, warnings, failures)?

**Step 3: Match Request to Reality**
- Does the output contain what the user asked for?
- If yes → Extract and present it clearly
- If no → Explain what's actually present vs what was requested
- If partial → Show available data and note what's missing

**DEVICE OUTPUT INTERPRETATION:**

**Status Messages** (% prefix, error messages):
- Recognize these as system status, not data
- Explain what the status means in networking terms
- Suggest next steps if appropriate

**Configuration Data** (config syntax, parameters):
- Present actual config lines or settings
- Explain key parameters if relevant to user's question
- Maintain hierarchical structure

**Operational Data** (statistics, states, tables):
- Extract specific values, counters, or states
- Present in organized, readable format
- Include units and context where relevant

**RESPONSE CONSTRUCTION:**

**Professional Opening**: "Here's [what you found] for [device]:"

**Data Presentation**:
- Use bullet points (•) for lists
- Use clear hierarchical structure
- Include specific values, not just summaries
- Group related information logically

**Technical Accuracy**:
- Use proper interface names, IP addresses, protocol states
- Include relevant technical details (AS numbers, metrics, timers)
- Be precise about quantities and states

**Clear Communication**:
- Start with direct answer to user's question
- Use networking terminology appropriately
- Explain status conditions clearly

**EXAMPLE RESPONSE PATTERNS:**

**For Status Messages**:
"BGP is not currently running on spine1. The device returned '% BGP inactive', which means the BGP routing process is not configured or enabled. To check BGP configuration, you could try 'show running-config section bgp'."

**For Actual Data**:
"Here are the active interfaces on spine1:
• Ethernet1/1: UP, 1.2M packets in/987K packets out, 0 errors
• Management1: UP, 45K packets in/23K packets out, 0 errors
• Ethernet1/2: DOWN (admin down)"

**For Mismatched Requests**:
"You requested BGP configuration, but the command returned neighbor status information instead. Here's what the output shows: [actual data]. For BGP configuration, try asking for 'show running-config bgp'."

**EXPERT GUIDELINES:**

✅ **Always Do:**
- Report actual device state accurately
- Explain networking concepts when relevant
- Use specific technical values from output
- Be clear about what data is/isn't available
- Maintain professional but approachable tone

❌ **Never Do:**
- Fabricate network data not in the output
- Give generic summaries without specific information
- Ignore error messages or status indicators
- Make assumptions about network state
- Use overly complex explanations for simple requests

**Remember: You are a knowledgeable network engineer helping a colleague. Be accurate, helpful, and technically sound.**
"""

        elif output_format == OutputFormat.JSON:
            return """

Format: Clean JSON for chat display - Network Engineer Focused
- Return valid JSON object with extracted technical data
- Include "status" field indicating data type (data/status/error)
- Add "technical_summary" for network engineer context
- Structure data appropriately for network information

Example:
{
 "status": "inactive_service",
 "technical_summary": "BGP routing process not running on spine1",
 "raw_message": "% BGP inactive",
 "suggested_action": "Check BGP configuration with 'show running-config section bgp'",
 "device_context": {
   "device": "spine1",
   "platform": "arista_eos",
   "service_queried": "bgp"
 }
}
"""

        else:
            return f"\n\nFormat: {output_format.value} suitable for network engineering chat display with technical accuracy"

    def _get_api_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """API-specific formatting instructions - ENHANCED FOR ACTUAL DATA"""

        if output_format == OutputFormat.JSON:
            return """

Format: Structured JSON for API consumption
- Return clean, consistent JSON structure
- Use standard field names (snake_case)
- Include metadata (success, device_count, execution_time)
- EXTRACT AND STRUCTURE ALL ACTUAL DATA from raw output
- Provide arrays for multi-device responses

Example structure:
{
 "success": true,
 "devices": [
   {
     "device": "spine1",
     "platform": "arista_eos",
     "command_executed": "show ip route",
     "success": true,
     "extracted_data": {
       "routes": [
         {
           "destination": "10.0.0.0/24",
           "next_hop": "192.168.1.1",
           "interface": "Ethernet1/1",
           "route_type": "connected"
         }
       ]
     },
     "discovery_used": true
   }
 ],
 "summary": "Retrieved 25 routes from spine1",
 "execution_metadata": {
   "total_devices": 1,
   "successful_devices": 1,
   "total_execution_time": "2.34s"
 }
}
"""

        elif output_format == OutputFormat.YAML:
            return """

Format: Clean YAML structure
- Use proper YAML formatting
- Include consistent indentation
- Provide clear hierarchy
- Include comments where helpful
- EXTRACT ACTUAL DATA and structure hierarchically
"""

        elif output_format == OutputFormat.XML:
            return """

Format: Well-formed XML
- Use descriptive tag names
- Include proper XML declaration
- Nest data logically
- Include attributes for metadata
- EXTRACT ACTUAL DATA into structured XML elements
"""

        else:
            return f"\n\nFormat: {output_format.value} for API consumption with actual data extraction"

    def _get_cli_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """CLI-specific formatting instructions - ENHANCED FOR ACTUAL DATA"""

        if output_format == OutputFormat.TABLE:
            return """

Format: ASCII table for CLI display
- Create clean, aligned tables with headers
- Use consistent column spacing
- EXTRACT ACTUAL DATA and display in tabular format
- Limit width to ~80 characters
- Use | for column separators

Example (Interface Statistics):
| Interface   | Status | Input Pkts | Output Pkts | Input Errors |
|-------------|--------|------------|-------------|--------------|
| Ethernet1/1 | UP     | 1,234,567  | 987,654     | 0            |
| Ethernet1/2 | UP     | 2,345,678  | 1,876,543   | 3            |
| Management1 | UP     | 45,123     | 23,456      | 0            |

Example (Route Table):
| Destination  | Next Hop      | Interface   | Type      |
|--------------|---------------|-------------|-----------|
| 10.0.0.0/24  | 192.168.1.1   | Ethernet1/1 | Connected |
| 172.16.0.0/16| 10.0.0.1      | Ethernet1/1 | Static    |
"""

        elif output_format == OutputFormat.CSV:
            return """

Format: CSV for CLI export
- Include headers
- Use comma separation
- Escape special characters
- One line per device/result
- EXTRACT ACTUAL DATA into CSV format

Example:
Device,Interface,Status,InputPackets,OutputPackets,InputErrors
spine1,Ethernet1/1,UP,1234567,987654,0
spine1,Ethernet1/2,UP,2345678,1876543,3
"""

        else:
            return f"\n\nFormat: {output_format.value} for command line display with actual data"

    def _get_web_format_instructions(
        self, output_format: OutputFormat, options: Dict
    ) -> str:
        """Web UI-specific formatting instructions - ENHANCED FOR ACTUAL DATA"""

        if output_format == OutputFormat.HTML:
            return """

Format: HTML for web display
- Use semantic HTML elements
- Include CSS classes for styling
- Create tables for structured data
- Use badges/alerts for status indicators
- Make it responsive and accessible
- EXTRACT ACTUAL DATA and present in structured HTML

Example:
<div class="device-result">
 <h3>spine1 <span class="badge badge-success">Connected</span></h3>

 <h4>Interface Statistics</h4>
 <table class="table table-striped">
   <thead>
     <tr><th>Interface</th><th>Status</th><th>Input Packets</th><th>Output Packets</th></tr>
   </thead>
   <tbody>
     <tr><td>Ethernet1/1</td><td><span class="badge badge-success">UP</span></td><td>1,234,567</td><td>987,654</td></tr>
   </tbody>
 </table>
</div>
"""

        elif output_format == OutputFormat.MARKDOWN:
            return """

Format: Markdown for web/documentation
- Use proper markdown syntax
- Include headers, lists, code blocks
- Create tables where appropriate
- Use emphasis for important data
- EXTRACT ACTUAL DATA and format in markdown

Example:
# Network Data for spine1

## Interface Statistics

| Interface | Status | Input Packets | Output Packets |
|-----------|--------|---------------|----------------|
| Ethernet1/1 | **UP** | 1,234,567 | 987,654 |
| Ethernet1/2 | **UP** | 2,345,678 | 1,876,543 |

## Route Table

- **10.0.0.0/24** via 192.168.1.1 (Ethernet1/1) - Connected
- **172.16.0.0/16** via 10.0.0.1 (Ethernet1/1) - Static
"""

        else:
            return f"\n\nFormat: {output_format.value} for web display with actual data extraction"

    async def _format_with_llm_direct(
        self, prompt: str, output_format: OutputFormat
    ) -> Any:
        """Direct LLM call optimized for maximum consistency and detailed responses"""

        try:
            import aiohttp

            # Optimized parameters for consistency and detailed responses
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                # Consistency optimizations
                "temperature": 0.01,  # EVEN LOWER: Maximum determinism
                "top_p": 0.9,  # Focus on high-probability tokens
                "top_k": 40,  # Limit token choices for consistency
                "repeat_penalty": 1.1,  # Avoid repetitive text
                # Token optimizations
                "max_tokens": 4000,  # INCREASED: More room for detailed data
                "min_tokens": 150,  # Ensure substantial responses
                # Response control
                "stream": False,
                "stop": ["Human:", "Assistant:", "```"],  # Stop at common break points
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

                            # Post-process response for better consistency
                            llm_response = self._post_process_response(
                                llm_response, output_format
                            )

                            # Handle response based on expected format
                            if output_format == OutputFormat.CONVERSATIONAL:
                                return llm_response

                            elif output_format == OutputFormat.JSON:
                                try:
                                    cleaned_json = self._clean_json_response(
                                        llm_response
                                    )
                                    return json.loads(cleaned_json)
                                except json.JSONDecodeError:
                                    return {"formatted_response": llm_response}

                            else:
                                return llm_response

                    # If HTTP request failed
                    error_text = await response.text()
                    logger.error(f"LLM API error {response.status}: {error_text}")
                    raise ValueError(f"LLM API returned {response.status}")

        except Exception as e:
            logger.error(f"Direct LLM call failed: {e}")
            raise ValueError("LLM formatting failed - no response") from e

    def _post_process_response(self, response: str, output_format: OutputFormat) -> str:
        """Post-process LLM response for better consistency"""

        # Remove common inconsistencies
        response = response.strip()

        # Remove any JSON wrapper if this should be conversational
        if output_format == OutputFormat.CONVERSATIONAL:
            # Remove JSON markers that sometimes appear
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            elif response.startswith("{") and response.endswith("}"):
                # Try to extract text from JSON
                try:
                    data = json.loads(response)
                    if isinstance(data, dict):
                        # Look for common text fields
                        for field in [
                            "response",
                            "result",
                            "message",
                            "content",
                            "formatted_response",
                        ]:
                            if field in data and isinstance(data[field], str):
                                response = data[field]
                                break
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # Keep original response if JSON parsing fails

        # Ensure proper formatting for network data
        if "Here" not in response and any(
            device in response.lower()
            for device in ["spine", "leaf", "switch", "router"]
        ):
            # Add a proper intro if missing
            response = f"Here's the requested information:\n\n{response}"

        # Clean up common formatting issues
        response = response.replace("\\n", "\n")  # Fix escaped newlines
        response = "\n".join(
            line.strip() for line in response.split("\n")
        )  # Clean spacing

        return response

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
