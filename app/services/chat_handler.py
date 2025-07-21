"""
Main Chat Handler - Orchestrates the complete ChatOps workflow
Connects Mattermost → NLP → Your existing Intent Executor → Response formatting
"""

import json
import logging
import time
from typing import Dict

from app.services.chat_nlp import ChatNLP, ParsedCommand
from app.services.mattermost_client import ChatMessage, MattermostClient

logger = logging.getLogger(__name__)


class ChatHandler:
    """Main chat interaction orchestrator"""

    def __init__(self):
        self.mattermost_client = MattermostClient()

        # Initialize NLP with LLM normalizer if available
        try:
            from app.services.local_llm_normalizer import local_llm_normalizer

            self.chat_nlp = ChatNLP(llm_normalizer=local_llm_normalizer)
            logger.info("ChatNLP initialized with LLM support")
        except ImportError as e:
            logger.warning(
                f"LLM normalizer not available: {e} - using pattern matching only"
            )
            self.chat_nlp = ChatNLP(llm_normalizer=None)

        self.is_running = False

        # Simple conversation memory (optional enhancement)
        self.user_contexts: Dict[str, Dict] = {}

    async def initialize(self):
        """Initialize all chat services"""
        logger.info("🚀 Initializing ChatOps handler...")

        # Initialize Mattermost connection
        if not await self.mattermost_client.initialize():
            logger.error("❌ Failed to initialize Mattermost client")
            return False

        logger.info("✅ Mattermost client initialized")

        # Connect WebSocket for real-time messages
        if not await self.mattermost_client.connect_websocket():
            logger.error("❌ Failed to connect WebSocket")
            return False

        logger.info("✅ WebSocket connected")
        logger.info("🎧 ChatOps handler ready to process commands!")
        return True

    async def start_listening(self):
        """Start listening for chat messages"""
        if not await self.initialize():
            return

        self.is_running = True
        logger.info("🎯 Starting ChatOps message listener...")

        try:
            await self.mattermost_client.listen_for_messages(self.handle_message)
        except Exception as e:
            logger.error(f"ChatOps listening error: {e}")
        finally:
            self.is_running = False

    async def handle_message(self, chat_message: ChatMessage):
        """Handle incoming chat message - the main workflow"""
        try:
            logger.info(
                f"📨 Processing command: '{chat_message.clean_message}' from user {chat_message.user_id}"
            )

            # Send "processing" indicator
            await self.mattermost_client.send_message(
                chat_message.channel_id, "🤖 Processing your network command..."
            )

            start_time = time.time()

            # Step 1: Check for help request
            if self.chat_nlp.is_help_request(chat_message.clean_message):
                await self._send_help_response(chat_message.channel_id)
                return

            # Step 2: Parse the command using NLP
            parsed_command = await self.chat_nlp.process_command(
                chat_message.clean_message
            )

            logger.info(
                f"🧠 Parsed intent: {parsed_command.intent} (confidence: {parsed_command.confidence:.2f}, method: {parsed_command.method})"
            )

            # Step 3: Handle unknown intents
            if parsed_command.intent == "unknown":
                await self._send_unknown_command_response(
                    chat_message.channel_id, chat_message.clean_message
                )
                return

            # Step 4: Execute the network operation
            execution_result = await self._execute_network_operation(parsed_command)

            # Step 5: Calculate timing and send response
            execution_time = time.time() - start_time
            await self._send_network_operation_response(
                chat_message.channel_id,
                parsed_command,
                execution_result,
                execution_time,
            )

            # Step 6: Update conversation context (optional)
            self._update_user_context(
                chat_message.user_id, parsed_command, execution_result
            )

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error_response(chat_message.channel_id, str(e))

    async def _execute_network_operation(self, command: ParsedCommand) -> Dict:
        """Execute network operation using your existing intent executor"""
        try:
            # Import your existing intent executor and models
            from app.models.intents import IntentRequest
            from app.services.intent_executor import intent_executor

            # Handle device selection
            devices = command.devices
            if "all" in devices:
                # Get all devices from your inventory
                try:
                    from app.services.inventory import inventory_service

                    all_devices = await inventory_service.get_all_device_names()
                    devices = (
                        all_devices if all_devices else ["spine1", "leaf1", "leaf2"]
                    )
                except ImportError:
                    # Fallback to lab devices if inventory service not available
                    devices = ["spine1", "leaf1", "leaf2"]

            logger.info(f"🔧 Executing {command.intent} on devices: {devices}")

            # Create IntentRequest object - CORRECTED
            intent_request = IntentRequest(
                intent=command.intent,
                devices=devices,
                parameters=command.parameters,
                timeout=30,  # Default timeout
            )

            # Execute via your existing intent executor
            result = await intent_executor.execute_intent(intent_request)

            # Convert IntentResponse to dict format for chat response
            return {
                "status": "success"
                if result.failed == 0
                else ("partial" if result.successful > 0 else "error"),
                "intent": result.intent,
                "total_devices": result.total_devices,
                "successful_devices": result.successful,
                "failed_devices": result.failed,
                "results": [
                    {
                        "device": r.device,
                        "success": r.status == "success",
                        "status": r.status,
                        "normalized_output": r.normalized_output,
                        "raw_output": r.raw_output,
                        "execution_time": r.execution_time,
                        "error_message": r.error_message,
                        "normalization_method": r.normalization_method,
                        "confidence_score": r.confidence_score,
                    }
                    for r in result.results
                ],
                "normalization_method": result.results[0].normalization_method
                if result.results
                else "none",
            }

        except Exception as e:
            logger.error(f"Network operation execution failed: {e}")
            return {
                "status": "error",
                "message": f"Execution failed: {str(e)}",
                "devices": command.devices,
                "intent": command.intent,
            }

    async def _send_network_operation_response(
        self,
        channel_id: str,
        command: ParsedCommand,
        result: Dict,
        execution_time: float,
    ):
        """Format and send network operation results"""

        # Determine response color based on status
        if result.get("status") == "success":
            color = "#36a64f"  # Green
            emoji = "✅"
        elif result.get("status") == "partial":
            color = "#ff9900"  # Orange
            emoji = "⚠️"
        else:
            color = "#ff0000"  # Red
            emoji = "❌"

        # Create title
        intent_display = command.intent.replace("_", " ").title()
        title = f"{emoji} {intent_display} Results"

        # Create summary message
        summary_lines = []
        if "results" in result:
            device_count = len(result["results"])
            success_count = sum(1 for r in result["results"] if r.get("success", False))
            summary_lines.append(
                f"**Devices:** {success_count}/{device_count} successful"
            )

        summary_lines.append(f"**Execution Time:** {execution_time:.2f}s")
        summary_lines.append(
            f"**Method:** {command.method} + {result.get('normalization_method', 'manual')}"
        )

        summary_text = "\n".join(summary_lines)

        # Create fields for structured data
        fields = []

        # Add device-specific results if available
        if (
            "results" in result and len(result["results"]) <= 3
        ):  # Don't overwhelm chat for many devices
            for device_result in result["results"][:3]:
                device_name = device_result.get("device", "Unknown")
                device_status = (
                    "✅ Success" if device_result.get("success") else "❌ Failed"
                )

                # Add key info from normalized output
                device_info = []
                if (
                    "normalized_output" in device_result
                    and device_result["normalized_output"]
                ):
                    norm_data = device_result["normalized_output"]

                    # Add relevant fields based on intent - UPDATED INTENT NAMES
                    if command.intent == "get_system_version":
                        if "version" in norm_data:
                            device_info.append(f"Version: {norm_data['version']}")
                        if "model" in norm_data:
                            device_info.append(f"Model: {norm_data['model']}")

                    elif command.intent == "get_interface_status":
                        if "total_interfaces" in norm_data:
                            total = norm_data.get("total_interfaces", 0)
                            up = norm_data.get("up_interfaces", 0)
                            device_info.append(f"Interfaces: {up}/{total} up")

                    elif command.intent == "get_bgp_summary":
                        if "total_neighbors" in norm_data:
                            total = norm_data.get("total_neighbors", 0)
                            established = norm_data.get("established_neighbors", 0)
                            device_info.append(
                                f"BGP: {established}/{total} established"
                            )

                field_value = device_status
                if device_info:
                    field_value += f"\n{', '.join(device_info[:2])}"  # Limit to 2 items

                fields.append(
                    {"title": device_name, "value": field_value, "short": True}
                )

        # Send the structured response
        await self.mattermost_client.send_structured_response(
            channel_id=channel_id,
            title=title,
            message=summary_text,
            color=color,
            fields=fields,
        )

        # For detailed results, send as code block (optional)
        if result.get("status") == "success" and len(str(result)) < 1500:  # Size limit
            normalized_results = {}
            if "results" in result:
                for r in result["results"]:
                    if r.get("normalized_output"):
                        normalized_results[r["device"]] = r["normalized_output"]

            if normalized_results:
                code_block = f"```json\n{json.dumps(normalized_results, indent=2)}\n```"
                await self.mattermost_client.send_message(
                    channel_id, f"**📊 Detailed Output:**\n{code_block}"
                )

    async def _send_help_response(self, channel_id: str):
        """Send help message"""
        help_text = self.chat_nlp.generate_help_message()

        await self.mattermost_client.send_structured_response(
            channel_id=channel_id,
            title="🤖 NetOps Bot Help",
            message=help_text,
            color="#439fe0",  # Blue
        )

    async def _send_unknown_command_response(
        self, channel_id: str, original_message: str
    ):
        """Send response for unknown commands"""
        message = f"""
🤔 I'm not sure how to process: *"{original_message}"*

**Try commands like:**
• `@netops-bot show version on spine1`
• `@netops-bot check interfaces on all devices`
• `@netops-bot bgp summary on leaf1,leaf2`
• `@netops-bot help` for full command list

**💡 Tip:** Describe what network information you want to check!
        """

        await self.mattermost_client.send_structured_response(
            channel_id=channel_id,
            title="❓ Command Not Recognized",
            message=message,
            color="#ff9900",  # Orange
        )

    async def _send_error_response(self, channel_id: str, error: str):
        """Send error response"""
        message = f"**Error Details:**\n{error}\n\n*Please try again or contact your network admin.*"

        await self.mattermost_client.send_structured_response(
            channel_id=channel_id,
            title="🚨 Execution Error",
            message=message,
            color="#ff0000",  # Red
        )

    def _update_user_context(self, user_id: str, command: ParsedCommand, result: Dict):
        """Update conversation context for user (optional feature)"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {"history": [], "last_devices": []}

        context = self.user_contexts[user_id]

        # Add to history (keep last 5 commands)
        context["history"].append(
            {
                "intent": command.intent,
                "devices": command.devices,
                "success": result.get("status") == "success",
                "timestamp": time.time(),
            }
        )

        if len(context["history"]) > 5:
            context["history"].pop(0)

        # Remember last devices used
        context["last_devices"] = command.devices

    async def stop_listening(self):
        """Stop the chat handler"""
        self.is_running = False
        if self.mattermost_client:
            await self.mattermost_client.close()
        logger.info("🔄 ChatOps handler stopped")


# Global instance for the chat handler
chat_handler = ChatHandler()
