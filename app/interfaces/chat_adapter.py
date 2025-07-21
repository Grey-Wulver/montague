# app/interfaces/chat_adapter.py
"""
Chat Interface Adapter - Complete Implementation
Replaces chat_handler.py with Universal Pipeline integration
Connects Mattermost → Universal Request Processor → Formatted Response
"""

import logging
import time
from typing import Dict

from app.models.universal_requests import (
    ChatRequest,
    OutputFormat,
    UniversalResponse,
    create_chat_request,
)
from app.services.mattermost_client import ChatMessage, MattermostClient

logger = logging.getLogger(__name__)


class ChatAdapter:
    """
    Chat interface adapter that connects chat platforms to the Universal Pipeline
    Handles Mattermost, Slack, Teams, or any chat platform through the same interface
    """

    def __init__(self):
        self.mattermost_client = MattermostClient()
        self._universal_processor = None
        self._intent_parser = None

        # Chat-specific configuration
        self.is_running = False
        self.response_timeout = 30  # seconds

        # Simple conversation memory (per session)
        self.user_contexts: Dict[str, Dict] = {}

        # Chat formatting preferences
        self.chat_config = {
            "show_discovery_info": True,
            "show_execution_time": True,
            "show_device_count": True,
            "max_devices_in_response": 10,
            "use_conversational_format": True,
        }

    async def initialize(self):
        """Initialize chat adapter and all dependencies"""
        logger.info("🚀 Initializing Chat Adapter...")

        try:
            # Initialize Universal Processor
            from app.services.intent_parser import intent_parser
            from app.services.universal_request_processor import universal_processor

            self._universal_processor = universal_processor
            self._intent_parser = intent_parser

            # Initialize universal processor if not already done
            if not await self._universal_processor.initialize():
                logger.error("❌ Failed to initialize Universal Processor")
                return False

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
            logger.info("🎧 Chat Adapter ready to process commands!")

            return True

        except Exception as e:
            logger.error(f"❌ Chat Adapter initialization failed: {e}")
            return False

    async def start_listening(self):
        """Start listening for chat messages"""
        if not await self.initialize():
            logger.error("Chat Adapter initialization failed")
            return

        self.is_running = True
        logger.info("🎯 Starting Chat Adapter message listener...")

        try:
            await self.mattermost_client.listen_for_messages(self.handle_message)
        except Exception as e:
            logger.error(f"Chat listening error: {e}")
        finally:
            self.is_running = False

    async def handle_message(self, chat_message: ChatMessage):
        """
        Handle incoming chat message using Universal Pipeline
        THE MAIN CHAT WORKFLOW - now powered by Universal Pipeline
        """
        try:
            logger.info(
                f"📨 Processing chat command: '{chat_message.clean_message}' from user {chat_message.user_id}"
            )

            # Send "processing" indicator
            await self.mattermost_client.send_message(
                chat_message.channel_id, "🤖 Processing your network command..."
            )

            start_time = time.time()

            # Step 1: Check for help request
            if self._is_help_request(chat_message.clean_message):
                await self._send_help_response(chat_message.channel_id)
                return

            # Step 2: Parse devices from message
            devices = self._intent_parser.extract_devices(chat_message.clean_message)

            # Step 3: Create Universal Request for chat interface
            chat_request = create_chat_request(
                message=chat_message.clean_message,
                devices=devices,
                user_id=chat_message.user_id,
                session_id=chat_message.channel_id,  # Use channel as session
            )

            # Configure chat-specific options
            chat_request.include_discovery_info = self.chat_config[
                "show_discovery_info"
            ]
            chat_request.output_format = OutputFormat.CONVERSATIONAL
            chat_request.timeout = self.response_timeout

            logger.info(
                f"🔧 Sending request to Universal Pipeline: intent extraction + execution on {len(devices)} devices"
            )

            # Step 4: Process through Universal Pipeline
            universal_response = await self._universal_processor.process_request(
                chat_request
            )

            # Step 5: Send formatted chat response
            await self._send_chat_response(chat_message.channel_id, universal_response)

            # Step 6: Update conversation context
            self._update_user_context(
                chat_message.user_id, chat_request, universal_response
            )

            execution_time = time.time() - start_time
            logger.info(f"✅ Chat workflow completed in {execution_time:.2f}s")

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            await self._send_error_response(chat_message.channel_id, str(e))

    async def _send_chat_response(self, channel_id: str, response: UniversalResponse):
        """
        Send Universal Response as a formatted chat message
        This replaces all the complex formatting logic in the old chat_handler
        """
        try:
            # The Universal Formatter already created a conversational response
            formatted_content = response.formatted_response

            # Determine status color
            if (
                response.success
                and response.successful_devices == response.total_devices
            ):
                color = "#36a64f"  # Green - complete success
            elif response.success and response.successful_devices > 0:
                color = "#ff9900"  # Orange - partial success
            else:
                color = "#ff0000"  # Red - failure

            # Create chat-friendly title
            title = self._create_chat_title(response)

            # Add metadata if configured
            metadata_lines = []
            if self.chat_config["show_device_count"]:
                metadata_lines.append(
                    f"**Devices:** {response.successful_devices}/{response.total_devices} successful"
                )

            if self.chat_config["show_execution_time"]:
                metadata_lines.append(
                    f"**Execution Time:** {response.execution_time:.2f}s"
                )

            if self.chat_config["show_discovery_info"] and response.discovery_used:
                metadata_lines.append("**Method:** AI Discovery + LLM")
                if response.new_discoveries > 0:
                    metadata_lines.append(
                        "💡 **Learning Update:** I learned new command mappings!"
                    )

            # Combine title, metadata, and formatted response
            message_parts = []
            if title:
                message_parts.append(f"## {title}")
                message_parts.append("")

            if metadata_lines:
                message_parts.extend(metadata_lines)
                message_parts.append("")

            # Add the main formatted response
            if isinstance(formatted_content, str):
                message_parts.append(formatted_content)
            elif isinstance(formatted_content, dict):
                # Handle JSON response in chat
                if "chat_friendly" in formatted_content:
                    message_parts.append(formatted_content["chat_friendly"])
                else:
                    message_parts.append(str(formatted_content))
            else:
                message_parts.append(str(formatted_content))

            final_message = "\n".join(message_parts)

            # Send structured response
            await self.mattermost_client.send_structured_response(
                channel_id=channel_id,
                title="",  # Empty since title is in message
                message=final_message,
                color=color,
            )

            logger.info(f"✅ Chat response sent - Success: {response.success}")

        except Exception as e:
            logger.error(f"Failed to send chat response: {e}")
            # Fallback to simple message
            fallback_message = f"**Command Results:** {response.successful_devices}/{response.total_devices} devices successful"
            await self.mattermost_client.send_message(channel_id, fallback_message)

    def _create_chat_title(self, response: UniversalResponse) -> str:
        """Create a user-friendly title for chat response"""

        # Extract intent from metadata
        intent = (
            response.interface_metadata.get("intent", "Command")
            if response.interface_metadata
            else "Command"
        )

        # Convert intent to human-readable title
        title = intent.replace("_", " ").title()

        # Add discovery indicator
        if response.discovery_used:
            if response.new_discoveries > 0:
                title += " (AI Discovered)"
            else:
                title += " (Cached Discovery)"

        # Add status emoji
        if response.success and response.successful_devices == response.total_devices:
            status_emoji = "✅"
        elif response.success and response.successful_devices > 0:
            status_emoji = "⚠️"
        else:
            status_emoji = "❌"

        return f"{status_emoji} {title} Results"

    def _is_help_request(self, message: str) -> bool:
        """Check if message is a help request"""
        help_keywords = ["help", "usage", "commands", "what can you do", "?"]
        return any(keyword in message.lower() for keyword in help_keywords)

    async def _send_help_response(self, channel_id: str):
        """Send help message using intent parser"""
        help_text = self._intent_parser.generate_help_message()

        await self.mattermost_client.send_structured_response(
            channel_id=channel_id,
            title="🤖 NetOps Bot Help",
            message=help_text,
            color="#439fe0",  # Blue
        )

    async def _send_error_response(self, channel_id: str, error: str):
        """Send error response"""
        message = f"**Error:** {error}\n\n*Please try again or ask for help.*"

        await self.mattermost_client.send_structured_response(
            channel_id=channel_id,
            title="🚨 Execution Error",
            message=message,
            color="#ff0000",  # Red
        )

    def _update_user_context(
        self, user_id: str, request: ChatRequest, response: UniversalResponse
    ):
        """Update conversation context for user"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {"history": [], "last_devices": []}

        context = self.user_contexts[user_id]

        # Add to history (keep last 5 commands)
        context["history"].append(
            {
                "request": request.user_input,
                "devices": request.devices,
                "success": response.success,
                "timestamp": time.time(),
                "discovery_used": response.discovery_used,
            }
        )

        if len(context["history"]) > 5:
            context["history"].pop(0)

        # Remember last devices used
        context["last_devices"] = request.devices

    async def stop_listening(self):
        """Stop the chat adapter"""
        self.is_running = False
        if self.mattermost_client:
            await self.mattermost_client.close()
        logger.info("🔄 Chat Adapter stopped")

    async def health_check(self) -> Dict[str, str]:
        """Check health of chat adapter"""
        status = {
            "chat_adapter": "healthy",
            "mattermost": "unknown",
            "universal_processor": "unknown",
        }

        try:
            # Check Mattermost connection
            if self.mattermost_client and hasattr(
                self.mattermost_client, "health_check"
            ):
                status["mattermost"] = (
                    "healthy"
                    if await self.mattermost_client.health_check()
                    else "unhealthy"
                )

            # Check Universal Processor
            if self._universal_processor:
                processor_health = await self._universal_processor.health_check()
                status["universal_processor"] = (
                    "healthy"
                    if processor_health.get("universal_processor") == "healthy"
                    else "unhealthy"
                )

        except Exception as e:
            status["chat_adapter"] = f"error: {str(e)}"

        return status

    def get_stats(self) -> Dict[str, any]:
        """Get chat adapter statistics"""

        total_users = len(self.user_contexts)
        total_conversations = sum(
            len(ctx["history"]) for ctx in self.user_contexts.values()
        )

        stats = {
            "active_users": total_users,
            "total_conversations": total_conversations,
            "is_running": self.is_running,
            "chat_config": self.chat_config,
        }

        # Add Universal Processor stats if available
        if self._universal_processor:
            stats["universal_processor_stats"] = self._universal_processor.get_stats()

        return stats


# Global instance
chat_adapter = ChatAdapter()
