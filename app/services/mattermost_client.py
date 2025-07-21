"""
Enhanced Mattermost client with WebSocket listening for real-time chat
"""

import json
import logging
import os
from typing import Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class ChatMessage:
    """Represents an incoming chat message"""

    def __init__(self, data: dict):
        self.id = data.get("id")
        self.channel_id = data.get("channel_id")
        self.user_id = data.get("user_id")
        self.message = data.get("message", "").strip()
        self.timestamp = data.get("create_at", 0)
        self.mentions_bot = "@netops-bot" in self.message.lower()

        # Clean message by removing bot mention
        if self.mentions_bot:
            self.clean_message = self.message.replace("@netops-bot", "").strip()
        else:
            self.clean_message = self.message


class MattermostClient:
    """Enhanced Mattermost client with WebSocket listening"""

    def __init__(self):
        self.base_url = os.getenv("MATTERMOST_URL", "http://192.168.1.100:8065")
        self.bot_token = os.getenv("MATTERMOST_BOT_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.bot_user_id: Optional[str] = None
        self.is_listening = False

    async def initialize(self):
        """Initialize client and verify connection"""
        self.session = aiohttp.ClientSession()

        if not self.bot_token:
            logger.warning("No bot token provided")
            return False

        try:
            async with self.session.get(
                f"{self.base_url}/api/v4/users/me", headers=self.headers
            ) as response:
                if response.status == 200:
                    user_data = await response.json()
                    self.bot_user_id = user_data["id"]
                    logger.info(
                        f"Connected as bot: {user_data.get('username', 'unknown')}"
                    )
                    return True
                else:
                    logger.error(f"Failed to authenticate: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def connect_websocket(self):
        """Connect to Mattermost WebSocket for real-time events"""
        if not self.session or not self.bot_user_id:
            raise Exception("Client not initialized")

        # WebSocket URL
        ws_url = (
            self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            + "/api/v4/websocket"
        )

        try:
            self.websocket = await self.session.ws_connect(
                ws_url, headers={"Authorization": f"Bearer {self.bot_token}"}
            )
            logger.info("WebSocket connected successfully")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def listen_for_messages(self, message_handler: Callable):
        """Listen for incoming messages and route to handler"""
        if not self.websocket:
            if not await self.connect_websocket():
                return

        self.is_listening = True
        logger.info("🎧 Started listening for chat messages...")

        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        # We're interested in 'posted' events (new messages)
                        if data.get("event") == "posted":
                            await self._handle_posted_event(data, message_handler)

                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON from WebSocket")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.websocket.exception()}")
                    break

        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
        finally:
            self.is_listening = False
            logger.info("Stopped listening for messages")

    async def _handle_posted_event(self, data: Dict, handler: Callable):
        """Process a 'posted' event from WebSocket"""
        try:
            # Extract post data from the event
            post_data = json.loads(data["data"]["post"])

            # Ignore our own messages
            if post_data.get("user_id") == self.bot_user_id:
                return

            # Create ChatMessage object
            chat_message = ChatMessage(post_data)

            # Only process messages that mention the bot or are direct messages
            if chat_message.mentions_bot:
                logger.info(
                    f"📨 Received command: '{chat_message.clean_message}' in channel {chat_message.channel_id}"
                )
                await handler(chat_message)

        except Exception as e:
            logger.error(f"Error handling posted event: {e}")

    async def send_message(self, channel_id: str, message: str) -> bool:
        """Send a simple message to a channel"""
        if not self.session or not self.bot_token:
            logger.error("Client not initialized")
            return False

        payload = {"channel_id": channel_id, "message": message}

        try:
            async with self.session.post(
                f"{self.base_url}/api/v4/posts", headers=self.headers, json=payload
            ) as response:
                if response.status == 201:
                    logger.info(f"📤 Message sent to channel {channel_id}")
                    return True
                else:
                    logger.error(f"Failed to send message: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def send_structured_response(
        self,
        channel_id: str,
        title: str,
        message: str,
        color: str = "#36a64f",
        fields: List[Dict] = None,
    ) -> bool:
        """Send a structured response with attachments"""

        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "fields": fields or [],
        }

        payload = {
            "channel_id": channel_id,
            "message": "",
            "props": {"attachments": [attachment]},
        }

        try:
            async with self.session.post(
                f"{self.base_url}/api/v4/posts", headers=self.headers, json=payload
            ) as response:
                return response.status == 201

        except Exception as e:
            logger.error(f"Failed to send structured message: {e}")
            return False

    async def send_typing_indicator(self, channel_id: str):
        """Send typing indicator (Mattermost doesn't support this, so we'll send a quick status)"""
        await self.send_message(channel_id, "🤖 Processing your request...")

    async def get_channel_info(self, channel_id: str) -> Dict:
        """Get information about a channel"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/v4/channels/{channel_id}", headers=self.headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return {}

    async def test_connection(self) -> Dict:
        """Test connection to Mattermost"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/v4/system/ping"
            ) as response:
                if response.status == 200:
                    ping_data = await response.json()
                    return {
                        "status": "connected",
                        "version": ping_data.get("version", "unknown"),
                        "bot_authenticated": self.bot_user_id is not None,
                        "websocket_ready": self.websocket is not None,
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def close(self):
        """Close all connections"""
        self.is_listening = False

        if self.websocket:
            await self.websocket.close()

        if self.session:
            await self.session.close()

        logger.info("Mattermost client connections closed")
