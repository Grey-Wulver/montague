"""
Simple Mattermost client for NetOps ChatBot integration
"""

import logging
import os
from typing import Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class MattermostClient:
    """Basic Mattermost client for bot interactions"""

    def __init__(self):
        self.base_url = os.getenv("MATTERMOST_URL", "http://192.168.1.100:8065")
        self.bot_token = os.getenv("MATTERMOST_BOT_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self.bot_user_id: Optional[str] = None

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
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
