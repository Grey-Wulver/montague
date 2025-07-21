"""
Test endpoints for Mattermost integration
"""

from fastapi import APIRouter

from app.services.mattermost_client import MattermostClient

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.get("/health")
async def chat_health():
    """Test Mattermost connection and bot authentication"""

    client = MattermostClient()
    await client.initialize()

    result = await client.test_connection()
    await client.close()

    return result


@router.get("/debug-env")
async def debug_environment():
    """Debug environment variable loading"""
    import os

    return {
        "MATTERMOST_URL": os.getenv("MATTERMOST_URL", "NOT_FOUND"),
        "MATTERMOST_BOT_TOKEN_SET": bool(os.getenv("MATTERMOST_BOT_TOKEN")),
        "MATTERMOST_BOT_TOKEN_LENGTH": len(os.getenv("MATTERMOST_BOT_TOKEN", "")),
        "ALL_ENV_VARS": [
            key for key in os.environ.keys() if "MATTERMOST" in key.upper()
        ],
    }


@router.post("/send-test")
async def send_test_message():
    """Send a test message to Town Square"""

    client = MattermostClient()
    await client.initialize()

    try:
        # Get the Town Square channel ID
        async with client.session.get(
            f"{client.base_url}/api/v4/teams/name/netops/channels/name/town-square",
            headers=client.headers,
        ) as response:
            if response.status == 200:
                channel_data = await response.json()
                channel_id = channel_data["id"]

                # Send test message
                payload = {
                    "channel_id": channel_id,
                    "message": "🤖 Hello from NetOps Bot! API integration test successful! 🚀",
                }

                async with client.session.post(
                    f"{client.base_url}/api/v4/posts",
                    headers=client.headers,
                    json=payload,
                ) as post_response:
                    if post_response.status == 201:
                        return {
                            "status": "success",
                            "message": "Test message sent to Town Square!",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Failed to send: HTTP {post_response.status}",
                        }
            else:
                return {
                    "status": "error",
                    "message": f"Town Square not found: HTTP {response.status}",
                }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        await client.close()
