import os
from dotenv import load_dotenv
from services.logger import get_logger

load_dotenv()
log = get_logger(__name__)


def post_to_slack(
    content:  str,
    channel:  str | None = None,
) -> bool:
    """
    Post content to a Slack channel.
    Falls back to SLACK_CHANNEL_ID from .env if no channel provided.
    """
    from slack_sdk         import WebClient
    from slack_sdk.errors  import SlackApiError

    token      = os.getenv("SLACK_BOT_TOKEN", "")
    channel_id = channel or os.getenv("SLACK_CHANNEL_ID", "")

    if not token:
        log.error("SLACK_BOT_TOKEN not set in .env")
        return False
    if not channel_id:
        log.error("SLACK_CHANNEL_ID not set in .env")
        return False

    client = WebClient(token=token)
    log.debug(f"posting to Slack channel {channel_id} — {len(content)} chars")

    try:
        client.chat_postMessage(channel=channel_id, text=content)
        log.debug("Slack post successful")
        return True
    except SlackApiError as e:
        log.error(f"Slack post failed: {e.response['error']}")
        return False