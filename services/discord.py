import requests
import os
from dotenv import load_dotenv
from services.logger import get_logger
log = get_logger(__name__)

load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# 1a
def post_to_discord(content: str) -> bool:
    if not WEBHOOK_URL:
        log.error("WEBHOOK_URL is not set — cannot post to Discord")
        return False

    log.debug(f"posting to Discord — {len(content)} chars")
    result = requests.post(WEBHOOK_URL, json={"content": content}, timeout=10)
    if result.status_code == 204:
        log.debug("Discord accepted — HTTP 204")
        return True
    log.error(f"Discord rejected — HTTP {result.status_code}: {result.text}")
    return False
