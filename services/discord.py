import requests
import os
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# 1a
def post_to_discord(content: str) -> bool:
    if not WEBHOOK_URL:
        print("  [Discord] WEBHOOK_URL not set in .env")
        return False
    result = requests.post(WEBHOOK_URL, json={"content": content})
    return result.status_code == 204