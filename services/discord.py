import requests
import os
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

def post_to_discord(content: str) -> bool:
    result = requests.post(WEBHOOK_URL, json={"content": content})
    return result.status_code == 204