# Saves campaign results as json files, and they go to testers/campaigns folder
# 1e

import json
import os
from datetime import datetime

def save_campaign(user_prompt: str, content: str, hashtags: list[str], status: str) -> str:
    os.makedirs("campaigns", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"campaigns/{status}_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "status": status, # posted or denied
        "user_prompt": user_prompt,
        "content": content,
        "hashtags": hashtags,
        "full_post": f"{content}\n\n{' '.join(hashtags)}",
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filename