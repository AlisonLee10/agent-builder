import json
import os
from datetime import datetime


def save_campaign(
    user_prompt: str,
    content:     str,
    hashtags:    list[str],
    status:      str,
    sources:     list[str] | None = None,
    articles:    list[dict] | None = None,
    full_post:   str | None = None,
) -> str:
    os.makedirs("campaigns", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"campaigns/{status}_{timestamp}.json"

    data = {
        "timestamp":   datetime.now().isoformat(),
        "status":      status,
        "user_prompt": user_prompt,
        "content":     content,
        "hashtags":    hashtags,
        "sources":     sources or [],
        "articles":    articles or [],
        "full_post":   full_post if full_post is not None else content,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filename