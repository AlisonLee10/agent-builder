import json
import os
from datetime import datetime


def sources_to_list(sources: str | list[str] | None) -> list[str]:
    """Normalize agent sources (string or list) for JSON storage."""
    if not sources:
        return []
    if isinstance(sources, list):
        return [s.strip() for s in sources if s and str(s).strip()]
    return [line.strip() for line in sources.splitlines() if line.strip()]


def save_campaign(
    user_prompt: str,
    content:     str,
    hashtags:    list[str],
    status:      str,
    sources:     list[str] | None = None,
    articles:    list[dict] | None = None,
    full_post:   str | None = None,
    verdict_info: dict | None = None,
) -> str:
    os.makedirs("campaigns", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"campaigns/{status}_{timestamp}.json"

    vi = verdict_info or {}

    data = {
        "timestamp":   datetime.now().isoformat(),
        "status":      status,
        "user_prompt": user_prompt,
        "content":     content,
        "hashtags":    hashtags,
        "sources":     sources or [],
        "articles":    articles or [],
        "verdict": vi.get("verdict", "needs_revision"),
        "issues": vi.get("issues", []),
        "denial_reason": vi.get("summary", ""),
        "full_post":   full_post if full_post is not None else content,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filename