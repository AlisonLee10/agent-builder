import json
import os
from datetime import datetime
from dotenv import load_dotenv

from services.database import insert_campaign
from services.logger import get_logger, get_run_id

load_dotenv()
log = get_logger(__name__)

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
    platform:    str | None = None,
) -> dict:
    os.makedirs("campaigns", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"campaigns/{status}_{timestamp}.json"
    vi = verdict_info or {}

    data = {
        "run_id":      get_run_id(),
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
        "platform":    platform or "",
        "full_post":   full_post if full_post is not None else content,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.debug(f"saving campaign: status={status}, platform={platform}")
    campaign_id = insert_campaign(data)
    data["id"]  = campaign_id
    log.info(f"campaign saved — id={campaign_id}, status={status}, platform={platform}")
    return {"id": campaign_id, "filename": filename}