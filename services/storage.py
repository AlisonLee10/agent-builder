import json
import os
from datetime import datetime
from dotenv import load_dotenv

from services.database import insert_campaign
from services.logger import get_logger, get_run_id
from services.post_content import build_publishable_post, strip_sources_block

load_dotenv()
log = get_logger(__name__)

def sources_to_list(sources: str | list[str] | None) -> list[str]:
    """Normalize agent sources (string or list) for JSON storage."""
    if not sources:
        return []
    if isinstance(sources, list):
        return [s.strip() for s in sources if s and str(s).strip()]
    return [line.strip() for line in sources.splitlines() if line.strip()]


def articles_to_source_lines(articles: list[dict]) -> list[str]:
    """Build source lines from structured article/trend records."""
    lines: list[str] = []
    for a in articles:
        title = (a.get("title") or "").strip()
        url   = (a.get("url") or "").strip()
        if title and url:
            lines.append(f"• {title} — {url}")
        elif title:
            lines.append(f"• {title}")
        elif url:
            lines.append(f"• {url}")
    return lines


def normalize_research_for_save(
    sources: str | list[str] | None,
    articles: list[dict] | None,
) -> tuple[list[str], list[dict]]:
    """Ensure posted/denied campaigns persist both sources and articles when available."""
    arts = list(articles or [])
    src  = sources_to_list(sources)
    if not src and arts:
        src = articles_to_source_lines(arts)
    return src, arts


def save_campaign(
    user_prompt: str,
    content:     str,
    hashtags:    list[str],
    status:      str,
    sources:     list[str] | None = None,
    articles:    list[dict] | None = None,
    full_post:   str | None = None,
    verdict_info: dict | None = None,
    platform:          str | None = None,
    posted_platforms:  list[str] | None = None,
    run_id:            str | None = None,
    user_denial_reason: str | None = None,
) -> dict:
    os.makedirs("campaigns", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"campaigns/{status}_{timestamp}.json"
    vi = verdict_info or {}
    verdict = vi.get("verdict", "needs_revision")

    src, arts = normalize_research_for_save(sources, articles)

    stored_full_post = strip_sources_block(
        full_post if full_post is not None else content
    )
    if hashtags:
        stored_full_post = build_publishable_post(content, hashtags)

    effective_run_id = (run_id or "").strip() or get_run_id()
    if effective_run_id == "--------":
        effective_run_id = ""

    denial_reason = ""
    if status == "denied":
        if user_denial_reason and user_denial_reason.strip():
            denial_reason = user_denial_reason.strip()
        elif verdict == "rejected":
            denial_reason = vi.get("summary", "")

    data = {
        "run_id":      effective_run_id,
        "timestamp":   datetime.now().isoformat(),
        "status":      status,
        "user_prompt": user_prompt,
        "content":     content,
        "hashtags":    hashtags,
        "sources":     src,
        "articles":    arts,
        "verdict":     verdict,
        "issues":      vi.get("issues", []),
        "denial_reason": denial_reason,
        "platform":         platform or "",
        "posted_platforms": posted_platforms or [],
        "full_post":        stored_full_post,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.debug(
        f"saving campaign: status={status}, platform={platform}, "
        f"posted_platforms={posted_platforms}"
    )
    campaign_id = insert_campaign(data)
    data["id"]  = campaign_id
    log.info(f"campaign saved — id={campaign_id}, status={status}, platform={platform}")
    return {"id": campaign_id, "filename": filename}