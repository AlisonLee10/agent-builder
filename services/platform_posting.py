"""Post approved content to platforms requested in the user's prompt."""

from __future__ import annotations

import asyncio

from services.logger import get_logger
from services.platform_parser import PlatformIntent
from services.post_content import prepare_for_platform

log = get_logger(__name__)


async def post_to_platforms(
    full_post: str,
    intent: PlatformIntent,
    *,
    content: str = "",
    hashtags: str | list[str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Post to each platform in intent.platforms.
    Returns (posted_platforms, failed_platforms, errors_by_platform).
    """
    posted: list[str] = []
    failed: list[str] = []
    errors: dict[str, str] = {}

    for platform in intent.platforms:
        ok, err = await _post_one(full_post, platform, intent, content, hashtags)
        if ok:
            posted.append(platform)
            log.info(f"posted to {platform}")
        else:
            failed.append(platform)
            errors[platform] = err or "unknown error"
            log.warning(f"post to {platform} failed: {err}")

    return posted, failed, errors


async def _post_one(
    full_post: str,
    platform: str,
    intent: PlatformIntent,
    content: str,
    hashtags: str | list[str] | None,
) -> tuple[bool, str | None]:
    body = prepare_for_platform(
        content or full_post,
        hashtags,
        platform,
    )
    limit = {"discord": 2000, "slack": 4000, "gmail": 100_000}.get(platform, 2000)
    if len(body) >= limit - 50:
        log.info(f"Posting to {platform} — {len(body)} chars (limit {limit})")

    if platform == "discord":
        from services.discord import post_to_discord

        ok = await asyncio.to_thread(post_to_discord, body)
        return ok, None if ok else "Discord post failed"

    if platform == "slack":
        from services.slack import post_to_slack

        ok = await asyncio.to_thread(post_to_slack, body)
        return ok, None if ok else "Slack post failed"

    if platform == "gmail":
        from services.gmail import send_email

        recipient = intent.gmail_to
        if not recipient:
            return False, "No recipient email address found in your prompt."
        subject = intent.gmail_subject or "Message from Agent"
        ok, err = await asyncio.to_thread(send_email, recipient, subject, body)
        return ok, err

    return False, f"Unknown platform: {platform}"
