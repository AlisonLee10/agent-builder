"""Prepare marketing copy for external platforms vs internal campaign storage."""

from __future__ import annotations

import re

# Platform message limits (characters)
PLATFORM_CHAR_LIMITS: dict[str, int] = {
    "discord": 2000,
    "slack":   4000,
    "gmail":   100_000,
}

_SOURCES_HEADER_RE = re.compile(
    r"\n*\s*📰\s*Sources\s*:.*",
    re.IGNORECASE | re.DOTALL,
)


def strip_sources_block(text: str) -> str:
    """Remove the Sources section from post text (not sent to platforms)."""
    return _SOURCES_HEADER_RE.sub("", text).strip()


def build_publishable_post(
    content: str,
    hashtags: str | list[str] | None = None,
) -> str:
    """Body + hashtags only — no source links (those live in campaign JSON)."""
    body = (content or "").strip()
    if isinstance(hashtags, list):
        tags = " ".join(h.strip() for h in hashtags if h and str(h).strip())
    else:
        tags = (hashtags or "").strip()
    parts = [p for p in (body, tags) if p]
    return "\n\n".join(parts)


def truncate_for_platform(text: str, platform: str) -> str:
    """Trim text to the platform's character limit, ending at a word boundary."""
    limit = PLATFORM_CHAR_LIMITS.get(platform, 2000)
    text  = strip_sources_block(text)
    if len(text) <= limit:
        return text

    suffix = "\n\n… (trimmed to fit platform limit)"
    max_body = limit - len(suffix)
    if max_body < 80:
        return text[:limit]

    cut = text[:max_body]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + suffix


def prepare_for_platform(
    content: str,
    hashtags: str | list[str] | None,
    platform: str,
) -> str:
    """Build and cap the text that is actually posted to a platform."""
    post = build_publishable_post(content, hashtags)
    return truncate_for_platform(post, platform)


def prepare_for_platforms(
    content: str,
    hashtags: str | list[str] | None,
    platforms: list[str],
) -> str:
    """Cap for the strictest limit among the requested platforms."""
    post = build_publishable_post(content, hashtags)
    post = strip_sources_block(post)
    if not platforms:
        return truncate_for_platform(post, "discord")
    limit = min(PLATFORM_CHAR_LIMITS.get(p, 2000) for p in platforms)
    strictest = min(platforms, key=lambda p: PLATFORM_CHAR_LIMITS.get(p, 2000))
    # Re-use truncate with synthetic platform key — pass limit via strictest
    if len(post) <= limit:
        return post
    return truncate_for_platform(post, strictest)
