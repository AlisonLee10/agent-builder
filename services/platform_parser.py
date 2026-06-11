"""Parse posting platforms and Gmail recipient from the user's initial prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Gmail/email posting temporarily disabled — re-enable when service account is ready.
GMAIL_POSTING_ENABLED = False
GMAIL_DISABLED_MESSAGE = (
    "Gmail/email posting is temporarily disabled. Use Discord or Slack for now."
)

SUPPORTED_PLATFORMS = ("discord", "slack")  # add "gmail" when GMAIL_POSTING_ENABLED

SUPPORTED_LABELS = "Discord / Slack"

MISSING_PLATFORMS_MESSAGE = (
    "Please include platforms to post (Discord / Slack)."
)

# Platform name -> aliases (for destination detection)
SUPPORTED_ALIASES: dict[str, list[str]] = {
    "discord": ["discord"],
    "slack": ["slack"],
    # "gmail": ["gmail", "e-mail", "email"],  # disabled — see GMAIL_POSTING_ENABLED
}

UNSUPPORTED_ALIASES: dict[str, list[str]] = {
    "instagram": ["instagram", "insta"],
    "facebook": ["facebook", "fb"],
    "twitter": ["twitter"],
    "x": [" x "],  # X/Twitter only when used as a posting destination
    "tiktok": ["tiktok", "tik tok"],
    "linkedin": ["linkedin", "linked in"],
    "youtube": ["youtube"],
    "pinterest": ["pinterest"],
    "snapchat": ["snapchat"],
    "threads": ["threads"],
    "whatsapp": ["whatsapp"],
    "telegram": ["telegram"],
    "reddit": ["reddit"],  # we have trends from reddit, not posting
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

POST_VERB_RE = re.compile(
    r"\b(?:post|send|publish|share|push|cross-?post)\b",
    re.I,
)

# "post to Discord and Slack", "post it on gmail and discord"
POST_DESTINATION_RE = re.compile(
    r"(?:post|send|publish|share|push)(?:\s+\w+){0,6}?\s+"
    r"(?:to|on|via|through|using)\s+"
    r"([^.!?\n;]+)",
    re.I,
)

AND_LIST_RE = re.compile(
    r"\b(discord|slack|gmail|e-?mail|email|instagram|facebook|twitter|tiktok|linkedin)\b",
    re.I,
)


@dataclass
class PlatformIntent:
    platforms: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    gmail_to: str | None = None
    gmail_subject: str | None = None

    @property
    def has_posting_intent(self) -> bool:
        return bool(self.platforms) or bool(self.unsupported)


def _normalize_platform_key(alias: str) -> str | None:
    a = alias.lower().strip()
    for platform, aliases in SUPPORTED_ALIASES.items():
        if a in aliases:
            return platform
    return None


def _find_emails(prompt: str) -> list[str]:
    return EMAIL_RE.findall(prompt)


def _gmail_posting_requested(text: str) -> bool:
    """True when the user asks to send/post via email (even if gmail is disabled)."""
    lower = text.lower()
    if not POST_VERB_RE.search(text):
        return False
    if re.search(r"\b(?:gmail|e-?mail|email)\b", lower):
        return True
    if EMAIL_RE.search(text) and re.search(
        r"\b(?:send|email|mail)\b", lower
    ):
        return True
    return False


def _alias_in_text(alias: str, text: str) -> bool:
    if alias.strip() == "x":
        return bool(re.search(r"\bpost\b.*\bto\s+x\b|\bto\s+x\b.*\bpost\b", text, re.I))
    return alias.lower() in text.lower()


def _mentioned_as_destination(prompt: str, aliases: list[str]) -> bool:
    lower = prompt.lower()
    if not any(_alias_in_text(a, prompt) for a in aliases):
        return False

    if POST_VERB_RE.search(prompt):
        for m in POST_DESTINATION_RE.finditer(prompt):
            chunk = m.group(1).lower()
            if any(a.lower() in chunk for a in aliases):
                return True
        if re.search(
            r"(?:to|on|via|through)\s+[^.!?\n;]*(?:"
            + "|".join(re.escape(a.lower()) for a in aliases)
            + r")",
            lower,
        ):
            return True
        for a in aliases:
            if re.search(
                rf"(?:discord|slack|gmail|e-?mail|email|instagram|facebook|twitter|tiktok|linkedin)\s+and\s+{re.escape(a.lower())}",
                lower,
            ):
                return True
            if re.search(
                rf"{re.escape(a.lower())}\s+and\s+(?:discord|slack|gmail|e-?mail|email)",
                lower,
            ):
                return True

    return False


def _collect_supported_from_chunk(chunk: str) -> list[str]:
    found: list[str] = []
    for token in re.split(r"[\s,/+\-&]+", chunk.lower()):
        key = _normalize_platform_key(token)
        if key and key not in found:
            found.append(key)
    for platform, aliases in SUPPORTED_ALIASES.items():
        if platform in found:
            continue
        if any(a.lower() in chunk.lower() for a in aliases):
            if platform not in found:
                found.append(platform)
    return found


def parse_platform_intent(prompt: str) -> PlatformIntent:
    """Extract requested platforms, unsupported names, and Gmail recipient."""
    text = prompt.strip()
    intent = PlatformIntent()
    if not text:
        return intent

    lower = text.lower()
    platforms: list[str] = []

    for m in POST_DESTINATION_RE.finditer(text):
        for p in _collect_supported_from_chunk(m.group(1)):
            if p not in platforms:
                platforms.append(p)

    if POST_VERB_RE.search(text):
        for m in AND_LIST_RE.finditer(text):
            key = _normalize_platform_key(m.group(1))
            if key and key not in platforms:
                platforms.append(key)

    # Gmail auto-detection disabled while GMAIL_POSTING_ENABLED is False
    # if GMAIL_POSTING_ENABLED and (...):
    #     platforms.append("gmail")

    intent.platforms = [p for p in SUPPORTED_PLATFORMS if p in platforms]

    unsupported: list[str] = []
    for name, aliases in UNSUPPORTED_ALIASES.items():
        if _mentioned_as_destination(text, aliases):
            unsupported.append(name.capitalize())
    intent.unsupported = unsupported

    emails = _find_emails(text)
    if emails:
        intent.gmail_to = emails[0]
    if "gmail" in intent.platforms and not intent.gmail_to:
        near = re.search(
            r"(?:to|send(?:\s+it)?\s+to|email\s+to)\s+"
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            text,
            re.I,
        )
        if near:
            intent.gmail_to = near.group(1)

    subj = re.search(
        r"subject\s*(?::|=)?\s*['\"]?([^'\".\n]+?)['\"]?(?:\s+and\s+|\s*$|\.)",
        text,
        re.I,
    )
    if subj:
        intent.gmail_subject = subj.group(1).strip()

    return intent


def unsupported_platform_message(unsupported: list[str]) -> str:
    names = ", ".join(unsupported)
    verb = "is" if len(unsupported) == 1 else "are"
    return (
        f"{names} {verb} not listed yet. "
        f"Please choose other platform: {SUPPORTED_LABELS}."
    )


def format_platform_plan(intent: PlatformIntent) -> str:
    if not intent.platforms:
        return ""
    labels = [p.capitalize() for p in intent.platforms]
    plan = "Will post to: " + ", ".join(labels)
    if GMAIL_POSTING_ENABLED and "gmail" in intent.platforms and intent.gmail_to:
        plan += f" (email to {intent.gmail_to})"
    return plan


def validate_posting_intent(prompt: str) -> tuple[bool, str, PlatformIntent]:
    """
    Ensure the prompt names supported platforms before content generation.
    Returns (ok, error_message, parsed_intent).
    """
    intent = parse_platform_intent(prompt)

    if not GMAIL_POSTING_ENABLED and _gmail_posting_requested(prompt):
        return False, GMAIL_DISABLED_MESSAGE, intent

    if intent.unsupported:
        return False, unsupported_platform_message(intent.unsupported), intent

    if not intent.platforms:
        return False, MISSING_PLATFORMS_MESSAGE, intent

    if GMAIL_POSTING_ENABLED and "gmail" in intent.platforms and not intent.gmail_to:
        return (
            False,
            "Gmail was requested but no recipient email was found in your prompt. "
            "Include an address, e.g. 'send to team@company.com via Gmail'.",
            intent,
        )

    return True, "", intent
