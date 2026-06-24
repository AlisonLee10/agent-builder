"""
API key store — reads from data/api_keys.json first, falls back to os.environ.
All engine code should call get_key() instead of os.getenv() for user-managed keys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_PATH = Path(__file__).parent.parent / "data" / "api_keys.json"

# Keys shown in the sidebar UI with their metadata
KEY_REGISTRY: list[dict] = [
    {
        "env":         "OPENAI_API_KEY",
        "label":       "OpenAI",
        "icon":        "🤖",
        "description": "Required for GPT-4o, DALL-E, Summarizer, Classifier, CSV Analyzer, Translation",
    },
    {
        "env":         "ANTHROPIC_API_KEY",
        "label":       "Anthropic",
        "icon":        "🧠",
        "description": "Required for Claude models (claude-sonnet-4-6, claude-haiku-4-5)",
    },
    {
        "env":         "NEWS_API_KEY",
        "label":       "NewsAPI",
        "icon":        "📰",
        "description": "Required for the News Fetch built-in tool",
    },
    {
        "env":         "RUNWAYML_API_KEY",
        "label":       "RunwayML",
        "icon":        "🎬",
        "description": "Required for the Video Generation built-in tool",
    },
    {
        "env":         "SERP_API_KEY",
        "label":       "SerpAPI",
        "icon":        "🔍",
        "description": "Used by the Web Search built-in tool (alternative to Tavily)",
    },
    {
        "env":         "TAVILY_API_KEY",
        "label":       "Tavily",
        "icon":        "🔎",
        "description": "Used by the Web Search built-in tool (alternative to SerpAPI)",
    },
    {
        "env":         "LANGCHAIN_API_KEY",
        "label":       "LangSmith",
        "icon":        "🔗",
        "description": "Optional — enables LangChain tracing and observability",
    },
    {
        "env":         "GMAIL_ADDRESS",
        "label":       "Gmail Address",
        "icon":        "📧",
        "description": "Your Gmail address used as the sender (e.g. you@gmail.com)",
    },
    {
        "env":         "GMAIL_APP_PASSWORD",
        "label":       "Gmail App Password",
        "icon":        "🔑",
        "description": "16-character App Password from Google Account → Security → App Passwords (requires 2-Step Verification)",
    },
    {
        "env":         "DISCORD_WEBHOOK_URL",
        "label":       "Discord Webhook URL",
        "icon":        "🎮",
        "description": "Incoming Webhook URL from Discord Server Settings → Integrations → Webhooks",
    },
]


def _load() -> dict[str, str]:
    if not _PATH.exists():
        return {}
    try:
        return json.loads(_PATH.read_text())
    except Exception:
        return {}


def _save(data: dict[str, str]) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(data, indent=2))


def get_key(env_name: str) -> str:
    """Return the key value: stored file takes priority over environment."""
    return _load().get(env_name) or os.environ.get(env_name, "")


def set_key(env_name: str, value: str) -> None:
    data = _load()
    data[env_name] = value.strip()
    _save(data)
    # Also inject into the live process env so libraries that read os.environ directly benefit
    os.environ[env_name] = value.strip()


def delete_key(env_name: str) -> bool:
    data = _load()
    if env_name not in data:
        return False
    del data[env_name]
    _save(data)
    os.environ.pop(env_name, None)
    return True


def list_keys() -> list[dict]:
    """Return KEY_REGISTRY entries annotated with whether the key is currently set."""
    stored = _load()
    result = []
    for entry in KEY_REGISTRY:
        env = entry["env"]
        is_set = bool(stored.get(env) or os.environ.get(env, ""))
        result.append({**entry, "is_set": is_set})
    return result
