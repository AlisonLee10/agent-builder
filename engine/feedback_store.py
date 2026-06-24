"""Rejection feedback store.

When a Human Approval node denies an output, the content is appended here.
AgentNode reads recent rejections and injects them into system prompts as
negative examples so the agent can learn to avoid similar outputs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_STORE = Path(__file__).parent.parent / "data" / "feedback" / "rejected.jsonl"


def record_rejection(content: str, node_id: str = "") -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "node_id":   node_id,
        "content":   content,
    }
    with _STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_rejections(limit: int = 5) -> list[dict]:
    """Return the most recent `limit` rejection entries, newest first."""
    if not _STORE.exists():
        return []
    lines = _STORE.read_text(encoding="utf-8").strip().splitlines()
    entries: list[dict] = []
    for line in reversed(lines):
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
        if len(entries) >= limit:
            break
    return entries


def clear_rejections() -> int:
    """Delete all stored rejections. Returns the number of entries cleared."""
    if not _STORE.exists():
        return 0
    lines = [l for l in _STORE.read_text(encoding="utf-8").splitlines() if l.strip()]
    count = len(lines)
    _STORE.write_text("", encoding="utf-8")
    return count


def list_all_rejections() -> list[dict]:
    """Return every stored rejection entry, newest first."""
    if not _STORE.exists():
        return []
    lines = _STORE.read_text(encoding="utf-8").strip().splitlines()
    entries: list[dict] = []
    for line in reversed(lines):
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries
