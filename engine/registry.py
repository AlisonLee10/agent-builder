"""User-managed tool and MCP registry backed by data/registry.json."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "registry.json"

_DEFAULT: dict = {"tools": [], "mcps": []}


def _load() -> dict:
    if not REGISTRY_PATH.exists():
        return {"tools": [], "mcps": []}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except json.JSONDecodeError:
        return {"tools": [], "mcps": []}


def _save(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2))


# ── Tools ─────────────────────────────────────────────────────────────────────

def list_tools() -> list[dict]:
    return _load()["tools"]


def add_tool(entry: dict) -> dict:
    data = _load()
    entry.setdefault("id", str(uuid.uuid4()))
    data["tools"].append(entry)
    _save(data)
    return entry


def remove_tool(tool_id: str) -> bool:
    data = _load()
    before = len(data["tools"])
    data["tools"] = [t for t in data["tools"] if t["id"] != tool_id]
    changed = len(data["tools"]) < before
    if changed:
        _save(data)
    return changed


def get_tool_instance(tool_id: str) -> Callable[[str], str] | None:
    """Return a callable(query) -> str for the registered tool, or None."""
    entry = next((t for t in list_tools() if t["id"] == tool_id), None)
    if not entry:
        return None

    kind = entry.get("kind", "http")

    if kind == "http":
        import requests
        url = entry.get("url", "")
        method = entry.get("method", "POST").upper()
        headers = entry.get("headers", {})

        def _http(query: str) -> str:
            payload = {"query": query, "input": query}
            resp = requests.request(method, url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text

        return _http

    if kind == "tavily":
        import os
        os.environ.setdefault("TAVILY_API_KEY", entry.get("api_key", ""))
        from langchain_community.tools.tavily_search import TavilySearchResults
        _tavily = TavilySearchResults(max_results=5)
        return lambda q: str(_tavily.run(q))

    if kind == "serpapi":
        from langchain_community.utilities import SerpAPIWrapper
        search = SerpAPIWrapper(serpapi_api_key=entry.get("api_key", ""))
        return search.run

    return None


# ── MCPs ──────────────────────────────────────────────────────────────────────

def list_mcps() -> list[dict]:
    return _load().get("mcps", [])


def add_mcp(entry: dict) -> dict:
    data = _load()
    entry.setdefault("id", str(uuid.uuid4()))
    data.setdefault("mcps", []).append(entry)
    _save(data)
    return entry


def remove_mcp(mcp_id: str) -> bool:
    data = _load()
    before = len(data.get("mcps", []))
    data["mcps"] = [m for m in data.get("mcps", []) if m["id"] != mcp_id]
    changed = len(data["mcps"]) < before
    if changed:
        _save(data)
    return changed
