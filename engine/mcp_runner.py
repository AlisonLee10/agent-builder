"""Load registered MCP servers as LangChain-compatible tools."""
from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REGISTRY_PATH = DATA_DIR / "registry.json"


def _is_native(mcp_env: dict) -> bool:
    """Return True for MCP entries that are handled natively (no subprocess)."""
    if "GMAIL_ADDRESS" in mcp_env and "GMAIL_APP_PASSWORD" in mcp_env:
        return True
    if "DISCORD_WEBHOOK_URL" in mcp_env:
        return True
    return False


def get_native_mcp_tools(mcp_ids: list[str]) -> list:
    """
    For MCP entries that have native Python implementations (e.g. Gmail SMTP),
    inject their credentials into key_store and return LangChain Tool instances
    directly instead of spawning a subprocess.
    """
    if not mcp_ids:
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return []
    mcps_by_id = {m["id"]: m for m in data.get("mcps", [])}
    tools = []
    for mcp_id in mcp_ids:
        mcp = mcps_by_id.get(mcp_id)
        if not mcp:
            continue
        mcp_env = mcp.get("env", {})
        if not _is_native(mcp_env):
            continue
        if "GMAIL_ADDRESS" in mcp_env and "GMAIL_APP_PASSWORD" in mcp_env:
            from engine.key_store import set_key
            set_key("GMAIL_ADDRESS",      mcp_env["GMAIL_ADDRESS"])
            set_key("GMAIL_APP_PASSWORD", mcp_env["GMAIL_APP_PASSWORD"])
            from engine.builtin_tools import get_builtin_tool
            tool = get_builtin_tool("gmail_send")
            if tool:
                tools.append(tool)
        elif "DISCORD_WEBHOOK_URL" in mcp_env:
            from engine.builtin_tools import get_builtin_tool
            from engine.key_store import set_key
            set_key("DISCORD_WEBHOOK_URL", mcp_env["DISCORD_WEBHOOK_URL"])
            tool = get_builtin_tool("discord_send")
            if tool:
                tools.append(tool)
    return tools


def get_mcp_server_configs(mcp_ids: list[str]) -> dict:
    """
    Return a MultiServerMCPClient-compatible config dict for the given MCP IDs.
    Keys are server display names; values have command/args/env/transport.
    The subprocess env merges os.environ with the stored credentials so that
    PATH, HOME, etc. are available to npx.
    """
    if not mcp_ids:
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return {}
    mcps_by_id = {m["id"]: m for m in data.get("mcps", [])}
    configs: dict = {}
    for mcp_id in mcp_ids:
        mcp = mcps_by_id.get(mcp_id)
        if not mcp:
            continue
        mcp_env = mcp.get("env", {})

        # Gmail (and any future native integrations) are handled via get_native_mcp_tools
        if _is_native(mcp_env):
            continue

        configs[mcp["name"]] = {
            "command":   mcp["command"],
            "args":      mcp.get("args", []),
            # Merge parent env first so PATH/HOME reach npx, then overlay credentials
            "env":       {**os.environ, **mcp_env},
            "transport": "stdio",
        }
    return configs
