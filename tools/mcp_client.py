"""
MCP Client Manager
==================
Defines which MCP servers are connected and provides
the async context manager used by the agent.

Location: tools/mcp_client.py
"""

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from services.logger import get_logger

load_dotenv()
log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BRAND_CONTEXT_SERVER = str(_PROJECT_ROOT / "mcp_servers" / "brand_context_server.py")

# Slack MCP can post before user approval — posting uses services/slack on approve.
AGENT_MCP_SKIP_SERVERS: frozenset[str] = frozenset({"slack"})

# MCP tools that duplicate native LangChain tools (native names kept).
MCP_SKIP_TOOL_NAMES: frozenset[str] = frozenset({"retrieve_brand_context"})


# -- Server registry ----------------------------------------------
# Uncomment each block as you complete the corresponding step.
# Key = server name used in logs

MCP_SERVERS: dict[str, Any] = {

    # --- #5c: Tavily (replaces SerpAPI trends) ------------
    "tavily": {
        "command":   "npx",
        "args":      ["-y", "tavily-mcp"],
        "env":       {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
        "transport": "stdio",
    },

    # ── #5e: Web Fetch (replaces urllib calls) ────────────────
    "fetch": {
        "command":   "uvx",
        "args":      ["mcp-server-fetch"],
        "transport": "stdio",
    },

    # ── #5e: Slack (new posting platform) ────────────────────
    "slack": {
        "command":   "npx",
        "args":      ["-y", "@modelcontextprotocol/server-slack"],
        "env":       {
            "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN", ""),
            "SLACK_TEAM_ID":   os.getenv("SLACK_TEAM_ID",   ""),
        },
        "transport": "stdio",
    },

    # ── SQLite (optional — enable when a working MCP sqlite package is installed) ──
    # "sqlite": {
    #     "command":   "npx",
    #     "args":      ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "campaigns.db"],
    #     "transport": "stdio",
    # },

    # ── Custom Brand Context MCP (built in-house) ────────────
    "brand_context": {
        "command":   sys.executable,
        "args":      [_BRAND_CONTEXT_SERVER],
        "transport": "stdio",
        "cwd":       str(_PROJECT_ROOT),
    },

    # ── #5g: Gmail (email newsletter) ────────────────────────
    # "gmail": {
    #     "command":   "npx",
    #     "args":      ["-y", "@modelcontextprotocol/server-gmail"],
    #     "env":       {"GMAIL_CREDENTIALS": os.getenv("GMAIL_CREDENTIALS", "")},
    #     "transport": "stdio",
    # },
}


def has_servers() -> bool:
    """True if at least one server is configured and uncommented."""
    return bool(MCP_SERVERS)


def get_mcp_client() -> "_EmptyClient | _MCPClientAdapter":
    """
    Returns an async context manager that connects to all configured
    MCP servers and exposes their tools.

    Usage (inside an async function):
        async with get_mcp_client() as client:
            mcp_tools = await client.get_tools()

    When no servers are configured, returns _EmptyClient()
    which behaves identically but returns an empty tool list.
    """
    if not has_servers():
        log.debug("[MCP] no servers configured - using empty client")
        return _EmptyClient()

    from langchain_mcp_adapters.client import MultiServerMCPClient

    log.info(f"[MCP] connecting to {len(MCP_SERVERS)} server(s): {list(MCP_SERVERS)}")
    return _MCPClientAdapter(MultiServerMCPClient(MCP_SERVERS))


class _MCPClientAdapter:
    """Wraps MultiServerMCPClient (no longer supports `async with` directly)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self) -> "_MCPClientAdapter":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def get_tools(self) -> list:
        return await self._client.get_tools()


def merge_agent_tools(native_tools: list, mcp_tools: list) -> list:
    """Combine native + MCP tools; skip duplicates and agent-blocklisted MCP tools."""
    native_names = {getattr(t, "name", "") for t in native_tools}
    merged       = list(native_tools)

    for tool in mcp_tools:
        name = getattr(tool, "name", "") or ""
        if not name or name in native_names or name in MCP_SKIP_TOOL_NAMES:
            continue
        merged.append(tool)
        native_names.add(name)

    return merged


async def load_mcp_tools_for_agent() -> list:
    """
    Load MCP tools server-by-server so one failure does not block the rest.
    Skips servers in AGENT_MCP_SKIP_SERVERS (e.g. Slack — approve flow posts separately).
    """
    if not has_servers():
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    all_tools: list = []
    for name, cfg in MCP_SERVERS.items():
        if name in AGENT_MCP_SKIP_SERVERS:
            log.debug(f"[MCP] skipping {name} for content agent (not used during /api/run)")
            continue
        try:
            client = MultiServerMCPClient({name: cfg})
            async with _MCPClientAdapter(client) as session:
                server_tools = await session.get_tools()
            all_tools.extend(server_tools)
            log.info(f"[MCP] loaded {len(server_tools)} tool(s) from '{name}'")
        except Exception as e:
            log.warning(f"[MCP] server '{name}' unavailable — {e}")

    return all_tools


class _EmptyClient:
    """
    Placeholder used when MCP_SERVERS is empty.
    Lets the agent code use the same pattern unconditionally
    without checking whether MCP is configured.
    """
    async def __aenter__(self) -> "_EmptyClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def get_tools(self) -> list:
        return []