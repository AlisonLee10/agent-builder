"""Node type definitions for the workflow engine."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class Node(ABC):
    def __init__(self, node_id: str, config: dict[str, Any]):
        self.node_id = node_id
        self.config = config

    @abstractmethod
    async def run(self, inputs: dict[str, Any]) -> Any:
        pass


class InputNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        # Runtime prompt (from Run modal) takes priority; node's typed prompt is the fallback.
        return inputs.get("prompt") or self.config.get("prompt", "")


class LLMNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        from datetime import datetime
        model       = self.config.get("model", "gpt-4o-mini")
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        temperature = float(self.config.get("temperature", 0.7))
        api_key     = self.config.get("api_key", "").strip() or None
        user_input  = str(inputs.get("input", ""))
        now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        system_prompt = f"Today is {now}.\n\n{system_prompt}"

        if model.startswith("claude"):
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
            msg = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            )
            return msg.content[0].text
        else:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            kwargs = {"model": model, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            llm = ChatOpenAI(**kwargs)
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ])
            return response.content


class ToolNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        from engine.registry import get_tool_instance
        tool_id = self.config.get("tool_id", "")
        if not tool_id:
            return {"error": "No tool configured on this node"}
        tool = get_tool_instance(tool_id)
        if not tool:
            return {"error": f"Tool '{tool_id}' not found in registry"}
        query = str(inputs.get("input", ""))
        result = await asyncio.to_thread(tool, query)
        return result


class ConditionNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        import json as _j
        text = str(inputs.get("input", ""))

        # Parse conditions list; fall back to legacy single-condition field
        raw = self.config.get("conditions", "[]")
        try:
            conditions = _j.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            conditions = []
        if not conditions:
            legacy = self.config.get("condition", "").strip()
            conditions = [{"label": "if", "expr": legacy}] if legacy else []

        if not conditions:
            return {"output": text, "route_port": "output_1"}

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        for i, cond in enumerate(conditions):
            expr = cond.get("expr", "").strip()
            if not expr:
                continue
            resp = await llm.ainvoke([
                SystemMessage(content="Answer ONLY 'true' or 'false'. No explanation."),
                HumanMessage(content=f"Condition: {expr}\n\nInput: {text}"),
            ])
            if "true" in resp.content.strip().lower():
                return {"output": text, "route_port": f"output_{i + 1}"}

        # No condition matched → else branch (last output port)
        return {"output": text, "route_port": f"output_{len(conditions) + 1}"}


class ApprovalNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        text = str(inputs.get("input", ""))
        raw = inputs.get("__decision__", "")
        # Accept either a plain string ("approve"/"reject") or a dict with edited content
        if isinstance(raw, dict):
            decision = raw.get("decision", "").strip().lower()
            text = raw.get("edited_content") or text
        else:
            decision = str(raw).strip().lower()
        if decision == "approve":
            return {"output": text, "route_port": "output_1"}
        if decision == "reject":
            return {"output": text, "route_port": "output_2"}
        return {
            "__approval_pending__": True,
            "node_id": self.node_id,
            "preview": text,
            "message": self.config.get("message", "Please review the content and choose to approve or reject."),
        }


class OutputNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        text = str(inputs.get("input", ""))
        delivery = self.config.get("delivery", "none")
        if delivery == "slack":
            from engine.delivery import deliver_slack
            deliver_slack(text)
        elif delivery == "gmail":
            from engine.delivery import deliver_gmail
            recipient = self.config.get("recipient", "").strip()
            subject   = self.config.get("subject", "").strip()
            deliver_gmail(text, recipient, subject)
        elif delivery == "discord":
            from engine.delivery import deliver_discord
            deliver_discord(text)
        return text


class AgentNode(Node):
    """A named sub-agent with its own instructions, model, and tool attachments."""

    async def run(self, inputs: dict[str, Any]) -> Any:
        name         = self.config.get("name", "Agent")
        instructions = self.config.get("instructions", "You are a helpful assistant.")
        model        = self.config.get("model", "gpt-4o-mini")
        domain_id    = self.config.get("domain_id", "")

        def _parse_ids(raw: Any) -> list[str]:
            if isinstance(raw, str):
                return [x.strip() for x in raw.split(",") if x.strip()]
            return [str(x) for x in raw if x]

        tool_ids = _parse_ids(self.config.get("tool_ids", ""))
        mcp_ids  = _parse_ids(self.config.get("mcp_ids",  ""))

        user_input = str(inputs.get("input", ""))

        # Build system prompt: date + identity + instructions, then domain context, then RAG
        from datetime import datetime
        now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        system = f"Today is {now}.\n\nYou are {name}. {instructions}"
        if domain_id:
            ctx = _load_domain_context(domain_id)
            if ctx:
                system = f"{system}\n\n{ctx}"
        rag = _load_rag_context()
        if rag:
            system = f"{system}\n\n## Company Context\n\n{rag}"

        # Collect built-in / HTTP tools
        from engine.builtin_tools import get_builtin_tool
        from engine.registry import get_tool_instance
        from langchain_core.tools import Tool as LCTool

        lc_tools: list[LCTool] = []
        for tid in tool_ids:
            t = get_builtin_tool(tid) or get_tool_instance(tid)
            if isinstance(t, LCTool):
                lc_tools.append(t)

        # Run inside MCP context if any MCP servers are selected
        from engine.mcp_runner import get_mcp_server_configs, get_native_mcp_tools
        mcp_configs   = get_mcp_server_configs(mcp_ids)
        lc_tools      = lc_tools + get_native_mcp_tools(mcp_ids)

        if mcp_configs:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            try:
                mcp_client = MultiServerMCPClient(mcp_configs)
                mcp_tools = await mcp_client.get_tools()
            except BaseException as exc:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                root = exc
                while hasattr(root, "exceptions") and root.exceptions:
                    root = root.exceptions[0]
                raise RuntimeError(
                    f"MCP server failed to start ({type(root).__name__}): {root}"
                ) from root
            return await self._invoke(model, system, user_input, lc_tools + mcp_tools)
        else:
            return await self._invoke(model, system, user_input, lc_tools)

    async def _invoke(self, model: str, system: str, user_input: str, lc_tools: list) -> Any:
        if not lc_tools:
            if model.startswith("claude"):
                import anthropic
                client = anthropic.AsyncAnthropic()
                msg = await client.messages.create(
                    model=model, max_tokens=4096,
                    system=system,
                    messages=[{"role": "user", "content": user_input}],
                )
                return msg.content[0].text
            else:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage
                llm = ChatOpenAI(model=model, temperature=0.7)
                resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user_input)])
                return resp.content

        # Agent with tools — use LangChain 1.x create_agent (backed by LangGraph)
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm    = ChatOpenAI(model=model, temperature=0.7)
        agent  = create_agent(llm, lc_tools, system_prompt=system)
        result = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})

        # The final AI reply is the last non-tool-call message
        for msg in reversed(result.get("messages", [])):
            content = getattr(msg, "content", None)
            if content and not getattr(msg, "tool_calls", None):
                return content if isinstance(content, str) else str(content)
        return ""


def _load_domain_context(domain_id: str) -> str:
    """
    Return context for a domain. If the domain entry has a "folder" key pointing
    to a structured domain pack (domain.yaml + governance/ + training_data/),
    loads rich context from there. Otherwise falls back to the plain "context"
    string in domains_config.json.
    """
    import json
    from pathlib import Path
    path = Path(__file__).parent.parent / "data" / "domains_config.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        for d in data.get("builtin", []) + data.get("user", []):
            if d["id"] == domain_id:
                folder = d.get("folder")
                if folder:
                    try:
                        from engine.domain_loader import load_rich_context
                        return load_rich_context(folder)
                    except Exception:
                        pass  # fall through to plain context
                return d.get("context", "")
    except Exception:
        pass
    return ""


def _load_rag_context() -> str:
    """Concatenate all user RAG documents into a single block for injection."""
    import json
    from pathlib import Path
    path = Path(__file__).parent.parent / "data" / "rag_docs.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        docs = data.get("docs", [])
        if not docs:
            return ""
        parts = []
        for doc in docs:
            header = f"### {doc.get('icon', '📄')} {doc['name']}"
            if doc.get("description"):
                header += f" — {doc['description']}"
            parts.append(f"{header}\n\n{doc.get('content', '').strip()}")
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""


NODE_TYPE_MAP: dict[str, type[Node]] = {
    "input":    InputNode,
    "llm":      LLMNode,
    "agent":    AgentNode,
    "tool":     ToolNode,
    "condition": ConditionNode,
    "approval": ApprovalNode,
    "output":   OutputNode,
}

# Schema consumed by the frontend to render the node palette and config panels.
NODE_TYPE_SCHEMA = [
    {
        "type": "input",
        "label": "Input",
        "icon": "📥",
        "color": "#10b981",
        "inputs": 0,
        "outputs": 1,
        "fields": [],
    },
    {
        "type": "llm",
        "label": "LLM",
        "icon": "🤖",
        "color": "#3b82f6",
        "inputs": 1,
        "outputs": 1,
        "fields": [
            {
                "key": "model",
                "label": "Model",
                "type": "select",
                "options": [
                    "gpt-4o", "gpt-4o-mini",
                    "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
                ],
                "default": "gpt-4o-mini",
            },
            {
                "key": "system_prompt",
                "label": "System Prompt",
                "type": "textarea",
                "default": "You are a helpful assistant.",
            },
            {
                "key": "temperature",
                "label": "Temperature",
                "type": "number",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
            },
            {
                "key": "api_key",
                "label": "API Key",
                "type": "password",
                "default": "",
            },
        ],
    },
    {
        "type": "tool",
        "label": "Tool",
        "icon": "🔧",
        "color": "#f59e0b",
        "inputs": 1,
        "outputs": 1,
        "fields": [
            {"key": "tool_id", "label": "Tool", "type": "tool_select"},
        ],
    },
    {
        "type": "condition",
        "label": "Condition",
        "icon": "🔀",
        "color": "#8b5cf6",
        "inputs": 1,
        "outputs": 2,
        "fields": [
            {
                "key": "conditions",
                "label": "Conditions",
                "type": "condition_list",
                "default": '[{"label":"if","expr":""}]',
            },
        ],
    },
    {
        "type": "agent",
        "label": "Agent",
        "icon": "🤖",
        "color": "#0891b2",
        "inputs": 1,
        "outputs": 1,
        "fields": [
            {"key": "name",         "label": "Agent Name",    "type": "text",     "default": "My Agent"},
            {"key": "instructions", "label": "Instructions",  "type": "textarea", "default": "You are a helpful assistant."},
            {
                "key": "model", "label": "Model", "type": "select",
                "options": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                "default": "gpt-4o-mini",
            },
            {"key": "tool_ids",  "label": "Tools",       "type": "tool_multiselect"},
            {"key": "mcp_ids",   "label": "MCP Servers", "type": "mcp_multiselect"},
            {"key": "domain_id", "label": "Domain",      "type": "domain_select"},
        ],
    },
    {
        "type": "approval",
        "label": "Human Approval",
        "icon": "🙋",
        "color": "#f97316",
        "inputs": 1,
        "outputs": 2,
        "fields": [
            {
                "key": "message",
                "label": "Review Prompt",
                "type": "textarea",
                "default": "Please review the content above and choose to approve or reject.",
            },
        ],
    },
    {
        "type": "output",
        "label": "Output",
        "icon": "📤",
        "color": "#ef4444",
        "inputs": 1,
        "outputs": 0,
        "fields": [
            {
                "key": "delivery",
                "label": "Deliver output to",
                "type": "select",
                "options": ["none", "slack", "gmail", "discord"],
                "default": "none",
            },
            {
                "key": "recipient",
                "label": "Recipient email (Gmail only)",
                "type": "text",
                "default": "",
            },
            {
                "key": "subject",
                "label": "Email subject (Gmail only)",
                "type": "text",
                "default": "Workflow Output",
            },
        ],
    },
]
