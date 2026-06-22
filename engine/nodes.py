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
        model       = self.config.get("model", "gpt-4o-mini")
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        temperature = float(self.config.get("temperature", 0.7))
        api_key     = self.config.get("api_key", "").strip() or None
        user_input  = str(inputs.get("input", ""))

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
        text = str(inputs.get("input", ""))
        condition = self.config.get("condition", "")
        if not condition:
            return {"output": text, "route": "true"}
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = await llm.ainvoke([
            SystemMessage(content="Answer ONLY 'true' or 'false'. No explanation."),
            HumanMessage(content=f"Condition: {condition}\n\nInput: {text}"),
        ])
        route = "true" if "true" in response.content.lower() else "false"
        return {"output": text, "route": route}


class OutputNode(Node):
    async def run(self, inputs: dict[str, Any]) -> Any:
        return inputs.get("input", "")


class AgentNode(Node):
    """A named sub-agent with its own instructions, model, and tool attachments."""

    async def run(self, inputs: dict[str, Any]) -> Any:
        name         = self.config.get("name", "Agent")
        instructions = self.config.get("instructions", "You are a helpful assistant.")
        model        = self.config.get("model", "gpt-4o-mini")
        domain_id    = self.config.get("domain_id", "")

        # tool_ids stored as comma-separated string or list
        raw_ids = self.config.get("tool_ids", "")
        tool_ids: list[str] = (
            [t.strip() for t in raw_ids.split(",") if t.strip()]
            if isinstance(raw_ids, str)
            else [str(t) for t in raw_ids if t]
        )

        user_input = str(inputs.get("input", ""))

        # Build system prompt: identity + instructions, then domain context, then RAG
        system = f"You are {name}. {instructions}"
        if domain_id:
            ctx = _load_domain_context(domain_id)
            if ctx:
                system = f"{system}\n\n{ctx}"
        rag = _load_rag_context()
        if rag:
            system = f"{system}\n\n## Company Context\n\n{rag}"

        # Collect LangChain tools
        from engine.builtin_tools import get_builtin_tool
        from engine.registry import get_tool_instance
        from langchain_core.tools import Tool as LCTool

        lc_tools: list[LCTool] = []
        for tid in tool_ids:
            t = get_builtin_tool(tid) or get_tool_instance(tid)
            if isinstance(t, LCTool):
                lc_tools.append(t)

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

        # Agent with tools
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model=model, temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent    = create_tool_calling_agent(llm, lc_tools, prompt)
        executor = AgentExecutor(agent=agent, tools=lc_tools, max_iterations=6, verbose=False)
        result   = await asyncio.to_thread(executor.invoke, {"input": user_input})
        return result.get("output", "")


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
    "input":     InputNode,
    "llm":       LLMNode,
    "agent":     AgentNode,
    "tool":      ToolNode,
    "condition": ConditionNode,
    "output":    OutputNode,
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
                "key": "condition",
                "label": "Condition (plain English)",
                "type": "text",
                "default": "",
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
            {"key": "tool_ids",   "label": "Tools",   "type": "tool_multiselect"},
            {"key": "domain_id",  "label": "Domain",  "type": "domain_select"},
        ],
    },
    {
        "type": "output",
        "label": "Output",
        "icon": "📤",
        "color": "#ef4444",
        "inputs": 1,
        "outputs": 0,
        "fields": [],
    },
]
