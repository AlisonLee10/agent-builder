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
        model = self.config.get("model", "gpt-4o-mini")
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        temperature = float(self.config.get("temperature", 0.7))
        user_input = str(inputs.get("input", ""))

        if model.startswith("claude"):
            import anthropic
            client = anthropic.AsyncAnthropic()
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
            llm = ChatOpenAI(model=model, temperature=temperature)
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


NODE_TYPE_MAP: dict[str, type[Node]] = {
    "input":     InputNode,
    "llm":       LLMNode,
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
        "type": "output",
        "label": "Output",
        "icon": "📤",
        "color": "#ef4444",
        "inputs": 1,
        "outputs": 0,
        "fields": [],
    },
]
