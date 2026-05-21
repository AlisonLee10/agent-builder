"""Send agent tool-loop scratchpad to LangSmith (not the terminal)."""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def tracing_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")


def build_agent_scratchpad_payload(
    intermediate_steps: list,
    final_output: str | None,
) -> dict[str, Any]:
    """Structured view of the growing agent_scratchpad (tool calls + results)."""
    iterations: list[dict[str, Any]] = []
    for i, (action, observation) in enumerate(intermediate_steps, 1):
        tool_input = getattr(action, "tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {"input": str(tool_input)}
        iterations.append(
            {
                "iteration": i,
                "tool": getattr(action, "tool", None),
                "tool_input": tool_input,
                "tool_output": str(observation),
                "scratchpad_entries_after_turn": i,
            }
        )
    return {
        "iterations": iterations,
        "total_tool_calls": len(iterations),
        "final_output": final_output,
    }


def _upload_scratchpad_run(prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Body wrapped by traceable when LangSmith tracing is on."""
    return {"user_prompt": prompt, "agent_scratchpad": payload}


def record_agent_scratchpad_to_langsmith(
    user_prompt: str,
    intermediate_steps: list,
    final_output: str | None,
) -> None:
    """
    Upload scratchpad as a LangSmith run (project from LANGCHAIN_PROJECT).
    No-op when LANGCHAIN_TRACING_V2 is not enabled.
    """
    if not tracing_enabled():
        return

    from langsmith import traceable

    scratchpad = build_agent_scratchpad_payload(intermediate_steps, final_output)
    traced_upload = traceable(name="agent_scratchpad", run_type="chain")(_upload_scratchpad_run)
    traced_upload(user_prompt, scratchpad)
