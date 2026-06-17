from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING, cast

from dotenv import load_dotenv
from langchain.chat_models           import init_chat_model
from langchain_classic.agents        import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts          import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph                 import StateGraph, START, END
from langgraph.graph.state           import CompiledStateGraph
from langgraph.types                 import interrupt

from schema          import AgentConfig, TaskType, StepConfig
from services.logger import get_logger, get_run_id
from services.progress import show_progress

if TYPE_CHECKING:
    from domain_pack import DomainPack

load_dotenv()
log = get_logger(__name__)

# =============================================================================
# compiler.py
#
# YAMLToLangGraph: reads a validated AgentConfig and dynamically builds a
# LangGraph StateGraph — one node per step, edges wired from the config.
#
# WHAT THIS REPLACES
#   agent.py currently has a hardcoded AgentExecutor with a fixed prompt and
#   a fixed set of tools for one task. The Compiler replaces that hardcoded
#   graph with a dynamic one built from the AgentConfig produced by
#   generator.py. agent.py is kept as a fallback for the original marketing
#   post flow; the Compiler is used by the new universal entry points in
#   main.py and server.py (Phase 5).
#
# ARCHITECTURE
#   Each StepConfig in AgentConfig.steps becomes one LangGraph node.
#   The node calls the tool declared in step.tool using the existing
#   tool functions from tools/tools.py (reused unchanged).
#   If step.hitl is set, an interrupt() node is inserted BEFORE that step
#   so a human reviewer can approve or reject via Slack (Phase 4b).
#
# TOOL RESOLUTION
#   step.tool names are resolved to callables via _TOOL_REGISTRY, which
#   maps the same string names used in domain.yaml tools: to the actual
#   @tool-decorated functions already in tools/tools.py.
#   No new tool functions are written here — 100% reuse.
#
# TECHNOLOGY
#   LangGraph StateGraph   — dynamic graph construction, same import already
#                            in services/verify.py so it is installed
#   langchain AgentExecutor — reused from agent.py for tool-calling nodes
#   tools/tools.py          — all tool functions reused unchanged
#   tools/mcp_client.py     — MCP tool loading reused unchanged
#   tools/tool_selector.py  — select_tools() reused, now domain-aware
# =============================================================================


# ── Tool registry ─────────────────────────────────────────────────────────────
# Maps tool name strings (from domain.yaml and AgentConfig step.tool) to the
# actual callable tool objects from tools/tools.py.
# MCP tools are loaded dynamically at compile time via load_mcp_tools_for_agent().

def _build_tool_registry() -> dict[str, Any]:
    from tools.tools import (
        brand_context_tool,
        news_tool,
        news_sources_tool,
        reddit_tool,
        generate_content_tool,
        generate_hashtags_tool,
    )
    return {
        "brand_context_tool":     brand_context_tool,
        "news_tool":              news_tool,
        "news_sources_tool":      news_sources_tool,
        "reddit_tool":            reddit_tool,
        "generate_content_tool":  generate_content_tool,
        "generate_hashtags_tool": generate_hashtags_tool,
    }


# ── Shared state schema ───────────────────────────────────────────────────────
# Passed between nodes as LangGraph state. Each node reads from and writes
# to this dict. Using TypedDict keeps it inspectable by LangSmith.

from typing import TypedDict

class WorkflowState(TypedDict):
    nl_input:       str             # original user NL prompt
    task_type:      str             # from AgentConfig.task_type
    domain:         str             # from AgentConfig.domain
    step_outputs:   dict[str, str]  # {step_name: output_text}
    final_output:   str             # assembled final output
    hitl_approved:  bool            # True after human approves
    hitl_rejected:  bool            # True after human rejects
    rejection_reason: str           # set by HITL rejection handler


# ── Compiler ──────────────────────────────────────────────────────────────────

class YAMLToLangGraph:
    """
    Compiles an AgentConfig into a runnable LangGraph StateGraph.

    Usage:
        compiler = YAMLToLangGraph(config, domain)
        result   = await compiler.run(nl_input)

    Or via the module-level helper:
        result   = await compile_and_run(config, domain, nl_input)
    """

    def __init__(self, config: AgentConfig, domain: "DomainPack"):
        self.config  = config
        self.domain  = domain
        self._graph  = None   # built lazily on first run()

    # ── Public ────────────────────────────────────────────────────────────

    async def run(self, nl_input: str, *, debug: bool = False) -> dict:
        """
        Compile the AgentConfig into a StateGraph and execute it.

        Returns the same dict shape as agent.run_agent() so existing
        server.py and main.py code needs minimal changes:
            {content, hashtags, sources, full_post, articles, step_outputs}
        """
        if self._graph is None:
            self._graph = await self._build_graph(debug=debug)

        initial_state: WorkflowState = {
            "nl_input":        nl_input,
            "task_type":       self.config.task_type.value,
            "domain":          self.config.domain,
            "step_outputs":    {},
            "final_output":    "",
            "hitl_approved":   False,
            "hitl_rejected":   False,
            "rejection_reason": "",
        }

        log.debug(
            f"Compiler running workflow — task_type: {self.config.task_type.value} | "
            f"steps: {[s.name for s in self.config.steps]}"
        )

        with show_progress("      Running workflow"):
            final_state = await self._graph.ainvoke(
                initial_state,
                config={
                    "metadata": {"run_id": get_run_id()},
                    "tags":     [f"agent-builder-{self.config.domain}"],
                },
            )

        return self._parse_final_state(cast(WorkflowState, final_state))

    # ── Graph construction ────────────────────────────────────────────────

    async def _build_graph(self, *, debug: bool = False) -> CompiledStateGraph:
        """
        Build the LangGraph StateGraph from AgentConfig.steps.

        Each step becomes either:
          - A tool node (calls the tool, writes output to step_outputs)
          - A HITL gate node (interrupts for human review) if step.hitl is set
        """
        # Load MCP tools once — shared across all tool nodes
        from tools.mcp_client import load_mcp_tools_for_agent, merge_agent_tools
        from tools.tool_selector import select_tools

        native_tools = select_tools(
            prompt       = "",
            domain_tools = self.domain.tools,
        )
        mcp_tools  = await load_mcp_tools_for_agent()
        all_tools  = merge_agent_tools(native_tools, mcp_tools)
        tool_map   = {getattr(t, "name", str(t)): t for t in all_tools}

        # Also include registry tools not returned by select_tools
        registry   = _build_tool_registry()
        for name, fn in registry.items():
            if name not in tool_map:
                tool_map[name] = fn

        log.debug(
            f"Compiler tool map: {list(tool_map.keys())}"
        )

        # Build the agent executor used by tool nodes
        # Reuses the same prompt/LLM pattern from agent.py
        agent_executor = self._build_agent_executor(
            list(tool_map.values()), debug=debug
        )

        # Build the StateGraph
        graph = StateGraph(WorkflowState)

        # Add one node per step (plus optional HITL gate node before it)
        for step in self.config.steps:
            if step.hitl is not None:
                # HITL gate node — inserted BEFORE the step node
                gate_name = f"{step.name}_hitl_gate"
                graph.add_node(
                    gate_name,
                    self._make_hitl_node(step),
                )

            # Tool node — calls the tool declared in step.tool
            graph.add_node(
                step.name,
                self._make_tool_node(step, agent_executor, tool_map),
            )

        # Wire edges: START → first node → ... → last node → END
        self._wire_edges(graph)

        log.debug(
            f"Graph compiled — {len(self.config.steps)} step node(s) | "
            f"HITL gates: {sum(1 for s in self.config.steps if s.hitl)}"
        )

        return graph.compile()

    def _wire_edges(self, graph: StateGraph) -> None:
        """
        Add directed edges between nodes in execution order.
        Accounts for HITL gate nodes inserted before steps that have hitl set.

        Execution order for a step with HITL:
          [previous node] → [step_hitl_gate] → [step] → [next node]

        Execution order for a step without HITL:
          [previous node] → [step] → [next node]
        """
        node_sequence: list[str] = []

        for step in self.config.steps:
            if step.hitl is not None:
                node_sequence.append(f"{step.name}_hitl_gate")
            node_sequence.append(step.name)

        # START → first node
        graph.add_edge(START, node_sequence[0])

        # Each node → next node
        for i in range(len(node_sequence) - 1):
            graph.add_edge(node_sequence[i], node_sequence[i + 1])

        # Last node → END
        graph.add_edge(node_sequence[-1], END)

    # ── Node factories ────────────────────────────────────────────────────

    def _make_tool_node(
        self,
        step:           StepConfig,
        agent_executor: AgentExecutor,
        tool_map:       dict[str, Any],
    ):
        """
        Returns a LangGraph node function for a single StepConfig.

        The node builds an input string from:
          - The original nl_input (if step.input_from is None)
          - The output of the named previous step (if step.input_from is set)
        Then invokes the AgentExecutor with the single tool for this step.
        """
        step_name  = step.name
        tool_name  = step.tool
        input_from = step.input_from

        async def tool_node(state: WorkflowState) -> dict:
            # Skip if HITL rejected
            if state.get("hitl_rejected"):
                log.debug(f"Step '{step_name}' skipped — HITL rejected")
                return {}

            # Build input for this step
            if input_from and input_from in state["step_outputs"]:
                node_input = state["step_outputs"][input_from]
            else:
                node_input = state["nl_input"]

            # Evaluate condition if present
            if step.condition:
                try:
                    should_run = eval(
                        step.condition,
                        {"task_type": state["task_type"], "domain": state["domain"]},
                    )
                    if not should_run:
                        log.debug(
                            f"Step '{step_name}' skipped — condition "
                            f"'{step.condition}' evaluated False"
                        )
                        return {}
                except Exception as e:
                    log.warning(
                        f"Step '{step_name}' condition eval failed: {e} — running anyway"
                    )

            # Call the tool directly if it is in the registry,
            # otherwise invoke the full AgentExecutor for this step
            log.debug(f"Executing step '{step_name}' — tool: '{tool_name}'")

            if tool_name in tool_map:
                # Direct tool call — faster, no LLM overhead for simple tools
                tool_fn = tool_map[tool_name]
                try:
                    output = tool_fn.invoke(node_input)
                except Exception as e:
                    if step.retry_on_failure:
                        log.warning(
                            f"Step '{step_name}' failed ({e}) — retrying once"
                        )
                        try:
                            output = tool_fn.invoke(node_input)
                        except Exception as e2:
                            log.error(f"Step '{step_name}' retry also failed: {e2}")
                            output = f"[Step '{step_name}' failed: {e2}]"
                    else:
                        log.error(f"Step '{step_name}' failed: {e}")
                        output = f"[Step '{step_name}' failed: {e}]"
            else:
                # Unknown tool — fall back to AgentExecutor with full tool set
                log.warning(
                    f"Tool '{tool_name}' not in tool_map — "
                    f"falling back to AgentExecutor"
                )
                result = await agent_executor.ainvoke(
                    {"input": f"Use {tool_name} to: {node_input}"},
                    config={"metadata": {"run_id": get_run_id()}},
                )
                output = result.get("output", "")

            new_step_outputs = dict(state["step_outputs"])
            new_step_outputs[step_name] = str(output)

            # If this is the last content-generating step, set final_output
            is_last = step_name == self.config.steps[-1].name
            return {
                "step_outputs": new_step_outputs,
                "final_output": str(output) if is_last else state["final_output"],
            }

        tool_node.__name__ = step_name
        return tool_node

    def _make_hitl_node(self, step: StepConfig):
        """
        Returns a LangGraph HITL gate node for a step with step.hitl set.

        Interrupts the graph with interrupt() — LangGraph pauses execution
        here until the graph is resumed externally (by server.py's
        /api/approve or /api/deny endpoints — reused unchanged from
        the existing marketing platform).

        On resume, the state will have hitl_approved=True or
        hitl_rejected=True set by the resume handler (Phase 4b).
        """
        step_name = step.name
        hitl_cfg  = step.hitl

        def hitl_gate_node(state: WorkflowState) -> dict:
            # Notify the reviewer via Slack (reusing existing services/slack.py)
            last_output = ""
            if state["step_outputs"]:
                last_key    = list(state["step_outputs"].keys())[-1]
                last_output = state["step_outputs"].get(last_key, "")

            _notify_reviewer(
                content    = last_output or state["nl_input"],
                step_name  = step_name,
                channel    = hitl_cfg.channel if hitl_cfg else "slack",
            )

            # LangGraph interrupt() — pauses graph execution here.
            # server.py /api/approve resumes with hitl_approved=True.
            # server.py /api/deny   resumes with hitl_rejected=True.
            interrupt({
                "step":    step_name,
                "message": f"Awaiting human approval for step '{step_name}'",
            })

            # Code below runs only after the graph is resumed
            return {}

        hitl_gate_node.__name__ = f"{step_name}_hitl_gate"
        return hitl_gate_node

    # ── Agent executor factory ────────────────────────────────────────────

    @staticmethod
    def _build_agent_executor(
        tools: list,
        *,
        debug: bool = False,
    ) -> AgentExecutor:
        """
        Build the AgentExecutor used as a fallback when a tool is not found
        in the direct tool_map. Reuses the same LLM and prompt pattern
        from agent.py — create_openai_tools_agent + AgentExecutor.
        """
        from agent import _build_prompt  # reuse the domain-aware prompt builder

        agent_llm = init_chat_model("gpt-4o", temperature=0)
        prompt    = _build_prompt()
        agent     = create_openai_tools_agent(agent_llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=debug,
            return_intermediate_steps=True,
        )

    # ── Output parsing ────────────────────────────────────────────────────

    def _parse_final_state(self, state: WorkflowState) -> dict:
        """
        Assemble the final output dict from the completed workflow state.
        Returns the same shape as agent.run_agent() for drop-in compatibility.
        """
        from agent import parse_agent_output
        from services.post_content import build_publishable_post
        from services.storage import normalize_research_for_save
        from services.news import get_last_fetched_articles
        from services.trends import get_last_fetched_trends

        raw_output = state.get("final_output", "")

        # Try structured parsing first (expects [CONTENT] / [HASHTAGS] / [SOURCES])
        parsed = parse_agent_output(raw_output)

        # If no structured output, put everything in content
        if not parsed.get("content"):
            parsed["content"] = raw_output

        # Collect research articles from services (same as agent.py)
        articles = get_last_fetched_articles() + get_last_fetched_trends()
        sources_list, articles = normalize_research_for_save(
            parsed.get("sources", ""),
            articles,
        )

        parsed["sources"]      = sources_list
        parsed["articles"]     = articles
        parsed["full_post"]    = build_publishable_post(
            parsed.get("content", ""),
            parsed.get("hashtags", ""),
        )
        parsed["step_outputs"] = state.get("step_outputs", {})
        parsed["hitl_rejected"] = state.get("hitl_rejected", False)

        return parsed


# ── Reviewer notification (reuses services/slack.py) ─────────────────────────

def _notify_reviewer(content: str, step_name: str, channel: str) -> None:
    """
    Send the generated content to Slack for human review.
    Reuses services/slack.py post_to_slack() unchanged.
    """
    from services.slack import post_to_slack
    from services.post_content import build_publishable_post

    message = (
        f"*[Agent Builder — Review Required]*\n"
        f"Step: `{step_name}`\n\n"
        f"{content[:1500]}"
        f"{'...' if len(content) > 1500 else ''}\n\n"
        f"Reply `/approve` or `/deny <reason>` to continue."
    )
    posted = post_to_slack(message, channel=channel if channel != "slack" else None)
    if not posted:
        log.warning(
            f"HITL notification failed for step '{step_name}' — "
            f"graph will still interrupt and wait for resume"
        )


# ── Module-level convenience function ─────────────────────────────────────────

async def compile_and_run(
    config:   AgentConfig,
    domain:   "DomainPack",
    nl_input: str,
    *,
    debug:    bool = False,
) -> dict:
    """
    One-call helper: compile config into a graph and run it.
    Used by server.py and main.py (Phase 5).

        result = await compile_and_run(config, domain, user_prompt)
    """
    compiler = YAMLToLangGraph(config, domain)
    return await compiler.run(nl_input, debug=debug)


def compile_and_run_sync(
    config:   AgentConfig,
    domain:   "DomainPack",
    nl_input: str,
    *,
    debug:    bool = False,
) -> dict:
    """Sync wrapper for CLI usage (Phase 5a)."""
    return asyncio.run(compile_and_run(config, domain, nl_input, debug=debug))