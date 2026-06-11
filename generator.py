from __future__ import annotations

import json
import asyncio
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic
from services.logger import get_logger
from shcema import AgentConfig, TaskType, StepConfig

if TYPE_CHECKING:
    from domain_pack import DomainPack

log = get_logger(__name__)

# =============================================================================
# generator.py
#
# The Generator takes a user's NL input and a loaded DomainPack and returns
# a validated AgentConfig — the universal workflow description that the
# Compiler (Phase 4a) turns into a running LangGraph graph.
#
# WHAT THIS REPLACES
#   Previously agent.py called run_agent(user_prompt) directly, which
#   hardcoded a fixed LangGraph graph for one task (marketing post).
#   The Generator sits upstream of the agent: it decides WHAT workflow
#   to run before the agent runs it.
#
# HOW IT WORKS
#   1. Assembles a system prompt from four domain-aware sources:
#        - AgentConfig JSON schema       → tells Claude the exact output structure
#        - Domain tool catalog           → only domain-relevant tools exposed
#        - GovernanceLoader.to_prompt()  → compliance rules injected
#        - FAISSRetriever.get_top_k()    → 3 approved examples as few-shot
#        - SemanticLayer.resolve_terms() → NL → YAML term hints
#   2. Makes a single Claude API call in JSON mode
#   3. Parses the response with AgentConfig.model_validate_json()
#   4. If parsing fails, retries once with the validation error fed back
#      to Claude so it can self-correct
#
# WHY CLAUDE API DIRECTLY (not via LangChain)
#   The existing marketing platform uses LangChain's init_chat_model for
#   the agent LLM. The Generator uses the Anthropic SDK directly because:
#     - The Generator needs precise JSON mode control
#     - It uses a different model (claude-sonnet) from the agent (gpt-4o)
#     - Keeping it decoupled means swapping models in domain.yaml model_hints
#       requires no code change
#
# TECHNOLOGY
#   anthropic SDK  — AsyncAnthropic for non-blocking Claude API calls
#   Pydantic v2    — AgentConfig.model_validate_json() validates the response
#   domain_pack    — provides all four context sources above
# =============================================================================

_client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env automatically


# ── Prompt assembly ────────────────────────────────────────────────────────────

def _build_system_prompt(domain: "DomainPack", nl_input: str) -> str:
    """
    Assemble the Generator system prompt from all domain context sources.
    This is the prompt that tells Claude exactly what schema to produce
    and what domain rules to respect.
    """
    # 1. AgentConfig JSON schema — Claude must match this exactly
    schema_json = json.dumps(AgentConfig.model_json_schema(), indent=2)

    # 2. Domain tool catalog — only tools declared in domain.yaml are listed
    tool_list = "\n".join(f"  - {t}" for t in domain.tools)

    # 3. Governance rules — from GovernanceLoader.to_prompt() (stub in Phase 1c,
    #    full implementation in Phase 2b)
    governance_block = domain.governance.to_prompt()

    # 4. Few-shot examples — top-3 approved workflows similar to nl_input
    #    from FAISSRetriever (stub in Phase 1c, full in Phase 3a)
    few_shots = domain.retriever.get_top_k(nl_input, k=3, task_type=domain.task_type)

    # 5. Semantic hints — NL term → YAML param mappings from SemanticLayer
    #    (stub in Phase 1c, full in Phase 2c)
    semantic_hints = domain.semantic.resolve_terms(nl_input)

    parts = [
        f"You are an agent workflow generator for the '{domain.name}' domain.",
        "",
        "Your job: given a natural language request, produce a valid AgentConfig JSON object.",
        "Return ONLY the JSON object — no markdown, no explanation, no code fences.",
        "",
        "## AgentConfig Schema",
        "Your output must exactly match this JSON schema:",
        "```json",
        schema_json,
        "```",
        "",
        "## Available Tools",
        "Only use tools from this list in step.tool fields:",
        tool_list,
    ]

    if governance_block:
        parts += [
            "",
            "## Governance Rules (enforced — do not violate)",
            governance_block,
        ]

    if semantic_hints:
        parts += [
            "",
            "## Domain Vocabulary",
            "Map these NL terms to the exact YAML values shown:",
            semantic_hints,
        ]

    if few_shots:
        parts += [
            "",
            "## Approved Workflow Examples (use as style reference)",
            few_shots,
        ]

    parts += [
        "",
        "## Rules",
        "1. task_type must be one of: "
        + ", ".join(t.value for t in TaskType),
        "2. Every step.tool must be in the Available Tools list above.",
        "3. brand_context_tool must be the FIRST step for email_generation "
        "and campaign_brief task types.",
        "4. input_from must reference an earlier step name, or be null.",
        "5. Do not include any field not in the schema.",
        "6. domain must be set to: " + domain.name,
    ]

    return "\n".join(parts)


def _build_retry_prompt(nl_input: str, bad_json: str, error: str) -> str:
    """
    Prompt for the single auto-retry. Feeds the validation error back so
    Claude can self-correct rather than making the same mistake again.
    """
    return (
        f"Your previous response failed validation with this error:\n"
        f"{error}\n\n"
        f"The invalid JSON was:\n{bad_json}\n\n"
        f"Fix the error and return the corrected JSON only. "
        f"Original request: {nl_input}"
    )


# ── Core generator function ────────────────────────────────────────────────────

async def generate_config(
    nl_input: str,
    domain:   "DomainPack",
) -> AgentConfig:
    """
    Generate a validated AgentConfig from a natural language input string.

    Parameters
    ----------
    nl_input : the user's raw prompt, e.g.
               "Every Tuesday, research B2B SaaS trends, draft a cold email
                to VP Sales personas, and route to Slack for approval."
    domain   : loaded DomainPack (from DomainPack.load())

    Returns
    -------
    AgentConfig — validated, ready to pass to the Compiler.

    Raises
    ------
    ValueError  — if Claude returns invalid JSON after one retry
    RuntimeError — if the Anthropic API call fails
    """
    system_prompt = _build_system_prompt(domain, nl_input)
    model         = domain.preferred_model(domain.task_type)

    log.debug(
        f"Generator → model: {model} | domain: {domain.name} | "
        f"task_type: {domain.task_type} | input length: {len(nl_input)}"
    )

    # ── First attempt ──────────────────────────────────────────────────────
    raw_json = await _call_claude(
        system  = system_prompt,
        user    = nl_input,
        model   = model,
    )

    config, error = _parse_config(raw_json)

    if config is not None:
        log.debug(
            f"Generator succeeded on first attempt — "
            f"{len(config.steps)} steps, task_type: {config.task_type}"
        )
        return config

    # ── One auto-retry with error fed back ────────────────────────────────
    log.warning(f"Generator first attempt failed: {error} — retrying once")

    retry_prompt = _build_retry_prompt(nl_input, raw_json, error)
    raw_json_2   = await _call_claude(
        system  = system_prompt,
        user    = retry_prompt,
        model   = model,
    )

    config2, error2 = _parse_config(raw_json_2)

    if config2 is not None:
        log.debug(
            f"Generator succeeded on retry — "
            f"{len(config2.steps)} steps, task_type: {config2.task_type}"
        )
        return config2

    # ── Both attempts failed ───────────────────────────────────────────────
    raise ValueError(
        f"Generator failed to produce a valid AgentConfig after 2 attempts.\n"
        f"Final error: {error2}\n"
        f"Final response: {raw_json_2}"
    )


# ── Sync wrapper (for CLI and FastAPI sync routes) ────────────────────────────

def generate_config_sync(nl_input: str, domain: "DomainPack") -> AgentConfig:
    """
    Synchronous wrapper around generate_config() for contexts that
    cannot use async/await (e.g. the existing CLI in main.py).
    """
    return asyncio.run(generate_config(nl_input, domain))


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _call_claude(system: str, user: str, model: str) -> str:
    """
    Make one Claude API call and return the raw response text.
    Uses the Anthropic AsyncAnthropic client (non-blocking).

    max_tokens is set to 1000 per the project spec.
    temperature is 0 — we want deterministic, schema-conformant JSON,
    not creative variation.
    """
    response = await _client.messages.create(
        model      = model,
        max_tokens = 1000,
        temperature= 0,
        system     = system,
        messages   = [{"role": "user", "content": user}],
    )
    # response.content is a list of blocks; we want the first text block
    return response.content[0].text.strip()


def _parse_config(raw: str) -> tuple[AgentConfig | None, str]:
    """
    Attempt to parse a raw string into a validated AgentConfig.

    Returns (config, "") on success.
    Returns (None, error_message) on failure.

    Strips markdown code fences defensively — Claude occasionally wraps
    JSON in ```json ... ``` even when instructed not to.
    """
    cleaned = (
        raw
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        config = AgentConfig.model_validate_json(cleaned)
        return config, ""
    except Exception as e:
        return None, str(e)