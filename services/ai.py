import os
from pathlib import Path

import jinja2
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from services.llm    import llm
from services.logger import get_logger

load_dotenv()

log = get_logger(__name__)

# =============================================================================
# WHAT CHANGED IN THIS FILE (Phase 1b)
#
# BEFORE:
#   Two hardcoded system prompt strings lived directly inside generate_content():
#     "You are a social media copywriter. Write an engaging marketing post..."
#     "You are a social media copywriter. Write an engaging marketing post..."
#   generate_hashtags() also had a hardcoded system prompt string.
#
# AFTER:
#   All three prompts are now loaded from Jinja2 templates inside the active
#   domain's templates/ folder:
#     domains/{domain}/templates/persona.j2     → generate_content() system prompt
#     domains/{domain}/templates/hashtags.j2    → generate_hashtags() system prompt
#
#   A domain-aware prompt loader (_load_prompt) is injected via set_domain().
#   set_domain() is called by DomainPack.load() in Phase 2a.
#
# WHAT DID NOT CHANGE:
#   - generate_content() signature and return type — identical
#   - generate_hashtags() signature and return type — identical
#   - set_campaign_memory(), set_few_shot_examples(), clear_few_shot_examples()
#     — all identical, called the same way from agent.py
#   - _memory_context_block() — identical
#   - The llm.invoke() call structure — identical
#
# WHY JINJA2
#   Jinja2 is the standard Python templating library (already installed as a
#   FastAPI/LangChain dependency). It supports variables, conditionals, and
#   inheritance — allowing persona.j2 to adapt its step instructions based on
#   task_type without branching logic in Python.
# =============================================================================

# ── Domain context (injected by DomainPack.load() in Phase 2a) ───────────────
# Default values make the file work exactly as before until a domain is loaded.
_domain_name:     str            = "marketing"
_task_type:       str            = "email_generation"
_jinja_env:       jinja2.Environment | None = None
_governance_rules: str           = ""
_semantic_hints:  str            = ""

def set_domain(
    domain_name:      str,
    task_type:        str,
    domain_folder:    str | Path,
    governance_rules: str = "",
    semantic_hints:   str = "",
) -> None:
    """
    Called by DomainPack.load() (Phase 2a) to activate a domain.

    Parameters
    ----------
    domain_name      : e.g. "marketing"
    task_type        : e.g. "email_generation" — determines which branch
                       persona.j2 renders
    domain_folder    : absolute or relative path to domains/{domain}/
    governance_rules : plain-text block from GovernanceLoader.to_prompt()
                       (Phase 2b) — injected into persona.j2
    semantic_hints   : resolved term string from SemanticLayer.resolve_terms()
                       (Phase 2c) — injected into persona.j2
    """
    global _domain_name, _task_type, _jinja_env
    global _governance_rules, _semantic_hints

    template_dir = Path(domain_folder) / "templates"
    if not template_dir.exists():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}\n"
            f"Expected domains/{domain_name}/templates/ to exist."
        )

    _domain_name      = domain_name
    _task_type        = task_type
    _jinja_env        = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        # undefined=jinja2.StrictUndefined raises an error for missing variables
        # instead of silently rendering empty string — catches bugs early
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    _governance_rules = governance_rules
    _semantic_hints   = semantic_hints

    log.debug(
        f"Domain activated: {domain_name} | task_type: {task_type} | "
        f"templates: {template_dir}"
    )


def _render_template(template_name: str, **extra_vars) -> str:
    """
    Render a Jinja2 template from the active domain's templates/ folder.

    Falls back to a hardcoded default string if no domain has been loaded yet
    (i.e. set_domain() has not been called). This keeps the file backward-
    compatible during Phase 1 before Phase 2a is implemented.
    """
    if _jinja_env is None:
        # No domain loaded yet — return the original hardcoded prompts so
        # existing behaviour is preserved during the transition.
        log.debug(
            f"No domain loaded — using fallback prompt for {template_name}. "
            f"Call set_domain() to activate domain-aware prompts."
        )
        return _fallback_prompt(template_name)

    try:
        template = _jinja_env.get_template(template_name)
        return template.render(
            domain_name      = _domain_name,
            task_type        = _task_type,
            governance_rules = _governance_rules,
            semantic_hints   = _semantic_hints,
            few_shot_examples= _few_shot_examples,
            denial_lessons   = _denial_lessons,
            **extra_vars,
        )
    except jinja2.TemplateNotFound:
        log.warning(
            f"Template '{template_name}' not found in "
            f"domains/{_domain_name}/templates/ — using fallback."
        )
        return _fallback_prompt(template_name)


def _fallback_prompt(template_name: str) -> str:
    """
    Exact copy of the original hardcoded prompts from this file.
    Used when no domain is active so existing behaviour is unchanged.
    """
    if template_name == "persona.j2":
        # Original string from generate_content() — unchanged
        return (
            "You are a social media copywriter. "
            "Write an engaging marketing post under 200 words. "
            "If approved style examples are provided, match their tone and quality. "
            "If rejection lessons are provided, avoid the mistakes users cited. "
            "If news or trend context is provided, reference real facts naturally. "
            "Do not invent statistics. No hashtags."
        )
    if template_name == "hashtags.j2":
        # Original string from generate_hashtags() — unchanged
        return (
            "Generate relevant hashtags for a social media post. "
            "Consider industry, niche, brand type, and region if mentioned. "
            "Return ONLY hashtags, one per line, minimum 3, maximum 8. "
            "Each must start with #."
        )
    return ""


# ── Campaign memory (unchanged from original) ─────────────────────────────────
_few_shot_examples: str = ""
_denial_lessons:    str = ""


def set_campaign_memory(
    approved_examples: str = "",
    denial_lessons:    str = "",
) -> None:
    global _few_shot_examples, _denial_lessons
    _few_shot_examples = approved_examples or ""
    _denial_lessons    = denial_lessons    or ""


def set_few_shot_examples(examples: str) -> None:
    set_campaign_memory(approved_examples=examples)


def clear_few_shot_examples() -> None:
    set_campaign_memory("", "")


def _memory_context_block() -> str:
    parts = [p for p in (_few_shot_examples, _denial_lessons) if p]
    return "\n\n".join(parts)


# ── Content generation (signature unchanged) ──────────────────────────────────
def generate_content(
    user_prompt:    str,
    news_context:   str = "",
    trends_context: str = "",
) -> str:
    """
    Generate marketing copy for the given prompt.

    Signature and return type are identical to the original.
    The only internal change: the system prompt string is now rendered from
    domains/{domain}/templates/persona.j2 instead of being hardcoded here.
    """
    log.debug(f"generate_content — prompt length {len(user_prompt)}")

    context_blocks = []
    if news_context:
        context_blocks.append(f"Recent news:\n{news_context}")
    if trends_context:
        context_blocks.append(f"Current trends:\n{trends_context}")

    # ── CHANGED: system prompt now comes from persona.j2 ──────────────────
    # Previously there were two almost-identical hardcoded strings here,
    # one for when context_blocks existed and one for when they didn't.
    # persona.j2 handles both cases via its own {% if %} logic.
    system = _render_template("persona.j2")
    # ── END CHANGE ─────────────────────────────────────────────────────────

    user_message = f"Topic: {user_prompt}"
    if context_blocks:
        user_message += "\n\n" + "\n\n".join(context_blocks)

    memory = _memory_context_block()
    if memory:
        user_message = f"{user_message}\n\n{memory}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_message),
    ])
    return response.content.strip()


# ── Hashtag generation (signature unchanged) ──────────────────────────────────
def generate_hashtags(user_prompt: str) -> list[str]:
    """
    Generate hashtags for the given topic.

    Signature and return type are identical to the original.
    The only internal change: the system prompt string is now rendered from
    domains/{domain}/templates/hashtags.j2 instead of being hardcoded here.
    """
    log.debug(f"generate_hashtags — topic length {len(user_prompt)}")

    # ── CHANGED: system prompt now comes from hashtags.j2 ─────────────────
    system = _render_template("hashtags.j2")
    # ── END CHANGE ─────────────────────────────────────────────────────────

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Generate hashtags for: {user_prompt}"),
    ])
    raw = str(response.content).strip()
    return [line.strip() for line in raw.splitlines() if line.strip().startswith("#")]
