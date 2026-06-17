from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from schema import AgentConfig
from services.logger import get_logger
from services.verify import run_verification

if TYPE_CHECKING:
    from domain_pack import DomainPack

log = get_logger(__name__)

# =============================================================================
# validator.py
#
# Two-layer validation that runs on every generated AgentConfig and its
# eventual text output:
#
#   Layer 1 — Schema validation (Pydantic)
#     AgentConfig.model_validate_json() already ran inside generator.py.
#     The Validator re-checks the assembled config against domain-specific
#     rules: are all step tools in the domain's tool catalog? Are required
#     steps present for this task type?
#
#   Layer 2 — Governance validation (GovernanceLoader)
#     GovernanceLoader.check() evaluates the generated TEXT output against
#     the rules in content_policy.yaml. Runs after the agent produces copy.
#
#   Layer 3 — LLM verification (existing services/verify.py)
#     The existing run_verification() from the marketing platform is reused
#     unchanged as a third pass — it catches semantic issues (false claims,
#     tone problems) that rule-based checks miss.
#
# WHAT THIS REPLACES
#   Previously services/verify.py was the only validation layer — a single
#   LLM call that checked content safety. The Validator adds two deterministic
#   layers before the LLM check, which:
#     - Catch schema and governance violations instantly (no API cost)
#     - Produce structured violation reports (not just pass/fail)
#     - Enable the self-learning loop: violations are written to
#       rejected/ Training Data so FAISS retrieval improves over time
#
# TECHNOLOGY
#   Pydantic v2           — AgentConfig re-validation
#   GovernanceLoader      — rule-based content_policy.yaml evaluation
#   services/verify.py    — existing LLM-based content verification (reused)
# =============================================================================


@dataclass
class ValidationResult:
    """
    Result of running all three validation layers on a config + output pair.
    Passed back to the caller so it can decide whether to retry, block, or
    write to rejected/ Training Data.
    """
    passed:     bool               # True only if no error-severity violations
    violations: list[dict]         = field(default_factory=list)
    warnings:   list[dict]         = field(default_factory=list)
    llm_verdict: str               = ""     # "approved" | "needs_revision" | "rejected"
    llm_summary: str               = ""
    revised_output: str | None     = None   # set if LLM verifier revised the text


# ── Layer 1: Schema + domain rule checks ──────────────────────────────────────

def validate_config(config: AgentConfig, domain: "DomainPack") -> ValidationResult:
    """
    Layer 1 validation — runs immediately after generate_config() returns.
    Checks the AgentConfig structure against domain-specific rules before
    any agent execution begins.

    Checks performed:
      - All step.tool values are in the domain's tool catalog
      - brand_context_tool is first for email/campaign_brief task types
      - No duplicate step names
      - schedule.cron is a valid cron expression (if present)

    Parameters
    ----------
    config : validated AgentConfig from generator.py
    domain : loaded DomainPack with tool catalog

    Returns
    -------
    ValidationResult — passed=True means safe to compile and run
    """
    violations = []
    warnings   = []

    # ── Check 1: all tools are in the domain catalog ───────────────────────
    allowed_tools = set(domain.tools)
    for step in config.steps:
        if step.tool not in allowed_tools:
            violations.append({
                "id":          "tool_not_in_catalog",
                "description": (
                    f"Step '{step.name}' uses tool '{step.tool}' which is not "
                    f"in the domain '{domain.name}' tool catalog. "
                    f"Allowed: {sorted(allowed_tools)}"
                ),
                "severity": "error",
            })

    # ── Check 2: brand_context_tool is first for content-heavy tasks ───────
    content_tasks = {"email_generation", "campaign_brief"}
    if config.task_type.value in content_tasks:
        first_tool = config.steps[0].tool if config.steps else None
        if first_tool != "brand_context_tool":
            violations.append({
                "id":          "brand_context_not_first",
                "description": (
                    f"task_type '{config.task_type.value}' requires brand_context_tool "
                    f"as the first step, but first step uses '{first_tool}'."
                ),
                "severity": "error",
            })

    # ── Check 3: no duplicate step names ──────────────────────────────────
    seen_names: set[str] = set()
    for step in config.steps:
        if step.name in seen_names:
            violations.append({
                "id":          "duplicate_step_name",
                "description": f"Duplicate step name '{step.name}'. Each step must have a unique name.",
                "severity":    "error",
            })
        seen_names.add(step.name)

    # ── Check 4: cron expression is non-empty if schedule is set ──────────
    if config.schedule and not config.schedule.cron.strip():
        violations.append({
            "id":          "empty_cron_expression",
            "description": "schedule.cron is set but empty. Provide a valid cron expression.",
            "severity":    "error",
        })

    # ── Check 5: HITL channel is 'slack' (only supported channel in MVP) ──
    for step in config.steps:
        if step.hitl and step.hitl.channel not in {"slack", "email"}:
            warnings.append({
                "id":          "unsupported_hitl_channel",
                "description": (
                    f"Step '{step.name}' HITL channel '{step.hitl.channel}' is not "
                    f"supported in MVP. Supported: slack, email."
                ),
                "severity": "warning",
            })

    errors = [v for v in violations if v["severity"] == "error"]
    passed = len(errors) == 0

    if violations or warnings:
        log.debug(
            f"Config validation: {len(errors)} error(s), "
            f"{len(warnings)} warning(s) for task_type '{config.task_type.value}'"
        )

    return ValidationResult(
        passed     = passed,
        violations = violations,
        warnings   = warnings,
    )


# ── Layer 2: Governance check on generated text output ────────────────────────

def validate_output(
    output:    str,
    config:    AgentConfig,
    domain:    "DomainPack",
) -> ValidationResult:
    """
    Layer 2 validation — runs after the agent produces text output.
    Evaluates the output string against GovernanceLoader rules from
    content_policy.yaml.

    Parameters
    ----------
    output  : raw generated text (email body, brief, research summary)
    config  : the AgentConfig that produced this output
    domain  : loaded DomainPack with GovernanceLoader

    Returns
    -------
    ValidationResult — passed=True means no error-severity governance violations
    """
    task_type  = config.task_type.value
    violations = domain.governance.check(output, task_type)

    errors   = [v for v in violations if v["severity"] == "error"]
    warnings = [v for v in violations if v["severity"] == "warning"]
    passed   = len(errors) == 0

    if violations:
        log.debug(
            f"Governance check: {len(errors)} error(s), {len(warnings)} warning(s) "
            f"for task_type '{task_type}'"
        )
        for v in errors:
            log.warning(f"  [ERROR]   [{v['id']}] {v['description'][:120]}")
        for v in warnings:
            log.debug(f"  [WARNING] [{v['id']}] {v['description'][:120]}")

    return ValidationResult(
        passed     = passed,
        violations = errors,
        warnings   = warnings,
    )


# ── Layer 3: LLM verification (existing services/verify.py reused) ────────────

def validate_output_llm(
    output:        str,
    max_revisions: int = 2,
) -> ValidationResult:
    """
    Layer 3 validation — reuses the existing run_verification() from
    services/verify.py unchanged. Catches semantic issues that rule-based
    checks miss: false statistical claims, inappropriate tone, misleading
    framing.

    Parameters
    ----------
    output        : text to verify
    max_revisions : max LLM revision attempts (passed to run_verification)

    Returns
    -------
    ValidationResult with llm_verdict, llm_summary, and revised_output set
    """
    result = run_verification(output, max_revisions=max_revisions)

    verdict        = result.get("verdict", "needs_revision")
    revised_output = result.get("content", output)
    passed         = verdict == "approved"

    log.debug(f"LLM verification verdict: {verdict} — {result.get('summary', '')}")

    return ValidationResult(
        passed         = passed,
        llm_verdict    = verdict,
        llm_summary    = result.get("summary", ""),
        revised_output = revised_output if revised_output != output else None,
    )


# ── Full pipeline: all three layers ───────────────────────────────────────────

def run_full_validation(
    config:        AgentConfig,
    output:        str,
    domain:        "DomainPack",
    skip_llm:      bool = False,
) -> ValidationResult:
    """
    Run all three validation layers in sequence.
    Short-circuits on the first error-severity failure so the caller gets
    a fast result without spending API tokens on LLM verification when
    a simple governance rule was already violated.

    Parameters
    ----------
    config   : the AgentConfig that produced the output
    output   : the generated text to validate
    domain   : loaded DomainPack
    skip_llm : if True, skip Layer 3 (useful for fast config-only checks)

    Returns
    -------
    Merged ValidationResult combining all three layers.
    passed=True only if all three layers pass.
    """
    # Layer 1: schema + domain rules
    schema_result = validate_config(config, domain)
    if not schema_result.passed:
        log.warning(
            f"Validation short-circuited at Layer 1 — "
            f"{len(schema_result.violations)} schema/domain error(s)"
        )
        return schema_result

    # Layer 2: governance rules on text output
    gov_result = validate_output(output, config, domain)
    if not gov_result.passed:
        log.warning(
            f"Validation short-circuited at Layer 2 — "
            f"{len(gov_result.violations)} governance error(s)"
        )
        # Merge warnings from Layer 1 into the result
        gov_result.warnings.extend(schema_result.warnings)
        return gov_result

    if skip_llm:
        gov_result.warnings.extend(schema_result.warnings)
        return gov_result

    # Layer 3: LLM semantic verification
    llm_result = validate_output_llm(output)

    # Merge all results into one
    return ValidationResult(
        passed         = llm_result.passed,
        violations     = gov_result.violations + llm_result.violations,
        warnings       = schema_result.warnings + gov_result.warnings,
        llm_verdict    = llm_result.llm_verdict,
        llm_summary    = llm_result.llm_summary,
        revised_output = llm_result.revised_output,
    )