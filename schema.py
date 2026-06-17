from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator

# =============================================================================
# schema.py
#
# The AgentConfig Pydantic model — the universal YAML/JSON structure that
# describes any agent workflow. This is the contract between:
#
#   Generator (generator.py, Phase 2b) — produces an AgentConfig from NL input
#   Compiler  (compiler.py,  Phase 4a) — consumes an AgentConfig to build a
#                                         LangGraph StateGraph dynamically
#
# WHAT THIS REPLACES
#   Previously, the agent had no explicit config schema. The workflow was
#   implicitly defined by the hardcoded prompt in agent.py and the fixed
#   LangGraph graph. Now every workflow is an explicit, validated, serialisable
#   data structure that can be:
#     - generated from NL input by the Generator
#     - saved as a YAML file (templates/, Git-native version control)
#     - loaded by name (agent run --template weekly_trend_post)
#     - compiled into a LangGraph graph at runtime
#     - compared between runs (diff two configs to understand behaviour change)
#
# TOOL NAMES
#   All tool names in StepConfig.tool must match a key in tools/tools.py or
#   an MCP server tool name. Valid values come from domain.yaml tools: list.
#   The Compiler (Phase 4a) resolves tool names to actual callable functions.
#
# TECHNOLOGY
#   Pydantic v2 — validation happens automatically on instantiation.
#   model_json_schema() is called by the Generator to inject the schema into
#   the Claude API system prompt so the model knows the exact output structure.
# =============================================================================


# ── Enums ─────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """
    All supported task types. Must match task_type_mappings in vocabulary.json
    and the task_type entity values in ontology.yaml.
    """
    EMAIL_GENERATION    = "email_generation"
    RESEARCH_SUMMARY    = "research_summary"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    CAMPAIGN_BRIEF      = "campaign_brief"
    APPROVAL_ROUTING    = "approval_routing"
    SCHEDULING          = "scheduling"


class OutputFormat(str, Enum):
    """How the final output should be formatted."""
    MARKDOWN  = "markdown"   # default — human-readable structured text
    JSON      = "json"       # machine-readable for downstream processing
    PLAIN     = "plain"      # plain text, no structure
    YAML      = "yaml"       # YAML output (for nested structured results)


class HITLAction(str, Enum):
    """What the Human-in-the-Loop gate does when a human responds."""
    APPROVE_OR_REJECT = "approve_or_reject"   # binary: proceed or stop
    APPROVE_OR_REVISE = "approve_or_revise"   # binary: proceed or loop back
    REVIEW_ONLY       = "review_only"         # notification only, always proceeds


# ── Sub-models ────────────────────────────────────────────────────────────────

class ScheduleConfig(BaseModel):
    """
    Cron-based schedule for recurring workflows.
    Only present when the workflow should run automatically on a schedule.
    """
    cron: str = Field(
        ...,
        description=(
            "Standard cron expression. "
            "Examples: '0 9 * * 2' (every Tuesday 9am), "
            "'0 9 * * *' (every day 9am). "
            "Resolved from NL by SemanticLayer.resolve_terms() — "
            "e.g. 'every tuesday' → '0 9 * * 2'."
        ),
        examples=["0 9 * * 2", "0 9 * * 1", "0 8 * * *"],
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for the cron schedule. e.g. 'America/New_York'.",
    )


class HITLConfig(BaseModel):
    """
    Human-in-the-Loop gate configuration.
    When present on a StepConfig, the Compiler inserts a LangGraph interrupt()
    before that step so a human can review and approve/reject.

    The existing Slack approval flow in services/slack.py is reused unchanged.
    Phase 4b wires this into the LangGraph compiler.
    """
    channel: str = Field(
        default="slack",
        description="Notification channel for the review request. 'slack' uses the existing MCP Slack tool.",
    )
    action: HITLAction = Field(
        default=HITLAction.APPROVE_OR_REJECT,
        description="What the reviewer can do: approve/reject, approve/revise, or review-only.",
    )
    timeout_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,  # max 24 hours
        description="How long to wait for human response before auto-escalating.",
    )
    on_timeout: str = Field(
        default="escalate",
        description=(
            "What to do if the reviewer does not respond within timeout_minutes. "
            "'escalate' notifies a manager. 'approve' auto-approves. 'reject' auto-rejects."
        ),
    )


class StepConfig(BaseModel):
    """
    A single step in the agent workflow.
    Each step maps to one LangGraph node in the compiled StateGraph.

    The Compiler (Phase 4a) reads step.tool to find the callable function
    in tools/tools.py or the MCP tool registry.
    """
    name: str = Field(
        ...,
        description="Short identifier for this step, used as the LangGraph node name.",
        examples=["fetch_brand_context", "fetch_news", "generate_email", "slack_approval"],
    )
    tool: str = Field(
        ...,
        description=(
            "The tool to call for this step. Must match a tool name in the "
            "domain's tool catalog (domain.yaml tools:) or an MCP tool name. "
            "Valid values for the marketing domain: brand_context_tool, "
            "generate_content_tool, generate_hashtags_tool, news_tool, "
            "news_sources_tool, reddit_tool, check_brand_compliance."
        ),
        examples=["brand_context_tool", "generate_content_tool", "news_tool"],
    )
    input_from: str | None = Field(
        default=None,
        description=(
            "Name of a previous step whose output is passed as input to this step. "
            "None means the step receives the original user NL input."
        ),
    )
    hitl: HITLConfig | None = Field(
        default=None,
        description=(
            "If set, a Human-in-the-Loop review gate is inserted before this step. "
            "The existing Slack HITL flow from the marketing platform is reused."
        ),
    )
    condition: str | None = Field(
        default=None,
        description=(
            "Optional Python expression evaluated at runtime to decide whether "
            "to execute this step. e.g. 'task_type == \"email_generation\"'. "
            "If False, the step is skipped and the workflow proceeds to the next step."
        ),
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Whether to auto-retry this step once if it raises an exception.",
    )


# ── Root model ────────────────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    """
    Universal agent workflow configuration.

    Generated by the Generator (Phase 2b) from a user's NL input.
    Validated by the Validator (Phase 2c).
    Compiled into a LangGraph StateGraph by the Compiler (Phase 4a).
    Serialisable to YAML for template storage and Git version control.

    Example minimal config (email generation):
        task_type: email_generation
        domain: marketing
        steps:
          - name: brand_context
            tool: brand_context_tool
          - name: fetch_news
            tool: news_tool
          - name: generate_email
            tool: generate_content_tool
            input_from: fetch_news
            hitl:
              channel: slack
              action: approve_or_reject
          - name: hashtags
            tool: generate_hashtags_tool
    """

    # ── Required fields ────────────────────────────────────────────────────
    task_type: TaskType = Field(
        ...,
        description=(
            "What kind of workflow this config describes. "
            "Resolved from NL by SemanticLayer — e.g. 'cold email' → email_generation."
        ),
    )
    domain: str = Field(
        ...,
        description="The domain pack to activate for this run. e.g. 'marketing'.",
        examples=["marketing", "hr", "legal"],
    )
    steps: list[StepConfig] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of steps. The Compiler executes them in order, "
            "passing outputs between steps via input_from references."
        ),
    )

    # ── Optional fields ────────────────────────────────────────────────────
    name: str | None = Field(
        default=None,
        description=(
            "Human-readable name for this workflow config. "
            "Used as the template file name if saved. "
            "e.g. 'weekly_trend_post', 'competitor_analysis'."
        ),
    )
    description: str | None = Field(
        default=None,
        description="One-sentence description of what this workflow does.",
    )
    schedule: ScheduleConfig | None = Field(
        default=None,
        description=(
            "If set, this workflow runs on a cron schedule. "
            "Compiled into a BullMQ cron job in Phase 5b."
        ),
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MARKDOWN,
        description="How to format the final output.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of full-workflow retries on validation failure.",
    )
    gdpr_mode: bool = Field(
        default=False,
        description=(
            "When True, GovernanceLoader enforces GDPR Article 13 rules: "
            "lawful basis statement required in email output."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for tracing, tagging, and debugging.",
    )

    # ── Validators ─────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def check_input_from_references(self) -> "AgentConfig":
        """
        Ensure every input_from reference points to a step that exists
        and appears earlier in the steps list.
        Catches wiring mistakes before the Compiler tries to build the graph.
        """
        step_names = []
        for step in self.steps:
            if step.input_from and step.input_from not in step_names:
                raise ValueError(
                    f"Step '{step.name}' has input_from='{step.input_from}' "
                    f"but no earlier step has that name. "
                    f"Steps defined so far: {step_names}"
                )
            step_names.append(step.name)
        return self

    @model_validator(mode="after")
    def check_brand_context_first(self) -> "AgentConfig":
        """
        Warn (not error) if brand_context_tool is not the first step.
        The domain guidelines require it to always run first, but some
        task types (e.g. scheduling) may not need it at all.
        """
        if not self.steps:
            return self
        first_tool = self.steps[0].tool
        if (
            first_tool != "brand_context_tool"
            and self.task_type
            in (TaskType.EMAIL_GENERATION, TaskType.CAMPAIGN_BRIEF)
        ):
            import warnings
            warnings.warn(
                f"AgentConfig for task_type '{self.task_type}' does not start with "
                f"brand_context_tool (first tool is '{first_tool}'). "
                f"Brand guidelines will not be available to subsequent steps.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ── Serialisation helpers ───────────────────────────────────────────────

    def to_yaml(self) -> str:
        """
        Serialise the config to a YAML string for template storage.
        Saved to domains/{domain}/templates/{name}.yaml in Phase 4c.
        """
        import yaml
        # model_dump excludes None values for cleaner YAML output
        data = self.model_dump(exclude_none=True, mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AgentConfig":
        """
        Load and validate an AgentConfig from a YAML string.
        Used by the CLI's --template flag (Phase 5a).
        """
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "AgentConfig":
        """Load and validate an AgentConfig from a YAML file path."""
        from pathlib import Path
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))