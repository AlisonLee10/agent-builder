import yaml
from pathlib import Path
from dataclasses import dataclass, field
from services.logger import get_logger

log = get_logger(__name__)

# =============================================================================
# domain_pack.py
#
# The DomainPack class is the single entry point for all domain-specific
# context. It reads domain.yaml and wires up three sub-components:
#
#   GovernanceLoader  — parses content_policy.yaml + brand_guidelines.md
#                       and exposes to_prompt() for system prompt injection.
#                       (full implementation in Phase 2b)
#
#   SemanticLayer     — loads ontology.yaml + vocabulary.json and exposes
#                       resolve_terms() for NL → YAML param mapping.
#                       (full implementation in Phase 2c)
#
#   FAISSRetriever    — wraps the existing campaign_memory FAISS index and
#                       adds task_type-filtered Top-K retrieval.
#                       (full implementation in Phase 3a)
#
# WHAT THIS REPLACES
#   Previously, domain knowledge was accessed directly from three places:
#     - services/rag.py        → loads company_data.json at import time
#     - services/ai.py         → hardcoded system prompt strings (fixed in 1b)
#     - tools/tools.py         → hardcoded tool names
#
#   After Phase 2a, all domain context flows through DomainPack.load().
#   Other files never touch domain files directly.
#
# HOW IT IS USED (Phase 2a onward)
#   In main.py / server.py:
#       from domain_pack import DomainPack
#       domain = DomainPack.load("marketing", task_type="email_generation",
#                                nl_input=user_prompt)
#       # domain is now active — services/ai.py, agent.py pick it up
#
# TECHNOLOGY
#   PyYAML  — parses domain.yaml (already installed as a LangChain dependency)
#   pathlib — resolves all paths relative to the domain folder so the pack is
#             portable regardless of where the project root is
# =============================================================================


# ── Stub classes (replaced by full implementations in Phase 2b and 2c) ────────

class GovernanceLoader:
    """
    Loads content_policy.yaml and brand_guidelines.md.
    Full implementation in Phase 2b — this stub returns empty strings
    so DomainPack.load() works end-to-end right now.
    """
    def __init__(self, governance_cfg: dict, domain_folder: Path):
        self._policy_path     = domain_folder / governance_cfg["content_policy"]
        self._guidelines_path = domain_folder / governance_cfg["brand_guidelines"]
        log.debug(f"GovernanceLoader stub initialised — policy: {self._policy_path}")

    def to_prompt(self) -> str:
        """
        Returns governance rules as a plain-text block for system prompt
        injection. Phase 2b will parse content_policy.yaml and render
        each rule as a human-readable line.
        """
        # Stub: read raw brand_guidelines.md as a best-effort fallback
        # so the system prompt gets *something* even before Phase 2b.
        try:
            return self._guidelines_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def check(self, output: str, task_type: str) -> list[dict]:
        """
        Evaluates output against content_policy.yaml rules.
        Returns list of violation dicts: [{id, description, severity}].
        Phase 2b implements the full rule evaluation loop.
        """
        return []  # stub — no violations yet


class SemanticLayer:
    """
    Loads ontology.yaml and vocabulary.json.
    Full implementation in Phase 2c — this stub returns empty string.
    """
    def __init__(self, semantic_cfg: dict, domain_folder: Path):
        self._vocab_path    = domain_folder / semantic_cfg["vocabulary"]
        self._ontology_path = domain_folder / semantic_cfg["ontology"]
        log.debug(f"SemanticLayer stub initialised — vocab: {self._vocab_path}")

    def resolve_terms(self, nl_input: str) -> str:
        """
        Maps domain vocabulary in nl_input to YAML parameter values.
        Phase 2c implements the full dict lookup against vocabulary.json.
        """
        return ""  # stub — no term resolution yet


class FAISSRetriever:
    """
    Wraps the existing campaign_memory FAISS index with task_type filtering.
    Full implementation in Phase 3a — this stub delegates directly to the
    existing get_few_shot_examples() and get_denial_lessons_for_agent().
    """
    def __init__(self, training_data_cfg: dict, domain_folder: Path):
        self._approved_path = domain_folder / training_data_cfg["approved"]
        self._rejected_path = domain_folder / training_data_cfg["rejected"]
        self._embed_model   = training_data_cfg.get(
            "embed_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        log.debug(f"FAISSRetriever stub initialised — model: {self._embed_model}")

    def get_top_k(self, nl_input: str, k: int = 3, task_type: str | None = None) -> str:
        """
        Returns k approved examples as a formatted string for few-shot
        injection. Phase 3a adds task_type-filtered FAISS index lookup.
        """
        # Stub: delegate to the existing function unchanged
        from services.campaign_memory import get_few_shot_examples
        return get_few_shot_examples(nl_input, k=k)

    def get_denial_lessons(self, nl_input: str, k: int = 2) -> str:
        """
        Returns k rejected examples and their reasons.
        Phase 3a adds task_type-filtered lookup.
        """
        from services.campaign_memory import get_denial_lessons_for_agent
        return get_denial_lessons_for_agent(nl_input, k=k)


# ── DomainPack ────────────────────────────────────────────────────────────────

@dataclass
class DomainPack:
    """
    Loaded domain context. Holds the three sub-components and exposes
    the tool catalog and model hints declared in domain.yaml.

    Do not instantiate directly — use DomainPack.load().
    """
    name:       str
    task_type:  str
    folder:     Path
    tools:      list[str]
    model_hints: dict[str, str]

    governance: GovernanceLoader = field(repr=False)
    semantic:   SemanticLayer    = field(repr=False)
    retriever:  FAISSRetriever   = field(repr=False)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        domain_name: str,
        task_type:   str  = "email_generation",
        nl_input:    str  = "",
        *,
        domains_root: str | Path | None = None,
    ) -> "DomainPack":
        """
        Load a domain pack by name and activate it across the platform.

        Parameters
        ----------
        domain_name  : folder name inside domains/  e.g. "marketing"
        task_type    : the task the agent will run  e.g. "email_generation"
                       If not provided, SemanticLayer.resolve_terms() will
                       infer it from nl_input in Phase 2c.
        nl_input     : the user's raw NL prompt — used for term resolution
                       and FAISS retrieval
        domains_root : override the default domains/ folder location.
                       Defaults to <project_root>/domains/

        Returns
        -------
        DomainPack instance with all sub-components initialised.
        Also calls services/ai.set_domain() so the Jinja2 prompt template
        is activated immediately.
        """
        # ── Resolve paths ──────────────────────────────────────────────────
        root          = Path(domains_root) if domains_root else Path("domains")
        domain_folder = root / domain_name

        if not domain_folder.exists():
            raise FileNotFoundError(
                f"Domain folder not found: {domain_folder}\n"
                f"Expected: domains/{domain_name}/domain.yaml to exist.\n"
                f"Run Phase 1a to create the domain folder structure."
            )

        domain_yaml_path = domain_folder / "domain.yaml"
        if not domain_yaml_path.exists():
            raise FileNotFoundError(
                f"domain.yaml not found at {domain_yaml_path}"
            )

        # ── Parse domain.yaml ──────────────────────────────────────────────
        # PyYAML: safe_load never executes arbitrary Python, safe for untrusted files
        with open(domain_yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        log.debug(
            f"Loading domain '{domain_name}' | task_type: '{task_type}' | "
            f"yaml version: {cfg.get('version', 'unknown')}"
        )

        # ── Instantiate sub-components ─────────────────────────────────────
        governance = GovernanceLoader(cfg["governance"],    domain_folder)
        semantic   = SemanticLayer(cfg["semantic"],         domain_folder)
        retriever  = FAISSRetriever(cfg["training_data"],   domain_folder)

        # ── Resolve task_type from NL if not explicitly provided ───────────
        # Phase 2c will implement full resolution via SemanticLayer.
        # For now: use the provided task_type or fall back to "email_generation"
        resolved_task_type = task_type or "email_generation"

        # ── Build the DomainPack instance ──────────────────────────────────
        pack = cls(
            name        = cfg["name"],
            task_type   = resolved_task_type,
            folder      = domain_folder,
            tools       = cfg.get("tools", []),
            model_hints = cfg.get("model_hints", {}),
            governance  = governance,
            semantic    = semantic,
            retriever   = retriever,
        )

        # ── Activate domain in services/ai.py ─────────────────────────────
        # This is the call that makes the Jinja2 persona.j2 template render
        # with domain context. From this point, generate_content() and
        # generate_hashtags() use the domain-aware prompts.
        from services.ai import set_domain
        set_domain(
            domain_name      = pack.name,
            task_type        = pack.task_type,
            domain_folder    = domain_folder,
            governance_rules = governance.to_prompt(),
            semantic_hints   = semantic.resolve_terms(nl_input),
        )

        log.debug(
            f"Domain '{domain_name}' active — "
            f"{len(pack.tools)} tools | "
            f"preferred model: {pack.model_hints.get('default', 'not set')}"
        )

        return pack

    # ── Convenience helpers ────────────────────────────────────────────────

    def preferred_model(self, task_type: str | None = None) -> str:
        """
        Return the preferred LLM model identifier for the given task type.
        Falls back to the domain default, then to claude-sonnet-4-20250514.
        Used by the LLM Router in Phase 5.
        """
        key = task_type or self.task_type
        return (
            self.model_hints.get(key)
            or self.model_hints.get("default")
            or "claude-sonnet-4-20250514"
        )

    def __repr__(self) -> str:
        return (
            f"DomainPack(name={self.name!r}, task_type={self.task_type!r}, "
            f"tools={self.tools})"
        )