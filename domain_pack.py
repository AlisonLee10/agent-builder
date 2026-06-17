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
    Parses content_policy.yaml and brand_guidelines.md.
    Exposes to_prompt() for system prompt injection and check() for
    runtime output validation.

    TECHNOLOGY
      PyYAML  — parses content_policy.yaml into rule dicts
      pathlib — resolves file paths relative to the domain folder
      re      — used by _check_approved_claims to find numeric patterns
    """

    def __init__(self, governance_cfg: dict, domain_folder: Path):
        self._policy_path     = domain_folder / governance_cfg["content_policy"]
        self._guidelines_path = domain_folder / governance_cfg["brand_guidelines"]
        self._rules: list[dict] = []
        self._guidelines_text: str = ""
        self._load()

    def _load(self) -> None:
        """Parse both files on initialisation so they are ready at runtime."""
        # brand_guidelines.md — read as plain text for to_prompt()
        try:
            self._guidelines_text = self._guidelines_path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning(f"GovernanceLoader: could not read brand_guidelines: {e}")
            self._guidelines_text = ""

        # content_policy.yaml — parsed into rule dicts for check()
        try:
            with open(self._policy_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._rules = data.get("rules", []) if isinstance(data, dict) else []
            log.debug(
                f"GovernanceLoader loaded {len(self._rules)} rules "
                f"from {self._policy_path.name}"
            )
        except (OSError, yaml.YAMLError) as e:
            log.warning(f"GovernanceLoader: could not parse content_policy.yaml: {e}")
            self._rules = []

    def to_prompt(self) -> str:
        """
        Returns a concise governance summary for injection into the
        Generator and agent system prompts.

        Renders the key rules as a plain-text list rather than the full
        brand_guidelines.md — keeping the prompt focused and token-efficient.
        """
        if not self._rules:
            # Fall back to raw guidelines text if no rules were parsed
            return self._guidelines_text

        error_rules   = [r for r in self._rules if r.get("severity") == "error"]
        warning_rules = [r for r in self._rules if r.get("severity") == "warning"]

        lines = []

        if error_rules:
            lines.append("HARD RULES (violations block output):")
            for rule in error_rules:
                desc = rule.get("description", "").strip().replace("\n", " ")
                lines.append(f"  [{rule['id']}] {desc}")

        if warning_rules:
            lines.append("SOFT RULES (violations logged as warnings):")
            for rule in warning_rules:
                desc = rule.get("description", "").strip().replace("\n", " ")
                lines.append(f"  [{rule['id']}] {desc}")

        return "\n".join(lines)

    def check(self, output: str, task_type: str) -> list[dict]:
        """
        Evaluate output text against all rules that apply to task_type.

        Parameters
        ----------
        output    : the generated text to check (email body, brief, etc.)
        task_type : e.g. "email_generation" — used to filter applicable rules

        Returns
        -------
        List of violation dicts, each with keys:
            id          — rule id from content_policy.yaml
            description — human-readable explanation
            severity    — "error" or "warning"
        Empty list means the output is compliant.
        """
        violations = []
        output_lower = output.lower()

        for rule in self._rules:
            # Filter by applies_to — ["*"] means all task types
            applies_to = rule.get("applies_to", ["*"])
            if "*" not in applies_to and task_type not in applies_to:
                continue

            rule_type = rule.get("type", "")
            violation = None

            if rule_type == "forbidden_word":
                violation = self._check_forbidden_word(rule, output_lower)

            elif rule_type == "forbidden_phrase_start":
                violation = self._check_forbidden_phrase_start(rule, output_lower)

            elif rule_type == "approved_claims_only":
                violation = self._check_approved_claims(rule, output_lower)

            elif rule_type == "required_present":
                violation = self._check_required_present(rule, output)

            elif rule_type == "required_sections":
                violation = self._check_required_sections(rule, output_lower)

            # Note: max_length, max_count, required_field, conditional_required_field
            # require structured output fields (not plain text) — these are evaluated
            # by validator.py against the AgentConfig fields, not the raw output string.

            if violation:
                violations.append({
                    "id":          rule["id"],
                    "description": rule.get("description", "").strip(),
                    "severity":    rule.get("severity", "error"),
                })

        return violations

    # ── Rule evaluators ────────────────────────────────────────────────────

    def _check_forbidden_word(self, rule: dict, output_lower: str) -> bool:
        """Return True (violation) if any forbidden word appears in output."""
        for word in rule.get("value", []):
            if word.lower() in output_lower:
                log.debug(f"Governance violation [{rule['id']}]: forbidden word '{word}'")
                return True
        return False

    def _check_forbidden_phrase_start(self, rule: dict, output_lower: str) -> bool:
        """Return True if output starts with any forbidden phrase."""
        stripped = output_lower.lstrip()
        for phrase in rule.get("value", []):
            if stripped.startswith(phrase.lower()):
                log.debug(f"Governance violation [{rule['id']}]: forbidden opening '{phrase}'")
                return True
        return False

    def _check_approved_claims(self, rule: dict, output_lower: str) -> bool:
        """
        Return True if output contains a numeric claim not in the
        approved_claims list. Matches patterns like '30%', '10,000+', '2 hours'.
        """
        import re
        approved = [c.lower() for c in rule.get("approved_claims", [])]
        # Find all numeric phrases: digits optionally followed by %, +, 'hours', etc.
        numeric_pattern = re.compile(
            r'\d[\d,]*\s*(?:%|\+|hours?|minutes?|days?|times?|x\b|k\b)?'
        )
        matches = numeric_pattern.findall(output_lower)
        for match in matches:
            match_clean = match.strip()
            # Check if this numeric appears in any approved claim
            if not any(match_clean in claim for claim in approved):
                log.debug(
                    f"Governance violation [{rule['id']}]: "
                    f"unapproved numeric claim '{match_clean}'"
                )
                return True
        return False

    def _check_required_present(self, rule: dict, output: str) -> bool:
        """Return True if none of the required tokens appear in output."""
        for token in rule.get("value", []):
            if token in output:
                return False  # at least one token found — compliant
        log.debug(f"Governance violation [{rule['id']}]: no required token found")
        return True

    def _check_required_sections(self, rule: dict, output_lower: str) -> bool:
        """Return True if any required section heading is missing."""
        for section in rule.get("value", []):
            # Match section names with spaces or underscores, case-insensitive
            normalized = section.replace("_", " ")
            if normalized not in output_lower and section not in output_lower:
                log.debug(
                    f"Governance violation [{rule['id']}]: "
                    f"missing required section '{section}'"
                )
                return True
        return False


class SemanticLayer:
    """
    Loads ontology.yaml and vocabulary.json and resolves NL terms to
    exact YAML parameter values.

    TECHNOLOGY
      json / yaml (stdlib + PyYAML) — parses vocabulary.json and ontology.yaml
      str.lower()                   — case-insensitive substring matching
                                      No ML needed — pure dict lookup is fast
                                      and deterministic.

    WHY NO ML HERE
      SemanticLayer does not use embeddings. The vocabulary.json mappings
      are explicit and curated — "every tuesday" should always map to
      "0 9 * * 2", not a probabilistic approximation of it.
      FAISSRetriever handles semantic similarity. SemanticLayer handles
      deterministic term resolution.
    """

    def __init__(self, semantic_cfg: dict, domain_folder: Path):
        self._vocab_path    = domain_folder / semantic_cfg["vocabulary"]
        self._ontology_path = domain_folder / semantic_cfg["ontology"]
        self._vocab:    dict = {}
        self._ontology: dict = {}
        self._load()

    def _load(self) -> None:
        import json as _json
        try:
            with open(self._vocab_path, encoding="utf-8") as f:
                self._vocab = _json.load(f)
            log.debug(
                f"SemanticLayer loaded vocabulary — "
                f"{sum(len(v) for k,v in self._vocab.items() if not k.startswith('_'))} terms"
            )
        except (OSError, ValueError) as e:
            log.warning(f"SemanticLayer: could not load vocabulary.json: {e}")

        try:
            with open(self._ontology_path, encoding="utf-8") as f:
                _loaded = yaml.safe_load(f)
                self._ontology = _loaded if isinstance(_loaded, dict) else {}
            log.debug(
                f"SemanticLayer loaded ontology — "
                f"{len(self._ontology.get('entities', []))} entities"
            )
        except (OSError, yaml.YAMLError) as e:
            log.warning(f"SemanticLayer: could not load ontology.yaml: {e}")

    def resolve_terms(self, nl_input: str) -> str:
        """
        Scan nl_input for known domain vocabulary and return a plain-text
        block of resolved mappings for injection into the Generator prompt.

        Example output:
            task_type     → email_generation
            email_type    → cold_outreach
            target_persona → vp_sales
            schedule.cron → 0 9 * * 2

        Returns empty string if no terms match (prompt is not enriched).
        """
        if not self._vocab or not nl_input:
            return ""

        lower      = nl_input.lower()
        resolved   = {}

        # Map groups in vocabulary.json to AgentConfig field names
        group_to_field = {
            "task_type_mappings":      "task_type",
            "schedule_mappings":       "schedule.cron",
            "email_type_mappings":     "email_type",
            "sequence_step_mappings":  "sequence_step",
            "sender_persona_mappings": "sender_persona",
            "target_persona_mappings": "target_persona",
            "lead_stage_mappings":     "lead_stage",
            "cta_type_mappings":       "cta_type",
        }

        for group_key, field_name in group_to_field.items():
            group = self._vocab.get(group_key, {})
            for phrase, value in group.items():
                if phrase.lower() in lower:
                    # First match wins per field — most specific match is
                    # preferred because longer phrases appear earlier in
                    # vocabulary.json (ordered by specificity)
                    if field_name not in resolved:
                        resolved[field_name] = value

        if not resolved:
            return ""

        lines = []
        for field, value in resolved.items():
            lines.append(f"  {field} → {value}")

        return "Resolved domain terms:\n" + "\n".join(lines)

    def infer_task_type(self, nl_input: str) -> str | None:
        """
        Return the most likely task_type for nl_input based on vocabulary
        mappings, or None if no match is found.

        Used by DomainPack.load() in Phase 2a when task_type is not
        explicitly provided by the caller.
        """
        if not self._vocab:
            return None

        lower    = nl_input.lower()
        mappings = self._vocab.get("task_type_mappings", {})

        # Sort by phrase length descending so longer, more specific phrases
        # match before shorter ones ("email sequence" before "email")
        for phrase in sorted(mappings, key=len, reverse=True):
            if phrase.lower() in lower:
                return mappings[phrase]

        return None


class FAISSRetriever:
    """
    Full implementation — delegates to embedder.FAISSRetriever.
    See embedder.py for full documentation.
    """
    def __init__(self, training_data_cfg: dict, domain_folder: Path):
        from embedder import FAISSRetriever as _Retriever
        self._impl = _Retriever(training_data_cfg, domain_folder)

    def get_top_k(self, nl_input: str, k: int = 3, task_type: str | None = None) -> str:
        return self._impl.get_top_k(nl_input, k=k, task_type=task_type)

    def get_denial_lessons(self, nl_input: str, k: int = 2, task_type: str | None = None) -> str:
        return self._impl.get_denial_lessons(nl_input, k=k, task_type=task_type)

    def add_rejection(self, text: str, task_type: str, rejection_reason: str, source_file: str = "") -> None:
        self._impl.add_rejection(text, task_type, rejection_reason, source_file)

    def rebuild(self) -> None:
        self._impl.rebuild()


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
        if not isinstance(cfg, dict):
            raise ValueError(
                f"domain.yaml must be a YAML mapping, got {type(cfg).__name__}"
            )

        log.debug(
            f"Loading domain '{domain_name}' | task_type: '{task_type}' | "
            f"yaml version: {cfg.get('version', 'unknown')}"
        )

        # ── Instantiate sub-components ─────────────────────────────────────
        governance = GovernanceLoader(cfg["governance"],    domain_folder)
        semantic   = SemanticLayer(cfg["semantic"],         domain_folder)
        retriever  = FAISSRetriever(cfg["training_data"],   domain_folder)

        # ── Resolve task_type from NL if not explicitly provided ───────────
        # SemanticLayer.infer_task_type() scans nl_input for vocabulary.json
        # keywords and returns the most likely task_type.
        # Explicit task_type arg always wins; inference is the fallback.
        if task_type:
            resolved_task_type = task_type
        elif nl_input:
            resolved_task_type = semantic.infer_task_type(nl_input) or "email_generation"
        else:
            resolved_task_type = "email_generation"

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
            or "gpt-4o"
        )

    def __repr__(self) -> str:
        return (
            f"DomainPack(name={self.name!r}, task_type={self.task_type!r}, "
            f"tools={self.tools})"
        )