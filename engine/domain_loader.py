"""
engine/domain_loader.py

Loads rich domain context from a structured domain folder (domains/<name>/).
Reads brand_guidelines.md, content_policy.yaml, and a sample of approved
training examples; returns a single formatted block for injection into an
LLM system prompt.

Called by _load_domain_context() in engine/nodes.py when a domain entry in
domains_config.json has a "folder" key pointing to a domain pack directory.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).parent.parent


def load_rich_context(folder: str | Path) -> str:
    """
    Build a rich system-prompt block from a domain folder.
    Sections included (each only if the source file exists):
      1. Brand & Style Guidelines  — from governance/brand_guidelines.md
      2. Enforced Content Rules    — from governance/content_policy.yaml
      3. Approved Style Examples   — 3 random .txt files from training_data/approved/
    """
    path = Path(folder)
    if not path.is_absolute():
        path = _REPO_ROOT / path

    cfg = _load_yaml(path / "domain.yaml") or {}
    gov = cfg.get("governance", {})
    td  = cfg.get("training_data", {})

    parts: list[str] = []

    # ── 1. Brand guidelines ────────────────────────────────────────────────────
    bg_path = _resolve(path, gov.get("brand_guidelines", "governance/brand_guidelines.md"))
    if bg_path.exists():
        cleaned = _clean_md(bg_path.read_text(encoding="utf-8"))
        if cleaned:
            parts.append("## Brand & Style Guidelines\n\n" + cleaned)

    # ── 2. Content policy rules ────────────────────────────────────────────────
    cp_path = _resolve(path, gov.get("content_policy", "governance/content_policy.yaml"))
    if cp_path.exists():
        policy = _load_yaml(cp_path) or {}
        rules_block = _format_rules(policy.get("rules", []))
        if rules_block:
            parts.append("## Enforced Content Rules\n\n" + rules_block)

    # ── 3. Approved style examples ─────────────────────────────────────────────
    approved_rel = td.get("approved", "training_data/approved")
    approved_path = _resolve(path, approved_rel)
    examples = _sample_examples(approved_path, n=3)
    if examples:
        parts.append(
            "## Approved Style Examples\n\n"
            "These are reference examples for tone, structure, and conciseness. "
            "Adapt the style — do not copy brand names or specific product claims:\n\n"
            + "\n\n---\n\n".join(examples)
        )

    # ── 4. Rejected style examples (if domain-specific rejections exist) ────────
    rejected_rel = td.get("rejected", "training_data/rejected")
    rejected_path = _resolve(path, rejected_rel)
    rejected_examples = _sample_examples(rejected_path, n=3)
    if rejected_examples:
        parts.append(
            "## Rejected Style Examples\n\n"
            "The following outputs were rejected by a human reviewer. "
            "Do NOT replicate their tone, structure, phrasing, or content:\n\n"
            + "\n\n---\n\n".join(rejected_examples)
        )

    return "\n\n".join(parts)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve(base: Path, rel: str) -> Path:
    return base / rel.lstrip("./")


def _load_yaml(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _clean_md(text: str) -> str:
    """
    Remove internal code-comment header blocks (lines starting with '# =' or
    '# SOURCE') that are meant for developers, not LLMs. Keeps all markdown
    content intact.
    """
    lines = []
    skip_block = False
    for line in text.splitlines():
        # Detect start/end of comment header block
        if line.startswith("# ===="):
            skip_block = not skip_block
            continue
        if skip_block or line.startswith("# SOURCE") or line.startswith("# WHY"):
            continue
        lines.append(line)
    # Trim leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _format_rules(rules: list) -> str:
    """
    Convert content_policy.yaml rule objects into a readable constraint list.
    Errors (⛔) are listed before warnings (⚠️).
    """
    errors:   list[str] = []
    warnings: list[str] = []

    for r in rules:
        severity = r.get("severity", "error")
        rtype    = r.get("type", "")
        val      = r.get("value", [])
        field    = r.get("field", "")

        line: str | None = None

        if rtype == "forbidden_word" and isinstance(val, list):
            line = f"Never use these words: {', '.join(val)}"
        elif rtype == "forbidden_phrase_start" and isinstance(val, list):
            shown = val[:5]
            more  = f" (+ {len(val)-5} more)" if len(val) > 5 else ""
            line  = f"Never open with: \"{'\"; \"'.join(shown)}\"{more}"
        elif rtype == "max_length" and field:
            unit = "characters" if "char" in field else "words"
            line = f"Keep {field} under {val} {unit}"
        elif rtype == "required_present" and isinstance(val, list):
            line = f"Must personalise — include one of: {', '.join(val)}"
        elif rtype == "max_count" and field:
            line = f"Maximum {val} {field} per output"
        elif rtype == "approved_claims_only":
            claims = r.get("approved_claims", [])
            if claims:
                line = "Only use approved statistics: " + " | ".join(f'"{c}"' for c in claims)

        if line:
            if severity == "error":
                errors.append(f"⛔ {line}")
            else:
                warnings.append(f"⚠️  {line}")

    return "\n".join(errors + warnings)


def _sample_examples(approved_path: Path, n: int = 3) -> list[str]:
    """
    Pick n representative .txt files from approved/ for use as few-shot examples.
    Tries to draw from different subdirectories for variety.
    Caps each example at 600 characters to keep token use reasonable.
    """
    if not approved_path.exists():
        return []

    # Build a pool: root-level files + files from each subdir
    root_files = list(approved_path.glob("*.txt"))
    subdir_pools: dict[str, list[Path]] = {}
    for subdir in sorted(approved_path.iterdir()):
        if subdir.is_dir():
            files = list(subdir.glob("*.txt"))
            if files:
                subdir_pools[subdir.name] = files

    # Try to pick one from each of the first n subdirs, fall back to root pool
    picked: list[Path] = []
    for name, pool in list(subdir_pools.items())[:n]:
        picked.append(random.choice(pool))
    while len(picked) < n and root_files:
        candidate = random.choice(root_files)
        if candidate not in picked:
            picked.append(candidate)
        root_files = [f for f in root_files if f not in picked]

    out: list[str] = []
    for f in picked[:n]:
        try:
            text = f.read_text(encoding="utf-8").strip()
            if not text:
                continue
            excerpt = text[:600] + ("…" if len(text) > 600 else "")
            out.append(f"*[{f.stem}]*\n\n{excerpt}")
        except Exception:
            continue
    return out
