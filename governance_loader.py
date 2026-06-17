from __future__ import annotations

# =============================================================================
# governance_loader.py
#
# Public re-export of GovernanceLoader from domain_pack.py, plus a
# standalone load() helper for importing GovernanceLoader directly
# without loading a full DomainPack.
#
# WHY THIS FILE EXISTS
# validator.py and generator.py import GovernanceLoader by name. Having a
# dedicated module makes imports clean:
#
#   from governance_loader import GovernanceLoader          # direct import
#   from governance_loader import load_governance           # path-based helper
#
# instead of:
#   from domain_pack import GovernanceLoader                # leaks internals
#
# TECHNOLOGY
#   Same as GovernanceLoader in domain_pack.py:
#   PyYAML + pathlib + re (stdlib)
# =============================================================================

from domain_pack import GovernanceLoader
from pathlib     import Path


def load_governance(domain_folder: str | Path) -> GovernanceLoader:
    """
    Load a GovernanceLoader directly from a domain folder path without
    instantiating a full DomainPack. Useful for standalone validation
    scripts and unit tests.

    Parameters
    ----------
    domain_folder : path to domains/{domain}/ folder
                    e.g. Path("domains/marketing")

    Returns
    -------
    GovernanceLoader with content_policy.yaml and brand_guidelines.md loaded.

    Example
    -------
        gov = load_governance("domains/marketing")
        violations = gov.check(email_text, task_type="email_generation")
        prompt_block = gov.to_prompt()
    """
    folder = Path(domain_folder)
    domain_yaml_path = folder / "domain.yaml"

    if not domain_yaml_path.exists():
        raise FileNotFoundError(
            f"domain.yaml not found at {domain_yaml_path}. "
            f"Run Phase 1a to create the domain folder structure."
        )

    import yaml
    with open(domain_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return GovernanceLoader(cfg["governance"], folder)


__all__ = ["GovernanceLoader", "load_governance"]