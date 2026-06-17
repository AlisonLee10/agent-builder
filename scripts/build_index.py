#!/usr/bin/env python3
"""
scripts/build_index.py

Populates and verifies the domain FAISS Training Data indexes.

Usage:
    # Build indexes for the marketing domain
    python scripts/build_index.py --domain marketing

    # Rebuild from scratch (clears existing indexes)
    python scripts/build_index.py --domain marketing --rebuild

    # Verify index contents without rebuilding
    python scripts/build_index.py --domain marketing --verify

    # Run a test query against the built index
    python scripts/build_index.py --domain marketing --query "cold email for VP Sales"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and verify domain FAISS Training Data indexes"
    )
    parser.add_argument(
        "--domain",
        required = True,
        metavar  = "DOMAIN",
        help     = "Domain folder name, e.g. 'marketing'",
    )
    parser.add_argument(
        "--rebuild",
        action = "store_true",
        help   = "Force full rebuild — clears existing indexes from disk",
    )
    parser.add_argument(
        "--verify",
        action = "store_true",
        help   = "Print index stats and sample documents without rebuilding",
    )
    parser.add_argument(
        "--query",
        type    = str,
        default = None,
        metavar = "QUERY",
        help    = "Run a test similarity query against the approved index",
    )
    return parser.parse_args()


def _load_retriever(domain: str):
    """Load a FAISSRetriever for the given domain."""
    import yaml
    from pathlib import Path
    from embedder import FAISSRetriever

    domain_folder    = Path("domains") / domain
    domain_yaml_path = domain_folder / "domain.yaml"

    if not domain_yaml_path.exists():
        print(f"❌ domain.yaml not found at {domain_yaml_path}")
        print(f"   Run Phase 1a to create the domain folder structure.")
        sys.exit(1)

    with open(domain_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print(f"❌ domain.yaml is empty or invalid at {domain_yaml_path}")
        sys.exit(1)

    return FAISSRetriever(cfg["training_data"], domain_folder), domain_folder


def cmd_rebuild(domain: str) -> None:
    """Force a full rebuild of both FAISS indexes."""
    print(f"\n🔄 Rebuilding indexes for domain '{domain}'...")
    retriever, domain_folder = _load_retriever(domain)

    # Load training data counts before rebuild
    approved_path = domain_folder / "training_data" / "approved"
    rejected_path = domain_folder / "training_data" / "rejected"

    approved_files = list(approved_path.glob("*")) if approved_path.exists() else []
    rejected_files = list(rejected_path.glob("*")) if rejected_path.exists() else []

    approved_files = [
        f for f in approved_files
        if f.suffix in {".txt", ".md", ".json", ".yaml", ".yml"}
    ]
    rejected_files = [
        f for f in rejected_files
        if f.suffix in {".txt", ".md", ".json", ".yaml", ".yml"}
    ]

    print(f"   Found {len(approved_files)} approved files")
    print(f"   Found {len(rejected_files)} rejected files")

    if len(approved_files) == 0:
        print(
            "\n⚠️  No approved training data files found.\n"
            f"   Add .txt, .json, or .md files to:\n"
            f"   {approved_path}\n"
            f"   See Phase 6a collection instructions for sources."
        )

    if len(rejected_files) == 0:
        print(
            "\n⚠️  No rejected training data files found.\n"
            f"   Add rejection .json files to:\n"
            f"   {rejected_path}"
        )

    print("\n   Loading embedding model (first run downloads ~90MB)...")
    retriever.rebuild()

    print(f"\n✅ Indexes rebuilt:")
    print(f"   Approved docs indexed: {len(retriever._approved_docs)}")
    print(f"   Rejected docs indexed: {len(retriever._rejected_docs)}")
    print(f"   Index saved to: memory/domain_index/")


def cmd_verify(domain: str) -> None:
    """Print index stats and sample documents."""
    retriever, _ = _load_retriever(domain)
    retriever._ensure_loaded()

    total_approved = len(retriever._approved_docs)
    total_rejected = len(retriever._rejected_docs)

    print(f"\n📊 Index stats for domain '{domain}':")
    print(f"   Approved documents : {total_approved}")
    print(f"   Rejected documents : {total_rejected}")

    if total_approved == 0 and total_rejected == 0:
        print(
            "\n⚠️  Both indexes are empty.\n"
            "   Run: python scripts/build_index.py --domain marketing --rebuild"
        )
        return

    # Task type breakdown
    from collections import Counter
    approved_by_type = Counter(
        d.task_type for d in retriever._approved_docs
    )
    rejected_by_type = Counter(
        d.task_type for d in retriever._rejected_docs
    )

    print(f"\n   Approved by task_type:")
    for task_type, count in sorted(approved_by_type.items()):
        bar = "█" * count
        print(f"     {task_type:<30} {count:>3}  {bar}")

    print(f"\n   Rejected by task_type:")
    for task_type, count in sorted(rejected_by_type.items()):
        bar = "█" * count
        print(f"     {task_type:<30} {count:>3}  {bar}")

    # Rejection reason breakdown
    if retriever._rejected_docs:
        reason_counter = Counter(
            d.rejection_reason.split(":")[0].strip()
            for d in retriever._rejected_docs
            if d.rejection_reason
        )
        print(f"\n   Rejection reason categories:")
        for reason, count in reason_counter.most_common():
            pct = count / total_rejected * 100
            print(f"     {reason:<40} {count:>3}  ({pct:.0f}%)")

    # Sample approved doc
    if retriever._approved_docs:
        sample = retriever._approved_docs[0]
        print(f"\n   Sample approved doc:")
        print(f"     task_type   : {sample.task_type}")
        print(f"     source_file : {Path(sample.source_file).name}")
        print(f"     text[:120]  : {sample.text[:120].strip()}...")

    # Sample rejected doc
    if retriever._rejected_docs:
        sample = retriever._rejected_docs[0]
        print(f"\n   Sample rejected doc:")
        print(f"     task_type        : {sample.task_type}")
        print(f"     rejection_reason : {sample.rejection_reason[:80]}")
        print(f"     text[:120]       : {sample.text[:120].strip()}...")

    # Sprint brief N >= 100 check
    print(f"\n   Sprint brief compliance:")
    status = "✅" if total_approved + total_rejected >= 100 else "❌"
    print(
        f"   {status} Total samples: {total_approved + total_rejected} "
        f"(target: N ≥ 100)"
    )

    approved_status = "✅" if total_approved >= 70 else "⚠️"
    rejected_status = "✅" if total_rejected >= 30 else "⚠️"
    print(f"   {approved_status} Approved: {total_approved} (target: ≥ 70)")
    print(f"   {rejected_status} Rejected: {total_rejected} (target: ≥ 30)")


def cmd_query(domain: str, query: str) -> None:
    """Run a test similarity query and print top-3 results."""
    retriever, _ = _load_retriever(domain)
    retriever._ensure_loaded()

    if not retriever._approved_docs:
        print(
            "⚠️  Approved index is empty — no results to return.\n"
            "   Run --rebuild first."
        )
        return

    print(f"\n🔍 Query: '{query}'")
    print(f"   Searching approved index ({len(retriever._approved_docs)} docs)...\n")

    # Test all task types
    for task_type in ["email_generation", "research_summary", "campaign_brief"]:
        results = retriever._search(
            query     = query,
            index     = retriever._approved_idx,
            docs      = retriever._approved_docs,
            k         = 2,
            task_type = task_type,
        )
        if results:
            print(f"   [{task_type}] top results:")
            for i, doc in enumerate(results, 1):
                print(f"     {i}. {Path(doc.source_file).name}")
                print(f"        {doc.text[:120].strip()}...")
            print()

    # Also test unfiltered top-3
    unfiltered = retriever._search(
        query     = query,
        index     = retriever._approved_idx,
        docs      = retriever._approved_docs,
        k         = 3,
        task_type = None,
    )
    print(f"   [all task types] top-3 unfiltered:")
    for i, doc in enumerate(unfiltered, 1):
        print(f"     {i}. [{doc.task_type}] {Path(doc.source_file).name}")
        print(f"        {doc.text[:120].strip()}...")


def cmd_rejection_distribution_check(domain: str) -> None:
    """
    Check whether the rejection reason distribution matches the target
    from the Domain Selection Brief (Compliance 40%, Tone 35%, Structure 25%).
    """
    retriever, _ = _load_retriever(domain)
    retriever._ensure_loaded()

    if not retriever._rejected_docs:
        print("⚠️  No rejected docs — skipping distribution check.")
        return

    from collections import Counter
    total   = len(retriever._rejected_docs)
    reasons = Counter()

    for doc in retriever._rejected_docs:
        reason = doc.rejection_reason.lower()
        if any(k in reason for k in ("compliance", "can-spam", "gdpr", "footer", "legal")):
            reasons["compliance"] += 1
        elif any(k in reason for k in ("tone", "forbidden", "opening", "hype", "brand")):
            reasons["tone"] += 1
        elif any(k in reason for k in ("structure", "length", "word", "cta", "personalization")):
            reasons["structure"] += 1
        else:
            reasons["other"] += 1

    targets = {"compliance": 0.40, "tone": 0.35, "structure": 0.25}

    print(f"\n📐 Rejection distribution check (target from Domain Selection Brief):")
    for category, target_pct in targets.items():
        actual_count = reasons.get(category, 0)
        actual_pct   = actual_count / total
        target_count = int(total * target_pct)
        delta        = actual_pct - target_pct
        status       = "✅" if abs(delta) <= 0.10 else "⚠️"
        print(
            f"   {status} {category:<12} "
            f"actual: {actual_count:>3} ({actual_pct:.0%})  "
            f"target: {target_count:>3} ({target_pct:.0%})  "
            f"delta: {delta:+.0%}"
        )


if __name__ == "__main__":
    args = parse_args()

    if args.rebuild:
        cmd_rebuild(args.domain)
        cmd_verify(args.domain)
        cmd_rejection_distribution_check(args.domain)

    elif args.verify:
        cmd_verify(args.domain)
        cmd_rejection_distribution_check(args.domain)

    elif args.query:
        cmd_query(args.domain, args.query)

    else:
        # Default: verify current state
        cmd_verify(args.domain)