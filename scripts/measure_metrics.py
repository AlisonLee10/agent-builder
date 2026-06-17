#!/usr/bin/env python3
"""
scripts/measure_metrics.py

D9-D10 Axis 1 metric measurement script.
Runs the same NL inputs 10 times each under two conditions:
  - Domain-specific  : full DomainPack (Training Data + Governance + Semantic Layer)
  - Baseline         : DomainPack loaded but all domain context stripped

Records all 6 KPIs from the Domain Selection Brief and prints a
structured comparison report. Also saves results to a JSON file
for further analysis.

Usage:
    # Full measurement (10 runs × 3 inputs × 2 conditions = 60 runs)
    python scripts/measure_metrics.py --domain marketing

    # Quick smoke test (3 runs instead of 10)
    python scripts/measure_metrics.py --domain marketing --runs 3

    # Single input only
    python scripts/measure_metrics.py --domain marketing --input-id A

    # Load from a previous results file and reprint the report
    python scripts/measure_metrics.py --report results/metrics_TIMESTAMP.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Test inputs (span multiple task types per sprint brief) ──────────────────

TEST_INPUTS = {
    "A": (
        "email_generation",
        "Write a 3-step cold outreach sequence targeting VP of Sales personas "
        "at Series A SaaS companies. Route each draft to Slack for approval.",
    ),
    "B": (
        "research_summary",
        "Research the top 5 competitors who launched new features this week "
        "in the B2B SaaS productivity space. Summarize their positioning and "
        "suggest a differentiation angle for FlowAI.",
    ),
    "C": (
        "email_generation",
        "Every Tuesday, research trending B2B SaaS topics, draft a cold email "
        "targeting Marketing Ops personas at companies that just raised a "
        "Series A, and route to Slack for approval.",
    ),
}

# Governance compliance checklist (5 items evaluated per run)
COMPLIANCE_CHECKLIST = [
    ("no_forbidden_phrases", [
        "revolutionary", "game-changer", "game changer", "disruptive",
        "world-class", "best-in-class", "synergy", "paradigm shift",
        "cutting-edge", "next-generation",
    ]),
    ("no_forbidden_openings", [
        "i hope this email finds you well",
        "my name is",
        "i wanted to reach out",
        "just following up",
        "i hope you're doing well",
    ]),
    ("has_personalization_token", [
        "{{first_name}}", "{{company}}", "{{recipient}}",
        "[name]", "[company]", "first_name", "company",
    ]),
    ("single_cta_only", None),       # evaluated separately
    ("no_spam_triggers", [
        "FREE", "WINNER", "CLICK NOW", "ACT FAST",
        "URGENT", "LIMITED TIME", "RISK-FREE",
    ]),
]


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SingleRunResult:
    run_number:        int
    input_id:          str
    condition:         str          # "domain" or "baseline"
    task_type:         str
    success:           bool         # did the pipeline complete without exception
    e2e_latency_sec:   float
    yaml_valid:        bool         # AgentConfig generated and validated
    yaml_attempts:     int          # 1 = first try, 2 = needed retry
    content_output:    str          # generated email / research text
    compliance_score:  float        # 0.0–1.0 (passed checklist items / total)
    compliance_detail: dict         # {check_name: passed}
    revision_count:    int          # reject→regenerate loops before approval
    error:             str = ""     # exception message if success=False


@dataclass
class ConditionSummary:
    condition:                  str
    total_runs:                 int
    successful_runs:            int
    yaml_success_rate:          float   # % valid on first try
    avg_compliance_score:       float   # brand compliance rate
    avg_revisions:              float   # avg revisions to approval
    avg_latency_sec:            float
    p95_latency_sec:            float
    self_learning_baseline:     int     # errors in first 5 runs
    self_learning_after:        int     # same error type in runs 6-10
    runs:                       list[SingleRunResult] = field(default_factory=list)


# ── Compliance evaluation ─────────────────────────────────────────────────────

def evaluate_compliance(content: str, task_type: str) -> tuple[float, dict]:
    """
    Run the content through the governance compliance checklist.
    Returns (score 0.0-1.0, detail dict).
    """
    if not content or not content.strip():
        return 0.0, {name: False for name, _ in COMPLIANCE_CHECKLIST}

    lower   = content.lower()
    results = {}

    for check_name, check_values in COMPLIANCE_CHECKLIST:
        if check_name == "no_forbidden_phrases":
            results[check_name] = not any(v.lower() in lower for v in check_values)

        elif check_name == "no_forbidden_openings":
            # Check only the first 200 chars (opening of email)
            opening = lower[:200].strip()
            results[check_name] = not any(opening.startswith(v) for v in check_values)

        elif check_name == "has_personalization_token":
            # For email tasks only — research summaries don't need tokens
            if task_type == "email_generation":
                results[check_name] = any(v.lower() in lower for v in check_values)
            else:
                results[check_name] = True  # not applicable

        elif check_name == "single_cta_only":
            # Count CTA indicators — more than 2 = likely multiple CTAs
            cta_signals = [
                "book a call", "schedule a call", "book a demo",
                "schedule a demo", "reply to this", "click here",
                "visit our website", "learn more at", "download",
                "sign up", "get started",
            ]
            cta_count = sum(1 for s in cta_signals if s in lower)
            results[check_name] = cta_count <= 1

        elif check_name == "no_spam_triggers":
            results[check_name] = not any(v in content for v in check_values)

    passed = sum(1 for v in results.values() if v)
    score  = passed / len(results)
    return score, results


# ── Pipeline invocation ───────────────────────────────────────────────────────

async def run_domain_pipeline(
    nl_input:       str,
    domain:         str,
    task_type:      str,
    baseline_mode:  bool,
) -> tuple[bool, float, bool, int, str, int]:
    """
    Invoke the full domain pipeline for one run.

    Returns:
        success, latency_sec, yaml_valid, yaml_attempts, content_output, revision_count
    """
    from domain_pack import DomainPack
    from generator  import generate_config
    from validator  import validate_config
    from compiler   import compile_and_run

    start = time.perf_counter()

    try:
        # Load domain pack
        domain_pack = DomainPack.load(
            domain_name = domain,
            task_type   = task_type,
            nl_input    = nl_input,
        )

        # Strip domain context for baseline condition
        if baseline_mode:
            from services.ai import set_domain
            set_domain(
                domain_name      = domain_pack.name,
                task_type        = domain_pack.task_type,
                domain_folder    = domain_pack.folder,
                governance_rules = "",
                semantic_hints   = "",
            )

        # Generate config — track attempts
        yaml_valid    = False
        yaml_attempts = 0
        config        = None

        try:
            config        = await generate_config(nl_input, domain_pack)
            yaml_valid    = True
            yaml_attempts = 1
        except Exception:
            # Generator already retried once internally — count as 2 attempts
            yaml_attempts = 2

        if config is None:
            latency = time.perf_counter() - start
            return False, latency, False, yaml_attempts, "", 0

        # Validate config
        schema_result = validate_config(config, domain_pack)
        if not schema_result.passed:
            latency = time.perf_counter() - start
            return False, latency, yaml_valid, yaml_attempts, "", 0

        # Compile and run — count revision loops
        revision_count = 0
        output         = {}

        # Wrap compile_and_run to detect HITL rejection loops
        # For measurement purposes we auto-approve (no human reviewer)
        # and count governance violations as "revision needed"
        output = await compile_and_run(config, domain_pack, nl_input)

        content = output.get("content", "")

        # Count revisions as governance violations that would require a loop
        from validator import validate_output
        gov_result = validate_output(content, config, domain_pack)
        # Each error-severity violation = one revision round needed
        revision_count = len([v for v in gov_result.violations
                              if v["severity"] == "error"])

        latency = time.perf_counter() - start
        return True, latency, yaml_valid, yaml_attempts, content, revision_count

    except Exception as e:
        latency = time.perf_counter() - start
        return False, latency, False, 0, "", 0


# ── Self-learning detection ───────────────────────────────────────────────────

def detect_self_learning_effect(runs: list[SingleRunResult]) -> tuple[int, int]:
    """
    Measure the self-learning KPI:
    Count compliance error occurrences in first 5 runs vs last 5 runs.
    A reduction indicates the FAISS rejected/ index is working.

    Returns: (errors_first_5, errors_last_5)
    """
    successful = [r for r in runs if r.success]
    if len(successful) < 6:
        return 0, 0

    first_5 = successful[:5]
    last_5  = successful[-5:]

    def count_errors(run_list):
        return sum(
            1 for r in run_list
            for passed in r.compliance_detail.values()
            if not passed
        )

    return count_errors(first_5), count_errors(last_5)


# ── Run measurement ───────────────────────────────────────────────────────────

async def run_condition(
    domain:        str,
    input_id:      str,
    task_type:     str,
    nl_input:      str,
    condition:     str,
    n_runs:        int,
) -> ConditionSummary:
    """Run one condition (domain or baseline) n_runs times for one input."""

    baseline_mode = (condition == "baseline")
    runs: list[SingleRunResult] = []

    print(f"\n  [{condition.upper()}] Input {input_id} — {n_runs} runs")
    print(f"  {'─' * 55}")

    for i in range(1, n_runs + 1):
        print(f"  Run {i:>2}/{n_runs} ...", end=" ", flush=True)

        success, latency, yaml_valid, yaml_attempts, content, revisions = \
            await run_domain_pipeline(nl_input, domain, task_type, baseline_mode)

        compliance_score, compliance_detail = evaluate_compliance(content, task_type)

        result = SingleRunResult(
            run_number        = i,
            input_id          = input_id,
            condition         = condition,
            task_type         = task_type,
            success           = success,
            e2e_latency_sec   = round(latency, 2),
            yaml_valid        = yaml_valid,
            yaml_attempts     = yaml_attempts,
            content_output    = content[:300],   # truncate for storage
            compliance_score  = round(compliance_score, 3),
            compliance_detail = compliance_detail,
            revision_count    = revisions,
        )
        runs.append(result)

        # Progress indicator
        status = "✅" if success else "❌"
        yaml_s = "YAML✓" if yaml_valid else "YAML✗"
        comp_s = f"comp:{compliance_score:.0%}"
        print(f"{status} {yaml_s} {comp_s} {latency:.1f}s")

    # Aggregate
    successful      = [r for r in runs if r.success]
    yaml_first_try  = [r for r in successful if r.yaml_attempts == 1]
    latencies       = [r.e2e_latency_sec for r in successful]
    compliance_vals = [r.compliance_score for r in successful]
    revision_vals   = [r.revision_count for r in successful]
    sl_first5, sl_last5 = detect_self_learning_effect(runs)

    yaml_success_rate    = len(yaml_first_try) / len(successful) if successful else 0.0
    avg_compliance_score = statistics.mean(compliance_vals) if compliance_vals else 0.0
    avg_revisions        = statistics.mean(revision_vals)   if revision_vals   else 0.0
    avg_latency          = statistics.mean(latencies)       if latencies        else 0.0
    p95_latency          = (
        sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2
        else (latencies[0] if latencies else 0.0)
    )

    return ConditionSummary(
        condition              = condition,
        total_runs             = n_runs,
        successful_runs        = len(successful),
        yaml_success_rate      = round(yaml_success_rate, 3),
        avg_compliance_score   = round(avg_compliance_score, 3),
        avg_revisions          = round(avg_revisions, 2),
        avg_latency_sec        = round(avg_latency, 2),
        p95_latency_sec        = round(p95_latency, 2),
        self_learning_baseline = sl_first5,
        self_learning_after    = sl_last5,
        runs                   = runs,
    )


# ── Report printing ───────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    """Print a formatted comparison report to stdout."""

    domain   = results["domain"]
    n_runs   = results["n_runs"]
    inputs   = results["inputs"]
    timestamp = results["timestamp"]

    TARGETS = {
        "yaml_success_rate":    (0.60, 0.90, ">90%"),
        "avg_compliance_score": (0.50, 0.85, ">85%"),
        "avg_revisions":        (3.5,  2.0,  "<2 rounds", True),  # lower is better
        "avg_latency_sec":      (62.0, 45.0, "<45 sec",   True),
    }

    def verdict(metric, value, target_info):
        baseline_val, target_val, target_str, *lower_better = target_info
        lower = bool(lower_better)
        if lower:
            met = value <= target_val
        else:
            met = value >= target_val
        return "✅" if met else "⚠️ "

    print("\n")
    print("=" * 70)
    print(f"  D9–D10 METRIC MEASUREMENT REPORT")
    print(f"  Domain: {domain} | Runs per condition: {n_runs} | {timestamp}")
    print("=" * 70)

    for input_id, input_data in inputs.items():
        task_type = input_data["task_type"]
        domain_s  = input_data.get("domain")
        baseline_s = input_data.get("baseline")

        if not domain_s or not baseline_s:
            continue

        print(f"\n── Input {input_id}: {task_type} ──")
        print(f"   Prompt: {input_data['nl_input'][:80]}...")
        print()
        print(f"  {'Metric':<32} {'Baseline':>10}  {'Domain':>10}  {'Target':>10}  {'Met?':>5}")
        print(f"  {'─' * 32} {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 5}")

        metrics = [
            ("YAML success rate (1st try)",
             "yaml_success_rate",
             f"{baseline_s['yaml_success_rate']:.0%}",
             f"{domain_s['yaml_success_rate']:.0%}"),

            ("Brand compliance rate",
             "avg_compliance_score",
             f"{baseline_s['avg_compliance_score']:.0%}",
             f"{domain_s['avg_compliance_score']:.0%}"),

            ("Avg revisions to approval",
             "avg_revisions",
             f"{baseline_s['avg_revisions']:.1f}",
             f"{domain_s['avg_revisions']:.1f}"),

            ("Avg E2E latency (sec)",
             "avg_latency_sec",
             f"{baseline_s['avg_latency_sec']:.1f}s",
             f"{domain_s['avg_latency_sec']:.1f}s"),

            ("P95 E2E latency (sec)",
             "p95_latency_sec",
             f"{baseline_s['p95_latency_sec']:.1f}s",
             f"{domain_s['p95_latency_sec']:.1f}s"),
        ]

        for label, key, base_val, dom_val in metrics:
            target_info = TARGETS.get(key)
            if target_info:
                v = verdict(key, domain_s[key], target_info)
                target_str = target_info[2]
            else:
                v = "  "
                target_str = "—"
            print(f"  {label:<32} {base_val:>10}  {dom_val:>10}  {target_str:>10}  {v:>5}")

        # Self-learning KPI
        sl_base   = domain_s["self_learning_baseline"]
        sl_after  = domain_s["self_learning_after"]
        if sl_base > 0:
            reduction = (sl_base - sl_after) / sl_base
            sl_met    = "✅" if reduction >= 0.50 else "⚠️ "
            print(f"  {'Self-learning error reduction':<32} {'—':>10}  "
                  f"{reduction:>10.0%}  {'≥50%':>10}  {sl_met:>5}")
        else:
            print(f"  {'Self-learning error reduction':<32} {'—':>10}  "
                  f"{'N/A (no errors)':>10}  {'≥50%':>10}  {'✅':>5}")

        # Success rate
        dom_success  = domain_s["successful_runs"]
        base_success = baseline_s["successful_runs"]
        print(f"\n  Successful runs: baseline {base_success}/{n_runs} | "
              f"domain {dom_success}/{n_runs}")

    # Aggregate across all inputs
    print(f"\n── Aggregate (all inputs combined) ──\n")

    all_domain   = [v["domain"]   for v in inputs.values() if v.get("domain")]
    all_baseline = [v["baseline"] for v in inputs.values() if v.get("baseline")]

    if all_domain and all_baseline:
        def avg(lst, key): return statistics.mean(d[key] for d in lst)

        agg_metrics = [
            ("YAML success rate (1st try)", "yaml_success_rate", ">90%",    False),
            ("Brand compliance rate",       "avg_compliance_score", ">85%", False),
            ("Avg revisions to approval",   "avg_revisions", "<2 rounds",   True),
            ("Avg E2E latency (sec)",        "avg_latency_sec", "<45 sec",   True),
        ]

        print(f"  {'Metric':<32} {'Baseline':>10}  {'Domain':>10}  {'Target':>10}  {'Met?':>5}")
        print(f"  {'─' * 32} {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 5}")

        for label, key, target_str, lower_better in agg_metrics:
            base_val = avg(all_baseline, key)
            dom_val  = avg(all_domain, key)
            if lower_better:
                threshold = float(target_str.replace("<","").replace(" rounds","").replace(" sec","").strip())
                met = dom_val <= threshold
            else:
                if "%" in target_str:
                    threshold = float(target_str.replace(">","").replace("%","").strip()) / 100
                else:
                    threshold = float(target_str.replace(">","").strip())
                met = dom_val >= threshold

            icon     = "✅" if met else "⚠️ "
            fmt      = ".0%" if key in ("yaml_success_rate", "avg_compliance_score") else ".1f"
            base_str = f"{base_val:{fmt}}"
            dom_str  = f"{dom_val:{fmt}}"
            print(f"  {label:<32} {base_str:>10}  {dom_str:>10}  {target_str:>10}  {icon:>5}")

    print("\n" + "=" * 70)


# ── Save / load ───────────────────────────────────────────────────────────────

def save_results(results: dict, domain: str) -> Path:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"metrics_{domain}_{ts}.json"

    # Convert SingleRunResult objects to dicts for JSON serialisation
    serialisable = json.loads(json.dumps(results, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
    path.write_text(json.dumps(serialisable, indent=2))
    print(f"\n  Results saved → {path}")
    return path


def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D9-D10 metric measurement — domain vs baseline comparison"
    )
    parser.add_argument("--domain",   type=str, default=None,
                        help="Domain folder name, e.g. 'marketing'")
    parser.add_argument("--runs",     type=int, default=10,
                        help="Number of runs per condition (default: 10)")
    parser.add_argument("--input-id", type=str, default=None,
                        choices=list(TEST_INPUTS.keys()),
                        help="Run only one input ID (A, B, or C)")
    parser.add_argument("--report",   type=str, default=None,
                        help="Path to a previous results JSON — reprint report only")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run baseline condition only (skip domain condition)")
    parser.add_argument("--domain-only", action="store_true",
                        help="Run domain condition only (skip baseline condition)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Reprint mode
    if args.report:
        results = load_results(args.report)
        print_report(results)
        return

    if not args.domain:
        print("❌ --domain is required. Example: --domain marketing")
        sys.exit(1)

    inputs_to_run = (
        {args.input_id: TEST_INPUTS[args.input_id]}
        if args.input_id
        else TEST_INPUTS
    )

    conditions = []
    if not args.baseline_only:
        conditions.append("domain")
    if not args.domain_only:
        conditions.append("baseline")
    if not conditions:
        conditions = ["domain", "baseline"]

    n_runs = args.runs
    total  = len(inputs_to_run) * len(conditions) * n_runs

    print(f"\n{'=' * 70}")
    print(f"  D9–D10 Metric Measurement")
    print(f"  Domain: {args.domain} | Conditions: {conditions}")
    print(f"  Inputs: {list(inputs_to_run.keys())} | Runs each: {n_runs}")
    print(f"  Total pipeline invocations: {total}")
    print(f"{'=' * 70}")

    results = {
        "domain":    args.domain,
        "n_runs":    n_runs,
        "timestamp": datetime.now().isoformat(),
        "inputs":    {},
    }

    for input_id, (task_type, nl_input) in inputs_to_run.items():
        print(f"\n{'─' * 70}")
        print(f"  Input {input_id}: {task_type}")
        print(f"  {nl_input[:100]}...")

        results["inputs"][input_id] = {
            "task_type": task_type,
            "nl_input":  nl_input,
        }

        for condition in conditions:
            summary = await run_condition(
                domain    = args.domain,
                input_id  = input_id,
                task_type = task_type,
                nl_input  = nl_input,
                condition = condition,
                n_runs    = n_runs,
            )
            results["inputs"][input_id][condition] = {
                "successful_runs":        summary.successful_runs,
                "yaml_success_rate":      summary.yaml_success_rate,
                "avg_compliance_score":   summary.avg_compliance_score,
                "avg_revisions":          summary.avg_revisions,
                "avg_latency_sec":        summary.avg_latency_sec,
                "p95_latency_sec":        summary.p95_latency_sec,
                "self_learning_baseline": summary.self_learning_baseline,
                "self_learning_after":    summary.self_learning_after,
                "runs":                   [asdict(r) for r in summary.runs],
            }

    save_results(results, args.domain)
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())