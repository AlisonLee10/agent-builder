from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agent                             import run_agent
from services.storage                  import save_campaign, normalize_research_for_save
from services.verify                   import run_verification
from services.campaign_memory          import load_or_build_index, add_campaign_to_index
from services.prompt_validation        import validate_user_prompt
from services.denial_reason_validation import validate_denial_reason
from services.platform_parser          import validate_posting_intent, format_platform_plan
from services.platform_posting         import post_to_platforms
from services.logger                   import (
    get_logger, new_run_id, clear_run_id, cleanup_old_logs, set_level
)

log = get_logger(__name__)

# =============================================================================
# main.py
#
# WHAT CHANGED (Phase 5a)
#
# parse_args() gains three new flags:
#
#   --domain <name>          Activate a domain pack (e.g. "marketing").
#                            Routes through the full domain-aware pipeline:
#                            DomainPack.load() → generate_config() →
#                            validate_config() → compile_and_run()
#
#   --template <name>        Load a pre-built AgentConfig YAML by template name
#                            (e.g. "weekly_trend_post") instead of calling the
#                            Generator. Skips the Claude API call entirely.
#                            Requires --domain.
#
#   --no-domain-pack         Run with DomainPack loaded but all domain context
#                            stripped (no Training Data, no Governance, no
#                            Semantic Layer). Used for D9-D10 baseline
#                            measurement against the domain-specific condition.
#
# run_campaign() is untouched — it still runs the original marketing agent
# path when no --domain flag is given. Full backward compatibility.
#
# Two new entry points are added:
#   run_domain_campaign()    → domain-aware interactive loop
#   run_template_campaign()  → one-shot template execution (no interaction)
#
# TECHNOLOGY
#   argparse       — existing, extended with three new arguments
#   DomainPack     — domain_pack.py (Phase 1c)
#   generate_config_sync — generator.py (Phase 2b)
#   validate_config      — validator.py (Phase 2c)
#   compile_and_run_sync — compiler.py  (Phase 4a)
#   AgentConfig.from_yaml_file — schema.py (Phase 2a)
# =============================================================================


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Agent Builder — domain-aware agent workflow platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original marketing agent (unchanged)
  python main.py

  # Domain-aware run — Generator produces AgentConfig from NL input
  python main.py --domain marketing

  # Run a pre-built template directly (no Generator call)
  python main.py --domain marketing --template weekly_trend_post

  # Baseline measurement run (no domain context injected)
  python main.py --domain marketing --no-domain-pack

  # Debug mode
  python main.py --domain marketing --debug
        """,
    )

    parser.add_argument(
        "--debug",
        action = "store_true",
        help   = "Enable DEBUG log level and AgentExecutor verbosity",
    )

    # ── New flags (Phase 5a) ───────────────────────────────────────────────

    parser.add_argument(
        "--domain",
        type    = str,
        default = None,
        metavar = "DOMAIN",
        help    = (
            "Activate a domain pack by folder name, e.g. 'marketing'. "
            "Routes through DomainPack → Generator → Compiler pipeline. "
            "If omitted, the original run_agent() path is used."
        ),
    )

    parser.add_argument(
        "--template",
        type    = str,
        default = None,
        metavar = "TEMPLATE_NAME",
        help    = (
            "Load a pre-built AgentConfig YAML template instead of calling "
            "the Generator. Template files live in "
            "domains/{domain}/templates/{name}.yaml. "
            "Requires --domain. Example: --template weekly_trend_post"
        ),
    )

    parser.add_argument(
        "--no-domain-pack",
        action  = "store_true",
        dest    = "no_domain_pack",
        help    = (
            "Load the domain pack but strip all domain context "
            "(Training Data, Governance, Semantic Layer). "
            "Used for D9-D10 baseline vs. domain-specific comparison. "
            "Requires --domain."
        ),
    )

    parser.add_argument(
        "--task-type",
        type    = str,
        default = None,
        metavar = "TASK_TYPE",
        help    = (
            "Override task type inference. "
            "e.g. 'email_generation', 'research_summary'. "
            "If omitted, SemanticLayer infers it from the prompt."
        ),
    )

    args = parser.parse_args()

    # Validate flag combinations
    if args.template and not args.domain:
        parser.error("--template requires --domain")
    if args.no_domain_pack and not args.domain:
        parser.error("--no-domain-pack requires --domain")

    return args


# ── Original campaign flow (unchanged) ───────────────────────────────────────

def run_campaign(*, debug: bool = False) -> None:
    """
    Original marketing agent loop — unchanged from the existing main.py.
    Used when --domain is not provided.
    """
    run_id = new_run_id()
    log.info(f"Campaign started - run ID: {run_id}")

    try:
        user_prompt = input("\nWhat do you want to post about?\n→ ").strip()
        if not user_prompt:
            log.warning("empty prompt — skipping")
            return

        log.info("[0/3] Analyzing your prompt...")
        valid, reason = validate_user_prompt(user_prompt)
        if not valid:
            log.warning(f"Prompt rejected — {reason}")
            log.info("No content generation — fix your prompt and try again")
            return

        log.debug("Prompt accepted")

        plat_ok, plat_reason, intent = validate_posting_intent(user_prompt)
        if not plat_ok:
            log.warning(plat_reason)
            print(f"\n⚠️  {plat_reason}")
            return

        log.info(format_platform_plan(intent))

        log.info("[1/3] Agent is researching and writing...")
        output        = run_agent(user_prompt, debug=debug)
        hashtags_list = [
            h.strip() for h in output["hashtags"].split()
            if h.strip().startswith("#")
        ]
        sources_list, articles = normalize_research_for_save(
            output.get("sources"),
            output.get("articles", []),
        )
        output["sources"]  = sources_list
        output["articles"] = articles
        log.debug(
            f"    Done. ({len(articles)} articles fetched)"
            if articles else "     Done."
        )

        log.info("[2/3] Verifying content...")
        verification = run_verification(output["content"])
        verdict      = verification["verdict"]
        icon         = {"approved": "✅", "needs_revision": "⚠️", "rejected": "❌"}.get(verdict, "?")
        log.debug(f"{icon} {verdict.upper()} — {verification['summary']}")

        if verification["revision_count"] > 0:
            log.debug(f"Revised {verification['revision_count']} time(s)")

        if verification["content"] != output["content"]:
            output["content"] = verification["content"]
            from services.post_content import build_publishable_post
            output["full_post"] = build_publishable_post(
                output["content"],
                output["hashtags"],
            )

        if verdict == "rejected":
            log.warning("Content rejected — not safe to post")
            saved = save_campaign(
                user_prompt, output["content"], hashtags_list,
                status="denied", verdict_info=verification,
                sources=sources_list, articles=articles,
                full_post=output["full_post"], run_id=run_id,
            )
            try:
                add_campaign_to_index(saved["filename"])
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")
            return

        log.info("[3/3] Review your draft:")
        print("─" * 50)
        print(output["full_post"])
        print("─" * 50)
        print(f"\n{format_platform_plan(intent)}")

        approval = input("\nApprove and post? (y/n): ").strip().lower()

        if approval == "y":
            log.info(f"Posting to {', '.join(intent.platforms)}...")
            posted, failed, errors = asyncio.run(
                post_to_platforms(
                    output["full_post"], intent,
                    content=output["content"], hashtags=hashtags_list,
                )
            )
            if failed:
                for p in failed:
                    log.warning(f"{p}: {errors.get(p, 'failed')}")
                if not posted:
                    log.warning("All platform posts failed")
                    return

            saved = save_campaign(
                user_prompt, output["full_post"], hashtags_list,
                status="posted", verdict_info=verification,
                platform=",".join(posted), posted_platforms=posted,
                sources=sources_list, articles=articles, run_id=run_id,
            )
            try:
                add_campaign_to_index(saved["filename"])
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")
        else:
            print("\nWhy are you denying this post?")
            print("(Be specific — e.g. too salesy, missing benefits, wrong tone)")
            while True:
                user_denial_reason = input("→ ").strip()
                if not user_denial_reason:
                    print("⚠️  A reason is required so the agent can learn from this.")
                    continue
                ok, msg = validate_denial_reason(
                    user_denial_reason, campaign_prompt=user_prompt
                )
                if ok:
                    break
                print(f"⚠️  {msg}")

            saved = save_campaign(
                user_prompt, output["content"], hashtags_list,
                status="denied", full_post=output["full_post"],
                verdict_info={"verdict": "user_denied", "issues": [], "summary": ""},
                platform="", posted_platforms=[],
                sources=sources_list, articles=articles,
                run_id=run_id, user_denial_reason=user_denial_reason,
            )
            try:
                add_campaign_to_index(saved["filename"])
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")

    finally:
        clear_run_id()


# ── Domain-aware campaign flow ────────────────────────────────────────────────

def run_domain_campaign(
    domain:        str,
    *,
    task_type:     str | None = None,
    no_domain_pack: bool = False,
    debug:         bool = False,
) -> None:
    """
    Domain-aware interactive campaign loop.
    Used when --domain is set and --template is not.

    Flow:
        user prompt → DomainPack.load() → generate_config() →
        validate_config() → compile_and_run() → review → approve/deny
    """
    run_id = new_run_id()
    log.info(f"Domain campaign started — domain: {domain} | run ID: {run_id}")

    try:
        user_prompt = input(
            f"\nWhat do you want the agent to do? (domain: {domain})\n→ "
        ).strip()
        if not user_prompt:
            log.warning("empty prompt — skipping")
            return

        valid, reason = validate_user_prompt(user_prompt)
        if not valid:
            print(f"\n⚠️  {reason}")
            return

        # ── Load domain pack ───────────────────────────────────────────────
        from domain_pack import DomainPack

        print(f"\n[1/4] Loading domain pack '{domain}'...")
        domain_pack = DomainPack.load(
            domain_name = domain,
            task_type   = task_type or "",
            nl_input    = user_prompt,
        )

        if no_domain_pack:
            # Baseline mode: zero out all domain context so the same
            # infrastructure runs with no domain intelligence injected.
            # Achieved by overriding the Jinja2 template back to the
            # fallback (original hardcoded prompt).
            from services.ai import _render_template
            log.info(
                "⚠️  --no-domain-pack: domain context stripped for baseline run"
            )
            print("⚠️  Baseline mode — domain context stripped")
            # Re-activate with empty governance and semantic hints
            from services.ai import set_domain
            set_domain(
                domain_name      = domain_pack.name,
                task_type        = domain_pack.task_type,
                domain_folder    = domain_pack.folder,
                governance_rules = "",   # stripped
                semantic_hints   = "",   # stripped
            )

        print(
            f"    ✓ Domain: {domain_pack.name} | "
            f"task_type: {domain_pack.task_type} | "
            f"model: {domain_pack.preferred_model()}"
        )

        # ── Generate AgentConfig ───────────────────────────────────────────
        print("\n[2/4] Generating workflow config from your prompt...")
        from generator import generate_config_sync
        config = generate_config_sync(user_prompt, domain_pack)
        print(
            f"    ✓ Config generated — {len(config.steps)} steps: "
            f"{[s.name for s in config.steps]}"
        )

        # ── Validate config ────────────────────────────────────────────────
        from validator import validate_config
        schema_result = validate_config(config, domain_pack)
        if not schema_result.passed:
            print("\n❌ Config failed domain validation:")
            for v in schema_result.violations:
                print(f"   [{v['id']}] {v['description']}")
            return
        if schema_result.warnings:
            for w in schema_result.warnings:
                print(f"   ⚠️  [{w['id']}] {w['description']}")

        # ── Compile and run ────────────────────────────────────────────────
        print("\n[3/4] Running workflow...")
        from compiler import compile_and_run_sync
        output = compile_and_run_sync(config, domain_pack, user_prompt, debug=debug)

        hashtags_list = (
            [h.strip() for h in output["hashtags"].split() if h.strip().startswith("#")]
            if isinstance(output.get("hashtags"), str)
            else output.get("hashtags", [])
        )
        sources_list, articles = normalize_research_for_save(
            output.get("sources"),
            output.get("articles", []),
        )

        # ── Governance check on output ─────────────────────────────────────
        print("\n[4/4] Review your output:")
        if output.get("content"):
            from validator import validate_output
            gov_result = validate_output(output["content"], config, domain_pack)
            if not gov_result.passed:
                print("⚠️  Governance violations detected:")
                for v in gov_result.violations:
                    print(f"   [{v['id']}] {v['description'][:100]}")
            if gov_result.warnings:
                for w in gov_result.warnings:
                    print(f"   ⚠️  [{w['id']}] {w['description'][:80]}")

        print("─" * 50)
        print(output.get("full_post", output.get("content", "[no output]")))
        print("─" * 50)

        # ── Human review ───────────────────────────────────────────────────
        approval = input("\nApprove? (y/n): ").strip().lower()

        if approval == "y":
            saved = save_campaign(
                user_prompt,
                output.get("full_post", output.get("content", "")),
                hashtags_list,
                status           = "posted",
                verdict_info     = {"verdict": "approved", "issues": [], "summary": ""},
                platform         = "",
                posted_platforms = [],
                sources          = sources_list,
                articles         = articles,
                run_id           = run_id,
            )
            try:
                add_campaign_to_index(saved["filename"])
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")
            print(f"\n✅ Saved → {saved['filename']}")

        else:
            print("\nWhy are you denying this output?")
            print("(Be specific — the agent learns from your reason)")
            while True:
                user_denial_reason = input("→ ").strip()
                if not user_denial_reason:
                    print("⚠️  A reason is required.")
                    continue
                ok, msg = validate_denial_reason(
                    user_denial_reason, campaign_prompt=user_prompt
                )
                if ok:
                    break
                print(f"⚠️  {msg}")

            saved = save_campaign(
                user_prompt,
                output.get("content", ""),
                hashtags_list,
                status             = "denied",
                full_post          = output.get("full_post", ""),
                verdict_info       = {"verdict": "user_denied", "issues": [], "summary": ""},
                platform           = "",
                posted_platforms   = [],
                sources            = sources_list,
                articles           = articles,
                run_id             = run_id,
                user_denial_reason = user_denial_reason,
            )
            try:
                add_campaign_to_index(saved["filename"])
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")

            # Self-learning: record rejection in domain FAISS index
            _record_domain_rejection(
                content          = output.get("content", ""),
                task_type        = domain_pack.task_type,
                rejection_reason = user_denial_reason,
                domain           = domain,
                source_file      = saved.get("filename", ""),
            )

            print(f"\n❌ Denied and saved → {saved['filename']}")

    finally:
        clear_run_id()


# ── Template execution flow ───────────────────────────────────────────────────

def run_template_campaign(
    domain:   str,
    template: str,
    *,
    debug:    bool = False,
) -> None:
    """
    One-shot template execution — no Generator call, no interactive prompt.
    Loads a pre-built AgentConfig YAML from domains/{domain}/templates/{name}.yaml
    and compiles + runs it directly.

    Used for:
        python main.py --domain marketing --template weekly_trend_post
    """
    run_id = new_run_id()
    log.info(f"Template run — domain: {domain} | template: {template} | run ID: {run_id}")

    try:
        # ── Resolve template path ──────────────────────────────────────────
        template_path = (
            Path("domains") / domain / "templates" / f"{template}.yaml"
        )
        if not template_path.exists():
            print(f"\n❌ Template not found: {template_path}")
            print(
                f"   Available templates in domains/{domain}/templates/:\n   "
                + "\n   ".join(
                    p.stem for p in
                    (Path("domains") / domain / "templates").glob("*.yaml")
                )
            )
            return

        # ── Load and validate AgentConfig from YAML ────────────────────────
        from schema import AgentConfig
        print(f"\n[1/3] Loading template '{template}'...")
        config = AgentConfig.from_yaml_file(str(template_path))
        print(
            f"    ✓ {len(config.steps)} steps: {[s.name for s in config.steps]} | "
            f"task_type: {config.task_type.value}"
        )

        # ── Load domain pack ───────────────────────────────────────────────
        from domain_pack import DomainPack
        domain_pack = DomainPack.load(
            domain_name = domain,
            task_type   = config.task_type.value,
            nl_input    = config.description or template,
        )

        # ── Validate config against domain ────────────────────────────────
        from validator import validate_config
        schema_result = validate_config(config, domain_pack)
        if not schema_result.passed:
            print("\n❌ Template failed domain validation:")
            for v in schema_result.violations:
                print(f"   [{v['id']}] {v['description']}")
            return

        # ── Compile and run ────────────────────────────────────────────────
        print(f"\n[2/3] Running template '{template}'...")
        from compiler import compile_and_run_sync
        nl_input = config.description or f"Run the {template} workflow"
        output   = compile_and_run_sync(config, domain_pack, nl_input, debug=debug)

        sources_list, articles = normalize_research_for_save(
            output.get("sources"),
            output.get("articles", []),
        )
        hashtags_list = (
            [h.strip() for h in output["hashtags"].split() if h.strip().startswith("#")]
            if isinstance(output.get("hashtags"), str)
            else output.get("hashtags", [])
        )

        # ── Output ─────────────────────────────────────────────────────────
        print(f"\n[3/3] Template '{template}' complete:")
        print("─" * 50)
        print(output.get("full_post", output.get("content", "[no output]")))
        print("─" * 50)

        # Auto-save template runs as posted (no interactive approval needed)
        saved = save_campaign(
            nl_input,
            output.get("full_post", output.get("content", "")),
            hashtags_list,
            status           = "posted",
            verdict_info     = {"verdict": "template_run", "issues": [], "summary": ""},
            platform         = "",
            posted_platforms = [],
            sources          = sources_list,
            articles         = articles,
            run_id           = run_id,
        )
        try:
            add_campaign_to_index(saved["filename"])
        except Exception as e:
            log.warning(f"Memory index update failed: {e}")

        print(f"\n✅ Saved → {saved['filename']}")

    finally:
        clear_run_id()


# ── Self-learning helper ──────────────────────────────────────────────────────

def _record_domain_rejection(
    content:          str,
    task_type:        str,
    rejection_reason: str,
    domain:           str,
    source_file:      str,
) -> None:
    """Feed a CLI rejection into the domain FAISS rejected/ index."""
    try:
        import yaml
        domain_folder    = Path("domains") / domain
        domain_yaml_path = domain_folder / "domain.yaml"
        if not domain_yaml_path.exists():
            return
        with open(domain_yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return
        from embedder import FAISSRetriever
        retriever = FAISSRetriever(cfg["training_data"], domain_folder)
        retriever.add_rejection(
            text             = content,
            task_type        = task_type,
            rejection_reason = rejection_reason,
            source_file      = source_file,
        )
        log.debug(f"Self-learning: CLI rejection recorded for domain '{domain}'")
    except Exception as e:
        log.warning(f"Self-learning index update failed (non-fatal): {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        set_level("DEBUG")
        log.debug("debug mode active")

    # ── Banner ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if args.domain:
        mode = "baseline (no domain pack)" if args.no_domain_pack else f"domain: {args.domain}"
        print(f"       AI Agent Builder — {mode}")
    else:
        print("       Marketing Agent — AI Powered")
    print("=" * 50)

    cleanup_old_logs()
    log.info("Initializing campaign memory...")
    load_or_build_index()

    # ── Route to correct execution mode ────────────────────────────────────
    if args.domain and args.template:
        # One-shot template run — no loop
        run_template_campaign(
            domain   = args.domain,
            template = args.template,
            debug    = args.debug,
        )

    elif args.domain:
        # Domain-aware interactive loop
        while True:
            run_domain_campaign(
                domain         = args.domain,
                task_type      = args.task_type,
                no_domain_pack = args.no_domain_pack,
                debug          = args.debug,
            )
            again = input("\nRun another? (y/n): ").strip().lower()
            if again != "y":
                log.info("session ended by user")
                print("\nGoodbye!\n")
                break

    else:
        # Original marketing agent loop — completely unchanged
        while True:
            run_campaign(debug=args.debug)
            again = input("\nRun another campaign? (y/n): ").strip().lower()
            if again != "y":
                log.info("session ended by user")
                print("\nGoodbye!\n")
                break