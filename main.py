import argparse
from agent                      import run_agent
from services.discord           import post_to_discord
from services.storage           import save_campaign, sources_to_list
from services.verify            import run_verification
from services.campaign_memory   import load_or_build_index, add_campaign_to_index
from services.prompt_validation import validate_user_prompt
from services.logger            import get_logger, new_run_id, clear_run_id, cleanup_old_logs, set_level

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marketing Agent - AI Powered")
    parser.add_argument(
        "--debug",
        action = "store_true",
        help = "Enable DEBUG log level and AgentExecutor verbosity"
    )
    return parser.parse_args()

def run_campaign(*, debug: bool = False) -> None:
    run_id = new_run_id()
    log.info(f"Campaign started - run ID: {run_id}")

    try:
        user_prompt = input("\nWhat do you want to post about?\n→ ").strip()
        if not user_prompt:
            log.warning("empty prompt — skipping")
            return

        log.debug("[0/3] Analyzing your prompt...")
        valid, reason = validate_user_prompt(user_prompt)
        if not valid:
            log.warning(f"Prompt rejected — {reason}")
            log.debug("No content generation — fix your prompt and try again")
            return

        log.debug("Prompt accepted")

        log.debug("[1/3] Agent is researching and writing...")
        output        = run_agent(user_prompt, debug=debug)
        hashtags_list = [
            h.strip() for h in output["hashtags"].split()
            if h.strip().startswith("#")
        ]
        articles = output["articles"]
        log.info(
            f"    Done. ({len(articles)} articles fetched)"
            if articles else "     Done."
        )
        if articles:
            log.debug(f"Agent done — {len(articles)} articles fetched")
        else:
            log.debug("Agent done")

        log.debug("[2/3] Verifying content...")
        verification = run_verification(output["content"])
        verdict      = verification["verdict"]
        icon         = {"approved": "✅", "needs_revision": "⚠️", "rejected": "❌"}.get(verdict, "?")
        log.debug(f"{icon} {verdict.upper()} — {verification['summary']}")

        if verification["revision_count"] > 0:
            log.debug(f"Revised {verification['revision_count']} time(s)")

        if verification["content"] != output["content"]:
            output["content"]   = verification["content"]
            parts               = [output["content"], output["hashtags"]]
            if output["sources"]:
                parts.append(f"📰 Sources:\n{output['sources']}")
            output["full_post"] = "\n\n".join(p for p in parts if p)

        if verdict == "rejected":
            log.warning("Content rejected — not safe to post")
            log.debug("Saving denied campaign...")
            saved = save_campaign(
                user_prompt,
                output["content"],
                hashtags_list,
                status="denied",
                verdict_info = verification,
            )
            log.debug("Updating memory index...")
            try:
                add_campaign_to_index(saved)
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")
            log.debug(f"Saved as denied → {saved}")
            return

        log.debug("[3/3] Review your draft:")
        print("─" * 50)
        print(output["full_post"])
        print("─" * 50)

        approval = input("\nApprove and post to Discord? (y/n): ").strip().lower()

        if approval == "y":
            log.debug("Posting to Discord...")
            success = post_to_discord(output["full_post"])
            if success:
                log.debug("Saving posted campaign...")
                saved = save_campaign(
                    user_prompt,
                    output["full_post"],
                    hashtags_list,
                    status="posted",
                    sources=sources_to_list(output.get("sources")),
                    articles=output.get("articles", []),
                    verdict_info=verification,
                )
                log.debug("Updating memory index...")
                try:
                    add_campaign_to_index(saved)
                except Exception as e:
                    log.warning(f"Memory index update failed: {e}")
                log.debug(f"Posted and saved → {saved}")
            else:
                log.warning("Discord post failed")
        else:
            log.debug("Saving user-denied campaign...")
            saved = save_campaign(
                user_prompt,
                output["content"],
                hashtags_list,
                status="denied",
                full_post=output["full_post"],
                verdict_info = {
                    "verdict": "user_denied",
                    "issues": [],
                    "summary": "User chose not to post",
                },
            )
            log.debug("Updating memory index...")
            try:
                add_campaign_to_index(saved)
            except Exception as e:
                log.warning(f"Memory index update failed: {e}")
            log.debug(f"Not posted. Saved → {saved}")

    finally:
        clear_run_id()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        set_level("DEBUG")
        log.debug("debug mode active")

    print("\n" + "="*50)
    print("       Marketing Agent — AI Powered")
    print("="*50)

    cleanup_old_logs()
    log.info("initializing campaign memory...")
    load_or_build_index()

    while True:
        run_campaign(debug=args.debug)
        again = input("\nRun another campaign? (y/n): ").strip().lower()
        if again != "y":
            log.info("session ended by user")
            print("\nGoodbye!\n")
            break