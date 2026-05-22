from agent                      import run_agent
from services.discord           import post_to_discord
from services.storage           import save_campaign, sources_to_list
from services.verify            import run_verification
from services.campaign_memory   import load_or_build_index, add_campaign_to_index
from services.prompt_validation import validate_user_prompt


def run_campaign():
    user_prompt = input("\nWhat do you want to post about?\n→ ").strip()
    if not user_prompt:
        print("No input provided.")
        return

    print("\n[0/3] Analyzing your prompt...")
    valid, reason = validate_user_prompt(user_prompt)
    if not valid:
        print(f"      ❌ Rejected — {reason}")
        print("      (No content generation — fix your prompt and try again.)")
        return

    print("      ✅ Prompt accepted")

    print("\n[1/3] Agent is researching and writing...")
    output        = run_agent(user_prompt)
    hashtags_list = [
        h.strip() for h in output["hashtags"].split()
        if h.strip().startswith("#")
    ]
    articles = output["articles"]
    print(f"      Done. ({len(articles)} articles fetched)" if articles else "      Done.")

    print("\n[2/3] Verifying content...")
    verification = run_verification(output["content"])
    verdict      = verification["verdict"]
    icon         = {"approved": "✅", "needs_revision": "⚠️", "rejected": "❌"}.get(verdict, "?")
    print(f"      {icon}  {verdict.upper()} — {verification['summary']}")

    if verification["revision_count"] > 0:
        print(f"      Revised {verification['revision_count']} time(s)")

    if verification["content"] != output["content"]:
        output["content"]   = verification["content"]
        parts               = [output["content"], output["hashtags"]]
        if output["sources"]:
            parts.append(f"📰 Sources:\n{output['sources']}")
        output["full_post"] = "\n\n".join(p for p in parts if p)

    # Rejected by verifier / reviser
    if verdict == "rejected":
        print("\n❌ Content rejected — not safe to post.")
        print(" Saving campaign...", end="", flush=True)
        saved = save_campaign(
            user_prompt,
            output["content"],
            hashtags_list,
            status="denied",
            verdict_info = verification,
        )
        print(" ✅")
        print("  Updating memory index...", end="", flush=True)
        try:
            add_campaign_to_index(saved)
            print(" ✅")
        except Exception as e:
            print(f" ⚠️  ({e})")
        print(f"   Saved as denied → {saved}")
        return

    print("\n[3/3] Review your draft:\n")
    print("─" * 50)
    print(output["full_post"])
    print("─" * 50)

    approval = input("\nApprove and post to Discord? (y/n): ").strip().lower()

    # approved  by human
    if approval == "y":
        print(" Posting to Discord...", end="", flush=True)
        success = post_to_discord(output["full_post"])
        if success:
            print(" ✅")
            print(" Saving campaign...", end="", flush=True)
            saved = save_campaign(
                user_prompt,
                output["full_post"],
                hashtags_list,
                status="posted",
                sources=sources_to_list(output.get("sources")),
                articles=output.get("articles", []),
                verdict_info=verification,
            )
            print(" ✅")
            print("  Updating memory index...", end="", flush=True)
            try:
                add_campaign_to_index(saved)
                print(" ✅")
            except Exception as e:
                print(f" ⚠️  ({e})")
            print(f"\n✅ Posted and saved → {saved}")
        else:
            # verified, but posting failed bc of webhook error
            print("\n❌ Discord post failed.")
    # denied by human
    else:
        print(" Saving campaign...", end="", flush=True)
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
        print(" ✅")
        print("  Updating memory index...", end="", flush=True)
        try:
            add_campaign_to_index(saved) 
            print(" ✅")
        except Exception as e:
            print(f" ⚠️  ({e})")
        print(f"\n🚫 Not posted. Saved → {saved}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("       Marketing Agent — AI Powered")
    print("="*50)

    print("\nInitializing campaign memory...")
    load_or_build_index()

    while True:
        run_campaign()
        again = input("\nRun another campaign? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!\n")
            break