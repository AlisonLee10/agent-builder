from agent                      import run_agent
from services.discord           import post_to_discord
from services.storage           import save_campaign
from services.verify            import run_verification
from services.campaign_memory   import load_or_build_index, add_campaign_to_index


def run_campaign():
    user_prompt = input("\nWhat do you want to post about?\n→ ").strip()
    if not user_prompt:
        print("No input provided.")
        return

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

    if verdict == "rejected":
        print("\n❌ Content rejected — not safe to post.")
        saved = save_campaign(
            user_prompt, output["full_post"], hashtags_list, status="denied",
        )
        add_campaign_to_index(saved)        # ← update memory
        print(f"   Saved as denied → {saved}")
        return

    print("\n[3/3] Review your draft:\n")
    print("─" * 50)
    print(output["full_post"])
    print("─" * 50)

    approval = input("\nApprove and post to Discord? (y/n): ").strip().lower()

    if approval == "y":
        success = post_to_discord(output["full_post"])
        if success:
            saved = save_campaign(
                user_prompt, output["full_post"], hashtags_list, status="posted",
            )
            add_campaign_to_index(saved)    # ← update memory
            print(f"\n✅ Posted and saved → {saved}")
        else:
            print("\n❌ Discord post failed.")
    else:
        saved = save_campaign(
            user_prompt, output["full_post"], hashtags_list, status="denied",
        )
        add_campaign_to_index(saved)        # ← update memory
        print(f"\n🚫 Not posted. Saved → {saved}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("       Marketing Agent — AI Powered")
    print("="*50)

    print("\nInitializing campaign memory...")
    load_or_build_index()                   # ← build or load index at startup

    while True:
        run_campaign()
        again = input("\nRun another campaign? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!\n")
            break