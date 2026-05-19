from agent                         import run_agent
from services.discord              import post_to_discord
from services.storage              import save_campaign
from services.verify               import run_verification
from services.prompt_validation    import validate_user_prompt
def _sources_for_save(output: dict, articles: list[dict]) -> list[str]:
    if articles:
        return [
            f"- [{a['title']}]({a['url']})"
            for a in articles
            if a.get("title") and a.get("url")
        ]
    raw = output.get("sources", "").strip()
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def print_header():
    print("\n" + "="*50)
    print("       Marketing Agent — AI Powered")
    print("="*50)


def run_campaign():
    user_prompt = input("\nWhat do you want to post about?\n→ ").strip()

    valid, reason = validate_user_prompt(user_prompt)
    if not valid:
        print(f"\n⚠️  {reason}")
        return

    print("\n[1/3] Agent is researching and writing...")
    output = run_agent(user_prompt)
    if output.get("rejected"):
        print("\n⚠️  Agent could not produce content — input is not a valid marketing topic.")
        return

    hashtags_list = [
        h.strip() for h in output["hashtags"].split()
        if h.strip().startswith("#")
    ]
    articles = output["articles"]
    sources  = _sources_for_save(output, articles)
    print(f"      Done. ({len(articles)} articles fetched)" if articles else "      Done.")

    print("\n[2/3] Verifying content...")
    verification = run_verification(output["content"], user_prompt=user_prompt)

    verdict = verification["verdict"]
    icon    = {"approved": "✅", "needs_revision": "⚠️", "rejected": "❌"}.get(verdict, "?")
    print(f"      {icon}  {verdict.upper()} — {verification['summary']}")

    if verification["revision_count"] > 0:
        print(f"      Revised {verification['revision_count']} time(s) to fix issues.")

    if verification["content"] != output["content"]:
        output["content"]   = verification["content"]
        parts               = [output["content"], output["hashtags"]]
        if output["sources"]:
            parts.append(f"📰 Sources:\n{output['sources']}")
        output["full_post"] = "\n\n".join(p for p in parts if p)

    if verdict == "rejected":
        print("\n❌ Content rejected — not safe to post.")
        saved = save_campaign(
            user_prompt, output["content"], hashtags_list,
            status="denied", sources=sources, articles=articles,
            full_post=output["full_post"],
        )
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
                user_prompt, output["content"], hashtags_list,
                status="posted", sources=sources, articles=articles,
                full_post=output["full_post"],
            )
            print(f"\n✅ Posted and saved → {saved}")
        else:
            print("\n❌ Discord post failed.")
    else:
        saved = save_campaign(
            user_prompt, output["content"], hashtags_list,
            status="denied", sources=sources, articles=articles,
            full_post=output["full_post"],
        )
        print(f"\n🚫 Not posted. Saved → {saved}")


if __name__ == "__main__":
    print_header()

    while True:
        run_campaign()
        again = input("\nRun another campaign? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!\n")
            break