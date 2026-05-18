from agent           import run_agent
from services.discord import post_to_discord
from services.storage import save_campaign
from services.verify  import run_verification

print("\n=== Marketing Agent ===")
user_prompt = input("What do you want to post about? ").strip()

if not user_prompt:
    print("No input. Exiting.")
    exit()

# Agent generates content
print("\nAgent is working...\n")
output = run_agent(user_prompt)

# Verification loop (LangGraph)
print("\nVerifying content...\n")
verification = run_verification(output["content"])

verdict = verification["verdict"]
icon    = {"approved": "✅", "needs_revision": "⚠️", "rejected": "❌"}.get(verdict, "?")
print(f"{icon} Verdict: {verdict.upper()}")
print(f"   {verification['summary']}")

if verification["issues"]:
    for issue in verification["issues"]:
        print(f"   · {issue}")

# If content was revised, update the full post
if verification["content"] != output["content"]:
    print("\n  Content was revised to fix issues.")
    output["content"]   = verification["content"]
    parts               = [output["content"], output["hashtags"]]
    if output["sources"]:
        parts.append(f"📰 Sources:\n{output['sources']}")
    output["full_post"] = "\n\n".join(p for p in parts if p)

# Hard stop if rejected
if verdict == "rejected":
    print("\n❌ Content rejected — not safe to post.")
    saved = save_campaign(user_prompt, output["full_post"], [], status="denied")
    print(f"Saved as denied → {saved}")
    exit()

# Show draft
print("\n" + "="*50)
print("DRAFT — Please review before posting:")
print("="*50)
print(output["full_post"])
print("="*50)

# Human approval
approval = input("\nApprove and post to Discord? (y/n): ").strip().lower()

if approval == "y":
    success = post_to_discord(output["full_post"])
    if success:
        saved = save_campaign(user_prompt, output["full_post"], [], status="posted")
        print(f"\n✅ Posted and saved → {saved}")
    else:
        print("\n❌ Discord post failed.")
else:
    saved = save_campaign(user_prompt, output["full_post"], [], status="denied")
    print(f"\n🚫 Not posted. Saved → {saved}")