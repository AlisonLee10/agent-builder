from agent import run_agent
from services.discord import post_to_discord
from services.storage import save_campaign
from services.news import fetch_news

print("\n=== Marketing Agent ===")
user_prompt = input("What do you want to post about? ").strip()

if not user_prompt:
    print("No input. Exiting.")
    exit()

print("\nAgent is working...\n")
full_post = run_agent(user_prompt)

# Show draft
print("\n" + "="*50)
print("DRAFT — Please review before posting:")
print("="*50)
print(full_post)
print("="*50)

# Human approval gate
approval = input("\nApprove and post to Discord? (y/n): ").strip().lower()

if approval == "y":
    success = post_to_discord(full_post)
    if success:
        saved = save_campaign(user_prompt, full_post, [], status="posted")
        print(f"\n✅ Posted and saved → {saved}")
    else:
        print("\n❌ Discord post failed.")
else:
    saved = save_campaign(user_prompt, full_post, [], status="denied")
    print(f"\n🚫 Not posted. Saved → {saved}")