from services.ai import generate_content, generate_hashtags
from services.discord import post_to_discord
from services.storage import save_campaign

print("\n=== Marketing Agent ===")
user_prompt = input("What do you want to post about? ").strip()

if not user_prompt:
    print("No input. Exiting.")
    exit()

print("\nGenerating content...")
content = generate_content(user_prompt)

print("Generating hashtags...")
hashtags = generate_hashtags(user_prompt)
full_post = f"{content}\n\n{' '.join(hashtags)}"

# Show full draft
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
        saved = save_campaign(user_prompt, content, hashtags, status="posted")
        print(f"\n✅ Posted to Discord and saved → {saved}")
    else:
        print("\n❌ Discord post failed. Not saved as posted.")
else:
    saved = save_campaign(user_prompt, content, hashtags, status="denied")
    print(f"\n🚫 Not posted. Saved as denied → {saved}")