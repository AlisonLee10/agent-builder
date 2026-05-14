import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#1b
def generate_content(user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a social media copywriter. Writean engaging marketing post under 100 words."},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

#1d
def generate_hashtags(user_prompt: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate relevant hashtags for a social media post. Consider industry, niche, brand type, and region if mentioned. These hashtags will guide you write the post.Return ONLY hashtags, one per line, minimum 3, maximum 7. Each must start with #."},
            {"role": "user", "content": f"Generate hashtags for: {user_prompt}"}
        ]
    )
    raw = response.choices[0].message.content.strip()
    return [line.strip() for line in raw.splitlines() if line.strip().startswith("#")]