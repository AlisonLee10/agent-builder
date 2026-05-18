from langchain_core.messages import SystemMessage, HumanMessage
from services.llm import llm


def generate_content(user_prompt: str, news_context: str = "", trends_context: str = "") -> str:
    context_blocks = []
    if news_context:
        context_blocks.append(f"Recent news:\n{news_context}")
    if trends_context:
        context_blocks.append(f"Current trends:\n{trends_context}")

    if context_blocks:
        system = (
            "You are a social media copywriter. "
            "Write an engaging marketing post under 200 words. "
            "Reference real facts from the context naturally. "
            "Do not invent statistics. No hashtags."
        )
        user_message = f"Topic: {user_prompt}\n\n" + "\n\n".join(context_blocks)
    else:
        system       = "You are a social media copywriter. Write an engaging marketing post under 200 words. No hashtags."
        user_message = user_prompt

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_message),
    ])
    return response.text.strip()


def generate_hashtags(user_prompt: str) -> list[str]:
    response = llm.invoke([
        SystemMessage(content=(
            "Generate relevant hashtags for a social media post. "
            "Consider industry, niche, brand type, and region if mentioned. "
            "Return ONLY hashtags, one per line, minimum 3, maximum 8. "
            "Each must start with #."
        )),
        HumanMessage(content=f"Generate hashtags for: {user_prompt}"),
    ])
    raw = response.text.strip()
    return [line.strip() for line in raw.splitlines() if line.strip().startswith("#")]