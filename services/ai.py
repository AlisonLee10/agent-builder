import os
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from services.llm import llm
from services.logger import get_logger

load_dotenv()

log = get_logger(__name__)

# Campaign memory injected into generate_content_tool
_few_shot_examples: str = ""
_denial_lessons: str    = ""


def set_campaign_memory(
    approved_examples: str = "",
    denial_lessons: str = "",
) -> None:
    global _few_shot_examples, _denial_lessons
    _few_shot_examples = approved_examples or ""
    _denial_lessons    = denial_lessons or ""


def set_few_shot_examples(examples: str) -> None:
    set_campaign_memory(approved_examples=examples)


def clear_few_shot_examples() -> None:
    set_campaign_memory("", "")


def _memory_context_block() -> str:
    parts = [p for p in (_few_shot_examples, _denial_lessons) if p]
    return "\n\n".join(parts)

def generate_content(user_prompt: str, news_context: str = "", trends_context: str = "") -> str:
    log.debug(f"generate_content — prompt length {len(user_prompt)}")
    context_blocks = []
    if news_context:
        context_blocks.append(f"Recent news:\n{news_context}")
    if trends_context:
        context_blocks.append(f"Current trends:\n{trends_context}")

    memory = _memory_context_block()
    if context_blocks:
        system = (
            "You are a social media copywriter. "
            "Write an engaging marketing post under 200 words. "
            "If approved style examples are provided, match their tone and quality. "
            "If rejection lessons are provided, avoid the mistakes users cited. "
            "If news or trend context is provided, reference real facts naturally. "
            "Do not invent statistics. No hashtags."
        )
        user_message = f"Topic: {user_prompt}\n\n" + "\n\n".join(context_blocks)
    else:
        system = (
            "You are a social media copywriter. "
            "Write an engaging marketing post under 200 words. "
            "If rejection lessons are provided, avoid those mistakes. "
            "No hashtags."
        )
        user_message = user_prompt

    if memory:
        user_message = f"{user_message}\n\n{memory}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_message),
    ])
    return response.text.strip()


def generate_hashtags(user_prompt: str) -> list[str]:
    log.debug(f"generate_hashtags — topic length {len(user_prompt)}")
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