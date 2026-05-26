import json

from langchain_openai          import ChatOpenAI
from langchain_core.messages   import SystemMessage, HumanMessage
from services.logger import get_logger

from services.tools import (
    brand_context_tool,
    news_tool,
    news_sources_tool,
    trends_tool,
    generate_content_tool,
    generate_hashtags_tool,
)

log = get_logger(__name__)

try:
    import tiktoken
    _enc              = tiktoken.encoding_for_model("gpt-4o")
    _count_tokens     = lambda text: len(_enc.encode(text))
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

_ALL_TOOLS = {
    "brand_context_tool":    brand_context_tool,
    "news_tool":             news_tool,
    "news_sources_tool":     news_sources_tool,
    "trends_tool":           trends_tool,
    "generate_content_tool": generate_content_tool,
    "generate_hashtags_tool":generate_hashtags_tool,
}

_CORE = ["brand_context_tool", "generate_content_tool", "generate_hashtags_tool"]

_selector_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def select_tools(prompt: str) -> list:
    """
    Pick only the tools this prompt needs.
    Core tools always included. Optional tools decided by a fast LLM call.
    """
    response = _selector_llm.invoke([
        SystemMessage(content=(
            "You select optional tools for a marketing agent.\n\n"
            "Optional tools:\n"
            "- news    → fetch real news articles "
              "(use when prompt asks for news, recent events, facts, or sources)\n"
            "- trends  → scan Google + Reddit "
              "(use when prompt asks about trends, viral topics, or what's popular)\n\n"
            'Reply with ONLY a JSON array. Examples: ["news"] ["trends"] '
            '["news","trends"] []'
        )),
        HumanMessage(content=f"Prompt: {prompt}"),
    ])

    try:
        needed = json.loads(response.text.strip())
    except (json.JSONDecodeError, ValueError):
        needed = []

    names = list(_CORE)
    if "news" in needed:
        names += ["news_tool", "news_sources_tool"]
    if "trends" in needed:
        names.append("trends_tool")

    # deduplicate, preserve order
    seen, unique = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    selected = [_ALL_TOOLS[n] for n in unique]

    not_used = [n for n in _ALL_TOOLS if n not in unique]

    if TIKTOKEN_AVAILABLE:
        all_tokens  = sum(_count_tokens(t.description) for t in _ALL_TOOLS.values())
        used_tokens = sum(_count_tokens(t.description) for t in selected)
        saved       = all_tokens - used_tokens
        log.debug(f"Tools {len(selected)}/6 selected · saved {saved} description tokens")
    else:
        log.debug(f"Tools {len(selected)}/6 selected")

    log.debug(f"Using: {', '.join(unique)}")
    log.debug(f"Skipped: {', '.join(not_used) if not_used else 'none'}")

    return selected


def _print_tool_report(selected_names: list[str], selected_tools: list) -> None:
    all_names   = list(_ALL_TOOLS.keys())
    skipped     = [n for n in all_names if n not in selected_names]
    total       = len(all_names)

    log.debug(f"Tools {len(selected_names)}/{total} selected")
    log.debug(f"Using: {', '.join(selected_names)}")
    log.debug(f"Skipped: {', '.join(skipped) if skipped else '(none)'}")

    if TIKTOKEN_AVAILABLE:
        all_tokens  = sum(_count_tokens(t.description) for t in _ALL_TOOLS.values())
        used_tokens = sum(_count_tokens(t.description) for t in selected_tools)
        saved       = all_tokens - used_tokens
        log.debug(f"Tool description tokens: {used_tokens} used (saved {saved})")