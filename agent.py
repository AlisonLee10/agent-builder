import asyncio

from dotenv import load_dotenv
from langchain.chat_models           import init_chat_model
from langchain_classic.agents        import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts          import ChatPromptTemplate, MessagesPlaceholder
from tools.tool_selector             import select_tools
from services.campaign_memory        import get_few_shot_examples, get_denial_lessons_for_agent
from services.ai                     import set_campaign_memory, clear_few_shot_examples
from services.progress               import show_progress
from services.agent_trace            import record_agent_scratchpad_to_langsmith
from services.logger                 import get_logger, get_run_id
from tools.mcp_client                import load_mcp_tools_for_agent, merge_agent_tools

load_dotenv()

log = get_logger(__name__)

agent_llm = init_chat_model("gpt-4o", temperature=0)

# =============================================================================
# WHAT CHANGED IN THIS FILE (Phase 1b)
#
# BEFORE:
#   The `prompt` variable was built at module load time with a hardcoded string:
#     ChatPromptTemplate.from_messages([
#         ("system", """You are a marketing content agent. Use available tools...
#                       Steps (use only the tools you have):
#                       1. Call brand_context_tool OR retrieve_brand_context FIRST...
#                       ...
#                    """),
#         ...
#     ])
#
# AFTER:
#   The system message string is no longer hardcoded here. Instead:
#     1. _build_system_prompt() reads the rendered output from services/ai.py's
#        _render_template("persona.j2"), which loads the active domain's
#        persona.j2 Jinja2 template.
#     2. prompt is built lazily inside run_agent_async() so it picks up
#        whatever domain was loaded before the agent runs.
#
# WHAT DID NOT CHANGE:
#   - run_agent_async() signature and return type — identical
#   - run_agent() signature and return type — identical
#   - All tool loading logic — identical
#   - parse_agent_output() — identical
#   - _extract_research_from_steps() — identical
#   - _build_all_tools() — identical
#   - AgentExecutor creation and invocation — identical
#   - LangSmith tracing — identical
#
# WHY BUILD PROMPT LAZILY
#   The original `prompt` was built once at import time, so the system string
#   was fixed for the lifetime of the process. Building it inside
#   run_agent_async() means set_domain() (called by DomainPack.load()) can
#   change the active template between calls — e.g. switching from
#   email_generation to research_summary changes the step instructions in
#   persona.j2 without restarting the server.
# =============================================================================


def _build_prompt() -> ChatPromptTemplate:
    """
    Build the ChatPromptTemplate using the currently active domain's
    persona.j2 template (rendered by services/ai._render_template).

    Falls back to the original hardcoded string if no domain is loaded,
    so the agent works identically to before until Phase 2a is complete.
    """
    from services.ai import _render_template  # import here to pick up live state

    system_message = _render_template("persona.j2")

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def _extract_research_from_steps(intermediate_steps: list) -> list[dict]:
    """Collect news articles and trend signals from tool calls in this run."""
    from services.news   import get_last_fetched_articles
    from services.trends import get_last_fetched_trends

    news_tools  = {"news_tool", "news_sources_tool"}
    trend_tools = {"reddit_tool", "trends_tool", "tavily_search", "tavily_extract"}

    articles:  list[dict] = []
    seen_urls: set[str]   = set()

    def _add(items: list[dict]) -> None:
        for item in items:
            url = (item.get("url") or "").strip()
            key = url or (item.get("title") or "")
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            articles.append(item)

    for action, _ in intermediate_steps:
        tool_name = getattr(action, "tool", None) or getattr(action, "name", None)
        if tool_name in news_tools:
            _add(get_last_fetched_articles())
        if tool_name in trend_tools:
            _add(get_last_fetched_trends())

    if not articles:
        _add(get_last_fetched_articles())
        _add(get_last_fetched_trends())

    return articles


def parse_agent_output(raw: str) -> dict:
    content  = ""
    hashtags = ""
    sources  = ""
    current  = None

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "[CONTENT]":
            current = "content"
        elif stripped == "[HASHTAGS]":
            current = "hashtags"
        elif stripped == "[SOURCES]":
            current = "sources"
        else:
            if current == "content":
                content  += line + "\n"
            elif current == "hashtags":
                hashtags += line + "\n"
            elif current == "sources":
                sources  += line + "\n"

    content  = content.strip()
    hashtags = hashtags.strip()
    sources  = sources.strip()

    from services.post_content import build_publishable_post

    sources_clean = sources if sources and sources.lower() != "none" else ""

    return {
        "content":   content,
        "hashtags":  hashtags,
        "sources":   sources_clean,
        "full_post": build_publishable_post(content, hashtags),
    }


async def _build_all_tools(user_prompt: str) -> list:
    native_tools = select_tools(user_prompt)
    mcp_tools    = await load_mcp_tools_for_agent()
    all_tools    = merge_agent_tools(native_tools, mcp_tools)

    if mcp_tools:
        mcp_names = [getattr(t, "name", "?") for t in mcp_tools]
        log.info(f"[MCP] attached {len(mcp_tools)} MCP tool(s): {mcp_names}")
    log.debug(
        f"Agent tool count: {len(all_tools)} "
        f"({len(native_tools)} native + {len(all_tools) - len(native_tools)} MCP)"
    )
    return all_tools


async def run_agent_async(user_prompt: str, *, debug: bool = False) -> dict:
    approved = get_few_shot_examples(user_prompt, k=2)
    lessons  = get_denial_lessons_for_agent(user_prompt, k=2)
    set_campaign_memory(approved_examples=approved, denial_lessons=lessons)

    if approved:
        log.debug("Similar approved campaigns found — using as style examples")
    if lessons:
        log.debug("Similar user-denied campaigns found — applying rejection lessons")
    if not approved and not lessons:
        log.debug("No similar campaigns yet — writing from scratch")

    all_tools = await _build_all_tools(user_prompt)

    # ── CHANGED: prompt now built lazily from persona.j2 ──────────────────
    prompt = _build_prompt()
    # ── END CHANGE ─────────────────────────────────────────────────────────

    agent    = create_openai_tools_agent(agent_llm, all_tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=debug,
        return_intermediate_steps=True,
    )

    log.debug("Invoking AgentExecutor (async)")
    with show_progress("      Generating content"):
        result = await executor.ainvoke(
            {"input": user_prompt},
            config={
                "metadata": {"run_id": get_run_id()},
                "tags":     ["marketing-agent"],
            },
        )

    record_agent_scratchpad_to_langsmith(
        user_prompt,
        result.get("intermediate_steps", []),
        result.get("output"),
    )

    log.debug("AgentExecutor finished — parsing output")
    output   = parse_agent_output(result["output"])
    articles = _extract_research_from_steps(result.get("intermediate_steps", []))

    from services.storage      import normalize_research_for_save
    from services.post_content import build_publishable_post

    sources_list, articles = normalize_research_for_save(
        output.get("sources", ""),
        articles,
    )
    output["sources"]   = sources_list
    output["articles"]  = articles
    output["full_post"] = build_publishable_post(
        output["content"],
        output["hashtags"],
    )

    if articles:
        log.debug(f"Attached {len(articles)} article(s) to campaign output")
    if sources_list:
        log.debug(f"Attached {len(sources_list)} source line(s) to campaign output")

    clear_few_shot_examples()
    return output


def run_agent(user_prompt: str, *, debug: bool = False) -> dict:
    """Sync entry point for CLI and FastAPI sync routes."""
    return asyncio.run(run_agent_async(user_prompt, debug=debug))
