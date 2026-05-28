from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.tool_selector import select_tools
from services.campaign_memory import get_few_shot_examples, get_denial_lessons_for_agent
from services.ai              import set_campaign_memory, clear_few_shot_examples
from services.progress        import show_progress
from services.agent_trace     import record_agent_scratchpad_to_langsmith
from services.logger          import get_logger, get_run_id
from tools.mcp_client    import get_mcp_client

load_dotenv()

log = get_logger(__name__)

agent_llm = init_chat_model("gpt-4o", temperature=0)

# Flexible prompt — works with any subset of tools
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a marketing content agent. Use available tools to create a post.

Steps (use only the tools you have):
1. Call brand_context_tool FIRST — always
2. If news_tool is available, fetch relevant articles
3. If reddit_tool is available, scan for current trends
4. If tavily-search is available, search the web for current trends and context
5. IF fetch is available and you have a specific URL worth reading, fetch it
6 Call generate_content_tool with topic and all context gathered
7. Call generate_hashtags_tool with the original topic
8. If news_sources_tool is available and you used news_tool, call it for source links
9. If check_brand_compliance is available, verify the generated content

Return the final post in exactly this format — no extra commentary:

[CONTENT]
<post copy>

[HASHTAGS]
<hashtags>

[SOURCES]
<source links or None>
"""),
    ("human", "{input}"),
    # The agent_scratchpad is where LangChain stores the growing list of tool calls and results between loop iterations.
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def _extract_research_from_steps(intermediate_steps: list) -> list[dict]:
    """Collect news articles and trend signals from tool calls in this run."""
    from services.news   import get_last_fetched_articles
    from services.trends import get_last_fetched_trends

    news_tools  = {"news_tool", "news_sources_tool"}
    trend_tools = {"reddit_tool", "trends_tool"}

    articles: list[dict] = []
    seen_urls: set[str]  = set()

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


def run_agent(user_prompt: str, *, debug: bool = False) -> dict:

    approved = get_few_shot_examples(user_prompt, k=2)
    lessons  = get_denial_lessons_for_agent(user_prompt, k=2)
    set_campaign_memory(approved_examples=approved, denial_lessons=lessons)

    if approved:
        log.debug("Similar approved campaigns found — using as style examples")
    if lessons:
        log.debug("Similar user-denied campaigns found — applying rejection lessons")
    if not approved and not lessons:
        log.debug("No similar campaigns yet — writing from scratch")

    # dynamic tool selection
    selected_tools = select_tools(user_prompt)

    # build and run agent
    log.debug(f"Building agent with {len(selected_tools)} tools")
    agent = create_openai_tools_agent(agent_llm, selected_tools, prompt)
    executor = AgentExecutor(
        agent = agent,
        tools = selected_tools,
        verbose = debug,
        return_intermediate_steps = True,
    )

    log.debug("Invoking AgentExecutor")
    with show_progress("      Generating content"):
        result = executor.invoke(
            {"input": user_prompt},
            config={
                "metadata": {"run_id": get_run_id()},
                "tags": ["marketing-agent"],
            },
        )

    record_agent_scratchpad_to_langsmith(
        user_prompt,
        result.get("intermediate_steps", []),
        result.get("output"),
    )

    log.debug("AgentExecutor finished — parsing output")
    output = parse_agent_output(result["output"])
    articles = _extract_research_from_steps(result.get("intermediate_steps", []))

    from services.storage import normalize_research_for_save
    from services.post_content import build_publishable_post

    sources_list, articles = normalize_research_for_save(
        output.get("sources", ""),
        articles,
    )
    output["sources"]  = sources_list
    output["articles"] = articles
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