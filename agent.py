from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from services.tool_selector import select_tools
from services.campaign_memory import get_few_shot_examples
from services.ai              import set_few_shot_examples, clear_few_shot_examples
from services.progress        import show_progress

load_dotenv()

agent_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

# Flexible prompt — works with any subset of tools
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a marketing content agent. Use available tools to create a post.

Steps (use only the tools you have):
1. Call brand_context_tool FIRST — always
2. If news_tool is available, fetch relevant articles
3. If trends_tool is available, scan for current trends
4. Call generate_content_tool with topic and all context gathered
5. Call generate_hashtags_tool with the original topic
6. If news_sources_tool is available and you used news_tool, call it for source links

Return the final post in exactly this format — no extra commentary:

[CONTENT]
<post copy>

[HASHTAGS]
<hashtags>

[SOURCES]
<source links or None>
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def _extract_articles_from_steps(intermediate_steps: list) -> list[dict]:
    from services.news import get_last_fetched_articles
    for action, _ in intermediate_steps:
        if hasattr(action, "tool") and action.tool in ("news_tool", "news_sources_tool"):
            return get_last_fetched_articles()
    return []


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

    parts = [content, hashtags]
    if sources and sources.lower() != "none":
        parts.append(f"📰 Sources:\n{sources}")

    return {
        "content":   content,
        "hashtags":  hashtags,
        "sources":   sources if sources.lower() != "none" else "",
        "full_post": "\n\n".join(p for p in parts if p),
    }


def run_agent(user_prompt: str) -> dict:

    # Retrieve few shot examples from memory (#3d)
    examples = get_few_shot_examples(user_prompt, k = 2)
    set_few_shot_examples(examples)

    if examples:
        print("   [Memory] Similar approved cammpaigns found - using as style examples")
    else:
        print("   [Memory] No similar campaigns yet - writing from scratch")

    # dynamic tool selection
    selected_tools = select_tools(user_prompt)

    # build and run agent
    agent = create_openai_tools_agent(agent_llm, selected_tools, prompt)
    executor = AgentExecutor(
        agent = agent,
        tools = selected_tools,
        verbose = False,
        return_intermediate_steps = True,
    )

    with show_progress("      Generating content"):
        result = executor.invoke({"input": user_prompt})
    output = parse_agent_output(result["output"])
    output["articles"] = _extract_articles_from_steps(
        result.get("intermediate_steps",[])
    )

    # Always clean up cache
    clear_few_shot_examples()

    return output