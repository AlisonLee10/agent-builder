from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from services.tools import (
    brand_context_tool,
    news_tool,
    news_sources_tool,
    trends_tool,
    generate_content_tool,
    generate_hashtags_tool,
)

load_dotenv()

agent_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

tools = [
    brand_context_tool,
    news_tool,
    news_sources_tool,
    trends_tool,
    generate_content_tool,
    generate_hashtags_tool,
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a marketing content agent. Given a user's topic, follow these steps in order:

IMPORTANT: If the user input is gibberish, keyboard smash, symbols only, vague non-topics
("post smth", "im tired"), or unrelated to marketing, do NOT call any tools. Return ONLY:

[CONTENT]
INVALID_INPUT: The request is not a valid marketing topic.
[HASHTAGS]
None
[SOURCES]
None

1. Call brand_context_tool FIRST to retrieve brand guidelines and tone rules
2. Decide if fetching recent news would make the content more credible — if yes, call news_tool
3. Decide if scanning current trends would make the content more timely — if yes, call trends_tool
4. Call generate_content_tool with the topic AND the brand context AND any news/trends gathered
5. Call generate_hashtags_tool with the original topic
6. If you called news_tool, call news_sources_tool to get the source links

Then return the final post in exactly this format — no extra commentary:

[CONTENT]
<the post copy here>

[HASHTAGS]
<hashtags here>

[SOURCES]
<source links here, or 'None' if no news was fetched>
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(agent_llm, tools, prompt)
executor = AgentExecutor(
    agent                  = agent,
    tools                  = tools,
    verbose                = False,
    return_intermediate_steps = True,    # ← expose tool call results
)

def _extract_articles_from_steps(intermediate_steps: list) -> list[dict]:
    """Return structured articles if news tools ran during the agent run."""
    from services.news import get_last_fetched_articles

    for action, _ in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        if tool_name in ("news_tool", "news_sources_tool"):
            articles = get_last_fetched_articles()
            if articles:
                return articles
    return get_last_fetched_articles()

def parse_agent_output(raw: str) -> dict:
    content  = ""
    hashtags = ""
    sources  = ""

    current_section = None
    for line in raw.splitlines():
        if line.strip() == "[CONTENT]":
            current_section = "content"
        elif line.strip() == "[HASHTAGS]":
            current_section = "hashtags"
        elif line.strip() == "[SOURCES]":
            current_section = "sources"
        else:
            if current_section == "content":
                content  += line + "\n"
            elif current_section == "hashtags":
                hashtags += line + "\n"
            elif current_section == "sources":
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
    result   = executor.invoke({"input": user_prompt})
    output   = parse_agent_output(result["output"])
    if output["content"].startswith("INVALID_INPUT"):
        output["rejected"] = True
        return output
    articles = _extract_articles_from_steps(result.get("intermediate_steps", []))
    output["articles"] = articles
    output["rejected"] = False
    return output