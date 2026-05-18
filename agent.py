from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from services.tools import (
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
    news_tool,
    news_sources_tool,
    trends_tool,
    generate_content_tool,
    generate_hashtags_tool,
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a marketing content agent. Given a user's topic, follow these steps in order:

    1. Decide if fetching recent news wouuld make the content more credible - if yes, call news_tool
    2. Decide if scanning current trends would make the content more timely - if yes, call trends_tool
    3. Call generate_content_tool with the topic and any context you gathered from steps 1 and 2
    4. Call generate_hashtags_tool with the original topicmessages
    5. If you called news_tool, call news_sources_tool to get the source LinkOutsideDestinationError
    
    Then return the final post in exactly this format - no extra commentary:

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
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


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
    result = executor.invoke({"input": user_prompt})
    return parse_agent_output(result["output"])