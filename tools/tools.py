from langchain_core.tools import tool
from services.logger import get_logger
from services.news    import fetch_news, format_for_prompt, format_sources
from services.trends  import fetch_google_trends, fetch_reddit_trends, format_trends_for_prompt

log = get_logger(__name__)


@tool
def brand_context_tool(query: str) -> str:
    """Retrieve brand guidelines, tone rules, and approved claims. Always call first."""
    log.debug(f"brand_context_tool — query='{query[:80]}'")
    from services.rag import retrieve_brand_context
    return retrieve_brand_context(query)


@tool
def news_tool(query: str) -> str:
    """Fetch recent news articles about a topic to ground content in real facts."""
    log.debug(f"news_tool — query='{query[:80]}'")
    articles = fetch_news(query)
    if not articles:
        return "No news articles found."
    return format_for_prompt(articles)


@tool
def news_sources_tool(query: str) -> str:
    """Get formatted source URLs for news articles fetched about a topic."""
    log.debug(f"news_sources_tool — query='{query[:80]}'")
    articles = fetch_news(query)
    if not articles:
        return "No sources found."
    return format_sources(articles)


@tool
def reddit_tool(query: str) -> str:
    """Search Reddit for hot posts and community discussion about a topic.
    Use this to find what real people are saying, asking, and feeling about a subject."""
    log.debug(f"reddit_tool called: '{query[:60]}'")
    posts = fetch_reddit_trends(query)
    if not posts:
        log.warning("reddit_tool: no Reddit posts found")
        return "No Reddit posts found."
    log.debug(f"reddit_tool: {len(posts)} posts returned")
    return format_trends_for_prompt([], posts)


@tool
def generate_content_tool(prompt_with_context: str) -> str:
    """Write a social media marketing post from a topic and any research context provided."""
    log.debug(f"generate_content_tool — context length {len(prompt_with_context)}")
    from services.ai import generate_content
    return generate_content(prompt_with_context)


@tool
def generate_hashtags_tool(topic: str) -> str:
    """Generate relevant hashtags for a social media post topic."""
    log.debug(f"generate_hashtags_tool — topic='{topic[:80]}'")
    from services.ai import generate_hashtags
    return " ".join(generate_hashtags(topic))