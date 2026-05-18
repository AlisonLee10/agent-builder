from langchain_core.tools import tool
from services.news import fetch_news, format_for_prompt, format_sources
from services.trends import fetch_google_trends, fetch_reddit_trends, format_trends_for_prompt, format_trends_for_prompt
from services.discord import post_to_discord


@tool
def news_tool(query: str) -> str:
    """
    Fetch recent news articles about a topic.
    Use this when the user wants content grounded in real, recent facts.
    Returns a formatted list of articles titles and descriptions.
    """
    articles = fetch_news(query)
    if not articles:
        return "No news articles found."
    return format_for_prompt(articles)


@tool
def news_sources_tool(query: str) -> str:
    """
    Fetch news article source URLs about a topic.
    Use this to get the source links to append at the end of the post.
    Returns a formatted list of articles titles and URLs.
    """
    articles = fetch_news(query)
    if not articles:
        return "No sources found."
    return format_sources(articles)


@tool
def trends_tool(query: str) -> str:
    """
    Scan Google and Reddit for what is currently trending about a topic.
    Use this when the user wants content that reflects recent trends and discussions.
    Return Google search snippets and hot Reddit post titles.
    """
    google = fetch_google_trends(query)
    reddit = fetch_reddit_trends(query)
    if not google and not reddit:
        return "No trend signals found."
    return format_trends_for_prompt(google, reddit)


@tool
def generate_content_tool(prompt_with_context: str) -> str:
    """
    Write a social media marketing post.
    Pass the user's topic plus any research context (news, trends) as one combined input.
    Returns the generated post copy without hashtags.
    """
    from services.ai import generate_content
    return generate_content(prompt_with_context)


@tool
def generate_hashtags_tool(topic: str) -> str:
    """
    Generate relevant hashtags for a social media post.
    Pass the original user topic or prompt.
    Returns a space-separated string of hashtags.
    """
    from services.ai import generate_hashtags
    hashtags = generate_hashtags(topic)
    return " ".join(hashtags)


@tool
def post_to_discord_tool(content: str) -> str:
    """
    Post the final content to Discord.
    Only call this after the user has approved the post.
    REturns confirmation of success or failure.
    """
    success = post_to_discord(content)
    return "O Posted to Discord successfully." if success else "X Failed to post to Discord."