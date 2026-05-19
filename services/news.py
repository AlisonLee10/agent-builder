import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

_last_fetched_articles: list[dict] = []

def fetch_news(query: str, max_articles: int = 5) -> list[dict]:
    global _last_fetched_articles
    params = {
        "q":        query,
        "pageSize": max_articles,
        "language": "en",
        "sortBy":   "publishedAt",
        "apiKey":   NEWS_API_KEY,
    }

    response = requests.get(NEWS_API_URL, params=params)

    if response.status_code != 200:
        print(f"  [NewsAPI] Error {response.status_code}: {response.text}")
        _last_fetched_articles = []
        return []

    articles = response.json().get("articles", [])
    result = [
        {
            "title":       a.get("title", ""),
            "description": a.get("description", ""),
            "url":         a.get("url", ""),
        }
        for a in articles
        if a.get("title") and a.get("description")
    ]

    _last_fetched_articles = result
    return result


def get_last_fetched_articles() -> list[dict]:
    return _last_fetched_articles


def format_for_prompt(articles: list[dict]) -> str:
    if not articles:
        return ""
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] {a['title']}\n    {a['description']}")
    return "\n\n".join(lines)

#1k
def format_sources(articles: list[dict]) -> str:
    if not articles:
        return ""
    lines = ["📰 Sources:"]
    for a in articles:
        lines.append(f"• {a.get('title', 'Article')} — {a.get('url', '')}")
    return "\n".join(lines)