import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_news(query: str, max_articles: int = 5) -> list[dict]:
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
        return []

    articles = response.json().get("articles", [])

    return [
        {
            "title":       a.get("title", ""),
            "description": a.get("description", ""),
            "url":         a.get("url", ""),
        }
        for a in articles
        if a.get("title") and a.get("description")
    ]


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
    lines = ["🗞️ Sources:"]
    for a in articles:
        title = a.get("title", "Article")
        url = a.get("url", "")
        lines.append(f"· {title} {url}")
    return "\n".join(lines)