import os
import requests
from dotenv import load_dotenv
from services.logger import get_logger

log = get_logger(__name__)

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")


def fetch_google_trends(query: str, max_results: int = 5) -> list[dict]:
    log.debug(f"SerpAPI request: '{query}'")

    params = {
        "engine":  "google",
        "q":       f"{query} trends",
        "api_key": SERP_API_KEY,
        "num":     max_results,
    }
    response = requests.get("https://serpapi.com/search", params=params)

    if response.status_code != 200:
        log.warning(f"SerpAPI error {response.status_code}")
        return []

    results = response.json().get("organic_results", [])
    parsed = [
        {
            "title":   r.get("title", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in results[:max_results]
        if r.get("title")
    ]
    log.debug(f"SerpAPI returned {len(parsed)} results")
    return parsed


def fetch_reddit_trends(query: str, max_results: int = 5) -> list[dict]:
    log.debug(f"Reddit search: '{query}'")
    headers  = {"User-Agent": "marketing-agent/1.0"}
    url      = f"https://www.reddit.com/search.json?q={query}&sort=hot&limit={max_results}&type=link"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        log.warning(f"Reddit error {response.status_code}")
        return []

    posts = response.json().get("data", {}).get("children", [])
    parsed = [
        {
            "title":     p["data"].get("title", ""),
            "subreddit": p["data"].get("subreddit", ""),
        }
        for p in posts
        if p.get("data", {}).get("title")
    ]
    log.debug(f"Reddit returned {len(parsed)} posts")
    return parsed


def format_trends_for_prompt(google: list[dict], reddit: list[dict]) -> str:
    lines = []

    if google:
        lines.append("=== Google Trends ===")
        for i, r in enumerate(google, 1):
            lines.append(f"[{i}] {r['title']}\n    {r['snippet']}")

    if reddit:
        lines.append("\n=== Reddit Hot Posts ===")
        for i, p in enumerate(reddit, 1):
            lines.append(f"[{i}] r/{p['subreddit']}: {p['title']}")

    return "\n".join(lines)
