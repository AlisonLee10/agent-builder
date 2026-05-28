import os
import requests
from dotenv import load_dotenv
from services.logger import get_logger

log = get_logger(__name__)

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")

_last_google_trends: list[dict] = []
_last_reddit_posts:  list[dict] = []


def fetch_google_trends(query: str, max_results: int = 5) -> list[dict]:
    global _last_google_trends
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
        _last_google_trends = []
        return []

    results = response.json().get("organic_results", [])
    parsed = [
        {
            "title":       r.get("title", ""),
            "description": r.get("snippet", ""),
            "url":         r.get("link", ""),
            "source_type": "google",
        }
        for r in results[:max_results]
        if r.get("title")
    ]
    log.debug(f"SerpAPI returned {len(parsed)} results")
    _last_google_trends = parsed
    return parsed


def fetch_reddit_trends(query: str, max_results: int = 5) -> list[dict]:
    global _last_reddit_posts
    log.debug(f"Reddit search: '{query}'")
    headers  = {"User-Agent": "marketing-agent/1.0"}
    url      = f"https://www.reddit.com/search.json?q={query}&sort=hot&limit={max_results}&type=link"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        log.warning(f"Reddit error {response.status_code}")
        _last_reddit_posts = []
        return []

    posts = response.json().get("data", {}).get("children", [])
    parsed = []
    for p in posts:
        data = p.get("data", {})
        title = data.get("title", "")
        if not title:
            continue
        permalink = data.get("permalink", "")
        url = f"https://www.reddit.com{permalink}" if permalink else data.get("url", "")
        parsed.append({
            "title":       title,
            "description": f"r/{data.get('subreddit', '')}",
            "url":         url,
            "source_type": "reddit",
        })
    log.debug(f"Reddit returned {len(parsed)} posts")
    _last_reddit_posts = parsed
    return parsed


def get_last_fetched_trends() -> list[dict]:
    """Return structured trend/reddit research from the most recent tool calls."""
    return list(_last_google_trends) + list(_last_reddit_posts)


def format_trends_for_prompt(google: list[dict], reddit: list[dict]) -> str:
    lines = []

    if google:
        lines.append("=== Google Trends ===")
        for i, r in enumerate(google, 1):
            snippet = r.get("description") or r.get("snippet", "")
            lines.append(f"[{i}] {r['title']}\n    {snippet}")

    if reddit:
        lines.append("\n=== Reddit Hot Posts ===")
        for i, p in enumerate(reddit, 1):
            sub = p.get("subreddit") or (p.get("description") or "").replace("r/", "")
            lines.append(f"[{i}] r/{sub}: {p['title']}")

    return "\n".join(lines)
