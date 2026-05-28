import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_COMPANY_DATA_PATH = Path(__file__).resolve().parent.parent / "company_data.json"

# Strong signals: clearly a FlowAI / company marketing brief
_STRONG_MARKETING_RE = re.compile(
    r"(?:"
    r"flow\s*ai|flowai|our\s+(?:company|product|brand|app|tool|platform|service)|"
    r"this\s+(?:company|product|app|tool|platform)|"
    r"marketing\s+post|social\s+media\s+(?:post|campaign)|"
    r"(?:write|create|draft|generate)\s+(?:a\s+)?(?:marketing\s+)?post\s+about|"
    r"promot(?:e|ing)\s+(?:flow|our|the\s+product)|"
    r"launch\s+(?:campaign|post)|"
    r"news\s+articles?\s+about|trends?\s+(?:on|about)|articles?\s+about|scan\s+(?:recent\s+)?trends?"
    r")",
    re.I,
)

# Weak words alone are NOT enough (post, discord, write, etc.)
_WEAK_ONLY_RE = re.compile(
    r"^(?:post|write|make|help|discord|slack|gmail|email|marketing|content|"
    r"something|anything|smth|sth|tired|bored|done|ok|yes|no|test)\s*"
    r"(?:post|write|make|discord|slack|gmail|smth|sth|something)?\s*\.?$",
    re.I,
)

_UNRELATED_TOPIC_RE = re.compile(
    r"(?:"
    r"\b(?:cookie|cake|brownie|recipe|cooking|bake|baking|pizza|banana|milk)\b|"
    r"\b(?:whale|dolphin|documentary|netflix|movie|film)\s+(?:about|on|recommend)?\b|"
    r"recommend\s+(?:me\s+)?a\s+\w+\s+documentary|"
    r"how\s+(?:big|large|hot)\s+is\s+(?:the\s+)?(?:sun|moon|earth)|"
    r"\b(?:weather|forecast|temperature)\s+(?:today|tomorrow)?\b|"
    r"\b(?:homework|essay|math|exam)\b|"
    r"what\s+is\s+\d+\s*\+\s*\d+"
    r")",
    re.I,
)

VAGUE_EXACT = {
    "x", "hi", "hey", "test", "asdf", "qwerty", "idk", "idc", "nothing",
    "something", "anything", "whatever", "post", "post smth", "post sth",
    "post something", "im tired", "i'm tired", "i am tired", "help",
    "write something", "make a post", "do smth", "idk what",
    "ok", "yes", "no", "lol", "haha", "marketing", "social media",
    "discord", "slack", "gmail", "email", "smth", "sth",
    "글써줘", "아무거나", "아무말", "뭐든지", "알아서", "아무거나 써줘", "글 써줘",
}

VAGUE_PATTERNS = [
    re.compile(r"^post\s+(something|smth|sth|anything|whatever)\.?$", re.I),
    re.compile(r"^i\s*'?m\s+(tired|bored|lazy|done)\.?$", re.I),
    re.compile(r"^(just\s+)?(post|write|make)\s+(a\s+)?(post|something)\.?$", re.I),
    re.compile(r"^write\s+(smth|something|anything)\.?$", re.I),
    re.compile(r"^(discord|slack|gmail|email)\s*\.?$", re.I),
    re.compile(r"^(im\s+)?tired[,.]?\s*(post|write)?", re.I),
    re.compile(r"^(글|포스트|게시글)\s*(써줘|작성|만들어).?$"),
    re.compile(r"^아무(거나|말).?$"),
]

_WORD_RE = re.compile(r"[a-zA-Z0-9\uac00-\ud7a3']+")

_GENERIC_WORDS = frozenset({
    "post", "write", "make", "about", "something", "anything", "help", "please",
    "the", "a", "an", "our", "my", "create", "generate", "marketing", "content",
    "discord", "slack", "gmail", "email", "smth", "sth", "tired", "bored",
    "done", "just", "output", "it", "to", "and", "then", "via",
    "글", "써", "써줘", "작성", "만들어", "해줘", "아무거나", "뭐든지",
})

_PLATFORM_WORDS = frozenset({
    "discord", "slack", "gmail", "email", "linkedin", "twitter", "instagram",
})


def _brand_context_for_judge() -> str:
    try:
        with open(_COMPANY_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "Brand: FlowAI — AI productivity software for busy professionals."

    brand = data.get("brand", {})
    product = (data.get("products") or [{}])[0]
    audience = data.get("target_audience", {})
    return (
        f"Brand: {brand.get('name', 'FlowAI')}. "
        f"Category: {brand.get('category', 'AI productivity software')}. "
        f"Mission: {brand.get('mission', '')}. "
        f"Product: {product.get('name', '')} — {product.get('description', '')}. "
        f"Audience: {audience.get('primary', '')}."
    )


def _judge_system_prompt() -> str:
    brand = _brand_context_for_judge()
    return f"""You gatekeep prompts for an internal marketing agent. It ONLY creates social posts for this company:

{brand}

REJECT (valid=false) — always reject:
- Gibberish, keyboard smash, symbols-only (e.g. "8uuq1ub", "////////")
- Careless / low-effort: "im tired", "post smth", "discord", "write something", only a platform name
- Unrelated topics: recipes, animals, documentaries, science trivia, weather, homework, jokes — unless clearly tied to promoting FlowAI
- No mention of the company/product and no clear marketing task for FlowAI

ACCEPT (valid=true) — only if:
- User wants a marketing/social post promoting FlowAI or "our company/product" with a clear angle
- OR explicit campaign brief (news/trends research + write post about FlowAI + optional platforms)

Reply with ONLY valid JSON:
{{"valid": true or false, "reason": "one short sentence for the user"}}"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_hangul(text: str) -> bool:
    return any("\uac00" <= c <= "\ud7a3" for c in text)


def _extract_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _is_alphanumeric_mash(token: str) -> bool:
    if len(token) < 4:
        return False
    digits  = sum(c.isdigit() for c in token)
    letters = sum(c.isalpha() for c in token)
    if digits and letters:
        if token[0].isdigit():
            return True
        if digits / len(token) >= 0.15:
            return True
    return False


def _looks_like_gibberish(token: str) -> bool:
    if _is_alphanumeric_mash(token):
        return True

    letters = [c.lower() for c in token if c.isalpha()]
    if len(letters) < 2:
        return True
    if len(letters) <= 2 and len(token) <= 3:
        return True

    vowels = sum(1 for c in letters if c in "aeiou")
    vowel_ratio = vowels / len(letters)

    if vowels == 0 and len(letters) >= 3:
        return True
    if len(letters) >= 5 and vowel_ratio < 0.2:
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", "".join(letters), re.I):
        return True

    return False


def _is_vague(normalized: str) -> bool:
    if normalized in VAGUE_EXACT:
        return True
    if _WEAK_ONLY_RE.match(normalized):
        return True
    return any(p.match(normalized) for p in VAGUE_PATTERNS)


def _is_unrelated_topic(prompt: str) -> bool:
    if _STRONG_MARKETING_RE.search(prompt):
        return False
    return bool(_UNRELATED_TOPIC_RE.search(prompt))


def _is_careless_prompt(normalized: str, meaningful: list[str]) -> bool:
    """Reject prompts with no real subject — only generic/platform words."""
    if not meaningful:
        return True
    lower = [w.lower() for w in meaningful]
    substantive = [
        w for w in lower
        if w not in _GENERIC_WORDS
        and w not in _PLATFORM_WORDS
        and len(w) >= 4
    ]
    if not substantive:
        return True
    if len(substantive) == 1 and substantive[0] in {"tired", "bored", "lazy", "done"}:
        return True
    return False


def _has_strong_marketing_intent(prompt: str) -> bool:
    return bool(_STRONG_MARKETING_RE.search(prompt))


def _heuristic_validate(prompt: str) -> tuple[bool, str]:
    text = prompt.strip()
    if not text:
        return False, "No input provided."

    normalized = _normalize(text)

    if _is_vague(normalized):
        return (
            False,
            "That's too vague for a marketing post. "
            "Describe what to promote (e.g. 'Write a FlowAI post for busy professionals, post to Discord').",
        )

    if _is_unrelated_topic(text):
        return (
            False,
            "This agent only creates marketing posts for FlowAI. "
            "Ask for a company campaign (e.g. 'Write a post about FlowAI for busy teams').",
        )

    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return (
            False,
            "Please use real words — not only symbols or numbers.",
        )

    if re.fullmatch(r"[\W_]+", text.replace(" ", "")):
        return (
            False,
            "Input looks like random symbols. Describe your marketing topic in plain language.",
        )

    non_word = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) >= 3 and non_word / len(text) > 0.45:
        return (
            False,
            "Too many special characters. Describe what you want to promote in plain language.",
        )

    words = _extract_words(text)
    if not words:
        return False, "Please enter a clear topic to post about."

    if len(words) == 1 and _looks_like_gibberish(words[0]):
        return (
            False,
            "That doesn't look like a real topic. "
            "Try a product or campaign theme (e.g. 'FlowAI productivity app').",
        )

    meaningful = [
        w for w in words
        if not _looks_like_gibberish(w) and (len(w) >= 3 or _has_hangul(w))
    ]
    if not meaningful:
        return (
            False,
            "Couldn't find a clear topic. Use specific words about what to promote.",
        )

    if _is_careless_prompt(normalized, meaningful):
        return (
            False,
            "That's too vague — name FlowAI, a feature, or campaign angle "
            "(not just 'post something' or a platform name).",
        )

    gibberish_count = sum(1 for w in words if _looks_like_gibberish(w))
    if gibberish_count == len(words):
        return (
            False,
            "Input looks like random characters. Describe your marketing topic clearly.",
        )

    if len(normalized) < 12 and len(meaningful) < 2:
        return (
            False,
            "Please add more detail (what to promote and optionally where to post).",
        )

    if not _has_strong_marketing_intent(text) and len(meaningful) < 4:
        return (
            False,
            "Please include a clear FlowAI marketing task "
            "(e.g. 'Write a post about FlowAI for busy professionals').",
        )

    return True, ""


def _llm_validate(prompt: str) -> tuple[bool, str]:
    response = _judge_llm.invoke([
        SystemMessage(content=_judge_system_prompt()),
        HumanMessage(content=prompt.strip()),
    ])
    raw = response.text.strip().replace("```json", "").replace("```", "")
    try:
        result = json.loads(raw)
        if result.get("valid"):
            return True, ""
        return False, result.get(
            "reason",
            "This doesn't look like a valid marketing brief for FlowAI.",
        )
    except json.JSONDecodeError:
        return (
            False,
            "Could not validate your prompt. Please describe a clear FlowAI marketing task.",
        )


def validate_user_prompt(prompt: str) -> tuple[bool, str]:
    """
    Run before any agent work. Heuristics reject obvious bad input;
    LLM judge runs for everything else (fail closed).
    """
    ok, reason = _heuristic_validate(prompt)
    if not ok:
        return False, reason

    if _has_strong_marketing_intent(prompt) and len(prompt.strip()) >= 40:
        return True, ""

    return _llm_validate(prompt)
