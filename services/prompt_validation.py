import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_COMPANY_DATA_PATH = Path(__file__).resolve().parent.parent / "company_data.json"

# User must show they want a FlowAI (or company) marketing post — not random topics
_MARKETING_INTENT_RE = re.compile(
    r"(?:"
    r"flow\s*ai|flowai|our\s+(?:app|product|brand|company|tool|platform|service)|"
    r"this\s+(?:app|product|tool|platform)|"
    r"(?:write|create|make|draft|generate|post|promot|market|announc|launch)|"
    r"(?:campaign|marketing\s+post|social\s+media|linkedin|twitter|discord)|"
    r"(?:productivity|automation|workflow|saas|b2b|professional)|"
    r"(?:target\s+audience|key\s+benefit|sign\s*ups?|free\s+trial)|"
    r"(?:news\s+about|trends?\s+about|articles?\s+about|scan\s+recent)"
    r")",
    re.I,
)

# Tokens / phrases with no real marketing topic (English + common Korean low-effort)
VAGUE_EXACT = {
    "x", "hi", "hey", "test", "asdf", "qwerty", "idk", "idc", "nothing",
    "something", "anything", "whatever", "post", "post smth", "post sth",
    "post something", "im tired", "i'm tired", "i am tired", "help",
    "write something", "make a post", "do smth", "idk what",
    "ok", "yes", "no", "lol", "haha", "marketing", "social media",
    "글써줘", "아무거나", "아무말", "뭐든지", "알아서", "아무거나 써줘", "글 써줘",
}

VAGUE_PATTERNS = [
    re.compile(r"^post\s+(something|smth|sth|anything|whatever)\.?$", re.I),
    re.compile(r"^i\s*'?m\s+(tired|bored|lazy|done)\.?$", re.I),
    re.compile(r"^(just\s+)?(post|write|make)\s+(a\s+)?(post|something)\.?$", re.I),
    re.compile(r"^write\s+(smth|something|anything)\.?$", re.I),
    re.compile(r"^(글|포스트|게시글)\s*(써줘|작성|만들어).?$"),
    re.compile(r"^아무(거나|말).?$"),
]

_WORD_RE = re.compile(r"[a-zA-Z0-9\uac00-\ud7a3']+")


def _has_hangul(text: str) -> bool:
    return any("\uac00" <= c <= "\ud7a3" for c in text)

_GENERIC_WORDS = frozenset({
    "post", "write", "make", "about", "something", "anything", "help", "please",
    "the", "a", "an", "our", "my", "create", "generate", "marketing", "content",
    "글", "써", "써줘", "작성", "만들어", "해줘", "아무거나", "뭐든지",
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

REJECT (valid=false):
- Gibberish, keyboard smash, symbols-only
- Vague or low-effort ("post something", "im tired", "help")
- Unrelated topics with no link to this brand (e.g. "banana milk", "cats", "weather today", "pizza recipe")
- Random noun phrases that are not a marketing brief for FlowAI / the product above
- Jokes, chit-chat, insults, or homework-style questions
- User names a consumer product/food/hobby unless they clearly tie it to promoting FlowAI

ACCEPT (valid=true):
- Explicitly about promoting FlowAI, its product, features, audience, or a defined campaign
- Clear instructions to write a marketing post about the company (may mention news/trends + FlowAI)
- Specific campaign angle for this B2B productivity brand (e.g. morning routine campaign, Q2 launch)

Reply with ONLY valid JSON:
{{"valid": true or false, "reason": "one short sentence for the user"}}"""


def _has_marketing_intent(prompt: str) -> bool:
    return bool(_MARKETING_INTENT_RE.search(prompt))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_alphanumeric_mash(token: str) -> bool:
    """Detect random letter+digit strings like 38grov1go8p or 183ygf83ge4bv."""
    if len(token) < 5:
        return False
    digits = sum(c.isdigit() for c in token)
    letters = sum(c.isalpha() for c in token)
    if digits == 0 or letters == 0:
        return False
    if token[0].isdigit():
        return True
    if digits / len(token) >= 0.2:
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

    if vowels == 0 and len(letters) >= 4:
        return True
    if len(letters) >= 5 and vowel_ratio < 0.2:
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", "".join(letters), re.I):
        return True

    return False


def _is_vague(normalized: str) -> bool:
    if normalized in VAGUE_EXACT:
        return True
    return any(p.match(normalized) for p in VAGUE_PATTERNS)


def _extract_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _is_low_effort(normalized: str, meaningful: list[str]) -> bool:
    """Careless / generic requests with no promotable subject."""
    if normalized in VAGUE_EXACT:
        return True
    if len(meaningful) < 2 and len(normalized) < 30:
        return True
    if meaningful and all(w.lower() in _GENERIC_WORDS for w in meaningful):
        return True
    return False


def _is_off_platform_topic(normalized: str, meaningful: list[str], prompt: str) -> bool:
    """
    Short prompts with no marketing/brand intent (e.g. 'banana milk') — reject locally, no agent.
    """
    if _has_marketing_intent(prompt):
        return False
    if len(meaningful) <= 3 and len(normalized) < 55:
        return True
    return False


def _heuristic_clear_accept(prompt: str, meaningful: list[str]) -> bool:
    """Only skip LLM judge for long, clearly on-brand briefs."""
    if not _has_marketing_intent(prompt):
        return False
    joined = " ".join(meaningful).lower()
    if "flowai" in joined or "flow ai" in joined:
        return len(prompt.strip()) >= 30
    if len(meaningful) >= 5 and len(prompt.strip()) >= 50:
        return True
    return False


def _heuristic_validate(prompt: str) -> tuple[bool, str]:
    text = prompt.strip()
    if not text:
        return False, "No input provided."

    normalized = _normalize(text)

    if _is_vague(normalized):
        return (
            False,
            "That's too vague for a marketing post. "
            "Describe a product, service, or topic (e.g. 'FlowAI for busy professionals').",
        )

    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return (
            False,
            "Please use real words — not only symbols or numbers.",
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
            "Try a product name, campaign theme, or audience (e.g. 'FlowAI productivity app').",
        )

    meaningful = [
        w for w in words
        if not _looks_like_gibberish(w) and (len(w) >= 3 or _has_hangul(w))
    ]
    if not meaningful:
        return (
            False,
            "Couldn't find a clear topic in your input. "
            "Use specific words about what to promote.",
        )

    if _is_low_effort(normalized, meaningful):
        return (
            False,
            "That's too vague for a marketing post. "
            "Name a product, service, or topic to promote (e.g. 'FlowAI for busy teams').",
        )

    if _is_off_platform_topic(normalized, meaningful, text):
        return (
            False,
            "This agent only creates posts for FlowAI. "
            "Describe a campaign, feature, or angle for FlowAI (e.g. 'Write about FlowAI saving 2 hours a day').",
        )

    if len(normalized) < 10 and len(meaningful) < 2:
        return (
            False,
            "Please add a bit more detail (at least ~10 characters or two clear words).",
        )

    gibberish_count = sum(1 for w in words if _looks_like_gibberish(w))
    if gibberish_count == len(words):
        return (
            False,
            "Input looks like random characters. Describe your marketing topic clearly.",
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
            "This doesn't look like a marketing topic. Describe what you want to promote.",
        )
    except json.JSONDecodeError:
        # If the judge fails to parse, fail closed — do not generate
        return (
            False,
            "Could not validate your topic. Please describe a clear product or campaign to promote.",
        )


def validate_user_prompt(prompt: str) -> tuple[bool, str]:
    """
    Analyze prompt before any agent/verification LLM runs.
    Heuristics first (no tokens); LLM judge only for short borderline cases.
    """
    ok, reason = _heuristic_validate(prompt)
    if not ok:
        return False, reason

    words = _extract_words(prompt.strip())
    meaningful = [
        w for w in words
        if not _looks_like_gibberish(w) and (len(w) >= 3 or _has_hangul(w))
    ]
    if _heuristic_clear_accept(prompt, meaningful):
        return True, ""

    return _llm_validate(prompt)
