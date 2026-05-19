import json
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

VAGUE_EXACT = {
    "x", "hi", "hey", "test", "asdf", "qwerty", "idk", "idc", "nothing",
    "something", "anything", "whatever", "post", "post smth", "post sth",
    "post something", "im tired", "i'm tired", "i am tired", "help",
    "write something", "make a post", "do smth", "idk what",
}

VAGUE_PATTERNS = [
    re.compile(r"^post\s+(something|smth|sth|anything|whatever)\.?$", re.I),
    re.compile(r"^i\s*'?m\s+(tired|bored|lazy|done)\.?$", re.I),
    re.compile(r"^(just\s+)?(post|write|make)\s+(a\s+)?(post|something)\.?$", re.I),
    re.compile(r"^write\s+(smth|something|anything)\.?$", re.I),
]

_JUDGE_SYSTEM = """You decide if the user input is a legitimate request to create a marketing or social media post.

REJECT (valid=false):
- Random characters, keyboard smash, gibberish (e.g. "38grov1go8p", "183ygf83ge4bv")
- Symbols only or mostly symbols
- Vague non-topics ("post smth", "im tired", "write something")
- Unrelated chat, jokes, insults, or commands with no promotable subject
- Single meaningless tokens

ACCEPT (valid=true):
- A clear product, service, brand, campaign, or topic to promote
- May include instructions like "fetch news about X and write about FlowAI"

Reply with ONLY valid JSON:
{"valid": true or false, "reason": "one short sentence for the user"}"""


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

    words = re.findall(r"[a-zA-Z0-9']+", text)
    if not words:
        return False, "Please enter a clear topic to post about."

    if len(words) == 1 and _looks_like_gibberish(words[0]):
        return (
            False,
            "That doesn't look like a real topic. "
            "Try a product name, campaign theme, or audience (e.g. 'FlowAI productivity app').",
        )

    meaningful = [w for w in words if not _looks_like_gibberish(w) and len(w) >= 3]
    if not meaningful:
        return (
            False,
            "Couldn't find a clear topic in your input. "
            "Use specific words about what to promote.",
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
        SystemMessage(content=_JUDGE_SYSTEM),
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
    """Fast heuristics, then LLM relevance check. Returns (is_valid, user_message)."""
    ok, reason = _heuristic_validate(prompt)
    if not ok:
        return False, reason
    return _llm_validate(prompt)
