import json
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_WEAK_ONLY_RE = re.compile(
    r"^(?:post|write|make|help|discord|slack|content|"
    r"something|anything|smth|sth|tired|bored|done|ok|yes|no|test)\s*"
    r"(?:post|write|make|discord|slack|smth|sth|something)?\s*\.?$",
    re.I,
)

VAGUE_EXACT = {
    "x", "hi", "hey", "test", "asdf", "qwerty", "idk", "idc", "nothing",
    "something", "anything", "whatever", "post", "post smth", "post sth",
    "post something", "im tired", "i'm tired", "i am tired", "help",
    "write something", "make a post", "do smth", "idk what",
    "ok", "yes", "no", "lol", "haha", "marketing", "social media",
    "discord", "slack", "smth", "sth",
    "글써줘", "아무거나", "아무말", "뭐든지", "알아서", "아무거나 써줘", "글 써줘",
}

VAGUE_PATTERNS = [
    re.compile(r"^post\s+(something|smth|sth|anything|whatever)\.?$", re.I),
    re.compile(r"^i\s*'?m\s+(tired|bored|lazy|done)\.?$", re.I),
    re.compile(r"^(just\s+)?(post|write|make)\s+(a\s+)?(post|something)\.?$", re.I),
    re.compile(r"^write\s+(smth|something|anything)\.?$", re.I),
    re.compile(r"^(discord|slack)\s*\.?$", re.I),
    re.compile(r"^(im\s+)?tired[,.]?\s*(post|write)?", re.I),
    re.compile(r"^(글|포스트|게시글)\s*(써줘|작성|만들어).?$"),
    re.compile(r"^아무(거나|말).?$"),
]

_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣']+")

_GENERIC_WORDS = frozenset({
    "post", "write", "make", "about", "something", "anything", "help", "please",
    "the", "a", "an", "our", "my", "create", "generate", "content",
    "discord", "slack", "gmail", "email", "smth", "sth", "tired", "bored",
    "done", "just", "output", "it", "to", "and", "then", "via",
    "글", "써", "써줘", "작성", "만들어", "해줘", "아무거나", "뭐든지",
})

_PLATFORM_WORDS = frozenset({
    "discord", "slack", "gmail", "email", "linkedin", "twitter", "instagram",
})


def _judge_system_prompt() -> str:
    return """\
You validate prompts for a general-purpose AI agent builder.

REJECT (valid=false) — only for:
- Gibberish, keyboard smash, or symbols-only input (e.g. "8uuq1ub", "////////")
- Careless / low-effort input: just a platform name, "im tired", "post smth"
- Completely empty or whitespace-only input

ACCEPT (valid=true) — for any reasonable task:
- Any specific question, task description, or request in plain language
- Any topic is fine: research, writing, coding, analysis, automation, etc.

Reply with ONLY valid JSON:
{"valid": true or false, "reason": "one short sentence for the user"}"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_hangul(text: str) -> bool:
    return any("가" <= c <= "힣" for c in text)


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
    vowels      = sum(1 for c in letters if c in "aeiou")
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


def _is_careless_prompt(normalized: str, meaningful: list[str]) -> bool:
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


def _heuristic_validate(prompt: str) -> tuple[bool, str]:
    text = prompt.strip()
    if not text:
        return False, "No input provided."

    normalized = _normalize(text)

    if _is_vague(normalized):
        return False, "That's too vague. Please describe what you'd like the agent to do."

    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False, "Please use real words — not only symbols or numbers."

    if re.fullmatch(r"[\W_]+", text.replace(" ", "")):
        return False, "Input looks like random symbols. Describe your task in plain language."

    non_word = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) >= 3 and non_word / len(text) > 0.45:
        return False, "Too many special characters. Describe what you want in plain language."

    words = _extract_words(text)
    if not words:
        return False, "Please enter a clear task or question."

    if len(words) == 1 and _looks_like_gibberish(words[0]):
        return False, "That doesn't look like a real task. Try describing what you want the agent to do."

    meaningful = [
        w for w in words
        if not _looks_like_gibberish(w) and (len(w) >= 3 or _has_hangul(w))
    ]
    if not meaningful:
        return False, "Couldn't find a clear task. Use specific words about what you want done."

    if _is_careless_prompt(normalized, meaningful):
        return False, "Please describe a specific task — not just a platform name or generic word."

    gibberish_count = sum(1 for w in words if _looks_like_gibberish(w))
    if gibberish_count == len(words):
        return False, "Input looks like random characters. Describe your task clearly."

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
        return False, result.get("reason", "Please describe a specific task for the agent.")
    except json.JSONDecodeError:
        return True, ""  # fail open for generic use


def validate_user_prompt(prompt: str) -> tuple[bool, str]:
    """
    Run before any agent work. Heuristics reject obvious bad input (gibberish,
    empty, careless); LLM judge handles edge cases.
    Accepts any substantive task — topic is not restricted.
    """
    ok, reason = _heuristic_validate(prompt)
    if not ok:
        return False, reason

    if len(prompt.strip()) >= 20:
        return True, ""

    return _llm_validate(prompt)
