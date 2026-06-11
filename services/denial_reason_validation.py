"""
Validate user-provided denial feedback before saving rejected campaigns.
Rejects gibberish, vague complaints, and prompt-like text (new briefs).
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from services.prompt_validation import (
    _STRONG_MARKETING_RE,
    _extract_words,
    _has_hangul,
    _looks_like_gibberish,
    _normalize,
)

_judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_DENIAL_VAGUE_EXACT = {
    "bad", "not good", "no good", "this is bad", "not great", "hmmm", "hmm", "hm",
    "meh", "nope", "nah", "wrong", "sucks", "terrible", "horrible", "awful",
    "idk", "idc", "don't like", "dont like", "don't like it", "dont like it",
    "not like", "not like it", "no", "reject", "deny", "denied", "trash", "garbage",
    "useless", "worst", "fail", "failed", "yuck", "bleh", "ugh", "lol", "lmao",
    "this sucks", "this is wrong", "not ok", "not okay", "worse", "poor",
    "별로", "안좋아", "싫어", "별로야", "안 좋아",
}

_DENIAL_VAGUE_PATTERNS = [
    re.compile(r"^this is (bad|wrong|not good|terrible|awful|garbage)\.?$", re.I),
    re.compile(r"^not good\.?$", re.I),
    re.compile(r"^too bad\.?$", re.I),
    re.compile(r"^i don'?t like (it|this)\.?$", re.I),
    re.compile(r"^don'?t (like|want) (it|this)\.?$", re.I),
    re.compile(r"^no(t)? (good|great|ok|okay)\.?$", re.I),
    re.compile(r"^h+m+\.?$", re.I),
    re.compile(r"^(it'?s|its) (bad|wrong|terrible)\.?$", re.I),
    re.compile(r"^(just\s+)?(bad|wrong|no)\.?$", re.I),
]

_CAMPAIGN_BRIEF_RE = re.compile(
    r"(?:"
    r"(?:write|create|draft|generate|make)\s+(?:a\s+)?(?:marketing\s+)?(?:social\s+)?post|"
    r"marketing\s+post\s+about|"
    r"post\s+(?:to|on)\s+(?:discord|slack|gmail)|"
    r"scan\s+(?:recent\s+)?(?:news|trends?)|"
    r"news\s+articles?\s+about|"
    r"promot(?:e|ing)\s+(?:flow|our|the\s+product)"
    r")",
    re.I,
)

_CRITIQUE_RE = re.compile(
    r"(?:"
    r"\btoo\s+\w+|not\s+\w+ enough|missing|lacks?|without\s+|needs?\s+more|"
    r"should\s+(?:be|sound|include|mention)|doesn'?t\s+(?:mention|include|sound)|"
    r"\b(?:tone|salesy|pushy|generic|vague|formal|casual|professional|convincing)\b|"
    r"\b(?:benefit|feature|detail|hashtag|emoji|headline|opening|cta|audience)\b|"
    r"\b(?:short|long|repetitive|boring|clickbait|off[- ]brand|inaccurate|misleading)\b|"
    r"\b(?:grammar|spelling|factual|claim|source|statistic|call[- ]to[- ]action)\b|"
    r"wrong\s+(?:tone|angle|focus|facts?)|more\s+(?:detail|context|specific)"
    r")",
    re.I,
)

_GENERIC_NEGATIVE = frozenset({
    "bad", "good", "great", "wrong", "right", "nice", "hate", "like", "love",
    "sucks", "terrible", "awful", "horrible", "meh", "hmm", "hmmm", "no", "yes",
    "nope", "nah", "trash", "garbage", "worst", "fail", "poor", "worse",
})


def _judge_system_prompt() -> str:
    return """You validate feedback when a user REJECTS a generated marketing post draft.

ACCEPT (valid=true) only if the text is specific critique of the draft, e.g.:
- tone problems (too salesy, too casual, not convincing)
- content gaps (missing benefits, wrong focus, needs more detail on X)
- format issues (too long, bad hashtags, weak opening)

REJECT (valid=false) for:
- Keyboard smash / gibberish / symbols only
- Vague disapproval with no actionable detail ("bad", "not good", "hmmm", "I don't like it")
- A new marketing brief or prompt ("Write a post about...", "Scan news and post to Discord")

Reply with ONLY JSON:
{"valid": true or false, "reason": "one short sentence for the user"}"""


def _is_denial_vague(normalized: str) -> bool:
    if normalized in _DENIAL_VAGUE_EXACT:
        return True
    return any(p.match(normalized) for p in _DENIAL_VAGUE_PATTERNS)


def _is_prompt_like_reason(text: str) -> bool:
    if _STRONG_MARKETING_RE.search(text):
        return True
    if _CAMPAIGN_BRIEF_RE.search(text):
        return True
    return False


def _too_similar_to_campaign(reason: str, campaign_prompt: str) -> bool:
    if not campaign_prompt.strip():
        return False
    a = _normalize(reason)
    b = _normalize(campaign_prompt)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 15 and (a in b or b in a):
        return True
    ra = set(_extract_words(a))
    pb = set(_extract_words(b))
    if len(ra) >= 4 and len(pb) >= 4:
        overlap = len(ra & pb) / max(len(ra), 1)
        if overlap >= 0.72:
            return True
    return False


def _has_specific_critique(text: str, meaningful: list[str]) -> bool:
    if _CRITIQUE_RE.search(text):
        return True
    substantive = [
        w.lower() for w in meaningful
        if w.lower() not in _GENERIC_NEGATIVE and len(w) >= 4
    ]
    return len(substantive) >= 2


def _heuristic_validate(reason: str, campaign_prompt: str = "") -> tuple[bool, str]:
    text = reason.strip()
    if not text:
        return False, "Please explain why you are denying this post."

    normalized = _normalize(text)

    if _is_denial_vague(normalized):
        return (
            False,
            "That's too vague. Say what to fix (e.g. 'Too salesy — add concrete benefits, not hype').",
        )

    if _is_prompt_like_reason(text):
        return (
            False,
            "That looks like a new post request, not feedback on this draft. "
            "Describe what is wrong with the generated post.",
        )

    if campaign_prompt and _too_similar_to_campaign(text, campaign_prompt):
        return (
            False,
            "Your reason looks like the original prompt, not feedback on the draft. "
            "Explain what you dislike about the generated post.",
        )

    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return (
            False,
            "Please use real words — explain what should change in the draft.",
        )

    if re.fullmatch(r"[\W_]+", text.replace(" ", "")):
        return (
            False,
            "Input looks like random symbols. Describe what is wrong with the post.",
        )

    non_word = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) >= 3 and non_word / len(text) > 0.45:
        return (
            False,
            "Too many special characters. Explain the issue in plain language.",
        )

    words = _extract_words(text)
    if not words:
        return False, "Please describe what is wrong with this draft."

    if len(words) == 1 and _looks_like_gibberish(words[0]):
        return (
            False,
            "That doesn't look like real feedback. "
            "Say what to change (tone, missing details, too salesy, etc.).",
        )

    meaningful = [
        w for w in words
        if not _looks_like_gibberish(w) and (len(w) >= 3 or _has_hangul(w))
    ]
    if not meaningful:
        return (
            False,
            "Couldn't read clear feedback. Use specific words about what to fix.",
        )

    gibberish_count = sum(1 for w in words if _looks_like_gibberish(w))
    if gibberish_count == len(words):
        return (
            False,
            "Input looks like random characters. Explain what should change in the post.",
        )

    if not _has_specific_critique(text, meaningful):
        if len(normalized) < 18:
            return (
                False,
                "Please be more specific — what is wrong with tone, content, or format?",
            )
        lower_words = [w.lower() for w in meaningful]
        if all(w in _GENERIC_NEGATIVE for w in lower_words):
            return (
                False,
                "That's too vague. Name what to fix (e.g. 'Too salesy' or 'Missing product benefits').",
            )

    return True, ""


def _llm_validate(reason: str, campaign_prompt: str = "") -> tuple[bool, str]:
    context = reason.strip()
    if campaign_prompt.strip():
        context = (
            f"Original campaign prompt (user should NOT repeat this as denial reason):\n"
            f"{campaign_prompt.strip()[:500]}\n\n"
            f"User denial feedback:\n{reason.strip()}"
        )
    response = _judge_llm.invoke([
        SystemMessage(content=_judge_system_prompt()),
        HumanMessage(content=context),
    ])
    raw = response.text.strip().replace("```json", "").replace("```", "")
    try:
        result = json.loads(raw)
        if result.get("valid"):
            return True, ""
        return False, result.get(
            "reason",
            "Please give specific feedback about what is wrong with this draft.",
        )
    except json.JSONDecodeError:
        return (
            False,
            "Could not validate your feedback. Describe what to change in the draft.",
        )


def validate_denial_reason(
    reason: str,
    *,
    campaign_prompt: str = "",
) -> tuple[bool, str]:
    """
    Run before saving a user-denied campaign.
    Heuristics reject obvious bad input; LLM judge for borderline cases (fail closed).
    """
    ok, msg = _heuristic_validate(reason, campaign_prompt)
    if not ok:
        return False, msg

    text = reason.strip()
    if _has_specific_critique(text, _extract_words(text)) and len(text) >= 24:
        return True, ""

    if _is_prompt_like_reason(text):
        return False, (
            "That looks like a new post request, not feedback on this draft."
        )

    return _llm_validate(reason, campaign_prompt)
