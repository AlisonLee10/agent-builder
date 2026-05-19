import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

verifier_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ── State ──────────────────────────────────────────────────────
class VerificationState(TypedDict):
    content:        str
    user_prompt:    str
    verdict:        str
    issues:         list[str]
    summary:        str
    revision_count: int
    max_revisions:  int


# ── Nodes ──────────────────────────────────────────────────────
def verify_node(state: VerificationState) -> dict:
    response = verifier_llm.invoke([
        SystemMessage(content="""You are a content safety and relevance reviewer.

First, check whether the ORIGINAL USER REQUEST is a legitimate marketing topic.
Reject if the request is gibberish, keyboard smash, symbols only, vague non-topics
("post smth", "im tired"), unrelated chat, or not asking to promote something specific.

Then evaluate the GENERATED POST for:
- Harmful, offensive, or inappropriate content
- False or unverified statistics
- Misleading claims
- Overly aggressive or spammy language
- Generic filler content that ignores a nonsense or irrelevant prompt

Reply ONLY with valid JSON, no markdown:
{
  "verdict": "approved" or "needs_revision" or "rejected",
  "issues": ["list of specific issues found"],
  "summary": "one sentence explanation"
}

If the user request is not a valid marketing topic, verdict must be "rejected".
If the content is clean and matches a valid request, return "approved" with empty issues."""),
        HumanMessage(content=(
            f"Original user request:\n{state['user_prompt']}\n\n"
            f"Generated post:\n{state['content']}"
        )),
    ])

    raw = response.text.strip().replace("```json", "").replace("```", "")
    try:
        result = json.loads(raw)
        return {
            "verdict": result.get("verdict", "needs_revision"),
            "issues":  result.get("issues",  []),
            "summary": result.get("summary", ""),
        }
    except json.JSONDecodeError:
        return {
            "verdict": "needs_revision",
            "issues":  ["Could not parse verification result"],
            "summary": "Parse error — defaulting to needs_revision",
        }


def revise_node(state: VerificationState) -> dict:
    issues_text = "\n".join(f"- {issue}" for issue in state["issues"])

    response = verifier_llm.invoke([
        SystemMessage(content=(
            "You are a copy editor. Fix the issues in the post "
            "while keeping the same message, tone, and length. "
            "Return ONLY the revised post text — no labels, no explanation."
        )),
        HumanMessage(content=(
            f"Original post:\n{state['content']}\n\n"
            f"Issues to fix:\n{issues_text}"
        )),
    ])

    return {
        "content":        response.text.strip(),
        "revision_count": state["revision_count"] + 1,
    }


# ── Routing ────────────────────────────────────────────────────
def route_after_verify(state: VerificationState) -> str:
    if state["verdict"] == "approved":
        return "approved"
    if state["verdict"] == "rejected":
        return "rejected"
    if state["revision_count"] >= state["max_revisions"]:
        return "rejected"
    return "revise"


# ── Graph ──────────────────────────────────────────────────────
def build_verification_graph():
    graph = StateGraph(VerificationState)

    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)

    graph.add_edge(START, "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "approved": END,
            "rejected": END,
            "revise":   "revise",
        }
    )
    graph.add_edge("revise", "verify")

    return graph.compile()


verification_graph = build_verification_graph()


def run_verification(
    content: str,
    user_prompt: str = "",
    max_revisions: int = 3,
) -> dict:
    return verification_graph.invoke({
        "content":        content,
        "user_prompt":    user_prompt,
        "verdict":        "",
        "issues":         [],
        "summary":        "",
        "revision_count": 0,
        "max_revisions":  max_revisions,
    })