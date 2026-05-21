import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from services.campaign_memory import get_denied_examples

load_dotenv()

verifier_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)



class VerificationState(TypedDict):
    content:         str
    verdict:         str
    issues:          list[str]
    summary:         str
    revision_count:  int
    max_revisions:   int
    denied_examples: str # past failure
    approved_examples: str # past success


def verify_node(state: VerificationState) -> dict:
    # Inject past denial patterns if available
    denied_section = ""
    if state.get("denied_examples"):
        denied_section = (
            f"\n\n{state['denied_examples']}\n"
        )

    response = verifier_llm.invoke([
        SystemMessage(content=(
            "You are a content safety reviewer."
            f"{denied_section}"
            "\nEvaluate the social media post for:\n"
            "- Harmful, offensive, or inappropriate content\n"
            "- False or unverified statistics\n"
            "- Misleading claims\n"
            "- Overly aggressive or spammy language\n\n"
            "Reply ONLY with valid JSON, no markdown:\n"
            "{\n"
            '  "verdict": "approved" or "needs_revision" or "rejected",\n'
            '  "issues": ["specific issue 1", "specific issue 2"],\n'
            '  "summary": "one sentence explanation"\n'
            "}"
        )),
        HumanMessage(content=f"Review this post:\n\n{state['content']}"),
    ])

    text = response.content if isinstance(response.content, str) else str(response.content)
    raw = text.strip().replace("```json", "").replace("```", "")
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
    issues_text = "\n".join(f"- {i}" for i in state["issues"])

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

    revised = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "content":        revised.strip(),
        "revision_count": state["revision_count"] + 1,
    }


def route_after_verify(state: VerificationState) -> str:
    if state["verdict"] == "approved":
        return "approved"
    if state["verdict"] == "rejected":
        return "rejected"
    if state["revision_count"] >= state["max_revisions"]:
        return "rejected"
    return "revise"


def build_verification_graph():
    graph = StateGraph(VerificationState)
    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"approved": END, "rejected": END, "revise": "revise"},
    )
    graph.add_edge("revise", "verify")
    return graph.compile()


verification_graph = build_verification_graph()


def run_verification(content: str, max_revisions: int = 3) -> dict:
    from services.campaign_memory import (
        get_denied_examples,
        get_approved_examples_for_verification,
    )

    denied_ex = get_denied_examples(content, k=2)
    approved_ex = get_approved_examples_for_verification(content, k = 2)

    # terminal report
    found = []
    if approved_ex:
        found.append("approved references")
    if denied_ex:
        found.append("denied referneces")

    if found:
        print(f"  [Memory] Verifier calibrated with: {' + '.join(found)}")
    else:
        print("  [Memory] No refernece campaigns yet - using rules only")

    from services.progress import show_progress

    with show_progress("      Verifying content"):
        return verification_graph.invoke({
            "content": content,
            "verdict": "",
            "issues": [],
            "summary": "",
            "revision_count": 0,
            "max_revisions": max_revisions,
            "denied_examples": denied_ex,
            "approved_examples": approved_ex,
        })

"""
def run_verification(content: str, max_revisions: int = 3) -> dict:

    denied_ex = get_denied_examples(content, k=2)
    if denied_ex:
        print("  [Memory] Similar denied campaigns found — verifier is calibrated")

    initial_state: VerificationState = {
        "content":         content,
        "verdict":         "",
        "issues":          [],
        "revision_count":  0,
        "max_revisions":   max_revisions,
        "summary":         "",
        "denied_examples": denied_ex,
    }
    return verification_graph.invoke(initial_state)
    """