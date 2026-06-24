"""Agent Builder API — workflow CRUD, tool/MCP registry, execution, domains, templates."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.executor import execute_workflow
from engine.nodes import NODE_TYPE_SCHEMA
from engine.builtin_tools import BUILTIN_TOOLS, is_available
from engine.builtin_mcps import PRESET_MCPS
from engine.registry import (
    add_mcp, add_tool, list_mcps, list_tools, remove_mcp, remove_tool,
)

DATA_DIR      = Path(__file__).parent.parent / "data"
WORKFLOWS_DIR = DATA_DIR / "workflows"
TEMPLATES_DIR = DATA_DIR / "templates"
DOMAINS_PATH  = DATA_DIR / "domains_config.json"
RAG_PATH      = DATA_DIR / "rag_docs.json"

WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

# ── RAG example templates (shown in sidebar, not stored in rag_docs.json) ──────
RAG_EXAMPLES = [
    {
        "id": "example_flowai",
        "name": "FlowAI",
        "icon": "🏢",
        "type": "company",
        "description": "B2B productivity SaaS — brand identity, approved claims, target audience",
        "content": (
            "## Company: FlowAI\n\n"
            "**Tagline:** Work less. Flow more.\n"
            "**Mission:** Give professionals back 2 hours every day through intelligent automation.\n"
            "**Category:** AI productivity software\n"
            "**Website:** https://flowai.com\n\n"
            "### Approved Statistics\n"
            "Only use these verified figures — do not invent or round differently:\n"
            "- \"Saves an average of 2 hours per day\"\n"
            "- \"Used by 10,000+ professionals\"\n"
            "- \"Integrates with 50+ popular tools\"\n\n"
            "### Target Audience\n"
            "**Primary:** Busy professionals aged 25–40\n"
            "**Industries:** Tech, Marketing, Finance, Consulting\n"
            "**Pain points:** Repetitive tasks · Context switching · Email overload\n\n"
            "When writing for this audience, assume they:\n"
            "- Are already familiar with productivity tools (Notion, Slack, Zapier).\n"
            "- Are skeptical of AI hype — prove value with specifics, not superlatives.\n"
            "- Make purchasing decisions based on ROI and time savings, not feature lists.\n\n"
            "### Brand Voice\n"
            "Inspirational, direct, empowering. Peer-to-peer tone."
        ),
    }
]

router = APIRouter(prefix="/api")


# ── Workflow CRUD ──────────────────────────────────────────────────────────────

class WorkflowBody(BaseModel):
    name: str
    nodes: list[dict]
    edges: list[dict]
    drawflow: dict | None = None


@router.get("/workflows")
def list_workflows():
    result = []
    for f in sorted(WORKFLOWS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            result.append({"id": data["id"], "name": data["name"]})
        except Exception:
            continue
    return result


@router.post("/workflows", status_code=201)
def create_workflow(body: WorkflowBody):
    wf_id = str(uuid.uuid4())
    data = {"id": wf_id, **body.model_dump()}
    (WORKFLOWS_DIR / f"{wf_id}.json").write_text(json.dumps(data, indent=2))
    return data


@router.get("/workflows/{wf_id}")
def get_workflow(wf_id: str):
    path = WORKFLOWS_DIR / f"{wf_id}.json"
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    return json.loads(path.read_text())


@router.put("/workflows/{wf_id}")
def update_workflow(wf_id: str, body: WorkflowBody):
    path = WORKFLOWS_DIR / f"{wf_id}.json"
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    data = {"id": wf_id, **body.model_dump()}
    path.write_text(json.dumps(data, indent=2))
    return data


@router.delete("/workflows/{wf_id}")
def delete_workflow(wf_id: str):
    path = WORKFLOWS_DIR / f"{wf_id}.json"
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    path.unlink()
    return {"deleted": wf_id}


class RunBody(BaseModel):
    prompt: str = ""
    approval_decisions: dict[str, Any] = {}
    run_id: str = ""


import asyncio as _asyncio
import os as _os

_WORKFLOW_TIMEOUT = int(_os.getenv("WORKFLOW_TIMEOUT", "120"))


@router.post("/workflows/{wf_id}/run")
async def run_workflow(wf_id: str, body: RunBody):
    path = WORKFLOWS_DIR / f"{wf_id}.json"
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    workflow = json.loads(path.read_text())
    if not workflow.get("nodes"):
        raise HTTPException(400, "Workflow has no nodes")
    try:
        return await _asyncio.wait_for(
            execute_workflow(workflow, body.prompt, body.approval_decisions, run_id=body.run_id or None),
            timeout=_WORKFLOW_TIMEOUT,
        )
    except _asyncio.TimeoutError:
        raise HTTPException(504, f"Workflow timed out after {_WORKFLOW_TIMEOUT} seconds.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ── User Tool Registry ─────────────────────────────────────────────────────────

class ToolBody(BaseModel):
    name: str
    kind: str
    description: str = ""
    url: str = ""
    api_key: str = ""
    method: str = "POST"
    headers: dict = {}


@router.get("/tools")
def get_tools():
    return list_tools()


@router.post("/tools", status_code=201)
def register_tool(body: ToolBody):
    return add_tool(body.model_dump())


@router.delete("/tools/{tool_id}")
def delete_tool(tool_id: str):
    if not remove_tool(tool_id):
        raise HTTPException(404, "Tool not found")
    return {"deleted": tool_id}


# ── User MCP Registry ──────────────────────────────────────────────────────────

class MCPBody(BaseModel):
    name: str
    command: str
    args: list[str] = []
    env: dict[str, str] = {}
    description: str = ""


@router.get("/mcps")
def get_mcps():
    return list_mcps()


@router.post("/mcps", status_code=201)
def register_mcp(body: MCPBody):
    return add_mcp(body.model_dump())


@router.delete("/mcps/{mcp_id}")
def delete_mcp(mcp_id: str):
    if not remove_mcp(mcp_id):
        raise HTTPException(404, "MCP not found")
    return {"deleted": mcp_id}


# ── Built-in tools catalog ─────────────────────────────────────────────────────

@router.get("/builtin-tools")
def get_builtin_tools():
    return [
        {**t, "available": is_available(t["id"])}
        for t in BUILTIN_TOOLS
    ]


# ── Preset MCP catalog ─────────────────────────────────────────────────────────

@router.get("/preset-mcps")
def get_preset_mcps():
    return PRESET_MCPS


# ── Node Type Catalog ──────────────────────────────────────────────────────────

@router.get("/node-types")
def node_types():
    return NODE_TYPE_SCHEMA


# ── Domain Packages ────────────────────────────────────────────────────────────

def _load_domains() -> dict:
    if not DOMAINS_PATH.exists():
        return {"builtin": [], "user": []}
    try:
        return json.loads(DOMAINS_PATH.read_text())
    except Exception:
        return {"builtin": [], "user": []}


def _save_domains(data: dict) -> None:
    DOMAINS_PATH.write_text(json.dumps(data, indent=2))


@router.get("/domains")
def get_domains():
    data = _load_domains()
    return {"builtin": data.get("builtin", []), "user": data.get("user", [])}


class DomainBody(BaseModel):
    name: str
    icon: str = "📦"
    description: str = ""
    context: str = ""


@router.post("/domains", status_code=201)
def create_domain(body: DomainBody):
    data = _load_domains()
    domain = {"id": str(uuid.uuid4()), **body.model_dump()}
    data.setdefault("user", []).append(domain)
    _save_domains(data)
    return domain


@router.delete("/domains/{domain_id}")
def delete_domain(domain_id: str):
    data = _load_domains()
    before = len(data.get("user", []))
    data["user"] = [d for d in data.get("user", []) if d["id"] != domain_id]
    if len(data["user"]) == before:
        raise HTTPException(404, "Domain not found (built-in domains cannot be deleted)")
    _save_domains(data)
    return {"deleted": domain_id}


# ── Templates ──────────────────────────────────────────────────────────────────

@router.get("/templates")
def get_templates():
    if not TEMPLATES_DIR.exists():
        return []
    result = []
    for f in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            result.append({"id": data["id"], "name": data["name"], "description": data.get("description", "")})
        except Exception:
            continue
    return result


@router.get("/templates/{template_id}")
def get_template(template_id: str):
    path = TEMPLATES_DIR / f"{template_id}.json"
    if not path.exists():
        raise HTTPException(404, "Template not found")
    return json.loads(path.read_text())


# ── RAG Documents ──────────────────────────────────────────────────────────────

def _load_rag() -> dict:
    if not RAG_PATH.exists():
        return {"docs": []}
    try:
        return json.loads(RAG_PATH.read_text())
    except Exception:
        return {"docs": []}


def _save_rag(data: dict) -> None:
    RAG_PATH.write_text(json.dumps(data, indent=2))


@router.get("/rag")
def get_rag():
    """Return user docs and the static example templates."""
    data = _load_rag()
    return {"docs": data.get("docs", []), "examples": RAG_EXAMPLES}


class RagDocBody(BaseModel):
    name: str
    icon: str = "📄"
    type: str = "company"
    description: str = ""
    content: str


@router.post("/rag", status_code=201)
def create_rag_doc(body: RagDocBody):
    data = _load_rag()
    doc = {"id": str(uuid.uuid4()), **body.model_dump()}
    data.setdefault("docs", []).append(doc)
    _save_rag(data)
    return doc


@router.delete("/rag/{doc_id}")
def delete_rag_doc(doc_id: str):
    data = _load_rag()
    before = len(data.get("docs", []))
    data["docs"] = [d for d in data.get("docs", []) if d["id"] != doc_id]
    if len(data["docs"]) == before:
        raise HTTPException(404, "RAG document not found")
    _save_rag(data)
    return {"deleted": doc_id}


# ── API Keys ───────────────────────────────────────────────────────────────────

@router.get("/keys")
def get_keys():
    from engine.key_store import list_keys
    return list_keys()


class KeyBody(BaseModel):
    env: str
    value: str


@router.post("/keys")
def set_key(body: KeyBody):
    from engine.key_store import set_key as _set, KEY_REGISTRY
    if not any(k["env"] == body.env for k in KEY_REGISTRY):
        raise HTTPException(400, f"Unknown key: {body.env}")
    _set(body.env, body.value)
    return {"env": body.env, "is_set": True}


@router.delete("/keys/{env_name}")
def delete_key(env_name: str):
    from engine.key_store import delete_key as _delete
    if not _delete(env_name):
        raise HTTPException(404, "Key not found")
    return {"env": env_name, "is_set": False}
