"""Agent Builder API — workflow CRUD, tool/MCP registry, and execution."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.executor import execute_workflow
from engine.nodes import NODE_TYPE_SCHEMA
from engine.registry import (
    add_mcp, add_tool, list_mcps, list_tools, remove_mcp, remove_tool,
)

WORKFLOWS_DIR = Path(__file__).parent.parent / "data" / "workflows"
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api")


# ── Workflow CRUD ──────────────────────────────────────────────────────────────

class WorkflowBody(BaseModel):
    name: str
    nodes: list[dict]
    edges: list[dict]
    drawflow: dict | None = None  # raw Drawflow export stored alongside engine format


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
    prompt: str


@router.post("/workflows/{wf_id}/run")
async def run_workflow(wf_id: str, body: RunBody):
    path = WORKFLOWS_DIR / f"{wf_id}.json"
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    workflow = json.loads(path.read_text())
    if not workflow.get("nodes"):
        raise HTTPException(400, "Workflow has no nodes")
    try:
        return await execute_workflow(workflow, body.prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Tool Registry ──────────────────────────────────────────────────────────────

class ToolBody(BaseModel):
    name: str
    kind: str           # "http" | "tavily" | "serpapi"
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


# ── MCP Registry ───────────────────────────────────────────────────────────────

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


# ── Node Type Catalog ──────────────────────────────────────────────────────────

@router.get("/node-types")
def node_types():
    return NODE_TYPE_SCHEMA
