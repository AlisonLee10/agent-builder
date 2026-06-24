"""Execute a workflow DAG definition."""
from __future__ import annotations

import uuid
from typing import Any

from engine.nodes import NODE_TYPE_MAP

# In-memory state store: run_id → {results, skipped, decisions}
# Keeps node outputs between approval submissions so LLM nodes don't re-run.
_STATE: dict[str, dict] = {}


def _topo_sort(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Kahn's algorithm — raises ValueError on cycles."""
    graph: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}

    for edge in edges:
        graph[edge["source"]].append(edge["target"])
        in_degree[edge["target"]] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for neighbor in graph[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise ValueError("Workflow contains a cycle — check your connections.")
    return order


async def execute_workflow(
    workflow: dict,
    user_input: str,
    approval_decisions: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Run workflow and return {"result": str, "trace": {node_id: output}}.

    On the first call run_id is None — a fresh run_id is created.
    When an approval node is reached without a decision, execution pauses:
    the current results and routing state are saved under run_id and returned
    to the caller so the frontend can re-submit with just the decision.
    On re-submission the caller passes the same run_id; cached node outputs
    are restored so LLM / Tool nodes are NOT re-run.
    """
    nodes_by_id = {n["id"]: n for n in workflow["nodes"]}
    edges       = workflow.get("edges", [])

    sources:    dict[str, list[str]]        = {}
    edge_ports: dict[tuple[str, str], str]  = {}
    for edge in edges:
        sources.setdefault(edge["target"], []).append(edge["source"])
        edge_ports[(edge["source"], edge["target"])] = edge.get("source_port", "output_1")

    order = _topo_sort(workflow["nodes"], edges)

    # ── Restore or initialise state ───────────────────────────────────────────
    if run_id and run_id in _STATE:
        saved     = _STATE[run_id]
        results   = dict(saved["results"])   # shallow copy of saved outputs
        skipped   = set(saved["skipped"])
        decisions = {**saved["decisions"], **(approval_decisions or {})}

        # Clear approval nodes that now have a decision so they re-run correctly
        for nid in list(decisions):
            if nid in results and isinstance(results[nid], dict) and results[nid].get("__approval_pending__"):
                del results[nid]
    else:
        run_id    = str(uuid.uuid4())
        results   = {}
        skipped   = set()
        decisions = dict(approval_decisions or {})

    # ── Main execution loop ───────────────────────────────────────────────────
    for node_id in order:
        # Skip nodes whose output is already cached from a previous partial run
        if node_id in results:
            continue

        node_def  = nodes_by_id[node_id]
        NodeClass = NODE_TYPE_MAP.get(node_def["type"])
        if not NodeClass:
            results[node_id] = {"error": f"Unknown node type: {node_def['type']}"}
            continue

        node = NodeClass(node_id, node_def.get("config", {}))

        if node_def["type"] == "input":
            effective  = user_input or node_def.get("config", {}).get("prompt", "")
            node_inputs = {"prompt": effective, "input": effective}
        else:
            src_ids    = sources.get(node_id, [])
            node_inputs = None

            for src_id in src_ids:
                if src_id in skipped:
                    continue
                raw = results.get(src_id, "")
                if isinstance(raw, dict) and "route_port" in raw:
                    active_port = raw["route_port"]
                    edge_port   = edge_ports.get((src_id, node_id), "output_1")
                    if edge_port != active_port:
                        continue
                    raw = raw.get("output", raw.get("result", str(raw)))
                elif isinstance(raw, dict):
                    raw = raw.get("output", raw.get("result", str(raw)))
                node_inputs = {"input": raw}
                break

            if node_inputs is None:
                if src_ids:
                    skipped.add(node_id)
                    continue
                else:
                    node_inputs = {"input": user_input}

        if node_def["type"] == "approval" and node_inputs is not None:
            node_inputs = {**node_inputs, "__decision__": decisions.get(node_id, "")}

        try:
            results[node_id] = await node.run(node_inputs)
        except Exception as exc:
            results[node_id] = {"error": str(exc)}
            continue

        r = results[node_id]
        if isinstance(r, dict) and r.get("__approval_pending__"):
            # Save state so the next call can resume without re-running anything
            _STATE[run_id] = {"results": dict(results), "skipped": set(skipped), "decisions": dict(decisions)}
            return {
                "status":   "pending_approval",
                "run_id":   run_id,
                "node_id":  r["node_id"],
                "preview":  r.get("preview", ""),
                "message":  r.get("message", ""),
            }

    # ── Workflow complete — discard saved state ───────────────────────────────
    _STATE.pop(run_id, None)

    executed_output = next(
        (n for n in reversed(workflow["nodes"])
         if n["type"] == "output" and n["id"] in results and n["id"] not in skipped),
        None,
    )
    if executed_output:
        final_id = executed_output["id"]
    else:
        final_id = next((nid for nid in reversed(order) if nid in results and nid not in skipped), None)

    final = results.get(final_id, "") if final_id else ""
    if isinstance(final, dict):
        final = final.get("output", final.get("result", str(final)))

    return {"result": final, "trace": results}
