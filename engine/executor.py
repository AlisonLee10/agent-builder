"""Execute a workflow DAG definition."""
from __future__ import annotations

from typing import Any

from engine.nodes import NODE_TYPE_MAP


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
    workflow: dict, user_input: str
) -> dict[str, Any]:
    """
    Run workflow and return {"result": str, "trace": {node_id: output}}.
    Nodes execute in topological order; each node receives the output of
    its upstream neighbor as "input".
    """
    nodes_by_id = {n["id"]: n for n in workflow["nodes"]}
    edges = workflow.get("edges", [])

    # Upstream mapping: target_id → [source_ids]
    sources: dict[str, list[str]] = {}
    for edge in edges:
        sources.setdefault(edge["target"], []).append(edge["source"])

    order = _topo_sort(workflow["nodes"], edges)
    results: dict[str, Any] = {}

    for node_id in order:
        node_def = nodes_by_id[node_id]
        NodeClass = NODE_TYPE_MAP.get(node_def["type"])
        if not NodeClass:
            results[node_id] = {"error": f"Unknown node type: {node_def['type']}"}
            continue

        node = NodeClass(node_id, node_def.get("config", {}))

        if node_def["type"] == "input":
            # If caller sent empty string, fall back to the node's pre-filled prompt
            effective = user_input or node_def.get("config", {}).get("prompt", "")
            node_inputs = {"prompt": effective, "input": effective}
        else:
            src_ids = sources.get(node_id, [])
            if src_ids:
                raw = results.get(src_ids[0], "")
                # Unwrap dict outputs from ConditionNode
                if isinstance(raw, dict):
                    raw = raw.get("output", raw.get("result", str(raw)))
                node_inputs = {"input": raw}
            else:
                node_inputs = {"input": user_input}

        try:
            results[node_id] = await node.run(node_inputs)
        except Exception as exc:
            results[node_id] = {"error": str(exc)}

    # Pick the output node's result, falling back to the last executed node.
    output_node = next(
        (n for n in reversed(workflow["nodes"]) if n["type"] == "output"), None
    )
    final_id = output_node["id"] if output_node else (order[-1] if order else None)
    final = results.get(final_id, "") if final_id else ""
    if isinstance(final, dict):
        final = final.get("output", final.get("result", str(final)))

    return {"result": final, "trace": results}
