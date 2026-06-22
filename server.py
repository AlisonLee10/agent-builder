from __future__ import annotations

import json
import os
import asyncio
from collections.abc import Callable
from pathlib import Path


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from api.routes      import router as builder_router
from agent           import run_agent
from services.verify import run_verification
from services.storage import sources_to_list, normalize_research_for_save
from services.logger  import get_logger, new_run_id, clear_run_id
from services.platform_parser import (
    validate_posting_intent,
    format_platform_plan,
)
from services.platform_posting   import post_to_platforms
from services.prompt_validation  import validate_user_prompt
from services.denial_reason_validation import validate_denial_reason
from sse_starlette.sse import EventSourceResponse

log = get_logger(__name__)

# =============================================================================
# server.py
#
# WHAT CHANGED IN THIS FILE (Phase 4b)
#
# 1. RunRequest gains an optional `domain` field (default "marketing") and an
#    optional `task_type` field. When present, the request goes through the
#    new domain-aware pipeline (DomainPack → Generator → Validator → Compiler).
#    When absent, the original run_agent() path is used unchanged — full
#    backward compatibility.
#
# 2. /api/deny wires add_rejection() from embedder.py into the denial handler.
#    Every human rejection now feeds the self-learning FAISS loop.
#    The existing save_campaign() + add_campaign_to_index() calls are kept.
#
# 3. /api/approve is UNCHANGED — the existing posting + indexing logic is
#    identical. No modification needed.
#
# 4. All existing endpoints, models, and startup logic are UNCHANGED.
#
# WHY BACKWARD COMPATIBLE
#   The frontend currently sends {prompt} with no domain field. RunRequest
#   defaults domain=None, which keeps the original run_agent() code path.
#   The new path only activates when domain is explicitly provided.
#   This means nothing breaks before Phase 5 updates the frontend/CLI.
# =============================================================================

app = FastAPI(
    title       = "AI Agent Builder API",
    description = (
        "Domain-aware AI agent workflow platform.\n\n"
        "**Original flow:** `POST /api/run` (no domain) → review → "
        "`POST /api/approve` or `POST /api/deny`\n\n"
        "**Domain-aware flow:** `POST /api/run` (with domain) → "
        "Generator → Compiler → review → approve/deny"
    ),
    version = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ── Agent Builder API (workflow engine, tool registry) ────────────────────────
app.include_router(builder_router)


# ── Request / Response models ─────────────────────────────────────────────────
# Models marked UNCHANGED are identical to the original server.py.

class RunRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length  = 5,
        max_length  = 500,
        description = "The task or topic for the agent",
        examples    = ["Write about FlowAI for busy professionals aged 25-40"],
    )
    # ── NEW fields (Phase 4b) ──────────────────────────────────────────────
    domain:    str | None = Field(
        default     = None,
        description = (
            "Domain pack to activate. e.g. 'marketing'. "
            "If omitted, the original run_agent() path is used."
        ),
    )
    task_type: str | None = Field(
        default     = None,
        description = (
            "Override task type inference. e.g. 'email_generation'. "
            "If omitted, SemanticLayer infers it from the prompt."
        ),
    )

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt cannot be empty")
        return stripped


class RunResponse(BaseModel):  # UNCHANGED
    run_id:              str
    content:             str
    hashtags:            list[str]
    verdict:             str
    summary:             str
    full_post:           str
    articles:            list[dict]
    sources:             list[str]
    requested_platforms: list[str] = []
    gmail_to:            str | None = None
    platform_plan:       str = ""


class PostRequest(BaseModel):  # UNCHANGED
    run_id:       str
    full_post:    str
    content:      str = ""
    prompt:       str
    hashtags:     list[str]
    verdict_info: dict = {}
    sources:      list[str] = []
    articles:     list[dict] = []


class DenyRequest(PostRequest):
    user_denial_reason: str = Field(
        ...,
        min_length  = 3,
        max_length  = 1000,
        description = "Why the user rejected this draft",
    )
    # ── NEW fields (Phase 4b) ──────────────────────────────────────────────
    domain:    str | None = Field(
        default     = None,
        description = "Domain pack that produced this output — used for self-learning indexing.",
    )
    task_type: str | None = Field(
        default     = None,
        description = "Task type of the rejected output — used as metadata in rejected/ index.",
    )

    @field_validator("user_denial_reason")
    @classmethod
    def clean_denial_reason(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Please explain why you are denying this post.")
        return stripped


class CampaignSummary(BaseModel):  # UNCHANGED
    run_id:           str
    timestamp:        str
    status:           str
    user_prompt:      str
    hashtags:         list[str]
    verdict:          str
    filename:         str
    sources:          list[str]
    articles:         list[dict]
    posted_platforms: list[str] = []
    denial_reason:    str = ""


class HealthResponse(BaseModel):  # UNCHANGED
    status:  str
    version: str


class RunStreamEvent(BaseModel):
    """
    A single Server-Sent Event in a streaming run.

    event types:
        step_start   — a workflow step has begun executing
        step_done    — a step finished, output is in data
        hitl_gate    — workflow paused for human review
        output       — final assembled output (content + hashtags + sources)
        error        — something went wrong
        done         — stream complete, no more events
    """
    event: str          # step_start | step_done | hitl_gate | output | error | done
    step:  str  = ""    # step name (empty for output/error/done events)
    data:  str  = ""    # payload — step output text, error message, or final output JSON


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return RedirectResponse(url="/frontend/index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/run", response_model=RunResponse, tags=["Agent"])
async def run(request: RunRequest):
    """
    Run the agent pipeline.

    If `domain` is provided → domain-aware path:
        DomainPack.load() → generate_config() → validate_config() →
        compile_and_run() → run_verification()

    If `domain` is omitted → original path (unchanged):
        run_agent() → run_verification()
    """
    run_id = new_run_id()
    log.info(f"POST /api/run — '{request.prompt[:60]}' | domain={request.domain}")

    valid, reason = validate_user_prompt(request.prompt)
    if not valid:
        log.warning(f"Prompt rejected — {reason}")
        raise HTTPException(status_code=400, detail=reason)

    plat_ok, plat_reason, intent = validate_posting_intent(request.prompt)
    if not plat_ok:
        log.warning(f"Prompt rejected — {plat_reason}")
        raise HTTPException(status_code=400, detail=plat_reason)

    try:
        if request.domain:
            output = await _run_domain_pipeline(
                prompt    = request.prompt,
                domain    = request.domain,
                task_type = request.task_type,
            )
        else:
            # ── Original path — unchanged ──────────────────────────────────
            output = run_agent(request.prompt)

        verification = run_verification(output["content"])

        hashtags_list = [
            h.strip() for h in output.get("hashtags", "").split()
            if h.strip().startswith("#")
        ] if isinstance(output.get("hashtags"), str) else output.get("hashtags", [])

        sources_list, articles_list = normalize_research_for_save(
            output.get("sources"),
            output.get("articles", []),
        )

        return RunResponse(
            run_id              = run_id,
            content             = output["content"],
            hashtags            = hashtags_list,
            verdict             = verification["verdict"],
            summary             = verification.get("summary", ""),
            full_post           = output["full_post"],
            sources             = sources_list,
            articles            = articles_list,
            requested_platforms = intent.platforms,
            gmail_to            = intent.gmail_to,
            platform_plan       = format_platform_plan(intent),
        )

    except Exception as e:
        log.error(f"POST /api/run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        clear_run_id()


@app.post("/api/approve", tags=["Campaigns"])
async def approve_campaign(request: PostRequest):
    """
    UNCHANGED from original server.py.
    Post to platforms named in the user's prompt, then save the campaign.
    """
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

    plat_ok, plat_reason, intent = validate_posting_intent(request.prompt)
    if not plat_ok:
        raise HTTPException(status_code=400, detail=plat_reason)

    posted, failed, errors = await post_to_platforms(
        request.full_post,
        intent,
        content  = request.content or request.full_post,
        hashtags = request.hashtags,
    )
    if failed:
        detail = "; ".join(f"{p}: {errors.get(p, 'failed')}" for p in failed)
        if posted:
            detail = f"Posted to {', '.join(posted)} but failed: {detail}"
        raise HTTPException(status_code=502, detail=detail)

    sources_list, articles_list = normalize_research_for_save(
        request.sources,
        request.articles,
    )

    saved = save_campaign(
        request.prompt,
        request.full_post,
        request.hashtags,
        status           = "posted",
        verdict_info     = request.verdict_info,
        platform         = ",".join(posted),
        posted_platforms = posted,
        sources          = sources_list,
        articles         = articles_list,
        run_id           = request.run_id,
    )
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"memory index update skipped: {e}")

    return {
        "status":           "posted",
        "posted_platforms": posted,
        "id":               saved["id"],
    }


@app.post("/api/deny", tags=["Campaigns"])
async def deny_campaign(request: DenyRequest):
    """
    Deny a generated output.

    CHANGED (Phase 4b): after saving the denial, calls add_rejection()
    from embedder.py to feed the self-learning FAISS loop.
    Everything else is identical to the original.
    """
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

    valid, denial_msg = validate_denial_reason(
        request.user_denial_reason,
        campaign_prompt = request.prompt,
    )
    if not valid:
        raise HTTPException(status_code=400, detail=denial_msg)

    sources_list, articles_list = normalize_research_for_save(
        request.sources,
        request.articles,
    )

    saved = save_campaign(
        request.prompt,
        request.content or request.full_post,
        request.hashtags,
        status             = "denied",
        full_post          = request.full_post,
        verdict_info       = {
            "verdict": "user_denied",
            "issues":  [],
            "summary": "",
        },
        platform           = "",
        posted_platforms   = [],
        sources            = sources_list,
        articles           = articles_list,
        run_id             = request.run_id,
        user_denial_reason = request.user_denial_reason,
    )

    # ── Existing index update (unchanged) ──────────────────────────────────
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"campaign memory index update skipped: {e}")

    # ── NEW: self-learning loop (Phase 4b) ─────────────────────────────────
    # Feed the rejection into the domain FAISS rejected/ index so future
    # Generator calls can learn from this failure.
    # Only runs when a domain was active for this run.
    if request.domain:
        _record_rejection_for_learning(
            content          = request.content or request.full_post,
            task_type        = request.task_type or "email_generation",
            rejection_reason = request.user_denial_reason,
            domain           = request.domain,
            source_file      = saved.get("filename", ""),
        )

    return {"status": "denied", "id": saved["id"]}


# ── UNCHANGED endpoints ───────────────────────────────────────────────────────

@app.get("/api/campaigns", response_model=list[CampaignSummary], tags=["Campaigns"])
def list_campaigns():
    """Return a list of all past campaigns, newest first."""
    campaigns_dir = Path("campaigns")
    if not campaigns_dir.exists():
        return []
    results = []
    for path in sorted(campaigns_dir.glob("*.json"), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(CampaignSummary(
                run_id           = data.get("run_id",      ""),
                timestamp        = data.get("timestamp",   ""),
                status           = data.get("status",      ""),
                user_prompt      = data.get("user_prompt", ""),
                hashtags         = data.get("hashtags",    []),
                verdict          = data.get("verdict",     ""),
                filename         = path.name,
                sources          = data.get("sources",     []),
                articles         = data.get("articles",    []),
                posted_platforms = data.get("posted_platforms", []),
                denial_reason    = data.get("denial_reason", ""),
            ))
        except (json.JSONDecodeError, OSError):
            continue
    return results


@app.get("/api/campaigns/{filename}", tags=["Campaigns"])
def get_campaign(filename: str):
    """Return the full JSON for one specific campaign."""
    path = Path("campaigns") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Campaign not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

@app.get("/api/run/stream", tags=["Agent"])
async def run_stream(
    prompt:    str,
    domain:    str | None = None,
    task_type: str | None = None,
):
    """
    Streaming version of /api/run using Server-Sent Events (SSE).

    The client connects and receives a stream of RunStreamEvent JSON objects,
    one per workflow step, then a final 'output' event with the assembled result.

    WHY SSE INSTEAD OF WEBSOCKET
      The existing /api/run uses HTTP request-response which blocks until
      the entire workflow completes. For long pipelines (research → email →
      brief → HITL) this can take 30-60 seconds with no feedback to the user.
      SSE streams progress events as each step completes — the frontend can
      show a live step-by-step trace.
      SSE (not WebSocket) because the stream is one-directional: server → client.
      The client only needs to listen, not send messages back during the run.

    QUERY PARAMETERS
      prompt    : the NL input (required)
      domain    : domain pack name, e.g. "marketing" (optional)
      task_type : override task type inference (optional)

    EVENT SEQUENCE (domain run)
      → step_start  (for each step as it begins)
      → step_done   (for each step as it finishes, with output snippet)
      → hitl_gate   (if a HITL step is reached — workflow pauses here)
      → output      (final assembled JSON: content, hashtags, sources, full_post)
      → done        (stream closed)

    EVENT SEQUENCE (original path, no domain)
      → step_start  "running_agent"
      → step_done   "running_agent"
      → output      (final assembled JSON)
      → done

    TECHNOLOGY
      sse-starlette  — EventSourceResponse wraps an async generator.
                       Installed as a new dependency (requirements.txt).
      asyncio.Queue  — decouples the workflow execution from the SSE emit loop.
                       The workflow runs in a background task and pushes events
                       into the queue. The SSE generator reads from the queue.
                       This prevents the workflow from blocking the event loop.
    """
    run_id = new_run_id()
    log.info(f"GET /api/run/stream — '{prompt[:60]}' | domain={domain}")

    # Validate prompt before opening the stream
    valid, reason = validate_user_prompt(prompt)
    if not valid:
        raise HTTPException(status_code=400, detail=reason)

    # asyncio.Queue bridges the workflow task and the SSE generator.
    # Sentinel value None signals the generator to close the stream.
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def _emit(event: str, step: str = "", data: str = "") -> None:
        """Push one event dict into the queue."""
        await queue.put({"event": event, "step": step, "data": data})

    async def _run_workflow() -> None:
        """
        Run the workflow in a background task and emit events to the queue.
        Always pushes a sentinel None when done so the generator closes cleanly.
        """
        try:
            if domain:
                await _run_domain_stream(
                    prompt    = prompt,
                    domain    = domain,
                    task_type = task_type,
                    emit      = _emit,
                    run_id    = run_id,
                )
            else:
                await _run_original_stream(
                    prompt = prompt,
                    emit   = _emit,
                    run_id = run_id,
                )
        except Exception as e:
            log.error(f"Streaming run failed: {e}")
            await _emit("error", data=str(e))
        finally:
            await queue.put(None)   # sentinel — closes the SSE stream
            clear_run_id()

    async def _event_generator():
        """
        AsyncGenerator consumed by EventSourceResponse.
        Reads from the queue and yields SSE-formatted dicts until sentinel.
        """
        import json as _json

        # Start the workflow as a background task so it runs concurrently
        # with the generator (which is reading from the queue)
        asyncio.create_task(_run_workflow())

        while True:
            event = await queue.get()
            if event is None:
                # Sentinel received — emit final done event and close
                yield {"event": "done", "data": ""}
                break
            yield {
                "event": event["event"],
                "data":  _json.dumps({
                    "step": event.get("step", ""),
                    "data": event.get("data", ""),
                }),
            }

    return EventSourceResponse(_event_generator())


# ── Streaming workflow helpers ────────────────────────────────────────────────

async def _run_domain_stream(
    prompt:    str,
    domain:    str,
    task_type: str | None,
    emit:      Callable,
    run_id:    str,
) -> None:
    """
    Domain-aware workflow with per-step SSE events.
    Mirrors _run_domain_pipeline() from the existing /api/run endpoint
    but emits progress events at each step boundary.
    """
    import json as _json
    from domain_pack  import DomainPack
    from generator    import generate_config
    from validator    import validate_config, validate_output
    from compiler     import YAMLToLangGraph

    # Step 1: Load domain pack
    await emit("step_start", step="load_domain", data=f"Loading domain '{domain}'")
    domain_pack = DomainPack.load(
        domain_name = domain,
        task_type   = task_type or "",
        nl_input    = prompt,
    )
    await emit("step_done",  step="load_domain",
               data=f"Domain '{domain_pack.name}' loaded | task_type: {domain_pack.task_type}")

    # Step 2: Generate config
    await emit("step_start", step="generate_config", data="Generating workflow config")
    config = await generate_config(prompt, domain_pack)
    await emit("step_done",  step="generate_config",
               data=f"{len(config.steps)} steps: {[s.name for s in config.steps]}")

    # Step 3: Validate config
    await emit("step_start", step="validate_config", data="Validating config")
    schema_result = validate_config(config, domain_pack)
    if not schema_result.passed:
        errors = "; ".join(v["id"] for v in schema_result.violations)
        await emit("error", step="validate_config",
                   data=f"Config validation failed: {errors}")
        return
    await emit("step_done", step="validate_config", data="Config valid")

    # Step 4: Compile graph and run each step with per-step events
    await emit("step_start", step="compile_graph", data="Compiling workflow graph")
    compiler = YAMLToLangGraph(config, domain_pack)

    # Patch the compiler to emit events at each node boundary.
    # We wrap _make_tool_node to emit step_start / step_done around the
    # original node function — no changes to compiler.py needed.
    original_make_tool_node = compiler._make_tool_node

    def _patched_make_tool_node(step, agent_executor, tool_map):
        original_node = original_make_tool_node(step, agent_executor, tool_map)

        async def instrumented_node(state):
            await emit("step_start", step=step.name,
                       data=f"Running step '{step.name}' (tool: {step.tool})")
            result = await original_node(state)
            output_snippet = ""
            if result and result.get("step_outputs"):
                latest = list(result["step_outputs"].values())[-1]
                output_snippet = str(latest)[:200]
            await emit("step_done", step=step.name, data=output_snippet)
            return result

        instrumented_node.__name__ = step.name
        return instrumented_node

    compiler._make_tool_node = _patched_make_tool_node
    await emit("step_done", step="compile_graph", data="Graph compiled")

    # Step 5: Run the compiled graph
    output = await compiler.run(prompt)

    # Step 6: Governance check
    if output.get("content"):
        gov_result = validate_output(output["content"], config, domain_pack)
        if not gov_result.passed:
            violations = [v["id"] for v in gov_result.violations]
            await emit("step_done", step="governance_check",
                       data=f"⚠️ Violations: {violations}")

    # Final output event — full structured result as JSON
    await emit("output", data=_json.dumps({
        "content":   output.get("content", ""),
        "hashtags":  output.get("hashtags", []),
        "sources":   output.get("sources",  []),
        "full_post": output.get("full_post", ""),
    }))


async def _run_original_stream(
    prompt: str,
    emit:   Callable,
    run_id: str,
) -> None:
    """
    Original run_agent() path wrapped with SSE events.
    No domain pack — same behaviour as the existing /api/run endpoint
    but with progress visibility.
    """
    import json as _json

    await emit("step_start", step="running_agent", data="Running marketing agent")

    # run_agent is sync — run it in a thread to avoid blocking the event loop
    loop   = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, run_agent, prompt)

    await emit("step_done", step="running_agent",
               data=output.get("content", "")[:200])

    await emit("output", data=_json.dumps({
        "content":   output.get("content", ""),
        "hashtags":  output.get("hashtags", []),
        "sources":   output.get("sources",  []),
        "full_post": output.get("full_post", ""),
    }))


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _run_domain_pipeline(
    prompt:    str,
    domain:    str,
    task_type: str | None,
) -> dict:
    """
    Domain-aware execution path:
        DomainPack.load() → generate_config() → validate_config() →
        compile_and_run()

    Called by /api/run when request.domain is set.
    Returns the same dict shape as run_agent() for drop-in compatibility.
    """
    from domain_pack  import DomainPack
    from generator    import generate_config
    from validator    import validate_config
    from compiler     import compile_and_run

    # 1. Load domain pack — activates Jinja2 templates, GovernanceLoader,
    #    SemanticLayer, and FAISSRetriever for this domain
    domain_pack = DomainPack.load(
        domain_name  = domain,
        task_type    = task_type or "",
        nl_input     = prompt,
    )

    log.debug(
        f"Domain pipeline — domain: {domain_pack.name} | "
        f"task_type: {domain_pack.task_type} | "
        f"model: {domain_pack.preferred_model()}"
    )

    # 2. Generate AgentConfig from NL input
    config = await generate_config(prompt, domain_pack)

    log.debug(
        f"Config generated — {len(config.steps)} steps: "
        f"{[s.name for s in config.steps]}"
    )

    # 3. Validate config against domain rules (Layer 1 only — no text yet)
    from validator import validate_config
    schema_result = validate_config(config, domain_pack)
    if not schema_result.passed:
        error_descriptions = "; ".join(
            v["description"] for v in schema_result.violations
        )
        raise ValueError(
            f"Generated config failed domain validation: {error_descriptions}"
        )

    # 4. Compile and run the workflow
    output = await compile_and_run(config, domain_pack, prompt)

    # 5. Layer 2 + 3 validation on the text output
    if output.get("content"):
        from validator import validate_output
        gov_result = validate_output(output["content"], config, domain_pack)
        if not gov_result.passed:
            violations = "; ".join(v["id"] for v in gov_result.violations)
            log.warning(
                f"Output has governance violations: {violations} — "
                f"returning anyway, violations logged"
            )
            output["governance_violations"] = gov_result.violations

    return output


def _record_rejection_for_learning(
    content:          str,
    task_type:        str,
    rejection_reason: str,
    domain:           str,
    source_file:      str,
) -> None:
    """
    Feed a human rejection into the domain's FAISS rejected/ index.
    Triggers automatic re-indexing when REINDEX_THRESHOLD is reached
    (defined in embedder.py as 5 rejections).

    Called by /api/deny when request.domain is set.
    """
    try:
        from domain_pack import DomainPack
        import yaml
        from pathlib import Path

        domain_folder    = Path("domains") / domain
        domain_yaml_path = domain_folder / "domain.yaml"

        if not domain_yaml_path.exists():
            log.warning(
                f"Self-learning skipped — domain folder not found: {domain_folder}"
            )
            return

        with open(domain_yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return

        from embedder import FAISSRetriever
        retriever = FAISSRetriever(cfg["training_data"], domain_folder)
        retriever.add_rejection(
            text             = content,
            task_type        = task_type,
            rejection_reason = rejection_reason,
            source_file      = source_file,
        )

        log.debug(
            f"Self-learning: rejection recorded for domain '{domain}' | "
            f"task_type: '{task_type}'"
        )

    except Exception as e:
        # Non-fatal — the denial was already saved, learning is best-effort
        log.warning(f"Self-learning index update failed (non-fatal): {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)