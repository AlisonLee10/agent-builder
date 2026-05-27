from __future__ import annotations

import json
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent import run_agent
from services.verify import run_verification
from services.storage import sources_to_list
from services.logger import get_logger, new_run_id, clear_run_id

log = get_logger(__name__)
app = FastAPI(
    title="Marketing Agent API",
    description = (
        "AI-powered marketing content piptline.\n\n"
        "**Flow:** `POST /api/run` → review content → `POST /api/post` or `POST /api/deny`"
    ),
    version = "1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────
# Allows the browser (on any port) to call this server.
# In production you'd restrict allow_origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# ══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# FastAPI uses these to validate incoming data and serialize
# outgoing data automatically. If the browser sends wrong data,
# FastAPI rejects it before your code runs.
# ══════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length = 5,
        max_length = 500,
        description = "The marketing goal or topic to write about",
        examples = ["Write about FlowAI for busy professionals aged 25-40"],
    )

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt cannot be empty")
        return stripped

class RunResponse(BaseModel):
    run_id:    str
    content:   str
    hashtags:  list[str]
    verdict:   str
    summary:   str
    full_post: str
    articles:  list[dict]
    sources:   list[str]

class PostRequest(BaseModel):
    run_id:      str
    full_post:   str
    prompt:      str
    hashtags:    list[str]
    verdict_info: dict = {}
    platform:    str = Field(default="")
    sources:     list[str] = []
    articles:    list[dict] = []

class CampaignSummary(BaseModel):
    run_id:      str
    timestamp:   str
    status:      str
    user_prompt: str
    hashtags:    list[str]
    verdict:     str
    filename:    str
    sources:     list[str]
    articles:    list[dict]

class HealthResponse(BaseModel):
    status:  str
    version: str

class EmailRequest(BaseModel):
    run_id:    str
    full_post: str
    prompt:    str
    hashtags:  list[str]
    to:        str  = Field(..., description="Recipient email address")
    subject:   str  = Field(default="Marketing Update")


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["System"])
def root():
    return RedirectResponse(url="/frontend/index.html")

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/run", response_model=RunResponse, tags=["Agent"])
def run(request: RunRequest):
    """
    Run the marketing agent pipeline.
    Receives a prompt, runs the agent + verification,
    returns the generated content. Does NOT post to Discord.
    The frontend handles the approval + post step separately.
    """
    run_id = new_run_id()
    log.info(f"POST /api/run — '{request.prompt[:60]}'")

    try:
        output       = run_agent(request.prompt)
        verification = run_verification(output["content"])

        hashtags_list = [
            h.strip() for h in output["hashtags"].split()
            if h.strip().startswith("#")
        ]

        return RunResponse(
            run_id    = run_id,
            content   = output["content"],
            hashtags  = hashtags_list,
            verdict   = verification["verdict"],
            summary   = verification.get("summary", ""),
            full_post = output["full_post"],
            sources   = sources_to_list(output.get("sources")),
            articles  = output.get("articles", []),
        )

    except Exception as e:
        log.error(f"POST /api/run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        clear_run_id()


@app.post("/api/post", tags=["Campaigns"])
async def post_campaign(request: PostRequest):
    from services.discord         import post_to_discord
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

    success = await asyncio.to_thread(post_to_discord, request.full_post)
    if not success:
        raise HTTPException(status_code=502, detail="Discord post failed")

    saved = save_campaign(
        request.prompt,
        request.full_post,
        request.hashtags,
        status       = "posted",
        verdict_info = request.verdict_info,
        platform     = request.platform or "discord",
        sources      = request.sources,
        articles     = request.articles,
    )
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"memory index update skipped: {e}")

    return {"status": "posted", "id": saved["id"]}


@app.post("/api/post-slack", tags=["Campaigns"])
async def post_slack(request: PostRequest):
    from services.slack           import post_to_slack
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

    success = await asyncio.to_thread(post_to_slack, request.full_post)
    if not success:
        raise HTTPException(status_code=502, detail="Slack post failed")

    saved = save_campaign(
        request.prompt,
        request.full_post,
        request.hashtags,
        status       = "posted",
        verdict_info = request.verdict_info,
        platform     = request.platform or "slack",
        sources      = request.sources,
        articles     = request.articles,
    )
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"memory index update skipped: {e}")

    return {"status": "posted", "platform": "slack", "id": saved["id"]}


@app.post("/api/deny", tags=["Campaigns"])
async def deny_campaign(request: PostRequest):
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

    saved = save_campaign(
        request.prompt,
        request.full_post,
        request.hashtags,
        status       = "denied",
        verdict_info = request.verdict_info or {
            "verdict": "user_denied",
            "issues":  [],
            "summary": "User chose not to post",
        },
        platform     = None,                   # ← denied = no platform
        sources      = request.sources,
        articles     = request.articles,
    )
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"memory index update skipped: {e}")

    return {"status": "denied", "id": saved["id"]}


@app.post("/api/post-email", tags=["Campaigns"])
async def post_email(request: EmailRequest):
    from services.gmail   import send_email
    from services.storage import save_campaign

    success = await asyncio.to_thread(
        send_email, request.to, request.subject, request.full_post
    )
    if not success:
        raise HTTPException(status_code=502, detail="Email send failed")

    saved = save_campaign(
        request.prompt, request.full_post, request.hashtags,
        status   = "posted",
        platform = "email",
    )

    return {"status": "posted", "platform": "email", "id": saved["id"]}


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
                run_id      = data.get("run_id",      ""),
                timestamp   = data.get("timestamp",   ""),
                status      = data.get("status",      ""),
                user_prompt = data.get("user_prompt", ""),
                hashtags    = data.get("hashtags",    []),
                verdict     = data.get("verdict",     ""),
                filename    = path.name,
                sources     = data.get("sources",     []),
                articles    = data.get("articles",    []),
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


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)