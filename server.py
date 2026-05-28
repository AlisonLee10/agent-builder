from __future__ import annotations

import json
import os
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
from services.storage import sources_to_list, normalize_research_for_save
from services.logger import get_logger, new_run_id, clear_run_id
from services.platform_parser import (
    validate_posting_intent,
    format_platform_plan,
)
from services.platform_posting import post_to_platforms
from services.prompt_validation import validate_user_prompt

log = get_logger(__name__)
app = FastAPI(
    title="Marketing Agent API",
    description = (
        "AI-powered marketing content piptline.\n\n"
        "**Flow:** `POST /api/run` → review content → `POST /api/approve` or `POST /api/deny` "
        "(platforms parsed from the prompt)"
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
    run_id:               str
    content:              str
    hashtags:             list[str]
    verdict:              str
    summary:              str
    full_post:            str
    articles:             list[dict]
    sources:              list[str]
    requested_platforms:  list[str] = []
    gmail_to:             str | None = None
    platform_plan:        str = ""

class PostRequest(BaseModel):
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
        min_length=3,
        max_length=1000,
        description="Why the user rejected this draft",
    )

    @field_validator("user_denial_reason")
    @classmethod
    def clean_denial_reason(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Please explain why you are denying this post.")
        return stripped

class CampaignSummary(BaseModel):
    run_id:            str
    timestamp:         str
    status:            str
    user_prompt:       str
    hashtags:          list[str]
    verdict:           str
    filename:          str
    sources:           list[str]
    articles:          list[dict]
    posted_platforms:  list[str] = []
    denial_reason:     str = ""

class HealthResponse(BaseModel):
    status:  str
    version: str

class EmailRequest(BaseModel):
    run_id:       str
    full_post:    str
    prompt:       str
    hashtags:     list[str]
    to:           str = Field(..., description="Recipient email address")
    subject:      str = Field(default="Marketing Update")
    verdict_info: dict = {}
    sources:      list[str] = []
    articles:     list[dict] = []

class GmailConfigResponse(BaseModel):
    configured:        bool
    default_recipient: str
    default_subject:   str


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

    valid, reason = validate_user_prompt(request.prompt)
    if not valid:
        log.warning(f"Prompt rejected — {reason}")
        raise HTTPException(status_code=400, detail=reason)

    plat_ok, plat_reason, intent = validate_posting_intent(request.prompt)
    if not plat_ok:
        log.warning(f"Prompt rejected — {plat_reason}")
        raise HTTPException(status_code=400, detail=plat_reason)

    try:
        output       = run_agent(request.prompt)
        verification = run_verification(output["content"])

        hashtags_list = [
            h.strip() for h in output["hashtags"].split()
            if h.strip().startswith("#")
        ]

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
    """Post to platforms named in the user's prompt, then save the campaign."""
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
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

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
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"memory index update skipped: {e}")

    return {"status": "denied", "id": saved["id"]}


@app.get("/api/gmail-config", response_model=GmailConfigResponse, tags=["Campaigns"])
def gmail_config():
    """Expose non-secret Gmail defaults for the frontend test form."""
    return GmailConfigResponse(
        configured        = bool(os.getenv("GMAIL_SENDER_EMAIL")),
        default_recipient = os.getenv("GMAIL_TEST_RECIPIENT", ""),
        default_subject   = os.getenv("GMAIL_DEFAULT_SUBJECT", "Marketing Update"),
    )


@app.post("/api/post-email", tags=["Campaigns"])
async def post_email(request: EmailRequest):
    from services.gmail           import send_email
    from services.storage         import save_campaign
    from services.campaign_memory import add_campaign_to_index

    if not os.getenv("GMAIL_SENDER_EMAIL"):
        raise HTTPException(
            status_code=503,
            detail="Gmail not configured — set GMAIL_SENDER_EMAIL in .env",
        )

    success = await asyncio.to_thread(
        send_email, request.to, request.subject, request.full_post
    )
    if not success:
        raise HTTPException(status_code=502, detail="Email send failed")

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
        platform         = "email",
        posted_platforms = ["gmail"],
        sources          = sources_list,
        articles         = articles_list,
        run_id           = request.run_id,
    )
    try:
        add_campaign_to_index(saved["filename"])
    except Exception as e:
        log.warning(f"memory index update skipped: {e}")

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


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)