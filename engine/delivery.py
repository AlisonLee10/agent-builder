"""Direct delivery integrations for the Output node.
Reads credentials from registry.json and key_store — no MCP subprocess needed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REGISTRY = Path(__file__).parent.parent / "data" / "registry.json"


def _load_mcp_env(name: str) -> dict:
    try:
        data = json.loads(_REGISTRY.read_text())
        for m in data.get("mcps", []):
            if m["name"].lower() == name.lower():
                return m.get("env", {})
    except Exception:
        pass
    return {}


def _extract_json(text: str) -> dict | None:
    """Try to parse the text as JSON; also handles code-fenced JSON blocks."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _extract_image_url(text: str) -> str | None:
    """Find the first image URL in the text (DALL-E CDN or generic image extension)."""
    patterns = [
        r"https://oaidalleapiprodscus\.blob\.core\.windows\.net/[^\s]+",
        r"https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s]*)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


# ── Slack ──────────────────────────────────────────────────────────────────────

def deliver_slack(text: str) -> str:
    """Post text (and optionally an embedded image URL) to the configured Slack channel."""
    import requests

    env        = _load_mcp_env("Slack")
    token      = env.get("SLACK_BOT_TOKEN", "").strip()
    channel_id = env.get("SLACK_CHANNEL_ID", "").strip()

    if not token:
        return "Slack delivery failed: SLACK_BOT_TOKEN not configured in the Slack MCP entry."
    if not channel_id:
        return "Slack delivery failed: SLACK_CHANNEL_ID not configured in the Slack MCP entry."

    image_url = _extract_image_url(text)
    if image_url:
        # Strip the URL from the main text, post as blocks so Slack renders the image
        clean = re.sub(re.escape(image_url), "", text).strip()
        payload = {
            "channel": channel_id,
            "text":    clean,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": clean}},
                {"type": "image",   "image_url": image_url, "alt_text": "Generated image"},
            ],
        }
    else:
        payload = {"channel": channel_id, "text": text}

    resp   = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=15,
    )
    result = resp.json()
    if result.get("ok"):
        return f"✅ Posted to Slack channel {channel_id}."
    return f"Slack API error: {result.get('error', resp.text)}"


# ── Gmail ──────────────────────────────────────────────────────────────────────

def deliver_discord(text: str) -> str:
    """Post text to a Discord channel via an Incoming Webhook URL."""
    import requests

    env         = _load_mcp_env("Discord")
    webhook_url = env.get("DISCORD_WEBHOOK_URL", "").strip()

    if not webhook_url:
        return "Discord delivery failed: DISCORD_WEBHOOK_URL not configured in the Discord MCP entry."

    resp = requests.post(webhook_url, json={"content": text}, timeout=15)
    if resp.status_code in (200, 204):
        return "✅ Posted to Discord."
    return f"Discord webhook error {resp.status_code}: {resp.text}"


# ── Gmail ──────────────────────────────────────────────────────────────────────

def deliver_gmail(text: str, recipient: str, subject: str = "") -> str:
    """Send text as an email via Gmail SMTP.

    If ``text`` is a JSON object with ``subject`` / ``body`` keys (as produced
    by newsletter-writer agents), those values are used automatically.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from engine.key_store import get_key

    # Prefer credentials from the Gmail MCP registry entry, fall back to key_store
    env     = _load_mcp_env("Gmail")
    sender  = (env.get("GMAIL_ADDRESS")      or get_key("GMAIL_ADDRESS")).strip()
    app_pwd = (env.get("GMAIL_APP_PASSWORD") or get_key("GMAIL_APP_PASSWORD")).strip()

    if not sender or not app_pwd:
        return "Gmail delivery failed: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not configured."
    if not recipient:
        return "Gmail delivery failed: no recipient address specified."

    # If the agent returned JSON with subject/body, unpack it
    parsed = _extract_json(text)
    if parsed:
        subject = subject or parsed.get("subject", "Workflow Output")
        text    = parsed.get("body", text)
    else:
        subject = subject or "Workflow Output"

    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(text, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, app_pwd)
            smtp.sendmail(sender, recipient, msg.as_string())
        return f"✅ Email sent to {recipient}."
    except Exception as exc:
        return f"Gmail delivery failed: {exc}"
