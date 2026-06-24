"""Built-in tools — always available without user registration.
Each tool requires the relevant API key in .env to function.
"""
from __future__ import annotations

from langchain_core.tools import Tool

BUILTIN_TOOLS = [
    {
        "id":           "image_gen",
        "name":         "Image Generation",
        "icon":         "🖼️",
        "description":  "Generate an image from a text description (DALL-E 3)",
        "requires_key": "OPENAI_API_KEY",
    },
    {
        "id":           "translate",
        "name":         "Translation",
        "icon":         "🌐",
        "description":  "Translate text to any language (GPT-4o-mini)",
        "requires_key": "OPENAI_API_KEY",
    },
    {
        "id":           "classify",
        "name":         "Classifier",
        "icon":         "🏷️",
        "description":  "Classify / categorise text and return a JSON result",
        "requires_key": "OPENAI_API_KEY",
    },
    {
        "id":           "pdf_reader",
        "name":         "PDF Reader",
        "icon":         "📄",
        "description":  "Extract all text from a PDF at a given URL",
        "requires_key": None,
    },
    {
        "id":           "csv_analyzer",
        "name":         "CSV Analyzer",
        "icon":         "📊",
        "description":  "Analyze CSV data or answer questions about it",
        "requires_key": "OPENAI_API_KEY",
    },
    {
        "id":           "web_fetch",
        "name":         "Web Fetch",
        "icon":         "🔗",
        "description":  "Fetch and extract readable text from any URL",
        "requires_key": None,
    },
    {
        "id":           "news_fetch",
        "name":         "News Fetch",
        "icon":         "📰",
        "description":  "Search and fetch recent news articles via NewsAPI",
        "requires_key": "NEWS_API_KEY",
    },
    {
        "id":           "gmail_send",
        "name":         "Gmail Send",
        "icon":         "📧",
        "description":  "Send an email via Gmail using SMTP App Password",
        "requires_key": "GMAIL_APP_PASSWORD",
    },
    {
        "id":           "discord_send",
        "name":         "Discord Send",
        "icon":         "🎮",
        "description":  "Post a message to Discord via an Incoming Webhook URL",
        "requires_key": "DISCORD_WEBHOOK_URL",
    },
    {
        "id":           "video_gen",
        "name":         "Video Generation",
        "icon":         "🎬",
        "description":  "Generate a short video clip via RunwayML API",
        "requires_key": "RUNWAYML_API_KEY",
    },
    {
        "id":           "summarize",
        "name":         "Summarizer",
        "icon":         "📝",
        "description":  "Summarize long text into bullet points",
        "requires_key": "OPENAI_API_KEY",
    },
]


def is_available(tool_id: str) -> bool:
    entry = next((t for t in BUILTIN_TOOLS if t["id"] == tool_id), None)
    if not entry:
        return False
    key = entry.get("requires_key")
    if not key:
        return True
    from engine.key_store import get_key
    return bool(get_key(key))


def get_builtin_tool(tool_id: str) -> Tool | None:
    if not is_available(tool_id):
        return None

    if tool_id == "image_gen":
        def _gen(prompt: str) -> str:
            from openai import OpenAI
            client = OpenAI()
            try:
                resp = client.images.generate(model="gpt-image-1", prompt=prompt, n=1, size="1024x1024")
                # gpt-image-1 may return base64 instead of URL
                item = resp.data[0]
                if getattr(item, "url", None):
                    return item.url
                if getattr(item, "b64_json", None):
                    return f"data:image/png;base64,{item.b64_json}"
            except Exception:
                # Fall back to dall-e-2 if gpt-image-1 is unavailable on this key
                resp = client.images.generate(model="dall-e-2", prompt=prompt, n=1, size="1024x1024")
                return resp.data[0].url or "Image generation failed"
            return "Image generation failed"
        return Tool(name="image_generation", func=_gen, description="Generate an image from a text prompt. Returns a URL.")

    if tool_id == "translate":
        def _translate(text: str) -> str:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            import json as _j
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            try:
                req = _j.loads(text)
                target = req.get("target_language", "English")
                content = req.get("text", text)
            except Exception:
                target, content = "English", text
            resp = llm.invoke([
                SystemMessage(content=f"Translate the following text to {target}. Reply with ONLY the translation."),
                HumanMessage(content=content),
            ])
            return resp.content
        return Tool(name="translate", func=_translate, description="Translate text. Pass JSON {text, target_language} or plain text.")

    if tool_id == "classify":
        def _classify(text: str) -> str:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            resp = llm.invoke([
                SystemMessage(content="Classify the input text. Return JSON with keys: category, confidence (0-1), reasoning."),
                HumanMessage(content=text),
            ])
            return resp.content
        return Tool(name="classify", func=_classify, description="Classify text into a category. Returns JSON.")

    if tool_id == "pdf_reader":
        def _pdf(url: str) -> str:
            import requests, io
            try:
                import pypdf
            except ImportError:
                return "pypdf not installed — run: pip install pypdf"
            try:
                resp = requests.get(url.strip(), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                reader = pypdf.PdfReader(io.BytesIO(resp.content))
                text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                return text[:10000] or "No text extracted."
            except Exception as exc:
                return f"PDF read failed: {exc}"
        return Tool(name="pdf_reader", func=_pdf, description="Extract text from a PDF URL.")

    if tool_id == "csv_analyzer":
        def _csv(query: str) -> str:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            resp = llm.invoke([
                SystemMessage(content="You are a data analyst. Analyze the CSV data provided or answer questions about it. Be concise."),
                HumanMessage(content=query),
            ])
            return resp.content
        return Tool(name="csv_analyzer", func=_csv, description="Analyze CSV data. Pass the CSV text or a question about it.")

    if tool_id == "news_fetch":
        from engine.key_store import get_key
        api_key = get_key("NEWS_API_KEY")
        def _news(query: str) -> str:
            import requests as _req
            params = {
                "q":        query,
                "pageSize": 5,
                "language": "en",
                "sortBy":   "publishedAt",
                "apiKey":   api_key,
            }
            try:
                r = _req.get("https://newsapi.org/v2/everything", params=params, timeout=15)
                if r.status_code != 200:
                    return f"NewsAPI error {r.status_code}: {r.text}"
                articles = r.json().get("articles", [])
                parts = []
                for i, a in enumerate(articles, 1):
                    title = a.get("title", "")
                    desc  = a.get("description", "")
                    url   = a.get("url", "")
                    if title and desc:
                        parts.append(f"[{i}] {title}\n    {desc}\n    {url}")
                return "\n\n".join(parts) if parts else "No articles found."
            except Exception as exc:
                return f"News fetch failed: {exc}"
        return Tool(name="news_fetch", func=_news, description="Search recent news articles. Input: search query. Returns titles, descriptions, and URLs.")

    if tool_id == "web_fetch":
        def _fetch(url: str) -> str:
            import requests
            from html.parser import HTMLParser
            class _Strip(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._buf: list[str] = []
                    self._skip = False
                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "footer", "header"): self._skip = True
                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "footer", "header"): self._skip = False
                def handle_data(self, data):
                    if not self._skip and data.strip(): self._buf.append(data.strip())
            try:
                r = requests.get(url.strip(), timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                p = _Strip(); p.feed(r.text)
                return " ".join(p._buf)[:8000] or "No content extracted."
            except Exception as exc:
                return f"Fetch failed: {exc}"
        return Tool(name="web_fetch", func=_fetch, description="Fetch readable text from a URL.")

    if tool_id == "gmail_send":
        from engine.key_store import get_key as _gk
        sender  = _gk("GMAIL_ADDRESS").strip()
        app_pwd = _gk("GMAIL_APP_PASSWORD").strip()
        def _gmail_send(payload: str) -> str:
            import smtplib, json as _j
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            try:
                req = _j.loads(payload)
            except Exception:
                # Accept plain "to|subject|body" fallback
                parts = payload.split("|", 2)
                req = {"to": parts[0].strip(), "subject": parts[1].strip() if len(parts) > 1 else "(no subject)", "body": parts[2].strip() if len(parts) > 2 else payload}
            to      = req.get("to", "").strip()
            subject = req.get("subject", "(no subject)").strip()
            body    = req.get("body", "").strip()
            if not to:
                return "Error: recipient address ('to') is required."
            msg = MIMEMultipart()
            msg["From"]    = sender
            msg["To"]      = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(sender, app_pwd)
                    smtp.sendmail(sender, to, msg.as_string())
                return f"Email sent successfully to {to}."
            except Exception as exc:
                return f"Failed to send email: {exc}"
        return Tool(
            name="gmail_send",
            func=_gmail_send,
            description=(
                "Send an email via Gmail. "
                'Input must be a JSON string with keys: "to" (recipient address), '
                '"subject" (email subject), "body" (plain-text body). '
                'Example: {"to": "alice@example.com", "subject": "Hello", "body": "Hi Alice!"}'
            ),
        )

    if tool_id == "discord_send":
        from engine.key_store import get_key as _gk
        webhook_url = _gk("DISCORD_WEBHOOK_URL").strip()
        def _discord_send(text: str) -> str:
            import requests as _req
            r = _req.post(webhook_url, json={"content": text}, timeout=15)
            if r.status_code in (200, 204):
                return "✅ Posted to Discord."
            return f"Discord webhook error {r.status_code}: {r.text}"
        return Tool(
            name="discord_send",
            func=_discord_send,
            description="Post a message to Discord via webhook. Input: the message text to post.",
        )

    if tool_id == "video_gen":
        def _video(prompt: str) -> str:
            import requests
            from engine.key_store import get_key
            api_key = get_key("RUNWAYML_API_KEY")
            resp = requests.post(
                "https://api.runwayml.com/v1/image_to_video",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"promptText": prompt, "model": "gen3a_turbo", "duration": 5},
                timeout=60,
            )
            data = resp.json()
            return data.get("url") or data.get("id") or "Video generation failed"
        return Tool(name="video_generation", func=_video, description="Generate a short video clip from a text prompt.")

    if tool_id == "summarize":
        def _summarize(text: str) -> str:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            resp = llm.invoke([
                SystemMessage(content="Summarize the following text into 5 concise bullet points. Use plain language."),
                HumanMessage(content=text[:8000]),
            ])
            return resp.content
        return Tool(name="summarize", func=_summarize, description="Summarize long text into bullet points.")

    return None
