"""Built-in tools — always available without user registration.
Each tool requires the relevant API key in .env to function.
"""
from __future__ import annotations

import os
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
    return not key or bool(os.getenv(key))


def get_builtin_tool(tool_id: str) -> Tool | None:
    if not is_available(tool_id):
        return None

    if tool_id == "image_gen":
        def _gen(prompt: str) -> str:
            from openai import OpenAI
            client = OpenAI()
            resp = client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
            return resp.data[0].url or "Image generation failed"
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

    if tool_id == "video_gen":
        def _video(prompt: str) -> str:
            import requests
            api_key = os.getenv("RUNWAYML_API_KEY", "")
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
