# Agent Builder MVP

A visual no-code/low-code platform for building and running AI agent workflows.
Drag-and-drop nodes onto a canvas, connect them, and execute the workflow against any prompt.

---

## English

### Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| OpenAI API key | — (required for GPT-4o / GPT-4o-mini nodes) |

---

### Installation

**1. Clone the repository**

```bash
git clone <repo-url>
cd agent-builder
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Environment Setup

Create a `.env` file in the project root (same directory as `server.py`):

```env
# ── Required ───────────────────────────────────────────────
OPENAI_API_KEY=sk-...            # GPT-4o / GPT-4o-mini

# ── Optional: Claude models ────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── Optional: Built-in tools ───────────────────────────────
NEWS_API_KEY=...                 # NewsAPI (news search tool)
SERP_API_KEY=...                 # SerpAPI (web search tool)
TAVILY_API_KEY=...               # Tavily (web search tool)

# ── Optional: Delivery integrations ───────────────────────
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# ── Optional: LangSmith tracing ────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=agent-builder

# ── Optional: Workflow execution timeout (seconds) ─────────
# WORKFLOW_TIMEOUT=120
```

> **Tip:** API keys can also be entered directly in the UI under the **Keys** tab — no restart required.

---

### Starting the Server

```bash
python server.py
OR input python3 server.py depending on the version of Python installed in your device.
```

The server starts on **http://localhost:8000** with hot-reload enabled.

To change the port:

```bash
uvicorn server:app --host 0.0.0.0 --port 9000 --reload
```

---

### Using the Workflow Builder

1. Open **http://localhost:8000** in your browser.
2. Click **New** to create a blank canvas.
3. Drag node types from the left palette onto the canvas:

| Node | Purpose |
|---|---|
| **Input** | Entry point — receives the user's prompt |
| **LLM** | Direct LLM call (GPT-4o, Claude, etc.) |
| **Agent** | LLM with tools, MCP servers, and domain context |
| **Condition** | Routes the flow with IF / ELSE IF / ELSE logic |
| **Human Approval** | Pauses for manual Approve / Deny before continuing |
| **Output** | Final result — optionally delivers via Slack, Gmail, or Discord |

4. Connect node handles by dragging from an output circle to an input circle.
5. Click a node to configure it in the right panel.
6. Click **▶ Run** to execute the workflow with a prompt.
7. Click **Save** to persist the workflow.
8. Click **📝** to create a sticky note over the canvas.

---

### Domain Packs

Domain packs inject rich context (brand guidelines, vocabulary, style examples) into Agent nodes.

The built-in **Email (Sales)** domain is located at `domains/marketing/` and includes:

| File | Purpose |
|---|---|
| `domain.yaml` | Entry point — wires all sub-files |
| `governance/brand_guidelines.md` | Brand voice and writing rules |
| `governance/content_policy.yaml` | Machine-readable compliance rules |
| `training_data/approved/` | Positive few-shot examples |
| `training_data/rejected/` | Negative examples (human-rejected outputs) |
| `semantic/ontology.yaml` | Domain entity types and relationships |
| `semantic/vocabulary.json` | Natural-language → parameter value mappings |
| `templates/persona.j2` | Jinja2 agent persona template |
| `templates/hashtags.j2` | Jinja2 hashtag generation rules |

To add a new domain, create a new folder under `domains/` with the same structure and register it in the UI under the **Email (Sales)** tab.

---

### LangSmith Tracing

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, every workflow run is logged to LangSmith as an `e2e_workflow_run` parent trace. Child LLM calls appear as nested spans.

The elapsed E2E time is also printed to the server console:

```
[E2E] Workflow 'abc123' completed in 4.32s
```

---

### Project Structure

```
agent-builder/
├── server.py               # FastAPI entry point (uvicorn)
├── api/
│   └── routes.py           # All REST endpoints (/api/...)
├── engine/
│   ├── executor.py         # Workflow graph execution (LangGraph)
│   ├── nodes.py            # Node type implementations
│   ├── domain_loader.py    # Domain context builder (Jinja2 + YAML)
│   ├── feedback_store.py   # Rejection feedback JSONL store
│   ├── builtin_tools.py    # Built-in tool catalog
│   ├── builtin_mcps.py     # Preset MCP server catalog
│   ├── mcp_runner.py       # MCP subprocess management
│   ├── registry.py         # User tool / MCP registry
│   ├── key_store.py        # API key persistence
│   └── delivery.py         # Slack / Gmail / Discord delivery
├── frontend/
│   └── index.html          # Single-page UI (Drawflow canvas)
├── domains/
│   └── marketing/          # Built-in marketing domain pack
├── data/
│   ├── workflows/          # Saved workflow JSON files
│   ├── templates/          # Workflow starter templates
│   ├── feedback/           # rejected.jsonl (human feedback store)
│   └── api_keys.json       # Persisted user API keys
└── .env                    # Environment variables (not committed)
```

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` inside the activated venv |
| LangSmith traces not appearing | Ensure `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set in `.env` and the server has been restarted |
| Workflow times out | Increase `WORKFLOW_TIMEOUT` in `.env` (default: 120 s) |
| MCP server fails to start | Check that the MCP command is installed and accessible in `$PATH` |
| Keys tab shows "needs key" | Enter the key in the UI Keys tab or add it to `.env` and restart |

---

---

## 한국어

### 사전 요구사항

| 항목 | 최소 버전 |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| OpenAI API 키 | — (GPT-4o / GPT-4o-mini 노드 사용 시 필수) |
| Anthropic API 키 | — (Claude 노드 사용 시 필요, 선택) |

---

### 설치

**1. 저장소 복제**

```bash
git clone <repo-url>
cd agent-builder
```

**2. 가상 환경 생성 및 활성화**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. 의존성 설치**

```bash
pip install -r requirements.txt
```

---

### 환경 변수 설정

프로젝트 루트(server.py와 같은 디렉터리)에 `.env` 파일을 생성합니다:

```env
# ── 필수 ───────────────────────────────────────────────────
OPENAI_API_KEY=sk-...            # GPT-4o / GPT-4o-mini

# ── 선택: Claude 모델 ──────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── 선택: 내장 도구 ────────────────────────────────────────
NEWS_API_KEY=...                 # NewsAPI (뉴스 검색)
SERP_API_KEY=...                 # SerpAPI (웹 검색)
TAVILY_API_KEY=...               # Tavily (웹 검색)

# ── 선택: 전송 통합 ────────────────────────────────────────
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# ── 선택: LangSmith 트레이싱 ───────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=agent-builder

# ── 선택: 워크플로우 실행 타임아웃 (초) ────────────────────
# WORKFLOW_TIMEOUT=120
```

> **팁:** API 키는 UI의 **Keys** 탭에서 직접 입력할 수도 있습니다. 서버 재시작 없이 즉시 반영됩니다.

---

### 서버 기동

```bash
python server.py
```

서버는 **http://localhost:8000** 에서 실행되며 코드 변경 시 자동으로 재시작됩니다.

포트를 변경하려면:

```bash
uvicorn server:app --host 0.0.0.0 --port 9000 --reload
```

---

### 워크플로우 빌더 사용법

1. 브라우저에서 **http://localhost:8000** 을 엽니다.
2. **＋ New Workflow** 를 클릭해 빈 캔버스를 만듭니다.
3. 왼쪽 팔레트에서 노드를 캔버스로 드래그합니다:

| 노드 | 역할 |
|---|---|
| **Input** | 진입점 — 사용자의 프롬프트를 받습니다 |
| **LLM** | 직접 LLM 호출 (GPT-4o, Claude 등) |
| **Agent** | 도구·MCP 서버·도메인 컨텍스트를 갖춘 LLM 에이전트 |
| **Condition** | IF / ELSE IF / ELSE 조건 분기 |
| **Human Approval** | 실행을 일시 중단하고 Approve / Deny 대기 |
| **Output** | 최종 결과 — Slack · Gmail · Discord 전송 옵션 포함 |

4. 출력 핸들(원)을 드래그해 다른 노드의 입력 핸들에 연결합니다.
5. 노드를 클릭하면 우측 패널에서 설정을 변경할 수 있습니다.
6. **▶ Run** 을 클릭해 프롬프트와 함께 워크플로우를 실행합니다.
7. **💾 Save** 를 클릭해 워크플로우를 저장합니다.

---

### 도메인 팩

도메인 팩은 Agent 노드에 브랜드 가이드라인·어휘·스타일 예시 등 풍부한 컨텍스트를 주입합니다.

내장 **Marketing** 도메인은 `domains/marketing/` 에 위치하며 다음 파일로 구성됩니다:

| 파일 | 역할 |
|---|---|
| `domain.yaml` | 진입점 — 하위 파일 경로 연결 |
| `governance/brand_guidelines.md` | 브랜드 보이스 및 작성 규칙 |
| `governance/content_policy.yaml` | 기계 판독형 컴플라이언스 규칙 |
| `training_data/approved/` | 긍정 예시 (Few-shot 학습용) |
| `training_data/rejected/` | 부정 예시 (인간 검토자가 거부한 출력) |
| `semantic/ontology.yaml` | 도메인 엔티티 타입 및 관계 정의 |
| `semantic/vocabulary.json` | 자연어 표현 → 파라미터 값 매핑 |
| `templates/persona.j2` | Jinja2 에이전트 페르소나 템플릿 |
| `templates/hashtags.j2` | Jinja2 해시태그 생성 규칙 템플릿 |

새 도메인을 추가하려면 `domains/` 하위에 같은 구조의 폴더를 만들고, UI의 **Domains** 탭에서 등록합니다.

---

### LangSmith 트레이싱

`.env` 에 `LANGCHAIN_TRACING_V2=true` 와 `LANGCHAIN_API_KEY` 가 설정되어 있으면, 모든 워크플로우 실행이 LangSmith에 `e2e_workflow_run` 부모 트레이스로 기록됩니다. 하위 LLM 호출은 중첩 스팬으로 표시됩니다.

E2E 소요 시간은 서버 콘솔에도 출력됩니다:

```
[E2E] Workflow 'abc123' completed in 4.32s
```

---

### 프로젝트 구조

```
agent-builder/
├── server.py               # FastAPI 진입점 (uvicorn)
├── api/
│   └── routes.py           # 전체 REST 엔드포인트 (/api/...)
├── engine/
│   ├── executor.py         # 워크플로우 그래프 실행 (LangGraph)
│   ├── nodes.py            # 노드 타입 구현체
│   ├── domain_loader.py    # 도메인 컨텍스트 빌더 (Jinja2 + YAML)
│   ├── feedback_store.py   # 거부 피드백 JSONL 저장소
│   ├── builtin_tools.py    # 내장 도구 카탈로그
│   ├── builtin_mcps.py     # 프리셋 MCP 서버 카탈로그
│   ├── mcp_runner.py       # MCP 서브프로세스 관리
│   ├── registry.py         # 사용자 도구 / MCP 레지스트리
│   ├── key_store.py        # API 키 영속성 관리
│   └── delivery.py         # Slack / Gmail / Discord 전송
├── frontend/
│   └── index.html          # 단일 페이지 UI (Drawflow 캔버스)
├── domains/
│   └── marketing/          # 내장 마케팅 도메인 팩
├── data/
│   ├── workflows/          # 저장된 워크플로우 JSON 파일
│   ├── templates/          # 워크플로우 스타터 템플릿
│   ├── feedback/           # rejected.jsonl (인간 피드백 저장소)
│   └── api_keys.json       # 사용자 API 키 (UI 입력 영속화)
└── .env                    # 환경 변수 (커밋하지 마세요)
```

---

### 문제 해결

| 증상 | 해결 방법 |
|---|---|
| 서버 시작 시 `ModuleNotFoundError` | 가상 환경이 활성화된 상태에서 `pip install -r requirements.txt` 실행 |
| LangSmith 트레이스가 보이지 않음 | `.env` 에 `LANGCHAIN_TRACING_V2=true` 와 `LANGCHAIN_API_KEY` 설정 후 서버 재시작 |
| 워크플로우 타임아웃 발생 | `.env` 의 `WORKFLOW_TIMEOUT` 값을 늘림 (기본값: 120초) |
| MCP 서버 시작 실패 | MCP 명령이 설치되어 있고 `$PATH` 에 등록되어 있는지 확인 |
| Keys 탭에 "needs key" 표시 | UI Keys 탭에서 직접 키를 입력하거나 `.env` 에 추가 후 재시작 |
