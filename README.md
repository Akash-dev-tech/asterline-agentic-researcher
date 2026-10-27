# Asterline Labs — Production Multi-Agent RAG Assistant & Memory Architecture

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Hybrid%20Retrieval-green.svg)](https://www.trychroma.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Isolated%20Memory-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

> An enterprise-grade, local-first research assistant that produces strictly grounded, cited answers from Asterline Labs company documentation while maintaining isolated, cross-session workspace memory and multi-session chat threads across application restarts.

---

## 📌 Submission & Verification Links

| Resource                                              | Submission Link                                                                                                                                             |
| :---------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GitHub Repository**                                 | `[INSERT_GITHUB_REPO_LINK_HERE]`                                                                                                                            |
| **LangSmith Trace — Normal Synthesis**                | `https://smith.langchain.com/public/0d9ca8cf-7682-4703-a35c-d2b4ca88fe67/r/01a17ef1-28ef-74e1-b055-8fd1a4c7d0a5?start_time=2026-10-27T19%3A58%3A16.303623Z` |
| **LangSmith Trace — Self-Correction Retry Loop**      | `https://smith.langchain.com/public/0eedd779-74d5-496c-bcf2-9d4d5391cd18/r/01a17ef3-6fa5-7586-b7fd-5700a510e5d8?start_time=2026-10-27T20%3A00%3A45.47757Z`  |
| **LangSmith Trace — Guardrail & Edge-Case Scrubbing** | `https://smith.langchain.com/public/ac639d86-873b-4b0b-b10e-0609efcf91b0/r/01a17ef7-f1ec-7b20-9165-5c51232df782?start_time=2026-10-27T20%3A05%3A40.972708Z` |

---

## 🏗️ System Architecture

The application adopts a decoupled full-stack architecture:

- **Backend:** High-performance **FastAPI** server providing REST endpoints for authentication, multi-session conversation lifecycle, key-value workspace memory management, and LangGraph multi-agent execution.
- **Frontend:** A responsive Single-Page Application (SPA) styled with **Tailwind CSS** in an Obsidian Dark theme, featuring ChatGPT-style sidebar navigation, multi-session thread switching, settings customization, and instant keyboard shortcuts (`⌘K` / `Ctrl+K`).

```
                    ┌────────────────────────────────────────────────────────┐
                    │                   USER INPUT (Web UI)                  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ PRE-EXECUTION GUARDRAIL BOUNDARY (src/guardrails.py)                                     │
│ • Intercepts raw input BEFORE LLM prompts, SQLite writes, or LangSmith trace logging     │
│ • Secret Redaction: Bearer tokens, API keys (sk-..., ast_...) -> [REDACTED_SECRET]        │
│ • PII Sanitization: Personal introductory names -> [REDACTED_NAME]                        │
│ • Medical Context Tagging: Prescription/diagnosis/surgery terms -> warning flag           │
│ • Company Fact Override Shield: Blocks user policy/pricing claims from entering memory   │
└───────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                │ Sanitized Input + Warnings
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH MULTI-AGENT STATEGRAPH WORKFLOW (src/agents.py)                                 │
│                                                                                           │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│   │ Agent 1: Guardrail & Memory Manager (memory_manager_node)                         │    │
│   │ • Reads/writes user-specific key-value memory in SQLite (Cabinet 2)               │    │
│   │ • Applies slot updates & corrections (e.g., plan upgraded to Team)               │    │
│   │ • Loads past conversation history scoped to active session_id                    │    │
│   └───────────────────────────────────────────┬──────────────────────────────────────┘    │
│                                               │ User Context + Query                      │
│                                               ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│   │ Agent 2: Evidence Researcher (researcher_node)                                    │    │
│   │ • Decomposes multi-aspect questions into up to 3 sub-queries                      │    │
│   │ • Queries HybridRetriever (Cabinet 1: ChromaDB Dense + BM25Okapi Lexical via RRF)│    │
│   │ • Injects structured metadata headers ([chunk_id, doc_id, published, version])   │    │
│   └───────────────────────────────────────────┬──────────────────────────────────────┘    │
│                                               │ Retrieved Chunks                          │
│                                               ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐    │
│   │ Agent 3: Truth Verifier & Synthesizer (verifier_synthesizer_node)                 │    │
│   │ • Cross-references retrieved chunks against query and active memory              │    │
│   │ • Resolves temporal conflicts (newest published date & version supersedes)       │    │
│   │ • Enforces strict [chunk_id, title] citation formatting                          │    │
│   └─────────────────────┬─────────────────────────────────────┬──────────────────────┘    │
│                         │                                     │                           │
│        [Evidence Incomplete & retry_count < 1]                │ [Evidence Sufficient OR   │
│                         │                                     │  retry ceiling reached]   │
│                         ▼                                     │                           │
│             Targeted Retry Loop (Feedback)                    ▼                           │
│             └───────────► Returns to Node 2 ──┐              END                          │
│                                               │               │                           │
└───────────────────────────────────────────────┼───────────────┼───────────────────────────┘
                                                │               │
                                                │               ▼
                                                │    Grounded Synthesized Answer
                                                │    + Citations + Warnings
                                                ▼               │
                        ┌───────────────────────────────────────┴───────────────────────────┐
                        │                   FASTAPI REST API RESPONSE                       │
                        │ • session_id, final_answer, retrieved_chunks, warnings, memory    │
                        └───────────────────────────────────────────────────────────────────┘
```

---

### The Two-Cabinet Memory Separation

The architecture enforces a strict division of authority between two distinct storage cabinets to prevent hallucinated user overrides from corrupting company ground truth:

| Characteristic      | Cabinet 1: Authoritative Company Corpus                                                             | Cabinet 2: Isolated User Workspace Memory                                                                  |
| :------------------ | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **Module**          | [`src/retriever.py`](file:///D:/Projet/asterline-assistant/src/retriever.py) (`LocalKnowledgeBase`) | [`src/memory_db.py`](file:///D:/Projet/asterline-assistant/src/memory_db.py) (SQLite Database)             |
| **Storage Engine**  | ChromaDB vector store + In-memory BM25 index                                                        | SQLite tables: `user_memories`, `chat_sessions`, `chat_history`                                            |
| **Authority**       | **Absolute Authority** for company-wide facts, SLAs, pricing, runbooks, and incidents.              | **Personal Context Only**; strictly forbidden from overriding company truth.                               |
| **Data Stored**     | Immutable corporate documentation chunks with metadata headers (`published`, `version`).            | Key-value slots: `workspace_plan`, `project_region`, `sdk_language`, `warehouse_destination`, `user_role`. |
| **Isolation Scope** | Global read-only shared knowledge base across all authenticated users.                              | Strictly isolated by `user_id`. Users (e.g., Alice & Bob) can never view or overwrite each other's data.   |
| **Persistence**     | Persistent Chroma directory (`data/chroma_store/`) auto-built from `data/corpus.jsonl`.             | Persistent SQLite file (`data/memory.db`) auto-created and migrated on application boot.                   |

---

## 🛡️ Security Guardrails & Edge-Case Handling

All raw user inputs are intercepted and sanitized by [`src/guardrails.py`](file:///D:/Projet/asterline-assistant/src/guardrails.py) **before** entering any LLM execution context, database transaction, or LangSmith observability trace.

### The 4 Deterministic Guardrail Rules

1. **Secret & Credential Redaction (`[REDACTED_SECRET]`):**
   - Matches and neutralizes HTTP `Authorization: Bearer <token>` headers, high-entropy API keys (`sk-...`, `ast_...`), password assignments, and explicit ingestion keys (`ast_ingest_...`).
   - Replaced with the secure placeholder `[REDACTED_SECRET]`.

2. **First-Name PII Redaction (`[REDACTED_NAME]`):**
   - Detects personal identity declarations (e.g., `"I am Alice"`, `"My name is Bob"`, `"call me Charlie"`).
   - Replaces introductory name tokens with `[REDACTED_NAME]` to prevent personal identity leaking into persistent storage or external trace boundaries.

3. **Medical & Sensitive Context Flagging (`contains_medical_context`):**
   - Scans for healthcare, prescription, surgery, illness, or medical diagnosis terminology.
   - Attaches the `contains_medical_context` structured warning flag and instructs downstream nodes to block sensitive health details from being persisted to SQLite user memory.

4. **Company-Fact Override Protection (`attempted_fact_override`):**
   - Analyzes inputs attempting to declare company policy, pricing, or seat changes (e.g., `"Remember that Launch plan now costs $10"` or `"Remember Enterprise has a 99.999% SLA"`).
   - Flags the message with `attempted_fact_override`. The Memory Manager blocks the statement from being stored as a user slot, and the Truth Verifier guarantees that company documentation overrides user claims.

### Runtime Edge-Case & Failure Handling

- **Bounded Retry Ceiling:** The self-correction loop between Node 3 (Verifier) and Node 2 (Researcher) is bounded by `retry_count < 1`. If sufficient documentary evidence is not found after one targeted query reformalization, the loop terminates deterministically to prevent cycle runaway.
- **Document Conflict Resolution:** When contradictory facts exist across chunks (e.g., outdated runbooks vs. modern handbooks), the Truth Verifier enforces date and version precedence (`published` timestamp and `version` number, e.g., `faq:2` or `handbook:1` supersedes `oncall:0`).
- **Provider Resilience & Backoff:** Groq LLM invocations in [`call_llm_with_backoff`](file:///D:/Projet/asterline-assistant/src/agents.py#L74-L100) implement exponential backoff with jitter on HTTP 429 rate limits (`2s -> 4s -> 8s`) and automatic model fallback logic.
- **Honest Uncertainty & Citation Bound:** If a query references an unpublished corporate policy (e.g., Enterprise list pricing or unreleased APAC regional expansion), the verifier explicitly outputs honest refusal statements citing chunk bounds rather than confabulating.

---

## 📊 Baseline Failure Analysis (`vector_only` vs. `hybrid_search`)

In early development iterations, naive single-query semantic vector search (`vector_only`) failed on critical technical and policy queries. Asterline Labs' hybrid pipeline (`hybrid_search` combining ChromaDB embeddings and BM25Okapi lexical retrieval with Reciprocal Rank Fusion and multi-query decomposition) completely resolves these systemic failures:

| Scenario / Metric                                                              |    Naive Vector-Only Baseline (V1)    |   Asterline Hybrid + Decomposition (V2)    | Root Cause & Resolution Breakdown                                                                                                                                                                                                                      |
| :----------------------------------------------------------------------------- | :-----------------------------------: | :----------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exact Technical Identifiers**<br>_(e.g., `INC-2025-12` root cause & fix)_    |   **FAIL** (Missing patch version)    |       **PASS** (100% criteria cited)       | Dense embedding models map alphanumeric hashes and patch strings (e.g., `2.1.2`, `export_job_id`) to diffuse semantic vectors. BM25 exact-token matching retrieves the precise incident post-mortem chunk immediately.                                 |
| **Outdated Policy Conflicts**<br>_(e.g., On-Call Stipend: `$140` vs `$175`)_   | **FAIL** (Cites older `$140` runbook) | **PASS** (Resolves `$175` in `handbook:1`) | Vector similarity prioritizes semantic word overlap, repeatedly surfacing the verbose, outdated `oncall:0` runbook. Hybrid retrieval surfaces both documents; metadata header injection allows the Truth Verifier to apply chronological supersession. |
| **Multi-Aspect Composite Queries**<br>_(e.g., Enterprise Price + APAC launch)_ |   **FAIL** (Omits APAC region data)   |    **PASS** (Retrieves both documents)     | Single queries asking about two distinct topics average out in embedding space, failing to reach the top-k threshold for the secondary document. Query decomposition splits the prompt into targeted sub-queries.                                      |
| **Citation Strictness & Traceability**<br>_(Format: `[chunk_id, title]`)_      |        Inconsistent formatting        |      **100% Strict Inline Footnotes**      | The Truth Verifier inspects chunk metadata and formats citations consistently as `[chunk_id, title]`.                                                                                                                                                  |

---

## 🚀 Step-by-Step Setup, Execution & Verification Guide

### 1. Prerequisites

- **Python:** Version `3.10`, `3.11`, or `3.12`.
- **Groq API Key:** Free API key from [Groq Console](https://console.groq.com).
- **LangSmith API Key (Optional):** Tracing key from [LangSmith](https://smith.langchain.com).

### 2. Environment Setup

Clone the repository and create an isolated virtual environment:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Redesigned Dependencies

Install all pinned, audited dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` configuration file from the provided template:

```powershell
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` to include your credentials:

```dotenv
# Required LLM Credentials
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b

# Security & Session Authentication
JWT_SECRET=asterline-secure-jwt-key-batch5-2026

# Optional LangSmith Observability Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGCHAIN_PROJECT=asterline-batch5-assistant
```

### 5. Automated Rubric Evaluation Suite

Run the comprehensive, reproducible evaluation benchmark:

```bash
python evaluate.py
```

**Expected Test Output:**

```
================ MEMORY BEHAVIOUR & ISOLATION EVALUATION ================
[PASS] Selective Memory & Secret/Medical Redaction -> Stored: {'project_region': 'US', 'workspace_plan': 'Launch'}
[PASS] Cross-User Isolation (Bob sees nothing of Alice's memory) -> Bob's Memory: {}
[PASS] Memory Correction (Launch -> Team) -> Stored: {'project_region': 'US', 'workspace_plan': 'Team'}
[PASS] Follow-up Continuity (Used 'Team' plan to find 1500 req/s in ingest:1)
[PASS] Company Evidence Precedence (Ignored user's $10 claim, cited $449 from pricing:0)
[PASS] Surgical Forgetting ('project_region' removed, plan kept) -> Stored: {'workspace_plan': 'Team'}

================ RESEARCH EVALUATION: V1 (Vector Only) vs V2 (Hybrid + Decomposition) ================

--- Running Mode: HYBRID ---
[PASS] Conflict 1: Weekly On-Call Stipend Supersession (4/4 criteria: ['175', 'handbook:1', '140', 'oncall:0'])
[PASS] Conflict 2: Cold Recovery Storage Retention Supersession (4/4 criteria: ['20', 'retention:0', '60'])
[PASS] Multi-Part: INC-2025-12 Cause & Fix (5/5 criteria: ['Heron', '2.1.2', 'export_job_id', 'record_hash', 'incident-12'])
[PASS] Honest Uncertainty: Enterprise Price & APAC Region (3/3 criteria: ['negotiated', 'pricing:0', 'regions:2'])
Total HYBRID Pass Rate: 4/4 (100%)
```

### 6. Launch the Web Application

Start the production FastAPI backend with hot-reload enabled:

```bash
uvicorn app:app --reload --port 8000
```

Open **`http://localhost:8000`** in any modern web browser.

---

## 🧪 Manual Web UI Testing Checklist

| Test Scenario                | Action in UI                                                                            | Expected System Behavior                                                                                                         |
| :--------------------------- | :-------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| **1. Authentication**        | Click "Create Account" -> Register `alice` / `password123`.                             | Account registered with bcrypt hash; JWT token stored in browser session; user workspace opens.                                  |
| **2. Persistent Memory**     | Send `"Hi, I am on the Launch plan in the US region."`                                  | Assistant acknowledges context; opens **Settings** modal and shows `workspace_plan=Launch` and `project_region=US`.              |
| **3. Multi-Session Chats**   | Click **"+ New Chat"** or press `⌘K` / `Ctrl+K`. Send `"What is our retention policy?"` | A new chat thread appears in the sidebar under "RECENT CONVERSATIONS". Previous chat remains intact and selectable.              |
| **4. Secret Redaction**      | Send `"My ingestion key is ast_ingest_9921_secret. What is our weekly stipend?"`        | Key is redacted to `[REDACTED_SECRET]` before hitting logs or LLM. Memory blocks the secret; assistant answers stipend question. |
| **5. Memory Correction**     | In a new session, send `"We upgraded to the Team plan."`                                | Assistant updates the memory slot. Opening **Settings** confirms `workspace_plan` updated to `Team` across all sessions.         |
| **6. Strict User Isolation** | Log out and register `bob` / `password123`.                                             | Bob's sidebar has **zero sessions**, and Bob's Settings shows **no memory slots**. Alice's data is completely hidden.            |

---

## 📁 Repository Structure

```text
asterline-assistant/
├── .env.example             # Documented environment variables template
├── .gitignore               # Strict git exclusion rules (stores, caches, environments)
├── app.py                   # FastAPI production REST API and static asset router
├── evaluate.py              # Automated 10-point evaluation test suite
├── requirements.txt         # Grouped and audited production dependencies
├── README.md                # System documentation, architecture & verification guide
├── data/
│   ├── corpus.jsonl         # Authoritative company documentation chunks (Cabinet 1 source)
│   ├── chroma_store/        # Persistent ChromaDB vector index (Auto-initialized on boot)
│   └── memory.db            # Persistent SQLite memory & chat history (Auto-initialized on boot)
├── src/
│   ├── agents.py            # LangGraph multi-agent orchestration (Agents 1, 2, 3)
│   ├── auth.py              # JWT token issuance, verification, and bcrypt password hashing
│   ├── guardrails.py        # Pre-execution sanitizer (secrets, PII, medical, fact overrides)
│   ├── memory_db.py         # SQLite persistence for Cabinet 2 user slots & chat sessions
│   └── retriever.py         # Cabinet 1 hybrid retrieval engine (ChromaDB + BM25Okapi + RRF)
└── static/
    └── index.html           # Modern Obsidian-themed SPA (Tailwind CSS, Lucide icons, Marked.js)
```

---

## 🔒 Security & Packaging Compliance

- **No Secrets in Source:** API keys, JWT secrets, and tokens are strictly loaded through environment variables.
- **Zero Runtime Bloat:** Runtime databases (`data/memory.db`, `data/chroma_store/`) and compiled bytecode (`__pycache__/`) are untracked and auto-generated dynamically on fresh initialization.
- **Licensing & Academic Integrity:** Built as a candidate technical assessment demonstration for Asterline Labs.
