# Multi-Tenant RAG Knowledge Base

> **Zero-Leakage Enterprise Knowledge Base grounded in private tenant documents with citation enforcement.**

Built following the **7-Stage GenAI Production Workflow**:
1. **Requirements**: Solves scattered institutional knowledge with strict tenant data isolation and zero hallucination.
2. **Scoping**: Focused on high-recall org-scoped retrieval, PDF/TXT/MD chunking, SSE streaming Q&A, and citation enforcement.
3. **Planning (Riskiest First)**: Verified cross-tenant vector isolation in Week 1 with automated leakage test suites.
4. **Architecture**: Defense-in-depth with vector store metadata filtering (`where={"org_id": current_tenant}`), unified LLM client (Gemini 2.0 Flash → Groq → OpenAI → Mock), and graceful degradation.
5. **Project Setup**: Versioned prompt templates, mock LLM fixtures for CI, and cost/token tracking.
6. **Task Breakdown**: Vertical slices from input to database to vector search to streaming answer.
7. **Production Dev & Eval**: 20+ item Golden Evaluation Suite measuring Retrieval Recall, Grounding Accuracy, and Latency.

---

## 🚀 Key Features

- 🛡️ **Zero-Leakage Vector Scoping**: Vector database queries are strictly scoped at the storage layer via `WHERE metadata.org_id = current_org_id`. Org A cannot retrieve Org B's private documents even on identical topics.
- ⚡ **Unified Multi-Provider LLM Client**:
  - **Primary**: Gemini 2.0 Flash
  - **Secondary**: Groq (`llama-3.3-70b-versatile`)
  - **Tertiary**: OpenAI (`gpt-4o-mini`)
  - **Deterministic Mock**: For offline testing and CI environments.
- 💬 **Real-Time Streaming with SSE**: Token-by-token streaming with Server-Sent Events (`/api/chat/stream`).
- 📄 **Source Citation Enforcement**: Non-negotiable citations with inline tags `[Doc: filename, Chunk: idx]` and clickable citation drawers.
- 🛡️ **Graceful Degradation**: If all LLM APIs are unreachable, the system gracefully serves the raw retrieved context chunks with a clear notice rather than crashing.
- 📊 **Golden Eval Suite**: Automated test runner evaluating 20+ test cases across recall, grounding, and isolation.
- 🎨 **Modern Dark-Mode Web Dashboard**: Glassmorphism UI with Tenant Switcher, Document Drag-and-Drop Ingestion, Real-Time Chat, Isolation Proof Lab, and Metrics scorecards.

---

## 📁 Repository Structure

```
├── app/
│   ├── main.py              # FastAPI app entry point & seed lifecycle
│   ├── config.py            # Pydantic environment configuration
│   ├── auth/                # Tenant identification middleware & onboarding router
│   ├── db/                  # Relational database models (SQLite/PostgreSQL)
│   ├── schemas/             # Pydantic schemas for requests, responses & citations
│   ├── documents/           # Upload, parsing, chunking & vector upsert router
│   ├── retrieval/           # ChromaDB vector retrieval service with org_id scoping
│   ├── chat/                # RAG Q&A endpoints & SSE streaming
│   ├── eval/                # Evaluation & Live Isolation check endpoints
│   └── static/              # Modern web dashboard (HTML5, Vanilla CSS, JS)
├── services/
│   ├── llm.py               # Unified LLM Client with multi-provider fallback
│   ├── embeddings.py        # Local sentence-transformers & deterministic vectorizer
│   ├── chunking.py          # Recursive document splitting (500 tokens / 50 overlap)
│   ├── cost_tracker.py      # Token counting and cost estimation logger
│   └── prompts/             # Versioned prompt templates (system_v1, qa_v1, refusal_v1)
├── tests/
│   ├── fixtures/            # Sample confidential documents for Org A & Org B
│   ├── mocks/               # Deterministic mock LLM for CI
│   ├── test_isolation.py    # CRITICAL: Week 1 Cross-tenant leakage tests
│   ├── test_grounding.py    # Hallucination & refusal tests
│   ├── test_chunking.py     # Chunk boundary & overlap unit tests
│   └── test_api.py          # End-to-end FastAPI integration tests
├── eval/
│   ├── golden_set.json      # 20+ hand-labeled test cases across tenants
│   └── run_eval.py          # Standalone CLI evaluation runner
├── docker-compose.yml       # Docker deployment orchestration
├── requirements.txt         # Pinned Python dependencies
└── .env.example             # Configuration template
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
# Clone and enter directory
cd "Multi-Tenant RAG Knowledge Base"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY
# If left blank, the system automatically uses the deterministic Mock engine!
```

### 3. Run FastAPI Application
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to: **`http://127.0.0.1:8000/`**

---

## 🧪 Testing & Validation

### Run Critical Cross-Tenant Isolation Tests (Week 1 Gate)
```bash
pytest tests/test_isolation.py -v
```

### Run Full Test Suite
```bash
pytest tests/ -v
```

### Run Golden Evaluation Suite
```bash
python eval/run_eval.py
```
