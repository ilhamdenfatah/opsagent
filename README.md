# OpsAgent — AI Operations Co-Pilot

> A multi-agent AI system that monitors business metrics, detects anomalies, diagnoses root causes, and recommends corrective actions.

**Status: 🚧 Phase 1 Complete — Phase 2 in progress**

---

## What This Is

OpsAgent is a production-grade multi-agent AI system built to demonstrate the full stack of skills that AI Engineer roles require in 2026: multi-agent orchestration, RAG pipelines, structured output, evaluation frameworks, and LLM observability.

Five specialized agents. One coordinator. Zero cloud costs.

### The Agent Pipeline

```
[Daily Metrics] → Coordinator
                    ├── Signal Detector      detects anomalies (hybrid: statistical + LLM)
                    ├── Root Cause Analyzer  diagnoses why via tool-use + RAG
                    ├── Action Recommender   ranks corrective actions by impact/effort
                    └── Report Generator     writes executive-grade incident reports
```

### Routing Logic

```
Severity: low      → Signal report only
Severity: medium   → Signal + Root Cause
Severity: high     → Full pipeline (all 4 agents)
Severity: critical → Full pipeline + immediate Telegram alert
```

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Agent orchestration | LangGraph | Explicit state machine, production-grade |
| LLM inference | Groq API (Llama 3.3 70B + 3.1 8B) | Fast, free tier, open-source models |
| Vector database | Qdrant (Docker) | Production architecture, great Python SDK |
| Embeddings | sentence-transformers (local) | Free, no API calls, 384-dim vectors |
| Evaluation | RAGAS + LLM-as-Judge | Industry standard, objective metrics |
| Observability | Langfuse | Full trace of every LLM call |
| Frontend | Streamlit | Fast to build, good enough for demo |
| Automation | n8n + Telegram | Scheduled runs, anomaly alerts |

**Total LLM API cost: $0** (Groq free tier + local embeddings)

---

## Why Synthetic Data?

This project generates its own dataset rather than using a public one. The reason: **objective evaluation requires controlled ground truth.**

```json
{
  "anomaly_id": "ANM-003",
  "date": "2025-12-03",
  "type": "sudden_drop",
  "metric": "daily_revenue",
  "magnitude": "-70%",
  "root_cause": "Payment gateway outage 08:00-20:00",
  "expected_actions": ["Activate backup payment provider", "..."]
}
```

15 anomalies across 4 categories (sudden drops, gradual degradation, spikes, correlated multi-metric). This ground truth becomes the golden test set for Phase 3 evaluation.

---

## Project Structure

```
opsagent/
├── src/
│   ├── config.py              # All config in one place
│   ├── data/                  # Synthetic dataset generation
│   ├── rag/                   # Embeddings + Qdrant + retrieval
│   ├── agents/                # Signal Detector + more coming
│   ├── evaluation/            # RAGAS + LLM-as-Judge (Phase 3)
│   ├── observability/         # Langfuse tracing (Phase 3)
│   └── pipeline/              # LangGraph state machine (Phase 2)
├── data/
│   ├── raw/                   # Generated dataset (gitignored, regenerate locally)
│   └── golden_testset/        # Evaluation test cases
├── tests/                     # pytest test suite
├── app/                       # Streamlit dashboard (Phase 4)
├── n8n/                       # Workflow automation (Phase 4)
└── docs/
    └── architecture.md        # Full system design + Mermaid diagrams
```

---

## Setup

**Requirements:** Python 3.11+, Docker

```bash
# 1. Clone and install
git clone https://github.com/ilhamdenfatah/opsagent.git
cd opsagent
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (free at console.groq.com)

# 3. Start Qdrant
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# 4. Generate dataset and ingest into Qdrant
python -m src.data.build_dataset
python -c "from src.rag.retriever import ingest_dataset; ingest_dataset()"

# 5. Run tests
pytest tests/ -v
```

---

## Current Progress

| Phase | Status | What's Built |
|-------|--------|-------------|
| Phase 1: Foundation | ✅ Complete | Data pipeline, RAG, Signal Detector |
| Phase 2: Multi-Agent | 🚧 In progress | LangGraph, 4 agents, coordinator |
| Phase 3: Evaluation | ⬜ Upcoming | RAGAS, LLM-as-Judge, Langfuse |
| Phase 4: Polish | ⬜ Upcoming | Streamlit, n8n, deploy |

---

## Skills Demonstrated

Built to cover the 8 most common AI Engineer job requirements in 2026:

| Skill | Coverage |
|-------|----------|
| RAG Pipelines | Qdrant + sentence-transformers + chunking strategies |
| Multi-Agent Systems | 5 agents + LangGraph coordinator |
| Production LLM Deployment | Guardrails, retry logic, structured output |
| Evaluation & Observability | RAGAS + LLM-as-Judge + Langfuse (Phase 3) |
| Prompt Engineering | System prompts, few-shot, CoT, JSON mode |
| Tool-Use / Function Calling | Agents query SQL, calculators, search |
| Cost & Latency Optimization | Token tracking, model routing, $0 cost |
| Context Engineering | Token budget management, RAG compression |

---

## Architecture

Full system design with Mermaid diagrams: [`docs/architecture.md`](docs/architecture.md)

---

*Built by Ilham Den Fatah — AI & Automation Developer*
