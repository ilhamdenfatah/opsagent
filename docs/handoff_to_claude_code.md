# Handoff: Day 10 — Switching to Claude Code

*Created: May 2026, end of Day 10 prep session*
*Purpose: Capture full project state so Claude Code can pick up without context loss.*

---

## TL;DR for Claude Code

Ilham is building **OpsAgent** — a multi-agent AI system for business metrics monitoring. Phase 1 done, Phase 2 Day 9 done (LangGraph skeleton with dummy nodes). Today is **Day 10**: replace the dummy `diagnose_node` with a real Root Cause Analyzer agent built as a LangGraph subgraph implementing ReAct, using Groq + Llama 3.3 70B.

Before Day 10 implementation can start, there's **one blocker**: the Qdrant vector store collection is missing (was lost during dev environment churn). Re-ingest is the first task.

---

## Where we are (as of this handoff)

### Phase status

| Phase | Day | Status |
|---|---|---|
| Phase 1 | 1-8 | ✅ Complete (data pipeline, RAG, Signal Detector) |
| Phase 2 | 9 | ✅ Complete (LangGraph skeleton + 3 dummy nodes + conditional edge) |
| **Phase 2** | **10** | 🟡 **In progress** (this handoff) |

### Day 10 prep work completed in chat session

1. **SQLite database built and loaded.**
   - File: `data/processed/metrics.db`
   - Table: `metrics_daily`, 180 rows
   - Schema:
     ```
     date              TIMESTAMP  (format: 'YYYY-MM-DD HH:MM:SS')
     daily_revenue     REAL       (IDR, baseline ~50M)
     order_count       INTEGER    (baseline ~800)
     customer_churn_rate     REAL    (percentage, baseline ~2.5)
     support_ticket_count    INTEGER (baseline ~45)
     conversion_rate         REAL    (percentage, baseline ~3.2)
     avg_order_value         REAL    (derived: revenue / order_count)
     ```
   - Index on `date` column already created.
   - Loader script: `scripts/csv_to_sqlite.py` (idempotent, can re-run safely).

2. **Pydantic schemas defined.**
   - File: `src/agents/schemas.py`
   - Main model: `RootCauseAnalysis` with rich output (primary cause, confidence, evidence, related metrics, investigation trace, alternative hypotheses, counterfactuals, severity assessment).
   - **Critical design decision**: alternative_hypotheses and counterfactuals are **confidence-gated** via `field_validator` — required only when `confidence >= 0.6`. Below that threshold, agent is allowed to admit uncertainty rather than fabricate analysis.
   - Supporting models: `InvestigationStep`, `Evidence`, `AlternativeHypothesis`, `CounterfactualCheck`.

3. **Inspection scripts (sys.path hack for one-off use).**
   - `scripts/inspect_db.py` — verifies SQLite state.
   - `scripts/inspect_qdrant.py` — verifies Qdrant state. **NOTE: collection `opsagent_metrics` is currently missing.**

4. **Config update.**
   - Added `METRICS_DB_FILE = PROCESSED_DATA_DIR / "metrics.db"` to `src/config.py`.

### What's still BLOCKED

**Qdrant collection `opsagent_metrics` does not exist.**
- Qdrant container itself is running (`http://localhost:6333` responds).
- Volume mount is set up correctly (`./qdrant_storage` → `/qdrant/storage`).
- But the collection is empty. Most likely cause: original collection was created in an earlier container (pre-volume) that was later removed.
- **Action needed**: build an idempotent ingest script (`scripts/build_vectorstore.py`) and run it. The Day 4 ingest logic likely lives in a Jupyter notebook (`notebooks/02_rag_experiments.ipynb`) — promote it to a proper script.

---

## Day 10 implementation plan (decided in chat)

### Architectural decisions already made

1. **ReAct loop implementation: LangGraph subgraph.**
   - Not custom while-loop, not AgentExecutor.
   - Rationale: LangGraph appears explicitly in 2026 AI Engineer job descriptions; subgraph pattern is portfolio signal; natural extension of Day 9 knowledge.
   - Trade-off accepted: more code (~150-200 lines) vs custom loop (~50 lines), justified by industry alignment and portfolio depth.

2. **Tools available to the agent (3 total):**
   - **`query_metrics`** — SQL query against `metrics_daily` table in `data/processed/metrics.db`.
   - **`calculate_metric_stats`** — statistical analysis: z-score, day-over-day change, correlation between metrics.
   - **`retrieve_similar_events`** — RAG retrieval from Qdrant `opsagent_metrics` collection.
   - Web search deliberately **excluded** — dataset is synthetic, no real-world events to cross-reference.

3. **Output: Rich structured output (already implemented in `schemas.py`).**
   - With guardrails: confidence-gating prevents hallucination of alternative hypotheses at low confidence.
   - Evidence MUST cite source (sql_query / metric_calculator / rag_retriever / reasoning).

4. **Model: Llama 3.3 70B via Groq.**
   - Already configured in `src/config.py`: `AGENT_MODEL_ROUTING["root_cause_analyzer"] = MODEL_PRIMARY`.
   - Groq supports OpenAI-compatible tool calling natively. No regex parsing.

### Build sequence

| Step | File(s) | What |
|---|---|---|
| 0 | `scripts/build_vectorstore.py` (NEW) | Re-ingest Qdrant. Idempotent. **Run this first.** |
| 1 | (verify) | Run `scripts/inspect_qdrant.py` to confirm collection is back. |
| 2 | `src/tools/data_query.py` (NEW) | SQL query tool with safety (read-only, parameterized). |
| 3 | `src/tools/metric_calculator.py` (NEW) | z-score, day-over-day change, correlation utilities. |
| 4 | `src/tools/rag_retriever.py` (NEW) | Thin wrapper around existing `src/rag/retriever.py` for tool interface. |
| 5 | `src/agents/root_cause_subgraph.py` (NEW) | LangGraph subgraph: ReAct loop with tool dispatcher. |
| 6 | `src/agents/root_cause.py` (NEW) | Public agent interface — node function that wraps the subgraph. |
| 7 | `src/pipeline/nodes.py` (EDIT) | Replace dummy `diagnose_node` with the real one. |
| 8 | `src/pipeline/state.py` (EDIT) | Add `root_cause_analysis: RootCauseAnalysis` field. |
| 9 | `tests/test_root_cause.py` (NEW) | Smoke test with a known anomaly from ground truth. |

---

## Project conventions (must follow)

### Code style (enforced by CLAUDE.md)

- **Type hints on all signatures.** `def foo(x: int) -> str:` not `def foo(x):`.
- **Docstrings on public functions.** Concise, useful, human-written. Explain *why* not *what*.
- **Pydantic for all structured data.** No raw dicts crossing module boundaries.
- **Config via `src/config.py`.** No magic numbers, no hardcoded paths.
- **Error handling explicit.** Catch specific exceptions, log meaningfully, decide retry vs fail.

### Anti-AI-ish rules (CRITICAL — Ilham flagged this explicitly)

- ❌ NO comments like "This robust function efficiently handles..."
- ❌ NO variable names like `metricsAnalysisRequestObject`
- ❌ NO emoji clusters in code or comments
- ❌ NO excessive comments explaining obvious things (`# increment counter` for `counter += 1`)
- ❌ NO corporate-speak: "Note that this approach ensures optimal..."
- ❌ NO README sections like "## ✨ Features" with marketing copy

- ✅ DO write comments that explain non-obvious *why*: `# Index on date — every agent query filters by date, so this stays fast`
- ✅ DO use concise variable names: `metrics_df`, `latest_anomaly`
- ✅ DO write like a thoughtful developer leaving notes for the next reader

### Imports and structure

- All `src/` code uses absolute imports: `from src.config import ...`
- Scripts in `scripts/` use a sys.path hack at the top (existing pattern, see `scripts/csv_to_sqlite.py`).
- `from __future__ import annotations` at the top of every Python file.

---

## Tech stack (for reference)

- **Python 3.11+**, package manager: pip (editable install via pyproject.toml).
- **LangGraph 1.1.10** + **langchain-core 1.3.2** + **langchain-groq 1.1.2** (1.x ecosystem, clean).
- **Groq SDK 0.37.1** for LLM inference (Llama 3.3 70B primary, Llama 3.1 8B fast/fallback).
- **Qdrant** (Docker, local) + **sentence-transformers** (all-MiniLM-L6-v2) for RAG.
- **SQLite** for structured queries.
- **Pydantic 2.x** for schemas.
- **Pytest** for testing.

---

## Resources and references

### Files Claude Code should read first (priority order)

1. `CLAUDE.md` (project root) — coding conventions and project rules.
2. `OpsAgent-Build-Plan.md` (project root) — full 28-day build plan, especially Day 10 section.
3. `src/config.py` — model routing, paths, thresholds.
4. `src/pipeline/state.py`, `src/pipeline/nodes.py`, `src/pipeline/graph.py` — existing Day 9 skeleton.
5. `src/agents/schemas.py` — output contracts.
6. `src/rag/retriever.py`, `src/rag/vector_store.py`, `src/rag/chunking.py`, `src/rag/embeddings.py` — for building rag_retriever tool and ingest script.

### Ground truth for testing

- `data/raw/anomaly_ground_truth.json` — 15 planted anomalies with documented causes.
- Use anomaly `ANM-001` (sudden revenue drop on 2025-11-15) as the primary smoke test case for Day 10.

---

## Communication style with Ilham

- **Language**: Bahasa Indonesia campur English, casual and warm.
- **Style**: "Vibe Coder" — Ilham describes intent, Claude Code writes the code, but **always explain what & why before/after coding**.
- **Don't be cheerleader.** Honest trade-offs, push back when needed, flag risks.
- **Address Ilham as "kaki" or "Ilham".** Claude can be called "Klo" or "Kaki".
- Phrases: "gas" / "lanjut" = proceed. "jelasin dong" = explain deeper.

---

## What Ilham needs from Claude Code TODAY

1. **Unblock**: build `scripts/build_vectorstore.py`, re-ingest Qdrant, verify.
2. **Step 2-4**: build 3 tools (`data_query`, `metric_calculator`, `rag_retriever`).
3. **Pause point**: before Step 5 (subgraph), Ilham wants to review the tool implementations.

Ilham will be working interactively in Claude Code, returning to chat session if architecture questions arise. Mode is **medium pendampingan** for Day 10-12 (Ilham is new to Claude Code).

---

*End of handoff. Welcome to the project, Claude Code.*
