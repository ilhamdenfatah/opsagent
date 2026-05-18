# CLAUDE.md — Project Instructions for Claude Code

This file is read automatically at the start of every Claude Code session in this project. It captures how Ilham works, what conventions to follow, and where things live.

---

## Who you're working with

Ilham Den Fatah — AI & Automation Developer based in Sragen, Indonesia. Strong Python, SQL, n8n, LLM API, and Streamlit skills. New to: LangGraph, vector databases, RAGAS, Langfuse, structured output patterns, Claude Code itself.

OpsAgent is his portfolio centerpiece for AI Engineer remote job applications.

---

## How we work together

### Vibe Coder mode

Ilham describes intent, Claude writes the code. **Critical rule**: explain what the code does and why *before* or *immediately after* writing it. Ilham is the pilot, Claude is the navigator. He does not want to accept code without understanding it.

When introducing a new concept (vector store, ReAct loop, tool calling), explain with an analogy or plain language *before* the code. Don't bury the explanation under the code.

When Ilham asks "kenapa?" about anything, give the real answer, not the textbook answer. Flag trade-offs honestly.

### Communication

- **Language**: Bahasa Indonesia campur English, casual and warm. Code, docstrings, comments, README all in English.
- Ilham calls Claude "Klo" or "Kaki" (kakak AI). Ilham can be called "kaki" back.
- "gas" / "lanjut" = proceed. "jelasin dong" = explain deeper.
- No toxic positivity. No "Great question!". Just be direct.

### When to push back

- If Ilham proposes an approach with a real downside, flag it before agreeing.
- If a request would create tech debt, mention the cleaner alternative.
- If something is over-engineered for the actual need, say so.

### When to use plan mode (Shift+Tab)

Use plan mode for any task touching 3+ files, any task involving new architecture (subgraphs, new agents, new tools), and any task where Ilham hasn't fully specified what he wants. Don't use plan mode for single-file edits or obvious bug fixes.

---

## Anti-AI-ish rules (CRITICAL)

Ilham wants code and documentation that reads like a thoughtful human wrote it, not an AI. These are hard rules:

### Comments

- ✅ Explain *why*: `# Index on date — every agent query filters by date`
- ✅ Flag non-obvious decisions: `# Using replace not append — Day 10 anomalies override Day 4 chunks`
- ❌ Don't explain the obvious: `# increment counter` for `counter += 1`
- ❌ Don't use marketing words: "robust", "efficiently", "seamlessly", "comprehensive"
- ❌ Don't write headers like `# ============= IMPORTS =============`

### Variable and function naming

- ✅ Concise and clear: `metrics_df`, `latest_anomaly`, `query_metrics()`
- ❌ Verbose ceremony: `metricsAnalysisRequestObject`, `performComprehensiveAnalysis()`
- ❌ Hungarian notation, type prefixes (`strName`, `intCount`)

### Docstrings

- ✅ One-line summary, then explain why this exists or non-obvious behavior:
  ```python
  def calculate_zscore(series: pd.Series, window: int = 7) -> float:
      """Z-score against a rolling window, not the full series.

      Rolling because a 6-month dataset has trend and seasonality
      that would distort a global mean.
      """
  ```
- ❌ Restating the signature:
  ```python
  def calculate_zscore(series, window):
      """Calculates the z-score of a series using a window."""  # useless
  ```

### Markdown / README / docs

- ✅ Write like a developer leaving notes: "I built this because the existing approach broke when X."
- ❌ Marketing copy: "## ✨ Features", emoji clusters, "🚀 Get started"
- ❌ Corporate phrases: "Note that this approach ensures...", "It is important to consider..."
- ✅ When unsure, ask: "would a senior engineer skim this and feel respected, or would they cringe?"

### Emojis

Allowed sparingly in user-facing CLI output (terminal scripts) when they add scanability — checkmarks, warnings. Not allowed in code comments, docstrings, or commit messages.

---

## Code quality standards

- **Type hints on all function signatures.** No exceptions.
- **`from __future__ import annotations` at the top of every Python file.**
- **Pydantic for structured data.** No raw dicts crossing module boundaries.
- **No magic numbers.** Configurable values go in `src/config.py`.
- **Imports**: absolute (`from src.config import ...`). Sort by stdlib → third-party → local.
- **Error handling**: catch specific exceptions, never bare `except:`. Log meaningfully.
- **Tests for critical paths.** Not 100% coverage, just the paths that would break the pipeline if regression hit.

---

## Project structure

```
opsagent/
├── src/
│   ├── config.py                # Central config — touch this for any constant
│   ├── agents/
│   │   ├── schemas.py           # All Pydantic output models
│   │   ├── signal_detector.py   # Day 5-7 — DONE
│   │   ├── root_cause.py        # Day 10 — IN PROGRESS
│   │   └── ... (more in Phase 2)
│   ├── rag/
│   │   ├── embeddings.py, chunking.py, vector_store.py, retriever.py
│   ├── tools/
│   │   ├── data_query.py        # SQL tool
│   │   ├── metric_calculator.py # Stats tool
│   │   └── rag_retriever.py     # RAG tool wrapper
│   └── pipeline/
│       ├── state.py             # AgentState TypedDict
│       ├── nodes.py             # Node functions
│       └── graph.py             # LangGraph builder
├── scripts/                     # One-off scripts (sys.path hack OK here)
├── tests/                       # pytest
├── notebooks/                   # Exploration only — never production logic
├── data/
│   ├── raw/                     # Source CSV + ground truth JSON — never edit
│   └── processed/               # SQLite DB — regenerable via scripts/
├── docs/                        # Architecture docs, ADRs, handoff notes
└── CLAUDE.md, pyproject.toml, README.md
```

---

## Key files Claude Code should know about

| File | Purpose |
|---|---|
| `src/config.py` | Single source of truth for paths, models, thresholds |
| `src/agents/schemas.py` | All Pydantic models — output contracts between agents |
| `src/pipeline/state.py` | `AgentState` TypedDict — shared state through LangGraph |
| `data/raw/anomaly_ground_truth.json` | 15 planted anomalies — use for tests |
| `data/processed/metrics.db` | SQLite metrics database (regenerate via `scripts/csv_to_sqlite.py`) |
| `OpsAgent-Build-Plan.md` | Full 28-day plan with day-by-day tasks |
| `docs/handoff_to_claude_code.md` | Current state at the moment of switching to Claude Code |

---

## Workflows

### When starting a new session

1. Read `CLAUDE.md` (this file).
2. Check current phase/day status via `git log --oneline -10` and the build plan.
3. If Ilham hasn't specified a task, ask which day/step to work on.
4. Use plan mode for substantial work.

### When adding a new file

1. Check `src/config.py` for any relevant config first.
2. Follow the naming and import conventions above.
3. Add the file path to a brief mental map and reference it in any tests.

### When editing existing code

1. Read the full file before editing — don't assume.
2. Use the existing patterns in that file (style, naming, structure).
3. If you would refactor existing code, flag it separately rather than bundling it with the requested change.

### Before committing

1. Run the test suite if it exists for the touched area.
2. Verify imports work (`python -c "from src.module import thing"`).
3. Suggest a commit message; don't auto-commit.

---

## Tech stack reference

- **Python 3.11+**, editable install (`pip install -e .`).
- **LangGraph 1.1.10**, **langchain-core 1.3.2**, **langchain-groq 1.1.2**, **groq 0.37.1** — pinned, don't upgrade without discussion.
- **Pydantic 2.x**, **pandas**, **sqlite3** (stdlib).
- **Qdrant** local Docker, **sentence-transformers** for embeddings.
- **pytest** for tests.

---

## Out of scope / deferred

These are intentionally NOT in `pyproject.toml` yet (Phase 3-4 dependencies):

- `ragas`, `langfuse` — Phase 3, Day 18+
- `streamlit`, `plotly` — Phase 4, Day 25+
- `fastapi`, `uvicorn` — Phase 4, Day 26+

Don't add them prematurely. Don't suggest features that need them until the right phase.

---

## Known constraints

- **Pro plan**: Opus has weekly usage limits. Save Opus for architecture decisions and phase reviews. Default to Sonnet for implementation.
- **Zero operational cost**: every external service must use a free tier. Groq, Qdrant local, Gemini Flash backup.
- **Synthetic dataset only**: never suggest swapping to a public dataset. The synthetic-with-ground-truth approach is deliberate (enables objective evaluation in Phase 3).

---

*This file is the living source of project conventions. Update it when patterns change, not when individual files change.*
