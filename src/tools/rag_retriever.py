"""
RAG retrieval tool for agents.

Thin wrapper around src/rag/retriever.py. Agents call this — not
retriever.py directly — so the tool layer stays consistent across
all three tools (SQL, stats, RAG).

The only real work here is converting RetrievalResult dataclasses
to Pydantic models and bundling the formatted context string alongside
the structured results, so agents don't have to call format_context_for_prompt
separately.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import TOP_K_RETRIEVAL
from src.rag.retriever import (
    RetrievalResult,
    format_context_for_prompt,
    retrieve,
)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class RAGResult(BaseModel):
    """One retrieved chunk — Pydantic mirror of retriever.RetrievalResult."""

    text: str
    score: float = Field(ge=0.0, le=1.0)
    chunk_type: str = Field(description="'weekly_summary' or 'anomaly_event'")
    metadata: dict


class RAGQueryResult(BaseModel):
    """
    Full result of a retrieval call.

    Agents get both structured results (for building Evidence objects)
    and a pre-formatted context string (for injecting into LLM prompts).
    """

    query_text: str
    results: list[RAGResult]
    result_count: int
    formatted_context: str = Field(
        description="Ready-to-inject string for LLM prompt context"
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_rag_result(r: RetrievalResult) -> RAGResult:
    return RAGResult(
        text=r.text,
        score=r.score,
        chunk_type=r.chunk_type,
        metadata=r.metadata,
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def retrieve_similar_events(
    query_text: str,
    top_k: int = TOP_K_RETRIEVAL,
    chunk_type: str | None = None,
    metric: str | None = None,
    severity: str | None = None,
) -> RAGQueryResult:
    """
    Retrieve historical context most similar to query_text.

    Args:
        query_text: natural language description of the anomaly or situation
        top_k: how many chunks to return
        chunk_type: optional filter — "weekly_summary" or "anomaly_event"
        metric: optional filter — e.g. "daily_revenue"
        severity: optional filter — e.g. "high", "critical"

    Returns:
        RAGQueryResult with structured results and a formatted context string
    """
    raw_results = retrieve(
        query=query_text,
        top_k=top_k,
        chunk_type=chunk_type,
        metric=metric,
        severity=severity,
    )

    pydantic_results = [_to_rag_result(r) for r in raw_results]

    return RAGQueryResult(
        query_text=query_text,
        results=pydantic_results,
        result_count=len(pydantic_results),
        formatted_context=format_context_for_prompt(raw_results),
    )
