"""
Retriever — the interface agents use to fetch relevant historical context.

Agents don't call vector_store.py directly. They call this module,
which handles the full retrieve pipeline:
    query text → embed → search Qdrant → format results → return to agent

Also handles ingestion: chunking the dataset and loading it into Qdrant.
"""

import pandas as pd
from dataclasses import dataclass

from src.rag.embeddings import embed_text, embed_batch
from src.rag.vector_store import (
    upsert_vectors,
    search_similar,
    get_collection_info,
    ensure_collection_exists,
    get_qdrant_client,
)
from src.rag.chunking import build_all_chunks, load_data, Chunk
from src.config import QDRANT_COLLECTION_NAME, TOP_K_RETRIEVAL


@dataclass
class RetrievalResult:
    """A single retrieved context item, ready to be injected into a prompt."""
    text: str
    score: float
    chunk_type: str       # "weekly_summary" | "anomaly_event"
    metadata: dict


def ingest_dataset(collection_name: str = QDRANT_COLLECTION_NAME) -> int:
    """
    Full ingestion pipeline: load data → chunk → embed → store in Qdrant.

    Run this once to populate the vector store. Safe to re-run —
    it will recreate the collection from scratch each time.

    Returns:
        Total number of vectors stored
    """
    print("Starting dataset ingestion...")

    # Load data
    df, ground_truth = load_data()
    print(f"Loaded {len(df)} days of metrics data")

    # Build chunks
    chunks = build_all_chunks(df, ground_truth)

    # Embed all chunks in one batch (faster than one by one)
    print(f"\nEmbedding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    vectors = embed_batch(texts, batch_size=32)

    # Prepare metadata
    metadata_list = [
        {"chunk_type": c.chunk_type, **c.metadata}
        for c in chunks
    ]

    # Wipe and recreate collection for clean ingestion
    client = get_qdrant_client()
    from src.rag.vector_store import delete_collection
    try:
        delete_collection(collection_name)
    except Exception:
        pass
    ensure_collection_exists(client, collection_name)

    # Store in Qdrant
    count = upsert_vectors(
        texts=texts,
        vectors=vectors,
        metadata=metadata_list,
        collection_name=collection_name,
    )

    print(f"\n✓ Ingested {count} chunks into '{collection_name}'")
    return count


def retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    chunk_type: str | None = None,
    metric: str | None = None,
    severity: str | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant historical context for a query.

    This is what agents call when they need historical context.
    Example: retrieve("revenue dropped significantly last week")

    Args:
        query: natural language description of what you're looking for
        top_k: how many results to return
        chunk_type: optional filter — "weekly_summary" or "anomaly_event"
        metric: optional filter — e.g. "daily_revenue"
        severity: optional filter — e.g. "high", "critical"

    Returns:
        List of RetrievalResult sorted by relevance (highest score first)
    """
    # Build optional metadata filters
    filter_by = {}
    if chunk_type:
        filter_by["chunk_type"] = chunk_type
    if metric:
        filter_by["metric"] = metric
    if severity:
        filter_by["severity"] = severity

    query_vector = embed_text(query)

    raw_results = search_similar(
        query_vector=query_vector,
        top_k=top_k,
        filter_by=filter_by if filter_by else None,
    )

    return [
        RetrievalResult(
            text=r["text"],
            score=r["score"],
            chunk_type=r.get("chunk_type", "unknown"),
            metadata={k: v for k, v in r.items()
                      if k not in ("text", "score", "chunk_type")},
        )
        for r in raw_results
    ]


def format_context_for_prompt(results: list[RetrievalResult]) -> str:
    """
    Format retrieval results into a clean string for injecting into an LLM prompt.

    Args:
        results: list of RetrievalResult from retrieve()

    Returns:
        Formatted string ready to paste into a system or user prompt
    """
    if not results:
        return "No relevant historical context found."

    lines = ["## Relevant Historical Context\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"**[{i}] Relevance score: {r.score:.3f}**")
        lines.append(r.text)
        lines.append("")  # blank line between results

    return "\n".join(lines)


def get_ingestion_status(collection_name: str = QDRANT_COLLECTION_NAME) -> dict:
    """Check if the collection exists and how many vectors are in it."""
    return get_collection_info(collection_name)
