"""
Qdrant vector store interface.

Handles all communication with the Qdrant database:
- Creating collections
- Upserting (insert or update) vectors with metadata
- Searching by semantic similarity
- Deleting vectors

Think of this as the "database driver" layer — agents don't talk to Qdrant
directly, they go through this module.
"""

from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    TOP_K_RETRIEVAL,
)


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client connected to our local instance."""
    return QdrantClient(url=QDRANT_URL)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> None:
    """
    Create the collection if it doesn't exist yet.

    Safe to call multiple times — won't error if collection already exists.
    A "collection" in Qdrant is like a table in SQL: it holds vectors of the
    same dimension with their associated metadata.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,  # cosine similarity = best for text
            ),
        )
        print(f"Created Qdrant collection: '{collection_name}'")
    else:
        print(f"Collection '{collection_name}' already exists — skipping creation")


def upsert_vectors(
    texts: list[str],
    vectors: list[list[float]],
    metadata: list[dict],
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> int:
    """
    Insert or update vectors in the collection.

    "Upsert" = insert if not exists, update if exists.
    Each vector is stored with its original text and any metadata you provide.

    Args:
        texts: original text for each vector (stored for retrieval)
        vectors: pre-computed embeddings
        metadata: dict of additional fields per vector (date, metric, severity, etc.)
        collection_name: which collection to write to

    Returns:
        Number of vectors upserted
    """
    client = get_qdrant_client()
    ensure_collection_exists(client, collection_name)

    points = []
    for text, vector, meta in zip(texts, vectors, metadata):
        payload = {"text": text, **meta}
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_similar(
    query_vector: list[float],
    top_k: int = TOP_K_RETRIEVAL,
    collection_name: str = QDRANT_COLLECTION_NAME,
    filter_by: dict | None = None,
) -> list[dict]:
    """
    Find the most semantically similar vectors to a query.

    Args:
        query_vector: embedding of the search query
        top_k: how many results to return
        collection_name: which collection to search
        filter_by: optional metadata filter, e.g. {"metric": "daily_revenue"}
                   only returns results where that metadata field matches

    Returns:
        List of dicts with keys: text, score, and all metadata fields
    """
    client = get_qdrant_client()

    # Build metadata filter if provided
    qdrant_filter = None
    if filter_by:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_by.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            **{k: v for k, v in hit.payload.items() if k != "text"},
        }
        for hit in results
    ]


def get_collection_info(collection_name: str = QDRANT_COLLECTION_NAME) -> dict:
    """Return basic stats about the collection — useful for debugging."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
    except Exception:
        return {"name": collection_name, "error": "collection not found"}


def delete_collection(collection_name: str = QDRANT_COLLECTION_NAME) -> None:
    """Delete the entire collection. Use with caution — this is irreversible."""
    client = get_qdrant_client()
    client.delete_collection(collection_name)
    print(f"Deleted collection: '{collection_name}'")
