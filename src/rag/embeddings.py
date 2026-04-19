"""
Embedding pipeline for OpsAgent.

Converts text into vectors using sentence-transformers (all-MiniLM-L6-v2).
This model runs locally — no API calls, no cost, no rate limits.

Why all-MiniLM-L6-v2?
- Fast: ~14k sentences/second on CPU
- Small: 80MB download
- Good enough: 384-dim vectors capture semantic meaning well for our use case
- Free: runs entirely on your machine
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    Uses lru_cache so the model is only loaded once per process —
    loading takes ~2 seconds, we don't want to pay that cost on every call.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a vector.

    Args:
        text: any string to embed

    Returns:
        List of floats (384 dimensions for all-MiniLM-L6-v2)
    """
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of texts efficiently in batches.

    Batching is faster than calling embed_text() in a loop because
    the model can process multiple texts in parallel on the same hardware.

    Args:
        texts: list of strings to embed
        batch_size: how many texts to process at once (tune based on RAM)

    Returns:
        List of vectors, one per input text
    """
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,  # show progress only for large batches
    )
    return vectors.tolist()
