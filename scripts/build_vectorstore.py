"""
Re-ingest the metrics dataset into Qdrant.

Run this whenever the vector store collection is missing or stale.
It's idempotent — ingest_dataset() wipes the collection before re-populating,
so running it twice gives the same result as running it once.

Typical usage:
    python scripts/build_vectorstore.py

Expected output: X chunks ingested, collection status printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# sys.path hack — only in scripts/, never in src/ proper.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import ingest_dataset, get_ingestion_status  # noqa: E402
from src.config import QDRANT_COLLECTION_NAME  # noqa: E402


def main() -> None:
    print(f"Target collection: {QDRANT_COLLECTION_NAME}")

    count = ingest_dataset()

    status = get_ingestion_status()
    print(f"Collection status: {status}")

    if count == 0:
        print("WARNING: 0 vectors ingested — check chunking logic and data files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
