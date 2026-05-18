"""Quick inspect: print Qdrant collection info + sample points."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `from src...` works.
# Hack only acceptable for one-off scripts — never in src/ proper.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_COLLECTION_NAME

client = QdrantClient(url=QDRANT_URL)

# 1. Collection info
info = client.get_collection(QDRANT_COLLECTION_NAME)
print(f"=== COLLECTION: {QDRANT_COLLECTION_NAME} ===")
print(f"Total points: {info.points_count}")
print(f"Vector size: {info.config.params.vectors.size}")
print(f"Distance: {info.config.params.vectors.distance}")

# 2. Sample 3 points to see payload structure
print("\n=== SAMPLE POINTS ===")
points, _ = client.scroll(
    collection_name=QDRANT_COLLECTION_NAME,
    limit=3,
    with_payload=True,
    with_vectors=False,  # skip vectors, terlalu panjang
)

for p in points:
    print(f"\n🔹 Point ID: {p.id}")
    print(f"  Payload keys: {list(p.payload.keys())}")
    print(f"  Payload sample:")
    for k, v in p.payload.items():
        # Truncate long values
        v_str = str(v)
        if len(v_str) > 150:
            v_str = v_str[:150] + "..."
        print(f"    {k}: {v_str}")