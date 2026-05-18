"""Quick inspect: print SQLite schema + sample rows."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path so `from src...` works.
# Hack only acceptable for one-off scripts in scripts/ — never in src/ proper.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402

DB_PATH = PROCESSED_DATA_DIR / "metrics.db"


def main() -> None:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        print(f"⚠️  Database is empty or missing at: {DB_PATH}")
        print(f"   Run: python scripts/csv_to_sqlite.py")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=== TABLES ===")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    if not tables:
        print("(no tables found — DB is empty)")
        conn.close()
        return

    for (table_name,) in tables:
        print(f"\n📋 {table_name}")

        cur.execute(f"PRAGMA table_info({table_name});")
        print("  Columns:")
        for col in cur.fetchall():
            print(f"    - {col[1]}: {col[2]}")

        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        print(f"  Row count: {cur.fetchone()[0]}")

        cur.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        print(f"  Sample rows:")
        for row in cur.fetchall():
            print(f"    {row}")

    conn.close()


if __name__ == "__main__":
    main()