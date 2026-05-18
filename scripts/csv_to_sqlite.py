"""
One-shot loader: convert metrics_daily.csv → SQLite database.

Why we need SQLite (not just CSV) for Day 10:
The Root Cause Analyzer agent needs a `query_metrics` tool that runs
real SQL. Pandas-on-CSV works but feels janky — production agents
hit a real database. SQLite is the lightest "real database" option:
zero setup, single file, full SQL support.

This script is idempotent — running it twice just overwrites the table.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# sys.path hack — only in scripts/, never in src/ proper.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_FILE, PROCESSED_DATA_DIR  # noqa: E402


DB_PATH = PROCESSED_DATA_DIR / "metrics.db"
TABLE_NAME = "metrics_daily"


def load_csv_to_sqlite() -> None:
    """Load metrics_daily.csv into a SQLite table with proper types."""

    # Ensure target directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Read CSV. parse_dates ensures the date column comes in as datetime,
    # which SQLite stores as ISO strings (queryable with date functions).
    print(f"📖 Reading CSV from {METRICS_FILE}")
    df = pd.read_csv(METRICS_FILE, parse_dates=["date"])
    print(f"   Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")

    # Connect & write
    print(f"\n💾 Writing to SQLite at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(
            TABLE_NAME,
            conn,
            if_exists="replace",  # idempotent: re-runs just overwrite
            index=False,
        )

        # Index on date — every agent query will filter by date.
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_date ON {TABLE_NAME}(date);")
        conn.commit()

        # Verify row count
        cur = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        count = cur.fetchone()[0]
        print(f"   ✅ Inserted {count} rows into '{TABLE_NAME}'")

        # Sample
        cur = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 2;")
        print(f"\n📋 Sample rows:")
        for row in cur.fetchall():
            print(f"   {row}")
    finally:
        conn.close()

    print(f"\n✨ Done. DB ready at: {DB_PATH}")


if __name__ == "__main__":
    load_csv_to_sqlite()