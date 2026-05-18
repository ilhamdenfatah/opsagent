"""
SQL query tool for agents.

Agents use this when they need actual numbers from the database —
"what was revenue on Day X?" or "show me the week before the anomaly."

Read-only by design: no INSERT/UPDATE/DELETE. Every query is
parameterized so there's no injection risk from agent-generated inputs.

The return types (MetricRow, QueryResult) are Pydantic models so
downstream agents can consume them without dict gymnastics.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from src.config import METRICS_DB_FILE


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class MetricRow(BaseModel):
    """One day of metrics from the database."""

    date: date
    daily_revenue: float
    order_count: int
    customer_churn_rate: float
    support_ticket_count: int
    conversion_rate: float
    avg_order_value: float


class QueryResult(BaseModel):
    """Wrapper around a list of rows — agents get metadata for free."""

    rows: list[MetricRow]
    row_count: int = Field(description="Number of rows returned")
    query_description: str = Field(description="Human-readable description of what was queried")


class MetricStats(BaseModel):
    """Descriptive stats for a single metric over a date range."""

    metric: str
    start_date: date
    end_date: date
    min: float
    max: float
    mean: float
    std: float
    row_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ALLOWED_METRICS = frozenset({
    "daily_revenue",
    "order_count",
    "customer_churn_rate",
    "support_ticket_count",
    "conversion_rate",
    "avg_order_value",
})


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(METRICS_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_metric(row: sqlite3.Row) -> MetricRow:
    raw_date = row["date"]
    if isinstance(raw_date, str):
        parsed = datetime.fromisoformat(raw_date).date()
    elif isinstance(raw_date, datetime):
        parsed = raw_date.date()
    else:
        parsed = raw_date
    return MetricRow(
        date=parsed,
        daily_revenue=row["daily_revenue"],
        order_count=row["order_count"],
        customer_churn_rate=row["customer_churn_rate"],
        support_ticket_count=row["support_ticket_count"],
        conversion_rate=row["conversion_rate"],
        avg_order_value=row["avg_order_value"],
    )


def _validate_metrics(metrics: list[str]) -> None:
    unknown = set(metrics) - _ALLOWED_METRICS
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Allowed: {_ALLOWED_METRICS}")


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def query_metrics(
    start_date: str | date,
    end_date: str | date,
) -> QueryResult:
    """
    Fetch all metric rows for a date range (inclusive on both ends).

    Args:
        start_date: ISO date string or date object
        end_date: ISO date string or date object

    Returns:
        QueryResult with all rows in range, sorted ascending by date
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    sql = """
        SELECT date, daily_revenue, order_count, customer_churn_rate,
               support_ticket_count, conversion_rate, avg_order_value
        FROM metrics_daily
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
    """
    with _get_connection() as conn:
        rows = conn.execute(sql, (start_date, end_date)).fetchall()

    metric_rows = [_row_to_metric(r) for r in rows]
    return QueryResult(
        rows=metric_rows,
        row_count=len(metric_rows),
        query_description=f"metrics_daily from {start_date} to {end_date}",
    )


def get_metric_stats(
    metric: str,
    start_date: str | date,
    end_date: str | date,
) -> MetricStats:
    """
    Descriptive stats for a single metric over a date range.

    Useful when the agent needs a baseline: "was revenue on Day X
    above or below the historical mean for this period?"

    Args:
        metric: column name in metrics_daily
        start_date: ISO date string or date object
        end_date: ISO date string or date object
    """
    _validate_metrics([metric])

    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    # Column name comes from our own allowlist, not user input — safe to interpolate
    sql = f"""
        SELECT
            MIN({metric})   AS min_val,
            MAX({metric})   AS max_val,
            AVG({metric})   AS mean_val,
            COUNT(*)        AS row_count,
            -- SQLite has no STDDEV; compute manually via variance formula
            SQRT(AVG({metric} * {metric}) - AVG({metric}) * AVG({metric})) AS std_val
        FROM metrics_daily
        WHERE date BETWEEN ? AND ?
    """
    with _get_connection() as conn:
        row = conn.execute(sql, (start_date, end_date)).fetchone()

    return MetricStats(
        metric=metric,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        min=row["min_val"],
        max=row["max_val"],
        mean=row["mean_val"],
        std=row["std_val"] or 0.0,
        row_count=row["row_count"],
    )


def get_rows_around_date(
    target_date: str | date,
    window_days: int = 7,
) -> QueryResult:
    """
    Fetch rows within ±window_days of target_date.

    This is the main tool for anomaly investigation context —
    "show me the week before and after the revenue drop."

    Args:
        target_date: the anomaly date
        window_days: how many days before and after to include
    """
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    start = (target_date - timedelta(days=window_days)).isoformat()
    end = (target_date + timedelta(days=window_days)).isoformat()
    target_str = target_date.isoformat()

    result = query_metrics(start, end)
    return QueryResult(
        rows=result.rows,
        row_count=result.row_count,
        query_description=(
            f"+/-{window_days} days around {target_str} "
            f"({start} to {end})"
        ),
    )


def get_single_day(target_date: str | date) -> MetricRow | None:
    """
    Fetch exactly one day's row. Returns None if date not in database.

    Handy for spot checks: "what were the exact numbers on the anomaly date?"
    """
    if isinstance(target_date, date):
        target_date = target_date.isoformat()

    # Dates in DB are stored as 'YYYY-MM-DD HH:MM:SS', so exact string match
    # fails without DATE() — BETWEEN works fine because of lexicographic ordering.
    sql = """
        SELECT date, daily_revenue, order_count, customer_churn_rate,
               support_ticket_count, conversion_rate, avg_order_value
        FROM metrics_daily
        WHERE DATE(date) = ?
    """
    with _get_connection() as conn:
        row = conn.execute(sql, (target_date,)).fetchone()

    return _row_to_metric(row) if row else None
