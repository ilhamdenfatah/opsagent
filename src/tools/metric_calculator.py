"""
Statistical calculation tool for agents.

Agents call this when they need derived metrics — z-scores, day-over-day
changes, correlations — rather than raw rows. Sits on top of data_query.py
so all DB access stays in one place.

No external stats libraries: stdlib math + statistics module only.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from src.tools.data_query import (
    _ALLOWED_METRICS,
    get_single_day,
    query_metrics,
)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class ZScoreResult(BaseModel):
    metric: str
    target_date: date
    target_value: float
    window_mean: float
    window_std: float
    zscore: float
    window_days: int
    window_start: date
    window_end: date


class DayOverDayResult(BaseModel):
    metric: str
    target_date: date
    target_value: float
    previous_value: float
    absolute_change: float
    percent_change: Optional[float] = Field(
        description="None when previous_value is zero (undefined, not inf)"
    )


class CorrelationResult(BaseModel):
    metric_a: str
    metric_b: str
    start_date: date
    end_date: date
    pearson_r: float
    row_count: int
    interpretation: str = Field(
        description="e.g. 'strong positive', 'weak negative', 'no correlation'"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_metric(metric: str) -> None:
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Allowed: {_ALLOWED_METRICS}")


def _extract_values(result, metric: str) -> list[float]:
    return [float(getattr(row, metric)) for row in result.rows]


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """
    Pearson correlation coefficient.

    r = Σ((x-x̄)(y-ȳ)) / sqrt(Σ(x-x̄)² · Σ(y-ȳ)²)

    Returns 0.0 when either series is constant — the correlation is
    undefined in that case, but 0.0 is the least surprising fallback
    for downstream agents.
    """
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denom = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )

    if denom == 0.0:
        return 0.0
    return numerator / denom


def _interpret_correlation(r: float) -> str:
    abs_r = abs(r)
    direction = "positive" if r >= 0 else "negative"
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.1:
        strength = "weak"
    else:
        return "no correlation"
    return f"{strength} {direction}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def calculate_zscore(
    metric: str,
    target_date: str | date,
    window_days: int = 30,
) -> ZScoreResult:
    """
    Z-score of target_date's value against the preceding window.

    Window is strictly BEFORE target_date so the anomaly itself doesn't
    contaminate the baseline mean/std.

    Args:
        metric: column name in metrics_daily
        target_date: the date to score
        window_days: how many prior days to use as the baseline

    Raises:
        ValueError: if window has fewer than 2 data points
    """
    _validate_metric(metric)

    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    window_end = target_date - timedelta(days=1)
    window_start = target_date - timedelta(days=window_days)

    window_result = query_metrics(window_start, window_end)
    if window_result.row_count < 2:
        raise ValueError(
            f"Need at least 2 data points for z-score baseline, "
            f"got {window_result.row_count} for window {window_start} to {window_end}"
        )

    target_row = get_single_day(target_date)
    if target_row is None:
        raise ValueError(f"No data for target_date={target_date}")

    window_values = _extract_values(window_result, metric)
    target_value = float(getattr(target_row, metric))

    w_mean = statistics.mean(window_values)
    w_std = statistics.stdev(window_values)

    zscore = (target_value - w_mean) / w_std if w_std > 0 else 0.0

    return ZScoreResult(
        metric=metric,
        target_date=target_date,
        target_value=target_value,
        window_mean=w_mean,
        window_std=w_std,
        zscore=zscore,
        window_days=window_days,
        window_start=window_start,
        window_end=window_end,
    )


def calculate_day_over_day_change(
    metric: str,
    target_date: str | date,
) -> DayOverDayResult:
    """
    Percent and absolute change vs the previous day.

    percent_change is None (not inf) when previous_value is zero —
    downstream agents should treat None as "undefined, not comparable."

    Raises:
        ValueError: if either target_date or previous day has no data
    """
    _validate_metric(metric)

    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    previous_date = target_date - timedelta(days=1)

    today_row = get_single_day(target_date)
    if today_row is None:
        raise ValueError(f"No data for target_date={target_date}")

    yesterday_row = get_single_day(previous_date)
    if yesterday_row is None:
        raise ValueError(f"No data for previous_date={previous_date}")

    today_val = float(getattr(today_row, metric))
    yesterday_val = float(getattr(yesterday_row, metric))

    absolute_change = today_val - yesterday_val
    percent_change = (
        (absolute_change / yesterday_val) * 100
        if yesterday_val != 0
        else None
    )

    return DayOverDayResult(
        metric=metric,
        target_date=target_date,
        target_value=today_val,
        previous_value=yesterday_val,
        absolute_change=absolute_change,
        percent_change=percent_change,
    )


def calculate_correlation(
    metric_a: str,
    metric_b: str,
    start_date: str | date,
    end_date: str | date,
) -> CorrelationResult:
    """
    Pearson correlation between two metrics over a date range.

    Useful for the root cause agent to ask: "when revenue dropped,
    did order_count also drop?" — a strong positive correlation supports
    a demand-side explanation over a pricing/AOV explanation.

    Raises:
        ValueError: if fewer than 2 rows in the date range
    """
    _validate_metric(metric_a)
    _validate_metric(metric_b)

    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    result = query_metrics(start_date, end_date)
    if result.row_count < 2:
        raise ValueError(
            f"Need at least 2 rows for correlation, "
            f"got {result.row_count} for {start_date} to {end_date}"
        )

    xs = _extract_values(result, metric_a)
    ys = _extract_values(result, metric_b)

    r = _pearson_r(xs, ys)

    return CorrelationResult(
        metric_a=metric_a,
        metric_b=metric_b,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        pearson_r=round(r, 6),
        row_count=result.row_count,
        interpretation=_interpret_correlation(r),
    )
