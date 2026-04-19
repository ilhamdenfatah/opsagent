"""
Synthetic business metrics generator.

Builds realistic time-series data using a 4-layer approach:
    Final Value = Baseline + Trend + Seasonality + Noise + [Anomaly]

Each layer adds a different kind of "realness" to the data. Without all four,
the data looks obviously fake — too smooth, too regular, or too random.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnomalySpec:
    """Defines a single anomaly to inject into a metric series."""
    anomaly_id: str
    date_start: str
    date_end: str
    metric: str
    anomaly_type: str          # sudden_drop | gradual_degradation | spike | correlated
    magnitude: float           # multiplier, e.g. 0.65 means "drop to 65% of normal"
    root_cause: str
    expected_actions: list[str] = field(default_factory=list)
    severity: str = "medium"   # low | medium | high | critical


def generate_baseline(
    dates: pd.DatetimeIndex,
    baseline_value: float,
    trend_pct_per_day: float = 0.0,
) -> np.ndarray:
    """
    Generate a flat baseline with optional linear trend.

    Args:
        dates: DatetimeIndex for the full period
        baseline_value: starting value on day 0
        trend_pct_per_day: daily growth rate, e.g. 0.003 = 0.3% per day

    Returns:
        Array of baseline values, one per date
    """
    n = len(dates)
    trend_multipliers = np.array([(1 + trend_pct_per_day) ** i for i in range(n)])
    return baseline_value * trend_multipliers


def add_weekly_seasonality(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    weekend_multiplier: float = 1.20,
) -> np.ndarray:
    """
    Add weekly seasonality — weekends behave differently from weekdays.

    For most business metrics, weekends either spike (consumer) or dip (B2B).
    weekend_multiplier > 1.0 means weekends are higher (e.g. e-commerce).

    Args:
        series: baseline series to modify
        dates: corresponding DatetimeIndex
        weekend_multiplier: how much to scale weekends (Sat=5, Sun=6)
    """
    result = series.copy()
    for i, date in enumerate(dates):
        if date.dayofweek >= 5:  # Saturday or Sunday
            result[i] *= weekend_multiplier
    return result


def add_monthly_seasonality(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    payday_boost_days: list[int] = None,
    payday_boost_multiplier: float = 1.15,
) -> np.ndarray:
    """
    Add monthly seasonality — payday effect and end-of-month patterns.

    In Indonesia, common paydays are around the 25th and 1st of each month.
    Spending typically spikes right after payday.

    Args:
        series: series to modify
        dates: corresponding DatetimeIndex
        payday_boost_days: days of month that get a boost (default: [1, 2, 25, 26])
        payday_boost_multiplier: how much to boost those days
    """
    if payday_boost_days is None:
        payday_boost_days = [1, 2, 25, 26]

    result = series.copy()
    for i, date in enumerate(dates):
        if date.day in payday_boost_days:
            result[i] *= payday_boost_multiplier
    return result


def add_noise(
    series: np.ndarray,
    rng: np.random.Generator,
    noise_level: float = 0.05,
) -> np.ndarray:
    """
    Add realistic random variation (Gaussian noise).

    Without noise, the data looks like a textbook graph — too perfect.
    Real business metrics always have day-to-day randomness.

    Args:
        series: series to add noise to
        rng: seeded random generator (use the project-wide seed for reproducibility)
        noise_level: std dev as fraction of value, e.g. 0.05 = ±5% daily variation
    """
    noise = rng.normal(loc=1.0, scale=noise_level, size=len(series))
    return series * noise


def inject_anomaly(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    spec: AnomalySpec,
) -> np.ndarray:
    """
    Inject a single anomaly into a series based on its spec.

    Handles three anomaly shapes:
    - sudden_drop / spike: instant change on start date, back to normal after end date
    - gradual_degradation: linear ramp from normal to magnitude over the date range
    - correlated: same as sudden_drop (correlation is handled at the dataset level)

    Args:
        series: the metric series to modify
        dates: DatetimeIndex corresponding to series
        spec: AnomalySpec defining what to inject and where

    Returns:
        Modified series with anomaly injected
    """
    result = series.copy()
    start = pd.Timestamp(spec.date_start)
    end = pd.Timestamp(spec.date_end)

    affected_mask = (dates >= start) & (dates <= end)
    affected_indices = np.where(affected_mask)[0]

    if len(affected_indices) == 0:
        return result

    if spec.anomaly_type in ("sudden_drop", "spike", "correlated"):
        # Instant change — all affected days get the same multiplier
        result[affected_mask] *= spec.magnitude

    elif spec.anomaly_type == "gradual_degradation":
        # Linear ramp: day 1 is still normal, last day hits full magnitude
        n = len(affected_indices)
        for step, idx in enumerate(affected_indices):
            # Linear interpolation from 1.0 to spec.magnitude
            progress = step / max(n - 1, 1)
            multiplier = 1.0 + progress * (spec.magnitude - 1.0)
            result[idx] *= multiplier

    return result
