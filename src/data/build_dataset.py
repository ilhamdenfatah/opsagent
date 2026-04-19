"""
Dataset builder — generates the full synthetic metrics dataset.

Run this script to (re)generate the dataset:
    python -m src.data.build_dataset

Outputs:
    data/raw/metrics_daily.csv          — main dataset with anomalies injected
    data/raw/metrics_daily_clean.csv    — same data without anomalies (for debugging)
    data/raw/anomaly_ground_truth.json  — ground truth for all 15 anomalies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    DATASET_START_DATE,
    DATASET_END_DATE,
    METRIC_BASELINES,
    RANDOM_SEED,
    RAW_DATA_DIR,
    METRICS_FILE,
    METRICS_CLEAN_FILE,
    GROUND_TRUTH_FILE,
)
from src.data.generator import (
    generate_baseline,
    add_weekly_seasonality,
    add_monthly_seasonality,
    add_noise,
    inject_anomaly,
)
from src.data.anomaly_catalog import ANOMALY_CATALOG


def build_metric_series(
    dates: pd.DatetimeIndex,
    metric_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a single metric's time series from scratch using the 4-layer approach.

    Each metric has its own personality — different trend rates, seasonality
    patterns, and noise levels to reflect real business behavior.
    """

    # --- Layer 1 + 2: Baseline + Trend ---
    # Each metric grows (or decays) at a different rate
    trend_rates = {
        "daily_revenue":         0.003,   # 0.3%/day growth — healthy business
        "order_count":           0.002,   # slightly slower than revenue (AOV growing)
        "customer_churn_rate":  -0.001,   # slowly improving retention
        "support_ticket_count":  0.001,   # growing with order volume
        "conversion_rate":       0.0005,  # slowly optimizing
    }
    series = generate_baseline(
        dates,
        baseline_value=METRIC_BASELINES[metric_name],
        trend_pct_per_day=trend_rates[metric_name],
    )

    # --- Layer 3: Seasonality ---
    # Different metrics respond differently to weekends and paydays
    weekend_multipliers = {
        "daily_revenue":         1.25,   # consumer spending spikes on weekends
        "order_count":           1.20,
        "customer_churn_rate":   0.90,   # people don't cancel on weekends
        "support_ticket_count":  0.70,   # support team is smaller on weekends
        "conversion_rate":       1.10,
    }
    series = add_weekly_seasonality(dates=dates, series=series,
                                    weekend_multiplier=weekend_multipliers[metric_name])
    series = add_monthly_seasonality(dates=dates, series=series)

    # --- Layer 4: Noise ---
    # Churn and conversion are noisier (harder to predict), revenue less so
    noise_levels = {
        "daily_revenue":         0.04,
        "order_count":           0.05,
        "customer_churn_rate":   0.08,
        "support_ticket_count":  0.10,
        "conversion_rate":       0.07,
    }
    series = add_noise(series=series, rng=rng, noise_level=noise_levels[metric_name])

    return series


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full dataset — clean version and anomaly-injected version.

    Returns:
        (df_with_anomalies, df_clean) — both as DataFrames indexed by date
    """
    rng = np.random.default_rng(seed=RANDOM_SEED)
    dates = pd.date_range(start=DATASET_START_DATE, end=DATASET_END_DATE, freq="D")

    print(f"Generating {len(dates)} days of data ({DATASET_START_DATE} → {DATASET_END_DATE})")

    # Build the 5 base metrics
    base_metrics = {}
    for metric in METRIC_BASELINES:
        print(f"  → Building {metric}...")
        base_metrics[metric] = build_metric_series(dates, metric, rng)

    # Build clean DataFrame (no anomalies) — useful for debugging and comparison
    df_clean = pd.DataFrame(base_metrics, index=dates)
    df_clean.index.name = "date"

    # Derive avg_order_value from revenue / orders (before anomaly injection)
    df_clean["avg_order_value"] = df_clean["daily_revenue"] / df_clean["order_count"]

    # Inject anomalies into a copy
    df_anomalies = df_clean.copy()

    print(f"\nInjecting {len(ANOMALY_CATALOG)} anomalies...")
    for spec in ANOMALY_CATALOG:
        if spec.metric in df_anomalies.columns:
            print(f"  → {spec.anomaly_id}: {spec.anomaly_type} on {spec.metric} "
                  f"({spec.date_start} → {spec.date_end})")
            df_anomalies[spec.metric] = inject_anomaly(
                series=df_anomalies[spec.metric].values,
                dates=dates,
                spec=spec,
            )

    # Recalculate avg_order_value after anomaly injection
    # (revenue and order_count anomalies affect AOV)
    df_anomalies["avg_order_value"] = (
        df_anomalies["daily_revenue"] / df_anomalies["order_count"]
    )

    # Round to sensible precision
    df_anomalies["daily_revenue"] = df_anomalies["daily_revenue"].round(0)
    df_anomalies["order_count"] = df_anomalies["order_count"].round(0).astype(int)
    df_anomalies["avg_order_value"] = df_anomalies["avg_order_value"].round(0)
    df_anomalies["customer_churn_rate"] = df_anomalies["customer_churn_rate"].round(4)
    df_anomalies["support_ticket_count"] = df_anomalies["support_ticket_count"].round(0).astype(int)
    df_anomalies["conversion_rate"] = df_anomalies["conversion_rate"].round(4)

    df_clean["daily_revenue"] = df_clean["daily_revenue"].round(0)
    df_clean["order_count"] = df_clean["order_count"].round(0).astype(int)
    df_clean["avg_order_value"] = df_clean["avg_order_value"].round(0)
    df_clean["customer_churn_rate"] = df_clean["customer_churn_rate"].round(4)
    df_clean["support_ticket_count"] = df_clean["support_ticket_count"].round(0).astype(int)
    df_clean["conversion_rate"] = df_clean["conversion_rate"].round(4)

    return df_anomalies, df_clean


def build_ground_truth_json() -> list[dict]:
    """
    Convert the anomaly catalog into a clean JSON structure for evaluation.

    This JSON is what Phase 3 evaluation compares AI output against.
    """
    ground_truth = []
    for spec in ANOMALY_CATALOG:
        ground_truth.append({
            "anomaly_id": spec.anomaly_id,
            "date_start": spec.date_start,
            "date_end": spec.date_end,
            "metric": spec.metric,
            "type": spec.anomaly_type,
            "severity": spec.severity,
            "magnitude": spec.magnitude,
            "root_cause": spec.root_cause,
            "expected_actions": spec.expected_actions,
        })
    return ground_truth


def main() -> None:
    """Run the full dataset generation pipeline."""
    # Ensure output directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate datasets
    df_anomalies, df_clean = build_dataset()

    # Save datasets
    df_anomalies.to_csv(METRICS_FILE)
    df_clean.to_csv(METRICS_CLEAN_FILE)
    print(f"\n✓ Dataset saved → {METRICS_FILE}")
    print(f"✓ Clean dataset saved → {METRICS_CLEAN_FILE}")

    # Save ground truth
    ground_truth = build_ground_truth_json()
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"✓ Ground truth saved → {GROUND_TRUTH_FILE}")

    # Quick sanity check
    print(f"\nDataset shape: {df_anomalies.shape}")
    print(f"Date range: {df_anomalies.index[0].date()} → {df_anomalies.index[-1].date()}")
    print(f"Columns: {list(df_anomalies.columns)}")
    print(f"Anomalies planted: {len(ground_truth)}")
    print("\nSample (first 3 rows):")
    print(df_anomalies.head(3).to_string())


if __name__ == "__main__":
    main()
