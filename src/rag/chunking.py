"""
Chunking strategies for historical metrics data.

We use two complementary strategies:
1. Weekly summaries — broad overview of each week's performance
2. Anomaly events — focused chunks around each planted anomaly

Why two strategies? Different queries need different granularity.
"How was Q4 performance?" → weekly chunks answer this well.
"What caused the November revenue drop?" → anomaly chunks answer this well.
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from src.config import METRICS_FILE, GROUND_TRUTH_FILE


@dataclass
class Chunk:
    """A single chunk ready to be embedded and stored in Qdrant."""
    text: str               # the text we'll embed
    chunk_type: str         # "weekly_summary" | "anomaly_event"
    metadata: dict = field(default_factory=dict)


def load_data() -> tuple[pd.DataFrame, list[dict]]:
    """Load the metrics dataset and ground truth anomalies."""
    df = pd.read_csv(METRICS_FILE, index_col="date", parse_dates=True)

    with open(GROUND_TRUTH_FILE) as f:
        ground_truth = json.load(f)

    return df, ground_truth


def build_weekly_summary_chunks(df: pd.DataFrame) -> list[Chunk]:
    """
    Create one chunk per week summarizing that week's metrics.

    Each chunk includes:
    - Date range
    - Average and trend for each metric
    - Whether any metric was notably high or low

    This gives the RAG system a "bird's eye view" of each week,
    useful for broad queries about periods of time.
    """
    chunks = []

    # Resample to weekly groups (week starting Monday)
    weekly = df.resample("W-MON")

    for week_start, week_data in weekly:
        if week_data.empty:
            continue

        week_end = week_data.index[-1]
        n_days = len(week_data)

        # Calculate weekly stats for each metric
        stats = {}
        for col in df.columns:
            stats[col] = {
                "mean": week_data[col].mean(),
                "min": week_data[col].min(),
                "max": week_data[col].max(),
                "pct_change": (
                    (week_data[col].iloc[-1] - week_data[col].iloc[0])
                    / week_data[col].iloc[0] * 100
                    if week_data[col].iloc[0] != 0 else 0
                ),
            }

        # Build natural language summary
        revenue_trend = "increased" if stats["daily_revenue"]["pct_change"] > 2 else \
                        "decreased" if stats["daily_revenue"]["pct_change"] < -2 else "was stable"

        churn_note = ""
        if stats["customer_churn_rate"]["mean"] > 3.5:
            churn_note = " Churn rate was elevated this week."
        elif stats["customer_churn_rate"]["mean"] < 1.8:
            churn_note = " Churn rate was notably low this week."

        support_note = ""
        if stats["support_ticket_count"]["mean"] > 70:
            support_note = " Support ticket volume was high."

        text = (
            f"Week of {week_start.strftime('%B %d, %Y')} "
            f"({week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}, "
            f"{n_days} days): "
            f"Daily revenue {revenue_trend}, averaging {stats['daily_revenue']['mean']:,.0f} IDR "
            f"(range: {stats['daily_revenue']['min']:,.0f} to {stats['daily_revenue']['max']:,.0f}). "
            f"Average order count was {stats['order_count']['mean']:.0f} per day. "
            f"Average order value was {stats['avg_order_value']['mean']:,.0f} IDR. "
            f"Conversion rate averaged {stats['conversion_rate']['mean']:.2f}%. "
            f"Customer churn rate averaged {stats['customer_churn_rate']['mean']:.2f}%. "
            f"Support tickets averaged {stats['support_ticket_count']['mean']:.0f} per day."
            f"{churn_note}{support_note}"
        )

        chunks.append(Chunk(
            text=text,
            chunk_type="weekly_summary",
            metadata={
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_end": week_end.strftime("%Y-%m-%d"),
                "avg_revenue": round(stats["daily_revenue"]["mean"], 0),
                "avg_orders": round(stats["order_count"]["mean"], 0),
                "avg_churn": round(stats["customer_churn_rate"]["mean"], 4),
                "revenue_pct_change": round(stats["daily_revenue"]["pct_change"], 2),
            },
        ))

    return chunks


def build_anomaly_event_chunks(
    df: pd.DataFrame,
    ground_truth: list[dict],
    context_days: int = 3,
) -> list[Chunk]:
    """
    Create one chunk per anomaly event, including context days before and after.

    The context window (default 3 days before + 3 days after) is important:
    it gives the RAG enough surrounding data to understand what "normal" looked
    like before the anomaly hit, and how fast recovery happened.

    Args:
        df: metrics DataFrame
        ground_truth: list of anomaly dicts from ground_truth.json
        context_days: how many days before/after to include
    """
    chunks = []

    for anomaly in ground_truth:
        anomaly_start = pd.Timestamp(anomaly["date_start"])
        anomaly_end = pd.Timestamp(anomaly["date_end"])
        metric = anomaly["metric"]

        # Get the window: context_days before → anomaly end + context_days after
        window_start = anomaly_start - pd.Timedelta(days=context_days)
        window_end = anomaly_end + pd.Timedelta(days=context_days)

        # Clip to dataset bounds
        window_start = max(window_start, df.index[0])
        window_end = min(window_end, df.index[-1])

        window_data = df.loc[window_start:window_end]

        if window_data.empty or metric not in df.columns:
            continue

        # Calculate magnitude vs pre-anomaly baseline
        pre_anomaly = df.loc[window_start:anomaly_start - pd.Timedelta(days=1)]
        if not pre_anomaly.empty:
            baseline_value = pre_anomaly[metric].mean()
            anomaly_value = df.loc[anomaly_start:anomaly_end, metric].mean()
            magnitude_pct = ((anomaly_value - baseline_value) / baseline_value * 100
                             if baseline_value != 0 else 0)
        else:
            magnitude_pct = 0

        direction = "increased" if magnitude_pct > 0 else "decreased"
        abs_pct = abs(magnitude_pct)

        # Build per-day breakdown for the anomaly period
        anomaly_period = df.loc[anomaly_start:anomaly_end, metric]
        day_values = ", ".join([
            f"{date.strftime('%b %d')}: {val:,.0f}" if val > 1000
            else f"{date.strftime('%b %d')}: {val:.3f}"
            for date, val in anomaly_period.items()
        ])

        text = (
            f"Anomaly event {anomaly['anomaly_id']} on {metric}: "
            f"{anomaly['type'].replace('_', ' ')} detected from "
            f"{anomaly['date_start']} to {anomaly['date_end']}. "
            f"Severity: {anomaly['severity']}. "
            f"The metric {direction} by approximately {abs_pct:.1f}% compared to the prior baseline. "
            f"Daily values during anomaly: {day_values}. "
            f"Root cause: {anomaly['root_cause']} "
            f"Recommended actions: {'; '.join(anomaly['expected_actions'][:3])}."
        )

        chunks.append(Chunk(
            text=text,
            chunk_type="anomaly_event",
            metadata={
                "anomaly_id": anomaly["anomaly_id"],
                "date_start": anomaly["date_start"],
                "date_end": anomaly["date_end"],
                "metric": metric,
                "anomaly_type": anomaly["type"],
                "severity": anomaly["severity"],
                "magnitude_pct": round(magnitude_pct, 2),
            },
        ))

    return chunks


def build_all_chunks(df: pd.DataFrame, ground_truth: list[dict]) -> list[Chunk]:
    """Build all chunks — weekly summaries + anomaly events combined."""
    weekly_chunks = build_weekly_summary_chunks(df)
    anomaly_chunks = build_anomaly_event_chunks(df, ground_truth)

    print(f"Built {len(weekly_chunks)} weekly summary chunks")
    print(f"Built {len(anomaly_chunks)} anomaly event chunks")
    print(f"Total: {len(weekly_chunks) + len(anomaly_chunks)} chunks")

    return weekly_chunks + anomaly_chunks
