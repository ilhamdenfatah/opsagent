"""
Signal Detector Agent — the first agent in the OpsAgent pipeline.

Watches daily metrics and decides: is something wrong here?

Uses a hybrid approach:
1. Statistical layer: z-score + % change vs 7-day rolling average
2. LLM layer: given the numbers + historical context, is this a real anomaly?

The statistical layer is fast and objective. The LLM layer adds judgment —
it knows that a 3x revenue spike during Lebaran is normal, not an emergency.
"""

import numpy as np
import pandas as pd
from groq import Groq
from pydantic import BaseModel

from src.config import (
    GROQ_API_KEY,
    AGENT_MODEL_ROUTING,
    ZSCORE_THRESHOLD,
    DAY_OVER_DAY_CHANGE_THRESHOLD,
)
from src.rag.retriever import retrieve, format_context_for_prompt
from src.agents.output_validator import call_llm_with_retry


# --- Output Schemas ---

class LLMJudgment(BaseModel):
    """Raw LLM output schema — validated before use."""
    anomaly_detected: bool
    severity: str
    confidence: float
    description: str


class AnomalySignal(BaseModel):
    """Final structured output from the Signal Detector."""
    anomaly_detected: bool
    metric: str
    current_value: float
    baseline_value: float
    pct_change: float
    zscore: float
    severity: str
    confidence: float
    description: str
    historical_context: str


class MetricSnapshot(BaseModel):
    """A single day's worth of metrics — input to the Signal Detector."""
    date: str
    daily_revenue: float
    order_count: int
    avg_order_value: float
    customer_churn_rate: float
    support_ticket_count: int
    conversion_rate: float


# --- Statistical Layer ---

def compute_statistics(
    current_value: float,
    historical_values: list[float],
) -> tuple[float, float]:
    """
    Compute z-score and % change for a single metric value.

    Returns:
        (zscore, pct_change) as floats
    """
    arr = np.array(historical_values)
    mean = arr.mean()
    std = arr.std()

    zscore = (current_value - mean) / std if std > 0 else 0.0
    pct_change = (current_value - mean) / mean if mean != 0 else 0.0

    return round(float(zscore), 4), round(float(pct_change), 4)


def is_statistically_anomalous(zscore: float, pct_change: float) -> bool:
    """
    Flag as suspicious if EITHER z-score OR % change exceeds threshold.

    OR logic because each catches different blind spots:
    z-score good for stable metrics, % change good for volatile metrics.
    """
    return (
        abs(zscore) > ZSCORE_THRESHOLD or
        abs(pct_change) > DAY_OVER_DAY_CHANGE_THRESHOLD
    )


def determine_severity(zscore: float, pct_change: float) -> str:
    """Map statistical extremity to a severity level."""
    abs_pct = abs(pct_change)
    abs_z = abs(zscore)

    if abs_pct > 0.50 or abs_z > 4.0:
        return "critical"
    elif abs_pct > 0.30 or abs_z > 3.0:
        return "high"
    elif abs_pct > 0.15 or abs_z > 2.0:
        return "medium"
    else:
        return "low"


# --- LLM Layer ---

SIGNAL_DETECTOR_SYSTEM_PROMPT = """You are a business metrics analyst for an e-commerce company in Indonesia.

Your job is to review flagged metric anomalies and make a final judgment:
- Is this a real anomaly that needs investigation?
- Or is it explainable by normal business patterns (seasonal effects, known events)?

You will receive:
1. The metric name and its current vs baseline value
2. Statistical scores (z-score, % change)
3. Relevant historical context from past similar events

Respond ONLY with a valid JSON object. No explanation outside the JSON.

Required format:
{
  "anomaly_detected": true or false,
  "severity": "low" | "medium" | "high" | "critical",
  "confidence": 0.0 to 1.0,
  "description": "1-2 sentence explanation of what is happening and why it matters"
}

Guidelines:
- If historical context shows this pattern is normal (holiday spike etc), set anomaly_detected=false
- Use lower confidence when context is ambiguous
- Description should be specific and actionable, not generic
"""


def llm_judgment(
    metric: str,
    current_value: float,
    baseline_value: float,
    pct_change: float,
    zscore: float,
    historical_context: str,
    date: str,
) -> LLMJudgment:
    """
    Ask the LLM to make the final call on whether this is a real anomaly.

    Now uses call_llm_with_retry for robust error handling and validation.
    Returns a validated LLMJudgment Pydantic model — never raw JSON.
    """
    client = Groq(api_key=GROQ_API_KEY)
    model = AGENT_MODEL_ROUTING["signal_detector"]

    user_message = f"""
Date: {date}
Metric: {metric}
Current value: {current_value:,.4f}
7-day baseline: {baseline_value:,.4f}
% change from baseline: {pct_change * 100:.1f}%
Z-score: {zscore:.2f}

{historical_context}

Based on the statistics and historical context above, is this a real anomaly?
"""

    return call_llm_with_retry(
        client=client,
        model=model,
        system_prompt=SIGNAL_DETECTOR_SYSTEM_PROMPT,
        user_message=user_message,
        output_schema=LLMJudgment,
    )


# --- Main Agent Function ---

def detect_anomalies(
    snapshot: MetricSnapshot,
    historical_df: pd.DataFrame,
    lookback_days: int = 7,
) -> list[AnomalySignal]:
    """
    Run the full Signal Detector pipeline on a single day's metrics.

    For each metric:
    1. Compute z-score + % change vs last N days
    2. If statistically suspicious → pull RAG context → ask LLM for judgment
    3. Return confirmed anomaly signals

    Args:
        snapshot: today's metric values
        historical_df: DataFrame with past metrics (index=date, columns=metrics)
        lookback_days: how many days to use as the rolling baseline

    Returns:
        List of AnomalySignal — one per confirmed anomaly (may be empty)
    """
    metrics_to_check = [
        "daily_revenue",
        "order_count",
        "customer_churn_rate",
        "support_ticket_count",
        "conversion_rate",
    ]

    signals = []

    for metric in metrics_to_check:
        current_value = getattr(snapshot, metric)
        recent_history = historical_df[metric].tail(lookback_days).tolist()

        if len(recent_history) < 3:
            continue

        baseline_value = float(np.mean(recent_history))
        zscore, pct_change = compute_statistics(current_value, recent_history)

        # Layer 1: statistical gate
        if not is_statistically_anomalous(zscore, pct_change):
            continue

        severity = determine_severity(zscore, pct_change)

        # Layer 2: RAG context
        rag_query = (
            f"{metric.replace('_', ' ')} "
            f"{'increased' if pct_change > 0 else 'decreased'} "
            f"{abs(pct_change)*100:.0f}%"
        )
        rag_results = retrieve(rag_query, top_k=3)
        historical_context = format_context_for_prompt(rag_results)

        # Layer 2: LLM judgment — now with retry + validation
        try:
            judgment = llm_judgment(
                metric=metric,
                current_value=current_value,
                baseline_value=baseline_value,
                pct_change=pct_change,
                zscore=zscore,
                historical_context=historical_context,
                date=snapshot.date,
            )
        except RuntimeError as e:
            # All retries exhausted — log and skip this metric
            print(f"  Warning: LLM judgment failed for {metric}: {e}")
            continue

        if judgment.anomaly_detected:
            signals.append(AnomalySignal(
                anomaly_detected=True,
                metric=metric,
                current_value=current_value,
                baseline_value=baseline_value,
                pct_change=pct_change,
                zscore=zscore,
                severity=judgment.severity,
                confidence=judgment.confidence,
                description=judgment.description,
                historical_context=historical_context,
            ))

    return signals
