"""
Dummy nodes for the OpsAgent pipeline skeleton (Day 9).

These nodes mimic the shape of real agents — they read state, do work,
and return partial state updates — but the "work" is hardcoded logic.
This lets us validate the LangGraph wiring before we plug in real LLMs
in Day 10 onwards.

Each node will eventually be replaced by:
- signal_detector_node  -> Signal Detector agent (Day 5–7, already done in src/agents/)
- diagnose_node         -> Root Cause Analyzer agent (Day 10)
- recommend_node        -> Action Recommender agent (Day 11)
"""

from __future__ import annotations

from src.pipeline.state import AgentState


# ---------------------------------------------------------------------
# Constants — keep in one place so they're easy to tweak during testing.
# In real agents, these come from config.py + LLM judgment.
# ---------------------------------------------------------------------

REVENUE_ANOMALY_THRESHOLD = 100
"""If revenue drops below this number, flag as anomaly. Dummy heuristic."""


# ---------------------------------------------------------------------
# Node 1: Signal Detector (dummy)
# ---------------------------------------------------------------------

def signal_detector_node(state: AgentState) -> dict:
    """
    Decide whether the incoming metrics contain an anomaly.

    Real version (already exists in src/agents/signal_detector.py) uses
    statistical checks + LLM judgment. This dummy version just checks
    if `revenue` is below a threshold.

    Returns a partial state update — only the keys this node owns.
    """
    metrics = state.get("metrics", {})
    revenue = metrics.get("revenue", 0)

    is_anomaly = revenue < REVENUE_ANOMALY_THRESHOLD

    log_msg = (
        f"[signal_detector] revenue={revenue}, "
        f"anomaly_detected={is_anomaly}"
    )

    return {
        "anomaly_detected": is_anomaly,
        "messages": [log_msg],
    }


# ---------------------------------------------------------------------
# Node 2: Root Cause Analyzer (dummy)
# ---------------------------------------------------------------------

def diagnose_node(state: AgentState) -> dict:
    """
    Diagnose the root cause of the detected anomaly.

    Real version (Day 10) will use tool-use: query the metrics database,
    correlate with historical events, and let the LLM reason. For now,
    this just returns a hardcoded narrative.
    """
    metrics = state.get("metrics", {})
    revenue = metrics.get("revenue", 0)

    # Hardcoded "diagnosis" — pretend the LLM figured this out.
    cause = (
        f"Revenue dropped to {revenue} (below threshold "
        f"{REVENUE_ANOMALY_THRESHOLD}). Likely cause: a marketing "
        f"campaign ended without a replacement, cutting traffic."
    )

    log_msg = f"[diagnose] root_cause identified"

    return {
        "root_cause": cause,
        "messages": [log_msg],
    }


# ---------------------------------------------------------------------
# Node 3: Action Recommender (dummy)
# ---------------------------------------------------------------------

def recommend_node(state: AgentState) -> dict:
    """
    Recommend an action based on the diagnosed root cause.

    Real version (Day 11) will use few-shot prompting + ranked output
    with impact-vs-effort scoring. For now, hardcoded suggestion.
    """
    cause = state.get("root_cause", "unknown cause")

    action = (
        "Launch a replacement marketing campaign within 48 hours. "
        "Re-engage churned segments via email. Monitor revenue daily."
    )

    log_msg = f"[recommend] action proposed for cause: {cause[:50]}..."

    return {
        "recommended_action": action,
        "messages": [log_msg],
    }