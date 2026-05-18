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

from src.agents.action_recommender import run_action_recommender
from src.agents.report_generator import run_report_generator
from src.agents.root_cause import run_root_cause_agent
from src.agents.schemas import ActionPlan, RootCauseAnalysis
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
    Run the Root Cause Analyzer ReAct agent and write structured output to state.

    Reads anomaly_context populated by the signal detector (or test harness).
    Writes root_cause (plain text, for downstream dummy nodes) and
    root_cause_analysis (serialized RootCauseAnalysis, for Day 11+ agents).
    """
    ctx = state.get("anomaly_context", {})
    rca = run_root_cause_agent(
        anomaly_date=ctx.get("date", ""),
        anomaly_metric=ctx.get("metric", "daily_revenue"),
        anomaly_severity=ctx.get("severity", "medium"),
    )
    return {
        "root_cause": rca.primary_cause,
        "root_cause_analysis": rca.model_dump(mode="json"),
        "messages": [
            f"[diagnose] confidence={rca.confidence:.2f}, "
            f"severity={rca.severity_assessment}, "
            f"steps={len(rca.investigation_trace)}"
        ],
    }


# ---------------------------------------------------------------------
# Node 3: Action Recommender (dummy)
# ---------------------------------------------------------------------

def recommend_node(state: AgentState) -> dict:
    """
    Run the Action Recommender agent and write a prioritized ActionPlan to state.

    Reads root_cause_analysis (serialized RootCauseAnalysis dict) from state,
    reconstructs the Pydantic model, then calls the recommender.
    """
    rca = RootCauseAnalysis(**state.get("root_cause_analysis", {}))
    plan = run_action_recommender(rca)
    return {
        "recommended_action": plan.actions[0].description,
        "action_plan": plan.model_dump(mode="json"),
        "messages": [
            f"[recommend] {len(plan.actions)} actions, "
            f"top: {plan.actions[0].title!r} (score={plan.actions[0].priority_score:.2f})"
        ],
    }


# ---------------------------------------------------------------------
# Node 4: Report Generator
# ---------------------------------------------------------------------

def report_node(state: AgentState) -> dict:
    """
    Generate a structured IncidentReport from all prior agent outputs.

    Reads anomaly_context, root_cause_analysis, and action_plan from state.
    report_markdown (assembled markdown) is included in the serialized dict.
    """
    ctx = state.get("anomaly_context", {})
    rca = RootCauseAnalysis(**state.get("root_cause_analysis", {}))
    plan = ActionPlan(**state.get("action_plan", {}))
    report = run_report_generator(ctx, rca, plan)
    return {
        "report": report.model_dump(mode="json"),
        "messages": [
            f"[report] generated, severity={report.metadata.severity}, "
            f"actions={report.metadata.action_count}"
        ],
    }