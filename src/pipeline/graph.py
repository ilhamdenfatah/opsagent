"""
LangGraph wiring for the OpsAgent pipeline (Day 12 — four real agents).

This module:
1. Builds the graph (state schema + nodes + edges).
2. Compiles it into an invokable object.
3. Provides a helper to export the graph as Mermaid for documentation.

Pipeline flow:
  START → signal_detector → (anomaly?) → diagnose → recommend → report → END
                                       ↘ END (no anomaly)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.pipeline.state import AgentState
from src.pipeline.nodes import (
    signal_detector_node,
    diagnose_node,
    recommend_node,
    report_node,
)


# ---------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------
# Conditional edges call these to decide where to go next based on state.
# Each function returns a STRING that maps to a destination in the
# conditional edge's routing dict.
# ---------------------------------------------------------------------

def route_after_signal(state: AgentState) -> str:
    """
    After Signal Detector runs, decide whether to continue or exit early.

    Returns:
        "diagnose" — anomaly detected, proceed to root cause analysis.
        "end"      — no anomaly, skip the rest of the pipeline.
    """
    if state.get("anomaly_detected", False):
        return "diagnose"
    return "end"


# ---------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------

def build_graph():
    """
    Construct and compile the OpsAgent graph.

    Why a builder function instead of module-level code?
    - Easier to test (each call returns a fresh graph).
    - Easier to parametrize later (e.g., swap nodes for testing).
    - Avoids side effects on import.
    """
    # 1. Initialize the graph with our state schema.
    #    LangGraph uses this schema to know what fields exist and how
    #    to merge updates from each node (using reducers, etc.).
    graph = StateGraph(AgentState)

    # 2. Register each node by name. The string name is how we reference
    #    nodes in edges and routing.
    graph.add_node("signal_detector", signal_detector_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("report", report_node)

    # 3. Wire the edges — the actual flow control.
    #
    #    START → signal_detector  (always)
    graph.add_edge(START, "signal_detector")

    #    signal_detector → diagnose OR END  (conditional)
    graph.add_conditional_edges(
        "signal_detector",
        route_after_signal,
        {
            "diagnose": "diagnose",   # routing returns "diagnose" -> go to diagnose node
            "end": END,               # routing returns "end" -> terminate
        },
    )

    #    diagnose → recommend  (always)
    graph.add_edge("diagnose", "recommend")

    #    recommend → report  (always)
    graph.add_edge("recommend", "report")

    #    report → END  (always)
    graph.add_edge("report", END)

    # 4. Compile. This validates the graph (no orphan nodes, valid edges,
    #    reachable END, etc.) and returns an invokable object.
    return graph.compile()


# ---------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------

def export_mermaid(output_path: str = "docs/architecture/pipeline_day9.mmd") -> str:
    """
    Export the compiled graph as a Mermaid diagram.

    Mermaid is a text-based diagram format that GitHub renders
    automatically in Markdown. We save the raw .mmd file for reference
    and embed it in README/architecture.md later.

    Returns the Mermaid source string.
    """
    compiled = build_graph()
    mermaid_str = compiled.get_graph().draw_mermaid()

    # Write to disk for documentation.
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(mermaid_str, encoding="utf-8")

    return mermaid_str


# ---------------------------------------------------------------------
# Quick sanity check when running this file directly.
# Usage: python -m src.pipeline.graph
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Building graph...")
    pipeline = build_graph()
    print("Graph compiled.\n")

    print("=" * 60)
    print("Test 1: Normal metrics (no anomaly expected)")
    print("=" * 60)
    result_normal = pipeline.invoke({"metrics": {"revenue": 500}})
    print(f"\nFinal state:\n{result_normal}\n")

    print("=" * 60)
    print("Test 2: Low revenue (anomaly expected)")
    print("=" * 60)
    result_anomaly = pipeline.invoke({"metrics": {"revenue": 50}})
    print(f"\nFinal state:\n{result_anomaly}\n")

    print("=" * 60)
    print("Mermaid diagram of the graph:")
    print("=" * 60)
    print(export_mermaid())