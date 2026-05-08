"""
Shared state for the OpsAgent multi-agent pipeline.

The state is a TypedDict that flows through every node in the LangGraph.
Each node reads from it and returns partial updates that LangGraph merges
back automatically.

Why TypedDict instead of Pydantic?
LangGraph internals optimize for dict-based state (faster merging,
native checkpointing, native serialization). We use Pydantic at the
boundary — for validating LLM outputs — but state itself stays as a
TypedDict. See the Day 9 notes for the full reasoning.
"""

from __future__ import annotations

from typing import Annotated, TypedDict
from operator import add


class AgentState(TypedDict, total=False):
    """
    Shared state passed between every node in the OpsAgent pipeline.

    Fields are marked optional via `total=False` because most of them
    get populated incrementally as the pipeline runs. The Coordinator
    receives partial state, fills in what it can, and routes onward.

    Day 9 (dummy nodes only) uses a minimal subset. Later phases will
    extend this with Pydantic models, message history, and memory.
    """

    # --- Input ---
    metrics: dict
    """Raw business metrics fed into the pipeline (e.g., daily KPIs)."""

    # --- Signal Detector output ---
    anomaly_detected: bool
    """True when the Signal Detector flags an anomaly worth investigating."""

    # --- Root Cause Analyzer output ---
    root_cause: str
    """Plain-text description of the diagnosed root cause."""

    # --- Action Recommender output ---
    recommended_action: str
    """Plain-text recommendation for what to do about the anomaly."""

    # --- Trace / observability ---
    messages: Annotated[list[str], add]
    """
    Append-only log of what each node did. The `add` reducer means
    when a node returns `{"messages": ["foo"]}`, LangGraph appends
    "foo" to the existing list instead of replacing it. This is the
    standard pattern for accumulating history across nodes.
    """