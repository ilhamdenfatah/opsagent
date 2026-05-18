"""
Root Cause Analyzer agent.

ReAct subgraph: LLM loops over investigation tools until it has enough
evidence, then a synthesis call produces a structured RootCauseAnalysis.

Two-phase design:
  Phase 1 — ReAct loop (llm.bind_tools): LLM investigates freely
  Phase 2 — Synthesis (llm.with_structured_output): extract structured output

Phases are separate because bind_tools + with_structured_output in one
call conflicts on Groq/Llama — the model gets confused about which
"function" to fill in.

langgraph.prebuilt is not available in the pinned version (1.1.10),
so ToolNode and routing are implemented manually here.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agents.schemas import InvestigationStep, RootCauseAnalysis
from src.config import (
    AGENT_MODEL_ROUTING,
    GROQ_API_KEY,
    MAX_PIPELINE_LOOPS,
    MAX_RETRIES_PER_AGENT,
)
from src.tools.data_query import get_rows_around_date, query_metrics
from src.tools.metric_calculator import (
    calculate_correlation,
    calculate_day_over_day_change,
    calculate_zscore,
)
from src.tools.rag_retriever import retrieve_similar_events


# ---------------------------------------------------------------------------
# LangChain @tool wrappers
# ---------------------------------------------------------------------------
# All tools return strings — ToolMessage content is always a string.
# Exceptions are caught and returned as error strings so the LLM can
# reason about failures rather than crashing the loop.

@tool
def query_metrics_tool(start_date: str, end_date: str) -> str:
    """
    Fetch daily metrics for a date range (both dates inclusive).

    Returns all 6 metrics per day: daily_revenue, order_count,
    customer_churn_rate, support_ticket_count, conversion_rate,
    avg_order_value. Use to see raw numbers over a period.

    Args:
        start_date: ISO date string, e.g. '2025-11-01'
        end_date: ISO date string, e.g. '2025-11-30'
    """
    try:
        return query_metrics(start_date, end_date).model_dump_json()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def get_rows_around_date_tool(target_date: str, window_days: int = 7) -> str:
    """
    Fetch daily metrics within ±window_days of target_date.

    Best tool for anomaly context — shows what changed in the days
    leading up to and following the anomaly date.

    Args:
        target_date: the anomaly date, e.g. '2025-11-15'
        window_days: days before and after to include (default 7)
    """
    try:
        return get_rows_around_date(target_date, window_days).model_dump_json()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def calculate_zscore_tool(
    metric: str, target_date: str, window_days: int = 30
) -> str:
    """
    Z-score of a metric on target_date vs the prior window_days baseline.

    Z-score beyond ±2.0 is statistically significant.
    Baseline window is strictly before target_date.

    Args:
        metric: one of daily_revenue, order_count, customer_churn_rate,
                support_ticket_count, conversion_rate, avg_order_value
        target_date: ISO date string, e.g. '2025-11-15'
        window_days: prior days used as baseline (default 30)
    """
    try:
        return calculate_zscore(metric, target_date, window_days).model_dump_json()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def calculate_dod_change_tool(metric: str, target_date: str) -> str:
    """
    Day-over-day percent change for a metric on target_date vs previous day.

    Use to quantify how sharply a metric moved on the anomaly date.

    Args:
        metric: one of daily_revenue, order_count, customer_churn_rate,
                support_ticket_count, conversion_rate, avg_order_value
        target_date: ISO date string, e.g. '2025-11-15'
    """
    try:
        return calculate_day_over_day_change(metric, target_date).model_dump_json()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def calculate_correlation_tool(
    metric_a: str, metric_b: str, start_date: str, end_date: str
) -> str:
    """
    Pearson correlation between two metrics over a date range.

    Use to determine if metrics moved together (correlated) or
    independently. Helps distinguish demand-side from supply-side causes.

    Args:
        metric_a: first metric name
        metric_b: second metric name
        start_date: ISO date string
        end_date: ISO date string
    """
    try:
        return calculate_correlation(
            metric_a, metric_b, start_date, end_date
        ).model_dump_json()
    except Exception as exc:
        return f"Error: {exc}"


@tool
def retrieve_similar_events_tool(
    query_text: str,
    top_k: int = 5,
    chunk_type: str | None = None,
    metric: str | None = None,
    severity: str | None = None,
) -> str:
    """
    Search historical context for events similar to the current anomaly.

    Returns past anomaly events and weekly summaries from the vector store.
    Use to find if this pattern has occurred before.

    Args:
        query_text: natural language description of what you're looking for
        top_k: number of results (default 5)
        chunk_type: optional filter — 'weekly_summary' or 'anomaly_event'
        metric: optional filter — e.g. 'daily_revenue'
        severity: optional filter — e.g. 'high', 'critical'
    """
    try:
        result = retrieve_similar_events(query_text, top_k, chunk_type, metric, severity)
        return result.formatted_context
    except Exception as exc:
        return f"Error: {exc}"


_TOOLS = [
    query_metrics_tool,
    get_rows_around_date_tool,
    calculate_zscore_tool,
    calculate_dod_change_tool,
    calculate_correlation_tool,
    retrieve_similar_events_tool,
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Root Cause Analyzer for OpsAgent, an automated business operations monitor.

An anomaly has been detected:
- Date: {anomaly_date}
- Metric: {anomaly_metric}
- Severity: {anomaly_severity}

Your task: determine why this anomaly occurred by systematically investigating the data.

Available tools:
- query_metrics_tool: raw daily metrics for a date range
- get_rows_around_date_tool: ±N days context around a date (start here for anomaly context)
- calculate_zscore_tool: statistical significance vs prior 30-day baseline
- calculate_dod_change_tool: day-over-day percent change
- calculate_correlation_tool: Pearson correlation between two metrics
- retrieve_similar_events_tool: search historical context for similar past events

Investigation approach:
1. Confirm the anomaly — check z-score and DoD change for {anomaly_metric} on {anomaly_date}
2. Check context — look at ±7 days to see if isolated or part of a trend
3. Check related metrics — are other metrics also anomalous? Check correlations
4. Check history — retrieve similar past events to see if this pattern recurs

Stop when you have sufficient evidence to form a confident hypothesis.
Only claim what the data shows — do not fabricate evidence.\
"""

_SYNTHESIS_PROMPT = """\
Based on your investigation above, produce a complete structured root cause analysis.

Field requirements:
- primary_cause: 1-3 sentences, the most likely explanation
- confidence: 0.0-1.0 based on evidence quality (0.5 = uncertain, 0.8 = strong support)
- evidence: every data point you found — include source (sql_query / metric_calculator / \
rag_retriever / reasoning) and strength (weak / moderate / strong)
- related_metrics: other metrics that moved with {anomaly_metric}, empty list if isolated
- investigation_trace: provide at least 1 step (will be replaced with actual tool calls)
- alternative_hypotheses: REQUIRED if confidence >= 0.6 — what else could explain it and why rejected
- counterfactuals: REQUIRED if confidence >= 0.6 — what you'd expect if true vs what you observed
- severity_assessment: your severity judgment after investigation (low / medium / high / critical)\
"""


# ---------------------------------------------------------------------------
# ReAct subgraph
# ---------------------------------------------------------------------------

class _RCAState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_subgraph(llm: ChatGroq):
    tool_map = {t.name: t for t in _TOOLS}
    llm_with_tools = llm.bind_tools(_TOOLS)

    def agent_node(state: _RCAState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(state: _RCAState) -> dict:
        last_msg = state["messages"][-1]
        results = []
        for tc in last_msg.tool_calls:
            try:
                result = tool_map[tc["name"]].invoke(tc)
            except Exception as exc:
                result = ToolMessage(
                    content=f"Tool error: {exc}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            results.append(result)
        return {"messages": results}

    def route(state: _RCAState) -> str:
        last = state["messages"][-1]
        # AIMessage has tool_calls when it wants to call a tool;
        # empty list or absent means it's done investigating.
        return "tools" if getattr(last, "tool_calls", None) else END

    g = StateGraph(_RCAState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_investigation_trace(
    messages: list[BaseMessage],
) -> list[InvestigationStep]:
    """
    Build the real InvestigationStep trace from the ReAct message history.

    One step per tool call. AIMessages without tool calls (pure reasoning)
    also become steps with tool_used=None, so the trace reflects the full
    thought process, not just the mechanical tool invocations.
    """
    steps: list[InvestigationStep] = []
    step_num = 1

    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue

        thought = msg.content if isinstance(msg.content, str) else str(msg.content)

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                # Find the matching ToolMessage by tool_call_id
                observation = "(no result)"
                for candidate in messages[i + 1:]:
                    if (
                        isinstance(candidate, ToolMessage)
                        and candidate.tool_call_id == tc["id"]
                    ):
                        observation = candidate.content
                        break

                steps.append(InvestigationStep(
                    step_number=step_num,
                    thought=thought or f"Calling {tc['name']}",
                    tool_used=tc["name"],
                    tool_input=tc["args"],
                    observation=observation,
                ))
                step_num += 1

        elif thought and thought.strip():
            # Pure reasoning step — include so evaluators can see the full trace
            steps.append(InvestigationStep(
                step_number=step_num,
                thought=thought,
                tool_used=None,
                tool_input=None,
                observation="(reasoning step, no tool called)",
            ))
            step_num += 1

    # Schema requires at least 1 step
    if not steps:
        steps.append(InvestigationStep(
            step_number=1,
            thought="Investigation completed without tool calls.",
            tool_used=None,
            tool_input=None,
            observation="(no tools called)",
        ))

    return steps


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_root_cause_agent(
    anomaly_date: str,
    anomaly_metric: str,
    anomaly_severity: str,
) -> RootCauseAnalysis:
    """
    Run the ReAct investigation loop and return a structured RootCauseAnalysis.

    Phase 1: bound-tools ReAct loop — LLM investigates freely with the 6 tools.
    Phase 2: synthesis call with structured output — produce validated RCA.
    The investigation_trace in the returned model reflects actual tool calls.
    """
    llm = ChatGroq(
        model=AGENT_MODEL_ROUTING["root_cause_analyzer"],
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    # Phase 1: ReAct loop
    subgraph = _build_subgraph(llm)

    initial_messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT.format(
            anomaly_date=anomaly_date,
            anomaly_metric=anomaly_metric,
            anomaly_severity=anomaly_severity,
        )),
        HumanMessage(content=(
            f"Investigate the anomaly: {anomaly_metric} on {anomaly_date} "
            f"(severity: {anomaly_severity}). Use the tools to find the root cause."
        )),
    ]

    loop_result = subgraph.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": MAX_PIPELINE_LOOPS * 2 + 2},
    )
    messages: list[BaseMessage] = loop_result["messages"]

    # Extract actual tool-call trace before synthesis can overwrite it
    trace = _extract_investigation_trace(messages)

    # Phase 2: Synthesis into structured output
    synthesis_messages = messages + [
        HumanMessage(content=_SYNTHESIS_PROMPT.format(anomaly_metric=anomaly_metric))
    ]
    llm_structured = llm.with_structured_output(RootCauseAnalysis)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_AGENT):
        try:
            rca: RootCauseAnalysis = llm_structured.invoke(synthesis_messages)
            # Replace LLM's recalled trace with the one we extracted from actual messages
            return rca.model_copy(update={"investigation_trace": trace})
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES_PER_AGENT - 1:
                synthesis_messages = synthesis_messages + [
                    HumanMessage(
                        content=(
                            f"Validation error on attempt {attempt + 1}: {exc}. "
                            f"Fix the response and try again."
                        )
                    )
                ]

    raise RuntimeError(
        f"Root cause synthesis failed after {MAX_RETRIES_PER_AGENT} attempts"
    ) from last_exc
