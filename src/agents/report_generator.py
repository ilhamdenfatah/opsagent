"""
Report Generator agent.

Single LLM call (MODEL_FAST — Llama 3.1 8B) that converts the structured
outputs of the three prior agents into a readable IncidentReport.

Context compression is the main work here: raw RCA + ActionPlan JSON is
~3000-4000 tokens. _compress_inputs() distills it to ~600-800 tokens of
essential facts so the 8B model isn't overloaded and stays focused.

No few-shot examples needed — the task is reformatting structured data
into prose, which small models handle reliably with clear section-level
instructions.
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.agents.schemas import ActionPlan, IncidentReport, RootCauseAnalysis
from src.config import AGENT_MODEL_ROUTING, GROQ_API_KEY, MAX_RETRIES_PER_AGENT


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Report Generator for OpsAgent, an automated business operations monitor.

You receive compressed facts from three prior agents and write a structured incident
report with four sections. Write as a thoughtful analyst — clear, direct, no filler.

Section instructions:

executive_summary
  2-3 sentences only. Audience: business stakeholders who won't read further.
  Cover: what happened (in plain business terms, no metric names), the root cause
  in one plain-language sentence, and the single most important action being taken.
  No percentages, no technical jargon, no metric column names.

analysis
  Start with "## Analysis". Audience: ops team and technical leads.
  Structure:
    - **Root cause**: one sentence, the main finding
    - **Confidence**: X% — brief reason for that confidence level
    - **Supporting evidence**: bullet list, each item citing what data was found
    - **Related metrics**: which other metrics were affected (if any)
    - **Alternatives considered**: briefly note what other explanations were ruled out

recommendations
  Start with "## Recommendations". Audience: team leads.
  Numbered list. Each item: "N. **[Title]** (Owner: X | Urgency: Y) — one-line description."
  Include all actions from the input, in priority order.

next_steps
  Start with "## Next Steps". Audience: whoever is on-call.
  Two subsections:
    - **Immediate (within 24h)**: bullet list of actions with urgency=immediate or within_24h
    - **Review**: one line suggesting a follow-up check in 7 days\
"""


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------

def _compress_inputs(
    anomaly_context: dict,
    rca: RootCauseAnalysis,
    plan: ActionPlan,
) -> str:
    """
    Distill RCA + ActionPlan into a compact fact block for the LLM.

    Keeps the 3 strongest evidence items and top 2 alternatives to stay
    within a token budget that works comfortably for MODEL_FAST.
    """
    # Anomaly context
    date = anomaly_context.get("date", "unknown")
    metric = anomaly_context.get("metric", "unknown")
    severity = anomaly_context.get("severity", "unknown")
    description = anomaly_context.get("description", "")

    # RCA — top evidence by strength order (strong first)
    strength_order = {"strong": 0, "moderate": 1, "weak": 2}
    sorted_evidence = sorted(
        rca.evidence, key=lambda e: strength_order.get(e.strength, 3)
    )
    evidence_lines = "\n".join(
        f"  - [{e.strength}] {e.description}"
        for e in sorted_evidence[:3]
    )

    # Alternatives (top 2)
    alt_lines = ""
    if rca.alternative_hypotheses:
        alts = rca.alternative_hypotheses[:2]
        alt_lines = "\nAlternatives ruled out:\n" + "\n".join(
            f"  - {a.hypothesis}: {a.disconfirming_evidence[:80]}"
            for a in alts
        )

    related = ", ".join(rca.related_metrics) if rca.related_metrics else "none"

    # ActionPlan — all actions (just title + first sentence of description)
    action_lines = "\n".join(
        f"  {i + 1}. [{a.priority_score:.2f}] {a.title}"
        f" | owner: {a.owner} | urgency: {a.urgency}"
        f"\n     {a.description.split('.')[0]}."
        for i, a in enumerate(plan.actions)
    )

    return (
        f"ANOMALY\n"
        f"Date: {date} | Metric: {metric} | Severity: {severity}\n"
        f"{('Description: ' + description) if description else ''}\n\n"
        f"ROOT CAUSE ANALYSIS\n"
        f"Primary cause (confidence={rca.confidence:.0%}): {rca.primary_cause}\n"
        f"Severity after investigation: {rca.severity_assessment}\n"
        f"Related metrics: {related}\n"
        f"Top evidence:\n{evidence_lines}"
        f"{alt_lines}\n\n"
        f"ACTION PLAN\n"
        f"Expected outcome: {plan.expected_outcome}\n"
        f"Actions:\n{action_lines}\n\n"
        f"Now write the IncidentReport with the four sections as instructed."
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_report_generator(
    anomaly_context: dict,
    rca: RootCauseAnalysis,
    plan: ActionPlan,
) -> IncidentReport:
    """
    Generate a structured IncidentReport from the pipeline's prior outputs.

    metadata.generated_at is set by Python after generation — the LLM
    is not trusted to know the current timestamp.
    """
    llm = ChatGroq(
        model=AGENT_MODEL_ROUTING["report_generator"],
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_compress_inputs(anomaly_context, rca, plan)),
    ]

    llm_structured = llm.with_structured_output(IncidentReport)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_AGENT):
        try:
            report: IncidentReport = llm_structured.invoke(messages)
            # Override generated_at — LLM doesn't know the real timestamp
            real_metadata = report.metadata.model_copy(
                update={"generated_at": datetime.now().isoformat()}
            )
            return report.model_copy(update={"metadata": real_metadata})
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES_PER_AGENT - 1:
                messages = messages + [
                    HumanMessage(
                        content=(
                            f"Validation error on attempt {attempt + 1}: {exc}. "
                            f"Fix the response and try again."
                        )
                    )
                ]

    raise RuntimeError(
        f"Report generation failed after {MAX_RETRIES_PER_AGENT} attempts"
    ) from last_exc
