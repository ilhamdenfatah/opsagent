"""
Action Recommender agent.

Single LLM call with few-shot prompting — no tools, no subgraph.
Takes a RootCauseAnalysis and produces a prioritized ActionPlan.

Why no ReAct loop: the RCA already contains all the evidence. This agent's
job is judgment (what to do), not investigation (what happened). That's a
pattern-matching + ranking task where few-shot examples outperform tool loops.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.agents.schemas import ActionPlan, Evidence, RootCauseAnalysis
from src.config import AGENT_MODEL_ROUTING, GROQ_API_KEY, MAX_RETRIES_PER_AGENT


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Action Recommender for OpsAgent, an automated business operations monitor.

You receive a root cause analysis for a detected business anomaly and produce a
prioritized action plan. Each action must be specific, ownable by a named team,
and assigned a priority_score (0.0–1.0) reflecting urgency × impact ÷ effort.

Study these examples, then apply the same reasoning to the new case.

---
EXAMPLE 1
Anomaly: daily_revenue -25%, order_count -28% | severity: high
Root cause (confidence=0.85): "November Sale" campaign ended Nov 14 with no replacement.
Traffic dropped immediately as paid search and social spend cut off.

Actions:
[0.95] "Relaunch replacement promotional campaign"
  owner: Marketing team | urgency: immediate | impact: high | effort: medium
  → Launch a replacement campaign targeting the same audience segments. Start with
    top-performing creatives from the ended campaign to minimize ramp-up time.

[0.75] "Run campaign attribution analysis"
  owner: Marketing analytics | urgency: within_24h | impact: medium | effort: low
  → Identify which channels (paid search, social, email) drove the most revenue
    during the ended campaign to inform replacement budget allocation.

[0.55] "Set up campaign expiry alerts"
  owner: Marketing operations | urgency: within_week | impact: medium | effort: low
  → Configure automated alerts 2 weeks before any active promotional campaign
    ends so there is always time to prepare a replacement.

Expected outcome: Revenue returns to baseline within 5-7 days as replacement ramps up.

---
EXAMPLE 2
Anomaly: support_ticket_count +180% | severity: high
Root cause (confidence=0.90): Payment gateway intermittent failures causing checkout errors.
Correlated with conversion_rate -22% on the same day.

Actions:
[0.98] "Escalate to payment provider SLA breach"
  owner: Engineering on-call | urgency: immediate | impact: high | effort: low
  → Open a P1 ticket with the payment provider. Attach error logs showing the
    failure rate and time window. Request an ETA for resolution.

[0.80] "Send proactive customer communication"
  owner: Customer support | urgency: within_24h | impact: medium | effort: low
  → Email affected customers acknowledging the issue, confirming their orders
    were not charged, and offering a discount code as goodwill.

[0.65] "Add payment failure rate to operations dashboard"
  owner: Engineering | urgency: within_week | impact: high | effort: medium
  → Instrument checkout failure rate as a real-time metric with a PagerDuty
    alert at >2% failure rate so future incidents are caught within minutes.

Expected outcome: Tickets return to baseline within 24h of payment gateway resolution.

---
EXAMPLE 3
Anomaly: customer_churn_rate +40% | severity: high
Root cause (confidence=0.75): Competitor launched aggressive 30%-off campaign targeting
our core price-sensitive segment.

Actions:
[0.90] "Offer loyalty discount to at-risk segments"
  owner: Marketing team | urgency: immediate | impact: high | effort: medium
  → Identify customers who signed up within the last 90 days (most price-sensitive)
    and send a targeted retention offer matching or slightly beating competitor pricing.

[0.70] "Conduct competitive pricing review"
  owner: Product | urgency: within_week | impact: high | effort: medium
  → Analyze the competitor's full pricing structure and identify specific tiers
    where we are most exposed. Propose pricing adjustments for leadership review.

[0.50] "Strengthen retention email sequence"
  owner: Marketing | urgency: within_week | impact: medium | effort: high
  → Audit the current post-signup nurture sequence. Add a "why us vs competitors"
    email at day 14 and a milestone reward email at day 30.

Expected outcome: Churn rate returns to baseline within 2 weeks as retention offers activate.

---
EXAMPLE 4
Anomaly: conversion_rate -30%, avg_order_value stable | severity: high
Root cause (confidence=0.88): Checkout UI regression introduced by yesterday's deployment.
Support tickets also +45% with complaints about the checkout flow.

Actions:
[0.98] "Roll back yesterday's deployment"
  owner: Engineering on-call | urgency: immediate | impact: high | effort: low
  → Immediately roll back to the previous stable build. Confirm conversion rate
    recovery by monitoring for 30 minutes post-rollback.

[0.72] "Add conversion rate to deployment health checks"
  owner: Engineering | urgency: within_24h | impact: high | effort: low
  → Add a canary check that blocks deployment promotion if conversion rate drops
    more than 5% relative to a 15-minute pre-deploy baseline.

[0.55] "Audit A/B test variants for regression source"
  owner: Engineering | urgency: within_week | impact: medium | effort: medium
  → Identify which code path in the rolled-back deployment caused the regression.
    Write a regression test before re-deploying.

Expected outcome: Conversion rate recovers to baseline within 1 hour of rollback.

---
EXAMPLE 5
Anomaly: daily_revenue +35%, order_count +40% (positive spike) | severity: low
Root cause (confidence=0.70): Product featured in a high-traffic social media post that
went viral. Inbound traffic spiked +300% for 6 hours.

Actions:
[0.90] "Amplify the viral content"
  owner: Marketing team | urgency: immediate | impact: high | effort: low
  → Share the viral post across all owned channels. Engage with the original
    creator. Consider a paid boost to extend the reach window.

[0.75] "Scale infrastructure for continued traffic"
  owner: Engineering | urgency: within_24h | impact: high | effort: medium
  → Review auto-scaling limits. If traffic remains elevated, increase instance
    capacity proactively to avoid degraded performance killing the conversion rate.

[0.45] "Document the trigger for the growth playbook"
  owner: Marketing | urgency: within_week | impact: low | effort: low
  → Record what drove the spike (which product, which creator, which platform)
    in the growth playbook for future campaign ideation.

Expected outcome: Revenue stays elevated for 3-5 days while amplification runs.

---
EXAMPLE 6
Anomaly: daily_revenue -22%, order_count -25%, conversion_rate -18% (all correlated)
severity: critical
Root cause (confidence=0.92): Major platform outage lasting 4 hours. All checkout
attempts failed. Support tickets +320%.

Actions:
[0.99] "Declare incident and mobilize SRE response"
  owner: Engineering on-call | urgency: immediate | impact: high | effort: low
  → Declare a P0 incident in the incident management system. Page the SRE team.
    Assign an incident commander and begin the recovery runbook.

[0.88] "Update status page and notify customers"
  owner: Customer support | urgency: immediate | impact: medium | effort: low
  → Post a status update acknowledging the outage. Set up automated updates
    every 30 minutes. Prepare a post-resolution email for affected customers.

[0.65] "Conduct blameless post-mortem"
  owner: Engineering | urgency: within_week | impact: high | effort: high
  → Document the full timeline, root cause, and contributing factors. Produce
    5 concrete prevention measures with owners and deadlines.

Expected outcome: Services restored within the incident SLA; post-mortem prevents recurrence.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(rca: RootCauseAnalysis) -> str:
    """Format RCA fields into the input block the LLM sees for the current case."""
    related = ", ".join(rca.related_metrics) if rca.related_metrics else "none"

    evidence_lines = "\n".join(
        f"  - [{e.strength}, {e.source}] {e.description}"
        for e in rca.evidence
    )

    return (
        f"Current anomaly to produce an action plan for:\n"
        f"Primary cause (confidence={rca.confidence:.2f}): {rca.primary_cause}\n"
        f"Severity: {rca.severity_assessment}\n"
        f"Related metrics affected: {related}\n"
        f"Evidence:\n{evidence_lines}\n\n"
        f"Produce a prioritized ActionPlan following the pattern in the examples above."
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_action_recommender(rca: RootCauseAnalysis) -> ActionPlan:
    """
    Produce a prioritized ActionPlan from an investigated RootCauseAnalysis.

    Single LLM call with few-shot system prompt — no tools needed since
    the RCA already contains all the evidence.
    """
    llm = ChatGroq(
        model=AGENT_MODEL_ROUTING["action_recommender"],
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(rca)),
    ]

    llm_structured = llm.with_structured_output(ActionPlan)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_AGENT):
        try:
            return llm_structured.invoke(messages)
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
        f"Action plan generation failed after {MAX_RETRIES_PER_AGENT} attempts"
    ) from last_exc
