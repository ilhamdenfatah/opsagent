"""
Pydantic schemas for agent outputs.

These models are the "contracts" between agents. Each agent's output
must conform to its schema — no exceptions. This gives us:
  1. Reliable parsing (no regex on freeform text)
  2. Type safety downstream (Action Recommender knows exactly what
     fields RootCauseAnalysis has)
  3. Free validation (Pydantic raises clear errors on bad outputs)
  4. Easy serialization (model.model_dump() → dict → SQLite/JSON)

Why one file for all schemas?
Schemas are shared across agents. Putting them in one module avoids
circular imports and makes the data flow easy to read at a glance.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# =====================================================================
# Shared types
# =====================================================================

SeverityLevel = Literal["low", "medium", "high", "critical"]
"""Used across all agents. Keep this in sync with config.SEVERITY_LEVELS."""

ConfidenceScore = float
"""Float between 0.0 and 1.0. Validated per-field."""


# =====================================================================
# Investigation primitives — used inside RootCauseAnalysis
# =====================================================================

class InvestigationStep(BaseModel):
    """
    A single step in the agent's reasoning trace.

    Captures one Reason → Act → Observe cycle. We log every step
    so Phase 3 evaluation can score "did the agent investigate
    sensibly?" — not just "was the final answer correct?".
    """

    step_number: int = Field(..., ge=1, description="1-indexed step in the loop")
    thought: str = Field(..., description="What the agent was thinking before acting")
    tool_used: str | None = Field(
        None,
        description="Name of the tool called, e.g. 'query_metrics'. None if pure reasoning step.",
    )
    tool_input: dict | None = Field(
        None,
        description="Arguments passed to the tool. None if no tool was used.",
    )
    observation: str = Field(
        ...,
        description="What the agent observed after the action (tool result summary or reasoning conclusion).",
    )


class Evidence(BaseModel):
    """
    A piece of evidence supporting (or refuting) a hypothesis.

    Every claim the agent makes about the root cause MUST be backed
    by evidence. This is the anti-hallucination guardrail: if the
    agent says "revenue dropped because of an outage," it has to
    point to specific data showing the outage's impact.
    """

    description: str = Field(..., description="What the evidence says")
    source: Literal["sql_query", "metric_calculator", "rag_retriever", "reasoning"] = Field(
        ...,
        description="Where this evidence came from. 'reasoning' means LLM inference, not external data.",
    )
    strength: Literal["weak", "moderate", "strong"] = Field(
        ...,
        description="How compelling this evidence is. Strong = direct measurement, weak = circumstantial.",
    )


class AlternativeHypothesis(BaseModel):
    """
    An alternative explanation the agent considered but rejected.

    Why we force this:
      - Real investigators consider multiple hypotheses.
      - Forcing the agent to articulate AND reject alternatives reduces
        overconfidence in the primary cause.
      - In Phase 3, we can score "did the agent steelman alternatives
        before committing to the main answer?"

    Critical guardrail: every alternative MUST have disconfirming_evidence.
    No "this could also be the cause but I have nothing to back it up"
    free-form speculation. If there's no disconfirming evidence, the
    hypothesis shouldn't be in the output.
    """

    hypothesis: str = Field(..., description="The alternative explanation")
    disconfirming_evidence: str = Field(
        ...,
        min_length=20,
        description="Why this hypothesis was rejected. Must reference specific data or reasoning.",
    )


class CounterfactualCheck(BaseModel):
    """
    A 'what would we expect to see if X were true vs what we actually see' check.

    This is the rigorous-investigator pattern. Instead of just asserting
    'revenue dropped because of outage', the agent says:
      - 'If outage caused it, we'd expect support tickets to spike too.'
      - 'Actual data: support tickets DID spike +180% on the same day.'
      - 'Therefore, outage hypothesis is consistent with observations.'
    """

    if_cause_were_true: str = Field(
        ..., description="Predicted observation under the proposed cause"
    )
    actual_observation: str = Field(
        ..., description="What was actually observed in the data"
    )
    consistency: Literal["consistent", "inconsistent", "ambiguous"] = Field(
        ..., description="Does the actual observation match the prediction?"
    )


# =====================================================================
# Main output: RootCauseAnalysis
# =====================================================================

class RootCauseAnalysis(BaseModel):
    """
    The complete output of the Root Cause Analyzer agent.

    This is the contract that downstream agents (Action Recommender,
    Report Generator) consume. Every field is mandatory unless marked
    optional — agents can't skip work just because it's hard.

    Confidence-gated fields:
      - alternative_hypotheses and counterfactuals are only required
        when confidence >= 0.6. Below that threshold, the agent is
        allowed to admit uncertainty rather than fabricate analysis.
        See the field_validator below.
    """

    # --- Primary diagnosis ---
    primary_cause: str = Field(
        ...,
        min_length=20,
        description="The main hypothesized root cause, in 1-3 sentences.",
    )
    confidence: ConfidenceScore = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in primary_cause. 0=guess, 1=certain.",
    )

    # --- Supporting evidence (always required) ---
    evidence: list[Evidence] = Field(
        ...,
        min_length=1,
        description="At least one piece of evidence is required. No bare assertions.",
    )

    # --- Related metrics (always required if multi-metric anomaly) ---
    related_metrics: list[str] = Field(
        default_factory=list,
        description="Other metrics that moved alongside the anomaly. Empty list if isolated.",
    )

    # --- Investigation trace (always required) ---
    investigation_trace: list[InvestigationStep] = Field(
        ...,
        min_length=1,
        description="Step-by-step reasoning trace. Used for evaluation in Phase 3.",
    )

    # --- Rich analysis (confidence-gated) ---
    alternative_hypotheses: list[AlternativeHypothesis] = Field(
        default_factory=list,
        description=(
            "Alternative causes considered and rejected. "
            "REQUIRED when confidence >= 0.6, optional below."
        ),
    )
    counterfactuals: list[CounterfactualCheck] = Field(
        default_factory=list,
        description=(
            "What-if checks supporting the primary cause. "
            "REQUIRED when confidence >= 0.6, optional below."
        ),
    )

    # --- Severity reassessment ---
    severity_assessment: SeverityLevel = Field(
        ...,
        description="The agent's severity judgment after investigation. May differ from initial flag.",
    )

    # --- Cross-cutting validation ---
    @field_validator("alternative_hypotheses")
    @classmethod
    def require_alternatives_when_confident(
        cls, v: list[AlternativeHypothesis], info
    ) -> list[AlternativeHypothesis]:
        """
        Guardrail: if the agent claims high confidence (>=0.6),
        it MUST have considered alternatives. This prevents the
        'I'm 95% sure but didn't think of any other options' failure mode.

        Note: at validation time, info.data contains already-validated fields.
        Since `confidence` is declared earlier in the class, it's available here.
        """
        confidence = info.data.get("confidence", 0.0)
        if confidence >= 0.6 and len(v) == 0:
            raise ValueError(
                f"At confidence={confidence}, at least 1 alternative_hypothesis is required. "
                f"High confidence demands consideration of alternatives."
            )
        return v

    @field_validator("counterfactuals")
    @classmethod
    def require_counterfactuals_when_confident(
        cls, v: list[CounterfactualCheck], info
    ) -> list[CounterfactualCheck]:
        """Same guardrail as alternatives, but for counterfactuals."""
        confidence = info.data.get("confidence", 0.0)
        if confidence >= 0.6 and len(v) == 0:
            raise ValueError(
                f"At confidence={confidence}, at least 1 counterfactual is required."
            )
        return v


# =====================================================================
# Action Recommender output
# =====================================================================

class ActionItem(BaseModel):
    """
    A single recommended action with priority scoring.

    priority_score drives the ordering of actions in ActionPlan — higher
    score means "do this sooner." The model_validator on ActionPlan
    re-sorts items by this score so consumers always get them in order.
    """

    title: str = Field(..., description="Short name for the action, ~60 chars max")
    description: str = Field(
        ...,
        min_length=30,
        description="What exactly to do, 2-4 sentences. Specific enough to act on.",
    )
    owner: str = Field(
        ...,
        description="Team or role responsible, e.g. 'Marketing team', 'Engineering on-call'",
    )
    urgency: Literal["immediate", "within_24h", "within_week", "within_month"] = Field(
        ...,
        description="When to act: immediate = within the hour, within_24h = today, etc.",
    )
    impact: Literal["low", "medium", "high"] = Field(
        ..., description="Expected business impact if this action is taken"
    )
    effort: Literal["low", "medium", "high"] = Field(
        ..., description="Relative effort required to execute this action"
    )
    priority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0–1.0. 1.0 = highest priority. Derived from urgency × impact ÷ effort.",
    )


class ActionPlan(BaseModel):
    """
    The complete output of the Action Recommender agent.

    actions is guaranteed to be sorted by priority_score descending
    (model_validator handles this), so index 0 is always the top priority.
    """

    anomaly_summary: str = Field(
        ...,
        min_length=20,
        description="1-2 sentence restatement of the anomaly being addressed",
    )
    root_cause_summary: str = Field(
        ...,
        min_length=20,
        description="Brief restatement of the diagnosed root cause for context",
    )
    actions: list[ActionItem] = Field(
        ...,
        min_length=2,
        description="Prioritized actions, at least 2 items. Sorted descending by priority_score.",
    )
    expected_outcome: str = Field(
        ...,
        min_length=20,
        description="What success looks like if all actions are taken",
    )

    @model_validator(mode="after")
    def sort_actions_by_priority(self) -> ActionPlan:
        # Re-sort in case the LLM returned items in the wrong order.
        # Silently corrects rather than adding a retry for a trivial ordering issue.
        self.actions = sorted(self.actions, key=lambda a: a.priority_score, reverse=True)
        return self


# =====================================================================
# Report Generator output
# =====================================================================

class ReportMetadata(BaseModel):
    """Structured metadata attached to every IncidentReport."""

    generated_at: str = Field(
        ..., description="ISO datetime when the report was generated. Set by Python, not the LLM."
    )
    anomaly_date: str = Field(..., description="The date of the anomaly, YYYY-MM-DD")
    anomaly_metric: str = Field(..., description="The primary metric that was anomalous")
    severity: SeverityLevel = Field(..., description="Final severity after investigation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="RCA confidence score")
    action_count: int = Field(..., ge=1, description="Number of recommended actions")


class IncidentReport(BaseModel):
    """
    The complete output of the Report Generator agent.

    Four markdown-formatted sections for different audiences, plus metadata.
    `report_markdown` is a computed field that joins all sections — it's
    included in model_dump() so state["report"]["report_markdown"] works
    without a separate assembly step.
    """

    executive_summary: str = Field(
        ...,
        min_length=30,
        description=(
            "2-3 sentences for a non-technical audience. "
            "Cover: what happened, root cause in plain language, #1 action. "
            "No metric names, no percentages."
        ),
    )
    analysis: str = Field(
        ...,
        min_length=50,
        description=(
            "Markdown H2 section. Root cause, confidence, evidence bullet list, "
            "related metrics, alternatives considered."
        ),
    )
    recommendations: str = Field(
        ...,
        min_length=50,
        description=(
            "Markdown H2 section. All actions as a numbered list with owner, "
            "urgency, and one-line description each."
        ),
    )
    next_steps: str = Field(
        ...,
        min_length=30,
        description=(
            "Markdown H2 section. Immediate actions (within 24h) "
            "and a suggested review date."
        ),
    )
    metadata: ReportMetadata

    @computed_field
    @property
    def report_markdown(self) -> str:
        """Full assembled incident report as a single markdown document."""
        return (
            "# Incident Report\n\n"
            f"{self.executive_summary}\n\n"
            f"{self.analysis}\n\n"
            f"{self.recommendations}\n\n"
            f"{self.next_steps}"
        )