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
from pydantic import BaseModel, Field, field_validator


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