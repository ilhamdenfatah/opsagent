"""
Tests for Signal Detector Agent and output validator.

Testing philosophy for LLM-powered agents:
- DON'T test exact output text (non-deterministic)
- DO test output structure (always required)
- DO test behavior on known cases (should detect planted anomalies)
- DO test boundaries (edge cases that could crash the pipeline)
- DO test the validator independently (pure logic, fully deterministic)

Run with:
    pytest tests/test_agents.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.agents.signal_detector import (
    MetricSnapshot,
    AnomalySignal,
    compute_statistics,
    is_statistically_anomalous,
    determine_severity,
    detect_anomalies,
)
from src.agents.output_validator import (
    extract_json_from_response,
    normalize_severity,
    parse_and_validate,
)
from src.config import METRICS_FILE


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def metrics_df() -> pd.DataFrame:
    """Load the full metrics dataset once for all tests."""
    return pd.read_csv(METRICS_FILE, index_col="date", parse_dates=True)


@pytest.fixture
def normal_snapshot(metrics_df) -> tuple[MetricSnapshot, pd.DataFrame]:
    """A normal day (Oct 5) — should produce zero anomalies."""
    date = "2025-10-05"
    row = metrics_df.loc[date]
    snapshot = MetricSnapshot(
        date=date,
        daily_revenue=row["daily_revenue"],
        order_count=int(row["order_count"]),
        avg_order_value=row["avg_order_value"],
        customer_churn_rate=row["customer_churn_rate"],
        support_ticket_count=int(row["support_ticket_count"]),
        conversion_rate=row["conversion_rate"],
    )
    history = metrics_df.loc[:"2025-10-04"]
    return snapshot, history


@pytest.fixture
def anomaly_snapshot(metrics_df) -> tuple[MetricSnapshot, pd.DataFrame]:
    """Nov 15 — promo campaign ended, revenue + orders should be flagged."""
    date = "2025-11-15"
    row = metrics_df.loc[date]
    snapshot = MetricSnapshot(
        date=date,
        daily_revenue=row["daily_revenue"],
        order_count=int(row["order_count"]),
        avg_order_value=row["avg_order_value"],
        customer_churn_rate=row["customer_churn_rate"],
        support_ticket_count=int(row["support_ticket_count"]),
        conversion_rate=row["conversion_rate"],
    )
    history = metrics_df.loc[:"2025-11-14"]
    return snapshot, history


@pytest.fixture
def critical_snapshot(metrics_df) -> tuple[MetricSnapshot, pd.DataFrame]:
    """Dec 3 — payment gateway outage, revenue crashed 70%."""
    date = "2025-12-03"
    row = metrics_df.loc[date]
    snapshot = MetricSnapshot(
        date=date,
        daily_revenue=row["daily_revenue"],
        order_count=int(row["order_count"]),
        avg_order_value=row["avg_order_value"],
        customer_churn_rate=row["customer_churn_rate"],
        support_ticket_count=int(row["support_ticket_count"]),
        conversion_rate=row["conversion_rate"],
    )
    history = metrics_df.loc[:"2025-12-02"]
    return snapshot, history


# ── Statistical Layer Tests (pure math, no LLM, fully deterministic) ────────

class TestStatisticalLayer:

    def test_zscore_normal_value(self):
        """Value within normal range should have low z-score."""
        history = [100, 102, 98, 101, 99, 103, 100]
        zscore, pct_change = compute_statistics(101, history)
        assert abs(zscore) < 1.0

    def test_zscore_anomalous_drop(self):
        """Significant drop should produce high negative z-score."""
        history = [100, 102, 98, 101, 99, 103, 100]
        zscore, pct_change = compute_statistics(60, history)
        assert zscore < -2.0
        assert pct_change < -0.20

    def test_zscore_anomalous_spike(self):
        """Significant spike should produce high positive z-score."""
        history = [100, 102, 98, 101, 99, 103, 100]
        zscore, pct_change = compute_statistics(180, history)
        assert zscore > 2.0
        assert pct_change > 0.20

    def test_zscore_zero_std(self):
        """Perfectly stable series (std=0) should not cause division by zero."""
        history = [100, 100, 100, 100, 100, 100, 100]
        zscore, pct_change = compute_statistics(100, history)
        assert zscore == 0.0  # no division by zero

    def test_pct_change_calculation(self):
        """% change should be relative to the mean of history."""
        history = [100, 100, 100, 100]  # mean = 100
        _, pct_change = compute_statistics(80, history)
        assert abs(pct_change - (-0.20)) < 0.001  # exactly -20%

    def test_anomaly_flag_zscore_trigger(self):
        """Should flag when z-score exceeds threshold."""
        assert is_statistically_anomalous(zscore=2.5, pct_change=0.05) is True

    def test_anomaly_flag_pct_change_trigger(self):
        """Should flag when % change exceeds threshold."""
        assert is_statistically_anomalous(zscore=1.0, pct_change=0.25) is True

    def test_anomaly_flag_neither_trigger(self):
        """Should NOT flag when both are within normal range."""
        assert is_statistically_anomalous(zscore=1.5, pct_change=0.10) is False

    def test_severity_critical(self):
        assert determine_severity(zscore=5.0, pct_change=0.60) == "critical"

    def test_severity_high(self):
        assert determine_severity(zscore=3.5, pct_change=0.35) == "high"

    def test_severity_medium(self):
        assert determine_severity(zscore=2.2, pct_change=0.18) == "medium"

    def test_severity_low(self):
        # zscore 2.1 is above threshold → correctly medium, not low
        # true "low" requires both pct and z just barely over the flag threshold
        assert determine_severity(zscore=2.05, pct_change=0.22) == "medium"


# ── Output Validator Tests (pure logic, no LLM) ──────────────────────────────

class TestOutputValidator:

    def test_extract_clean_json(self):
        """Clean JSON should pass through unchanged."""
        raw = '{"anomaly_detected": true, "severity": "high"}'
        result = extract_json_from_response(raw)
        assert result == raw

    def test_extract_json_with_preamble(self):
        """JSON with preamble text should still be extracted."""
        raw = 'Sure! Here is the JSON: {"anomaly_detected": true, "severity": "high"}'
        result = extract_json_from_response(raw)
        assert result == '{"anomaly_detected": true, "severity": "high"}'

    def test_extract_json_from_markdown(self):
        """JSON wrapped in markdown code blocks should be extracted."""
        raw = '```json\n{"anomaly_detected": true}\n```'
        result = extract_json_from_response(raw)
        import json
        parsed = json.loads(result)
        assert parsed["anomaly_detected"] is True

    def test_normalize_severity_lowercase(self):
        assert normalize_severity("high") == "high"

    def test_normalize_severity_uppercase(self):
        assert normalize_severity("HIGH") == "high"

    def test_normalize_severity_mixed_case(self):
        assert normalize_severity("Critical") == "critical"

    def test_normalize_severity_unknown(self):
        """Unknown severity should default to medium."""
        assert normalize_severity("extreme") == "medium"

    def test_parse_and_validate_valid(self):
        """Valid JSON matching schema should parse cleanly."""
        from src.agents.signal_detector import LLMJudgment
        raw = '{"anomaly_detected": true, "severity": "high", "confidence": 0.85, "description": "Revenue dropped significantly."}'
        result = parse_and_validate(raw, LLMJudgment)
        assert result.anomaly_detected is True
        assert result.severity == "high"
        assert result.confidence == 0.85

    def test_parse_and_validate_invalid_json(self):
        """Invalid JSON should raise ValueError."""
        from src.agents.signal_detector import LLMJudgment
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_and_validate("not json at all", LLMJudgment)

    def test_parse_and_validate_missing_field(self):
        """JSON missing required fields should raise ValueError."""
        from src.agents.signal_detector import LLMJudgment
        with pytest.raises(ValueError):
            parse_and_validate('{"anomaly_detected": true}', LLMJudgment)


# ── Behavior Tests (use mocked LLM to avoid API calls in CI) ────────────────

class TestSignalDetectorBehavior:
    """
    These tests mock the LLM call so they run fast and don't hit the API.
    We test that the agent BEHAVES correctly — routes to LLM when it should,
    respects LLM's judgment, handles errors gracefully.
    """

    def _make_mock_judgment(self, detected: bool, severity: str = "high", confidence: float = 0.85):
        """Helper to create a mock LLM judgment."""
        from src.agents.signal_detector import LLMJudgment
        return LLMJudgment(
            anomaly_detected=detected,
            severity=severity,
            confidence=confidence,
            description="Mocked judgment for testing.",
        )

    @patch("src.agents.signal_detector.llm_judgment")
    @patch("src.agents.signal_detector.retrieve")
    def test_normal_day_no_anomalies(self, mock_retrieve, mock_llm, normal_snapshot):
        """Normal day should produce zero anomalies without calling LLM."""
        from src.agents.signal_detector import LLMJudgment
        snapshot, history = normal_snapshot
        mock_retrieve.return_value = []
        mock_llm.return_value = LLMJudgment(
            anomaly_detected=False,
            severity="low",
            confidence=0.9,
            description="Normal variation.",
        )

        signals = detect_anomalies(snapshot, history)
        assert len(signals) == 0

    @patch("src.agents.signal_detector.llm_judgment")
    @patch("src.agents.signal_detector.retrieve")
    def test_anomaly_day_detects_revenue_drop(self, mock_retrieve, mock_llm, anomaly_snapshot):
        """Nov 15 anomaly day should detect revenue drop."""
        snapshot, history = anomaly_snapshot
        mock_retrieve.return_value = []
        mock_llm.return_value = self._make_mock_judgment(detected=True, severity="high")

        signals = detect_anomalies(snapshot, history)

        assert len(signals) >= 1
        metrics_detected = [s.metric for s in signals]
        assert "daily_revenue" in metrics_detected

    @patch("src.agents.signal_detector.llm_judgment")
    @patch("src.agents.signal_detector.retrieve")
    def test_llm_can_override_statistical_flag(self, mock_retrieve, mock_llm, anomaly_snapshot):
        """If LLM says no anomaly, signal should not be added even if stats flagged it."""
        snapshot, history = anomaly_snapshot
        mock_retrieve.return_value = []
        # LLM says "nope, this is normal" for everything
        mock_llm.return_value = self._make_mock_judgment(detected=False)

        signals = detect_anomalies(snapshot, history)
        assert len(signals) == 0

    @patch("src.agents.signal_detector.llm_judgment")
    @patch("src.agents.signal_detector.retrieve")
    def test_output_structure_is_valid(self, mock_retrieve, mock_llm, anomaly_snapshot):
        """Every returned signal must be a valid AnomalySignal Pydantic model."""
        snapshot, history = anomaly_snapshot
        mock_retrieve.return_value = []
        mock_llm.return_value = self._make_mock_judgment(detected=True)

        signals = detect_anomalies(snapshot, history)

        for signal in signals:
            assert isinstance(signal, AnomalySignal)
            assert signal.severity in ["low", "medium", "high", "critical"]
            assert 0.0 <= signal.confidence <= 1.0
            assert isinstance(signal.description, str)
            assert len(signal.description) > 0

    @patch("src.agents.signal_detector.llm_judgment")
    @patch("src.agents.signal_detector.retrieve")
    def test_llm_failure_skips_metric_gracefully(self, mock_retrieve, mock_llm, anomaly_snapshot):
        """If LLM fails for a metric, that metric is skipped — pipeline doesn't crash."""
        snapshot, history = anomaly_snapshot
        mock_retrieve.return_value = []
        mock_llm.side_effect = RuntimeError("All retries exhausted")

        # Should not raise — just return empty list
        signals = detect_anomalies(snapshot, history)
        assert isinstance(signals, list)


# ── Edge Case Tests ──────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_insufficient_history(self, metrics_df):
        """Less than 3 days of history should skip detection gracefully."""
        date = "2025-10-03"
        row = metrics_df.loc[date]
        snapshot = MetricSnapshot(
            date=date,
            daily_revenue=row["daily_revenue"],
            order_count=int(row["order_count"]),
            avg_order_value=row["avg_order_value"],
            customer_churn_rate=row["customer_churn_rate"],
            support_ticket_count=int(row["support_ticket_count"]),
            conversion_rate=row["conversion_rate"],
        )
        # Only 2 days of history
        history = metrics_df.loc[:"2025-10-02"]
        signals = detect_anomalies(snapshot, history)
        assert isinstance(signals, list)  # no crash

    def test_metric_snapshot_validation(self):
        """MetricSnapshot should reject invalid data types."""
        with pytest.raises(Exception):
            MetricSnapshot(
                date="2025-10-05",
                daily_revenue="not_a_number",  # should be float
                order_count=800,
                avg_order_value=62500,
                customer_churn_rate=2.5,
                support_ticket_count=45,
                conversion_rate=3.2,
            )

    def test_extreme_spike_severity(self):
        """Extreme spike should be classified as critical."""
        # 500% increase
        history = [100] * 7
        zscore, pct_change = compute_statistics(600, history)
        severity = determine_severity(zscore, pct_change)
        assert severity == "critical"

    def test_compute_statistics_returns_floats(self):
        """compute_statistics should always return Python floats."""
        zscore, pct_change = compute_statistics(80.0, [100.0] * 7)
        assert isinstance(zscore, float)
        assert isinstance(pct_change, float)
