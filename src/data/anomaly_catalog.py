"""
Anomaly catalog — the 15 "crime scenes" planted in our synthetic dataset.

This is the ground truth for Phase 3 evaluation. Every anomaly here has:
- Exact dates so we know WHEN to look
- Affected metrics so we know WHAT to look at
- Root cause narrative so we can judge if the AI diagnosed it correctly
- Expected actions so we can score recommendation quality

Think of this file as the answer key. The AI agents are the students.
"""

from src.data.generator import AnomalySpec


ANOMALY_CATALOG: list[AnomalySpec] = [

    # ---------- SUDDEN DROPS (5) ----------

    AnomalySpec(
        anomaly_id="ANM-001",
        date_start="2025-11-15",
        date_end="2025-11-17",
        metric="daily_revenue",
        anomaly_type="sudden_drop",
        magnitude=0.62,  # drops to 62% of normal = -38%
        severity="high",
        root_cause=(
            "November Sale promo campaign ended abruptly on Nov 14. "
            "Traffic from campaign ads stopped overnight, causing a 38% drop "
            "in revenue and order count over 3 days."
        ),
        expected_actions=[
            "Launch a replacement promo within 48 hours to maintain traffic momentum",
            "Analyze which product categories saw steepest drop",
            "Re-engage customers from campaign segment via email/push notification",
            "Review campaign end date alignment with marketing calendar",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-002",
        date_start="2025-11-15",
        date_end="2025-11-17",
        metric="order_count",
        anomaly_type="correlated",
        magnitude=0.65,
        severity="high",
        root_cause=(
            "Correlated with ANM-001. Same campaign end caused order volume "
            "to drop 35% alongside revenue."
        ),
        expected_actions=[
            "See ANM-001 — root cause and actions are the same incident",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-003",
        date_start="2025-12-03",
        date_end="2025-12-03",
        metric="daily_revenue",
        anomaly_type="sudden_drop",
        magnitude=0.30,  # drops to 30% of normal = -70%
        severity="critical",
        root_cause=(
            "Payment gateway outage from 08:00 to 20:00 on Dec 3. "
            "Most checkout attempts failed silently. Revenue dropped 70% "
            "and support tickets spiked simultaneously."
        ),
        expected_actions=[
            "Activate backup payment provider immediately",
            "Send customer communication acknowledging checkout issues",
            "Quantify lost orders and issue discount codes for affected customers",
            "Conduct post-mortem with payment gateway provider",
            "Add payment gateway health check to monitoring alerts",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-004",
        date_start="2025-12-03",
        date_end="2025-12-03",
        metric="support_ticket_count",
        anomaly_type="correlated",
        magnitude=3.50,  # spikes to 350% of normal
        severity="critical",
        root_cause=(
            "Correlated with ANM-003 (payment gateway outage). "
            "Customers unable to checkout flooded support with complaints."
        ),
        expected_actions=[
            "Prepare canned response for support team",
            "Escalate to engineering immediately",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-005",
        date_start="2026-02-10",
        date_end="2026-02-12",
        metric="conversion_rate",
        anomaly_type="sudden_drop",
        magnitude=0.55,
        severity="high",
        root_cause=(
            "A/B test on checkout flow pushed wrong variant to 60% of users on Feb 10. "
            "The variant had a broken promo code field, causing users to abandon at checkout. "
            "Conversion dropped 45% for 3 days until the test was rolled back."
        ),
        expected_actions=[
            "Roll back the failing A/B test variant immediately",
            "Audit all active A/B tests for broken UI elements",
            "Add conversion rate as a guardrail metric for future A/B tests",
            "Compensate users who encountered errors with a discount",
        ],
    ),

    # ---------- GRADUAL DEGRADATIONS (3) ----------

    AnomalySpec(
        anomaly_id="ANM-006",
        date_start="2025-12-15",
        date_end="2025-12-29",
        metric="customer_churn_rate",
        anomaly_type="gradual_degradation",
        magnitude=1.80,  # ramps up to 180% of normal = +80% churn
        severity="high",
        root_cause=(
            "A competitor launched an aggressive promo on Dec 15 targeting "
            "our customer segment. Churn rate crept up 80% over 2 weeks as "
            "price-sensitive customers churned to the competitor."
        ),
        expected_actions=[
            "Analyze churned customer profiles — identify price sensitivity segment",
            "Launch retention campaign targeting at-risk customers",
            "Review competitive pricing and adjust if margin allows",
            "Accelerate loyalty program rollout to increase switching cost",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-007",
        date_start="2026-01-10",
        date_end="2026-01-24",
        metric="conversion_rate",
        anomaly_type="gradual_degradation",
        magnitude=0.72,
        severity="medium",
        root_cause=(
            "Website performance degraded gradually after a Jan 10 backend deploy "
            "introduced an unoptimized database query. Page load time increased "
            "from 1.2s to 3.8s over 2 weeks, silently killing conversion rate."
        ),
        expected_actions=[
            "Run performance profiling to identify the slow query",
            "Roll back or hotfix the Jan 10 deploy",
            "Add p95 page load time to the anomaly detection watchlist",
            "Set up automated performance regression tests for deploys",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-008",
        date_start="2026-02-20",
        date_end="2026-03-05",
        metric="support_ticket_count",
        anomaly_type="gradual_degradation",
        magnitude=1.60,
        severity="medium",
        root_cause=(
            "A product quality issue with a top-selling item (batch from Feb 18 "
            "supplier) led to increasing customer complaints. Ticket volume "
            "crept up 60% over 2 weeks before QC caught the defective batch."
        ),
        expected_actions=[
            "Quarantine all items from the Feb 18 supplier batch",
            "Proactively contact customers who received affected items",
            "Issue refunds or replacements for defective products",
            "Review supplier QC process and add incoming inspection step",
        ],
    ),

    # ---------- SPIKES (4) ----------

    AnomalySpec(
        anomaly_id="ANM-009",
        date_start="2025-10-28",
        date_end="2025-10-31",
        metric="daily_revenue",
        anomaly_type="spike",
        magnitude=2.20,  # 220% of normal = +120%
        severity="low",  # spikes are usually good news
        root_cause=(
            "Halloween campaign + end-of-month payday coincided. "
            "Revenue doubled for 4 days. Nothing is broken — this is expected "
            "seasonal behavior worth documenting for next year's planning."
        ),
        expected_actions=[
            "Document Halloween + payday combo as a repeatable revenue pattern",
            "Ensure inventory was sufficient — check stockout rates during this period",
            "Plan larger inventory buffer for same period next year",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-010",
        date_start="2025-10-28",
        date_end="2025-10-31",
        metric="order_count",
        anomaly_type="correlated",
        magnitude=2.00,
        severity="low",
        root_cause="Correlated with ANM-009. Order volume doubled alongside revenue.",
        expected_actions=["See ANM-009"],
    ),

    AnomalySpec(
        anomaly_id="ANM-011",
        date_start="2025-12-23",
        date_end="2025-12-26",
        metric="daily_revenue",
        anomaly_type="spike",
        magnitude=2.50,
        severity="low",
        root_cause=(
            "Christmas shopping peak. Revenue 2.5x normal for 4 days. "
            "Expected seasonal spike — worth monitoring fulfillment capacity."
        ),
        expected_actions=[
            "Verify fulfillment team has sufficient capacity for order volume",
            "Monitor delivery SLA during peak period",
            "Capture learnings for Christmas 2026 capacity planning",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-012",
        date_start="2026-01-28",
        date_end="2026-01-30",
        metric="daily_revenue",
        anomaly_type="spike",
        magnitude=1.85,
        severity="low",
        root_cause=(
            "Viral social media post by a micro-influencer on Jan 27 drove "
            "unexpected traffic. Revenue up 85% for 3 days. "
            "Conversion rate held steady, confirming genuine demand spike."
        ),
        expected_actions=[
            "Identify the influencer and initiate a formal partnership",
            "Capture new customer emails from this cohort for retention",
            "Ensure inventory levels can sustain if virality continues",
        ],
    ),

    # ---------- MULTI-METRIC CORRELATED (3) ----------

    AnomalySpec(
        anomaly_id="ANM-013",
        date_start="2026-01-05",
        date_end="2026-01-07",
        metric="daily_revenue",
        anomaly_type="sudden_drop",
        magnitude=0.45,
        severity="critical",
        root_cause=(
            "New Year holiday skeleton crew caused fulfillment delays. "
            "Orders piled up, customers cancelled, revenue dropped 55%. "
            "Support tickets tripled. Churn rate ticked up slightly in the week after."
        ),
        expected_actions=[
            "Implement holiday staffing plan for Jan 1-7 in future years",
            "Set up automated order delay notifications",
            "Offer affected customers expedited shipping or discount",
        ],
    ),

    AnomalySpec(
        anomaly_id="ANM-014",
        date_start="2026-01-05",
        date_end="2026-01-07",
        metric="support_ticket_count",
        anomaly_type="correlated",
        magnitude=3.00,
        severity="critical",
        root_cause="Correlated with ANM-013. Fulfillment delays caused support surge.",
        expected_actions=["See ANM-013"],
    ),

    AnomalySpec(
        anomaly_id="ANM-015",
        date_start="2026-01-08",
        date_end="2026-01-12",
        metric="customer_churn_rate",
        anomaly_type="gradual_degradation",
        magnitude=1.40,
        severity="medium",
        root_cause=(
            "Lagging effect of ANM-013 (holiday fulfillment failure). "
            "Customers who experienced delays churned in the following week. "
            "This 3-5 day lag between operational failure and churn spike "
            "is a key pattern for the Root Cause Analyzer to learn."
        ),
        expected_actions=[
            "Run win-back campaign targeting customers from Jan 5-7 order cohort",
            "Offer loyalty points to customers affected by holiday delays",
        ],
    ),
]


# Quick lookup by anomaly_id
ANOMALY_BY_ID: dict[str, AnomalySpec] = {a.anomaly_id: a for a in ANOMALY_CATALOG}
