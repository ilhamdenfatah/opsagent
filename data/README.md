# OpsAgent Dataset

## Overview

This project uses a **synthetic business metrics dataset** — not a public Kaggle dataset. The dataset is generated programmatically with controlled anomaly injection, giving us objective ground truth for evaluation.

## Why Synthetic?

Three reasons:

1. **Controlled ground truth** — we know exactly when and why each anomaly occurs. This makes Phase 3 evaluation objective: we can measure detection accuracy, false positive rate, and root cause alignment against known answers.

2. **Business universality** — revenue, orders, churn, conversion rate. Metrics every recruiter understands regardless of industry.

3. **Portfolio differentiation** — building a dataset generator demonstrates deeper engineering than downloading a CSV.

## Regenerating the Dataset

```bash
python -m src.data.build_dataset
```

Output is deterministic — `seed=42` ensures identical results every run.

## Dataset Specification

| Property | Value |
|----------|-------|
| Timeframe | 180 days (2025-10-01 to 2026-03-29) |
| Granularity | Daily |
| Metrics | 6 columns |
| Planted anomalies | 15 events |
| Random seed | 42 (reproducible) |

## Metrics

| Column | Type | Baseline | Description |
|--------|------|----------|-------------|
| `daily_revenue` | float (IDR) | ~50,000,000 | Total daily revenue |
| `order_count` | int | ~800 | Number of orders per day |
| `avg_order_value` | float (IDR) | ~62,500 | Revenue / orders (derived) |
| `customer_churn_rate` | float (%) | ~2.5% | Daily churn rate |
| `support_ticket_count` | int | ~45 | Support tickets opened |
| `conversion_rate` | float (%) | ~3.2% | Visitor-to-buyer conversion |

## Generation Method: 4-Layer Approach

```
Final Value = Baseline × Trend × Seasonality × Noise × [Anomaly Multiplier]
```

1. **Baseline** — starting value per metric
2. **Trend** — daily growth rate (e.g. revenue +0.3%/day)
3. **Seasonality** — weekly (weekend effect) + monthly (payday effect)
4. **Noise** — Gaussian noise for realistic day-to-day variation
5. **Anomaly injection** — controlled multipliers at specific dates

## Planted Anomalies: Summary

15 anomalies across 4 categories:

| Category | Count | Example |
|----------|-------|---------|
| Sudden drops | 5 | Promo campaign ended, payment gateway outage |
| Gradual degradations | 3 | Competitor price war, slow performance regression |
| Spikes | 4 | Halloween campaign, Christmas peak, viral influencer |
| Multi-metric correlated | 3 | Holiday staffing failure (revenue + support + churn) |

Full ground truth with root causes and expected actions: `data/raw/anomaly_ground_truth.json`

## Files

| File | Description | In Git? |
|------|-------------|---------|
| `raw/metrics_daily.csv` | Main dataset with anomalies | ❌ Regenerate locally |
| `raw/metrics_daily_clean.csv` | Clean version without anomalies | ❌ Regenerate locally |
| `raw/anomaly_ground_truth.json` | Ground truth for all 15 anomalies | ❌ Regenerate locally |
| `golden_testset/` | Curated evaluation test cases (Phase 3) | ✅ |

Data files are gitignored — regenerate with `python -m src.data.build_dataset`.
