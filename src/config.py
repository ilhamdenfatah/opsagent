"""
Central configuration for OpsAgent.

All constants, model names, thresholds, and paths live here.
If you need to tweak a value, this is the only place you should touch.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_TESTSET_DIR = DATA_DIR / "golden_testset"

# Key data files
METRICS_FILE = RAW_DATA_DIR / "metrics_daily.csv"
METRICS_CLEAN_FILE = RAW_DATA_DIR / "metrics_daily_clean.csv"
GROUND_TRUTH_FILE = RAW_DATA_DIR / "anomaly_ground_truth.json"

# --- API Keys (loaded from .env) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- LLM Models ---
# Llama 3.3 70B: best reasoning quality, use for heavy lifting
MODEL_PRIMARY = "llama-3.3-70b-versatile"
# Llama 3.1 8B: fast and cheap(er), use for simple tasks or when rate-limited
MODEL_FAST = "llama-3.1-8b-instant"
# Gemini Flash: backup evaluator, fallback if Groq is down
MODEL_EVAL_BACKUP = "gemini-1.5-flash"

# Which agent uses which model
AGENT_MODEL_ROUTING: dict[str, str] = {
    "signal_detector": MODEL_FAST,        # simple classification task
    "root_cause_analyzer": MODEL_PRIMARY,  # heavy reasoning
    "action_recommender": MODEL_PRIMARY,   # needs good judgment
    "report_generator": MODEL_FAST,        # mostly formatting, not reasoning
    "coordinator": MODEL_FAST,             # routing decisions, lightweight
    "llm_judge": MODEL_PRIMARY,            # needs careful scoring
}

# --- Vector Database ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "opsagent_metrics")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # output size of all-MiniLM-L6-v2
TOP_K_RETRIEVAL = 5        # how many chunks to retrieve per query

# --- Dataset Config ---
DATASET_START_DATE = "2025-10-01"
DATASET_END_DATE = "2026-03-29"
DATASET_DAYS = 180
RANDOM_SEED = 42  # for reproducibility — never change this

# Metric baselines (daily values)
METRIC_BASELINES: dict[str, float] = {
    "daily_revenue": 50_000_000,    # IDR 50 juta
    "order_count": 800,
    "customer_churn_rate": 2.5,     # percentage
    "support_ticket_count": 45,
    "conversion_rate": 3.2,         # percentage
    # avg_order_value is derived: daily_revenue / order_count
}

# --- Anomaly Detection Thresholds ---
ZSCORE_THRESHOLD = 2.0             # flag if z-score > this
DAY_OVER_DAY_CHANGE_THRESHOLD = 0.20  # flag if change > 20%
WEEK_OVER_WEEK_CHANGE_THRESHOLD = 0.15

# --- Agent Behavior ---
MAX_RETRIES_PER_AGENT = 3
AGENT_TIMEOUT_SECONDS = 30
MAX_PIPELINE_LOOPS = 5  # loop prevention in LangGraph

# Severity levels (used throughout the codebase)
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# Routing thresholds: what severity triggers full pipeline
FULL_PIPELINE_SEVERITY = "high"    # high + critical → all 4 agents
PARTIAL_PIPELINE_SEVERITY = "medium"  # medium → signal + root cause only

# --- Evaluation ---
GOLDEN_TESTSET_SIZE = 100
RAGAS_EVAL_SAMPLE_SIZE = 20  # how many cases to run RAGAS on per evaluation cycle

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
