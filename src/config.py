"""Central place for paths and constants, so every module agrees on them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "day01_metrics.json"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
