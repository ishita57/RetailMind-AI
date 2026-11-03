"""Run end to end: ingest -> split -> train -> evaluate -> persist.

Usage (from the repo root):  python -m src.train
"""
import json

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_loader import inspect, load_data, split_features_target
from src.pipeline import build_pipeline


def main():
    # 1. Ingest
    df = load_data()
    inspect(df)
    X, y = split_features_target(df)

    # 2. Split (stratify keeps the churn ratio equal in train and test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Train
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # 4. Evaluate
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "LogisticRegression (baseline)",
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    print("\n=== Baseline metrics (test set) ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")

    # 5. Persist
    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved   -> {MODEL_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
