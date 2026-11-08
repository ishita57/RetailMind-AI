"""RetailMind's ML service: load a saved model and predict churn for one customer.

By default it uses the Day 1 model. Pass model_path to use a newer one
(e.g. models/churn_v4.joblib).
"""
import joblib
import pandas as pd

from src.config import MODEL_PATH

_models = {}   # cache, so each model file is only loaded once


def _get_model(model_path):
    key = str(model_path)
    if key not in _models:
        _models[key] = joblib.load(model_path)
    return _models[key]


def predict(customer: dict, threshold: float = 0.5, model_path=MODEL_PATH) -> dict:
    model = _get_model(model_path)
    proba = float(model.predict_proba(pd.DataFrame([customer]))[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": "Yes" if proba >= threshold else "No",
        "risk_level": "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low",
    }
