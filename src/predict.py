"""The platform's first 'ML service': a thin predict() helper around the saved model.

Usage:
    from src.predict import predict
    predict({...customer fields...})      # -> {"churn_probability": 0.71, "churn_prediction": "Yes", ...}

Or run `python -m src.predict` for a demo with a sample customer.
"""
import joblib

from src.config import MODEL_PATH

_model = None  # loaded once, reused across calls


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(customer: dict, threshold: float = 0.5) -> dict:
    """Score one customer (a dict of raw fields, same columns as the CSV minus customerID/Churn)."""
    import pandas as pd

    proba = float(_get_model().predict_proba(pd.DataFrame([customer]))[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": "Yes" if proba >= threshold else "No",
        "risk_level": "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low",
    }


SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 140.7,
}

if __name__ == "__main__":
    print(predict(SAMPLE_CUSTOMER))
