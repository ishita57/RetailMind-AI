"""Used by the modelling days: score a model the same way each time."""
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def get_metrics_from_proba(y_test, proba, threshold=0.5) -> dict:
    """Score predicted churn probabilities (used for the Keras ANN)."""
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
    }


def get_metrics(model, X_test, y_test) -> dict:
    """Score a scikit-learn model (anything with predict and predict_proba)."""
    proba = model.predict_proba(X_test)[:, 1]
    return get_metrics_from_proba(y_test, proba)
