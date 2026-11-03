"""Step 2 of the pipeline: PREPROCESS + MODEL, bundled into ONE sklearn Pipeline.

Keeping preprocessing and model together means the saved file can score raw
customer data directly, and there is no train/test leakage because scaling and
encoding are learned from the training split only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def build_pipeline(X) -> Pipeline:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # fixes blank TotalCharges
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")

    preprocess = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

    # Baseline classifier. Day 2 swaps this for stronger algorithms.
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    return Pipeline([("preprocess", preprocess), ("model", model)])
