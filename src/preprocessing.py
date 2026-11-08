"""Day 1's preprocessing as a reusable piece, so every model trains on identical features."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Numbers: fill missing with median, then scale. Text columns: one-hot encode."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    text_cols = X.select_dtypes(exclude="number").columns.tolist()

    numbers = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numbers, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), text_cols),
    ])
