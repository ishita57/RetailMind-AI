"""Step 1 of the pipeline: INGEST. Load the CSV and clean the known issues."""
import pandas as pd

from src.config import DATA_PATH, ID_COLUMN, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the Telco churn CSV and fix its quirks.

    Known quirk: `TotalCharges` is stored as text and has blank values
    (brand-new customers with tenure = 0). We convert it to numbers; blanks
    become NaN, which the pipeline imputes later.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def inspect(df: pd.DataFrame) -> None:
    """Print schema, target balance and missing values (task 2)."""
    print("Shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nTarget balance:\n", df[TARGET].value_counts())
    print("\nTarget balance (%):\n", (df[TARGET].value_counts(normalize=True) * 100).round(1))
    missing = df.isna().sum()
    print("\nMissing values:\n", missing[missing > 0])


def split_features_target(df: pd.DataFrame):
    """Return X (features) and y (1 = churned, 0 = stayed). Drops the ID column."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, ID_COLUMN])
    return X, y


if __name__ == "__main__":
    inspect(load_data())
