"""Day 4: domain features built from the raw customer columns.

Works on the raw columns, so it can sit at the start of a pipeline and the saved
model can still be given a plain customer record.
"""
import pandas as pd

ADD_ON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # tenure bucket: new customers behave very differently from long-term ones
    X["tenure_group"] = pd.cut(X["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

    # charge ratios
    X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)
    X["charge_ratio"] = X["MonthlyCharges"] / (X["avg_monthly_spend"] + 1)

    # contract / service flags (1 = yes, 0 = no)
    X["num_add_ons"] = (X[ADD_ON_SERVICES] == "Yes").sum(axis=1)
    X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
    X["has_internet"] = (X["InternetService"] != "No").astype(int)
    X["auto_pay"] = X["PaymentMethod"].str.contains("automatic").astype(int)
    X["has_support"] = ((X["OnlineSecurity"] == "Yes") | (X["TechSupport"] == "Yes")).astype(int)
    return X
