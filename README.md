# RetailMind AI

An end-to-end retail AI platform, built **one module per day**. Every day adds a new
increment to this same codebase, and each day is committed separately so the git
history tells the story of the product.

## Product vision
Give a retail/telecom business an immediate, trustworthy way to **flag at-risk customers
before they leave**, then grow that into a full AI platform (better models, recommendations,
forecasting, deployment) over 35 days.

## Project structure
```
RetailMind-AI/
├── data/            # raw datasets (Telco Customer Churn, from Kaggle: blastchar/telco-customer-churn)
├── src/             # all reusable code
│   ├── config.py        # paths + constants
│   ├── data_loader.py   # ingest + cleaning + inspection
│   ├── pipeline.py      # preprocessing + model (one sklearn Pipeline)
│   ├── train.py         # ingest -> split -> train -> evaluate -> persist
│   └── predict.py       # thin predict() helper = the first "ML service"
├── models/          # saved models (joblib)
├── notebooks/       # exploration
├── reports/         # metrics for each day
└── requirements.txt
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train                # trains, evaluates, saves model + metrics
python -m src.predict              # scores a sample customer with the saved model
```

Use it from code:
```python
from src.predict import predict
predict({"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
         "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
         "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
         "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
         "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
         "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 140.7})
# {'churn_probability': 0.6928, 'churn_prediction': 'Yes', 'risk_level': 'Medium'}
```

---

## Daily progress log

### Day 1 – End-to-end ML pipeline & ML-as-a-Service
**What was built**
1. Repo structure + this README.
2. Loaded the CSV with pandas and inspected schema, target balance and missing values.
3. One pipeline: stratified 80/20 split → impute + scale numeric columns, one-hot encode categoricals → Logistic Regression baseline → evaluate.
4. Saved the whole fitted pipeline with `joblib` and wrapped it in a `predict()` helper.
5. Recorded baseline metrics (below).

**Data findings**
- 7,043 customers, 21 columns; target `Churn` is imbalanced: 73.5% No / 26.5% Yes.
- Only missing data: `TotalCharges` (11 blanks, stored as text, all new customers) → converted to numeric and median-imputed.

**Baseline metrics (Logistic Regression, 20% test set, `random_state=42`)**

| Metric | Value |
|---|---|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1 | 0.6040 |
| **ROC-AUC** | **0.8419** |

> This is the number every later day tries to beat. Recall (0.56) is the weak spot: the model misses
> ~44% of customers who actually churn, which is the thing the business cares about most.

**Next (Day 2):** replace the single baseline with Decision Tree, Random Forest and SVM, compare, pick a champion.
