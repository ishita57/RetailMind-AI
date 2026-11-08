# RetailMind AI

A retail AI platform built step by step, one module per day. Each day's work is a separate
notebook and a separate commit, so the git history shows how the product grew.

## Product vision

Help a business **spot customers who are about to leave** so it can act before they go, then grow
this into a full AI platform (better models, recommendations, forecasting, deployment).

## Project structure

```
RetailMind-AI/
├── data/        telco\_customer\_churn.csv (Kaggle: blastchar/telco-customer-churn)
├── notebooks/   one notebook per day (the work and the results)
├── src/         reusable code shared by the notebooks
│   ├── config.py, data\_loader.py, pipeline.py, train.py   (Day 1)
│   ├── preprocessing.py   Day 1 preprocessing, reusable for any model
│   ├── evaluate.py        one function to score any model the same way
│   ├── features.py        Day 4 engineered features
│   └── predict.py         predict() service, works with any saved model
├── models/      saved models: churn\_pipeline (Day 1), churn\_v2, churn\_v3, churn\_v4
├── reports/
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
```

Run the notebooks **in order** (Day 3 needs Day 2's folder, Day 4 needs Day 3's saved model):
`day01\_churn\_pipeline` → `day02\_model\_comparison` → `day03\_tuning\_bias\_variance` → `day04\_features\_ensembles`

Use the latest model:

```python
from src.predict import predict
predict({...one customer's fields...}, model\_path="models/churn\_v4.joblib")
```

\---

## Progress log

### Day 1 - End-to-end ML pipeline and ML-as-a-Service

* Loaded and inspected the Telco churn data: 7,043 customers, **26.5% churn** (imbalanced).
`TotalCharges` was text with 11 blanks (new customers), so it is converted to numbers and the blanks are filled in the pipeline.
* Pipeline: split, fill missing + scale numbers + one-hot encode text, Logistic Regression baseline.
* Saved the pipeline with `joblib` and wrapped it in `predict()`.

|Accuracy|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|
|0.8055|0.6572|0.5588|0.6040|**0.8419**|

Recall is the weak spot: about 44% of customers who really churn are missed.

### Day 2 - Decision Tree, Random Forest and SVM

All models use the same Day 1 preprocessing, the same train/test split and `random\_state=42`.
With default settings none of them beat the baseline on recall. Adding `class\_weight="balanced"` helped the SVM a lot,
but not the fully grown trees (they overfit, which Day 3 addresses).

|Model|Accuracy|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|-|
|Day 1 baseline (Logistic Regression)|0.8055|0.6572|0.5588|0.6040|0.8419|
|Decision Tree|0.7289|0.4896|0.5053|0.4974|0.6573|
|Random Forest|0.7779|0.6034|0.4759|0.5321|0.8162|
|SVM|0.7913|0.6418|0.4840|0.5518|0.7905|
|Decision Tree (balanced)|0.7282|0.4878|0.4813|0.4845|0.6496|
|Random Forest (balanced)|0.7821|0.6184|0.4679|0.5327|0.8192|
|**SVM (balanced)**|0.7438|0.5115|**0.7727**|0.6155|0.8211|

**Champion: SVM (balanced)**, saved as `models/churn\_v2.joblib`. It won on recall (0.56 to 0.77), which is the business priority.
The trade-off is lower precision and ROC-AUC than the baseline, meaning more false alarms.

### Day 3 - Tuning and bias-variance

`GridSearchCV` with 5-fold cross-validation for four models, scored on ROC-AUC (recall also tracked).

|Model|Best settings|Train ROC-AUC|Validation ROC-AUC|Gap|Validation recall|
|-|-|-|-|-|-|
|Logistic Regression|C=10|0.8497|0.8455|0.004|0.8027|
|Decision Tree|max\_depth=5, min\_samples\_leaf=30|0.8537|0.8255|0.028|0.7860|
|**Random Forest**|n\_estimators=200, max\_depth=8, min\_samples\_leaf=10|0.8890|**0.8489**|0.040|0.7813|
|SVM|C=1, gamma=0.01|0.8490|0.8412|0.008|0.8261|

* No model badly underfits or overfits; the Random Forest has a mild variance gap (0.04).
* The learning curve shows validation score rising only slightly with more data (0.844 to 0.849), so new **features** are a better next step than more data.
* **Champion: tuned Random Forest**, saved as `models/churn\_v3.joblib`. Hold-out test: ROC-AUC 0.8432, recall 0.7888.

### Day 4 - Feature engineering and ensembles

* New features (`src/features.py`): tenure group, average monthly spend, charge ratio, number of add-ons, and flags for month-to-month contract, internet, auto-pay and support.
* PCA reduced 56 features to 20 but lowered recall, so it is not used in the final model.
* The saved model is one pipeline: features, then preprocessing, then model.

|Model|Accuracy|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|-|
|Day 3 champion (tuned RF)|0.7559|0.5268|0.7888|0.6317|0.8432|
|**RF + features**|0.7594|0.5318|0.7834|0.6335|**0.8447**|
|RF + features + PCA|0.7530|0.5242|0.7540|0.6184|0.8373|
|XGBoost + features|0.7509|0.5204|0.7861|0.6262|0.8373|

**Final Phase-1 model:** Random Forest + engineered features, saved as `models/churn\_v4.joblib`.
The feature gain is small (ROC-AUC +0.0015), because the original columns already hold most of the signal.

### Phase 1 summary 

|Day|Best model|Recall|F1|ROC-AUC|
|-|-|-|-|-|
|1|Logistic Regression (baseline)|0.5588|0.6040|0.8419|
|2|SVM (balanced)|0.7727|0.6155|0.8211|
|3|Random Forest (tuned)|0.7888|0.6317|0.8432|
|4|Random Forest + features|0.7834|0.6335|0.8447|

Recall improved from 0.56 to about 0.78 (we now catch about 4 in 5 churners), at the cost of lower precision.

## Phase 2 - Deep learning

### Day 5 - Neural networks and ANN (Keras)

Built a small Keras ANN for churn (normalised inputs, two dense layers, sigmoid output) with early stopping on the
validation loss and class weights (so recall stays a priority). Warmed up on MNIST first.
Benchmarked on the same hold-out customers as Phase 1.

|Model|Accuracy|Precision|Recall|F1|ROC-AUC|
|-|-|-|-|-|-|
|Phase 1 champion (churn\_v4)|0.7594|0.5318|0.7834|0.6335|0.8447|
|Keras ANN|0.7211|0.4851|0.8262|0.6113|0.8366|

**Did depth help?** The classical champion is better, so depth did not help here.

**When to use which**

* **Classical ML (Random Forest, XGBoost):** small or medium tables like this one. Fast to train, easy to explain, often just as accurate.
* **Neural networks:** images, text, audio, or very large datasets with complex patterns, where they can learn features by themselves.

Saved: `models/churn\\\_ann.keras` and `models/churn\\\_ann\\\_preprocessor.joblib`.

### Day 6 - CNN image classification (catalog auto-tagging)

* Prototyped the CNN on Fashion-MNIST, then trained it on Fashion Product Images (Kaggle `paramaggarwal/fashion-product-images-small`):
the 10 biggest article types, up to 600 images each, 80x60 pixels, with flip/rotation/zoom augmentation.
* CNN: conv, pool, conv, pool, dropout, dense (`src/cnn.py`). Early stopping on validation loss.
* The image dataset is not stored in the repo because of its size.

|Dataset|Test accuracy|
|-|-|
|Fashion-MNIST (prototype)|0.8819|
|Fashion Product Images (catalog classifier)|0.8575|

**Most confused categories:** Casual shoes and Sports shoes,Tshirts and Tops, Casual shoes and Heels

Saved: `models/catalog\\\_cnn.keras` and `models/catalog\\\_classes.json`.

**Next:** transfer learning for production-level accuracy, then object detection (YOLO).



