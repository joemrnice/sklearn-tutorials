# Module 8 — Ensembles (Random Forest, Gradient Boosting, XGBoost via sklearn API)

## Real-world motivation
On real tabular problems (credit risk, churn, resource allocation), ensembles often provide a strong accuracy-to-effort ratio. Your job is to use them responsibly: proper CV, calibration, interpretability, and monitoring.

## Step-by-step code
```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)

hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=500,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in {"rf": rf, "hgb": hgb}.items():
    p = Pipeline([("preprocess", preprocess), ("model", mdl)])
    out = cross_validate(p, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "ap": "average_precision"}, n_jobs=-1)
    print(name, out["test_roc_auc"].mean(), out["test_ap"].mean())