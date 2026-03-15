# Scikit-learn Mastery: From Zero to Production-Ready Machine Learning in Every Life Domain
**Created by Alex Morgan – Scikit-learn Mastery Course**  
**Estimated completion time:** 40–60 hours (plus capstones)

> This course is engineered to take you from “basic Python” to building **production-ready**, **ethical**, and **domain-aware** machine learning systems with scikit-learn. Every module begins with a real-life problem and ends with deployable patterns.

---

## Table of Contents

- [Course Conventions](#course-conventions)
- [Module 1 — Why Scikit-learn? ML in Daily Life + Setup](#module-1--why-scikit-learn-ml-in-daily-life--setup)
- [Module 2 — First End-to-End Project: No-Show Prediction](#module-2--first-end-to-end-project-no-show-prediction)
- [Module 3 — Data Preprocessing Mastery](#module-3--data-preprocessing-mastery)
- [Module 4 — Core Supervised Learning + Metrics](#module-4--core-supervised-learning--metrics)
- [Module 5 — Core Unsupervised Learning](#module-5--core-unsupervised-learning)
- [Module 6 — Pipelines & ColumnTransformer](#module-6--pipelines--columntransformer)
- [Module 7 — Cross-Validation That Matches Reality](#module-7--cross-validation-that-matches-reality)

---

## Course Conventions

### Tooling assumptions
- Python 3.10+
- scikit-learn ≥ 1.5
- pandas, numpy
- matplotlib, seaborn (plotly optional)
- shap (for explainability)
- joblib (serialization)
- scikit-optimize (Bayesian search)
- imbalanced-learn (imbalance handling)
- xgboost (optional; via sklearn API)
- skl2onnx + onnxruntime (optional deployment)

### “Production-grade” in this course
A model is “production-grade” when:
1. **Preprocessing is inside a Pipeline** (leakage-resistant).
2. **Evaluation strategy matches deployment** (e.g., GroupKFold, TimeSeriesSplit).
3. **Metrics match decisions** (thresholding and costs considered).
4. **Artifacts are versioned** (model + schema + package versions).
5. **Responsible ML checks exist** (fairness + safety + monitoring plan).

### Dataset policy
We use a mix of:
- **Public datasets** (where feasible)
- **Realistically simulated datasets** (where privacy blocks public release, e.g., healthcare)
Simulated does **not** mean “toy”—we include missingness, imbalance, outliers, and drift patterns.

---

# Module 1 — Why Scikit-learn? ML in Daily Life + Setup

## Real-world motivation
Scikit-learn powers a large fraction of practical machine learning in business, government, education, and science because it makes **end-to-end modeling** reliable: clean preprocessing, honest evaluation, and deployable pipelines.

In the real world, ML is not a Kaggle contest. It is a system of decisions: who gets outreach, who gets flagged for review, which farms get intervention resources, which patients get follow-up.

## Step-by-step code (Colab-ready)

```python
import sys
import sklearn
import numpy as np
import pandas as pd

print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
```

### Install (local)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -U numpy pandas scikit-learn matplotlib seaborn plotly shap joblib scikit-optimize imbalanced-learn
pip install -U xgboost skl2onnx onnxruntime fastapi uvicorn
```

## Visualizations (standard styling)
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
```

## Why This Matters in Real Life
If your environment is inconsistent, your results cannot be trusted. Reproducibility is not “nice to have”: it is operational safety.

## Common pitfalls & how professionals avoid them
- **Pitfall:** “Works on my machine.”  
  **Avoid:** pinned dependencies, documented build steps.
- **Pitfall:** Randomness causes different results each run.  
  **Avoid:** set `random_state` consistently and record it.
- **Pitfall:** Notebook-only code is hard to review and deploy.  
  **Avoid:** move reusable code into `src/` modules.

## Exercises (3–5)
1. Create a local environment and print package versions.
2. Save versions to `artifacts/versions.txt`.
3. Confirm that a train/test split is identical when `random_state` is fixed.

## Mini-quiz + solution
**Q:** Why does version pinning matter for ML?  
**A:** Small library changes can alter preprocessing defaults, numerical stability, or model behavior—breaking reproducibility and deployments.

## Next-Level Tip
Use an experiment log (even a CSV) from day one: dataset hash, git commit SHA, model params, metrics.

---

# Module 2 — First End-to-End Project: No-Show Prediction

## Real-world motivation
A clinic wants to reduce missed appointments (“no-shows”). If the clinic can predict who is at risk, it can send targeted reminders or offer transport support. Done well, this improves health outcomes and reduces waste. Done poorly, it can unfairly burden certain populations.

**Task:** binary classification (`no_show`: 1/0).

## Step-by-step code (data creation + baseline pipeline)

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 6000

df = pd.DataFrame({
    "age": rng.integers(0, 90, size=n),
    "days_to_appointment": rng.integers(0, 60, size=n),
    "sms_received": rng.integers(0, 2, size=n),
    "prior_no_shows": rng.poisson(0.4, size=n),
    "clinic": rng.choice(["north", "south", "east", "west"], size=n, p=[0.3, 0.3, 0.2, 0.2]),
    "insurance_type": rng.choice(["private", "medicaid", "medicare", "self_pay"], size=n, p=[0.4, 0.25, 0.25, 0.10]),
})

# Realistic missingness
missing_mask = rng.random(n) < 0.08
df.loc[missing_mask, "insurance_type"] = None

# Target with signal + noise
logit = (
    -2.2
    + 0.03 * df["days_to_appointment"]
    + 0.55 * df["prior_no_shows"]
    - 0.35 * df["sms_received"]
    + 0.02 * (df["age"] < 8).astype(int)
    + 0.15 * (df["insurance_type"].fillna("unknown") == "self_pay").astype(int)
)
p = 1 / (1 + np.exp(-logit))
df["no_show"] = (rng.random(n) < p).astype(int)

df.head()
```

### Build a leakage-resistant pipeline
```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay

X = df.drop(columns=["no_show"])
y = df["no_show"]

num_cols = ["age", "days_to_appointment", "sms_received", "prior_no_shows"]
cat_cols = ["clinic", "insurance_type"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred, digits=3))
RocCurveDisplay.from_predictions(y_test, proba);
```

## Visualizations (what to look for)
- ROC curve: ranking quality.
- Classification report: precision/recall. In outreach systems, recall is often prioritized.

## Why This Matters in Real Life
This project mirrors real deployments: mixed data types, missing values, class imbalance, and operational actions tied to predicted risk.

## Common pitfalls & professional fixes
- **Leakage trap:** features only known after the appointment (e.g., “checked_in_time”).  
  **Fix:** build an “availability time” checklist for each feature.
- **Wrong metric:** accuracy is misleading when no-shows are rare.  
  **Fix:** prioritize PR-AUC, recall, or recall@k.
- **No threshold strategy:** probabilities are not decisions.  
  **Fix:** set thresholds by budget and cost.

## Exercises
1. Add features: `weekday`, `hour_of_day` (simulate from appointment timestamp).
2. Compare `class_weight="balanced"` vs none.
3. Choose a threshold maximizing recall with precision ≥ 0.35.

## Mini-quiz + solution
**Q:** Why do we keep preprocessing inside the pipeline?  
**A:** To prevent leakage and ensure the same preprocessing runs in CV and production inference.

## Next-Level Tip
Create a “decision memo”: the action, costs, false positive consequences, and false negative consequences. Pick metrics accordingly.

---

# Module 3 — Data Preprocessing Mastery

## Real-world motivation
Most ML work is data work: missingness, outliers, skew, inconsistent categories, and hidden leakage. If preprocessing is fragile, the model is fragile.

## Step-by-step code (preprocessing patterns)

### 1) Missing values (numeric and categorical)
```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
```

### 2) Encoding categories safely
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore")
```

### 3) Scaling (when it matters)
```python
from sklearn.preprocessing import StandardScaler, RobustScaler

standard = StandardScaler()
robust = RobustScaler()
```

### 4) Put it together with ColumnTransformer
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", RobustScaler())])

cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])
```

## Visualizations (diagnose before you “fix”)
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["days_to_appointment"], kde=True)
plt.title("Days-to-appointment distribution");
```

## Why This Matters in Real Life
Poor preprocessing causes silent failures: new categories crash your model, missing values create NaNs, scaling errors break distance-based models.

## Common pitfalls & professional fixes
- **Fitting imputers on full dataset** (leakage).  
  **Fix:** always fit inside pipeline + CV.
- **Over-encoding** high-cardinality categories.  
  **Fix:** consider hashing or careful grouping; measure sparsity and latency.
- **Scaling trees unnecessarily.**  
  **Fix:** scale for linear/KNN/SVM; trees usually do not require it.

## Exercises
1. Add missingness indicators (`SimpleImputer(add_indicator=True)`).
2. Compare RobustScaler vs StandardScaler for heavy-tailed features.
3. Create a preprocessing report: missingness rate, outlier counts, cardinalities.

## Mini-quiz + solution
**Q:** Why use `handle_unknown="ignore"`?  
**A:** Production data contains unseen categories; ignoring prevents runtime failures.

## Next-Level Tip
Treat “missingness” as a feature, not just a problem. Missingness often signals process issues (e.g., tests not ordered for healthy patients).

---

# Module 4 — Core Supervised Learning + Metrics

## Real-world motivation
Supervised learning supports decisions: credit limits, early interventions, risk scoring, targeted support. The difference between “good” and “harmful” is often evaluation and thresholding, not the model class.

## Step-by-step (compare models fairly)

```python
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    "logreg": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=25),
    "tree": DecisionTreeClassifier(max_depth=6, random_state=42),
}

scoring = {
    "f1": make_scorer(f1_score),
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", mdl)])
    out = cross_validate(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    print(name, {k: out[f"test_{k}"].mean() for k in scoring})
```

## Visualizations
Confusion matrix translates model results to operational discussion.

```python
from sklearn.metrics import ConfusionMatrixDisplay

pipe = Pipeline([("preprocess", preprocess),
                 ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
pipe.fit(X_train, y_train)
ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test);
```

## Why This Matters in Real Life
A model is only useful if the metric and threshold align to the intervention constraints and harm profile.

## Common pitfalls & professional fixes
- **Accuracy as main metric on imbalance.**  
  **Fix:** PR-AUC, recall@k, cost curves.
- **Comparing models with different preprocessing quality.**  
  **Fix:** consistent pipeline structure.

## Exercises
1. Add `average_precision` scoring and compare to ROC-AUC.
2. Show how different thresholds change the confusion matrix.
3. Build a cost table: cost(FP), cost(FN) and compute expected cost.

## Mini-quiz + solution
**Q:** When do you prefer MAE to RMSE in regression?  
**A:** When outliers should not dominate; MAE is more robust.

## Next-Level Tip
Make the intervention budget explicit: “We can call 200 patients/day.” Evaluate recall@200.

---

# Module 5 — Core Unsupervised Learning

## Real-world motivation
When labels do not exist, unsupervised learning helps you discover structure: customer segments, unusual devices, patient subtypes, and compressed representations.

## Step-by-step (segmentation)

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 3000

dfc = pd.DataFrame({
    "annual_spend": np.exp(rng.normal(8.5, 0.7, n)),
    "visits_per_month": rng.poisson(4, n) + rng.integers(0, 6, n),
    "returns_rate": np.clip(rng.normal(0.08, 0.05, n), 0, 0.6),
    "avg_basket_size": np.clip(rng.normal(3.2, 1.1, n), 1, 15)
})
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

cluster_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, n_init="auto", random_state=42))
])

dfc["segment"] = cluster_pipe.fit_predict(dfc)
dfc["segment"].value_counts()
```

## Visualizations (PCA projection)
```python
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

X_scaled = StandardScaler().fit_transform(dfc.drop(columns=["segment"]))
X2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plot_df = pd.DataFrame({"pc1": X2[:, 0], "pc2": X2[:, 1], "segment": dfc["segment"]})
sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="segment", palette="tab10")
plt.title("Segments (PCA projection)");
```

## Why This Matters in Real Life
Segmentation drives messaging, pricing, support tiers—but can also create exclusion. Always evaluate downstream effects.

## Common pitfalls & fixes
- **Not scaling before distance-based methods.**  
- **Treating clusters as ground truth.**  
- **Using t-SNE as a quantitative tool** (it is a visualization method).

## Exercises
1. Choose k with silhouette score.
2. Compare KMeans vs DBSCAN for outlier-heavy data.
3. Interpret segments using group means.

## Mini-quiz + solution
**Q:** Why does KMeans need scaling?  
**A:** Because Euclidean distance is sensitive to feature scale.

## Next-Level Tip
Test stability: cluster today vs next month. Unstable segments are hard to operationalize.

---

# Module 6 — Pipelines & ColumnTransformer

## Real-world motivation
Pipelines are the single most important deployability feature in scikit-learn. They eliminate many silent errors and make model training and inference consistent.

## Step-by-step (feature engineering + preprocessing + model)
```python
from sklearn.preprocessing import FunctionTransformer

def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "days_to_appointment" in X.columns and "prior_no_shows" in X.columns:
        X["risk_proxy"] = X["prior_no_shows"] / (X["days_to_appointment"].replace(0, np.nan))
    return X

feat = FunctionTransformer(add_ratio_features, feature_names_out="one-to-one")
```

```python
pipe = Pipeline([
    ("features", feat),
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
])
```

## Why This Matters in Real Life
Without pipelines, deployment is a manual recreation of transformations—one mismatch causes wrong predictions.

## Pitfalls & fixes
- **Not handling unknown categories** → production crashes.
- **Preprocessing done outside CV** → leakage.
- **Feature logic not unit tested** → silent bugs.

## Exercises
1. Write a quick unit test: ratio feature never inf or NaN after preprocessing.
2. Export pipeline with joblib (preview of Module 15).
3. Add a `predict_one(record: dict)` helper that validates schema.

## Mini-quiz + solution
**Q:** Why store feature engineering inside Pipeline?  
**A:** To guarantee identical transformations at training and inference.

## Next-Level Tip
Treat your pipeline like an API contract: define required/optional fields and expected types.

---

# Module 7 — Cross-Validation That Matches Reality

## Real-world motivation
The biggest ML lie is a single lucky train/test split. CV helps estimate stability, but only if you choose the right CV type.

## Step-by-step (CV strategies)
- **StratifiedKFold:** imbalanced classification.
- **GroupKFold:** avoid user/patient/store leakage.
- **TimeSeriesSplit:** time-ordered evaluation.

```python
from sklearn.model_selection import GroupKFold, cross_val_score

# Example: repeated entities, e.g. multiple rows per patient
# groups = df["patient_id"]
# scores = cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring="roc_auc")
# print(scores.mean(), scores.std())
```

## Why This Matters in Real Life
If you overestimate performance, you deploy a system that fails in production and damages trust.

## Pitfalls & fixes
- **Random CV for time-series** → future leakage.
- **Tuning using the test set** → inflated estimates.

## Exercises
1. Implement TimeSeriesSplit for a time-ordered dataset.
2. Compare KFold vs GroupKFold on a “multiple records per entity” dataset.
3. Report mean and std; decide if variance is operationally acceptable.

## Mini-quiz + solution
**Q:** What does high variance across folds imply?  
**A:** The model is unstable; performance depends heavily on which data it sees.

## Next-Level Tip
Use nested CV when comparing tuned models under tight governance (Module 13).

---

**End of Part 01. Continue with Part 02.**