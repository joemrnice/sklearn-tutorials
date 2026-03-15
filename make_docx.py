# Module 8 — Ensembles (Random Forest, Gradient Boosting, XGBoost via sklearn API)

## Real-world motivation
A regional bank wants to identify **customers at risk of missing a loan payment** so it can offer proactive restructuring options (short-term deferral, repayment plan changes). This is a high-stakes setting: false positives can cause unnecessary friction and stigma; false negatives can lead to default and significant financial harm.

Ensembles are often the most effective “default choice” for structured/tabular data because they capture non-linear interactions and handle mixed feature types well (especially with proper preprocessing).

**Key real-world constraints introduced in this module**
- Class imbalance (missed payments are relatively rare)
- Cost-sensitive decisions (collection resources are limited)
- Probability calibration (we need meaningful risk scores, not only rank order)

---

## Step-by-step code (Colab-ready)

### 8.1 Create a realistic “missed payment risk” dataset (simulated but messy)
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(12)
n = 12000

df = pd.DataFrame({
    "age": rng.integers(18, 85, n),
    "income": np.exp(rng.normal(10.4, 0.6, n)),  # heavy-tailed
    "balance": np.exp(rng.normal(9.4, 0.9, n)),
    "utilization": np.clip(rng.normal(0.35, 0.18, n), 0, 1.5),
    "tenure_months": rng.integers(1, 240, n),
    "late_payments_12m": rng.poisson(0.35, n),
    "region": rng.choice(["north", "south", "east", "west"], size=n, p=[0.24, 0.28, 0.22, 0.26]),
    "account_type": rng.choice(["checking", "credit", "loan", "mortgage"], size=n, p=[0.36, 0.30, 0.24, 0.10]),
})

# Missingness that correlates with risk (realistic: incomplete income verification)
miss_income = rng.random(n) < (0.03 + 0.10 * (df["utilization"] > 0.9))
df.loc[miss_income, "income"] = np.nan

# Outliers
outlier_idx = rng.choice(n, size=60, replace=False)
df.loc[outlier_idx, "balance"] *= 15

# Target generation with noise + nonlinearity
logit = (
    -3.2
    + 0.9 * (df["utilization"] > 0.85).astype(int)
    + 0.35 * df["late_payments_12m"]
    + 0.20 * (df["account_type"] == "credit").astype(int)
    + 0.15 * (df["region"] == "south").astype(int)
    + 0.00000008 * df["balance"].fillna(df["balance"].median())
    - 0.00000005 * df["income"].fillna(df["income"].median())
    + 0.12 * (df["tenure_months"] < 12).astype(int)
)
p = 1 / (1 + np.exp(-logit))
df["miss_next_payment"] = (rng.random(n) < p).astype(int)

df["miss_next_payment"].mean()