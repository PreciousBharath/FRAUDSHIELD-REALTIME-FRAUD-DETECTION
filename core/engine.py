"""FraudShield engine: synthetic transaction generator, model training & evaluation, explainability, live stream."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CATEGORIES = ["grocery", "electronics", "travel", "fuel", "restaurant", "online_retail", "jewellery", "gaming", "pharmacy", "atm_withdrawal"]
CAT_RISK = {"grocery": 0.2, "electronics": 1.2, "travel": 1.0, "fuel": 0.3, "restaurant": 0.3, "online_retail": 1.0, "jewellery": 1.5, "gaming": 1.3, "pharmacy": 0.2, "atm_withdrawal": 1.1}
CITIES = {"Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Hyderabad": (17.38, 78.48),
          "Kolkata": (22.57, 88.36), "Madurai": (9.93, 78.12), "Pune": (18.52, 73.86), "Jaipur": (26.91, 75.79), "Kochi": (9.93, 76.27)}
FEATURES = ["amount", "hour", "distance_km", "txn_last_hour", "account_age_days", "foreign", "card_present", "new_device", "category_risk", "amount_vs_avg"]


def generate(n=20000, seed=7):
    rng = np.random.default_rng(seed)
    cat = rng.choice(CATEGORIES, n, p=[.2, .08, .06, .12, .14, .16, .03, .05, .1, .06])
    avg_spend = rng.lognormal(7.2, 0.6, n)
    amount = np.round(avg_spend * rng.lognormal(0, 0.7, n), 2)
    hour = rng.integers(0, 24, n)
    distance = np.round(rng.exponential(8, n), 1)
    velocity = rng.poisson(1.2, n)
    age = rng.integers(10, 3650, n)
    foreign = (rng.random(n) < 0.04).astype(int)
    present = (rng.random(n) < 0.6).astype(int)
    new_device = (rng.random(n) < 0.07).astype(int)
    crisk = np.array([CAT_RISK[c] for c in cat])
    ratio = amount / avg_spend
    signal = (1.8 * np.log1p(ratio) + 1.1 * ((hour < 5) | (hour > 23)) + 0.025 * np.minimum(distance, 200) + 0.45 * velocity
              + 1.8 * foreign - 1.2 * present + 1.6 * new_device + 0.9 * crisk - 0.0006 * age)
    z = -10.5 + 2.2 * signal + rng.normal(0, 0.4, n)
    fraud = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    city = rng.choice(list(CITIES), n)
    df = pd.DataFrame({"amount": amount, "hour": hour, "distance_km": distance, "txn_last_hour": velocity, "account_age_days": age,
                       "foreign": foreign, "card_present": present, "new_device": new_device, "category": cat, "category_risk": crisk,
                       "amount_vs_avg": np.round(ratio, 3), "city": city, "is_fraud": fraud})
    return df


def train(df):
    X, y = df[FEATURES], df["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=3, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, p)
        pr, rc, _ = precision_recall_curve(yte, p)
        results[name] = {"model": m, "proba": p, "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                         "precision": precision_score(yte, pred, zero_division=0), "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                         "cm": confusion_matrix(yte, pred), "roc": (fpr, tpr), "pr": (pr, rc)}
    iso = IsolationForest(n_estimators=200, contamination=float(y.mean()), random_state=42).fit(Xtr[ytr == 0])
    gb = models["Gradient Boosting"]
    importance = pd.Series(gb.feature_importances_, index=FEATURES).sort_values(ascending=True)
    return {"results": results, "y_test": yte.values, "X_test": Xte, "iso": iso, "importance": importance, "means": Xtr.mean(), "best": "Gradient Boosting"}


def score(bundle, X, model_name=None):
    m = bundle["results"][model_name or bundle["best"]]["model"]
    p = m.predict_proba(X[FEATURES])[:, 1]
    anomaly = -bundle["iso"].score_samples(X[FEATURES])
    return p, anomaly


def explain(bundle, row: pd.DataFrame, model_name=None):
    """Local explanation: change in fraud probability when each feature is reset to the population average."""
    base, _ = score(bundle, row, model_name)
    contrib = {}
    for f in FEATURES:
        r = row.copy()
        r[f] = bundle["means"][f]
        p, _ = score(bundle, r, model_name)
        contrib[f] = float(base[0] - p[0])
    return float(base[0]), pd.Series(contrib).sort_values()


def stream_batch(rng, n=1):
    """Generate n fresh live transactions (with ~4% injected attacks for demo)."""
    df = generate(n, seed=int(rng.integers(0, 1e9)))
    attack = rng.random(n) < 0.04
    df.loc[attack, "amount"] *= rng.uniform(4, 12, attack.sum())
    df.loc[attack, "amount_vs_avg"] *= 8
    df.loc[attack, ["new_device", "foreign"]] = 1
    df.loc[attack, "card_present"] = 0
    df.loc[attack, "hour"] = rng.integers(0, 4, attack.sum())
    df["txn_id"] = [f"TX{rng.integers(10**7, 10**8)}" for _ in range(n)]
    df["merchant"] = [f"{c.replace('_', ' ').title()} #{rng.integers(100, 999)}" for c in df["category"]]
    df["lat"] = [CITIES[c][0] + rng.normal(0, .08) for c in df["city"]]
    df["lon"] = [CITIES[c][1] + rng.normal(0, .08) for c in df["city"]]
    return df


def decision(p, threshold):
    if p >= threshold:
        return "BLOCK", "red"
    if p >= threshold * 0.5:
        return "REVIEW", "amber"
    return "APPROVE", "green"
