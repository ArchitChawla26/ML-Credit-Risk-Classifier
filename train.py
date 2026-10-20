"""
ML Credit-Risk Classifier with Explainability
Author: Archit Chawla

Predicts next-month credit-card default on the UCI "Default of Credit Card
Clients" dataset (30,000 borrowers, Taiwan, 2005) and benchmarks a logistic
regression baseline against a gradient-boosting model, with SHAP-based
explainability (permutation importance as fallback).

Reproducibility: fixed seed (42) for the split, cross-validation and models.
Preprocessing is fit on the training partition only — no leakage.

Run:  python train.py
Outputs: results.json, charts/*.png
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "default of credit card clients.xls"
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)

TARGET = "default payment next month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
FEATURES = NUMERIC + CATEGORICAL

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit (NT$)",
    "AGE": "Age",
    "PAY_0": "Repayment status, Sep 2005 (most recent)",
    "PAY_2": "Repayment status, Aug 2005",
    "PAY_3": "Repayment status, Jul 2005",
    "PAY_4": "Repayment status, Jun 2005",
    "PAY_5": "Repayment status, May 2005",
    "PAY_6": "Repayment status, Apr 2005 (oldest)",
    "BILL_AMT1": "Bill amount, Sep 2005",
    "BILL_AMT6": "Bill amount, Apr 2005",
    "PAY_AMT1": "Payment amount, Sep 2005",
    "PAY_AMT2": "Payment amount, Aug 2005",
}


# ---------------------------------------------------------------- data ----
def load_data() -> pd.DataFrame:
    """Load the raw UCI .xls and apply documented, deterministic cleaning.

    The .xls carries its real header in the second row. Two categorical
    fields contain undocumented codes in the raw file (EDUCATION: 0, 5, 6;
    MARRIAGE: 0); these are collapsed into each field's 'others' category.
    This is a fixed mapping, not learned from the data, so applying it
    before the split cannot leak information.
    """
    df = pd.read_excel(DATA_FILE, header=1)
    df = df.rename(columns={TARGET: "default"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df[FEATURES + ["default"]]


# -------------------------------------------------------------- metrics ----
def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() > 0:
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate(y_true, y_prob, threshold) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece_deciles": round(expected_calibration_error(y_true, y_prob), 4),
    }


def youden_threshold(y_true, y_prob) -> float:
    """Operating threshold picked on TRAINING predictions only (Youden's J)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def cv_auc(make_model, X, y, seed=SEED) -> dict:
    """Manual 5-fold stratified CV — identical protocol for both models."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return {"mean": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)}


# --------------------------------------------------------------- models ----
def make_logistic():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=SEED)),
    ])


def make_booster(X_train, y_train):
    """Gradient boosting. XGBoost if installed, else sklearn's
    HistGradientBoostingClassifier — the choice is recorded in results."""
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            random_state=SEED, n_jobs=4,
        )
        return model, f"XGBoost {__import__('xgboost').__version__}"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=400, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=SEED,
        )
        return model, "sklearn HistGradientBoostingClassifier (XGBoost unavailable)"


# ---------------------------------------------------------------- main ----
def main():
    rng = np.random.default_rng(SEED)
    df = load_data()
    X, y = df[FEATURES], df["default"]
    base_rate = float(y.mean())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"rows={len(df)}  train={len(X_tr)}  test={len(X_te)}  "
          f"base_rate={base_rate:.4f}  test_base_rate={y_te.mean():.4f}")

    results = {
        "dataset": {
            "name": "UCI Default of Credit Card Clients",
            "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "rows": int(len(df)),
            "features": int(len(FEATURES)),
            "base_rate_default": round(base_rate, 4),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "test_base_rate": round(float(y_te.mean()), 4),
            "seed": SEED,
        },
        "imbalance_handling": (
            "No resampling. Logistic regression uses class_weight='balanced'; "
            "the booster uses scale_pos_weight = negatives/positives "
            "(HistGradientBoosting fallback: class_weight='balanced'). "
            "Operating threshold selected on training predictions via Youden's J "
            "and then frozen before touching the test set."
        ),
        "models": {},
    }

    # ---- logistic regression baseline
    lr = make_logistic()
    results["models"]["logistic_regression"] = {"cv_auc_train": cv_auc(make_logistic, X_tr, y_tr)}
    lr.fit(X_tr, y_tr)
    lr_thr = youden_threshold(y_tr, lr.predict_proba(X_tr)[:, 1])
    lr_prob = lr.predict_proba(X_te)[:, 1]
    results["models"]["logistic_regression"]["test"] = evaluate(y_te, lr_prob, lr_thr)
    results["models"]["logistic_regression"]["test_at_0.50"] = evaluate(y_te, lr_prob, 0.50)
    print("LR  test:", results["models"]["logistic_regression"]["test"])

    # ---- gradient boosting
    booster, booster_name = make_booster(X_tr, y_tr)
    results["models"]["gradient_boosting"] = {
        "implementation": booster_name,
        "cv_auc_train": cv_auc(lambda: make_booster(X_tr, y_tr)[0], X_tr, y_tr),
    }
    booster.fit(X_tr, y_tr)
    gb_thr = youden_threshold(y_tr, booster.predict_proba(X_tr)[:, 1])
    gb_prob = booster.predict_proba(X_te)[:, 1]
    results["models"]["gradient_boosting"]["test"] = evaluate(y_te, gb_prob, gb_thr)
    results["models"]["gradient_boosting"]["test_at_0.50"] = evaluate(y_te, gb_prob, 0.50)
    print("GB  test:", results["models"]["gradient_boosting"]["test"])

    # ---------------------------------------------------------- charts ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for prob, label in [(lr_prob, "Logistic regression"), (gb_prob, "Gradient boosting")]:
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax.plot(fpr, tpr, lw=1.8, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — held-out test set (n=6,000)")
    ax.legend(loc="lower right"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(CHARTS / "roc_curves.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for prob, label in [(lr_prob, "Logistic regression"), (gb_prob, "Gradient boosting")]:
        frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", lw=1.5, ms=4, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability"), ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration — predicted vs. observed (test set, deciles)")
    ax.legend(loc="upper left"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(CHARTS / "calibration.png", dpi=150), plt.close(fig)

    # -------------------------------------------------- explainability ----
    explain = {"method": None, "top_drivers": []}
    try:
        import shap

        explainer = shap.TreeExplainer(booster)
        sample = X_te.sample(n=min(2000, len(X_te)), random_state=SEED)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        explain["method"] = f"SHAP (TreeExplainer, shap {shap.__version__}), 2,000-row test sample"
        explain["top_drivers"] = [
            {"feature": FEATURES[i], "label": FEATURE_LABELS.get(FEATURES[i], FEATURES[i]),
             "mean_abs_shap": round(float(mean_abs[i]), 4)}
            for i in order[:10]
        ]
        shap.summary_plot(sv, sample, show=False, max_display=15)
        plt.title("SHAP summary — gradient boosting model")
        plt.tight_layout(), plt.savefig(CHARTS / "shap_summary.png", dpi=150,
                                         bbox_inches="tight"), plt.close()
        print("explainability: SHAP")
    except Exception as exc:  # shap missing or incompatible -> honest fallback
        from sklearn.inspection import permutation_importance

        r = permutation_importance(booster, X_te, y_te, n_repeats=10,
                                   random_state=SEED, scoring="roc_auc")
        order = np.argsort(r.importances_mean)[::-1]
        explain["method"] = ("Permutation importance (test set, AUC drop, 10 repeats) "
                             f"— SHAP unavailable: {type(exc).__name__}")
        explain["top_drivers"] = [
            {"feature": FEATURES[i], "label": FEATURE_LABELS.get(FEATURES[i], FEATURES[i]),
             "auc_drop": round(float(r.importances_mean[i]), 4)}
            for i in order[:10]
        ]
        fig, ax = plt.subplots(figsize=(7, 5))
        top = order[:12][::-1]
        ax.barh([FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in top],
                r.importances_mean[top], xerr=r.importances_std[top],
                color="#2c5f8a", alpha=0.85)
        ax.set_xlabel("Mean AUC drop when feature is shuffled")
        ax.set_title("Permutation importance — gradient boosting (test set)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(), fig.savefig(CHARTS / "permutation_importance.png", dpi=150)
        plt.close(fig)
        print("explainability: permutation importance (fallback)")

    results["explainability"] = explain
    (BASE / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(explain, indent=2))
    print("wrote results.json and charts/")


if __name__ == "__main__":
    main()
