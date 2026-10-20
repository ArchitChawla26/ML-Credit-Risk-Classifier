# Model Card — Credit-Card Default Classifier

**Author:** Archit Chawla
**Date:** October 20, 2026
**Code:** `train.py` (seed 42 throughout) · **Full metrics:** `results.json`

## Purpose

Predict whether a credit-card borrower will default in the next month, benchmark a transparent baseline against a gradient-boosting model, and explain the winning model's predictions well enough that a credit officer — not just a data scientist — can interrogate them. This is a methods demonstrator built on public research data, not a production lending model.

## Dataset and provenance

- **Source:** UCI Machine Learning Repository, dataset 350, "Default of Credit Card Clients" — downloaded October 20, 2026 from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (the `.xls` mirrored in `data/`).
- **Origin:** Credit-card clients of a Taiwanese bank, 2005 (Yeh & Lien, 2009). Behavior window April–September 2005; the target is default in the following month.
- **Size:** 30,000 borrowers, 23 features, **base default rate 22.12%** (6,636 defaults).
- **Features:** credit limit and demographics (age, sex, education, marital status), six months of repayment-status codes (−1 = paid duly, 1–8 = months of delay), bill amounts, and payment amounts. Amounts are in New Taiwan dollars.
- **Cleaning (deterministic, documented in `train.py`):** the raw file's undocumented education codes (0, 5, 6) and marriage code (0) are collapsed into each field's "others" category. No rows dropped, no imputation needed.

## Method

- **Split:** stratified 80/20 — 24,000 train / 6,000 test, seed 42. Test base rate 22.12%, matching the full sample.
- **No leakage:** all preprocessing (scaling, one-hot encoding) is fit on the training partition only, inside a scikit-learn pipeline. The operating threshold is selected on *training* predictions (Youden's J) and frozen before the test set is scored.
- **Models:** (1) logistic regression baseline — one-hot categoricals, standardized numerics; (2) XGBoost 2.0.3 gradient boosting (400 trees, depth 4, learning rate 0.05).
- **Class imbalance (78/22):** no resampling and no synthetic data. Logistic regression uses `class_weight='balanced'`; XGBoost uses `scale_pos_weight = negatives/positives`. Cross-validated AUC (5-fold, stratified, on train) is reported alongside held-out test metrics so the imbalance handling can't hide behind a lucky split.
- **Validation:** ROC/AUC and the KS statistic (the standard separation measure in credit scoring) on the test set, precision/recall/F1 at the frozen operating threshold, full confusion matrices, and calibration assessed by Brier score, decile expected calibration error, and a calibration plot.
- **Explainability:** SHAP (TreeExplainer, shap 0.54.1) on a 2,000-row test sample — beeswarm summary plus mean |SHAP| ranking. `train.py` falls back to permutation importance if SHAP is unavailable and labels the output accordingly.

## Results (held-out test set, n = 6,000)

| Metric | Logistic regression | XGBoost |
|---|---|---|
| CV AUC (train, 5-fold) | 0.728 ± 0.011 | 0.781 ± 0.008 |
| **Test AUC** | 0.710 | **0.775** |
| **Test KS** | 0.360 | **0.422** |
| Operating threshold (from train) | 0.572 | 0.438 |
| Precision @ threshold | 0.512 | 0.414 |
| Recall @ threshold | 0.482 | 0.679 |
| F1 @ threshold | 0.497 | 0.515 |
| Confusion matrix (TN/FP/FN/TP) | 4062 / 611 / 687 / 640 | 3399 / 1274 / 426 / 901 |
| Brier score | 0.209 | 0.178 |
| Calibration error (ECE, deciles) | 0.234 | 0.190 |

The booster is the better *ranker* by a clear margin (+0.065 AUC, +0.06 KS) and, at its operating threshold, catches 68% of defaults versus the baseline's 48% — at the cost of more false alarms (precision 0.41 vs. 0.51). Which trade-off is "better" is a business decision about the cost of a missed default versus a wrongly flagged good customer; the model's job is to make that trade-off explicit, which the table above does.

**Calibration caveat, stated plainly:** both models' raw probabilities run *hot* — they overstate default probabilities on average (hence the high ECE), because class weighting deliberately shifts scores upward to fight the imbalance. The ranking is trustworthy; the raw probabilities are not, and would need recalibration (Platt scaling or isotonic regression on a validation set) before being used for pricing, provisioning, or cutoff decisions. The calibration plot in `charts/` shows this directly.

## Explainability findings

Mean |SHAP| ranking for the XGBoost model (full top-10 in `results.json`):

1. **PAY_0 — repayment status in the most recent month (Sep 2005): 0.539.** More than twice the weight of any other feature. A borrower already delaying in the latest month is, by a wide margin, the most likely to default next month; the beeswarm shows long delays pushing risk sharply up and on-time payment pushing it down.
2. **LIMIT_BAL — credit limit: 0.238.** Higher limits point to *lower* predicted risk — the limit is partly the bank's own prior judgment of the borrower showing up in the data — while low limits flag risk.
3. **BILL_AMT1 — most recent bill amount: 0.168.** A larger current balance owed raises predicted risk.
4. **PAY_AMT2 / PAY_AMT1 — recent payment amounts: 0.122 / 0.111.** Larger recent payments pull predicted risk down: borrowers who are actually paying down balances behave differently from borrowers who are merely not yet late.

The pattern across the top ten is consistent: **recent behavior dominates** — five of the top ten are the last two months' repayment and payment variables, and no demographic feature (sex, education, marital status, age) makes the top ten at all.

## Fairness and limitations

- **Not a US lending model.** The data is a single 2005 Taiwanese card portfolio, pre-financial-crisis, in NT dollars, under a different credit culture and regulatory regime. Nothing here should be transported to US underwriting.
- **Demographics are inputs.** Sex, education, and marital status are in the feature set because the dataset ships them; SHAP shows they carry little weight in this model, but low importance is not proof of fairness — correlated behavioral variables can act as proxies, and a real deployment would require formal disparate-impact testing and, in the US, ECOA/Reg B adverse-action reason codes. SHAP values are a starting point for those explanations, not a substitute for the legal process.
- **SHAP explains the model, not the world.** It attributes the model's output to features; it does not establish that changing a feature would change the borrower's outcome.
- **Data vintage and scope.** Twenty-year-old data, one lender-type portfolio, no income, employment, or bureau variables. AUC of 0.775 is respectable for this benchmark dataset but reflects those missing variables as much as model quality.
- **Reproducibility:** every number in this card regenerates from `train.py` with the pinned `requirements.txt`; a second clean run reproduced `results.json` exactly.
