# Who Defaults Next Month? A Credit-Risk Classifier You Can Interrogate

**Archit Chawla** · October 20, 2026
*Data: UCI "Default of Credit Card Clients" (30,000 borrowers, Taiwan, 2005). Code, model card, and full metrics: github.com/ArchitChawla26/ML-Credit-Risk-Classifier*

## The question

A lender's oldest problem: of the borrowers on the books today, which ones won't pay next month? I built a small, fully checkable version of the machine that answers it — two competing models, an honest scoreboard, and an explanation layer that says *why* the machine flags the people it flags.

## Approach

The dataset is real loan-level data: 30,000 Taiwanese credit-card clients, six months of repayment behavior (April–September 2005), and whether each defaulted the following month. 22.1% did — imbalanced enough that a lazy model can look accurate by predicting "no default" for everyone, so the evaluation leans on ranking metrics (AUC, and the KS statistic credit scorers actually use) instead of raw accuracy.

I benchmarked a logistic regression — the transparent incumbent in credit scoring — against XGBoost gradient boosting, under rules designed to keep the contest honest: a stratified 80/20 split with a fixed seed, preprocessing fit on training data only, the decision threshold chosen on training predictions and frozen before the test set was touched, and class imbalance handled with class weights rather than resampling tricks.

## Findings

The booster wins the ranking contest clearly: **test AUC 0.775 vs. 0.710, KS 0.42 vs. 0.36**. At its operating threshold it catches 68% of the defaults the baseline's 48% — though it flags more good borrowers to do it (precision 0.41 vs. 0.51). Neither number is "the answer"; together they price the trade-off a credit committee actually faces between a missed default and a wrongly declined customer.

The honest blemish: both models' probabilities run hot. Because class weighting pushes scores up to fight the imbalance, the raw outputs overstate default risk on average (calibration error ≈ 0.19–0.23 by decile). As rankers, both models are sound; as probability estimates, they would need recalibration before anyone priced a loan off them. I would rather publish that caveat than a prettier chart.

## What the explainability adds

A black box that says "0.78, decline" is unusable in a regulated lending decision — someone has to give the borrower a reason. SHAP explanations turn the booster into something a credit officer can argue with:

- **The most recent month's repayment status dominates everything** — more than twice the influence of the next feature. A borrower already delaying in September is the single clearest predictor of an October default, and the effect is monotone: the longer the delay, the higher the flagged risk.
- **Credit limit cuts the other way.** Higher limits predict *lower* risk — the limit is the bank's own earlier judgment embedded in the data — while low-limit borrowers are flagged more often.
- **Money actually moving matters.** Larger recent payments pull predicted risk down; a larger current bill pushes it up. Recent behavior, not demographics, fills the top ten: sex, education, marital status, and age don't appear in it at all.

That ordering is commercially sensible — it is close to what an experienced underwriter would say — and that is precisely the point of the exercise: the explanation layer is what lets you check the model against domain judgment instead of trusting it blind.

## Limits

This is 2005 Taiwan card data, one portfolio, no income or bureau variables, and none of it is a US lending model; demographics are model inputs and would need formal fairness testing and adverse-action reason codes in any real deployment. The full caveats, including the calibration issue and feature-by-feature SHAP values, are in the model card.

## Why I built it

This project extends my earlier SVM research on banking complaints and loan approval from comparing classifiers on paper to the full credit workflow — benchmark, threshold policy, calibration, and explanations — in code anyone can rerun and check.
