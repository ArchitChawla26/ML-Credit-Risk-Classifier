# Resume entry — ML Credit-Risk Classifier with Explainability

## Full entry (Projects section)

ML Credit-Risk Classifier with Explainability | Oct 2026

- Built to demonstrate applied machine learning for credit roles: trained and benchmarked logistic regression against XGBoost gradient boosting to predict next-month default on 30,000 real borrowers (UCI credit-card dataset, 22% base default rate), with a leakage-free pipeline — train-only preprocessing, stratified split, decision threshold set on training data alone. Boosting won on ranking power: test AUC 0.775 vs. 0.710, KS 0.42 vs. 0.36.
- Added the explainability layer a black box lacks: SHAP analysis showing the most recent month's repayment status drives default risk more than twice as much as any other factor, ahead of credit limit and current bill size, while demographic features fall outside the top ten — with calibration checks and a full model card documenting limits. github.com/ArchitChawla26/ML-Credit-Risk-Classifier

## One-line version (summary sections / LinkedIn)

Credit-risk ML: benchmarked logistic regression vs. XGBoost on 30,000 real borrowers (AUC 0.775) with SHAP explainability and a published model card — github.com/ArchitChawla26/ML-Credit-Risk-Classifier

## Skills line addition

Machine Learning (scikit-learn, XGBoost, SHAP) · Credit Risk Modeling · Python (pandas, model evaluation & calibration)

## Placement notes

- Pairs with the research paper: the SVM work is the theory, this is the runnable artefact — if both are listed, put this in Projects and keep the paper in Research.
- Lead with the benchmark numbers (AUC/KS) in interviews; the SHAP finding (most recent repayment status dominates) is the 30-second story.
