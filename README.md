# credit-risk-lab

A credit-risk modeling study on public data, in Python and R. It compares logit, GLMs, a decision
tree, a random forest and XGBoost for predicting credit-card default, segments customers with
k-means, forecasts the U.S. credit-card delinquency rate against naive baselines, and monitors
drift with PSI and an alert threshold that triggers a retrain. CI fails if the champion model
falls below a minimum AUC or if R and Python disagree.

## What it does

| Part | Code | Method |
|---|---|---|
| PD models | `src/credit_risk/models.py` | L2 logit, statsmodels binomial GLM (logit and probit links), decision tree, random forest, XGBoost, in scikit-learn pipelines |
| Validation | `src/credit_risk/evaluate.py`, `metrics.py` | Stratified 70/30 holdout, nested 5-fold CV for selection, AUC with bootstrap CI, Gini, KS, Brier, ECE, calibration slope |
| R benchmark | `r/glm_benchmark.R`, `src/credit_risk/compare_r.py` | Base-R `glm()` on the same split, compared on coefficients, predictions and metrics |
| Segmentation | `src/credit_risk/segments.py` | K-means on behavioural features, k chosen by silhouette, default rate profiled on the holdout |
| Forecast | `src/credit_risk/forecast.py` | Rolling-origin backtest (40 origins, h=1 and h=4), ARIMA(1,1,0) and damped ETS against naive and seasonal naive |
| Monitoring | `src/credit_risk/monitor.py` | PSI on features and scores, green / amber (0.10) / red (0.25), red queues a retrain |
| Explainability | `src/credit_risk/explain.py` | Permutation importance, XGBoost TreeSHAP, per-applicant median-substitution reasons |
| Demo | `src/credit_risk/score.py` | CLI that returns a PD, a band and the fields raising risk most |
| Gate | `src/credit_risk/gate.py` | Champion holdout AUC at least 0.75, beats the PAY_0-only baseline, R/Python parity |

The champion is the highest CV AUC on the training set. The holdout is not used to choose.

## Data

Real public data only, details in [`data/SOURCES.md`](data/SOURCES.md).

- Default of Credit Card Clients (Yeh, UCI, CC BY 4.0): 30,000 Taiwanese card holders, 22.1% default rate.
- Federal Reserve delinquency rate on credit-card loans, all commercial banks, seasonally adjusted,
  quarterly, 1991:1 to 2026:2 (public domain per the Board's notice).

The drift feedback experiment applies a **simulated** corruption to the real holdout (amounts in
thousands, PAY_0 shifted by 2). It is labelled `simulated` in its output.

## How to run

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src python -m credit_risk.evaluate     # full study, several minutes
Rscript r/glm_benchmark.R                         # base R only
PYTHONPATH=src python -m credit_risk.compare_r
PYTHONPATH=src python -m credit_risk.gate
PYTHONPATH=src python -m credit_risk.score --example
```

`.github/workflows/ci.yml` runs all of this. Running it with `workflow_dispatch` also commits
`results/` and `models/` back to main.

## Results

Measured on GitHub Actions (`ubuntu-latest`, Python 3.11.18, scikit-learn 1.7.2, xgboost 3.1.2, R 4.4), run
[42539358827](https://github.com/ryannaidji/credit-risk-lab/actions/runs/42539358827). Data: 21,000 train and
9,000 holdout customers, stratified 70/30, seed 42, 22.1% default rate. Files: [`results/`](results/).

### Default models (`results/models.json`)

| Model | CV AUC (5-fold) | Holdout AUC [95% CI] | Gini | KS | Brier | ECE | Cal. slope |
|---|---|---|---|---|---|---|---|
| Base rate (baseline) | - | 0.5000 | 0.0000 | - | 0.1723 | - | - |
| PAY_0-only logit (baseline) | - | 0.6833 [0.671, 0.698] | 0.3667 | 0.3599 | 0.1489 | 0.0644 | 0.958 |
| GLM probit (statsmodels) | 0.7254 ± 0.0081 | 0.7153 [0.702, 0.730] | 0.4307 | 0.3689 | 0.1473 | 0.0556 | 0.978 |
| GLM logit (statsmodels) | 0.7260 ± 0.0079 | 0.7164 [0.704, 0.731] | 0.4328 | 0.3721 | 0.1461 | 0.0501 | 0.936 |
| Logit, L2 (scikit-learn) | 0.7260 ± 0.0080 | 0.7165 [0.704, 0.731] | 0.4329 | 0.3717 | 0.1461 | 0.0500 | 0.938 |
| Decision tree (tuned) | 0.7672 ± 0.0070 | 0.7546 [0.743, 0.766] | 0.5093 | 0.3938 | 0.1391 | 0.0193 | 0.901 |
| **Random forest (champion)** | 0.7853 ± 0.0081 | 0.7790 [0.767, 0.790] | 0.5579 | 0.4266 | 0.1353 | 0.0146 | 1.037 |
| XGBoost (tuned) | 0.7845 ± 0.0097 | 0.7811 [0.770, 0.792] | 0.5623 | 0.4285 | 0.1351 | 0.0141 | 0.989 |

The random forest won on CV AUC by 0.0009, well inside one standard deviation, and XGBoost then scored
slightly higher on the holdout. Their confidence intervals overlap almost entirely, so the data does not separate them.
The linear models are clearly weaker here (about 0.06 AUC behind). The gate threshold of 0.75 passes.

### R against Python (`results/r_vs_python.json`, 27 terms, 9,000 holdout predictions)

| Link | Max coefficient difference | Max prediction difference | AUC (Python / R) | KS (Python / R) | Brier (Python / R) |
|---|---|---|---|---|---|
| logit | 7.6e-12 | 1.1e-11 | 0.71642 / 0.71642 | 0.37208 / 0.37208 | 0.14608 / 0.14608 |
| probit | 1.7e-06 | 6.1e-06 | 0.71534 / 0.71534 | 0.36895 / 0.36895 | 0.14733 / 0.14733 |

### Forecast (`results/forecast.json`)

Credit-card delinquency rate, 142 quarters, expanding-window backtest over 40 origins, targets 2016:3 to 2026:2.

| Model | MAE h=1 | MAE vs naive h=1 | MAE h=4 | MAE vs naive h=4 |
|---|---|---|---|---|
| Naive (last value) | 0.101 | 1.00 | 0.332 | 1.00 |
| Seasonal naive | 0.332 | 3.28 | 0.332 | 1.00 |
| ARIMA(1,1,0) | 0.081 | 0.80 | 0.289 | 0.87 |
| Damped ETS | 0.080 | 0.79 | 0.298 | 0.90 |

Both models beat naive, modestly, with no significance test. Seasonal naive is a poor baseline for a seasonally
adjusted series, and at h=4 it equals naive by construction.

### Segmentation (`results/segments.json`)

Silhouette picked k=2 (0.381; k=3 to 8 scored 0.311 to 0.342), which is only moderate separation. Assignments were
identical across two seeds (ARI 1.0).

| Segment | Holdout share | Holdout default rate | Avg months late (of 6) | Avg utilization |
|---|---|---|---|---|
| Mostly current | 75.9% | 13.9% | 0.13 | 0.30 |
| Repeatedly late | 24.1% | 48.0% | 3.00 | 0.60 |

This is a coarse split, and the default gap is unsurprising because the features include payment delays.

### Monitoring (`results/monitoring.json`)

| Check | Score PSI | Max feature PSI | Alert | Action |
|---|---|---|---|---|
| Real: Apr-Jun 2005 vs Jul-Sep 2005 features | 0.044 | 0.173 (PAY_AMT1) | amber | investigate |
| Clean control: train vs holdout | 0.008 | 0.003 | green | none |
| **Simulated** unit bug on half the holdout | 6.29 | 10.84 (LIMIT_BAL) | red | retrain |

A 3-month-window logit scored AUC 0.717 on current features and 0.634 when given the same customers'
features from three months earlier.

Feedback loop (**simulated** corruption, champion retrained on 4,500 corrupted labelled rows, AUC on the other 4,500):
0.806 on clean data, 0.703 corrupted with no action, 0.793 after retraining.

### Explainability (`results/explain.json`)

Champion permutation importance (holdout AUC drop): PAY_0 0.067, PAY_2 0.015, LIMIT_BAL 0.014, BILL_AMT1 0.009,
PAY_AMT1 0.009. XGBoost TreeSHAP also ranks PAY_0 first (mean absolute 0.47 log-odds), then LIMIT_BAL (0.18) and PAY_2 (0.13).
The most recent repayment status dominates.

### Demo

`python -m credit_risk.score --example` on a made-up applicant, recorded by CI in
[`results/demo_output.txt`](results/demo_output.txt): PD 0.5747, band high, top reason PAY_0 (+0.168 against the
training median). There is no screenshot, as the demo is a CLI.

## Limitations

- **This is public data, not production credit experience.** It is one Taiwanese issuer in 2005.
  Nothing here was validated on a live portfolio or reviewed under model-risk rules such as SR 11-7.
- There is no application date, so a true out-of-time holdout is impossible. The holdout is stratified.
- The default label covers a single month (October 2005).
- "Real" drift compares two windows of the **same customers** three months apart, not a new cohort.
  PAY_5 and PAY_6 never take the value 1 in the source, a coding quirk that may affect PSI on those fields, though the measured PSI of the affected windows (PAY_2, PAY_3) is under 0.03.
- The feedback loop is a simulated bug, retrained on 4,500 rows of the holdout. The two halves have no time order, and AUC on that half alone (0.806 clean) sits above the full holdout (0.779), so compare only within the experiment.
- SEX, AGE and MARRIAGE are used as features to match the published dataset. A real lender could
  not use them this way, and no fairness analysis is done.
- Risk bands (10% and 30%) are illustrative, not cost-based.
- The committed model is a pickle: only load one you trust.
