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

**Full results are pending the first GitHub Actions run.** The model comparison, segmentation,
forecast backtest and scoring demo have not been executed end to end yet, so no numbers are quoted.

Partial smoke check, run once on the author's Windows laptop (R 4.2.2, statsmodels 0.14.0, 21,000
training rows, 9,000 holdout rows):

| Check | Result |
|---|---|
| R vs Python GLM, logit link, max coefficient difference | 7.6e-12 |
| R vs Python GLM, probit link, max coefficient difference | 1.7e-06 |
| GLM logit holdout (AUC / KS / Brier), both languages | 0.7164 / 0.3721 / 0.1461 |

No screenshot. The demo is a CLI, and its recorded output will be `results/demo_output.txt`.

## Limitations

- **This is public data, not production credit experience.** It is one Taiwanese issuer in 2005.
  Nothing here was validated on a live portfolio or reviewed under model-risk rules such as SR 11-7.
- There is no application date, so a true out-of-time holdout is impossible. The holdout is stratified.
- The default label covers a single month (October 2005).
- "Real" drift compares two windows of the **same customers** three months apart, not a new cohort.
  PAY_5 and PAY_6 never take the value 1 in the source, which inflates PSI on PAY fields.
- The feedback loop is a simulated bug, retrained on 4,500 rows of the holdout.
- SEX, AGE and MARRIAGE are used as features to match the published dataset. A real lender could
  not use them this way, and no fairness analysis is done.
- Risk bands (10% and 30%) are illustrative, not cost-based.
- The committed model is a pickle: only load one you trust.
