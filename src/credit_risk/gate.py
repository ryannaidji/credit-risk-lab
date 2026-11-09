"""CI gate. Thresholds were fixed before the first full run."""
import json
import sys

from .data import RESULTS

MIN_AUC = 0.75
MAX_COEF_DIFF = 1e-3
MAX_METRIC_DIFF = 1e-4


def check(models: dict, parity: dict) -> list[str]:
    failures = []
    champion = models["champion"]
    auc = models["models"][champion]["holdout"]["auc"]
    baseline = models["baselines"]["baseline_pay0_logit"]["holdout"]["auc"]
    if auc < MIN_AUC:
        failures.append(f"champion {champion} AUC {auc:.4f} < {MIN_AUC}")
    if auc <= baseline:
        failures.append(f"champion AUC {auc:.4f} does not beat PAY_0 baseline {baseline:.4f}")
    for link in ("logit", "probit"):
        row = parity[link]
        if row["max_abs_coef_diff"] > MAX_COEF_DIFF:
            failures.append(f"{link}: R and Python coefficients differ by {row['max_abs_coef_diff']:.2e}")
        for metric in ("auc", "ks", "brier"):
            diff = abs(row["python"][metric] - row["r"][metric])
            if diff > MAX_METRIC_DIFF:
                failures.append(f"{link}: {metric} differs between R and Python by {diff:.2e}")
    return failures


def main() -> None:
    models = json.loads((RESULTS / "models.json").read_text())
    parity = json.loads((RESULTS / "r_vs_python.json").read_text())
    failures = check(models, parity)
    for failure in failures:
        print("GATE FAIL:", failure)
    if failures:
        sys.exit(1)
    print("Gate passed.")


if __name__ == "__main__":
    main()
