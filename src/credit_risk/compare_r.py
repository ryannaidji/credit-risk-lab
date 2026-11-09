"""Compare R glm() with statsmodels on the same split: coefficients, predictions, metrics."""
import json
import re

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import RESULTS
from .metrics import ks_statistic


def normalize_term(term: str) -> str:
    """statsmodels C(SEX)[T.2] and R factor(SEX)2 both become SEX=2."""
    term = term.strip()
    match = re.fullmatch(r"C\((\w+)\)\[T\.(\w+)\]", term) or re.fullmatch(r"factor\((\w+)\)(\w+)", term)
    if match:
        return f"{match.group(1)}={match.group(2)}"
    return "Intercept" if term == "(Intercept)" else term


def _coefs(frame: pd.DataFrame, link: str) -> pd.Series:
    rows = frame[frame["link"] == link]
    return pd.Series(rows["estimate"].to_numpy(), index=rows["term"].map(normalize_term).to_numpy())


def compare() -> dict:
    py = pd.read_csv(RESULTS / "python_glm_coefs.csv")
    r = pd.read_csv(RESULTS / "r_glm_coefs.csv")
    py_preds = pd.read_csv(RESULTS / "python_glm_preds.csv")
    r_preds = pd.read_csv(RESULTS / "r_glm_preds.csv").drop(columns="default")
    r_metrics = pd.read_csv(RESULTS / "r_metrics.csv").set_index("link")
    preds = py_preds.merge(r_preds, on="ID", suffixes=("_py", "_r"), validate="one_to_one")
    out: dict = {"n_holdout": int(len(preds))}
    y = preds["default"]
    for link in ("logit", "probit"):
        a, b = _coefs(py, link), _coefs(r, link)
        if set(a.index) != set(b.index):
            raise ValueError(f"term mismatch for {link}: {sorted(set(a.index) ^ set(b.index))}")
        p = preds[f"{link}_py"]
        out[link] = {
            "n_terms": int(len(a)),
            "max_abs_coef_diff": float((a - b.loc[a.index]).abs().max()),
            "max_abs_pred_diff": float((p - preds[f"{link}_r"]).abs().max()),
            "python": {
                "auc": float(roc_auc_score(y, p)),
                "ks": ks_statistic(y, p),
                "brier": float(brier_score_loss(y, p)),
            },
            "r": {m: float(r_metrics.loc[link, m]) for m in ("auc", "ks", "brier")},
        }
    return out


def main() -> None:
    result = compare()
    (RESULTS / "r_vs_python.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
