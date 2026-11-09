"""CLI scoring demo: python -m credit_risk.score --example"""
import argparse
import json

import joblib
import pandas as pd

from .data import FEATURES, MODELS

LOW, HIGH = 0.10, 0.30
EXAMPLE = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 34,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 46000,
    "BILL_AMT4": 44000, "BILL_AMT5": 43000, "BILL_AMT6": 42000,
    "PAY_AMT1": 0, "PAY_AMT2": 1500, "PAY_AMT3": 1600,
    "PAY_AMT4": 1500, "PAY_AMT5": 1500, "PAY_AMT6": 1500,
}


def band(pd_value: float) -> str:
    return "low" if pd_value < LOW else "medium" if pd_value < HIGH else "high"


def score(model, reference: dict, applicant: dict, top: int = 3) -> dict:
    """PD plus reasons: how much PD changes when one field is set to the training median."""
    missing = [f for f in FEATURES if f not in applicant]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    row = pd.DataFrame([{f: applicant[f] for f in FEATURES}])
    value = float(model.predict_proba(row)[0, 1])
    deltas = {}
    for f in FEATURES:
        alt = row.copy()
        alt[f] = reference[f]
        deltas[f] = value - float(model.predict_proba(alt)[0, 1])
    ranked = sorted(deltas, key=lambda f: -deltas[f])[:top]
    return {
        "pd": round(value, 4),
        "band": band(value),
        "raises_risk_most": [
            {"feature": f, "value": applicant[f], "pd_vs_median": round(deltas[f], 4)} for f in ranked
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="applicant as a JSON object")
    parser.add_argument("--example", action="store_true", help="score a made-up example applicant")
    args = parser.parse_args()
    if not args.example and not args.json:
        parser.error("pass --example or --json")
    applicant = EXAMPLE if args.example else json.loads(args.json)
    meta = json.loads((MODELS / "champion_meta.json").read_text())
    model = joblib.load(MODELS / "champion.joblib")
    print(json.dumps({"model": meta["champion"], **score(model, meta["medians"], applicant)}, indent=2))


if __name__ == "__main__":
    main()
