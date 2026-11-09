"""Drift monitoring (PSI), alert thresholds, and a retrain-on-alert feedback loop."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .data import BILL, FEATURES, NUMERIC, PAY_AMT, TARGET
from .models import logit_pipeline

AMBER, RED = 0.10, 0.25


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. Bin edges are reference quantiles, deduplicated for discrete fields."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins, endpoint=False)))
    size = len(edges) + 1
    ref = np.bincount(np.searchsorted(edges, reference, side="left"), minlength=size) / len(reference)
    cur = np.bincount(np.searchsorted(edges, current, side="left"), minlength=size) / len(current)
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def level(value: float) -> str:
    return "red" if value >= RED else "amber" if value >= AMBER else "green"


def _all_psi(ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str]) -> tuple[dict, str]:
    values = {c: psi(ref[c], cur[c]) for c in columns}
    return values, max(values, key=values.get)


def decide(score_psi: float, feature_psi: dict) -> dict:
    """Feedback rule: amber asks for investigation, red queues a retrain on newly labelled data."""
    worst = max(feature_psi.values())
    state = level(max(score_psi, worst))
    return {
        "alert": state,
        "score_psi": score_psi,
        "max_feature_psi": worst,
        "action": {"green": "none", "amber": "investigate", "red": "retrain"}[state],
    }


SEP_WINDOW = ["PAY_0", "PAY_2", "PAY_3"] + BILL[:3] + PAY_AMT[:3]
JUN_WINDOW = ["PAY_4", "PAY_5", "PAY_6"] + BILL[3:] + PAY_AMT[3:]
STATIC_NUMERIC = ["LIMIT_BAL", "AGE"]
STATIC = STATIC_NUMERIC + ["SEX", "EDUCATION", "MARRIAGE"]
WINDOW_FEATURES = STATIC + SEP_WINDOW


def jun_as_sep(df: pd.DataFrame) -> pd.DataFrame:
    """The same customers three months earlier, in the columns the Jul-Sep model expects."""
    out = df[STATIC + JUN_WINDOW].rename(columns=dict(zip(JUN_WINDOW, SEP_WINDOW)))
    return out[WINDOW_FEATURES]


def calendar_drift(train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """Real time drift: Apr-Jun 2005 snapshot as reference, Jul-Sep 2005 as the later one."""
    model = logit_pipeline(STATIC_NUMERIC + SEP_WINDOW)
    model.fit(train[WINDOW_FEATURES], train[TARGET])
    sep, jun = holdout[WINDOW_FEATURES], jun_as_sep(holdout)
    p_sep, p_jun = model.predict_proba(sep)[:, 1], model.predict_proba(jun)[:, 1]
    feature_psi, worst = _all_psi(jun, sep, SEP_WINDOW)
    score = psi(p_jun, p_sep)
    return {
        "reference": "holdout customers, Apr-Jun 2005 features",
        "current": "same customers, Jul-Sep 2005 features",
        "model": "3-month-window logit trained on Jul-Sep features of the training set",
        "decision": decide(score, feature_psi),
        "worst_feature": worst,
        "feature_psi": feature_psi,
        "auc_on_current_features": float(roc_auc_score(holdout[TARGET], p_sep)),
        "auc_on_3_month_stale_features": float(roc_auc_score(holdout[TARGET], p_jun)),
        "note": "PAY_5 and PAY_6 never take value 1 in the source data, which inflates PSI on PAY fields.",
    }


def corrupt(df: pd.DataFrame) -> pd.DataFrame:
    """SIMULATED upstream bug: amounts arrive in thousands and PAY_0 is shifted by +2."""
    out = df.copy()
    amounts = ["LIMIT_BAL"] + BILL + PAY_AMT
    out[amounts] = out[amounts] / 1000.0
    out["PAY_0"] = (out["PAY_0"] + 2).clip(upper=8)
    return out


def feedback_experiment(model, train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """Clean control, then a simulated bug: does the alert fire, and does retraining help?"""
    train_scores = model.predict_proba(train[FEATURES])[:, 1]
    clean_scores = model.predict_proba(holdout[FEATURES])[:, 1]
    clean_feats, _ = _all_psi(train, holdout, NUMERIC)
    control = decide(psi(train_scores, clean_scores), clean_feats)

    bad = corrupt(holdout)
    half = len(bad) // 2
    labelled, later = bad.iloc[:half], bad.iloc[half:]
    drift_feats, worst = _all_psi(train, labelled, NUMERIC)
    drift_scores = model.predict_proba(labelled[FEATURES])[:, 1]
    alarm = decide(psi(train_scores, drift_scores), drift_feats)

    y_later = later[TARGET]
    clean_later = holdout.iloc[half:]
    retrained = clone(model).fit(labelled[FEATURES], labelled[TARGET])
    auc = lambda m, X: float(roc_auc_score(y_later, m.predict_proba(X[FEATURES])[:, 1]))
    return {
        "simulated": True,
        "scenario": "amounts divided by 1000, PAY_0 + 2 (clipped at 8), applied to the real holdout",
        "control_clean_holdout": control,
        "drifted_batch": alarm,
        "worst_feature": worst,
        "n_labelled_for_retrain": int(half),
        "n_evaluated": int(len(later)),
        "auc_later_clean_data": auc(model, clean_later),
        "auc_later_drifted_no_action": auc(model, later),
        "auc_later_drifted_after_retrain": auc(retrained, later),
    }
