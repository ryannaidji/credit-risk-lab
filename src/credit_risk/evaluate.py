"""Run the whole Python study and write results/*.json. Run on CI, not on a laptop."""
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import explain, forecast, metrics, monitor, segments
from .data import FEATURES, MODELS, RESULTS, SEED, TARGET, load_delinquency, load_split
from .models import StatsmodelsGLM, baselines, candidates


def dump(name: str, obj: dict) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2) + "\n")


def conditions(train, holdout) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "default_rate_train": float(train[TARGET].mean()),
        "split": "stratified 70/30 on default, data/split.csv, seed 42",
    }


def glm_export(train, holdout) -> None:
    """Coefficients and holdout predictions, for the comparison against R."""
    rows, preds = [], {"ID": holdout["ID"].to_numpy(), "default": holdout[TARGET].to_numpy()}
    for link in ("logit", "probit"):
        model = StatsmodelsGLM(link).fit(train[FEATURES], train[TARGET])
        for name, value in model.result_.params.items():
            rows.append({"link": link, "term": name, "estimate": value})
        preds[link] = model.predict_proba(holdout[FEATURES])[:, 1]
    pd.DataFrame(rows).to_csv(RESULTS / "python_glm_coefs.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTS / "python_glm_preds.csv", index=False)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, holdout = load_split()
    Xtr, ytr, Xho, yho = train[FEATURES], train[TARGET], holdout[FEATURES], holdout[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted = {}, {}
    for name, estimator in candidates().items():
        start = time.perf_counter()
        scores = cross_val_score(clone(estimator), Xtr, ytr, cv=cv, scoring="roc_auc")
        model = clone(estimator).fit(Xtr, ytr)
        fitted[name] = model
        rows[name] = {
            "cv_auc_mean": float(scores.mean()),
            "cv_auc_std": float(scores.std()),
            "holdout": metrics.evaluate(yho, model.predict_proba(Xho)[:, 1]),
            "fit_and_cv_seconds": round(time.perf_counter() - start, 1),
        }
        if hasattr(model, "best_params_"):
            rows[name]["best_params"] = model.best_params_
        print(name, round(rows[name]["cv_auc_mean"], 4), flush=True)

    base = {}
    for name, estimator in baselines().items():
        model = clone(estimator).fit(Xtr, ytr)
        p = model.predict_proba(Xho)[:, 1]
        if name == "baseline_prior":
            base[name] = {"holdout": {"auc": 0.5, "gini": 0.0, "brier": float(brier_score_loss(yho, p))}}
        else:
            base[name] = {"holdout": metrics.evaluate(yho, p)}

    champion = max(rows, key=lambda k: rows[k]["cv_auc_mean"])
    model = fitted[champion]
    pipe = getattr(model, "best_estimator_", model)
    dump(
        "models.json",
        {
            "conditions": conditions(train, holdout),
            "selection": "highest 5-fold CV AUC on the training set; the holdout is not used to choose",
            "champion": champion,
            "models": rows,
            "baselines": base,
            "champion_calibration_table": metrics.calibration_table(yho, model.predict_proba(Xho)[:, 1]),
        },
    )
    glm_export(train, holdout)
    joblib.dump(pipe, MODELS / "champion.joblib", compress=3)
    medians = {f: float(np.median(Xtr[f])) for f in FEATURES}
    meta = {"champion": champion, "medians": medians}
    (MODELS / "champion_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    dump(
        "explain.json",
        {
            "champion": champion,
            "permutation_importance_auc_drop": explain.permutation(pipe, holdout),
            "xgboost_treeshap_global": explain.xgb_shap(fitted["xgboost"], holdout),
        },
    )
    dump("segments.json", segments.run(train, holdout))
    dump("forecast.json", forecast.run(load_delinquency()))
    dump(
        "monitoring.json",
        {
            "thresholds": {"amber": monitor.AMBER, "red": monitor.RED},
            "calendar_drift_real": monitor.calendar_drift(train, holdout),
            "feedback_loop_simulated": monitor.feedback_experiment(pipe, train, holdout),
        },
    )


if __name__ == "__main__":
    main()
