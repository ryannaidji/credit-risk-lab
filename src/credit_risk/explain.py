"""Global explanations: permutation importance (any model) and XGBoost TreeSHAP."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance

from .data import FEATURES, SEED, TARGET


def permutation(model, holdout: pd.DataFrame, n: int = 3000) -> list[dict]:
    sample = holdout.sample(min(n, len(holdout)), random_state=SEED)
    result = permutation_importance(
        model, sample[FEATURES], sample[TARGET], scoring="roc_auc", n_repeats=5, random_state=SEED
    )
    order = np.argsort(-result.importances_mean)
    return [
        {
            "feature": FEATURES[i],
            "auc_drop": float(result.importances_mean[i]),
            "std": float(result.importances_std[i]),
        }
        for i in order[:10]
    ]


def xgb_shap(search, holdout: pd.DataFrame, n: int = 3000) -> list[dict]:
    """Native TreeSHAP from the booster, in log-odds units."""
    pipe = search.best_estimator_
    sample = holdout.sample(min(n, len(holdout)), random_state=SEED)
    encoded = pipe.named_steps["prep"].transform(sample[FEATURES])
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    contribs = pipe.named_steps["clf"].get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": names[i], "mean_abs_shap_log_odds": float(mean_abs[i])} for i in order[:10]]
