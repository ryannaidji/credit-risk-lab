"""Risk-model metrics: discrimination (AUC, Gini, KS) and calibration (Brier, ECE, slope)."""
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y, p) -> float:
    """Max distance between the score CDFs of defaulters and non-defaulters."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.unique(p)
    cdf1 = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / (y == 1).sum()
    cdf0 = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / (y == 0).sum()
    return float(np.abs(cdf1 - cdf0).max())


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    """Equal-frequency reliability table."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable")
    return [
        {"n": int(len(i)), "mean_pd": float(p[i].mean()), "observed_rate": float(y[i].mean())}
        for i in np.array_split(order, bins)
    ]


def ece(y, p, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    total = sum(r["n"] for r in table)
    return float(sum(r["n"] * abs(r["mean_pd"] - r["observed_rate"]) for r in table) / total)


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration of y on logit(p); a perfect model gives slope 1, intercept 0."""
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    fit = sm.GLM(np.asarray(y), sm.add_constant(logit), family=sm.families.Binomial()).fit()
    return float(fit.params[1]), float(fit.params[0])


def bootstrap_auc_ci(y, p, n: int = 300, seed: int = 0) -> tuple[float, float]:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(y, p) -> dict:
    auc = float(roc_auc_score(y, p))
    slope, intercept = calibration_slope_intercept(y, p)
    low, high = bootstrap_auc_ci(y, p)
    return {
        "auc": auc,
        "auc_ci95": [low, high],
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
