"""K-means customer segmentation on behavioural features, profiled on the holdout."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import BILL, PAY, PAY_AMT, SEED, TARGET

SEGMENT_FEATURES = ["log_limit", "utilization", "pay_ratio", "months_late", "max_delay"]


def behaviour(df: pd.DataFrame) -> pd.DataFrame:
    avg_bill = df[BILL].mean(axis=1)
    return pd.DataFrame(
        {
            "log_limit": np.log(df["LIMIT_BAL"]),
            "utilization": (avg_bill / df["LIMIT_BAL"]).clip(-0.5, 1.5),
            "pay_ratio": (df[PAY_AMT].mean(axis=1) / avg_bill.clip(lower=1)).clip(0, 2),
            "months_late": (df[PAY] > 0).sum(axis=1),
            "max_delay": df[PAY].max(axis=1).clip(lower=0),
        },
        index=df.index,
    )


def _kmeans(k: int, seed: int = SEED) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("km", KMeans(n_clusters=k, n_init=10, random_state=seed))]
    )


def run(train: pd.DataFrame, holdout: pd.DataFrame, ks=range(2, 9), sample: int = 5000) -> dict:
    X = behaviour(train)
    scaled = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    silhouette = {}
    for k in ks:
        labels = _kmeans(k).fit_predict(X)
        silhouette[k] = float(silhouette_score(scaled[idx], labels[idx]))
    best = max(silhouette, key=silhouette.get)
    model = _kmeans(best).fit(X)
    alt = _kmeans(best, SEED + 1).fit(X)
    Xh = behaviour(holdout)
    labels = model.predict(Xh)
    profile = []
    for k in np.unique(labels):
        mask = labels == k
        profile.append(
            {
                "segment": int(k),
                "holdout_share": float(mask.mean()),
                "holdout_default_rate": float(holdout[TARGET].values[mask].mean()),
                **{c: float(Xh[c].values[mask].mean()) for c in SEGMENT_FEATURES},
            }
        )
    return {
        "features": SEGMENT_FEATURES,
        "fit_on": "train only",
        "silhouette_by_k": {str(k): v for k, v in silhouette.items()},
        "chosen_k": int(best),
        "seed_stability_ari": float(adjusted_rand_score(model.predict(X), alt.predict(X))),
        "holdout_profile": sorted(profile, key=lambda r: r["holdout_default_rate"]),
    }
