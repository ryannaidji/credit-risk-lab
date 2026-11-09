"""The model zoo: every candidate is a scikit-learn compatible estimator."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import AMOUNTS, CATEGORICAL, FEATURES, NUMERIC, SEED


def glm_formula(columns=None) -> str:
    cols = FEATURES if columns is None else list(columns)
    terms = [f"C({c})" if c in CATEGORICAL else c for c in cols]
    return "y ~ " + " + ".join(terms)


class StatsmodelsGLM(ClassifierMixin, BaseEstimator):
    """Unpenalized binomial GLM, the same specification as r/glm_benchmark.R."""

    def __init__(self, link: str = "logit"):
        self.link = link

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in AMOUNTS:
            if col in out:
                out[col] = out[col] / 1e4
        return out

    def fit(self, X, y):
        links = {"logit": sm.families.links.Logit(), "probit": sm.families.links.Probit()}
        frame = self._frame(X).assign(y=np.asarray(y))
        family = sm.families.Binomial(links[self.link])
        self.result_ = smf.glm(glm_formula(X.columns), frame, family=family).fit(tol=1e-10, maxiter=100)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(self.result_.predict(self._frame(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _encoder(scale: bool, numeric=None) -> ColumnTransformer:
    numeric = list(NUMERIC if numeric is None else numeric)
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", numeric_step, numeric),
        ]
    )


def logit_pipeline(numeric=None) -> Pipeline:
    return Pipeline(
        [("prep", _encoder(True, numeric)), ("clf", LogisticRegression(max_iter=2000))]
    )


def _tuned(pipe: Pipeline, grid: dict) -> GridSearchCV:
    return GridSearchCV(
        pipe,
        grid,
        scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        n_jobs=1,
    )


def candidates() -> dict:
    """Candidates compared by nested CV: tuning happens only inside each training fold."""
    tree = Pipeline(
        [("prep", _encoder(False)), ("clf", DecisionTreeClassifier(random_state=SEED))]
    )
    forest = Pipeline(
        [
            ("prep", _encoder(False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED
                ),
            ),
        ]
    )
    xgb = Pipeline(
        [
            ("prep", _encoder(False)),
            (
                "clf",
                XGBClassifier(
                    tree_method="hist",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )
    return {
        "logit_sklearn_l2": logit_pipeline(),
        "glm_logit_statsmodels": StatsmodelsGLM("logit"),
        "glm_probit_statsmodels": StatsmodelsGLM("probit"),
        "decision_tree": _tuned(
            tree,
            {"clf__max_depth": [3, 5, 7], "clf__min_samples_leaf": [50, 200]},
        ),
        "random_forest": forest,
        "xgboost": _tuned(
            xgb,
            {
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.08],
                "clf__n_estimators": [200, 400],
            },
        ),
    }


def baselines() -> dict:
    """Reference points: base rate, and a logit on the most recent repayment status only."""
    pay0 = Pipeline(
        [
            ("pick", ColumnTransformer([("p", "passthrough", ["PAY_0"])])),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return {"baseline_prior": DummyClassifier(strategy="prior"), "baseline_pay0_logit": pay0}
