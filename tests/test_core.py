import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk import compare_r, data, forecast, gate, metrics, monitor, score, segments
from credit_risk.models import StatsmodelsGLM, candidates, glm_formula


@pytest.fixture(scope="module")
def sample():
    return data.load_credit().sample(1500, random_state=1)


def test_split_is_stratified_disjoint_and_reproducible():
    df = data.load_credit()
    saved = pd.read_csv(data.DATA / "split.csv")
    fresh = data.make_split(df).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, fresh)
    train, holdout = data.load_split()
    assert not set(train["ID"]) & set(holdout["ID"])
    assert abs(train["default"].mean() - holdout["default"].mean()) < 0.001
    assert len(holdout) == 9000


def test_clean_merges_undocumented_codes():
    df = data.load_credit()
    assert set(df["EDUCATION"]) == {1, 2, 3, 4}
    assert set(df["MARRIAGE"]) == {1, 2, 3}


def test_ks_perfect_and_known_value():
    y = np.array([0, 0, 1, 1])
    assert metrics.ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert metrics.ks_statistic(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(0.5)


def test_calibrated_scores_have_slope_near_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.9, 20000)
    y = rng.binomial(1, p)
    slope, intercept = metrics.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1, abs=0.1)
    assert intercept == pytest.approx(0, abs=0.1)
    assert metrics.ece(y, p) < 0.03


def test_psi_zero_for_same_and_large_for_shift():
    a = np.random.default_rng(0).normal(size=20000)
    assert monitor.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert monitor.psi(a, a + 2) > monitor.RED


def test_psi_handles_constant_reference():
    assert monitor.psi(np.zeros(100), np.zeros(100)) == pytest.approx(0, abs=1e-9)
    assert monitor.psi(np.zeros(100), np.ones(100)) > monitor.RED


def test_alert_levels():
    assert monitor.level(0.05) == "green"
    assert monitor.level(0.1) == "amber"
    assert monitor.level(0.25) == "red"
    assert monitor.decide(0.01, {"a": 0.3})["action"] == "retrain"
    assert monitor.decide(0.15, {"a": 0.01})["action"] == "investigate"
    assert monitor.decide(0.01, {"a": 0.02})["action"] == "none"


def test_jun_window_uses_earlier_columns(sample):
    jun = monitor.jun_as_sep(sample)
    assert list(jun.columns) == monitor.WINDOW_FEATURES
    assert (jun["PAY_0"].values == sample["PAY_4"].values).all()
    assert (jun["BILL_AMT3"].values == sample["BILL_AMT6"].values).all()


def test_corrupt_changes_units_only_on_copy(sample):
    bad = monitor.corrupt(sample)
    assert bad["LIMIT_BAL"].iloc[0] == sample["LIMIT_BAL"].iloc[0] / 1000
    assert bad["PAY_0"].max() <= 8
    assert (sample["LIMIT_BAL"] >= 10000).all()


def test_candidates_are_all_named_models():
    assert set(candidates()) >= {
        "logit_sklearn_l2", "glm_logit_statsmodels", "decision_tree", "random_forest", "xgboost",
    }


def test_glm_fits_and_formula_marks_categoricals(sample):
    assert "C(SEX)" in glm_formula() and "C(AGE)" not in glm_formula()
    model = StatsmodelsGLM("logit").fit(sample[data.FEATURES], sample["default"])
    p = model.predict_proba(sample[data.FEATURES])[:, 1]
    assert 0.6 < roc_auc_score(sample["default"], p) <= 1
    assert model.predict(sample[data.FEATURES]).shape == (len(sample),)


def test_normalize_r_and_python_terms_match():
    assert compare_r.normalize_term("C(SEX)[T.2]") == compare_r.normalize_term("factor(SEX)2") == "SEX=2"
    assert compare_r.normalize_term("(Intercept)") == "Intercept"
    assert compare_r.normalize_term("PAY_0") == "PAY_0"


def test_naive_and_seasonal_naive():
    h = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    assert forecast.naive(h, 4) == 8
    assert forecast.seasonal_naive(h, 1) == 5
    assert forecast.seasonal_naive(h, 4) == 8


def test_backtest_never_peeks():
    series = pd.Series(np.arange(30.0), index=[str(i) for i in range(30)])
    seen = []

    def spy(history, h):
        seen.append(len(history))
        return history[-1]

    result = forecast.backtest(series, n_origins=5, forecasters={"naive": spy})
    assert result["h1"]["models"]["naive"]["mae"] == pytest.approx(1)
    assert result["h4"]["models"]["naive"]["mae"] == pytest.approx(4)
    assert max(seen) == 29
    assert result["h1"]["last_target"] == "29"


def test_segments_run_and_are_sized(sample):
    train, holdout = sample.iloc[:1000], sample.iloc[1000:]
    out = segments.run(train, holdout, ks=range(2, 4), sample=500)
    assert out["chosen_k"] in (2, 3)
    assert sum(r["holdout_share"] for r in out["holdout_profile"]) == pytest.approx(1)


class FakeModel:
    def predict_proba(self, X):
        p = np.clip(X["PAY_0"].to_numpy() * 0.1 + 0.2, 0, 1)
        return np.column_stack([1 - p, p])


def test_score_returns_band_and_reason():
    reference = {f: 0 for f in data.FEATURES}
    result = score.score(FakeModel(), reference, score.EXAMPLE)
    assert result["pd"] == pytest.approx(0.4)
    assert result["band"] == "high"
    assert result["raises_risk_most"][0]["feature"] == "PAY_0"


def test_score_rejects_missing_field():
    with pytest.raises(ValueError):
        score.score(FakeModel(), {}, {"PAY_0": 1})


def _gate_inputs(auc=0.78, base=0.65, coef=1e-6):
    metric = {"auc": 0.7, "ks": 0.3, "brier": 0.14}
    row = {"max_abs_coef_diff": coef, "python": metric, "r": metric}
    models = {
        "champion": "x",
        "models": {"x": {"holdout": {"auc": auc}}},
        "baselines": {"baseline_pay0_logit": {"holdout": {"auc": base}}},
    }
    return models, {"logit": row, "probit": row}


def test_gate_pass_and_failures():
    assert gate.check(*_gate_inputs()) == []
    assert any("AUC" in f for f in gate.check(*_gate_inputs(auc=0.7)))
    assert any("baseline" in f for f in gate.check(*_gate_inputs(base=0.9)))
    assert any("coefficients" in f for f in gate.check(*_gate_inputs(coef=0.1)))
