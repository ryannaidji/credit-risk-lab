"""Rolling-origin backtest of the U.S. credit-card delinquency rate against naive baselines."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HORIZONS = (1, 4)


def naive(history: np.ndarray, h: int) -> float:
    return float(history[-1])


def seasonal_naive(history: np.ndarray, h: int, period: int = 4) -> float:
    return float(history[-period + (h - 1) % period])


def arima(history: np.ndarray, h: int) -> float:
    return float(ARIMA(history, order=(1, 1, 0)).fit().forecast(h)[-1])


def ets(history: np.ndarray, h: int) -> float:
    fit = ExponentialSmoothing(history, trend="add", damped_trend=True).fit()
    return float(fit.forecast(h)[-1])


FORECASTERS = {"naive": naive, "seasonal_naive": seasonal_naive, "arima_1_1_0": arima, "ets_damped": ets}


def backtest(series: pd.Series, n_origins: int = 40, forecasters=None) -> dict:
    """Expanding window. Models only see data up to each forecast origin."""
    forecasters = forecasters or FORECASTERS
    y = series.to_numpy(dtype=float)
    out: dict = {}
    for h in HORIZONS:
        last_origin = len(y) - h
        origins = range(last_origin - n_origins + 1, last_origin + 1)
        scale = float(np.mean(np.abs(y[h : origins[0]] - y[: origins[0] - h])))
        rows = {}
        for name, fn in forecasters.items():
            errors = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for t in origins:
                    errors.append(y[t + h - 1] - fn(y[:t], h))
            e = np.array(errors)
            rows[name] = {
                "mae": float(np.abs(e).mean()),
                "rmse": float(np.sqrt((e**2).mean())),
                "mase": float(np.abs(e).mean() / scale),
            }
        for row in rows.values():
            row["mae_vs_naive"] = row["mae"] / rows["naive"]["mae"]
        out[f"h{h}"] = {
            "n_origins": n_origins,
            "first_target": str(series.index[origins[0] + h - 1]),
            "last_target": str(series.index[origins[-1] + h - 1]),
            "models": rows,
        }
    return out


def run(series: pd.Series) -> dict:
    return {
        "series": "U.S. commercial banks, credit-card delinquency rate, SA, percent, quarterly",
        "n_quarters": int(len(series)),
        "span": [str(series.index[0]), str(series.index[-1])],
        "backtest": backtest(series),
    }
