"""Walk-forward LightGBM long-short backtest.

Design (anti-lookahead):
- At each rebalance date t, train on days in [t - train_window, t), i.e. strictly past data.
- Predict forward return for days [t, t + step). The realized return of day t
  (close-to-close t -> t+1) is earned after the position is formed at t close.
- Transaction costs applied on turnover (|weight change| summed across names).
"""
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # OSError on macOS when libomp is missing
    lgb = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor as _HGBR
except ImportError:
    _HGBR = None

if lgb is None and _HGBR is None:
    raise ImportError("need either lightgbm or scikit-learn installed")


def _make_model(params: dict):
    """GBDT backend: prefer LightGBM, fall back to sklearn HistGradientBoosting
    (same family of algorithm; on macOS lightgbm additionally needs libomp)."""
    if lgb is not None:
        return ("lgb", lgb.LGBMRegressor(**params))
    return ("sk", _HGBR(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, early_stopping=False))


def stack_long(panels: dict, fwd_panel: pd.DataFrame) -> pd.DataFrame:
    """Stack factor panels into long format: columns = factors + fwd_ret + symbol + date."""
    stacked = []
    for name, panel in panels.items():
        s = panel.stack().rename(name)
        stacked.append(s)
    X = pd.concat(stacked, axis=1)
    y = fwd_panel.stack().rename("fwd_ret")
    df = X.join(y, how="inner").dropna()
    df = df.reset_index()
    df.columns = ["date", "symbol"] + list(panels.keys()) + ["fwd_ret"]
    return df


def walk_forward_predict(long_df: pd.DataFrame, factor_names: list[str],
                         train_window: int = 180, step: int = 20,
                         **lgb_params) -> pd.DataFrame:
    """Return DataFrame with columns [date, symbol, pred] for out-of-sample predictions."""
    params = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
              "min_child_samples": 40, "verbose": -1}
    params.update(lgb_params)
    dates = np.sort(long_df["date"].unique())
    preds = []
    for t in range(train_window, len(dates) - 1, step):
        past_dates = dates[t - train_window: t]
        future_dates = dates[t: min(t + step, len(dates) - 1)]
        train = long_df[long_df["date"].isin(past_dates)]
        pred_mask = long_df["date"].isin(future_dates)
        if len(train) < 500:  # not enough history yet
            continue
        kind, model = _make_model(params)
        model.fit(train[factor_names], train["fwd_ret"])
        chunk = long_df[pred_mask][["date", "symbol"] + factor_names].copy()
        chunk["pred"] = model.predict(chunk[factor_names])
        preds.append(chunk[["date", "symbol", "pred"]])
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def long_short_backtest(preds: pd.DataFrame, fwd_panel: pd.DataFrame,
                        top_frac: float = 0.2, cost_bps: float = 5.0) -> dict:
    """Form long-top / short-bottom portfolios each predicted day; report performance.

    Returns dict with daily return series and summary stats.
    """
    daily = preds.groupby("date")
    rets, turnovers = [], []
    prev_long, prev_short = set(), set()
    for date, chunk in daily:
        k = max(int(len(chunk) * top_frac), 1)
        ranked = chunk.sort_values("pred", ascending=False)
        long_names = set(ranked.head(k)["symbol"])
        short_names = set(ranked.tail(k)["symbol"])
        long_ret = np.mean([fwd_panel.at[date, s] for s in long_names if s in fwd_panel.columns])
        short_ret = np.mean([fwd_panel.at[date, s] for s in short_names if s in fwd_panel.columns])
        ret = 0.5 * long_ret - 0.5 * short_ret
        turnover = (len(long_names - prev_long) + len(short_names - prev_short)) / (2 * k)
        cost = turnover * cost_bps / 1e4
        rets.append((date, ret - cost))
        turnovers.append(turnover)
        prev_long, prev_short = long_names, short_names
    series = pd.Series({d: r for d, r in rets}).sort_index()
    if series.empty:
        return {}
    ann = np.sqrt(365)
    sharpe = series.mean() / series.std() * ann if series.std() > 0 else np.nan
    cum = (1 + series).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {
        "daily_returns": series,
        "cumulative": cum,
        "ann_return": round(series.mean() * 365, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "avg_turnover": round(np.mean(turnovers), 4),
        "n_days": len(series),
    }
