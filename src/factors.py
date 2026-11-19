"""Factor library: price-volume factors adapted from WorldQuant '101 Formulaic Alphas' style.

Each factor is computed per symbol on OHLCV data; cross-sectional evaluation
(IC / group backtest) happens in evaluate.py.
"""
import numpy as np
import pandas as pd

FACTOR_NAMES = [
    "mom_20", "rev_5", "ret_1", "vol_20", "volat_ratio",
    "vwap_dev", "volume_z", "amihud", "corr_pv_10", "range_20",
]


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all factors for one symbol. df indexed by date with OHLCV columns."""
    c, v, qv = df["close"], df["volume"], df["quote_volume"]
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)

    # Trend / reversal
    out["mom_20"] = c / c.shift(20) - 1                       # 20d momentum
    out["rev_5"] = -(c / c.shift(5) - 1)                      # 5d reversal (short-term)
    out["ret_1"] = ret

    # Volatility
    out["vol_20"] = ret.rolling(20).std()
    out["volat_ratio"] = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-12)

    # Price-volume / liquidity
    vwap20 = qv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    out["vwap_dev"] = c / vwap20 - 1                          # deviation from 20d VWAP
    out["volume_z"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)
    out["amihud"] = (ret.abs() / (qv + 1e-12)).rolling(20).mean() * 1e9  # Amihud illiquidity
    out["corr_pv_10"] = c.rolling(10).corr(v)                 # price-volume corr (101-alpha style)
    out["range_20"] = ((df["high"] - df["low"]) / c).rolling(20).mean()  # avg daily range
    return out


def build_panels(raw: dict[str, pd.DataFrame], fwd_days: int = 1):
    """Build {factor_name: panel(date x symbol)} and forward-return panel.

    Forward return of date t = close-to-close return from t to t+fwd_days,
    so a prediction made with information up to t close earns exactly this return.
    """
    factor_tables = {s: compute_factors(df) for s, df in raw.items()}
    fwd = {}
    for s, df in raw.items():
        fwd[s] = df["close"].shift(-fwd_days) / df["close"] - 1

    panels = {}
    for f in FACTOR_NAMES:
        panels[f] = pd.DataFrame({s: t[f] for s, t in factor_tables.items()})
    fwd_panel = pd.DataFrame(fwd)
    return panels, fwd_panel


def winsorize_panel(p: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
    """Clip cross-sectional outliers: clip each row at mean +/- n_std * std."""
    mu, sd = p.mean(axis=1), p.std(axis=1)
    lower, upper = mu - n_std * sd, mu + n_std * sd
    return p.clip(lower=lower, upper=upper, axis=0)
