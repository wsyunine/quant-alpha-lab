"""Data utilities: quality checks + synthetic panel for offline smoke tests.

Real-data path is Tushare (see data_tushare.py). The old Binance kline
downloader was removed when the project switched to A-share data.
"""
import numpy as np
import pandas as pd


def check_data(df: pd.DataFrame, name: str = "") -> None:
    """Basic data quality checks: gaps and extreme moves."""
    gaps = df.index.to_series().diff().dt.days.dropna()
    big = (df["close"].pct_change().abs() > 0.30).sum()
    print(f"[check] {name}: rows={len(df)}, max_gap_days={gaps.max():.0f}, "
          f"days with |ret|>30%: {big}")


def make_synthetic(symbols: list[str], n_days: int = 500, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV panel with embedded momentum regimes, for offline smoke tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    out = {}
    for i, sym in enumerate(symbols):
        n = len(dates)
        # momentum regimes: drift switches sign every ~40 days
        regime = np.sign(rng.standard_normal(n // 40 + 1)).repeat(40)[:n]
        drift = regime * 0.002 + rng.standard_normal(n) * 0.018
        close = 100 * (1 + drift).cumprod() * (1 + i * 0.1)
        open_ = np.roll(close, 1); open_[0] = 100
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
        volume = np.exp(rng.normal(14, 0.5, n))
        quote_volume = volume * close
        out[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume, "quote_volume": quote_volume}, index=dates)
    return out
