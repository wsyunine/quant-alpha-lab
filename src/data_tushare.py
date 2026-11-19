"""A-share data layer via Tushare.

Output format (shared contract with src/data.py utilities):

    raw: dict[ts_code] -> DataFrame indexed by date (DatetimeIndex),
         columns: open, high, low, close, volume, quote_volume

Notes:
- Prices are forward-adjusted (qfq): p_adj = p * adj_factor / latest_adj_factor.
  adj_factor requires >=2000 Tushare points; if unavailable we fall back to
  unadjusted prices and print a warning (returns will be distorted around
  dividend dates -- state this in your report).
- volume: Tushare `vol` is in hands (100 shares) -> convert to shares.
- quote_volume: Tushare `amount` is in 1000 CNY -> convert to CNY.
"""
import os
import sys
import time

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 10 liquid, well-known names for the first run (replace with HS300 members later)
DEMO_CODES = [
    "000001.SZ",  # 平安银行
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "601318.SH",  # 中国平安
    "600036.SH",  # 招商银行
    "000333.SZ",  # 美的集团
    "600276.SH",  # 恒瑞医药
    "002415.SZ",  # 海康威视
    "601888.SH",  # 中国中免
    "300750.SZ",  # 宁德时代
]


def _pro():
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit(
            "Missing token. Run:  export TUSHARE_TOKEN=your_token   "
            "(token from tushare.pro -> 个人主页 -> 接口TOKEN)")
    ts.set_token(token)
    return ts.pro_api()


def fetch_symbol(code: str, start: str, end: str, pro=None) -> pd.DataFrame:
    """Download one symbol -> qfq OHLCV panel indexed by date."""
    pro = pro or _pro()
    start_s = start.replace("-", "")
    end_s = end.replace("-", "")

    daily = pro.daily(ts_code=code, start_date=start_s, end_date=end_s)
    if daily is None or daily.empty:
        raise RuntimeError(f"no daily data for {code} (check points/permission)")
    daily = daily[["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"]]
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.set_index("trade_date").sort_index()
    daily["volume"] = daily.pop("vol") * 100.0        # hands -> shares
    daily["quote_volume"] = daily.pop("amount") * 1000.0  # kCNY -> CNY

    # --- adjustment ---
    # Try adj_factor first (needs >=2000 points). On failure, reconstruct the
    # cumulative adjustment factor from pre_close, which Tushare already sets to
    # the ex-dividend-adjusted previous close:
    #   daily ratio q(t) = close(t-1) / pre_close(t)   (= 1 when no corporate action)
    #   hfq factor  f(t) = cumprod(q)
    #   qfq scale        = f(t) / f(latest)  -> applied to OHLC
    # (Equivalent to (1 + pct_chg/100) * close(t-1)/close(t), but exact: prices
    #  carry full precision while pct_chg is rounded to 2 decimals.)
    try:
        adj = pro.adj_factor(ts_code=code, start_date=start_s, end_date=end_s)
        adj["trade_date"] = pd.to_datetime(adj["trade_date"])
        adj = adj.set_index("trade_date").sort_index()["adj_factor"]
        factor = adj / adj.iloc[-1]  # latest price = real price
        for col in ["open", "high", "low", "close"]:
            daily[col] = daily[col] * factor.reindex(daily.index)
        print(f"[adj  ] {code}: adjusted via adj_factor")
    except Exception:  # noqa: BLE001
        ratio = (daily["close"].shift(1) / daily["pre_close"]).fillna(1.0)
        hfq_factor = ratio.cumprod()
        scale = hfq_factor / hfq_factor.iloc[-1]
        for col in ["open", "high", "low", "close"]:
            daily[col] = daily[col] * scale
        print(f"[adj  ] {code}: adjusted via pre_close reconstruction "
              f"(no adj_factor permission)")

    return daily[["open", "high", "low", "close", "volume", "quote_volume"]]


def download_all(codes: list[str], start: str, end: str, sleep: float = 0.35) -> dict[str, pd.DataFrame]:
    """Download all symbols with caching to data/ashare_<code>.csv."""
    pro = _pro()
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {}
    for code in codes:
        cache = os.path.join(DATA_DIR, f"ashare_{code}.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            print(f"[cache] {code}: {len(df)} rows")
        else:
            df = fetch_symbol(code, start, end, pro=pro)
            df.to_csv(cache)
            print(f"[down ] {code}: {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")
            time.sleep(sleep)
        out[code] = df
    return out


def load_cached(codes: list[str]) -> dict[str, pd.DataFrame]:
    """Load previously downloaded panels from data/ (offline)."""
    out = {}
    for code in codes:
        cache = os.path.join(DATA_DIR, f"ashare_{code}.csv")
        if not os.path.exists(cache):
            raise SystemExit(f"cache missing: {cache}. Run data_tushare.py first.")
        out[code] = pd.read_csv(cache, index_col=0, parse_dates=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="download A-share daily data via Tushare")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--codes", nargs="+", default=DEMO_CODES)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    if args.force:
        import glob
        for f in glob.glob(os.path.join(DATA_DIR, "ashare_*.csv")):
            os.remove(f)
        print("[force] cache cleared")
    panels = download_all(args.codes, args.start, args.end)
    print(f"\nOK: {len(panels)} symbols cached under {DATA_DIR}")
