"""Build dashboard/index.html from real pipeline outputs.

Runs the full pipeline on the Tushare cache (data/), collects:
  - OHLCV per stock          -> section 1 (K-line + table)
  - factor panels            -> section 2 (line + cross-section snapshot)
  - IC/ICIR/t/LS-sharpe      -> section 3 (table + bars + group heatmap)
  - walk-forward backtest    -> section 4 (nav / drawdown / turnover)
and injects everything as JSON into template.html.

Usage:  python dashboard/build_dashboard.py
Output: dashboard/index.html  (self-contained, open in browser)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd

import data_tushare as T
import factors as F
import evaluate as E
import backtest as B

STOCK_NAMES = {
    "000001.SZ": "平安银行", "600519.SH": "贵州茅台", "000858.SZ": "五粮液",
    "601318.SH": "中国平安", "600036.SH": "招商银行", "000333.SZ": "美的集团",
    "600276.SH": "恒瑞医药", "002415.SZ": "海康威视", "601888.SH": "中国中免",
    "300750.SZ": "宁德时代",
}
TRAIN_WINDOW, STEP, COST_BPS = 180, 20, 5.0

# 因子元数据：集中定义（名称 -> 一句话金融含义），dashboard 直接读取。
# 复现 101 Alphas 时，把每个新因子在这里登记一句话即可自动进可视化。
FACTOR_DEFS = {
    "mom_20": "动量：收盘价÷20天前收盘价−1。过去一个月涨势强不强（趋势惯性假说，Jegadeesh & Titman 1993）",
    "rev_5": "短期反转：负的5日涨幅。短期涨太多会不会跌回来（Lehmann 1990）",
    "ret_1": "隔日反转：昨天一天的涨跌幅，最极端的短期反转",
    "vol_20": "波动率：最近20天日收益率标准差。波动大的股票未来跑输（低波异象，Ang et al. 2006）",
    "volat_ratio": "波动加速：5日波动÷20日波动。近期波动突然放大的信号（自构因子）",
    "vwap_dev": "VWAP偏离：收盘价÷20日成交量加权均价−1。现价相对机构平均成本的位置",
    "volume_z": "放量异常：今日成交量相对20日均量偏离几个标准差。异常资金进出",
    "amihud": "非流动性：|日涨跌幅|÷成交额再取20日均值。撬动价格所需资金越少越'薄'（Amihud 2002）",
    "corr_pv_10": "量价相关：最近10天价格与成交量的相关系数。放量时涨还是跌",
    "range_20": "振幅：平均每日(最高−最低)÷收盘。日内分歧程度（与 Parkinson 波动率估计器同源）",
}


def r2(x):
    return None if pd.isna(x) else round(float(x), 2)


def main():
    print("[1/5] loading cached A-share data ...")
    codes = T.DEMO_CODES
    raw = T.load_cached(codes)

    print("[2/5] computing factors ...")
    panels, fwd_panel = F.build_panels(raw)

    print("[3/5] cross-sectional evaluation ...")
    summary = E.evaluate_all(panels, fwd_panel)

    print(f"[4/5] walk-forward backtest (window={TRAIN_WINDOW}, step={STEP}) ...")
    long_df = B.stack_long(panels, fwd_panel)
    preds = B.walk_forward_predict(long_df, F.FACTOR_NAMES,
                                   train_window=TRAIN_WINDOW, step=STEP)
    res = B.long_short_backtest(preds, fwd_panel, cost_bps=COST_BPS)
    if not res:
        raise SystemExit("walk-forward produced no predictions; not enough history")

    print("[5/5] building dashboard ...")

    # ---- section 1: OHLCV ----
    stocks = {}
    for code, df in raw.items():
        stocks[code] = {
            "name": STOCK_NAMES.get(code, code),
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "open": [r2(v) for v in df["open"]],
            "high": [r2(v) for v in df["high"]],
            "low": [r2(v) for v in df["low"]],
            "close": [r2(v) for v in df["close"]],
            "volume": [r2(v) for v in df["volume"]],                 # 股
            "amount": [r2(v) for v in df["quote_volume"]],          # 元
        }

    # ---- section 2: factor panels (date x symbol) ----
    # symbols use ts_codes（与板块一的股票选择键一致）；中文显示名在模板端映射
    factors = {}
    for fname, panel in panels.items():
        factors[fname] = {
            "dates": panel.index.strftime("%Y-%m-%d").tolist(),
            "symbols": list(panel.columns),
            "values": [[None if pd.isna(v) else round(float(v), 4) for v in row]
                       for row in panel.values],
        }

    # ---- section 3: evaluation ----
    eval_rows = []
    group_means = {}
    for fname, panel in panels.items():
        s = summary.loc[fname]
        g = E.group_backtest(panel, fwd_panel)
        gm = g[[f"G{i}" for i in range(1, 6)]].mean() * 100  # -> %
        group_means[fname] = gm
        eval_rows.append({
            "factor": fname,
            "IC_mean": float(s["IC_mean"]), "ICIR": float(s["ICIR"]),
            "t_stat": float(s["t_stat"]), "ic_pos": float(s["IC>0_ratio"]),
            "n_days": int(s["n_days"]), "ls_sharpe": float(s["LS_sharpe_ann"]),
        })

    # ---- section 4: backtest ----
    nav_dates = res["cumulative"].index.strftime("%Y-%m-%d").tolist()
    nav = [round(float(v), 4) for v in res["cumulative"]]
    cum = res["cumulative"]
    drawdown = [round(float(v), 4) for v in (cum / cum.cummax() - 1)]
    turnover_by_date = res["daily_returns"].index  # turnover needs recompute align
    # recompute per-day turnover series (same logic as long_short_backtest)
    to_series = []
    prev_long, prev_short = set(), set()
    for date, chunk in preds.groupby("date"):
        k = max(int(len(chunk) * 0.2), 1)
        ranked = chunk.sort_values("pred", ascending=False)
        long_names = set(ranked.head(k)["symbol"])
        short_names = set(ranked.tail(k)["symbol"])
        to_series.append((date, (len(long_names - prev_long) +
                                 len(short_names - prev_short)) / (2 * k)))
        prev_long, prev_short = long_names, short_names
    to_s = pd.Series(dict(to_series)).sort_index()
    to_s = to_s.reindex(res["daily_returns"].index)

    # equal-weight benchmark (long-only) over the same OOS dates
    bench = fwd_panel.loc[nav_dates].mean(axis=1)
    bench_nav = (1 + bench).cumprod()

    # model backend actually used
    import backtest as _B
    backend = "LightGBM" if _B.lgb is not None else "sklearn HistGradientBoosting"

    backtest = {
        "dates": nav_dates,
        "nav": nav,
        "drawdown": drawdown,
        "turnover": [None if pd.isna(v) else round(float(v), 3) for v in to_s],
        "bench_nav": [round(float(v), 4) for v in bench_nav],
        "stats": {
            "ann_return": float(res["ann_return"]), "sharpe": float(res["sharpe"]),
            "max_drawdown": float(res["max_drawdown"]),
            "avg_turnover": float(res["avg_turnover"]),
            "n_days": int(res["n_days"]), "cost_bps": COST_BPS, "backend": backend,
        },
    }

    payload = {
        "meta": {
            "n_stocks": len(codes), "n_days": len(next(iter(raw.values()))),
            "n_factors": len(panels),
            "range": (f"{min(s['dates'][0] for s in stocks.values())} ~ "
                      f"{max(s['dates'][-1] for s in stocks.values())}"),
        },
        "stocks": stocks,
        "factors": factors,
        "factorDefs": FACTOR_DEFS,
        "eval": eval_rows,
        "groupMeans": {
            "factors": list(group_means.keys()),
            "groups": [f"G{i}" for i in range(1, 6)],
            "cells": [[f, f"G{i}", round(float(group_means[f][f"G{i}"]), 3)]
                      for f in group_means for i in range(1, 6)],
        },
        "backtest": backtest,
    }

    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                              separators=(",", ":")))
    out = os.path.join(HERE, "index.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"OK: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
