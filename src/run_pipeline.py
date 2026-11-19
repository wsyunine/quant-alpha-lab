"""One-shot pipeline entry: data -> factors -> IC evaluation -> walk-forward backtest."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import data as D
import factors as F
import evaluate as E
import backtest as B
import data_tushare as T

DEFAULT_SYMBOLS = None  # A-share defaults live in data_tushare.DEMO_CODES


def main():
    ap = argparse.ArgumentParser(description="multi-factor alpha mining pipeline (A-share / Tushare)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (offline test)")
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="ts_codes; defaults to data_tushare.DEMO_CODES")
    ap.add_argument("--train-window", type=int, default=180)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()

    if args.synthetic:
        print("[data] generating synthetic panel ...")
        syms = args.symbols or T.DEMO_CODES
        raw = D.make_synthetic([s.split(".")[0] + "-SYN" for s in syms])
    else:
        codes = args.symbols or T.DEMO_CODES
        print(f"[data] loading A-share panel (Tushare cache): {codes}")
        raw = T.load_cached(codes)
        for sym, df in raw.items():
            D.check_data(df, sym)

    print("\n[stage 1] computing factors ...")
    panels, fwd_panel = F.build_panels(raw)

    print("\n[stage 2] cross-sectional IC / ICIR / group backtest ...")
    summary = E.evaluate_all(panels, fwd_panel)
    print(summary.to_string())

    print(f"\n[stage 3] walk-forward LightGBM long-short "
          f"(train_window={args.train_window}, step={args.step}, cost={args.cost_bps}bps) ...")
    long_df = B.stack_long(panels, fwd_panel)
    preds = B.walk_forward_predict(long_df, F.FACTOR_NAMES,
                                   train_window=args.train_window, step=args.step)
    if preds.empty:
        print("not enough history for walk-forward; increase data or reduce train-window")
    else:
        res = B.long_short_backtest(preds, fwd_panel, cost_bps=args.cost_bps)
        print(f"out-of-sample days : {res['n_days']}")
        print(f"annualized return  : {res['ann_return']:.2%}")
        print(f"sharpe (ann)       : {res['sharpe']}")
        print(f"max drawdown       : {res['max_drawdown']:.2%}")
        print(f"avg daily turnover : {res['avg_turnover']:.1%}")

    print("\ndone. next: write report/week1_report.md per the TODO doc.")


if __name__ == "__main__":
    main()
