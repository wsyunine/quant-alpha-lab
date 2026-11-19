"""Cross-sectional evaluation: Spearman IC, ICIR, t-stat, quantile group backtest."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_NAMES = 5  # minimum symbols with valid data per date to compute cross-sectional IC


def daily_ic(factor_panel: pd.DataFrame, fwd_panel: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman IC between factor and forward return."""
    ics = {}
    common_idx = factor_panel.index.intersection(fwd_panel.index)
    for date in common_idx:
        f = factor_panel.loc[date]
        r = fwd_panel.loc[date]
        mask = f.notna() & r.notna()
        if mask.sum() >= MIN_NAMES:
            ics[date] = spearmanr(f[mask], r[mask])[0]
    return pd.Series(ics).sort_index()


def ic_summary(ic: pd.Series) -> dict:
    """Mean IC, ICIR, t-stat, positive-day ratio."""
    if ic.empty:
        return {}
    mean, std = ic.mean(), ic.std()
    icir = mean / std if std > 0 else np.nan
    return {
        "IC_mean": round(mean, 4),
        "ICIR": round(icir, 3),
        "t_stat": round(icir * np.sqrt(len(ic)), 2) if std > 0 else np.nan,
        "IC>0_ratio": round((ic > 0).mean(), 3),
        "n_days": len(ic),
    }


def group_backtest(factor_panel: pd.DataFrame, fwd_panel: pd.DataFrame,
                   n_groups: int = 5) -> pd.DataFrame:
    """Quantile group backtest: equal-weight mean forward return per group per day.

    Group 1 = lowest factor values, group n = highest.
    Returns DataFrame indexed by date with columns G1..Gn and LS (long-short = top - bottom).
    """
    common_idx = factor_panel.index.intersection(fwd_panel.index)
    rows = []
    for date in common_idx:
        f = factor_panel.loc[date]
        r = fwd_panel.loc[date]
        mask = f.notna() & r.notna()
        if mask.sum() < n_groups:
            continue
        labels = pd.qcut(f[mask].rank(method="first"), n_groups, labels=False) + 1
        row = {f"G{g}": r[mask][labels == g].mean() for g in range(1, n_groups + 1)}
        row["LS"] = row[f"G{n_groups}"] - row["G1"]
        rows.append((date, row))
    return pd.DataFrame({d: r for d, r in rows}).T.sort_index()


def evaluate_all(panels: dict, fwd_panel: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """Summary table across all factors: IC stats + annualized long-short Sharpe."""
    summary = {}
    for name, panel in panels.items():
        ic = daily_ic(panel, fwd_panel)
        stat = ic_summary(ic)
        groups = group_backtest(panel, fwd_panel, n_groups)
        if not groups.empty:
            ls = groups["LS"]
            stat["LS_sharpe_ann"] = round(
                ls.mean() / ls.std() * np.sqrt(243), 3) if ls.std() > 0 else np.nan  # A-share: ~243 trading days/year
        summary[name] = stat
    return pd.DataFrame(summary).T.sort_values("ICIR", ascending=False)
