# quant-alpha-lab

多因子 Alpha 挖掘研究管线（学习/求职项目）。数据源：A 股（Tushare Pro）。

## 流水线

```
数据下载 (Tushare daily, 前复权) → 因子计算 (10+ 量价因子) → 截面 IC/ICIR 评价
→ 分位数分组回测 → walk-forward GBDT 多空策略 (含交易成本)
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 下载 A 股日线（token 从 tushare.pro 个人主页获取）
export TUSHARE_TOKEN=你的token
python src/data_tushare.py --start 20230101 --end 20260101
#   --force: 清空缓存重新下载

# 2. 无网络冒烟测试（合成数据，不调任何 API）
python src/run_pipeline.py --synthetic

# 3. 用真实 A 股缓存跑全链路
python src/run_pipeline.py
#   或指定股票子集：
python src/run_pipeline.py --symbols 000001.SZ 600519.SH 300750.SZ
```

## 目录结构

- `src/data_tushare.py` — Tushare 日线下载、复权（adj_factor 优先，积分不足时用 pct_chg 重建）、CSV 缓存
- `src/data.py` — 数据质量检查 + 合成数据生成（离线冒烟测试用）
- `src/factors.py` — 因子库：动量/反转/波动/量价/流动性类
- `src/evaluate.py` — 截面 Spearman IC、ICIR、t 统计量、分位数分组回测
- `src/backtest.py` — walk-forward GBDT 预测 + 多空组合 + 成本
- `src/run_pipeline.py` — 一键运行入口

## GBDT 后端说明

优先使用 LightGBM；若 macOS 上未安装 `libomp` 导致 LightGBM 无法加载，
自动回退到 sklearn 的 `HistGradientBoostingRegressor`（同类算法，结果可直接对比）。
想用 LightGBM 时执行 `brew install libomp` 即可。

## 已知简化（诚实声明，面试可讲）

- 股票池为少数现存成分股 → 截面过窄、幸存者偏差；扩展方向：按交易日拉全市场、用历史成分股回溯
- 日频、等权、简单成本模型 → 扩展方向：滑点模型、T+1 与涨跌停约束、协方差风控
- 复权：积分不足时用 pct_chg 重建（有两位小数舍入误差）；有 2000 积分后建议切换 adj_factor
