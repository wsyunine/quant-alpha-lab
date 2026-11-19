# quant-alpha-lab

多因子 Alpha 挖掘研究管线（学习 / 求职项目）。数据源：A 股（Tushare Pro）。

## 流水线

```
数据下载 (Tushare daily, 前复权) → 因子计算 (10 个量价因子) → 截面 IC/ICIR 评价
→ 分位数分组回测 → walk-forward GBDT 多空策略 (含交易成本) → 可视化 dashboard
```

## 快速开始

```bash
# 准备环境（推荐 venv，避免 Homebrew Python 的 PEP 668 报错）
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. 下载 A 股日线（token 从 tushare.pro 个人主页获取）
export TUSHARE_TOKEN=你的token
python src/data_tushare.py --start 20230101 --end 20260101
#   --force: 清空旧缓存重新下载（复权算法升级后需要重跑）

# 2. 无网络冒烟测试（合成数据，不调任何 API）
python src/run_pipeline.py --synthetic

# 3. 用真实 A 股缓存跑全链路
python src/run_pipeline.py
#   指定股票子集：
python src/run_pipeline.py --symbols 000001.SZ 600519.SH 300750.SZ

# 4. 生成可视化 dashboard（自包含 index.html，浏览器直接打开）
python dashboard/build_dashboard.py
#   产物：dashboard/index.html
```

## 目录结构

- `src/data_tushare.py` — Tushare 日线下载、前复权（adj_factor 优先；积分不足时用 `pre_close` 精确重建复权因子）、CSV 缓存
- `src/data.py` — 数据质量检查（缺日 / 异常涨跌幅）+ 合成数据生成（离线冒烟测试用）
- `src/factors.py` — 因子库：动量 / 反转 / 波动率 / 量价 / 流动性类共 10 个
- `src/evaluate.py` — 截面 Spearman IC、ICIR、t 统计量、分位数分组回测
- `src/backtest.py` — walk-forward GBDT 预测 + 多空组合 + 名单法换手成本
- `src/run_pipeline.py` — 一键运行入口（stage 1–3）
- `dashboard/build_dashboard.py` — 从 pipeline 真实输出构建自包含可视化页面
- `dashboard/template.html` — dashboard 模板（ECharts）
- `notes/` — 学习笔记（不上传，已在 .gitignore 中）

## Dashboard

四个标签页，全部基于 pipeline 真实计算结果，数据以 JSON 形式内嵌进 HTML：

1. **数据** — 10 只股票前复权 K 线 + 成交量（可缩放拖动）+ 最近交易日明细表
2. **因子** — 单因子单股票时序折线 + 最近一天截面快照（每行一只股票）
3. **评价** — 汇总表（IC / ICIR / t / LS Sharpe，按 |t| 排序）+ t 值与 ICIR 条形图 + 分组回测热力图（因子 × G1–G5 日均收益，观察单调性）
4. **回测** — 净值曲线（策略多空 vs 等权基准）+ 回撤 + 日换手率 + 指标卡

数据更新后只需重跑 `python dashboard/build_dashboard.py`，页面自动刷新。

## GBDT 后端

优先使用 LightGBM；若 macOS 上未安装 `libomp` 导致 LightGBM 无法加载，
自动回退到 sklearn 的 `HistGradientBoostingRegressor`（同类算法，结果可直接对比）。
想用 LightGBM 时执行 `brew install libomp` 即可。

## 年化口径

A 股约 243 个交易日 / 年。`evaluate.py` 的 LS Sharpe 与 `backtest.py` 的年化收益 / Sharpe
均使用 `√243` / `×243` 口径（非加密货币的 365）。

## 已知简化

- 股票池为 10 只现存成分股 → 截面过窄（多空各仅 2 只）、存在幸存者偏差；
  扩展方向：按交易日 `pro.daily(trade_date=...)` 拉全市场、用历史成分股回溯
- 日频、等权、简单成本模型（单边 5bps、名单法换手）；
  扩展方向：权重型换手、滑点 / 冲击成本、T+1 与涨跌停约束、协方差风控
- 因子无预处理；扩展方向：去极值（MAD）→ 标准化 → 行业 / 市值中性化
- 因子无相关性分析；扩展方向：N×N 相关性矩阵聚类，簇内选代表防冗余

## 演进路线（向《101 Formulaic Alphas》量级扩展）

1. **因子注册表**：`factors.py` 从硬编码改为装饰器注册，每个因子带元数据（名称 / 定义 / 参数），新增因子只需追加一个函数
2. **算子 DSL**：实现 `ts_corr / ts_rank / decay_linear / scale / rank / sign` 等向量化算子，101 Alphas 每个写成一行业达式字符串即可批量生成
3. **标准化研究流水线**：因子预处理三件套（去极值 / 标准化 / 中性化）+ 多进程批量评价 + 结果落盘 parquet
4. **可视化升级**：因子动物园散点图（IC_mean × ICIR，点大小 = |t|）+ IC 时序 / 累计 IC + 因子相关性热力图 + 可搜索汇总表

> 务实顺序：**先扩池到全市场，再加因子**。10 只股票的截面上再多因子 IC 也都不可信。
