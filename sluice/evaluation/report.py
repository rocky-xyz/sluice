"""The portfolio evaluation of pipeline configs on one segment (docs/specs/05, task 6.4).

Each config's ratings become a tilt on equal weight (`portfolio.py`), one
portfolio per repeat, rebalanced at the open after each decision day and held
until the close of the next grid decision day after the segment. Baselines run
through the same engine: equal weight rebalanced each period, equal weight
bought once, SPY and QQQ bought once. Every result is given at the main cost
and at the doubled one.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from sluice.agents.batch import decision_days
from sluice.agents.prefetch import backtest_decision_time
from sluice.evaluation import metrics, quality
from sluice.evaluation.backtest import PriceBar, Prices, simulate
from sluice.evaluation.decisions import Decision, load_decisions
from sluice.evaluation.portfolio import target_weights
from sluice.evaluation.split import TEST_1, VALIDATION, Segment, require_open
from sluice.market_calendar import close_of, next_open_after
from sluice.prices.store import PriceStore
from sluice.sizing import plan
from sluice.sizing.config import SizingConfig

COSTS = (0.001, 0.002)  # the main cost and the sensitivity check
BENCHMARKS = ("SPY", "QQQ")
MISSING_BASELINE = "原版 TradingAgents"  # an external file, from task 6.5


@dataclass(frozen=True)
class Calendar:
    decisions: list[date]  # decision days in the segment
    rebalances: list[date]  # the trading day after each, when the trade is made
    days: list[date]  # every trading day simulated
    end: date  # the next grid decision day after the segment: the last period ends at its close


def calendar(requested: Segment, final: bool = False) -> Calendar:
    require_open(requested, final)
    bounds = {"validation": VALIDATION, "test_1": TEST_1}
    if requested not in bounds:
        raise ValueError(f"no portfolio calendar for {requested} yet")
    first, last = bounds[requested]
    grid = decision_days(first, last + timedelta(days=60))
    count = sum(day <= last for day in grid)
    rebalances = [next_open_after(backtest_decision_time(day)) for day in grid[:count]]
    end = grid[count]
    days, day = [], rebalances[0]
    while day <= end:
        if close_of(day) is not None:
            days.append(day)
        day += timedelta(days=1)
    return Calendar(decisions=grid[:count], rebalances=rebalances, days=days, end=end)


def load_prices(store: PriceStore, tickers: list[str], cal: Calendar) -> Prices:
    """Adjusted opens and closes for every simulated day. Refuses a gap."""
    prices: Prices = {}
    missing = []
    for ticker in tickers:
        bars = store.history(ticker, cal.end, limit=len(cal.days) + 10)
        by_day = {bar.day: PriceBar(open=bar.open, close=bar.close) for bar in bars}
        if gaps := [day for day in cal.days if day not in by_day]:
            more = f" and {len(gaps) - 1} more" if gaps[1:] else ""
            missing.append(f"{ticker}: {gaps[0]}{more}")
        prices[ticker] = by_day
    if missing:
        raise ValueError("no prices for " + "; ".join(missing))
    return prices


@dataclass(frozen=True)
class Result:
    name: str
    cost: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    total_return: float
    turnover: float | None  # mean per rebalance after the first purchase; None if there is none
    costs: float
    neutral: int | None = None  # ratings filled in as hold: failed or missing runs


def _result(name: str, cost: float, schedule, cal: Calendar, prices: Prices, neutral=None):
    path = simulate(schedule, cal.days, prices, cost)
    return Result(
        name=name,
        cost=cost,
        sharpe=metrics.sharpe(path.returns),
        sortino=metrics.sortino(path.returns),
        max_drawdown=metrics.max_drawdown(path.values),
        total_return=metrics.total_return(path.values),
        turnover=sum(path.turnover[1:]) / len(path.turnover[1:]) if path.turnover[1:] else None,
        costs=path.costs,
        neutral=neutral,
    )


def rating_schedules(
    decisions: list[Decision], tickers: list[str], cal: Calendar
) -> dict[int, tuple[dict[date, dict[str, float]], int]]:
    """Per repeat: the target weights at each rebalance, and how many ratings were filled in."""
    by_repeat: dict[int, dict[tuple[date, str], Decision]] = defaultdict(dict)
    for decision in decisions:
        by_repeat[decision.repeat][(decision.day, decision.ticker)] = decision
    schedules = {}
    for repeat, cells in sorted(by_repeat.items()):
        schedule, neutral = {}, 0
        for day, rebalance in zip(cal.decisions, cal.rebalances, strict=True):
            ratings = {}
            for ticker in tickers:
                decision = cells.get((day, ticker))
                ratings[ticker] = decision.rating if decision else None
                neutral += ratings[ticker] is None
            schedule[rebalance] = target_weights(ratings)
        schedules[repeat] = (schedule, neutral)
    return schedules


Extra = list[tuple[str, dict[date, dict[str, float]], int | None]]  # name, schedule, filled-in


def evaluate(
    run_dirs: list[Path],
    tickers: list[str],
    store: PriceStore,
    requested: Segment = "validation",
    final: bool = False,
    extra: Extra | None = None,
) -> list[Result]:
    """Each config's rating tilt, then `extra` schedules (from the sizing formula), then baselines."""
    cal = calendar(requested, final)
    prices = load_prices(store, [*tickers, *BENCHMARKS], cal)
    equal = {t: 1 / len(tickers) for t in tickers}
    baselines = {
        "等权，每期再平衡": {day: equal for day in cal.rebalances},
        "等权，买入后不再平衡": {cal.rebalances[0]: equal},
        **{f"{b} 买入持有": {cal.rebalances[0]: {b: 1.0}} for b in BENCHMARKS},
    }
    results = []
    for cost in COSTS:
        for folder in run_dirs:
            decisions = load_decisions(folder, requested, final)
            for repeat, (schedule, neutral) in rating_schedules(decisions, tickers, cal).items():
                name = f"{folder.name} r{repeat}"
                results.append(_result(name, cost, schedule, cal, prices, neutral))
        for name, schedule, neutral in extra or []:
            results.append(_result(name, cost, schedule, cal, prices, neutral))
        for name, schedule in baselines.items():
            results.append(_result(name, cost, schedule, cal, prices))
    return results


def render(
    results: list[Result], cal: Calendar, requested: Segment, missing_baseline: bool = True
) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{x:+.2%}"

    def num(x: float | None) -> str:
        return "-" if x is None else f"{x:.2f}"

    lines = [
        f"# 组合回测：{requested}",
        "",
        (
            f"决策日 {cal.decisions[0]} 至 {cal.decisions[-1]}（{len(cal.decisions)} 期），"
            f"在下一个开盘调仓，最后一期持有到 {cal.end} 收盘；"
            f"共 {len(cal.days)} 个交易日。无风险利率和现金收益取 0。"
        ),
    ]
    for cost in COSTS:
        lines += [
            "",
            f"## 成本 {cost:.1%}（单边，按成交金额）",
            "",
            "| 策略 | Sharpe | Sortino | 最大回撤 | 总收益 | 每期换手（不含建仓） | 累计成本 | 按 hold 补的评级 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in (r for r in results if r.cost == cost):
            lines.append(
                f"| {r.name} | {num(r.sharpe)} | {num(r.sortino)} | {r.max_drawdown:.2%} "
                f"| {pct(r.total_return)} | {num(r.turnover)} | {r.costs:.3%} "
                f"| {'-' if r.neutral is None else r.neutral} |"
            )
        if missing_baseline:
            lines.append(f"| {MISSING_BASELINE} | 缺（推迟到 7.7，只在测试期跑） | | | | | | |")
    return "\n".join(lines) + "\n"


def render_quality(configs: dict[str, list[Decision]], labelled: dict[str, list]) -> str:
    """Decision quality per config and repeat, its calibration and its consistency (task 6.7)."""

    def pct(x: float | None) -> str:
        return "-" if x is None else f"{x:.1%}"

    def num(x: float | None, pattern: str = "{:+.3f}") -> str:
        return "-" if x is None else pattern.format(x)

    lines = [
        "",
        "## 决策质量（不经组合）",
        "",
        (
            f"标签：下一个开盘起 {quality.HORIZON} 个交易日相对 SPY 的对数超额收益。"
            "命中率只算有方向的评级（不含 hold）；Spearman 含 hold。"
        ),
        "",
        "| 配置 | 重复 | 运行 | 失败 | 有标签 | 看多比例 | 看空比例 | 有方向 | 命中率 | Spearman |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in labelled.items():
        repeats = sorted({r.decision.repeat for r in rows})
        for repeat in repeats:
            m = quality.summary([r for r in rows if r.decision.repeat == repeat])
            lines.append(
                f"| {name} | {repeat} | {m.runs} | {m.failed} | {m.labelled} | {pct(m.bullish)} "
                f"| {pct(m.bearish)} | {m.directional} | {pct(m.hit_rate)} | {num(m.spearman)} |"
            )
    lines += ["", "### 校准：研究经理自报的置信度对命中率（只含有方向的评级）", ""]
    lines += ["| 配置 | 置信度 | 次数 | 平均置信度 | 命中率 |", "|---|---|---|---|---|"]
    for name, rows in labelled.items():
        for b in quality.confidence_calibration(rows):
            lines.append(
                f"| {name} | {b.name} | {b.n} | {num(b.mean_confidence, '{:.2f}')} "
                f"| {pct(b.hit_rate)} |"
            )
    lines += ["", "### 校准：各档评级随后的平均超额收益", ""]
    lines += ["| 配置 | 评级 | 次数 | 平均标签 |", "|---|---|---|---|"]
    for name, rows in labelled.items():
        for rating, n, mean in quality.rating_calibration(rows):
            lines.append(f"| {name} | {rating} | {n} | {'-' if mean is None else f'{mean:+.2%}'} |")
    lines += ["", "### 一致性（同一格多次重复）", ""]
    for name, decisions in configs.items():
        c = quality.consistency(decisions)
        if c.cells == 0:
            lines.append(f"- {name}：只有 1 次重复，无法计算（规格 05 要求每格 3 次）。")
        else:
            lines.append(
                f"- {name}：{c.cells} 格，评级完全一致 {pct(c.agree)}，"
                f"评级分数平均标准差 {c.mean_std:.2f}"
            )
    return "\n".join(lines) + "\n"


def build(
    folders: list[Path],
    sized: list[Path],
    tickers: list[str],
    prices: PriceStore,
    src: plan.Sources | None,
    sizing: SizingConfig | None,
    now: datetime,
    requested: Segment = "validation",
    final: bool = False,
) -> str:
    """The whole report: rating tilts of `folders`, the sizing formula on `sized`, the
    formula's two controls, the baselines, and decision quality. Writes nothing."""
    cal = calendar(requested, final)
    extra: Extra = []
    if sizing is not None and src is not None:
        markets = {d: plan.market(src, tickers, d, sizing) for d in cal.decisions}
        rebalances = dict(zip(cal.decisions, cal.rebalances, strict=True))
        label = f"sizing {sizing.version}-{sizing.digest}"
        for folder in sized:
            decisions = load_decisions(folder, requested, final)
            for repeat, (schedule, missing) in plan.formula_schedules(
                decisions, markets, rebalances, src, sizing
            ).items():
                extra.append((f"{folder.name} r{repeat} + {label}", schedule, missing))
        for name, schedule in plan.ablations(markets, rebalances, sizing).items():
            extra.append((f"{name}，{label}", schedule, None))
    results = evaluate(folders, tickers, prices, requested, final, extra)
    decided = {f.name: load_decisions(f, requested, final) for f in folders}
    labelled = {name: quality.label_decisions(ds, prices, now) for name, ds in decided.items()}
    has_external = any(f.name.startswith("tradingagents") for f in folders)
    return render(results, cal, requested, missing_baseline=not has_external) + render_quality(
        decided, labelled
    )
