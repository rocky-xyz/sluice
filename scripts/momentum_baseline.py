"""Are the pipeline's ratings mostly a restatement of the recent move? (H1)

docs/specs/11-momentum-baseline.md. A one-line momentum baseline (the stock's
log excess return over SPY in the N trading days to the decision close) is set
beside each config's ratings on the validation segment:
- restatement: Spearman of rating score and the 10-day past excess;
- forward Spearman of the rating and of the baseline against the label, and
  their difference;
- hit rate, the baseline cut into bullish / hold / bearish in the same shares
  as the config's ratings;
- partial correlation of rating and label once the past excess is taken out
  of both, ranks regressed on its rank within each unit.

A unit is one decision day and one repeat. Correlations are taken within a
unit, over its stocks, then averaged over units. Intervals resample whole
decision days, 1000 times; the two sides of a difference share the draws.
The label is the frozen one of sluice.evaluation.quality, and only validation
runs are read (load_decisions refuses the sealed segments). Prices are read
from the stores as they are, never fetched.

Beyond the validation runs: over the S&P 500 price store of the pattern study,
on every 10th SPY session from 2016, the cross-sectional Spearman of the past
excess against the same label (next open to the 10th session's close, as the
pattern study's), to see whether short-term reversal is general. Only
decision days whose label settles before test segment 1 opens are used, and no
bar from that day on is read. The members are today's (survivors).

Usage:
    python -m scripts.momentum_baseline [--config v1 --config v2 --config v3] [--out ...]
"""

import argparse
import math
import statistics
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from scripts.compare_baseline import EXTERNAL, past_excess, spearman
from sluice.agents.batch import run_dir
from sluice.agents.config import load_config
from sluice.evaluation import quality
from sluice.evaluation.decisions import load_decisions
from sluice.evaluation.split import TEST_1, VALIDATION
from sluice.patterns import universe
from sluice.patterns.prices import DEFAULT_DB as PATTERN_DB
from sluice.patterns.study import _log_excess
from sluice.prices.store import Bar, PriceStore
from sluice.universe import COMPANIES

WINDOWS = (5, 10, 20)  # trading days of past excess; 10 is the label's own length
MAIN = 10
RESAMPLES = 1000
MIN_STOCKS = 3  # fewer in a unit and its correlations are not taken
CONFIGS = ("v1", "v2", "v3")
STEP = 10  # sessions between decision days of the S&P 500 check: labels do not overlap
CUTOFF = TEST_1[0]  # labels there must settle before test segment 1 opens
MIN_UNIVERSE = 30  # fewer stocks on a day of the whole index and the day is left out


@dataclass(frozen=True)
class Cell:
    ticker: str
    day: date
    repeat: int
    score: int  # quality.SCORES of the rating
    label: float
    past: dict[int, float] = field(hash=False)  # window -> past excess


@dataclass(frozen=True)
class Counts:
    runs: int
    failed: int
    unlabelled: int  # successful runs whose label is not known
    no_past: int  # successful, labelled runs missing a past excess in some window


def load_cells(
    folder: Path,
    prices: PriceStore,
    now: datetime,
    past: Callable[[str, date, int], float | None],
) -> tuple[list[Cell], Counts]:
    """The validation runs of one run directory that have a rating, a label and every past."""
    decisions = load_decisions(folder, "validation")
    cells, failed, unlabelled, no_past = [], 0, 0, 0
    for item in quality.label_decisions(decisions, prices, now):
        d = item.decision
        if d.rating is None:
            failed += 1
            continue
        if item.label is None:
            unlabelled += 1
            continue
        values = {n: past(d.ticker, d.day, n) for n in WINDOWS}
        if any(v is None for v in values.values()):
            no_past += 1
            continue
        cells.append(Cell(d.ticker, d.day, d.repeat, quality.SCORES[d.rating], item.label, values))
    return cells, Counts(len(decisions), failed, unlabelled, no_past)


def units(cells: list[Cell]) -> dict[tuple[date, int], list[Cell]]:
    grouped: dict[tuple[date, int], list[Cell]] = defaultdict(list)
    for c in cells:
        grouped[c.day, c.repeat].append(c)
    return dict(sorted(grouped.items()))


# A measure is kept as (numerator, denominator) per decision day, so that a
# resample of days only has to add them up. A mean over units is (sum of the
# units' values, number of units with a value); a hit rate is (hits, ratings
# with a direction).
Parts = dict[date, tuple[float, int]]


def _add(parts: Parts, day: date, num: float, den: int) -> None:
    old = parts.get(day, (0.0, 0))
    parts[day] = (old[0] + num, old[1] + den)


def unit_mean(cells: list[Cell], value: Callable[[list[Cell]], float | None]) -> Parts:
    """Every day's (sum, count) of a per-unit value; a unit without one is left out, not 0."""
    parts: Parts = {}
    for (day, _), group in units(cells).items():
        v = value(group) if len(group) >= MIN_STOCKS else None
        _add(parts, day, 0.0 if v is None else v, 0 if v is None else 1)
    return parts


def rank_correlation(x: Callable[[Cell], float], y: Callable[[Cell], float]):
    return lambda group: spearman([(x(c), y(c)) for c in group])


def score(c: Cell) -> float:
    return c.score


def label(c: Cell) -> float:
    return c.label


def past_of(n: int) -> Callable[[Cell], float]:
    return lambda c: c.past[n]


def _residuals(ys: list[float], xs: list[float]) -> list[float] | None:
    """Least-squares residuals of ys on a line in xs; None when xs do not vary."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return [y - my - slope * (x - mx) for x, y in zip(xs, ys, strict=True)]


def partial(n: int) -> Callable[[list[Cell]], float | None]:
    """Rank correlation of rating and label once the rank of the n-day past is taken out."""

    def value(group: list[Cell]) -> float | None:
        rp = quality._rank([c.past[n] for c in group])
        rs = _residuals(quality._rank([c.score for c in group]), rp)
        ry = _residuals(quality._rank([c.label for c in group]), rp)
        if rs is None or ry is None:
            return None
        if max(map(abs, rs)) < 1e-12 or max(map(abs, ry)) < 1e-12:
            return None  # one side is all explained by the past: nothing left to correlate
        return statistics.correlation(rs, ry)

    return value


def matched_tiers(cells: list[Cell], n: int) -> dict[Cell, int]:
    """The baseline's direction per cell: the highest n-day past bullish, the lowest bearish,
    in the same numbers as the ratings; ties are settled by ticker, day and repeat."""
    bullish = sum(c.score > 0 for c in cells)
    bearish = sum(c.score < 0 for c in cells)
    order = sorted(cells, key=lambda c: (-c.past[n], c.ticker, c.day, c.repeat))
    tiers = dict.fromkeys(order, 0)
    for c in order[:bullish]:
        tiers[c] = 1
    for c in order[len(order) - bearish :]:
        tiers[c] = -1
    return tiers


def hits(cells: list[Cell], direction: Callable[[Cell], int]) -> Parts:
    """Every day's (hits, ratings with a direction): hold is left out, as in quality."""
    parts: Parts = {}
    for c in cells:
        if (s := direction(c)) != 0:
            _add(parts, c.day, float((s > 0) == (c.label > 0)), 1)
    return parts


def value(parts: Parts) -> float | None:
    num = sum(n for n, _ in parts.values())
    den = sum(d for _, d in parts.values())
    return num / den if den else None


@dataclass(frozen=True)
class Estimate:
    value: float | None
    low: float | None = None  # 95% interval by resampling decision days
    high: float | None = None


@dataclass(frozen=True)
class Paired:
    llm: Estimate
    baseline: Estimate
    difference: Estimate  # llm - baseline, the two from the same draws


def _arrays(parts: Parts, days: list[date]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([parts.get(d, (0.0, 0))[0] for d in days]),
        np.array([parts.get(d, (0.0, 0))[1] for d in days]),
    )


def _draws(days: list[date], seed: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    return rng.integers(0, len(days), size=(RESAMPLES, len(days)))


def _resampled(parts: Parts, days: list[date], picks: np.ndarray) -> np.ndarray:
    """The measure in every draw; NaN in a draw with nothing to divide by."""
    nums, dens = _arrays(parts, days)
    num, den = nums[picks].sum(axis=1), dens[picks].sum(axis=1)
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)


def _interval(draws: np.ndarray) -> tuple[float | None, float | None]:
    if np.isnan(draws).all():
        return None, None
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def estimate(parts: Parts, days: list[date], seed: str | None) -> Estimate:
    """The measure, with its interval unless `seed` is None."""
    if seed is None:
        return Estimate(value(parts))
    return Estimate(value(parts), *_interval(_resampled(parts, days, _draws(days, seed))))


def paired(llm: Parts, baseline: Parts, days: list[date], seed: str | None) -> Paired:
    a, b = value(llm), value(baseline)
    gap = None if a is None or b is None else a - b
    if seed is None:
        return Paired(Estimate(a), Estimate(b), Estimate(gap))
    picks = _draws(days, seed)
    da, db = _resampled(llm, days, picks), _resampled(baseline, days, picks)
    return Paired(
        Estimate(a, *_interval(da)), Estimate(b, *_interval(db)), Estimate(gap, *_interval(da - db))
    )


@dataclass(frozen=True)
class Result:
    name: str
    counts: Counts
    cells: int
    units: int  # day x repeat
    days: int
    directional: int  # ratings that are not hold; the baseline has as many
    restatement: Estimate
    forward: dict[int, Paired]
    hit_rate: dict[int, Paired]
    partial: dict[int, Estimate]


def analyse(name: str, cells: list[Cell], counts: Counts, intervals: bool = True) -> Result:
    days = sorted({c.day for c in cells})

    def seed(measure: str) -> str | None:
        return f"{name}|{measure}" if intervals else None

    forward, hit_rate, part = {}, {}, {}
    rated = hits(cells, lambda c: c.score)
    for n in WINDOWS:
        forward[n] = paired(
            unit_mean(cells, rank_correlation(score, label)),
            unit_mean(cells, rank_correlation(past_of(n), label)),
            days,
            seed(f"forward|{n}"),
        )
        tiers = matched_tiers(cells, n)
        hit_rate[n] = paired(rated, hits(cells, tiers.__getitem__), days, seed(f"hits|{n}"))
        part[n] = estimate(unit_mean(cells, partial(n)), days, seed(f"partial|{n}"))
    return Result(
        name=name,
        counts=counts,
        cells=len(cells),
        units=len(units(cells)),
        days=len(days),
        directional=sum(c.score != 0 for c in cells),
        restatement=estimate(
            unit_mean(cells, rank_correlation(score, past_of(MAIN))), days, seed("restatement")
        ),
        forward=forward,
        hit_rate=hit_rate,
        partial=part,
    )


def _num(e: Estimate, pct: bool = False) -> str:
    if e.value is None:
        return "-"
    fmt = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:+.3f}")
    if e.low is None or e.high is None:
        return fmt(e.value)
    return f"{fmt(e.value)}（{fmt(e.low)} 至 {fmt(e.high)}）"


def _counts(r: Result) -> str:
    c = r.counts
    return (
        f"运行 {c.runs}，失败 {c.failed}，标签未知 {c.unlabelled}，缺过去超额 {c.no_past}；"
        f"用到 {r.cells} 格、{r.units} 个单元、{r.days} 个决策日，有方向的评级 {r.directional} 个。"
    )


def render_validation(results: list[Result], skipped: list[str]) -> list[str]:
    lines = [
        "# H1：评级是否只是在复述最近的涨跌",
        "",
        (
            "验证期。标签：下一个开盘起 10 个交易日相对 SPY 的对数超额收益。"
            "动量基线：决策日收盘前 N 个交易日相对 SPY 的对数超额。"
            "单元 = 一个决策日 × 一次重复；相关在单元内的股票之间算，再对单元取平均。"
            f"括号里是按决策日整天重抽样 {RESAMPLES} 次的 95% 区间。"
        ),
        "",
    ]
    if skipped:
        lines += [f"跳过（没有验证期运行）：{', '.join(skipped)}", ""]
    for r in results:
        f, h = r.forward[MAIN], r.hit_rate[MAIN]
        lines += [
            f"## {r.name}",
            "",
            _counts(r),
            "",
            "| 度量 | LLM 评级 | 动量基线（10 日） | 差值（LLM − 基线） |",
            "|---|---|---|---|",
            f"| 复述程度：评级 vs 过去 10 日超额 | {_num(r.restatement)} | | |",
            f"| 向前 Spearman | {_num(f.llm)} | {_num(f.baseline)} | {_num(f.difference)} |",
            (
                f"| 命中率（基线按比例分档） | {_num(h.llm, True)} | {_num(h.baseline, True)} "
                f"| {_num(h.difference, True)} |"
            ),
            f"| 偏相关：去掉过去 10 日超额后 | {_num(r.partial[MAIN])} | | |",
            "",
        ]
    lines += [
        "## 稳健性：过去 5 / 10 / 20 日",
        "",
        "| 配置 | 窗口 | 基线向前 Spearman | LLM − 基线 | 基线命中率 | 偏相关 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for n in WINDOWS:
            f, h = r.forward[n], r.hit_rate[n]
            lines.append(
                f"| {r.name} | {n} | {_num(f.baseline)} | {_num(f.difference)} "
                f"| {_num(h.baseline, True)} | {_num(r.partial[n])} |"
            )
    return lines + [""]


def render_external(r: Result) -> list[str]:
    f, h = r.forward[MAIN], r.hit_rate[MAIN]
    return [
        "## 原版 TradingAgents（初筛 5 天，只给点估计）",
        "",
        _counts(r),
        "",
        "| 度量 | LLM 评级 | 动量基线（10 日） |",
        "|---|---|---|",
        f"| 复述程度 | {_num(r.restatement)} | |",
        f"| 向前 Spearman | {_num(f.llm)} | {_num(f.baseline)} |",
        f"| 命中率 | {_num(h.llm, True)} | {_num(h.baseline, True)} |",
        f"| 偏相关 | {_num(r.partial[MAIN])} | |",
        "",
    ]


# The S&P 500 check.

Point = tuple[dict[int, float], float]  # past excess per window, label


def decision_days(sessions: list[date]) -> list[date]:
    """Every STEP-th session, from the first with the longest past window behind it."""
    return sessions[max(WINDOWS) :: STEP]


def points(
    bars: list[Bar], spy: dict[date, Bar], days: list[date], cutoff: date
) -> dict[date, Point]:
    """One stock's past excess and label on each decision day it has the data for."""
    index = {b.day: i for i, b in enumerate(bars)}
    found = {}
    for day in days:
        i = index.get(day)
        if i is None or i < max(WINDOWS):
            continue
        labelled = _log_excess(bars, spy, i, quality.HORIZON)
        if labelled is None or labelled[0] >= cutoff:
            continue
        past = {}
        for n in WINDOWS:
            start, end = bars[i - n], bars[i]
            s_start, s_end = spy.get(start.day), spy.get(end.day)
            if s_start is None or s_end is None or min(start.close, s_start.close) <= 0:
                break
            past[n] = math.log(end.close / start.close) - math.log(s_end.close / s_start.close)
        else:
            found[day] = (past, labelled[1])
    return found


def cross_sections(
    by_ticker: dict[str, dict[date, Point]], days: list[date], minimum: int
) -> dict[int, Parts]:
    """Per window, each decision day's Spearman of past excess and label over the stocks."""
    parts: dict[int, Parts] = {n: {} for n in WINDOWS}
    for day in days:
        here = [found[day] for found in by_ticker.values() if day in found]
        for n in WINDOWS:
            rho = spearman([(past[n], y) for past, y in here]) if len(here) >= minimum else None
            parts[n][day] = (0.0, 0) if rho is None else (rho, 1)
    return parts


def load_points(
    store: PriceStore, tickers: list[str], cutoff: date
) -> tuple[list[date], dict[str, dict[date, Point]]]:
    """Decision days and every ticker's points, from bars before `cutoff` only."""
    through = cutoff - timedelta(days=1)
    spy_bars = store.history("SPY", through, 4000)
    spy = {b.day: b for b in spy_bars}
    days = decision_days([b.day for b in spy_bars])
    found = {t: points(store.history(t, through, 4000), spy, days, cutoff) for t in tickers}
    return days, {t: f for t, f in found.items() if f}


@dataclass(frozen=True)
class Period:
    name: str
    days: int  # decision days with a value
    stocks: float  # mean stocks per such day
    windows: dict[int, Estimate]


def periods(
    parts: dict[int, Parts], by_ticker: dict[str, dict[date, Point]], label: str
) -> list[Period]:
    """The whole span, each year, and the validation segment."""
    spans = [("全部", lambda d: True)]
    years = sorted({d.year for d, (_, n) in parts[MAIN].items() if n})
    spans += [(str(y), lambda d, y=y: d.year == y) for y in years]
    spans.append(("验证期", lambda d: VALIDATION[0] <= d <= VALIDATION[1]))
    rows = []
    for name, keep in spans:
        valued = [d for d, (_, n) in parts[MAIN].items() if n and keep(d)]
        if not valued:
            continue
        windows = {
            n: estimate(
                {d: v for d, v in parts[n].items() if keep(d)}, valued, f"sp500|{label}|{name}|{n}"
            )
            for n in WINDOWS
        }
        stocks = statistics.mean(sum(d in f for f in by_ticker.values()) for d in valued)
        rows.append(Period(name, len(valued), stocks, windows))
    return rows


def render_sp500(tables: dict[str, list[Period]], missing: list[str], cutoff: date) -> list[str]:
    lines = [
        "## 标普 500：过去超额与之后 10 日超额（短期反转是否普遍）",
        "",
        (
            f"从 2016 年起每隔 {STEP} 个交易日取一个决策日；每个决策日在股票之间算"
            "过去 N 日超额与之后 10 日超额（下一个开盘到第 10 个交易日收盘，减 SPY）的 Spearman，"
            "再对决策日取平均，括号里是按决策日重抽样的 95% 区间。负数表示短期反转。"
            f"只用标签在 {cutoff} 之前结算的决策日。成分股是 2026-10 的名单（幸存者偏差）。"
        ),
        "",
    ]
    if missing:
        lines += [f"价格库里没有：{', '.join(missing)}", ""]
    for name, rows in tables.items():
        lines += [
            f"### {name}",
            "",
            "| 时期 | 决策日 | 平均股票数 | 过去 5 日 | 过去 10 日 | 过去 20 日 |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            cells = " | ".join(_num(r.windows[n]) for n in WINDOWS)
            lines.append(f"| {r.name} | {r.days} | {r.stocks:.0f} | {cells} |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", help="Pipeline config; may repeat.")
    parser.add_argument("--runs", type=Path, default=Path("data/local/runs"))
    parser.add_argument("--out", type=Path, default=Path("data/local/evaluation/h1-momentum.md"))
    parser.add_argument("--patterns-db", type=Path, default=PATTERN_DB)
    parser.add_argument("--universe", default="sp500-2026-10")
    args = parser.parse_args()

    now = datetime.now(UTC)
    results, skipped, external = [], [], None
    with PriceStore() as prices:  # no fetch: read what is stored, skip what is not
        cache: dict[tuple[str, date, int], float | None] = {}

        def past(ticker: str, day: date, n: int) -> float | None:
            if (ticker, day, n) not in cache:
                cache[ticker, day, n] = past_excess(prices, ticker, day, n)
            return cache[ticker, day, n]

        for name in args.config or CONFIGS:
            folder = run_dir(args.runs, load_config(name))
            cells, counts = load_cells(folder, prices, now, past)
            if not cells:
                skipped.append(folder.name)
                continue
            results.append(analyse(folder.name, cells, counts))
        if (args.runs / EXTERNAL).is_dir():
            cells, counts = load_cells(args.runs / EXTERNAL, prices, now, past)
            if cells:
                external = analyse(EXTERNAL, cells, counts, intervals=False)
    lines = render_validation(results, skipped)
    lines += render_external(external) if external else [f"{EXTERNAL}：没有运行结果。", ""]
    if args.patterns_db.exists():
        with PriceStore(args.patterns_db) as store:
            members = universe.load(args.universe)  # TSM, an ADR, is not one of them
            days, found = load_points(store, sorted({*members, *COMPANIES}), CUTOFF)
        everyone = {t: f for t, f in found.items() if t in members}
        ours = {t: f for t, f in found.items() if t in COMPANIES}
        tables = {
            f"全部（{len(everyone)} 只）": periods(
                cross_sections(everyone, days, MIN_UNIVERSE), everyone, "all"
            ),
            f"项目的 {len(ours)} 只": periods(cross_sections(ours, days, MIN_STOCKS), ours, "ours"),
        }
        missing = sorted(set(COMPANIES) - set(ours))
        lines += render_sp500(tables, missing, CUTOFF)
    else:
        lines += [f"标普 500 价格库不存在：{args.patterns_db}", ""]
    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
