"""The original TradingAgents against this project's pipelines on the 55 screening
cells, per docs/specs/05-evaluation.md ("已确认（2026-10-27，原版 TradingAgents 小样本对照）").

Decision quality only, no portfolio: five decision days are too few for one.
For each config and repeat, on the cells of the evolution screen (validation
decision days 1, 4, 7, 10 and 13, every company):
- the share of bullish, hold and bearish ratings;
- the hit rate (non-hold ratings whose sign matches the label) with its Wilson
  95% interval;
- the Spearman of rating score and label within each decision day, averaged
  over the days;
- the Spearman of rating score and the stock's excess return over SPY in the
  10 trading days up to the decision close: how much a rating restates the
  recent move;
- how often the original's rating equals each repeat's on the same cell.

The label is the frozen one of sluice.evaluation.quality. Only validation runs
are read (load_decisions refuses the sealed segments).

Usage:
    python -m scripts.compare_baseline [--config v1 --config v3] [--out ...]
"""

import argparse
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from sluice.agents.batch import run_dir
from sluice.agents.config import load_config
from sluice.evaluation import quality
from sluice.evaluation.decisions import load_decisions
from sluice.evaluation.report import calendar
from sluice.evolution.pipeline_gate import SCREEN_DAYS
from sluice.prices.store import PriceStore
from sluice.prices.yahoo import fetch_bars
from sluice.universe import COMPANIES

EXTERNAL = "tradingagents-1394a3f"
LOOKBACK = 10  # trading days of past excess return, the same length as the label
Z = 1.96  # two-sided 95%


@dataclass(frozen=True)
class Row:
    ticker: str
    day: date
    repeat: int
    rating: str | None  # None when the run failed
    label: float | None  # None until known
    past: float | None  # excess return over SPY in the LOOKBACK days to the decision close


@dataclass(frozen=True)
class Metrics:
    runs: int
    failed: int
    bullish: float | None
    hold: float | None
    bearish: float | None
    directional: int
    hits: int
    hit_rate: float | None
    interval: tuple[float, float] | None
    daily_spearman: float | None  # mean over days that have one
    days: int  # days with a daily Spearman
    past_spearman: float | None


def wilson(hits: int, n: int) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; stays inside [0, 1] for small n."""
    if n == 0:
        return None
    p = hits / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return centre - half, centre + half


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Rank correlation; None with fewer than 3 pairs or no spread on either side."""
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return None
    xs, ys = zip(*pairs, strict=True)
    return statistics.correlation(quality._rank(list(xs)), quality._rank(list(ys)))


def metrics(rows: list[Row]) -> Metrics:
    ok = [r for r in rows if r.rating is not None]
    scores = [quality.SCORES[r.rating] for r in ok]

    def share(test) -> float | None:
        return sum(map(test, scores)) / len(scores) if scores else None

    known = [(quality.SCORES[r.rating], r.label, r) for r in ok if r.label is not None]
    directional = [(s, y) for s, y, _ in known if s != 0]
    hits = sum((s > 0) == (y > 0) for s, y in directional)
    by_day = defaultdict(list)
    for s, y, r in known:
        by_day[r.day].append((s, y))
    daily = [rho for pairs in by_day.values() if (rho := spearman(pairs)) is not None]
    return Metrics(
        runs=len(rows),
        failed=len(rows) - len(ok),
        bullish=share(lambda s: s > 0),
        hold=share(lambda s: s == 0),
        bearish=share(lambda s: s < 0),
        directional=len(directional),
        hits=hits,
        hit_rate=hits / len(directional) if directional else None,
        interval=wilson(hits, len(directional)),
        daily_spearman=statistics.mean(daily) if daily else None,
        days=len(daily),
        past_spearman=spearman(
            [(quality.SCORES[r.rating], r.past) for r in ok if r.past is not None]
        ),
    )


def agreement(a: list[Row], b: list[Row]) -> tuple[int, float | None]:
    """(cells, share with the same rating) over cells where both runs succeeded."""
    rated = {(r.ticker, r.day): r.rating for r in b if r.rating is not None}
    both = [
        (r.rating, rated[r.ticker, r.day]) for r in a if r.rating and (r.ticker, r.day) in rated
    ]
    return len(both), sum(x == y for x, y in both) / len(both) if both else None


def past_excess(prices: PriceStore, ticker: str, day: date) -> float | None:
    """Log return over SPY from the close LOOKBACK trading days back to `day`'s close."""

    def log_return(symbol: str) -> float | None:
        bars = prices.history(symbol, through=day, limit=LOOKBACK + 1)
        if len(bars) < LOOKBACK + 1 or bars[-1].day != day:
            return None
        return math.log(bars[-1].close / bars[0].close)

    stock, spy = log_return(ticker), log_return("SPY")
    return None if stock is None or spy is None else stock - spy


def _pct(x: float | None, signed: bool = False) -> str:
    return "-" if x is None else f"{x:+.1%}" if signed else f"{x:.1%}"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:+.3f}"


def render(
    table: dict[tuple[str, int], Metrics],
    agree: dict[tuple[str, int], tuple[int, float | None]],
    days: list[date],
    tickers: list[str],
) -> str:
    lines = [
        "# 原版 TradingAgents 对照：初筛 55 格",
        "",
        (
            f"决策日 {', '.join(map(str, days))} × {len(tickers)} 只股票。"
            "标签：下一个开盘起 10 个交易日相对 SPY 的对数超额收益。"
            "命中率只算有方向的评级（不含 hold），区间为 Wilson 95%；"
            "每日 Spearman 在同一决策日内算，再对各天取平均；"
            f'"追随过去" 是评级与决策日收盘前 {LOOKBACK} 个交易日相对 SPY 超额的 Spearman。'
        ),
        "",
        "原版在历史日期取不到新闻（规格 05），它的判断只来自技术面和基本面。",
        "",
        (
            "| 配置 | 重复 | 运行 | 失败 | 看多 | hold | 看空 | 有方向 | 命中率 | 95% 区间 "
            "| 每日 Spearman（天数） | 追随过去 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (name, repeat), m in table.items():
        interval = "-" if m.interval is None else f"{m.interval[0]:.0%}–{m.interval[1]:.0%}"
        lines.append(
            f"| {name} | {repeat} | {m.runs} | {m.failed} | {_pct(m.bullish)} | {_pct(m.hold)} "
            f"| {_pct(m.bearish)} | {m.directional} | {_pct(m.hit_rate)} | {interval} "
            f"| {_num(m.daily_spearman)}（{m.days}） | {_num(m.past_spearman)} |"
        )
    lines += [
        "",
        "## 与原版同格评级一致的比例",
        "",
        "| 配置 | 重复 | 两边都成功的格 | 评级相同 |",
        "|---|---|---|---|",
    ]
    for (name, repeat), (n, share) in agree.items():
        lines.append(f"| {name} | {repeat} | {n} | {_pct(share)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", help="Pipeline config; may repeat.")
    parser.add_argument("--runs", type=Path, default=Path("data/local/runs"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/local/evaluation/baseline-screen.md")
    )
    args = parser.parse_args()

    validation_days = calendar("validation").decisions
    days = [validation_days[i] for i in SCREEN_DAYS]
    tickers = list(COMPANIES)
    folders = {EXTERNAL: args.runs / EXTERNAL} | {
        name: run_dir(args.runs, load_config(name)) for name in args.config or ["v1", "v3"]
    }
    now = datetime.now(UTC)
    rows: dict[tuple[str, int], list[Row]] = defaultdict(list)
    with PriceStore(fetch=fetch_bars) as prices:
        past = {(t, d): past_excess(prices, t, d) for t in tickers for d in days}
        for name, folder in folders.items():
            decisions = [
                d
                for d in load_decisions(folder, "validation")
                if d.day in days and d.ticker in tickers
            ]
            for item in quality.label_decisions(decisions, prices, now):
                d = item.decision
                rows[name, d.repeat].append(
                    Row(d.ticker, d.day, d.repeat, d.rating, item.label, past[d.ticker, d.day])
                )
    table = {key: metrics(rs) for key, rs in sorted(rows.items(), key=lambda kv: kv[0])}
    external = rows.get((EXTERNAL, 0), [])
    agree = {key: agreement(external, rs) for key, rs in rows.items() if key[0] != EXTERNAL}
    report = render(table, agree, days, tickers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
