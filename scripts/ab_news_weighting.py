"""Task 5.3 (tasks.md): unweighted (pipeline v1) against weighted news (v2) on the
validation period, per docs/specs/07-agent-pipeline.md ("已确认（任务 5.3）").

Each run's rating becomes a score (buy +2 .. sell -2) and is compared with the
stock's log excess return over SPY from the next open through the close of the
10th trading day. Per config and per repeat: hit rate (non-hold ratings whose
sign matches the label) and Spearman rank correlation (hold included). Repeats
are compared with each other to show run-to-run noise; with a single repeat
there is no noise estimate and the report says so.

This is a preliminary check, not the frozen evaluation of step 6. Until the
test period is sealed in code (task 6.3), this script never opens a run decided
on or after the first test-period day (docs/specs/05-evaluation.md); it only
counts them.

Usage:
    python scripts/ab_news_weighting.py [--a v1] [--b v2] [--runs data/local/runs]
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from scripts.news_label_correlation import _rank
from sluice.agents.batch import run_dir
from sluice.agents.config import load_config
from sluice.agents.state import cited_refs
from sluice.news.labels import excess_return_label
from sluice.prices.store import PriceStore
from sluice.prices.yahoo import fetch_bars

VALIDATION_FIRST = date(2026, 1, 2)
TEST_FIRST = date(2026, 8, 10)  # first day of test period one; never read here
HORIZON = 10  # trading days: one two-week decision period
SCORES = {"buy": 2, "overweight": 1, "hold": 0, "underweight": -1, "sell": -2}
SIGNAL_REF = "S:news_signal"


@dataclass(frozen=True)
class Run:
    ticker: str
    day: date
    repeat: int
    decision_time: datetime
    rating: str | None  # None when the run failed
    report_cites_signal: bool
    evidence_cites_signal: bool


def load_runs(folder: Path) -> tuple[list[Run], int]:
    """Validation-period states of one config, and how many test-period files were skipped.

    A test-period file is recognised by its name and never opened.
    """
    runs, sealed = [], 0
    for path in sorted(folder.glob("*/*.json")):
        day_text, repeat = path.stem.split("_r")
        day = date.fromisoformat(day_text)
        if day >= TEST_FIRST:
            sealed += 1
            continue
        if day < VALIDATION_FIRST:
            continue
        state = json.loads(path.read_text())
        decision = state["decision"] or {}
        news_report = (state["reports"].get("news") or {}).get("text", "")
        evidence_refs = {ref for e in decision.get("evidence", []) for ref in e["refs"]}
        runs.append(
            Run(
                ticker=path.parent.name,
                day=day,
                repeat=int(repeat),
                decision_time=datetime.fromisoformat(state["meta"]["decision_time"]),
                rating=decision.get("rating"),
                report_cites_signal=SIGNAL_REF in cited_refs(news_report),
                evidence_cites_signal=SIGNAL_REF in evidence_refs,
            )
        )
    return runs, sealed


def hit_rate(pairs: list[tuple[int, float]]) -> tuple[int, float | None]:
    """Among non-zero scores: the share whose sign matches the label's. (count, rate)."""
    directional = [(s, y) for s, y in pairs if s != 0]
    if not directional:
        return 0, None
    hits = sum((s > 0) == (y > 0) for s, y in directional)
    return len(directional), hits / len(directional)


def spearman(pairs: list[tuple[int, float]]) -> float | None:
    """Rank correlation; None when either side has no spread."""
    if len(pairs) < 3:
        return None
    xs, ys = _rank([s for s, _ in pairs]), _rank([y for _, y in pairs])
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


@dataclass(frozen=True)
class Metrics:
    runs: int
    failed: int
    labelled: int  # successful runs with a finished label
    directional: int
    hit_rate: float | None
    spearman: float | None
    ratings: dict[str, int]


def metrics(runs: list[Run], labels: dict[tuple[str, date], float | None]) -> Metrics:
    ok = [r for r in runs if r.rating is not None]
    pairs = [
        (SCORES[r.rating], label)
        for r in ok
        if (label := labels.get((r.ticker, r.day))) is not None
    ]
    directional, rate = hit_rate(pairs)
    return Metrics(
        runs=len(runs),
        failed=len(runs) - len(ok),
        labelled=len(pairs),
        directional=directional,
        hit_rate=rate,
        spearman=spearman(pairs),
        ratings=dict(Counter(r.rating for r in ok)),
    )


def by_repeat(runs: list[Run]) -> dict[int, list[Run]]:
    grouped: dict[int, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[run.repeat].append(run)
    return dict(sorted(grouped.items()))


def paired(a: list[Run], b: list[Run]) -> tuple[list[Run], list[Run]]:
    """Only the (ticker, day, repeat) cells that succeeded under both configs."""
    ok_a = {(r.ticker, r.day, r.repeat): r for r in a if r.rating is not None}
    ok_b = {(r.ticker, r.day, r.repeat): r for r in b if r.rating is not None}
    common = sorted(ok_a.keys() & ok_b.keys())
    return [ok_a[k] for k in common], [ok_b[k] for k in common]


def agreement(runs: list[Run]) -> tuple[int, float | None]:
    """Cells with every repeat successful and more than one: share whose ratings all agree."""
    cells: dict[tuple[str, date], list[Run]] = defaultdict(list)
    for run in runs:
        cells[(run.ticker, run.day)].append(run)
    full = [
        cell for cell in cells.values()
        if len(cell) > 1 and all(r.rating is not None for r in cell)
    ]  # fmt: skip
    if not full:
        return 0, None
    same = sum(len({r.rating for r in cell}) == 1 for cell in full)
    return len(full), same / len(full)


def _fmt(value: float | None, pattern: str = "{:.3f}") -> str:
    return "-" if value is None else pattern.format(value)


def render(
    names: tuple[str, str], runs: tuple[list[Run], list[Run]], labels: dict, sealed: int = 0
) -> str:
    lines = [
        f"# 5.3 A/B：{names[0]}（未加权）对 {names[1]}（加权）",
        "",
        (
            f"验证期 {VALIDATION_FIRST} 起、{TEST_FIRST} 之前；"
            f"标签为下一个开盘起 {HORIZON} 个交易日相对 SPY 的对数超额收益。"
            f"测试期的状态文件 {sealed} 个，未打开。"
        ),
        "",
        "## 每个配置、每次重复",
        "",
        "| 配置 | 重复 | 运行 | 失败 | 有标签 | 有方向 | 命中率 | Spearman | 评级分布 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    repeats_seen = set()
    for name, config_runs in zip(names, runs, strict=True):
        for repeat, group in by_repeat(config_runs).items():
            repeats_seen.add(repeat)
            m = metrics(group, labels)
            ratings = ", ".join(f"{k} {v}" for k, v in sorted(m.ratings.items()))
            lines.append(
                f"| {name} | {repeat} | {m.runs} | {m.failed} | {m.labelled} | {m.directional} "
                f"| {_fmt(m.hit_rate, '{:.1%}')} | {_fmt(m.spearman)} | {ratings} |"
            )

    pa, pb = paired(*runs)
    ma, mb = metrics(pa, labels), metrics(pb, labels)
    lines += [
        "",
        f"## 配对比较（两个配置都成功的 {len(pa)} 格）",
        "",
        "| 配置 | 命中率 | Spearman |",
        "|---|---|---|",
        f"| {names[0]} | {_fmt(ma.hit_rate, '{:.1%}')} | {_fmt(ma.spearman)} |",
        f"| {names[1]} | {_fmt(mb.hit_rate, '{:.1%}')} | {_fmt(mb.spearman)} |",
        "",
        "## 过程指标",
        "",
    ]
    for name, config_runs in zip(names, runs, strict=True):
        n, share = agreement(config_runs)
        lines.append(f"- {name}：多次重复评级完全一致的格 {_fmt(share, '{:.1%}')}（{n} 格）")
    ok_b = [r for r in runs[1] if r.rating is not None]
    if ok_b:
        report_share = sum(r.report_cites_signal for r in ok_b) / len(ok_b)
        evidence_share = sum(r.evidence_cites_signal for r in ok_b) / len(ok_b)
        lines.append(
            f"- {names[1]}：新闻报告引用 {SIGNAL_REF} 的 {report_share:.1%}，"
            f"研究经理证据引用的 {evidence_share:.1%}"
        )
    if len(repeats_seen) < 2:
        lines += ["", "只有 1 次重复：无法估计运行间噪声，不能判断差异是否超过噪声。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="v1", help="Unweighted pipeline config.")
    parser.add_argument("--b", default="v2", help="Weighted pipeline config.")
    parser.add_argument("--runs", type=Path, default=Path("data/local/runs"))
    parser.add_argument("--prices-db", type=Path, default=Path("data/local/prices.db"))
    parser.add_argument("--out", type=Path, default=Path("data/local/ab_news_weighting.md"))
    args = parser.parse_args()

    names = (args.a, args.b)
    loaded = [load_runs(run_dir(args.runs, load_config(name))) for name in names]
    runs = (loaded[0][0], loaded[1][0])
    sealed = loaded[0][1] + loaded[1][1]
    now = datetime.now(UTC)
    with PriceStore(args.prices_db, fetch=fetch_bars) as prices:
        labels = {
            (r.ticker, r.day): excess_return_label(r.ticker, r.decision_time, HORIZON, prices, now)
            for config_runs in runs
            for r in config_runs
        }
    report = render(names, runs, labels, sealed)
    args.out.write_text(report)
    print(report)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
