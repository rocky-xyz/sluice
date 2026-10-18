"""Task 4.6 (tasks.md): assemble the learned source/event-type weights into a
readable report — the latest walk-forward fit (task 4.3/4.4) as the main
table, plus a month-by-month stability check across the walk-forward run.

Sources with fewer than `--min-source-n` rows are not dropped from the fit
(ridge still gives each its own, heavily shrunk coefficient — see
docs/specs/03-news-weights.md), only from this report's main table; they are
summarized in one "其他" row instead, per the task-4.1 decision.

Usage:
    python scripts/news_weight_report.py [--start 2025-10-01] [--min-source-n 20]
"""

import argparse
import csv
from datetime import UTC, date, datetime
from pathlib import Path

from sluice.news.store import NewsStore
from sluice.news.walkforward import Refit, monthly_cutoffs, walk_forward_fit
from sluice.news.weights import WeightEstimate, WeightTable
from sluice.prices.store import PriceStore
from sluice.prices.yahoo import fetch_bars
from sluice.scoring.store import ScoreStore
from sluice.universe import COMPANIES

CONFIG_VERSION = "v1"


def _other_row(estimates: list[WeightEstimate]) -> dict[str, float | int]:
    total_n = sum(e.n for e in estimates)
    return {
        "count": len(estimates),
        "total_n": total_n,
        "weighted_mean": sum(e.weight * e.n for e in estimates) / total_n,
        "min": min(e.weight for e in estimates),
        "max": max(e.weight for e in estimates),
    }


def render_report(
    latest_as_of: datetime, table: WeightTable, refits: list[Refit], min_source_n: int
) -> str:
    out = [f"# 4.6 来源/类型权重表（scoring config {CONFIG_VERSION}，截至 {latest_as_of.date()}）\n"]
    out.append(
        f"λ0 = {table.lambda0}。若等于网格上限，说明这次拟合几乎没有可区分于 0 的信号"
        "（见 docs/specs/03-news-weights.md 的 4.3 已确认小节），权重要读成"
        "“几乎都被收缩到接近 0”，不是可信的点估计。\n"
    )

    out.append("## event_type 权重\n")
    out.append("| event_type | n | weight | 95% CI |")
    out.append("|---|---|---|---|")
    for t, e in sorted(table.event_type.items(), key=lambda kv: -abs(kv[1].weight)):
        out.append(f"| {t} | {e.n} | {e.weight:+.5f} | [{e.low:+.5f}, {e.high:+.5f}] |")
    out.append("")

    out.append(f"## 来源权重（n >= {min_source_n}，按 \\|weight\\| 排序）\n")
    out.append("| source | n | weight | 95% CI |")
    out.append("|---|---|---|---|")
    big = {s: e for s, e in table.source.items() if e.n >= min_source_n}
    for s, e in sorted(big.items(), key=lambda kv: -abs(kv[1].weight)):
        out.append(f"| {s} | {e.n} | {e.weight:+.5f} | [{e.low:+.5f}, {e.high:+.5f}] |")
    small = [e for e in table.source.values() if e.n < min_source_n]
    if small:
        other = _other_row(small)
        out.append(
            f"| 其他（{other['count']} 个来源合并） | {other['total_n']} | "
            f"{other['weighted_mean']:+.5f}（按样本量加权平均） "
            f"| 各自范围 [{other['min']:+.5f}, {other['max']:+.5f}] |"
        )
    out.append("")

    out.append("## walk-forward 稳定性（每月样本量和总惩罚力度）\n")
    out.append("| as_of | n_rows | λ0 |")
    out.append("|---|---|---|")
    for r in refits:
        lam = r.table.lambda0 if r.table else "n/a（样本不足）"
        out.append(f"| {r.as_of.date()} | {r.n_rows} | {lam} |")
    out.append("")
    return "\n".join(out) + "\n"


def write_source_csv(path: Path, table: WeightTable) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "n", "weight", "low", "high"])
        for s, e in sorted(table.source.items(), key=lambda kv: -abs(kv[1].weight)):
            writer.writerow([s, e.n, e.weight, e.low, e.high])


def write_event_type_csv(path: Path, table: WeightTable) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_type", "n", "weight", "low", "high"])
        for t, e in sorted(table.event_type.items(), key=lambda kv: -abs(kv[1].weight)):
            writer.writerow([t, e.n, e.weight, e.low, e.high])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, default=Path("data/news.db"))
    parser.add_argument("--scores-db", type=Path, default=Path("data/local/scores.db"))
    parser.add_argument("--prices-db", type=Path, default=Path("data/local/prices.db"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/local"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 1))
    parser.add_argument("--min-source-n", type=int, default=20)
    args = parser.parse_args()

    now = datetime.now(UTC)
    cutoffs = monthly_cutoffs(datetime.combine(args.start, datetime.min.time(), tzinfo=UTC), now)
    with (
        NewsStore(args.news_db) as news,
        ScoreStore(args.scores_db) as scores,
        PriceStore(args.prices_db, fetch=fetch_bars) as prices,
    ):
        refits = walk_forward_fit(
            news, scores, prices, list(COMPANIES), cutoffs, config_version=CONFIG_VERSION
        )

    fitted = [r for r in refits if r.table is not None]
    if not fitted:
        raise SystemExit("no month had enough resolved rows to fit a weight table")
    latest = fitted[-1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / f"news_weight_report_{CONFIG_VERSION}.md"
    source_csv = args.out_dir / f"news_weights_source_{CONFIG_VERSION}.csv"
    type_csv = args.out_dir / f"news_weights_event_type_{CONFIG_VERSION}.csv"

    report = render_report(latest.as_of, latest.table, refits, args.min_source_n)
    md_path.write_text(report)
    write_source_csv(source_csv, latest.table)
    write_event_type_csv(type_csv, latest.table)

    print(report)
    print(f"写入 {md_path}")
    print(f"写入 {source_csv}")
    print(f"写入 {type_csv}")


if __name__ == "__main__":
    main()
