"""Task 3.6 (tasks.md): a preliminary check of how each Jev score dimension
correlates with next-open-based excess returns, before task 4.1 commits to a
regression feature set for the weight-learning step.

One row per scored news item (not per ticker): a news item tagged to several
of the tracked companies gets one row, whose label is the equal-weighted
average of those companies' own excess-return labels (see docs/decisions/ for
why — the Jev score itself never sees which ticker it is about, so splitting
one score across several "independent" rows would overstate the sample).

Usage:
    python scripts/news_label_correlation.py [--out-dir data/local]
"""

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sluice.news.labels import item_label
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from sluice.prices.yahoo import fetch_bars
from sluice.scoring.store import ScoreStore
from sluice.universe import COMPANIES

HORIZONS = (1, 3, 5)
CONFIG_VERSION = "v1"
CONTINUOUS_FIELDS = ("relevance", "novelty", "rumour", "stance_score")


def stance_score(probabilities: dict[str, float]) -> float:
    """A signed stand-in for the categorical stance, for correlation only."""
    return probabilities.get("bullish", 0.0) - probabilities.get("bearish", 0.0)


class _CachedPrices:
    """Memoizes `(ticker, day)` bar lookups: many items share an entry or exit day."""

    def __init__(self, inner: PriceStore) -> None:
        self._inner = inner
        self._cache: dict[tuple[str, object], object] = {}

    def get(self, ticker, day):
        key = (ticker, day)
        if key not in self._cache:
            self._cache[key] = self._inner.get(ticker, day)
        return self._cache[key]


def collect_rows(
    news: NewsStore, scores: ScoreStore, prices, now: datetime
) -> list[dict[str, object]]:
    rows = []
    for news_id, score in scores.iter_all(CONFIG_VERSION):
        item = news.get(news_id)
        if item is None or item.published_at is None:
            continue
        tickers = sorted(set(item.tickers) & set(COMPANIES))
        if not tickers:
            continue
        row: dict[str, object] = {
            "news_id": news_id,
            "tickers": "+".join(tickers),
            "published_at": item.published_at.isoformat(),
            "relevance": score.relevance,
            "stance": score.stance,
            "stance_score": stance_score(score.stance_probabilities),
            "novelty": score.novelty,
            "rumour": score.rumour,
            "event_type": score.event_type,
        }
        for horizon in HORIZONS:
            row[f"label_{horizon}d"] = item_label(tickers, item.published_at, horizon, prices, now)
        rows.append(row)
    return rows


def _rank(values: list[float]) -> list[float]:
    """Average rank, ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def correlations(rows: list[dict], field: str, horizon: int) -> tuple[int, float | None, float | None]:
    pairs = [(r[field], r[f"label_{horizon}d"]) for r in rows if r[f"label_{horizon}d"] is not None]
    if len(pairs) < 2:
        return len(pairs), None, None
    xs, ys = zip(*pairs, strict=True)
    pearson = statistics.correlation(xs, ys)
    spearman = statistics.correlation(_rank(list(xs)), _rank(list(ys)))
    return len(pairs), pearson, spearman


def event_type_breakdown(rows: list[dict], horizon: int) -> list[tuple[str, int, float | None]]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        label = r[f"label_{horizon}d"]
        if label is not None:
            by_type[r["event_type"]].append(label)
    return sorted(
        (
            (event_type, len(labels), statistics.fmean(labels) if labels else None)
            for event_type, labels in by_type.items()
        ),
        key=lambda row: -row[1],
    )


def render_report(rows: list[dict]) -> str:
    out = [f"# 3.6 初步相关性检查（scoring config {CONFIG_VERSION}）\n"]
    out.append(f"已打分新闻中可定位到标的的：{len(rows)} 条（标签随窗口陆续结清，各周期的样本量见下）。\n")

    out.append("## 连续分数与超额收益标签的相关系数\n")
    out.append("| 分数维度 | 周期 | n | Pearson | Spearman |")
    out.append("|---|---|---|---|---|")
    for field in CONTINUOUS_FIELDS:
        for horizon in HORIZONS:
            n, pearson, spearman = correlations(rows, field, horizon)
            p = f"{pearson:.4f}" if pearson is not None else "n/a"
            s = f"{spearman:.4f}" if spearman is not None else "n/a"
            out.append(f"| {field} | {horizon}日 | {n} | {p} | {s} |")
    out.append("")

    out.append("## event_type 分组的平均超额收益（3 日）\n")
    out.append("| event_type | n | 平均 3 日超额收益（对数） |")
    out.append("|---|---|---|")
    for event_type, n, mean in event_type_breakdown(rows, 3):
        out.append(f"| {event_type} | {n} | {mean:.4f} |" if mean is not None else f"| {event_type} | {n} | n/a |")
    out.append("")
    return "\n".join(out) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, default=Path("data/news.db"))
    parser.add_argument("--scores-db", type=Path, default=Path("data/local/scores.db"))
    parser.add_argument("--prices-db", type=Path, default=Path("data/local/prices.db"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/local"))
    args = parser.parse_args()

    now = datetime.now(UTC)
    with (
        NewsStore(args.news_db) as news,
        ScoreStore(args.scores_db) as scores,
        PriceStore(args.prices_db, fetch=fetch_bars) as prices,
    ):
        rows = collect_rows(news, scores, _CachedPrices(prices), now)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / f"news_label_correlation_{CONFIG_VERSION}.md"
    csv_path = args.out_dir / f"news_label_correlation_{CONFIG_VERSION}.csv"
    report = render_report(rows)
    md_path.write_text(report)
    write_csv(rows, csv_path)

    print(report)
    print(f"写入 {md_path}")
    print(f"写入 {csv_path}")


if __name__ == "__main__":
    main()
