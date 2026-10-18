from datetime import UTC, date, datetime

from sluice.market_calendar import close_of, next_open_after, nth_trading_day
from sluice.news.models import NewsItem
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.news.walkforward import monthly_cutoffs, walk_forward_fit
from sluice.prices.store import Bar, PriceStore
from sluice.scoring.store import ItemScores, ScoreStore


def test_monthly_cutoffs_spans_whole_months_across_a_year_boundary():
    start = datetime(2025, 11, 15, tzinfo=UTC)  # mid-month: still counts November
    end = datetime(2026, 2, 3, tzinfo=UTC)  # early Feb: still counts February
    assert monthly_cutoffs(start, end) == [
        datetime(2025, 11, 1, tzinfo=UTC),
        datetime(2025, 12, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
    ]


def test_monthly_cutoffs_of_a_single_month_is_one_cutoff():
    assert monthly_cutoffs(datetime(2026, 6, 5, tzinfo=UTC), datetime(2026, 6, 20, tzinfo=UTC)) == [
        datetime(2026, 6, 1, tzinfo=UTC)
    ]


def _score(**overrides) -> ItemScores:
    fields = {
        "relevance": 0.9,
        "stance": "bullish",
        "stance_confidence": 0.8,
        "stance_probabilities": {"bullish": 0.7, "bearish": 0.1, "neutral": 0.2},
        "novelty": 0.5,
        "rumour": 0.1,
        "event_type": "earnings",
        "event_type_confidence": 0.8,
        "event_type_probabilities": {"earnings": 0.8},
    }
    fields.update(overrides)
    return ItemScores(**fields)


def test_walk_forward_fit_only_sees_each_month_what_had_resolved_by_then(tmp_path):
    cutoff1 = datetime(2026, 2, 1, tzinfo=UTC)
    cutoff2 = datetime(2026, 3, 1, tzinfo=UTC)

    early_published = datetime(2025, 12, 1, 11, 0, tzinfo=ET)
    late_published = datetime(2026, 1, 28, 11, 0, tzinfo=ET)

    # Confirm the premise this test relies on, rather than assuming hand-picked
    # dates land where intended: `late`'s label must still be open at cutoff1
    # but resolved by cutoff2; `early`'s must already be resolved at cutoff1.
    def resolves_by(published, cutoff) -> bool:
        entry = next_open_after(published)
        exit_day = nth_trading_day(entry, 3)
        return close_of(exit_day) <= cutoff

    assert resolves_by(early_published, cutoff1)
    assert not resolves_by(late_published, cutoff1)
    assert resolves_by(late_published, cutoff2)

    with (
        NewsStore(tmp_path / "news.db") as news,
        ScoreStore(tmp_path / "scores.db") as scores,
        PriceStore(tmp_path / "prices.db") as prices,
    ):
        news.add(
            [
                NewsItem(
                    title="Early item",
                    source="yfinance",
                    tickers=["NVDA"],
                    published_at=early_published,
                    fetched_at=early_published,
                ),
                NewsItem(
                    title="Late item",
                    source="yfinance",
                    tickers=["NVDA"],
                    published_at=late_published,
                    fetched_at=late_published,
                ),
            ]
        )
        for item in news.query("NVDA", before=cutoff2):
            scores.put(item.id, "v1", "digest", _score())

        for published in (early_published, late_published):
            entry_day = next_open_after(published)
            exit_day = nth_trading_day(entry_day, 3)
            for ticker in ("NVDA", "SPY"):
                existing = prices.history(ticker, date(2026, 12, 31), limit=10_000)
                bars = {b.day: b for b in existing}
                bars[entry_day] = Bar(day=entry_day, open=100.0, close=100.0, raw_close=100.0, volume=1.0)
                bars[exit_day] = Bar(day=exit_day, open=100.0, close=101.0, raw_close=101.0, volume=1.0)
                prices.replace(ticker, sorted(bars.values(), key=lambda b: b.day))

        refits = walk_forward_fit(
            news, scores, prices, ["NVDA"], [cutoff1, cutoff2], min_rows=2
        )

    assert [r.as_of for r in refits] == [cutoff1, cutoff2]
    assert refits[0].n_rows == 1  # only the early item has resolved
    assert refits[1].n_rows == 2  # the late item has joined by cutoff2
    assert refits[0].table is None  # below min_rows
    assert refits[1].table is not None


def test_a_cutoff_below_min_rows_gets_no_table(tmp_path):
    cutoff = datetime(2026, 2, 1, tzinfo=UTC)
    published = datetime(2025, 12, 1, 11, 0, tzinfo=ET)

    with (
        NewsStore(tmp_path / "news.db") as news,
        ScoreStore(tmp_path / "scores.db") as scores,
        PriceStore(tmp_path / "prices.db") as prices,
    ):
        news.add(
            [NewsItem(title="Solo item", source="yfinance", tickers=["NVDA"], published_at=published, fetched_at=published)]
        )
        (item,) = news.query("NVDA", before=cutoff)
        scores.put(item.id, "v1", "digest", _score())

        entry_day = next_open_after(published)
        exit_day = nth_trading_day(entry_day, 3)
        for ticker in ("NVDA", "SPY"):
            prices.replace(
                ticker,
                [
                    Bar(day=entry_day, open=100.0, close=100.0, raw_close=100.0, volume=1.0),
                    Bar(day=exit_day, open=100.0, close=101.0, raw_close=101.0, volume=1.0),
                ],
            )

        (refit,) = walk_forward_fit(news, scores, prices, ["NVDA"], [cutoff], min_rows=2)

    assert refit.n_rows == 1
    assert refit.table is None
