import json
from datetime import UTC, date, datetime

import pytest

from scripts.ab_news_weighting import (
    Run,
    agreement,
    hit_rate,
    load_runs,
    metrics,
    paired,
    render,
    spearman,
)

CLOSE = datetime(2026, 3, 3, 21, 0, tzinfo=UTC)


def run(ticker="NVDA", day=date(2026, 3, 3), repeat=0, rating="overweight", **flags) -> Run:
    return Run(
        ticker=ticker,
        day=day,
        repeat=repeat,
        decision_time=CLOSE,
        rating=rating,
        report_cites_signal=flags.get("report", False),
        evidence_cites_signal=flags.get("evidence", False),
    )


def test_hold_is_left_out_of_the_hit_rate():
    pairs = [(2, 0.01), (-1, 0.02), (0, 0.05), (1, -0.01), (-2, -0.03)]
    assert hit_rate(pairs) == (4, 0.5)
    assert hit_rate([(0, 0.01)]) == (0, None)


def test_spearman_includes_hold_and_needs_spread():
    assert spearman([(-1, -0.02), (0, 0.0), (1, 0.03)]) == pytest.approx(1.0)
    assert spearman([(0, -0.02), (0, 0.0), (0, 0.03)]) is None  # every rating the same
    assert spearman([(1, 0.01), (2, 0.02)]) is None  # too few


def test_metrics_skip_failed_runs_and_unfinished_labels():
    runs = [
        run("NVDA", rating="buy"),
        run("AMD", rating=None),
        run("KO", rating="sell"),
        run("MSFT", rating="hold"),
    ]
    labels = {("NVDA", date(2026, 3, 3)): 0.02, ("KO", date(2026, 3, 3)): 0.01}
    m = metrics(runs, labels)
    assert (m.runs, m.failed, m.labelled, m.directional, m.hit_rate) == (4, 1, 2, 2, 0.5)
    assert m.ratings == {"buy": 1, "sell": 1, "hold": 1}


def test_pairing_keeps_only_cells_both_configs_finished():
    a = [run("NVDA"), run("AMD", rating=None), run("KO")]
    b = [run("NVDA", rating="hold"), run("AMD"), run("MSFT")]
    pa, pb = paired(a, b)
    assert [r.ticker for r in pa] == [r.ticker for r in pb] == ["NVDA"]
    assert (pa[0].rating, pb[0].rating) == ("overweight", "hold")


def test_agreement_counts_cells_whose_repeats_all_finished():
    runs = [
        run("NVDA", repeat=0), run("NVDA", repeat=1),  # agree
        run("AMD", repeat=0), run("AMD", repeat=1, rating="hold"),  # disagree
        run("KO", repeat=0), run("KO", repeat=1, rating=None),  # a failed repeat: left out
        run("TSM", repeat=0),  # a single repeat: left out
    ]  # fmt: skip
    assert agreement(runs) == (2, 0.5)
    assert agreement([run()]) == (0, None)


def _save(folder, ticker, day, repeat, rating, news="Up [N1].", refs=("N1",)):
    path = folder / ticker / f"{day}_r{repeat}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = [{"claim": "x", "direction": "bullish", "refs": list(refs)}]
    decision = None if rating is None else {"rating": rating, "evidence": evidence}
    state = {
        "meta": {"decision_time": "2026-03-03T21:00:00Z"},
        "reports": {"news": {"text": news}},
        "decision": decision,
    }
    path.write_text(json.dumps(state))
    return path


def test_test_period_states_are_counted_but_never_opened(tmp_path):
    _save(tmp_path, "NVDA", "2026-03-03", 0, "buy")
    sealed = tmp_path / "NVDA" / "2026-08-10_r0.json"
    sealed.write_text("not json: opening it would fail")
    runs, skipped = load_runs(tmp_path)
    assert [(r.ticker, r.day, r.rating) for r in runs] == [("NVDA", date(2026, 3, 3), "buy")]
    assert skipped == 1


def test_citations_of_the_news_signal_are_read_from_report_and_evidence(tmp_path):
    _save(tmp_path, "NVDA", "2026-03-03", 0, "buy", "Flow leans up [N1, S:news_signal].",
          refs=("S:news_signal",))  # fmt: skip
    _save(tmp_path, "AMD", "2026-03-03", 0, None)
    runs, _ = load_runs(tmp_path)
    by_ticker = {r.ticker: r for r in runs}
    assert by_ticker["NVDA"].report_cites_signal and by_ticker["NVDA"].evidence_cites_signal
    assert by_ticker["AMD"].rating is None


def test_a_single_repeat_reports_no_noise_estimate():
    a, b = [run()], [run(report=True)]
    report = render(("v1", "v2"), (a, b), {("NVDA", date(2026, 3, 3)): 0.01})
    assert "只有 1 次重复" in report
    assert "新闻报告引用 S:news_signal 的 100.0%" in report
