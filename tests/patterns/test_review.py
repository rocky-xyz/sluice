import pytest

from sluice.patterns.candles import candles
from sluice.patterns.detect import cuorou, wedge
from sluice.patterns.review import Example, chart, choose, margin, page
from tests.patterns.test_detect import RULES, calm, cuorou_bars, wedge_bars


def test_the_margin_is_the_slack_over_the_closest_threshold():
    series = candles(cuorou_bars())
    (event,) = cuorou(series, RULES.cuorou)
    # day 2's volume is 600/900 = 0.667 of day 1's, the limit 0.7: 5% to spare, the least
    assert margin(event, series, RULES) == pytest.approx(0.7 / (600 / 900) - 1)
    tight = candles(cuorou_bars(v2=629.0))
    (just,) = cuorou(tight, RULES.cuorou)
    assert 0 <= margin(just, tight, RULES) < 0.002


def examples(n: int, pattern_margins) -> list[Example]:
    series = candles(cuorou_bars())
    (event,) = cuorou(series, RULES.cuorou)
    return [
        Example(f"T{i}", event, series, m, "")
        for i, m in zip(range(n), pattern_margins, strict=True)
    ]


def test_choose_takes_the_four_closest_calls_and_a_fixed_draw_of_the_rest():
    found = examples(30, [i / 100 for i in range(30)])
    first = choose(found, seed=1)
    assert [e.ticker for e in first if e.pick == "borderline"] == ["T0", "T1", "T2", "T3"]
    assert len(first) == 10 and len({e.ticker for e in first}) == 10
    assert [e.ticker for e in choose(found, seed=1)] == [e.ticker for e in first]
    assert [e.ticker for e in choose(found, seed=2)] != [e.ticker for e in first]


def test_a_wedge_chart_draws_both_lines_and_the_breakout_level():
    bars, _ = wedge_bars([(21.6, 21.5), (21.7, 21.55), (21.8, 21.65)])
    series = candles(bars)
    candidate, confirmed = wedge(series, calm(bars), RULES.wedge)
    ex = Example("IP", candidate, series, margin(candidate, series, RULES), "random")
    svg = chart(ex, {("IP", candidate.marks[0]): confirmed.marks[1]})
    assert svg.count('class="trend"') == 2 and svg.count('class="level"') == 1
    assert svg.count('class="mark"') == 2  # the breakout and the confirmation


def test_the_page_asks_for_verdicts_and_shows_no_later_returns():
    found = examples(12, [i / 100 for i in range(12)])
    text = page(choose(found), RULES, {}, ["搓揉线：共 12 次"])
    assert text.count('<section class="card"') == 10
    assert text.count('value="算"') == 10 and "生成结果" in text
    assert "超额" not in text and "收益" not in text  # the definition, not the payoff
