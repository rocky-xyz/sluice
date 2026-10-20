"""A page of examples for the user to judge each pattern's definition by eye (task 10.5).

docs/specs/09-patterns.md: for each pattern, 6 examples drawn at random (a fixed
seed, so the page can be made again) and the 4 that passed by the smallest
margin, which show best whether a threshold sits where the user means it. Each
is a candlestick chart with volume, the rule's bars shaded and a wedge's lines
drawn. The page lets the user mark "counts / does not count" with a note and
gives back the verdicts as text to paste. It shows no returns after the event:
this is about the definition, not about whether the pattern pays.
"""

import html
import json
import math
import random
from dataclasses import dataclass
from datetime import date

from sluice.patterns.candles import Candle
from sluice.patterns.config import PatternConfig
from sluice.patterns.detect import PatternEvent

PER_PATTERN = 10
BORDERLINE = 4
SEED = 1
NAMES = {
    "cuorou": "搓揉线",
    "harami_daily": "孕线（日线）",
    "harami_weekly": "孕线（周线）",
    "wedge": "楔形突破",
}
BEFORE, AFTER = 30, 10  # bars around a two-bar pattern


@dataclass(frozen=True)
class Example:
    ticker: str
    event: PatternEvent
    series: list[Candle]  # the daily or weekly candles it was found in
    margin: float  # how far past its closest threshold, as a share; small is borderline
    pick: str  # "random" or "borderline"


def _at(series: list[Candle], day: date) -> int:
    return next(i for i, c in enumerate(series) if c.day == day)


def _ratio(have: float | None, need: float) -> float:
    if have is None:
        return math.inf
    return have / need if need else math.inf


def margin(event: PatternEvent, series: list[Candle], config: PatternConfig) -> float:
    """The smallest slack over the thresholds the event passed, minus 1 (0 = right on one)."""
    i = _at(series, event.marks[-1] if event.pattern != "wedge" else event.marks[0])
    if event.pattern == "cuorou":
        r, a, b = config.cuorou, series[i - 1], series[i]
        slack = [
            _ratio(a.upper, r.shadow_to_body * a.body),
            _ratio(a.upper, r.shadow_to_range * a.range),
            _ratio(b.lower, r.shadow_to_body * b.body),
            _ratio(b.lower, r.shadow_to_range * b.range),
            _ratio(r.max_volume_vs_prev, b.vol_prev),
            _ratio(r.max_volume_vs_avg5, a.vol_avg5),
            _ratio(r.max_volume_vs_avg5, b.vol_avg5),
            _ratio(r.max_abs_ret20, abs(b.ret20) if b.ret20 else 0.0),
        ]
    elif event.pattern.startswith("harami"):
        r, mother, baby = config.harami, series[i - 1], series[i]
        big = (
            _ratio(mother.body, r.mother_body_to_avg * mother.avg_body20)
            if mother.bearish and mother.avg_body20
            else 0.0
        )
        shadow = _ratio(mother.lower, r.mother_lower_to_range * mother.range)
        slack = [max(big, shadow), _ratio(r.baby_to_mother_range * mother.range, baby.range)]
    else:
        r, c = config.wedge, series[i]
        level = event.levels["breakout_level"]
        slack = [_ratio(c.close, level * (1 + r.breakout)), _ratio(c.vol_avg20, r.breakout_volume)]
    return min(slack) - 1


def choose(examples: list[Example], seed: int = SEED) -> list[Example]:
    """For each pattern: the borderline few, then a fixed random draw from the rest."""
    chosen = []
    for pattern in NAMES:
        found = [e for e in examples if e.event.pattern == pattern]
        if pattern == "wedge":
            found = [e for e in found if e.event.stage == "candidate"]
        edge = sorted(found, key=lambda e: e.margin)[:BORDERLINE]
        rest = [e for e in found if e not in edge]
        draw = random.Random(f"{seed}-{pattern}").sample(
            rest, min(PER_PATTERN - len(edge), len(rest))
        )
        chosen += [_as(e, "borderline") for e in edge] + [_as(e, "random") for e in draw]
    return chosen


def _as(example: Example, pick: str) -> Example:
    return Example(example.ticker, example.event, example.series, example.margin, pick)


def _window(example: Example, confirmations: dict) -> tuple[int, int]:
    s, e = example.series, example.event
    if e.pattern == "wedge":
        first = min(_at(s, e.lines[0][0]), _at(s, e.lines[1][0]))
        done = confirmations.get((example.ticker, e.day))
        last = _at(s, done) if done else _at(s, e.day)
        return max(0, first - 8), min(len(s), last + 8)
    i = _at(s, e.marks[-1])
    return max(0, i - BEFORE), min(len(s), i + AFTER)


def chart(example: Example, confirmations: dict, width: int = 720) -> str:
    """An SVG candlestick chart with volume; shaded bars are the ones the rule read."""
    start, end = _window(example, confirmations)
    bars = example.series[start:end]
    top, bottom = max(c.high for c in bars), min(c.low for c in bars)
    pad = (top - bottom) * 0.05 or 1
    top, bottom = top + pad, bottom - pad
    ph, vh, gap = 240, 60, 8
    step = width / len(bars)
    peak = max(c.volume for c in bars) or 1

    def y(price: float) -> float:
        return ph - (price - bottom) / (top - bottom) * ph

    marks = set(example.event.marks) | (
        {confirmations[(example.ticker, example.event.day)]}
        if (example.ticker, example.event.day) in confirmations
        else set()
    )
    parts = [f'<svg viewBox="0 0 {width} {ph + gap + vh}" class="chart" role="img">']
    for k, c in enumerate(bars):
        x = k * step
        mid = x + step / 2
        if c.day in marks:
            parts.append(
                f'<rect x="{x:.1f}" y="0" width="{step:.1f}" height="{ph + gap + vh}" class="mark"/>'
            )
        cls = "up" if c.close >= c.open else "down"
        if not c.valid:
            cls = "bad"
        parts.append(
            f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="{y(c.high):.1f}" y2="{y(c.low):.1f}" class="{cls}"/>'
        )
        body_top, body_bottom = y(max(c.open, c.close)), y(min(c.open, c.close))
        parts.append(
            f'<rect x="{x + step * 0.15:.1f}" y="{body_top:.1f}" width="{step * 0.7:.1f}" '
            f'height="{max(body_bottom - body_top, 0.8):.1f}" class="{cls} body"/>'
        )
        vol = c.volume / peak * vh
        parts.append(
            f'<rect x="{x + step * 0.15:.1f}" y="{ph + gap + vh - vol:.1f}" width="{step * 0.7:.1f}" '
            f'height="{vol:.1f}" class="vol {cls}"/>'
        )
    days = [c.day for c in bars]
    for d1, p1, d2, p2 in example.event.lines:
        if d1 in days and d2 in days:
            i1, i2 = days.index(d1), days.index(d2)
            slope = (p2 - p1) / (i2 - i1)
            stop = len(bars) - 1
            parts.append(
                f'<line x1="{(i1 + 0.5) * step:.1f}" y1="{y(p1):.1f}" '
                f'x2="{(stop + 0.5) * step:.1f}" y2="{y(p1 + slope * (stop - i1)):.1f}" class="trend"/>'
            )
    if "breakout_level" in example.event.levels:
        level = y(example.event.levels["breakout_level"])
        parts.append(f'<line x1="0" x2="{width}" y1="{level:.1f}" y2="{level:.1f}" class="level"/>')
    parts.append("</svg>")
    return "".join(parts)


def _facts(example: Example) -> str:
    e = example.event
    shown = ", ".join(
        f"{k} {v:.2f}" if abs(v) < 1000 else f"{k} {v:,.0f}" for k, v in e.levels.items()
    )
    extra = f"；确认方式 {e.how}" if e.how else ""
    return f"{shown}{extra}；离最近的阈值还有 {example.margin:+.1%}"


def page(
    examples: list[Example], config: PatternConfig, confirmations: dict, notes: list[str]
) -> str:
    cards = []
    for n, ex in enumerate(examples, start=1):
        key = f"{ex.event.pattern}|{ex.ticker}|{ex.event.day}"
        title = f"{n}. {NAMES[ex.event.pattern]} · {ex.ticker} · {ex.event.day}"
        tag = "刚好过线" if ex.pick == "borderline" else "随机"
        cards.append(
            f'<section class="card" data-key="{html.escape(key)}">'
            f'<h3>{html.escape(title)} <span class="tag {ex.pick}">{tag}</span></h3>'
            f'<p class="facts">{html.escape(_facts(ex))}</p>'
            f"{chart(ex, confirmations)}"
            '<div class="verdict">'
            f'<label><input type="radio" name="v{n}" value="算"> 算</label>'
            f'<label><input type="radio" name="v{n}" value="不算"> 不算</label>'
            f'<label><input type="radio" name="v{n}" value="说不准"> 说不准</label>'
            f'<input type="text" class="note" placeholder="备注：哪里像、哪里不像">'
            "</div></section>"
        )
    rules = html.escape(
        json.dumps(config.model_dump(exclude={"version", "digest"}), ensure_ascii=False, indent=1)
    )
    listed = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>形态例子 {config.version}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#fff; --line:#e4e1d8;
  --up:#1a7f4b; --down:#c0392b; --mark:rgba(240,180,40,.18); --trend:#2b6cb0; --level:#8e44ad; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161614; --fg:#ecebe6; --muted:#9a988f; --card:#201f1c;
  --line:#34322d; --mark:rgba(240,180,40,.16); }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 -apple-system,"PingFang SC",sans-serif; }}
main {{ max-width:780px; margin:0 auto; padding:24px 16px 80px; }}
h1 {{ font-size:22px; margin:0 0 8px; }} h3 {{ font-size:16px; margin:0 0 4px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:16px 0; }}
.facts, .muted {{ color:var(--muted); font-size:13px; margin:0 0 8px; }}
.chart {{ width:100%; height:auto; display:block; }}
.up {{ stroke:var(--up); fill:var(--up); }} .down {{ stroke:var(--down); fill:var(--down); }}
.bad {{ stroke:#999; fill:#999; }} .vol {{ opacity:.45; }} .mark {{ fill:var(--mark); }}
.trend {{ stroke:var(--trend); stroke-width:1.5; stroke-dasharray:5 3; }}
.level {{ stroke:var(--level); stroke-width:1; stroke-dasharray:2 3; }}
.tag {{ font-size:12px; padding:1px 6px; border-radius:6px; background:var(--line); color:var(--muted); }}
.tag.borderline {{ background:#f6e3b4; color:#6b4e00; }}
.verdict {{ display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin-top:8px; }}
.note {{ flex:1; min-width:180px; padding:4px 8px; border:1px solid var(--line); border-radius:6px; background:transparent; color:var(--fg); }}
textarea {{ width:100%; height:180px; font:12px/1.4 ui-monospace,monospace; background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px; }}
button {{ padding:6px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }}
details pre {{ font-size:12px; overflow:auto; }}
</style></head><body><main>
<h1>形态例子：请判断"算 / 不算"</h1>
<p class="muted">配置 {config.version}-{config.digest}。每种形态 10 个：4 个"刚好过线"（离某个阈值最近），6 个随机（固定种子）。
浅黄底色是规则读到的 K 线；蓝色虚线是楔形的上下沿；紫色点线是突破价位。这里不显示之后的涨跌：只判断定义像不像你心里的形态。</p>
<ul class="muted">{listed}</ul>
<details><summary>当前阈值</summary><pre>{rules}</pre></details>
{"".join(cards)}
<h2>结果</h2>
<p class="muted">判断完点下面的按钮，把文本框里的内容复制给 Claude。</p>
<button id="make">生成结果</button>
<textarea id="out" readonly></textarea>
<script>
(function () {{
  const cards = Array.from(document.querySelectorAll('.card'));
  const KEY = 'pattern-review-{config.version}-{config.digest}';
  let saved = {{}};
  try {{ saved = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
  cards.forEach(function (card) {{
    const k = card.dataset.key, s = saved[k] || {{}};
    card.querySelectorAll('input[type=radio]').forEach(function (r) {{ if (r.value === s.v) r.checked = true; }});
    const note = card.querySelector('.note'); if (s.n) note.value = s.n;
    card.addEventListener('change', store); note.addEventListener('input', store);
  }});
  function collect() {{
    return cards.map(function (card, i) {{
      const r = card.querySelector('input[type=radio]:checked');
      return {{ k: card.dataset.key, v: r ? r.value : '', n: card.querySelector('.note').value.trim(), i: i + 1 }};
    }});
  }}
  function store() {{
    const all = {{}}; collect().forEach(function (x) {{ all[x.k] = {{ v: x.v, n: x.n }}; }});
    try {{ localStorage.setItem(KEY, JSON.stringify(all)); }} catch (e) {{}}
  }}
  document.getElementById('make').addEventListener('click', function () {{
    document.getElementById('out').value = collect().map(function (x) {{
      return x.i + '. ' + x.k + ' → ' + (x.v || '未判断') + (x.n ? '（' + x.n + '）' : '');
    }}).join('\\n');
  }});
}})();
</script>
</main></body></html>
"""
