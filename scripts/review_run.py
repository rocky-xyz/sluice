"""Export a saved pipeline run to a human-readable file, for the task-2.7 spot
check (tasks.md): is any report quoting a number that is not actually in the
evidence, or using information that would not have existed at decision time?

Usage:
    python scripts/review_run.py data/local/runs/v1-97b47162/AMD/2026-01-02_r0.json

This only does the mechanical part of that: it checks that every bracketed
citation like [N16253] or [F:revenue_q0] points at evidence that was actually
fetched for this run, and that no cited evidence's own timestamp is after the
decision. A hit here is a real bug (a leaked future number, or a reference
nobody fetched), not a judgment call.

A first version also tried to match the number written next to each citation
against the stored value, to catch invented figures automatically. Dropped:
reports pack several citations and figures into one sentence, sometimes citing
before the number and sometimes after, and the character-window heuristic
flagged most citations as mismatched even when they were fine. Whether a
number in the prose is actually the cited figure, or invented outright with no
citation at all, still has to be read by a person - this script only gets you
a clean evidence catalog and the full text to read it against.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sluice.news.timeutil import ET

REF = re.compile(r"\[((?:N\d+)|(?:P:[A-Za-z0-9_]+)|(?:F:[A-Za-z0-9_]+))\]")


@dataclass
class Evidence:
    kind: str  # news, price, fundamentals
    display: str
    as_of: datetime | None  # when this became knowable; None if not checkable


def catalog(inputs: dict, decision_time: datetime) -> dict[str, Evidence]:
    items: dict[str, Evidence] = {}
    for n in inputs["news"]:
        published = datetime.fromisoformat(n["published_at"])
        title, summary = n["title"] or "", n["summary"] or ""
        items[n["ref"]] = Evidence(kind="news", display=f"{title} — {summary[:200]}", as_of=published)
    for p in inputs["prices"]:
        items[p["ref"]] = Evidence(kind="price", display=f"{p['name']} = {p['value']}", as_of=None)
    for f in inputs["fundamentals"]:
        visible = datetime.fromisoformat(f["visible_from"]).replace(tzinfo=ET)
        display = f"{f['name']} = {f['value']} ({f['unit']}, quarter {f['quarter_end']}, visible from {f['visible_from']})"
        items[f["ref"]] = Evidence(kind="fundamentals", display=display, as_of=visible)
    return items


def check_refs(label: str, refs: list[str], items: dict[str, Evidence], decision_time: datetime) -> list[str]:
    flags = []
    for ref in refs:
        item = items.get(ref)
        if item is None:
            flags.append(f"[{label}] 引用了不存在的证据编号 {ref}")
        elif item.as_of is not None and item.as_of > decision_time:
            flags.append(f"[{label}] {ref} 的可见时间 {item.as_of} 晚于决策时刻 {decision_time}（point-in-time 违规）")
    return flags


def render(state: dict) -> str:
    meta, inputs, decision = state["meta"], state["inputs"], state["decision"] or {}
    decision_time = datetime.fromisoformat(meta["decision_time"])
    items = catalog(inputs, decision_time)

    flags: list[str] = []
    for role, report in state["reports"].items():
        refs = [m.group(1) for m in REF.finditer(report["text"])]
        flags += check_refs(f"{role} 分析师报告", refs, items, decision_time)
    for turn in state["debate"]:
        refs = [m.group(1) for m in REF.finditer(turn["text"])]
        flags += check_refs(f"辩论 第{turn['round']}轮 {turn['side']}", refs, items, decision_time)
    for i, ev in enumerate(decision.get("evidence", [])):
        flags += check_refs(f"决策证据 #{i}", ev["refs"], items, decision_time)

    out = [f"# {meta['ticker']} — {meta['decision_time']} ({meta['run_id']})\n"]
    out.append(f"状态：{meta['status']}" + (f"（{meta['failure']}）" if meta.get("failure") else ""))
    if decision:
        out.append(f"评级：{decision['rating']}，置信度 {decision['confidence']}\n")

    out.append("## 自动检查结果\n")
    out.append("（只查引用是否存在、是否违反 point-in-time；数字是否编造需要你往下读——见本脚本顶部的说明）\n")
    if flags:
        out += [f"- {f}" for f in dict.fromkeys(flags)]  # de-duplicate, keep order
    else:
        out.append("- 没有发现引用缺失或 point-in-time 违规。")
    out.append("")

    out.append("## 证据清单\n")
    for ref, item in sorted(items.items()):
        out.append(f"- `{ref}` ({item.kind}): {item.display}")
    out.append("")

    out.append("## 分析师报告\n")
    for role, report in state["reports"].items():
        out.append(f"### {role}\n\n{report['text']}\n")

    out.append("## 辩论\n")
    for turn in state["debate"]:
        out.append(f"### 第{turn['round']}轮 — {turn['side']}\n\n{turn['text']}\n")

    out.append("## 最终决策\n")
    if decision:
        out.append(f"评级：**{decision['rating']}**，置信度 {decision['confidence']}\n")
        for ev in decision.get("evidence", []):
            out.append(f"- [{ev['direction']}] {ev['claim']} {' '.join(f'[{r}]' for r in ev['refs'])}")
    else:
        out.append("（这次运行没有产出决策。）")

    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_files", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None, help="默认写到每个输入文件所在 run 目录下的 review/ 子目录")
    args = parser.parse_args()

    for path in args.state_files:
        state = json.loads(path.read_text())
        out_dir = args.out_dir or (path.parent.parent / "review")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.parent.name}_{path.stem}.md"
        out_path.write_text(render(state))
        print(out_path)


if __name__ == "__main__":
    main()
