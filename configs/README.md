# configs

Every parameter the evolution loop may change lives here: weights, Jev question
specs, the news taxonomy, thresholds and prompts. One file per version; a change
is a new version, so every change can be diffed and reverted.

Nothing under `sluice/evaluation/` reads its metrics or data splits from a file
the evolution loop can write.

`sizing/CHAMPION` names the sizing version in use. The evolution loop writes a
new `sizing/vN.toml` only when its gatekeeper accepts a change, copied from the
champion with just the changed values replaced; rolling back is pointing
`CHAMPION` at an older version. No version file is edited or deleted.

`sizing/experiments.jsonl` is the evolution loop's experiment log: one line per
proposal, appended and never rewritten. Each proposal that looked at the
validation period has a trial number; there are at most 20 for the whole loop.

`pipeline/CHAMPION` and `pipeline/experiments.jsonl` do the same for the pipeline
(prompts and tool-loop rules), with two budgets: up to 30 screens on 55 fixed
validation runs, and up to 5 full evaluations, each only after a passed screen.

`patterns/vN.toml` holds the candlestick rules of the pattern study, and
`universe/` the fixed stock lists it runs on (today's S&P 500, read once, with
the date it was read: looked back on, it carries survivorship bias).
