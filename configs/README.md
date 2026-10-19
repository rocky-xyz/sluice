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
