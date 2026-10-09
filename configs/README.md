# configs

Every parameter the evolution loop may change lives here: weights, Jev question
specs, the news taxonomy, thresholds and prompts. One file per version; a change
is a new version, so every change can be diffed and reverted.

Nothing under `sluice/evaluation/` reads its metrics or data splits from a file
the evolution loop can write.
