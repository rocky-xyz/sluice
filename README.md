# sluice

A multi-agent equity research pipeline that scores every news item, learns how
much each source and event type should count, and turns the result into
position sizes with code rather than with an LLM.

This is a research and decision-support tool. It does not place orders, and
nothing here is financial advice.

> **Status: early.** The repository currently contains the news store and a
> Yahoo Finance collector. The sections below describe the design being built;
> the roadmap shows what exists.

## Why

Multi-agent LLM pipelines for stock research usually hand analysts a raw list of
headlines, filter noise with hand-picked thresholds, and stop at a rating. sluice
is built around the parts that are normally skipped:

- **Per-item news scoring.** Each news item is scored for relevance, stance,
  novelty, rumour and event type, instead of being summarised as free text.
- **Learned weights.** Source and event-type weights are fitted from
  point-in-time event studies, with shrinkage and uncertainty, not set by hand.
- **Sizing in code.** LLMs and the scorer only extract information. Position
  sizes come from a plain Python function: market regime, weighted signals,
  volatility scaling, caps.
- **Portfolio-level evaluation.** Walk-forward backtests with holdings, fills
  and costs, reported against buy-and-hold and index baselines.
- **Guarded self-improvement.** An LLM proposes config changes; a gate accepts
  one only if its validation gain exceeds run-to-run noise.

The goal is rigour, not a claim of profitability.

## Architecture

```
data (yfinance,        ->  news store: timestamp, source, ticker, text      [cached]
 Alpha Vantage)
        |
scoring                per-item scores: relevance, stance, novelty, rumour,
                       event type + code features (source, time of day,
                       duplicate count)                                     [cached]
        |
aggregation            learned weights by source x type x regime
                       -> weighted news signal
        |
agents                 multi-agent graph; the news analyst receives filtered,
                       weighted items and the aggregate signal              [cached]
        |
sizing                 regime (computed in code) + scored analyst stances
(pure Python)          -> target position -> volatility scaling -> caps
        |
evaluation (frozen)    portfolio walk-forward backtest, baselines, hit rate,
                       calibration, consistency across repeated runs
        ^
evolution              proposer (LLM) reads failures -> config diff -> gate
                       accepts only if validation gain > noise -> versioned
                       config store
```

## Design rules

- **Point-in-time.** A feature may only use information published before the
  decision or news timestamp. Trades and labels start at the next open.
- **Everything is cached.** Every LLM and scorer call is cached, so changing
  weights, the sizing formula or the regime definition never triggers new calls.
- **Evolvable things live in `configs/`.** One file per version; every change
  is a diff that can be reverted.
- **Evaluation is frozen.** The evolution loop cannot change metrics or data
  splits. The test period is sealed and evaluated once.
- **Walk-forward only.** No shuffling across time.

## Layout

```
sluice/
  news/         collection, storage, deduplication, features, event study, weights
  scoring/      versioned question specs, scorer client, score cache
  agents/       multi-agent pipeline: graph, state, roles, prompts, LLM client
  sizing/       regime detection, weighting, position sizing, risk caps
  evaluation/   portfolio backtest, metrics, data splits, calibration (frozen)
  evolution/    proposer, gate, config store, experiment log
configs/        every evolvable parameter, one file per version
scripts/  tests/
```

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/rocky-xyz/sluice.git
cd sluice
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
sluice --version
sluice news collect                      # fetch recent Yahoo Finance news for the universe
sluice news collect --tickers NVDA,AMD   # or for chosen tickers
```

`news collect` adds new items to `data/news.db` and archives the raw vendor
responses under `data/raw/` as JSONL, so the database can be rebuilt. Further
subcommands will be added as each roadmap step lands.

## Development

```bash
ruff check .
pytest -q          # tests marked `integration` are skipped by default
```

## Roadmap

- [x] Packaging and package skeleton
- [ ] News store and daily collection
- [ ] Multi-agent pipeline skeleton
- [ ] Per-item news scoring
- [ ] News event study and learned weights
- [ ] Weighted news wired into the pipeline
- [ ] Position sizing and portfolio backtest
- [ ] Evolution loop

## Acknowledgements

The overall architecture (parallel analysts, a bull/bear debate, a risk debate
and a final decision) is inspired by
[TradingAgents](https://github.com/TauricResearch/TradingAgents). No code or
prompts are copied from it; it is used only as an architectural reference and
as an external baseline run outside this repository.

## License

[MIT](LICENSE)
