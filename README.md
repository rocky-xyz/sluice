# sluice

A multi-agent equity research pipeline that scores every news item, learns how
much each source and event type should count, and turns the result into
position sizes with code rather than with an LLM.

This is a research and decision-support tool. It does not place orders, and
nothing here is financial advice.

> **Status: early.** The repository currently contains the news store and
> collectors for Yahoo Finance and Finnhub. The sections below describe the design being built;
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
 Finnhub)
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
sluice news collect --source all         # Yahoo Finance and Finnhub (needs FINNHUB_API_KEY)
sluice news backfill                     # archive a year of Finnhub company news (resumable)
sluice news resolve-links                # record where Finnhub's article links lead (resumable)
sluice news import                       # rebuild the store from the raw archive
sluice fundamentals fetch                # quarterly statements from Alpha Vantage (resumable; 25 calls a day)
sluice analyze NVDA --date 2026-09-30    # run the pipeline as of that day's close (needs DEEPSEEK_API_KEY)
```

`news collect` adds new items to `data/news.db` and archives the raw vendor
responses as JSONL. `news backfill` archives history from Finnhub, one ticker
and one day per request, and resumes where it stopped. `news resolve-links`
follows Finnhub's redirect links to learn the original address, and with it the
publisher, of articles that Finnhub only attributes to an aggregator. `news import` loads
archived items from every source into the store, so the database can always be
rebuilt. Reprints (same headline within 12 hours) are grouped: a query returns
each story once, with a reprint count as of the decision time, its publisher
when known, and the trading session it was published in (pre-market, regular,
post-market or closed, from the NYSE calendar). Scheduled
collection runs in a separate private data repository,
cloned to `data/`; items collected by hand are archived under `data/local/` so
they never conflict with it.

`analyze` runs the pipeline for one ticker as of one trading day's close. So far
the pipeline has no debate yet: code fetches what was known at the close, then
a market analyst, a news analyst and a fundamentals analyst each write a report
that cites its sources by id, side by side, and a research manager gives a
five-step rating with a confidence and an evidence list. Answers that cite an
id that was never given are rejected and asked again. Model calls
are cached in `data/local/llm_cache.db` and each run is checkpointed, so
repeating a finished run costs nothing; `--out` writes the full run state as
JSON. Models, prompts and limits live in `configs/pipeline/`, one file per
version.

Daily prices come from Yahoo Finance and are kept in `data/local/prices.db`; a
ticker is fetched the first time it is needed. A day's bar counts only once its
session has closed, so a run at the close of day T sees T's bar and nothing
later. From them the code computes ratio-only indicators (trailing and excess
returns, volatility, distance from moving averages, RSI, relative volume) for
the market analyst.

Quarterly statements come from Alpha Vantage (`ALPHA_VANTAGE_API_KEY`) and are
kept as the vendor sent them under `data/local/raw/alphavantage/`. The free key
allows 25 calls a day and a ticker takes three, so `fundamentals fetch` stops at
the limit and continues the next day; the pipeline itself only reads what is
stored. A quarter counts as published 60 days after it ends. The fundamentals
analyst gets the latest four such quarters (revenue, margins, net income, free
cash flow, each with its change from a year earlier) and, for companies that
report in dollars, price-to-earnings and price-to-sales over four quarters.
The statements are the vendor's current version: a later restatement of an old
quarter is not undone.

API keys are read from the environment or from a local `.env` file; see
`.env.example`. Further subcommands will be added as each roadmap step lands.

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
