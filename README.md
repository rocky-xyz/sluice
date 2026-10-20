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
sluice news resolve-publishers           # read the publisher named on Yahoo-hosted article pages (resumable)
sluice news import                       # rebuild the store from the raw archive
sluice news fit-weights                  # fit and save one source/event-type weight table per month
sluice fundamentals fetch                # quarterly statements from Alpha Vantage (resumable; 25 calls a day)
sluice analyze NVDA --date 2026-09-30    # run the pipeline as of that day's close (needs DEEPSEEK_API_KEY)
sluice analyze TSLA                      # or as of now, for any ticker: collects its news first
sluice batch                             # backtest the universe on every tenth trading day (resumable)
sluice batch --config v2                 # the same with weighted news (needs TYPESAFE_API_KEY and fit-weights)
sluice score                             # score stored news with Jev (needs TYPESAFE_API_KEY; resumable)
sluice evaluate                          # portfolio backtest of v1 and v2 ratings on the validation period
sluice evaluate --sizing v1              # the same, plus the sizing formula and its controls
sluice evaluate --final test_1           # the one evaluation of the sealed test period: champion vs every baseline
sluice score-reports                     # Jev's stance on each analyst report of the validation runs
sluice evolve seed                       # log the sizing configs already seen; put the two controls through the gate
sluice evolve try --set k=0.3           # one hand-written sizing change through the gate (uses a trial)
sluice evolve propose --dry-run          # ask the LLM proposer for one sizing change, without using a trial
sluice evolve propose                    # the same, then put it through the gate (needs DEEPSEEK_API_KEY)
sluice evolve status                     # champion, trials left, experiment log
sluice evolve rollback v1                # point the champion back at an older sizing version
sluice evolve screen --set tools.max_calls=4         # estimate a pipeline change's screen; add --yes to run it
sluice evolve screen --prompt news_analyst=new.txt   # the same for one role's whole prompt
sluice evolve full --set tools.max_calls=4 --yes     # full validation run of a screened change
sluice evolve pipeline-status                        # pipeline champion, screens and full evaluations left
sluice evolve propose-pipeline                       # ask the LLM for one pipeline change; screen it with --proposal
sluice patterns universe                 # read today's S&P 500 once into configs/universe/ (pattern study)
sluice patterns fetch                    # ten years of daily bars for it, in a store of its own (resumable)
```

`news collect` adds new items to `data/news.db` and archives the raw vendor
responses as JSONL. `news backfill` archives history from Finnhub, one ticker
and one day per request, and resumes where it stopped. `news resolve-links`
follows Finnhub's redirect links to learn the original address, and with it the
publisher, of articles that Finnhub only attributes to an aggregator. For those
whose link lands back on a Yahoo page, `news resolve-publishers` reads the
publisher that page names in its metadata; it keeps only that name, not the
page. `news import` loads
archived items from every source into the store, so the database can always be
rebuilt. Reprints (same headline within 12 hours) are grouped: a query returns
each story once, with a reprint count as of the decision time, its publisher
when known, and the trading session it was published in (pre-market, regular,
post-market or closed, from the NYSE calendar). Scheduled
collection runs in a separate private data repository,
cloned to `data/`; items collected by hand are archived under `data/local/` so
they never conflict with it.

`analyze` runs the pipeline for one ticker. Code fetches what was known at the
decision time; a market analyst, a news analyst and a fundamentals analyst each
write a report that cites its sources by id, side by side; a bullish and a
bearish debater argue from those reports for a configured number of rounds; and
a research manager gives a five-step rating with a confidence and an evidence
list. Only the analysts see the data. Answers that cite an id that was never
given, or that no report cites, are rejected and asked again. The pipeline
extracts and weighs information; it does not size positions.

With `--date` the run is a backtest that decides at that trading day's close
and touches no network except the model. Without it the run is ad hoc: it
decides now, first collects the ticker's news from Yahoo Finance and, if none
are stored, its statements from Alpha Vantage (three calls), and uses the last
finished trading day's prices. Both use the same graph, roles and config. A
ticker outside the studied universe is marked as having no backtest evidence. Model calls
are cached in `data/local/llm_cache.db` and each run is checkpointed, so
repeating a finished run costs nothing; `--out` writes the full run state as
JSON. Models, prompts and limits live in `configs/pipeline/`, one file per
version.

`batch` runs the backtest over a grid: by default the eleven companies of the
universe on every tenth trading day since 2026-01-02, four runs at a time. Runs
are independent of each other, so neither their order nor the number of workers
changes a result. Each run's full state is saved as JSON under
`data/local/runs/<config>-<digest>/<ticker>/`, next to a `ratings.jsonl` with
one line per run (rating, confidence, gaps, tokens). A finished run is never
paid for twice: rerunning the command resumes what broke off and returns the
rest from checkpoints. It refuses to start while statements or prices are
missing for a ticker, because such a run would finish with a gap and stay that
way.

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
