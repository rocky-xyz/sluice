from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from sluice.news.models import StoredNews
from sluice.news.store import NewsStore
from sluice.scoring.client import (
    CachedJev,
    ChoiceAnswer,
    JevError,
    JevQuestion,
    JevRequest,
    JevResponse,
    NoulAnswer,
)
from sluice.scoring.config import ScoringConfig
from sluice.scoring.similarity import most_similar, split_sentences
from sluice.scoring.store import ItemScores, ScoreStore

# Candidate pool of prior same-ticker headlines to search for novelty references.
HISTORY_POOL = 50


def build_state(title: str, summary: str | None, references: list[str]) -> str:
    lines = [f"Headline: {title}"]
    if summary:
        lines.append(f"Summary: {summary}")
    if references:
        lines.append("Recent headlines about the same company, most similar first:")
        lines.extend(f"- {h}" for h in references)
    return "\n".join(lines)


def _state_variants(
    title: str, summary: str | None, history_titles: list[str], reference_k: int
) -> Iterator[str]:
    """States to try in order when the previous one came back too large.

    Shrinks the novelty reference list first (cheapest to lose), then the
    item's own summary down to its most title-relevant sentences, then the
    title alone. In this project items only ever have a short title and
    summary (no full article text), so reaching the later variants is
    expected to be rare.
    """
    if reference_k > 0 and history_titles:
        yield build_state(title, summary, most_similar(title, history_titles, reference_k))
    yield build_state(title, summary, [])
    if summary:
        sentences = split_sentences(summary)
        ranked = most_similar(title, sentences, len(sentences))
        for n in range(len(ranked) - 1, 0, -1):
            yield build_state(title, " ".join(ranked[:n]), [])
    yield build_state(title, None, [])


def _questions(config: ScoringConfig) -> dict[str, JevQuestion]:
    return {
        name: JevQuestion(type=spec.type, instructions=spec.instructions, criteria=spec.criteria)
        for name, spec in config.questions.items()
    }


def _parse(response: JevResponse) -> ItemScores:
    stance = response.answers["stance"]
    event_type = response.answers["event_type"]
    assert isinstance(stance, ChoiceAnswer)  # the config's "stance" question is a choice
    assert isinstance(event_type, ChoiceAnswer)  # same for "event_type"
    relevance = response.answers["relevance"]
    novelty = response.answers["novelty"]
    rumour = response.answers["rumour"]
    assert isinstance(relevance, NoulAnswer)
    assert isinstance(novelty, NoulAnswer)
    assert isinstance(rumour, NoulAnswer)
    return ItemScores(
        relevance=relevance.noul,
        stance=stance.choice,
        stance_confidence=stance.confidence,
        stance_probabilities=stance.probabilities,
        novelty=novelty.noul,
        rumour=rumour.noul,
        event_type=event_type.choice,
        event_type_confidence=event_type.confidence,
        event_type_probabilities=event_type.probabilities,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def score_item(
    item: StoredNews, history_titles: list[str], config: ScoringConfig, jev: CachedJev
) -> tuple[ItemScores | None, str | None]:
    """Score one item, shrinking the state and asking again if it comes back too large.

    Returns `(scores, None)` on success, or `(None, reason)` when every
    shrunk variant was still too large. A non-`too_large` `JevError`
    (network, rate limit, retries exhausted) is not caught here: the caller
    decides what "leave it for the next run" means.
    """
    questions = _questions(config)
    for state in _state_variants(item.title, item.summary, history_titles, config.novelty_reference_k):
        request = JevRequest(model=config.model, state=state, questions=questions)
        try:
            response = jev.answer(request)
        except JevError as exc:
            if not exc.too_large:
                raise
            continue
        return _parse(response), None
    return None, "state_too_large"


class ScoreResult(BaseModel):
    news_id: int
    ticker: str
    status: Literal["scored", "skipped", "done", "error"]
    detail: str | None = None  # the skip reason, or the error message
    input_tokens: int = 0
    output_tokens: int = 0


def jobs_for_ticker(
    items: list[StoredNews], history_pool: int = HISTORY_POOL
) -> list[tuple[StoredNews, list[str]]]:
    """Items oldest-first, each paired with up to `history_pool` of its most recent
    predecessors' titles. Point-in-time by construction: an item only ever sees the
    titles of items that come before it in this same ticker's chronological order.
    """
    ordered = sorted(items, key=lambda i: i.published_at or datetime.min.replace(tzinfo=UTC))
    titles = [i.title for i in ordered]
    return [(item, titles[max(0, i - history_pool) : i]) for i, item in enumerate(ordered)]


def score_all(
    tickers: list[str],
    store: NewsStore,
    scores: ScoreStore,
    config: ScoringConfig,
    jev: CachedJev,
    workers: int = 1,
    limit: int | None = None,
    on_done: Callable[[ScoreResult], None] | None = None,
) -> list[ScoreResult]:
    """Score every item for `tickers` not already done under `config.version`.

    One query per ticker (not per item); the point-in-time history window for
    each item comes from slicing that ticker's own chronologically sorted
    list, in memory. An item linked to more than one of `tickers` is scored
    once (first ticker it is seen under supplies its history window and is
    the one reported in its `ScoreResult`).
    Safe to repeat: `ScoreStore` makes already-done items a no-op.
    """
    now = datetime.now(UTC)
    seen: dict[int, tuple[str, StoredNews, list[str]]] = {}
    for ticker in tickers:
        for item, history in jobs_for_ticker(store.query(ticker, before=now, dedup=True)):
            seen.setdefault(item.id, (ticker, item, history))
    jobs = list(seen.values())[:limit]

    def work(job: tuple[str, StoredNews, list[str]]) -> ScoreResult:
        ticker, item, history = job
        if scores.done(item.id, config.version):
            result = ScoreResult(news_id=item.id, ticker=ticker, status="done")
        else:
            try:
                parsed, skip_reason = score_item(item, history, config, jev)
            except JevError as exc:
                result = ScoreResult(news_id=item.id, ticker=ticker, status="error", detail=str(exc))
            else:
                if parsed is not None:
                    scores.put(item.id, config.version, config.digest, parsed)
                    result = ScoreResult(
                        news_id=item.id,
                        ticker=ticker,
                        status="scored",
                        input_tokens=parsed.input_tokens,
                        output_tokens=parsed.output_tokens,
                    )
                else:
                    scores.skip(item.id, config.version, config.digest, skip_reason or "")
                    result = ScoreResult(
                        news_id=item.id, ticker=ticker, status="skipped", detail=skip_reason
                    )
        if on_done is not None:
            on_done(result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(work, jobs))
