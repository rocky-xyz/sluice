import json

import pytest
import requests

from sluice.agents.cache import CallCache
from sluice.scoring.client import (
    RETRIES,
    RETRY_WAITS,
    CachedJev,
    ChoiceAnswer,
    JevClient,
    JevError,
    JevQuestion,
    JevRequest,
    JevResponse,
    NoulAnswer,
    cache_key,
)


def request(state: str = "hello", **overrides) -> JevRequest:
    fields = {
        "model": "jev-1.13.0",
        "state": state,
        "questions": {"relevant": JevQuestion(type="noul", instructions="is this relevant?")},
    }
    fields.update(overrides)
    return JevRequest(**fields)


def noul_body(value: float = 0.9) -> dict:
    return {
        "answers": {"relevant": {"type": "noul", "noul": value}},
        "usage": {"input_tokens": 40, "output_tokens": 5},
    }


class FakeHTTP:
    """Stands in for requests.post: answers in order, records what was sent."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        response = requests.Response()
        response.status_code = status
        response._content = (body if isinstance(body, str) else json.dumps(body)).encode()
        return response


class FakeClient:
    """Stands in for JevClient: answers in order, counts calls."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def answer(self, request):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return JevResponse(answers=answer, input_tokens=40, output_tokens=5)


@pytest.fixture
def cache(tmp_path):
    with CallCache(tmp_path / "jev.db") as cache:
        yield cache


def jev(cache, *answers):
    client = FakeClient(*answers)
    waits = []
    return CachedJev(cache, client, sleep=waits.append), client, waits


# --- cache key ---


def test_key_is_stable_and_depends_on_every_field():
    base = cache_key(request())
    assert base == cache_key(request())
    assert len(
        {
            base,
            cache_key(request("something else")),
            cache_key(request(model="jev-preview")),
            cache_key(
                request(
                    questions={
                        "relevant": JevQuestion(type="noul", instructions="different question")
                    }
                )
            ),
            cache_key(request(), repeat=1),
        }
    ) == 5


# --- JevClient ---


def test_client_sends_the_request_and_reads_the_answer():
    http = FakeHTTP((200, noul_body(0.9)))
    answer = JevClient("secret", post=http).answer(request())

    assert answer == JevResponse(
        answers={"relevant": NoulAnswer(noul=0.9)}, input_tokens=40, output_tokens=5
    )
    url, sent = http.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"] == {"Authorization": "Bearer secret"}
    assert sent["json"] == {
        "model": "jev-1.13.0",
        "state": "hello",
        "questions": {"relevant": {"type": "noul", "instructions": "is this relevant?"}},
    }


def test_client_parses_a_choice_answer():
    body = {
        "answers": {
            "stance": {
                "type": "choice",
                "choice": "bullish",
                "probabilities": {"bullish": 0.8, "bearish": 0.2},
                "confidence": 0.8,
            }
        },
        "usage": {"input_tokens": 50, "output_tokens": 4},
    }
    answer = JevClient("secret", post=FakeHTTP((200, body))).answer(
        request(questions={"stance": JevQuestion(type="choice", instructions="pick", criteria={"bullish": "x", "bearish": "y"})})
    )
    assert answer.answers["stance"] == ChoiceAnswer(
        choice="bullish", probabilities={"bullish": 0.8, "bearish": 0.2}, confidence=0.8
    )


@pytest.mark.parametrize(
    ("answer", "retryable"),
    [
        ((429, "slow down"), True),
        ((500, "oops"), True),
        ((503, "busy"), True),
        ((400, "bad request"), False),
        ((401, "bad key"), False),
        ((200, "not json"), True),
        ((200, {"answers": {}}), True),  # missing the question we asked
        (requests.ConnectionError("down"), True),
        (requests.Timeout(), True),
    ],
)
def test_client_says_which_failures_may_pass(answer, retryable):
    with pytest.raises(JevError) as raised:
        JevClient("secret", post=FakeHTTP(answer)).answer(request())
    assert raised.value.retryable is retryable


def test_a_422_for_an_oversized_state_is_flagged_too_large():
    http = FakeHTTP((422, "state exceeds the 32k context limit"))
    with pytest.raises(JevError) as raised:
        JevClient("secret", post=http).answer(request())
    assert raised.value.retryable is False
    assert raised.value.too_large is True


def test_a_422_for_a_bad_request_is_not_flagged_too_large():
    http = FakeHTTP((422, "unknown field 'foo'"))
    with pytest.raises(JevError) as raised:
        JevClient("secret", post=http).answer(request())
    assert raised.value.too_large is False


def test_error_text_does_not_contain_the_key():
    with pytest.raises(JevError) as raised:
        JevClient("secret", post=FakeHTTP((401, "bad key"))).answer(request())
    assert "secret" not in str(raised.value)


# --- cached calls ---


def test_second_identical_call_comes_from_the_cache(cache):
    answers = {"relevant": {"type": "noul", "noul": 0.9}}
    model, client, _ = jev(cache, answers)
    first = model.answer(request())
    second = model.answer(request())

    assert client.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.model_copy(update={"cached": False}) == first


def test_a_different_repeat_is_a_new_call(cache):
    model, client, _ = jev(
        cache, {"relevant": {"type": "noul", "noul": 0.1}}, {"relevant": {"type": "noul", "noul": 0.9}}
    )
    assert model.answer(request(), repeat=0).answers["relevant"].noul == 0.1
    assert model.answer(request(), repeat=1).answers["relevant"].noul == 0.9
    assert model.answer(request(), repeat=0).answers["relevant"].noul == 0.1
    assert client.calls == 2


def test_a_failed_call_is_not_cached(cache):
    busy = JevError("HTTP 500", retryable=True)
    model, _, _ = jev(cache, *[busy] * (RETRIES + 1), {"relevant": {"type": "noul", "noul": 0.5}})
    with pytest.raises(JevError):
        model.answer(request())
    assert len(cache) == 0
    assert model.answer(request()).answers["relevant"].noul == 0.5


def test_without_a_client_only_cached_answers_are_served(cache):
    model, _, _ = jev(cache, {"relevant": {"type": "noul", "noul": 0.5}})
    model.answer(request())

    replay = CachedJev(cache)
    assert replay.answer(request()).answers["relevant"].noul == 0.5
    with pytest.raises(JevError):
        replay.answer(request("something new"))


def test_a_retryable_failure_is_retried_then_succeeds(cache):
    model, _, waits = jev(
        cache, JevError("HTTP 503", retryable=True), {"relevant": {"type": "noul", "noul": 0.7}}
    )
    answer = model.answer(request())
    assert answer.answers["relevant"].noul == 0.7
    assert waits == [RETRY_WAITS[0]]


def test_a_non_retryable_failure_is_not_retried(cache):
    model, client, waits = jev(cache, JevError("HTTP 401"), {"relevant": {"type": "noul", "noul": 0.7}})
    with pytest.raises(JevError):
        model.answer(request())
    assert client.calls == 1
    assert waits == []


def test_gives_up_after_the_allowed_retries(cache):
    busy = JevError("HTTP 503", retryable=True)
    model, client, _ = jev(cache, *[busy] * (RETRIES + 2))
    with pytest.raises(JevError):
        model.answer(request())
    assert client.calls == RETRIES + 1
