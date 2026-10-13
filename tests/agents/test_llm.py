import json

import pytest
import requests
from pydantic import BaseModel

from sluice.agents.cache import CallCache
from sluice.agents.llm import (
    RETRIES,
    RETRY_WAITS,
    CachedLLM,
    DeepSeekClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    cache_key,
)


def request(text: str = "Say hi.", **overrides) -> LLMRequest:
    fields = {"model": "deepseek-flash", "messages": (Message(role="user", content=text),)}
    fields.update(overrides)
    return LLMRequest(**fields)


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


def completion(text: str = "hi", **message) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text, **message}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


class FakeClient:
    """Stands in for DeepSeekClient: answers in order, counts calls."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return LLMResponse(text=answer, input_tokens=10, output_tokens=2)


@pytest.fixture
def cache(tmp_path):
    with CallCache(tmp_path / "llm.db") as cache:
        yield cache


def llm(cache, *answers):
    client = FakeClient(*answers)
    waits = []
    return CachedLLM(cache, client, sleep=waits.append), client, waits


# --- cache key ---


def test_key_is_stable_and_depends_on_every_field():
    base = cache_key(request())
    assert base == cache_key(request())
    assert len(
        {
            base,
            cache_key(request("Say bye.")),
            cache_key(request(model="deepseek-v4-pro")),
            cache_key(request(temperature=0.7)),
            cache_key(request(json_mode=True)),
            cache_key(request(), repeat=1),
            cache_key(request(), attempt=1),
        }
    ) == 7


# --- DeepSeek client ---


def test_client_sends_the_request_and_reads_the_answer():
    http = FakeHTTP((200, completion("hi", reasoning_content="thinking")))
    answer = DeepSeekClient("secret", post=http).complete(request(json_mode=True))

    assert answer == LLMResponse(text="hi", reasoning="thinking", input_tokens=12, output_tokens=3)
    url, sent = http.calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer secret"}
    assert sent["json"] == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "Say hi."}],
        "temperature": 0.0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def test_client_leaves_out_response_format_by_default():
    http = FakeHTTP((200, completion()))
    DeepSeekClient("secret", post=http).complete(request())
    assert "response_format" not in http.calls[0][1]["json"]


@pytest.mark.parametrize(
    ("answer", "retryable"),
    [
        ((429, "slow down"), True),
        ((500, "oops"), True),
        ((503, "busy"), True),
        ((400, "bad request"), False),
        ((401, "bad key"), False),
        ((402, "no balance"), False),
        ((200, "not json"), True),
        ((200, {"choices": []}), True),
        (requests.ConnectionError("down"), True),
        (requests.Timeout(), True),
    ],
)
def test_client_says_which_failures_may_pass(answer, retryable):
    with pytest.raises(LLMError) as raised:
        DeepSeekClient("secret", post=FakeHTTP(answer)).complete(request())
    assert raised.value.retryable is retryable


def test_error_text_does_not_contain_the_key():
    with pytest.raises(LLMError) as raised:
        DeepSeekClient("secret", post=FakeHTTP((401, "bad key"))).complete(request())
    assert "secret" not in str(raised.value)


# --- cached calls ---


def test_second_identical_call_comes_from_the_cache(cache):
    model, client, _ = llm(cache, "hi")
    first = model.complete(request())
    second = model.complete(request())

    assert client.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.model_copy(update={"cached": False}) == first


def test_a_different_repeat_is_a_new_call(cache):
    model, client, _ = llm(cache, "one", "two")
    assert model.complete(request(), repeat=0).text == "one"
    assert model.complete(request(), repeat=1).text == "two"
    assert model.complete(request(), repeat=0).text == "one"
    assert client.calls == 2


def test_a_failed_call_is_not_cached(cache):
    model, _, _ = llm(cache, LLMError("HTTP 500", retryable=True), "hi")
    with pytest.raises(LLMError):
        model.complete(request())
    assert len(cache) == 0
    assert model.complete(request()).text == "hi"


def test_without_a_client_only_cached_answers_are_served(cache):
    model, _, _ = llm(cache, "hi")
    model.complete(request())

    replay = CachedLLM(cache)
    assert replay.complete(request()).text == "hi"
    with pytest.raises(LLMError):
        replay.complete(request("Something new."))


# --- checked calls ---


class Verdict(BaseModel):
    rating: str


def parse(response: LLMResponse) -> Verdict:
    return Verdict.model_validate_json(response.text)


GOOD = '{"rating": "hold"}'


def test_a_rejected_answer_is_asked_again(cache):
    model, client, waits = llm(cache, "not json", GOOD)
    verdict, answers = model.complete_checked(request(), parse)

    assert verdict == Verdict(rating="hold")
    assert [a.text for a in answers] == ["not json", GOOD]
    assert client.calls == 2
    assert waits == []  # nothing to wait for: the request itself went through


def test_a_rerun_replays_rejections_without_new_calls(cache):
    model, client, _ = llm(cache, "not json", GOOD)
    first = model.complete_checked(request(), parse)

    verdict, answers = CachedLLM(cache).complete_checked(request(), parse)
    assert verdict == first[0]
    assert [a.text for a in answers] == ["not json", GOOD]
    assert all(a.cached for a in answers)
    assert client.calls == 2


def test_gives_up_after_the_allowed_retries(cache):
    model, client, _ = llm(cache, *["not json"] * (RETRIES + 2))
    with pytest.raises(LLMError):
        model.complete_checked(request(), parse)
    assert client.calls == RETRIES + 1


def test_a_failed_request_is_sent_again_after_a_wait(cache):
    model, _, waits = llm(cache, LLMError("HTTP 503", retryable=True), GOOD)
    verdict, answers = model.complete_checked(request(), parse)

    assert verdict == Verdict(rating="hold")
    assert len(answers) == 1
    assert waits == [RETRY_WAITS[0]]


def test_failed_requests_and_rejections_share_one_budget(cache):
    busy = LLMError("HTTP 503", retryable=True)
    model, client, _ = llm(cache, busy, "not json", "still not json", GOOD)
    with pytest.raises(LLMError):
        model.complete_checked(request(), parse)
    assert client.calls == RETRIES + 1


def test_a_failure_that_cannot_pass_is_not_retried(cache):
    model, client, waits = llm(cache, LLMError("HTTP 401"), GOOD)
    with pytest.raises(LLMError):
        model.complete_checked(request(), parse)
    assert client.calls == 1
    assert waits == []
