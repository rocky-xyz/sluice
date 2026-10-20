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
    ToolCall,
    ToolSpec,
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


def test_the_number_of_retries_can_be_set(cache):
    model, client, _ = llm(cache, *["not json"] * 5)
    with pytest.raises(LLMError):
        model.complete_checked(request(), parse, retries=0)
    assert client.calls == 1

    busy = LLMError("HTTP 503", retryable=True)
    model, client, waits = llm(cache, busy, busy, busy, GOOD)
    verdict, _ = model.complete_checked(request("Another."), parse, retries=3)
    assert verdict == Verdict(rating="hold")
    assert waits == [RETRY_WAITS[0], RETRY_WAITS[1], RETRY_WAITS[1]]


def test_giving_up_hands_back_the_answers_already_received(cache):
    model, _, _ = llm(cache, "not json", "still not json", "nor this")
    with pytest.raises(LLMError) as raised:
        model.complete_checked(request(), parse)
    assert [a.text for a in raised.value.answers] == ["not json", "still not json", "nor this"]


# --- function calling (task 9.2) ---

TOOL = ToolSpec(
    name="get_indicators",
    description="Price indicators as of the decision time.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)
CALL = ToolCall(id="call_1", name="get_indicators", arguments="{}")


def tool_loop_request(**overrides) -> LLMRequest:
    messages = (
        Message(role="system", content="Use the tool."),
        Message(role="user", content="How is NVDA doing?"),
        Message(role="assistant", content="", tool_calls=(CALL,)),
        Message(role="tool", content='{"P:rsi_14": 61}', tool_call_id="call_1"),
    )
    fields = {"model": "deepseek-flash", "messages": messages, "tools": (TOOL,)}
    return LLMRequest(**(fields | overrides))


def test_requests_made_before_function_calling_keep_their_cache_keys():
    # Computed with the code before task 9.2; 117k answers are cached under such keys.
    old = LLMRequest(
        model="deepseek-flash",
        messages=(
            Message(role="system", content="Be brief."),
            Message(role="user", content="Say hi."),
        ),
        temperature=0.0,
        json_mode=True,
    )
    assert cache_key(old) == "a4d98fe6993f02a3d37c65df4407730b3eac46ac15cd72134769fa5867ad775d"
    assert cache_key(old, 2, 1) == (
        "59f4d3fe79aa2a9abd5521c043099763f551bcb7371de2de3a9fd693f0600f24"
    )


def test_tools_tool_choice_and_each_turn_of_the_loop_change_the_key():
    base = cache_key(tool_loop_request())
    without_result = tool_loop_request().model_copy(
        update={"messages": tool_loop_request().messages[:3]}
    )
    assert len({
        base,
        cache_key(tool_loop_request(tool_choice="none")),
        cache_key(tool_loop_request(tools=())),
        cache_key(without_result),
    }) == 4  # fmt: skip


def test_client_sends_tools_calls_and_results_in_the_api_format():
    http = FakeHTTP((200, completion("NVDA looks strong [P:rsi_14].")))
    DeepSeekClient("sk", post=http).complete(tool_loop_request(tool_choice="none"))
    body = http.calls[0][1]["json"]
    assert body["tools"] == [{"type": "function", "function": TOOL.model_dump()}]
    assert body["tool_choice"] == "none"
    assert body["messages"][2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_indicators", "arguments": "{}"}},
        ],
    }  # fmt: skip
    assert body["messages"][3] == {
        "role": "tool", "content": '{"P:rsi_14": 61}', "tool_call_id": "call_1",
    }  # fmt: skip
    assert body["messages"][0] == {"role": "system", "content": "Use the tool."}


def test_client_leaves_tools_out_when_there_are_none():
    http = FakeHTTP((200, completion()))
    DeepSeekClient("sk", post=http).complete(request())
    assert "tools" not in http.calls[0][1]["json"]
    assert "tool_choice" not in http.calls[0][1]["json"]


def test_client_reads_the_calls_the_model_asks_for():
    asked = {
        "id": "call_9", "type": "function",
        "function": {"name": "search_news", "arguments": '{"query": "export", "limit": 5}'},
    }  # fmt: skip
    answer = completion("", tool_calls=[asked])
    answer["choices"][0]["finish_reason"] = "tool_calls"
    response = DeepSeekClient("sk", post=FakeHTTP((200, answer))).complete(tool_loop_request())
    assert response.tool_calls == (
        ToolCall(id="call_9", name="search_news", arguments='{"query": "export", "limit": 5}'),
    )
    assert (response.text, response.finish_reason) == ("", "tool_calls")


def test_tool_calls_survive_the_cache(cache):
    class Calling:
        calls = 0

        def complete(self, request):
            self.calls += 1
            return LLMResponse(text="", tool_calls=(CALL,), finish_reason="tool_calls")

    client = Calling()
    llm_ = CachedLLM(cache, client)
    first = llm_.complete(tool_loop_request())
    second = llm_.complete(tool_loop_request())
    assert second.cached and client.calls == 1
    assert second.tool_calls == first.tool_calls == (CALL,)
    assert second.finish_reason == "tool_calls"
