import hashlib
import json
import time
from collections.abc import Callable
from typing import Literal, Protocol, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field

from sluice.agents.cache import CallCache

ENV_KEY = "DEEPSEEK_API_KEY"
BASE_URL = "https://api.deepseek.com"
TIMEOUT = 300  # seconds; the deep model can think for minutes
# Default retries per answer, shared by failed requests and answers that fail the caller's check.
RETRIES = 2
# Seconds before re-sending a failed request; the last one repeats when there are more retries.
RETRY_WAITS = (2.0, 4.0)

T = TypeVar("T")


class LLMError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        # Answers received before giving up; they were paid for and belong in the usage record.
        self.answers: list[LLMResponse] = []


class ToolCall(BaseModel):
    """One function call the model asked for; `arguments` is its JSON text, as sent."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str


class ToolSpec(BaseModel):
    """A function the model may call: its name, what it does, its parameters' JSON Schema."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str
    parameters: dict


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # The model's function calls, on an assistant message (task 9.2).
    tool_calls: tuple[ToolCall, ...] = ()
    # Which call a tool message answers.
    tool_call_id: str | None = None
    # The thinking that came with an assistant message, sent back when the API asks for it.
    reasoning_content: str | None = None


class LLMRequest(BaseModel):
    """Everything that decides the answer. Two equal requests share one cache entry."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    temperature: float = 0.0
    # DeepSeek only honours JSON mode when the prompt itself mentions "json".
    json_mode: bool = False
    tools: tuple[ToolSpec, ...] = ()
    # "none" makes the model answer in text although tools are listed: the last turn of a loop.
    tool_choice: Literal["auto", "none"] | None = None


class LLMResponse(BaseModel):
    text: str
    reasoning: str | None = None  # the deep model's thinking, returned apart from the answer
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # True when no call was made for this answer
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None  # "stop", "tool_calls", "length", ...


class Completer(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


def cache_key(request: LLMRequest, repeat: int = 0, attempt: int = 0) -> str:
    """Hash of the request plus which sample of it this is.

    `repeat` separates deliberate re-runs of the same grid cell; `attempt`
    separates retries after an answer was rejected. Without them every re-run
    and every retry would get the first answer back.
    """
    payload = {"request": _keyed(request), "repeat": repeat, "attempt": attempt}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# Fields added for function calling (task 9.2), left out of the key while empty, so that every
# request made before them keeps the key it was cached under.
_REQUEST_EXTRAS = {"tools": (), "tool_choice": None}
_MESSAGE_EXTRAS = {"tool_calls": (), "tool_call_id": None, "reasoning_content": None}


def _keyed(request: LLMRequest) -> dict:
    dumped = request.model_dump(mode="json")
    for field, empty in _REQUEST_EXTRAS.items():
        if getattr(request, field) == empty:
            del dumped[field]
    for message, out in zip(request.messages, dumped["messages"], strict=True):
        for field, empty in _MESSAGE_EXTRAS.items():
            if getattr(message, field) == empty:
                del out[field]
    return dumped


def _wire(message: Message) -> dict:
    """A message as the OpenAI-style chat API takes it."""
    out: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.reasoning_content is not None:
        out["reasoning_content"] = message.reasoning_content
    return out


class DeepSeekClient:
    """One chat completion per call, no retries. The key travels in a header."""

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        post: Callable[..., requests.Response] = requests.post,
    ) -> None:
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._post = post

    def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, object] = {
            "model": request.model,
            "messages": [_wire(m) for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": spec.model_dump()} for spec in request.tools
            ]
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        try:
            response = self._post(
                self._url,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}".rstrip(": "), retryable=True) from exc
        status = response.status_code
        if status != 200:
            # Rate limits and server faults pass; a bad key, model or request does not.
            raise LLMError(
                f"HTTP {status}: {response.text[:200]}", retryable=status == 429 or status >= 500
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage") or {}
            calls = tuple(
                ToolCall(
                    id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"]
                )
                for c in message.get("tool_calls") or []
            )
            return LLMResponse(
                text=message.get("content") or "",
                reasoning=message.get("reasoning_content") or None,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                tool_calls=calls,
                finish_reason=choice.get("finish_reason"),
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"unexpected answer: {response.text[:200]}", retryable=True) from exc


class CachedLLM:
    """The only way roles reach the model: cache first, the client on a miss.

    Without a client it only replays, so re-running downstream code on cached
    answers cannot spend anything.
    """

    def __init__(
        self,
        cache: CallCache,
        client: Completer | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = cache
        self._client = client
        self._sleep = sleep

    def complete(self, request: LLMRequest, repeat: int = 0, attempt: int = 0) -> LLMResponse:
        """The cached answer, or one call whose answer is then cached. Failures are not stored."""
        key = cache_key(request, repeat, attempt)
        stored = self._cache.get(key)
        if stored is not None:
            return LLMResponse.model_validate_json(stored).model_copy(update={"cached": True})
        if self._client is None:
            raise LLMError("not in the cache, and no client to ask")
        response = self._client.complete(request)
        self._cache.put(
            key, request.model_dump_json(), response.model_dump_json(exclude={"cached"})
        )
        return response

    def complete_checked(
        self,
        request: LLMRequest,
        check: Callable[[LLMResponse], T],
        repeat: int = 0,
        retries: int = RETRIES,
    ) -> tuple[T, list[LLMResponse]]:
        """Ask until `check` accepts an answer, with at most `retries` retries in total.

        `check` turns an answer into what the caller needs and raises ValueError
        (pydantic's ValidationError is one) to reject it. Returns its result and
        every answer received, rejected ones included, for usage accounting.

        A rejected answer stays cached under its attempt number, so a re-run
        replays the same rejections and the same final answer without new calls.
        """
        answers: list[LLMResponse] = []
        failures = 0
        attempt = 0
        while True:
            try:
                response = self.complete(request, repeat, attempt)
            except LLMError as exc:
                if not exc.retryable or failures >= retries:
                    exc.answers = answers
                    raise
                self._sleep(RETRY_WAITS[min(failures, len(RETRY_WAITS) - 1)])
                failures += 1
                continue
            answers.append(response)
            try:
                return check(response), answers
            except ValueError as exc:
                if failures >= retries:
                    error = LLMError(f"no acceptable answer after {attempt + 1} tries: {exc}")
                    error.answers = answers
                    raise error from exc
                failures += 1
                attempt += 1
