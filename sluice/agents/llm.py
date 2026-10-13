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


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Everything that decides the answer. Two equal requests share one cache entry."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    temperature: float = 0.0
    # DeepSeek only honours JSON mode when the prompt itself mentions "json".
    json_mode: bool = False


class LLMResponse(BaseModel):
    text: str
    reasoning: str | None = None  # the deep model's thinking, returned apart from the answer
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # True when no call was made for this answer


class Completer(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


def cache_key(request: LLMRequest, repeat: int = 0, attempt: int = 0) -> str:
    """Hash of the request plus which sample of it this is.

    `repeat` separates deliberate re-runs of the same grid cell; `attempt`
    separates retries after an answer was rejected. Without them every re-run
    and every retry would get the first answer back.
    """
    payload = {"request": request.model_dump(mode="json"), "repeat": repeat, "attempt": attempt}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


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
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
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
            message = data["choices"][0]["message"]
            usage = data.get("usage") or {}
            return LLMResponse(
                text=message.get("content") or "",
                reasoning=message.get("reasoning_content") or None,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
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
