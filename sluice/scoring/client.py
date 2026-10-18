import hashlib
import json
import time
from collections.abc import Callable
from typing import Annotated, Literal, Protocol

import requests
from pydantic import BaseModel, ConfigDict, Field

from sluice.agents.cache import CallCache

ENV_KEY = "TYPESAFE_API_KEY"
BASE_URL = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"
TIMEOUT = 60  # seconds
# Default retries per answer, shared with the DeepSeek client for consistency.
RETRIES = 2
RETRY_WAITS = (2.0, 4.0)
PRICE_PER_MILLION_INPUT = 0.042  # USD; output tokens are free


class JevError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False, too_large: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        # HTTP 422 for a state/question over Jev's 64k/32k limit; the caller (not this
        # module) decides whether to shrink the state and ask again.
        self.too_large = too_large


class JevQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["noul", "choice"]
    instructions: str
    criteria: dict[str, str] | None = None


class JevRequest(BaseModel):
    """Everything that decides the answer. Two equal requests share one cache entry."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    state: str = Field(min_length=1)
    questions: dict[str, JevQuestion] = Field(min_length=1)


class NoulAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)  # calibrated probability; no separate confidence


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)


JevAnswer = Annotated[NoulAnswer | ChoiceAnswer, Field(discriminator="type")]


class JevResponse(BaseModel):
    answers: dict[str, JevAnswer]
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # True when no call was made for this answer


class Answerer(Protocol):
    def answer(self, request: JevRequest) -> JevResponse: ...


def cache_key(request: JevRequest, repeat: int = 0) -> str:
    """Hash of the request plus which sample of it this is, as with the LLM cache."""
    payload = {"request": request.model_dump(mode="json"), "repeat": repeat}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class JevClient:
    """One /v1/systemone call per call, no retries. The key travels in a header."""

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        post: Callable[..., requests.Response] = requests.post,
    ) -> None:
        self._api_key = api_key
        self._url = base_url.rstrip("/") + ENDPOINT
        self._post = post

    def answer(self, request: JevRequest) -> JevResponse:
        body = {
            "model": request.model,
            "state": request.state,
            "questions": {
                name: q.model_dump(exclude_none=True) for name, q in request.questions.items()
            },
        }
        try:
            response = self._post(
                self._url,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise JevError(f"{type(exc).__name__}: {exc}".rstrip(": "), retryable=True) from exc
        status = response.status_code
        if status == 422:
            text = response.text[:200]
            too_large = any(s in text.lower() for s in ("too large", "context", "32", "64"))
            raise JevError(f"HTTP 422: {text}", retryable=False, too_large=too_large)
        if status != 200:
            # Rate limits and server faults pass; a bad key, model or request does not.
            raise JevError(
                f"HTTP {status}: {response.text[:200]}", retryable=status == 429 or status >= 500
            )
        try:
            data = response.json()
            raw_answers = data["answers"]
            missing = set(request.questions) - set(raw_answers)
            if missing:
                raise KeyError(f"missing answers for {sorted(missing)}")
            usage = data.get("usage") or {}
            return JevResponse(
                answers=raw_answers,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise JevError(f"unexpected answer: {response.text[:200]}", retryable=True) from exc


class CachedJev:
    """The only way scoring reaches Jev: cache first, the client on a miss.

    Without a client it only replays, so re-scoring on cached answers after a
    downstream change (weights, thresholds) cannot spend anything.
    """

    def __init__(
        self,
        cache: CallCache,
        client: Answerer | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = cache
        self._client = client
        self._sleep = sleep

    def answer(self, request: JevRequest, repeat: int = 0) -> JevResponse:
        """The cached answer, or one call whose answer is then cached. Failures are not stored.

        Retries transient failures up to `RETRIES` times. A non-retryable or
        exhausted failure raises `JevError`; marking the item as missing and
        moving on is the caller's job, not this method's.
        """
        key = cache_key(request, repeat)
        stored = self._cache.get(key)
        if stored is not None:
            return JevResponse.model_validate_json(stored).model_copy(update={"cached": True})
        if self._client is None:
            raise JevError("not in the cache, and no client to ask")
        failures = 0
        while True:
            try:
                response = self._client.answer(request)
            except JevError as exc:
                if not exc.retryable or failures >= RETRIES:
                    raise
                self._sleep(RETRY_WAITS[min(failures, len(RETRY_WAITS) - 1)])
                failures += 1
                continue
            self._cache.put(
                key, request.model_dump_json(), response.model_dump_json(exclude={"cached"})
            )
            return response
