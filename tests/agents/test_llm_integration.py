import os

import pytest
from dotenv import load_dotenv

from sluice.agents.cache import CallCache
from sluice.agents.llm import ENV_KEY, CachedLLM, DeepSeekClient, LLMRequest, Message

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
def test_real_call_then_cache(tmp_path, model):
    load_dotenv()
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        pytest.skip(f"{ENV_KEY} is not set")
    request = LLMRequest(
        model=model, messages=(Message(role="user", content="Reply with the single word: ok"),)
    )
    with CallCache(tmp_path / "llm.db") as cache:
        llm = CachedLLM(cache, DeepSeekClient(key))
        first = llm.complete(request)
        second = llm.complete(request)

    assert "ok" in first.text.lower()
    assert first.input_tokens > 0 and first.output_tokens > 0
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text
