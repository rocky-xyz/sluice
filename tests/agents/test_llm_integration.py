import os

import pytest
from dotenv import load_dotenv

from sluice.agents.cache import CallCache
from sluice.agents.llm import ENV_KEY, CachedLLM, DeepSeekClient, LLMRequest, Message, ToolSpec

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


INDICATORS = ToolSpec(
    name="get_indicators",
    description="Price indicators of the stock as of the decision time.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
def test_real_tool_call_round_trip(tmp_path, model):
    """The model asks for the tool, gets its result, answers from it (task 9.2)."""
    load_dotenv()
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        pytest.skip(f"{ENV_KEY} is not set")
    asking = (
        Message(role="system", content="Answer only from tool results."),
        Message(role="user", content="What is NVDA's 14-day RSI? Use the tool."),
    )
    with CallCache(tmp_path / "llm.db") as cache:
        llm = CachedLLM(cache, DeepSeekClient(key))
        first = llm.complete(LLMRequest(model=model, messages=asking, tools=(INDICATORS,)))
        assert first.tool_calls and first.tool_calls[0].name == "get_indicators"
        turn = Message(
            role="assistant", content=first.text, tool_calls=first.tool_calls,
            reasoning_content=first.reasoning,
        )  # fmt: skip
        result = Message(
            role="tool", content='{"P:rsi_14": 61.3}', tool_call_id=first.tool_calls[0].id
        )
        follow = LLMRequest(
            model=model, messages=(*asking, turn, result), tools=(INDICATORS,), tool_choice="none"
        )
        second = llm.complete(follow)
        replay = llm.complete(follow)
    print(f"\n{model}: reasoning returned {first.reasoning is not None}; "
          f"answer {second.text!r}; finish {second.finish_reason}")  # fmt: skip
    assert "61" in second.text and not second.tool_calls
    assert replay.cached
