import hashlib
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sluice.agents import indicators

CONFIG_ROOT = Path("configs")
# The analysts a config may pick (docs/specs/10-configurable-components.md), in the
# order their reports are keyed; the same names as their report channels.
ANALYST_KEYS = ("market", "news", "fundamentals")
AnalystKey = Literal["market", "news", "fundamentals"]


def _no_repeats(names: tuple[str, ...], what: str) -> None:
    if not names:
        raise ValueError(f"{what} is empty; leave it out for the default")
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"{what} names {', '.join(repeated)} more than once")


class NewsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lookback_days: int = Field(gt=0)
    max_items: int = Field(gt=0)
    # Weighted news (docs/specs/07-agent-pipeline.md, task 5.1). Off by default, so a
    # config file written before it means what it always meant.
    weighted: bool = False
    min_relevance: float = Field(default=0.5, ge=0, le=1)  # Jev relevance below it is left out
    scoring_config: str = "v1"  # whose scores and weight tables are used


class PricesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lookback_days: int = Field(gt=0)  # trading days of history the indicators are computed on
    benchmark: str = Field(min_length=1)
    # Which indicators the market analyst sees (task 11.3). None: the original eleven, so a
    # config written before it means what it always meant.
    indicators: tuple[str, ...] | None = None

    @field_validator("indicators")
    @classmethod
    def _known_indicators(cls, names: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if names is None:
            return None
        _no_repeats(names, "indicators")
        if unknown := sorted(set(names) - set(indicators.ALL)):
            raise ValueError(f"unknown indicators: {', '.join(unknown)}")
        return names

    @property
    def selected_indicators(self) -> tuple[str, ...]:
        """The indicators shown, in the registry's order whatever the order written."""
        chosen = set(self.indicators or indicators.NAMES)
        return tuple(name for name in indicators.ALL if name in chosen)


class FundamentalsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lag_days: int = Field(ge=0)  # a quarter counts as published this long after it ends
    quarters: int = Field(gt=0)  # how many of the latest visible quarters are shown


class ToolsConfig(BaseModel):
    """The rules of the analysts' tool loop (docs/specs/08-agent-harness.md, task 9.3).

    A config without a [tools] section prefetches, as before; with one, the
    analysts call tools within these limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_calls: int = Field(default=6, ge=0)  # per analyst; then it must write its report
    followup_rounds: int = Field(default=1, ge=0)  # the research manager sending questions back
    followup_calls: int = Field(default=3, ge=0)  # per analyst answering one
    history_max_days: int = Field(default=250, gt=0)  # get_price_history
    search_max_limit: int = Field(default=20, gt=0)  # items per search_news call
    # Send the model's thinking back with its tool calls; off, since answers came back
    # right without it in the task 9.2 check, and it costs input tokens every step.
    pass_reasoning: bool = False


class PipelineConfig(BaseModel):
    """One version of everything about the pipeline that may be changed and compared."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str  # the file's name
    digest: str  # short hash of the file, so an edit in place does not pass as the same version
    debate_rounds: int = Field(ge=0)  # one round is a bull turn then a bear turn; 0 skips it
    temperature: float = Field(ge=0)
    retries: int = Field(ge=0)
    models: dict[str, str]  # role -> model
    prices: PricesConfig
    news: NewsConfig
    fundamentals: FundamentalsConfig
    prompts: dict[str, str]  # role -> system prompt
    tools: ToolsConfig | None = None  # None: the data is prefetched, as in v1 and v2
    analysts: tuple[AnalystKey, ...] | None = None  # None: all three (task 11.3)

    @field_validator("analysts")
    @classmethod
    def _analysts_once_each(cls, keys: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if keys is not None:
            _no_repeats(keys, "analysts")
        return keys

    @model_validator(mode="after")
    def _choices_fit_together(self) -> Self:
        chosen = self.selected_analysts
        if self.prices.indicators is not None and "market" not in chosen:
            raise ValueError("[prices] indicators is set but the market analyst is not chosen")
        if self.news.weighted and "news" not in chosen:
            raise ValueError("[news] weighted is on but the news analyst is not chosen")
        if self.tools is not None and (
            chosen != ANALYST_KEYS or self.prices.indicators is not None
        ):
            raise ValueError("a config with [tools] takes all three analysts and no indicators")
        return self

    @property
    def selected_analysts(self) -> tuple[str, ...]:
        """The analysts that run, in the order of `ANALYST_KEYS`."""
        chosen = set(self.analysts or ANALYST_KEYS)
        return tuple(key for key in ANALYST_KEYS if key in chosen)


def load_config(version: str = "v1", root: Path = CONFIG_ROOT) -> PipelineConfig:
    """Read `<root>/pipeline/<version>.toml`. A misspelt or unknown key is an error."""
    raw = (root / "pipeline" / f"{version}.toml").read_bytes()
    return PipelineConfig(
        version=version,
        digest=hashlib.sha256(raw).hexdigest()[:8],
        **tomllib.loads(raw.decode()),
    )
