import hashlib
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CONFIG_ROOT = Path("configs")


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


class FundamentalsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lag_days: int = Field(ge=0)  # a quarter counts as published this long after it ends
    quarters: int = Field(gt=0)  # how many of the latest visible quarters are shown


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


def load_config(version: str = "v1", root: Path = CONFIG_ROOT) -> PipelineConfig:
    """Read `<root>/pipeline/<version>.toml`. A misspelt or unknown key is an error."""
    raw = (root / "pipeline" / f"{version}.toml").read_bytes()
    return PipelineConfig(
        version=version,
        digest=hashlib.sha256(raw).hexdigest()[:8],
        **tomllib.loads(raw.decode()),
    )
