import hashlib
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CONFIG_ROOT = Path("configs")


class Coefficients(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rating: float = Field(ge=0)
    reports: float = Field(ge=0)
    news: float = Field(ge=0)


class InputsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    news_lookback_days: int = Field(gt=0)
    scoring_config: str = Field(min_length=1)
    report_spec: str = Field(min_length=1)


class StocksConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vol_lookback: int = Field(gt=1)
    max_weight: float = Field(gt=0, le=1)
    # Divide each stock's tilt by its volatility (task 7.2 made it a key, so the
    # exposure-only control is a config too). On by default: v1 means what it meant.
    vol_scaling: bool = True


class ExposureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str = Field(min_length=1)
    vol_lookback: int = Field(gt=1)
    target_vol: float = Field(gt=0)
    floor: float = Field(ge=0, le=1)
    trend_filter: bool
    trend_window: int = Field(gt=1)
    trend_exposure: float = Field(ge=0, le=1)


class SizingConfig(BaseModel):
    """One version of everything about sizing that may be changed and compared."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str  # the file's name
    digest: str  # short hash of the file, so an edit in place does not pass as the same version
    k: float = Field(ge=0)
    coefficients: Coefficients
    inputs: InputsConfig
    stocks: StocksConfig
    exposure: ExposureConfig


def load_sizing_config(version: str = "v1", root: Path = CONFIG_ROOT) -> SizingConfig:
    """Read `<root>/sizing/<version>.toml`. A misspelt or unknown key is an error."""
    raw = (root / "sizing" / f"{version}.toml").read_bytes()
    return SizingConfig(
        version=version,
        digest=hashlib.sha256(raw).hexdigest()[:8],
        **tomllib.loads(raw.decode()),
    )
