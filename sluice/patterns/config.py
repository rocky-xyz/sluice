import hashlib
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CONFIG_ROOT = Path("configs")


class _Rules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CuorouRules(_Rules):
    shadow_to_body: float = Field(gt=0)
    shadow_to_range: float = Field(gt=0, le=1)
    max_volume_vs_prev: float = Field(gt=0)
    max_volume_vs_avg5: float = Field(gt=0)
    above_sma: int = Field(gt=1)
    max_abs_ret20: float = Field(gt=0)


class HaramiRules(_Rules):
    mother_body_to_avg: float = Field(gt=0)
    mother_lower_to_range: float = Field(gt=0, le=1)
    baby_to_mother_range: float = Field(gt=0, le=1)
    daily_below_sma: int = Field(gt=1)
    weekly_bearish_weeks: int = Field(ge=1)


class WedgeRules(_Rules):
    min_pivots: int = Field(ge=2)
    max_pivots: int = Field(ge=2)
    min_days: int = Field(gt=0)
    max_days: int = Field(gt=0)
    middle_tolerance: float = Field(ge=0)
    max_gaps: int = Field(ge=0)
    gap: float = Field(gt=0)
    breakout: float = Field(ge=0)
    breakout_volume: float = Field(gt=0)
    market_sma: int = Field(gt=1)
    hold_days: int = Field(ge=1)
    retest_days: int = Field(ge=1)
    retest_band: float = Field(ge=0)
    doji_body: float = Field(gt=0, le=1)
    doji_volume: float = Field(gt=0)


class PatternConfig(BaseModel):
    """One version of every pattern's thresholds (docs/specs/09-patterns.md)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    digest: str  # short hash of the file
    cuorou: CuorouRules
    harami: HaramiRules
    wedge: WedgeRules


def load_pattern_config(version: str = "v1", root: Path = CONFIG_ROOT) -> PatternConfig:
    raw = (root / "patterns" / f"{version}.toml").read_bytes()
    return PatternConfig(
        version=version, digest=hashlib.sha256(raw).hexdigest()[:8], **tomllib.loads(raw.decode())
    )
