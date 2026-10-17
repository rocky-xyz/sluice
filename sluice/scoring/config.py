import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_ROOT = Path("configs")


class QuestionSpec(BaseModel):
    """One Jev question, in the shape TypeSafe's /v1/systemone API expects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["noul", "choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | None = None  # choice only; a noul question has none

    @model_validator(mode="after")
    def _criteria_matches_type(self) -> "QuestionSpec":
        if self.type == "choice" and not self.criteria:
            raise ValueError("a choice question needs criteria")
        if self.type == "noul" and self.criteria is not None:
            raise ValueError("a noul question takes no criteria")
        return self


class ScoringConfig(BaseModel):
    """One version of the Jev problem spec: which questions, on which model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str  # the file's name
    digest: str  # short hash of the file, so an edit in place does not pass as the same version
    model: str = Field(min_length=1)
    novelty_reference_k: int = Field(gt=0)
    questions: dict[str, QuestionSpec] = Field(min_length=1)


def load_scoring_config(version: str = "v1", root: Path = CONFIG_ROOT) -> ScoringConfig:
    """Read `<root>/scoring/<version>.toml`. A misspelt or unknown key is an error."""
    raw = (root / "scoring" / f"{version}.toml").read_bytes()
    return ScoringConfig(
        version=version,
        digest=hashlib.sha256(raw).hexdigest()[:8],
        **tomllib.loads(raw.decode()),
    )
