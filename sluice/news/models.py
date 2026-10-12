from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

TimePrecision = Literal["exact", "day", "missing"]
Session = Literal["pre", "regular", "post", "closed"]


class NewsItem(BaseModel):
    """One item as handed over by a collector, before it is stored."""

    title: str = Field(min_length=1)
    source: str = Field(min_length=1)  # data vendor, e.g. "yfinance"
    tickers: list[str] = Field(min_length=1)
    fetched_at: AwareDatetime
    url: str | None = None
    summary: str | None = None
    publisher: str | None = None
    published_at: AwareDatetime | None = None
    time_precision: TimePrecision = "exact"

    @field_validator("tickers")
    @classmethod
    def _upper_tickers(cls, value: list[str]) -> list[str]:
        tickers = sorted({t.strip().upper() for t in value})
        if not all(tickers):
            raise ValueError("empty ticker")
        return tickers

    @model_validator(mode="after")
    def _precision_matches_timestamp(self) -> "NewsItem":
        if (self.published_at is None) != (self.time_precision == "missing"):
            raise ValueError("time_precision must be 'missing' exactly when published_at is None")
        return self


class StoredNews(BaseModel):
    """One article as read back from the store. Timestamps are UTC."""

    id: int
    url: str | None
    title: str
    summary: str | None
    publisher: str | None
    source: str
    published_at: AwareDatetime | None
    time_precision: TimePrecision
    fetched_at: AwareDatetime
    tickers: list[str]
    reprint_count: int = 0  # reprints published before the query's decision time
    session: Session | None = None  # part of the trading day; None without an exact time


class AddResult(BaseModel):
    inserted: int = 0
    skipped: int = 0  # already stored; any new tickers were still linked
