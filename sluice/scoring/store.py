import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PATH = Path("data/local/scores.db")

SCHEMA_VERSION = 1

# No migrations: scores can always be rebuilt by rerunning `sluice score`, which
# skips news already scored under a config version and only re-asks Jev for new
# items or a new config version.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    news_id                  INTEGER NOT NULL,
    config_version           TEXT NOT NULL,
    config_digest             TEXT NOT NULL,
    skipped                  TEXT,  -- NULL when scored; else a short reason (e.g. state_too_large)
    relevance                 REAL,
    stance                    TEXT,
    stance_confidence         REAL,
    stance_probabilities      TEXT,  -- JSON {choice: probability}
    novelty                   REAL,
    rumour                    REAL,
    event_type                TEXT,
    event_type_confidence     REAL,
    event_type_probabilities  TEXT,  -- JSON {choice: probability}
    input_tokens              INTEGER NOT NULL DEFAULT 0,
    output_tokens             INTEGER NOT NULL DEFAULT 0,
    scored_at                 TEXT NOT NULL,
    PRIMARY KEY (news_id, config_version)
);
"""


class ItemScores(BaseModel):
    """One news item's Jev scores under one scoring config version."""

    model_config = ConfigDict(frozen=True)

    relevance: float = Field(ge=0, le=1)
    stance: str
    stance_confidence: float = Field(ge=0, le=1)
    stance_probabilities: dict[str, float]
    novelty: float = Field(ge=0, le=1)
    rumour: float = Field(ge=0, le=1)
    event_type: str
    event_type_confidence: float = Field(ge=0, le=1)
    event_type_probabilities: dict[str, float]
    input_tokens: int = 0
    output_tokens: int = 0


class ScoreStore:
    """SQLite store of Jev scores, keyed by news item and config version.

    A row already present for (news_id, config_version) means that item is
    done for that version — scored or permanently skipped — so a rerun of
    `sluice score` only asks Jev about what is still missing. A plain failure
    (network, rate limit, retries exhausted) writes no row, so it is retried
    on the next run rather than remembered as skipped.
    """

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Scoring runs with worker threads sharing one store.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise RuntimeError(
                f"{self.path} has schema version {version}, expected {SCHEMA_VERSION}; "
                "rerun `sluice score` into a fresh file to rebuild it"
            )
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def done(self, news_id: int, config_version: str) -> bool:
        """Whether this item already has a row (scored or skipped) for this version."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM scores WHERE news_id = ? AND config_version = ?",
                (news_id, config_version),
            ).fetchone()
        return row is not None

    def put(
        self, news_id: int, config_version: str, config_digest: str, scores: ItemScores
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO scores ("
                " news_id, config_version, config_digest, skipped,"
                " relevance, stance, stance_confidence, stance_probabilities,"
                " novelty, rumour,"
                " event_type, event_type_confidence, event_type_probabilities,"
                " input_tokens, output_tokens, scored_at"
                ") VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    news_id,
                    config_version,
                    config_digest,
                    scores.relevance,
                    scores.stance,
                    scores.stance_confidence,
                    json.dumps(scores.stance_probabilities),
                    scores.novelty,
                    scores.rumour,
                    scores.event_type,
                    scores.event_type_confidence,
                    json.dumps(scores.event_type_probabilities),
                    scores.input_tokens,
                    scores.output_tokens,
                    datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )

    def skip(self, news_id: int, config_version: str, config_digest: str, reason: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO scores (news_id, config_version, config_digest, skipped, scored_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    news_id,
                    config_version,
                    config_digest,
                    reason,
                    datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )

    def get(self, news_id: int, config_version: str) -> ItemScores | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scores WHERE news_id = ? AND config_version = ? AND skipped IS NULL",
                (news_id, config_version),
            ).fetchone()
        if row is None:
            return None
        return ItemScores(
            relevance=row["relevance"],
            stance=row["stance"],
            stance_confidence=row["stance_confidence"],
            stance_probabilities=json.loads(row["stance_probabilities"]),
            novelty=row["novelty"],
            rumour=row["rumour"],
            event_type=row["event_type"],
            event_type_confidence=row["event_type_confidence"],
            event_type_probabilities=json.loads(row["event_type_probabilities"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT count(*) FROM scores").fetchone()[0]
