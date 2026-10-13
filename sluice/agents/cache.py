import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

DEFAULT_PATH = Path("data/local/llm_cache.db")

# Unlike the news database this cannot be rebuilt: every row was paid for.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key        TEXT PRIMARY KEY,
    request    TEXT NOT NULL,  -- what was asked, kept for inspection; the key is its hash
    response   TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class CallCache:
    """Answers of LLM calls by request hash. An answer, once stored, is never replaced."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Analysts run side by side on worker threads and share this connection.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM calls WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, request: str, response: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO calls (key, request, response, created_at)"
                " VALUES (?, ?, ?, ?)",
                (key, request, response, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
            )

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT count(*) FROM calls").fetchone()[0]
