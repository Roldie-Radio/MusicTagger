"""Two tiny sqlite stores: an HTTP response cache and the persisted UI state.

Caching web lookups is not just politeness towards MusicBrainz (whose rate limit
is one request per second) - it is what makes re-running a scan over a large
library tolerable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .config import CACHE_DB, STATE_DB


class SqliteKV:
    """Thread-safe key/value store with optional TTL."""

    def __init__(self, path: Path, table: str = "kv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.table = table
        self._local = threading.local()
        self._init_lock = threading.Lock()
        with self._init_lock:
            conn = self._conn()
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                " key TEXT PRIMARY KEY,"
                " value TEXT NOT NULL,"
                " created REAL NOT NULL)"
            )
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def get(self, key: str, *, max_age: Optional[float] = None, default: Any = None) -> Any:
        """Look up ``key``, returning ``default`` on a miss, expiry, or bad data.

        ``default`` exists so a caller that legitimately caches ``None`` as a
        value (a confirmed negative lookup, say) can tell "nothing cached"
        apart from "cached, and the answer was None" - passing a private
        sentinel object as ``default`` makes that distinguishable. Every
        existing caller here stores real dicts and never ``None``, so the
        ordinary default of ``None`` is unchanged for them.
        """
        row = self._conn().execute(
            f"SELECT value, created FROM {self.table} WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        value, created = row
        if max_age is not None and (time.time() - created) > max_age:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def set(self, key: str, value: Any) -> None:
        conn = self._conn()
        conn.execute(
            f"INSERT INTO {self.table} (key, value, created) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, created = excluded.created",
            (key, json.dumps(value), time.time()),
        )
        conn.commit()

    def delete(self, key: str) -> None:
        conn = self._conn()
        conn.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
        conn.commit()

    def keys(self) -> list[str]:
        return [r[0] for r in self._conn().execute(f"SELECT key FROM {self.table}")]

    def clear(self) -> None:
        conn = self._conn()
        conn.execute(f"DELETE FROM {self.table}")
        conn.commit()

    def count(self) -> int:
        return self._conn().execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]


_http_cache: SqliteKV | None = None
_state_store: SqliteKV | None = None


def http_cache() -> SqliteKV:
    """Cache for MusicBrainz / AcoustID / Cover Art Archive responses."""
    global _http_cache
    if _http_cache is None:
        _http_cache = SqliteKV(CACHE_DB, "http")
    return _http_cache


def state_store() -> SqliteKV:
    """Persisted scan results, so closing the window does not lose your work."""
    global _state_store
    if _state_store is None:
        _state_store = SqliteKV(STATE_DB, "tracks")
    return _state_store
