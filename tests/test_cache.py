"""The generic sqlite key/value store, and specifically the miss-vs-None
distinction the HTTP layer relies on to make negative caching actually work.
"""

from __future__ import annotations

from musictag.cache import SqliteKV


def test_round_trip(tmp_path):
    kv = SqliteKV(tmp_path / "c.db")
    kv.set("k", {"a": 1})
    assert kv.get("k") == {"a": 1}


def test_missing_key_returns_the_default(tmp_path):
    kv = SqliteKV(tmp_path / "c.db")
    assert kv.get("nope") is None
    sentinel = object()
    assert kv.get("nope", default=sentinel) is sentinel


def test_a_cached_none_is_distinguishable_from_a_miss(tmp_path):
    """The bug this guards: a plain `.get(key) is not None` check cannot tell
    "nothing cached" from "cached, and the value really is None" - both a
    missing key and a stored null come back as None without a sentinel."""
    kv = SqliteKV(tmp_path / "c.db")
    kv.set("negative", None)

    sentinel = object()
    assert kv.get("negative", default=sentinel) is None, \
        "a genuinely cached None must come back as None, not the sentinel"
    assert kv.get("truly-missing", default=sentinel) is sentinel, \
        "a key that was never set must come back as the sentinel, not None"


def test_expired_entries_return_the_default(tmp_path):
    import time
    kv = SqliteKV(tmp_path / "c.db")
    kv.set("k", {"a": 1})
    time.sleep(0.02)
    assert kv.get("k", max_age=0.005) is None


def test_delete_and_clear(tmp_path):
    kv = SqliteKV(tmp_path / "c.db")
    kv.set("a", 1)
    kv.set("b", 2)
    kv.delete("a")
    assert kv.get("a") is None
    assert kv.count() == 1
    kv.clear()
    assert kv.count() == 0


def test_overwrite_replaces_the_value(tmp_path):
    kv = SqliteKV(tmp_path / "c.db")
    kv.set("k", "first")
    kv.set("k", "second")
    assert kv.get("k") == "second"
    assert kv.count() == 1


def test_purge_drops_only_entries_older_than_the_limit(tmp_path, monkeypatch):
    import time as _time
    from musictag.cache import SqliteKV
    kv = SqliteKV(tmp_path / "c.db", "http")
    now = _time.time()
    monkeypatch.setattr("musictag.cache.time.time", lambda: now - 40 * 86400)
    kv.set("old", {"x": 1})
    monkeypatch.setattr("musictag.cache.time.time", lambda: now)
    kv.set("new", {"x": 2})
    assert kv.purge_older_than(30 * 86400) == 1
    assert kv.get("old") is None
    assert kv.get("new") == {"x": 2}
