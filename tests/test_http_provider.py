"""The HTTP layer shared by every metadata provider: caching, retries, and -
the specific bug this guards against - what a 404 means.

A 404 means two different things depending on the endpoint. A direct lookup
by id (``/recording/{mbid}``) 404ing is authoritative: that id does not
exist, full stop. A *search* query 404ing is not - MusicBrainz answers "no
match" with 200 and an empty list, so a 404 there is an anomaly, almost
certainly a transient service hiccup. Treating both the same way used to mean
one flaky response got permanently cached as "this song does not exist" for
30 days, which is exactly the failure mode a user reported: a track's
confidence tanked because a single 404 was taken as gospel.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from musictag.cache import SqliteKV
from musictag.providers.http import HttpClient, ProviderError


class FakeResponse:
    def __init__(self, status_code: int, json_data=None, headers=None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


@pytest.fixture
def client(tmp_path, monkeypatch) -> HttpClient:
    kv = SqliteKV(tmp_path / "cache.db", "http")
    monkeypatch.setattr("musictag.providers.http.http_cache", lambda: kv)
    c = HttpClient(user_agent="test/1.0")
    c.cache = kv
    return c


def fast_sleep(monkeypatch):
    """Retries in these tests use real backoff delays; skip the wait."""
    monkeypatch.setattr("musictag.providers.http.time.sleep", lambda _: None)


class TestDirectLookup404(object):
    """not_found_is_empty=True (the default) - unchanged, correct behaviour."""

    def test_a_404_is_returned_as_none_on_the_first_try(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        get = MagicMock(return_value=FakeResponse(404))
        client.session.get = get
        result = client.get_json("http://x/recording/nonexistent")
        assert result is None
        assert get.call_count == 1, "a direct-lookup 404 must not be retried"

    def test_the_negative_result_is_cached(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        client.session.get = MagicMock(return_value=FakeResponse(404))
        client.get_json("http://x/recording/nonexistent", cache_key="k1")
        # A second call must not hit the network at all - it should be served
        # straight from the cache.
        client.session.get = MagicMock(side_effect=AssertionError("should not be called"))
        assert client.get_json("http://x/recording/nonexistent", cache_key="k1") is None


class TestSearch404(object):
    """not_found_is_empty=False - the fix."""

    def test_a_404_is_retried_not_trusted(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        get = MagicMock(side_effect=[FakeResponse(404), FakeResponse(404), FakeResponse(200, {"ok": 1})])
        client.session.get = get
        result = client.get_json("http://x/recording", not_found_is_empty=False)
        assert result == {"ok": 1}
        assert get.call_count == 3, "must have retried past the two 404s"

    def test_repeated_404_raises_rather_than_returning_a_false_empty(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        client.session.get = MagicMock(return_value=FakeResponse(404))
        with pytest.raises(ProviderError):
            client.get_json("http://x/recording", not_found_is_empty=False, retries=3)

    def test_a_search_404_is_never_cached(self, client, monkeypatch):
        """The actual regression: a transient 404 must not poison the cache
        for 30 days. The next call has to hit the network again, not be
        silently served a stale 'nothing found'."""
        fast_sleep(monkeypatch)
        client.session.get = MagicMock(return_value=FakeResponse(404))
        with pytest.raises(ProviderError):
            client.get_json("http://x/recording", cache_key="k2", not_found_is_empty=False, retries=2)

        # Network recovers. A fresh call must actually retry, not read a
        # cached negative result.
        client.session.get = MagicMock(return_value=FakeResponse(200, {"recordings": ["found"]}))
        result = client.get_json("http://x/recording", cache_key="k2", not_found_is_empty=False)
        assert result == {"recordings": ["found"]}

    def test_a_successful_search_result_is_still_cached_normally(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        client.session.get = MagicMock(return_value=FakeResponse(200, {"recordings": []}))
        client.get_json("http://x/recording", cache_key="k3", not_found_is_empty=False)
        client.session.get = MagicMock(side_effect=AssertionError("should not be called"))
        result = client.get_json("http://x/recording", cache_key="k3", not_found_is_empty=False)
        assert result == {"recordings": []}


class TestOtherFailureModes(object):
    """These paths were already correct; confirm the new parameter didn't
    disturb them."""

    def test_a_timeout_is_retried_regardless_of_the_new_flag(self, client, monkeypatch):
        import requests
        fast_sleep(monkeypatch)
        get = MagicMock(side_effect=[requests.Timeout("slow"), FakeResponse(200, {"ok": 1})])
        client.session.get = get
        assert client.get_json("http://x/recording", not_found_is_empty=False) == {"ok": 1}
        assert get.call_count == 2

    def test_rate_limiting_is_still_retried_not_raised_immediately(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        get = MagicMock(side_effect=[FakeResponse(503), FakeResponse(200, {"ok": 1})])
        client.session.get = get
        assert client.get_json("http://x/recording") == {"ok": 1}

    def test_a_genuine_server_error_still_raises_immediately(self, client, monkeypatch):
        fast_sleep(monkeypatch)
        client.session.get = MagicMock(return_value=FakeResponse(500, text="boom"))
        with pytest.raises(ProviderError):
            client.get_json("http://x/recording")


class TestSharedRateLimiter:
    """MusicBrainz blocks clients that exceed one request per second. A limiter
    only limits what shares it, so every caller in the process has to get the
    same one - which stopped being true the moment two of anything ran at once.
    """

    def test_every_caller_gets_the_same_limiter(self):
        from musictag.providers.http import shared_limiter
        assert shared_limiter("musicbrainz", 1.0) is shared_limiter("musicbrainz", 1.0)

    def test_different_services_are_limited_separately(self):
        from musictag.providers.http import shared_limiter
        assert shared_limiter("musicbrainz", 1.0) is not shared_limiter("acoustid", 0.34)

    def test_two_clients_built_separately_still_share_one_allowance(self):
        from musictag.config import Config
        from musictag.providers.musicbrainz import MusicBrainzClient

        cfg = Config()
        first, second = MusicBrainzClient(cfg), MusicBrainzClient(cfg)
        assert first.http.limiter is second.http.limiter, \
            "a second client must not come with a second request allowance"

    def test_changing_the_interval_does_not_forget_the_last_request(self):
        from musictag.providers.http import shared_limiter

        limiter = shared_limiter("test-service", 1.0)
        limiter.wait()
        marked = limiter._last
        again = shared_limiter("test-service", 2.0)
        assert again is limiter
        assert again.min_interval == 2.0
        assert again._last == marked, \
            "rebuilding on a settings change would hand out a free request"
