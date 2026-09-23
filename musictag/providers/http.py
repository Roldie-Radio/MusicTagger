"""Shared HTTP plumbing: rate limiting, retries and an on-disk response cache."""

from __future__ import annotations

import logging
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import requests

from ..cache import HTTP_CACHE_MAX_AGE_S, http_cache

log = logging.getLogger(__name__)

#: Distinguishes "nothing cached" from "cached, and the answer really is
#: None" (a confirmed-not-found lookup) - both would otherwise look identical
#: to a plain ``is not None`` check, silently defeating negative caching.
_CACHE_MISS = object()


class ProviderError(Exception):
    """A metadata source failed in a way the caller should surface, not swallow."""


#: Worth another try after a pause. 429/503 are the rate limiter talking;
#: 500/502/504 are a busy server or proxy having a bad moment, which
#: MusicBrainz does under load - failing the lookup on the first one turned
#: a blip into a missing match.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Never sleep longer than this on one response, whatever the server asks.
MAX_RETRY_WAIT_S = 30.0


def retry_after_seconds(value: Optional[str], fallback: float) -> float:
    """How long a ``Retry-After`` header asks us to wait.

    The header is either a number of seconds or an HTTP date, and a server
    may send either. Anything unreadable - or a date already in the past -
    falls back to our own backoff rather than raising out of the request.
    """
    if not value:
        return fallback
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return fallback
    if when is None:
        return fallback
    return max(0.0, when.timestamp() - time.time()) or fallback


class RateLimiter:
    """Simple process-wide minimum-interval limiter.

    MusicBrainz asks for no more than one request per second from a given
    client. Exceeding that gets you a 503 and, eventually, a block - so this is
    enforced globally rather than per-thread.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_shared_limiters: dict[str, RateLimiter] = {}
_shared_lock = threading.Lock()


def shared_limiter(service: str, min_interval: float) -> RateLimiter:
    """The one limiter for ``service`` in this process.

    A limiter only limits what shares it, and these used to be created per
    client instance - so two jobs running at once, or a second Matcher built
    to handle one request, each got their own allowance and the process as a
    whole could sail past the one request per second MusicBrainz asks for.
    That is the kind of thing that gets an IP blocked rather than throttled,
    and it gets likelier the more work is done concurrently.

    The interval is updated in place rather than by replacing the limiter, so
    changing it in Settings takes effect without also forgetting when the last
    request went out.
    """
    with _shared_lock:
        limiter = _shared_limiters.get(service)
        if limiter is None:
            limiter = RateLimiter(min_interval)
            _shared_limiters[service] = limiter
        else:
            limiter.min_interval = min_interval
        return limiter


class HttpClient:
    """A ``requests`` session with caching, rate limiting and bounded retries."""

    def __init__(self, user_agent: str, rate_limiter: Optional[RateLimiter] = None,
                 cache_ttl: float = HTTP_CACHE_MAX_AGE_S):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.limiter = rate_limiter
        self.cache_ttl = cache_ttl
        self.cache = http_cache()

    def get_json(self, url: str, params: dict[str, Any] | None = None, *,
                 cache_key: Optional[str] = None, retries: int = 3,
                 timeout: float = 20.0, not_found_is_empty: bool = True) -> Any:
        """GET JSON, with caching, rate limiting and retries.

        ``not_found_is_empty`` controls what a 404 means, because it means two
        different things depending on the endpoint:

        - For a **direct lookup by ID** (``/recording/{mbid}``), a 404 is
          authoritative: that id genuinely does not exist, and is not going to
          start existing on a retry. The default (``True``) treats it that way
          and caches the negative result for the full TTL.
        - For a **search query** (``/recording?query=...``), MusicBrainz
          returns 200 with an empty result list when nothing matches - it
          should never 404 for an ordinary query. A 404 there is anomalous,
          almost certainly a transient service hiccup, not a real answer.
          Treating it as authoritative was a real bug: one flaky response
          would get cached as "this song does not exist" for 30 days, tanking
          that track's confidence with no way to recover short of clearing the
          cache. Callers doing a search pass ``False`` so a 404 is retried
          exactly like a 429/503, and only becomes a raised error (not a
          cached negative) if every retry fails.
        """
        key = cache_key or f"GET {url} {sorted((params or {}).items())}"
        cached = self.cache.get(key, max_age=self.cache_ttl, default=_CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached

        last_error: Optional[Exception] = None
        for attempt in range(retries):
            if self.limiter:
                self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 8))
                continue

            if resp.status_code == 404:
                if not_found_is_empty:
                    self.cache.set(key, None)
                    return None
                last_error = ProviderError(f"404 (unexpected for a search) from {url}")
                time.sleep(min(2 ** attempt, 8))
                continue
            if resp.status_code in RETRYABLE_STATUS:
                # Backing off is mandatory here, not optional.
                wait = retry_after_seconds(resp.headers.get("Retry-After"), 2 ** attempt)
                log.info("%s from %s, waiting %.1fs", resp.status_code, url, wait)
                last_error = ProviderError(f"{resp.status_code} from {url}")
                if attempt + 1 < retries:
                    time.sleep(min(wait, MAX_RETRY_WAIT_S))
                continue
            if not resp.ok:
                raise ProviderError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")

            try:
                data = resp.json()
            except ValueError as exc:
                raise ProviderError(f"Non-JSON response from {url}") from exc
            self.cache.set(key, data)
            return data

        raise ProviderError(f"Request to {url} failed after {retries} attempts: {last_error}")

    def get_bytes(self, url: str, *, timeout: float = 30.0,
                  max_bytes: int = 12 * 1024 * 1024) -> Optional[bytes]:
        """Fetch binary content (cover art). Not cached in sqlite - too large."""
        if self.limiter:
            self.limiter.wait()
        try:
            resp = self.session.get(url, timeout=timeout, stream=True)
        except requests.RequestException as exc:
            log.debug("Binary fetch failed for %s: %s", url, exc)
            return None
        if resp.status_code == 404:
            return None
        if not resp.ok:
            log.debug("Binary fetch %s returned %s", url, resp.status_code)
            return None
        chunks, size = [], 0
        for chunk in resp.iter_content(64 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                log.debug("Cover art at %s exceeded %d bytes, discarding", url, max_bytes)
                return None
        return b"".join(chunks)
