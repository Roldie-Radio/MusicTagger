"""Checking GitHub Releases for a newer MusicTagger.

Every other network call this app makes happens because the user asked for
something - identify this track, fetch that cover. This one is the exception:
it goes out on the app's own initiative. That is the whole reason it is behind
a setting that turns it off completely, and the reason it never downloads or
installs anything by itself. It reports what exists and links to it; deciding
to install stays with the user.

A failed check is not a failed app. Every error path here ends in an
``UpdateInfo`` describing what went wrong, never an exception reaching the UI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from . import __version__
from .config import Config, get_config
from .providers.http import HttpClient, ProviderError, shared_limiter

log = logging.getLogger(__name__)

GITHUB_OWNER = "Roldie-Radio"
GITHUB_REPO = "MusicTagger"

LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)
RELEASES_PAGE = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"

#: How long an answer is reused before asking GitHub again. Unauthenticated API
#: calls are capped at 60 an hour *per IP address* - shared with anything else
#: on that address - and a published release does not change often enough to
#: justify spending more of that budget than this.
CHECK_TTL_SECONDS = 6 * 60 * 60

#: An update check is the least important request the app makes, so it gives up
#: quickly rather than inheriting the provider default of three tries at a
#: twenty-second timeout - which on an unreachable network would leave the
#: request hanging for the better part of a minute over something nobody asked
#: for. One attempt, six seconds, then say so and move on.
CHECK_RETRIES = 1
CHECK_TIMEOUT = 6.0

#: Release bodies are free text and can run to pages of changelog. The UI shows
#: a short excerpt and links out for the rest, so there is no reason to carry
#: the whole thing through the API response.
NOTES_LIMIT = 2000

_VERSION_RE = re.compile(
    r"^v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?(?:[-+](?P<pre>.+))?$"
)


def parse_version(text: str) -> Optional[tuple[int, int, int, int, str]]:
    """A sortable key for a version string, or ``None`` if it is not one.

    The parts are compared as numbers rather than as text, because "0.10.0" is
    newer than "0.9.0" and a plain string comparison says the opposite - which
    is the kind of bug that only shows up on the tenth release and then tells
    every user they are up to date when they are not.

    A prerelease suffix sorts *below* the same version without one, so
    "1.0.0-beta.1" never looks newer than the "1.0.0" it leads up to.
    """
    if not text:
        return None
    match = _VERSION_RE.match(text.strip())
    if not match:
        return None
    pre = match.group("pre") or ""
    return (
        int(match.group("major")),
        int(match.group("minor") or 0),
        int(match.group("patch") or 0),
        0 if pre else 1,
        pre,
    )


@dataclass
class UpdateInfo:
    """What the UI needs to say something honest about updates.

    The three "we did not get an answer" cases are deliberately distinct.
    ``enabled`` false means the user switched checking off; ``error`` set means
    we tried and could not reach GitHub; both false with ``checked`` true and
    no ``latest`` means GitHub answered and there are simply no releases yet.
    Collapsing those into one "unknown" would make the UI either alarmist or
    silent in cases where it should be the other.
    """

    current: str
    latest: Optional[str] = None
    available: bool = False
    enabled: bool = True
    checked: bool = False
    url: str = RELEASES_PAGE
    name: str = ""
    published_at: str = ""
    notes: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _client(cfg: Config) -> HttpClient:
    return HttpClient(
        user_agent=cfg.user_agent,
        # Shared with nothing else, but still limited: a burst of UI reloads
        # should not turn into a burst of API calls against a 60/hour budget.
        rate_limiter=shared_limiter("github", 1.0),
        cache_ttl=CHECK_TTL_SECONDS,
    )


def check_for_update(cfg: Config | None = None, *, force: bool = False,
                     client: HttpClient | None = None) -> UpdateInfo:
    """Ask GitHub whether there is a release newer than the running version.

    ``client`` is injectable so the tests can exercise every branch without a
    network round trip - this is the one part of the app whose behaviour is
    almost entirely "what did the remote say", and mocking at the transport
    boundary is the only way to cover the cases that matter.
    """
    cfg = cfg or get_config()
    info = UpdateInfo(current=__version__)

    if not cfg.update_check_enabled:
        info.enabled = False
        return info

    http = client or _client(cfg)
    if force:
        # Any cached answer is older than this, so the lookup always misses and
        # goes back to GitHub. The fresh response still gets cached afterwards.
        http.cache_ttl = -1

    try:
        data = http.get_json(LATEST_RELEASE_API, not_found_is_empty=True,
                             retries=CHECK_RETRIES, timeout=CHECK_TIMEOUT)
    except ProviderError as exc:
        # Includes GitHub's 403 for an exhausted rate limit, which is not an
        # error worth alarming anyone about - it means "ask again later".
        log.debug("Update check could not reach GitHub: %s", exc)
        info.error = str(exc)
        return info
    except Exception as exc:                          # noqa: BLE001
        # An update check is never important enough to take the app down with
        # it, whatever unexpected thing the transport does.
        log.debug("Update check failed unexpectedly: %s", exc)
        info.error = str(exc)
        return info

    info.checked = True

    if not data:
        # A 404 from /releases/latest means the repository has no published
        # releases at all. Nothing is wrong; there is just nothing to compare
        # against yet.
        return info

    tag = str(data.get("tag_name") or "")
    latest = parse_version(tag)
    current = parse_version(__version__)

    info.latest = tag.lstrip("v") or None
    info.url = str(data.get("html_url") or RELEASES_PAGE)
    info.name = str(data.get("name") or "")
    info.published_at = str(data.get("published_at") or "")
    info.notes = str(data.get("body") or "")[:NOTES_LIMIT]

    if latest is None or current is None:
        # An unparseable tag is a packaging mistake, not a new version. Saying
        # nothing is better than claiming an update that may not exist.
        log.debug("Could not compare versions: running %r, latest tag %r",
                  __version__, tag)
        info.latest = None
        return info

    info.available = latest > current
    return info
