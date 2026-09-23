"""The update check.

This module is almost entirely "what did the remote say, and what did we
conclude from it", so the tests drive it through an injected client rather
than the network. What matters is that every answer GitHub can give - a newer
release, the same one, no releases at all, a rate-limit refusal, a tag nobody
can parse - produces a sensible, non-alarming result, and that the check stays
completely silent when the user has switched it off.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musictag import __version__
from musictag.config import Config
from musictag.providers.http import ProviderError
from musictag.update import (
    AUTO_UPDATE_ENV, CHECK_RETRIES, CHECK_TIMEOUT, CHECK_TTL_SECONDS, LATEST_RELEASE_API,
    NOTES_LIMIT, RELEASES_PAGE, _client, check_for_update, parse_version,
)


class FakeClient:
    """Stands in for HttpClient, recording what it was asked for."""

    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls: list[str] = []
        self.cache_ttl = 6 * 60 * 60

    def get_json(self, url, params=None, **kwargs):
        self.calls.append(url)
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.payload


def release(tag="v9.9.9", **extra):
    payload = {
        "tag_name": tag,
        "html_url": f"https://github.com/Roldie-Radio/MusicTagger/releases/tag/{tag}",
        "name": f"MusicTagger {tag}",
        "published_at": "2026-09-15T12:00:00Z",
        "body": "Fixed some things.",
    }
    payload.update(extra)
    return payload


@pytest.fixture
def cfg() -> Config:
    config = Config()
    config.update_check_enabled = True
    return config


class TestParseVersion:
    def test_reads_the_usual_shapes(self):
        assert parse_version("1.2.3")[:3] == (1, 2, 3)
        assert parse_version("v1.2.3")[:3] == (1, 2, 3)
        assert parse_version("0.1.0")[:3] == (0, 1, 0)

    def test_missing_parts_default_to_zero(self):
        assert parse_version("2")[:3] == (2, 0, 0)
        assert parse_version("2.1")[:3] == (2, 1, 0)

    def test_compares_as_numbers_not_text(self):
        """The bug this exists to prevent only appears at the tenth release.

        As strings, "0.10.0" sorts *below* "0.9.0" - so a text comparison
        would tell every user they were up to date from 0.10.0 onwards, and
        nothing before that release would reveal it.
        """
        assert parse_version("0.10.0") > parse_version("0.9.0")
        assert parse_version("1.0.0") > parse_version("0.99.99")
        assert parse_version("2.0.1") > parse_version("2.0.0")

    def test_a_prerelease_precedes_its_own_release(self):
        assert parse_version("1.0.0-beta.1") < parse_version("1.0.0")
        assert parse_version("1.0.0-rc.1") > parse_version("0.9.9")

    def test_junk_is_not_a_version(self):
        for text in ("", "latest", "release-candidate", "v", "1.2.x", "nightly"):
            assert parse_version(text) is None, text


class TestCheckForUpdate:
    def test_reports_a_newer_release(self, cfg):
        client = FakeClient(release("v9.9.9"))
        info = check_for_update(cfg, client=client)
        assert info.available is True
        assert info.latest == "9.9.9"
        assert info.current == __version__
        assert info.checked is True
        assert not info.error
        assert client.calls == [LATEST_RELEASE_API]

    def test_the_running_version_is_not_an_update(self, cfg):
        info = check_for_update(cfg, client=FakeClient(release(f"v{__version__}")))
        assert info.available is False
        assert info.latest == __version__
        assert info.checked is True

    def test_a_dev_build_ahead_of_the_release_is_not_an_update(self, cfg):
        """Running 0.3.0 against a published 0.2.0 must not offer a downgrade."""
        info = check_for_update(cfg, client=FakeClient(release("v0.0.1")))
        assert info.available is False

    def test_no_releases_yet_is_not_an_error(self, cfg):
        """/releases/latest 404s on a repo that has never published one.

        That is the state this feature shipped into, so it has to read as
        "nothing to compare against" rather than as a broken update check.
        """
        info = check_for_update(cfg, client=FakeClient(None))
        assert info.checked is True
        assert info.latest is None
        assert info.available is False
        assert info.error == ""

    def test_an_unreachable_github_reports_an_error_and_no_update(self, cfg):
        info = check_for_update(cfg, client=FakeClient(error=ProviderError("403 rate limited")))
        assert info.available is False
        assert info.checked is False
        assert "403" in info.error

    def test_an_unexpected_transport_failure_is_still_contained(self, cfg):
        """An update check must never be the thing that takes the app down."""
        info = check_for_update(cfg, client=FakeClient(error=RuntimeError("socket exploded")))
        assert info.available is False
        assert "socket exploded" in info.error

    def test_an_unparseable_tag_never_claims_an_update(self, cfg):
        """A mis-tagged release is a packaging mistake, not a new version."""
        info = check_for_update(cfg, client=FakeClient(release("nightly-build")))
        assert info.available is False
        assert info.latest is None
        assert info.checked is True

    def test_disabled_makes_no_request_at_all(self, cfg):
        """Off has to mean no traffic, not a request whose answer is ignored."""
        cfg.update_check_enabled = False
        client = FakeClient(release("v9.9.9"))
        info = check_for_update(cfg, client=client)
        assert info.enabled is False
        assert info.available is False
        assert client.calls == []

    def test_carries_the_release_details_the_ui_shows(self, cfg):
        info = check_for_update(cfg, client=FakeClient(release("v9.9.9")))
        assert info.name == "MusicTagger v9.9.9"
        assert info.published_at == "2026-09-15T12:00:00Z"
        assert info.notes == "Fixed some things."
        assert info.url.endswith("/releases/tag/v9.9.9")

    def test_falls_back_to_the_releases_page_without_a_url(self, cfg):
        info = check_for_update(cfg, client=FakeClient(release("v9.9.9", html_url="")))
        assert info.url == RELEASES_PAGE

    def test_long_release_notes_are_truncated(self, cfg):
        info = check_for_update(cfg, client=FakeClient(release("v9.9.9", body="x" * 50_000)))
        assert len(info.notes) == NOTES_LIMIT

    def test_force_bypasses_the_cache(self, cfg):
        client = FakeClient(release("v9.9.9"))
        check_for_update(cfg, client=client, force=True)
        # Negative means every cached entry reads as expired, so the lookup
        # always goes back to GitHub.
        assert client.cache_ttl < 0

    def test_gives_up_quickly_rather_than_hanging(self, cfg):
        """Nobody asked for this request, so it must not block on a dead network.

        The provider default is three attempts at a twenty-second timeout with
        backoff between them - the better part of a minute of a hung request
        for something the user never initiated.
        """
        client = FakeClient(release("v9.9.9"))
        check_for_update(cfg, client=client)
        assert client.kwargs["retries"] == CHECK_RETRIES == 1
        assert client.kwargs["timeout"] == CHECK_TIMEOUT <= 10

    def test_the_result_serialises_for_the_api(self, cfg):
        info = check_for_update(cfg, client=FakeClient(release("v9.9.9")))
        data = info.to_dict()
        assert json.loads(json.dumps(data))["available"] is True
        assert set(data) >= {"current", "latest", "available", "enabled",
                             "checked", "url", "error"}


class TestRealClient:
    """The client every test above replaces with a fake.

    Injecting a fake into each behavioural test is what makes them fast and
    offline, but it also means the real constructor runs in none of them - and
    it shipped broken exactly once, calling ``cfg.user_agent()`` on what is a
    property, so every check raised TypeError while the whole suite stayed
    green. Building it for real here costs nothing and closes that gap.
    """

    def test_builds_without_touching_the_network(self, cfg):
        cfg.musicbrainz_contact = "someone@example.com"
        client = _client(cfg)
        assert client.session.headers["User-Agent"] == cfg.user_agent
        assert "MusicTagger" in client.session.headers["User-Agent"]

    def test_caches_for_the_documented_window(self, cfg):
        """The 6h TTL is what keeps this inside GitHub's 60-an-hour budget."""
        assert _client(cfg).cache_ttl == CHECK_TTL_SECONDS
        assert CHECK_TTL_SECONDS >= 60 * 60

    def test_is_rate_limited(self, cfg):
        assert _client(cfg).limiter is not None


class TestVersionSources:
    def test_python_and_electron_versions_agree(self):
        """Two files carry this app's version, and an updater compares both.

        musictag/__init__.py is what the running app reports through
        /api/status and what the update check compares against; the Electron
        package.json is what electron-builder stamps onto the installer and
        the release tag. If they drift, the app tells the user it is a version
        it is not, and the update banner is wrong in whichever direction the
        drift went. Nothing else enforces that they match.
        """
        package = json.loads(
            (Path(__file__).resolve().parents[1] / "desktop" / "electron" / "package.json")
            .read_text(encoding="utf-8")
        )
        assert package["version"] == __version__, (
            f"desktop/electron/package.json says {package['version']}, "
            f"musictag/__init__.py says {__version__} - these must match"
        )


class TestAutoInstallFlag:
    """The desktop shell installs updates itself and says so via the env."""

    def test_off_unless_the_shell_says_so(self, cfg, monkeypatch):
        monkeypatch.delenv(AUTO_UPDATE_ENV, raising=False)
        info = check_for_update(cfg, client=FakeClient(payload=release()))
        assert info.auto_install is False

    def test_on_when_the_shell_installs_updates(self, cfg, monkeypatch):
        monkeypatch.setenv(AUTO_UPDATE_ENV, "1")
        info = check_for_update(cfg, client=FakeClient(payload=release()))
        assert info.auto_install is True
        assert info.to_dict()["auto_install"] is True

    def test_reported_even_when_checking_is_off(self, cfg, monkeypatch):
        """The UI still needs to describe the right behaviour in Settings."""
        monkeypatch.setenv(AUTO_UPDATE_ENV, "1")
        cfg.update_check_enabled = False
        info = check_for_update(cfg, client=FakeClient(payload=release()))
        assert info.enabled is False
        assert info.auto_install is True

    def test_only_an_exact_one_counts(self, cfg, monkeypatch):
        monkeypatch.setenv(AUTO_UPDATE_ENV, "0")
        info = check_for_update(cfg, client=FakeClient(payload=release()))
        assert info.auto_install is False
