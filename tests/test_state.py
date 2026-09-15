"""Library state and its persistence.

Results survive a restart, so the round trip has to be exercised for real -
an in-memory check would have passed while the stored copy was being wiped.
"""

from __future__ import annotations

import pytest

from musictag.cache import SqliteKV
from musictag.models import (
    MatchResult, QualityIssue, QualityReport, Track, TrackTags,
)
from musictag.state import AppState, effective_tag


@pytest.fixture
def store(tmp_path, monkeypatch):
    kv = SqliteKV(tmp_path / "state.db", "tracks")
    monkeypatch.setattr("musictag.state.state_store", lambda: kv)
    monkeypatch.setattr("musictag.cache.state_store", lambda: kv)
    return kv


@pytest.fixture
def state(store):
    s = AppState()
    s._loaded = True
    return s


def make_track(path="C:/M/a.mp3", mtime=100.0, *, title="Old") -> Track:
    track = Track(path=path, filename=path.split("/")[-1], mtime=mtime)
    track.current = TrackTags(title=title)
    return track


def analysed(score=70) -> QualityReport:
    return QualityReport(score=score, analysed=True,
                         issues=[QualityIssue("x", "high", "Bad", "Detail")])


def reload_from(store) -> AppState:
    fresh = AppState()
    fresh.load()
    return fresh


class TestPersistence:
    def test_a_track_round_trips(self, state, store):
        state.add([make_track()])
        state.persist()
        assert len(reload_from(store).all()) == 1

    def test_quality_survives_a_restart(self, state, store):
        track = make_track()
        state.add([track])
        track.quality = analysed()
        state.persist([track])

        restored = reload_from(store).all()[0]
        assert restored.quality is not None
        assert restored.quality.analysed is True
        assert restored.quality.score == 70
        assert restored.quality.issues[0].title == "Bad"

    def test_match_survives_a_restart(self, state, store):
        track = make_track()
        state.add([track])
        track.match = MatchResult(confidence=88.0,
                                  proposed=TrackTags(title="New", artist="A"))
        state.persist([track])

        restored = reload_from(store).all()[0]
        assert restored.match.confidence == 88.0
        assert restored.match.proposed.title == "New"

    def test_rescanning_an_unchanged_file_keeps_its_results(self, state, store):
        """The regression: a re-scan used to wipe stored quality and matches.

        `add` keeps the existing object, but `persist` was handed the freshly
        scanned one - so memory stayed correct and the database quietly lost
        everything until the next restart.
        """
        track = make_track(mtime=100.0)
        state.add([track])
        track.quality = analysed()
        track.match = MatchResult(confidence=90.0, proposed=TrackTags(title="New"))
        state.persist([track])

        # A re-scan produces new objects with no results attached.
        rescanned = make_track(mtime=100.0, title="Old")
        state.add([rescanned])
        state.persist([rescanned])

        assert state.all()[0].quality is not None, "in-memory results must survive"
        restored = reload_from(store).all()[0]
        assert restored.quality is not None, "stored results must survive too"
        assert restored.match.confidence == 90.0

    def test_a_changed_file_drops_stale_results(self, state, store):
        """Different bytes means the old analysis is describing something else."""
        track = make_track(mtime=100.0)
        state.add([track])
        track.quality = analysed()
        state.persist([track])

        state.add([make_track(mtime=999.0)])
        state.persist()
        assert reload_from(store).all()[0].quality is None

    def test_persisting_an_unknown_track_still_writes_it(self, state, store):
        state.persist([make_track("C:/M/never-added.mp3")])
        assert len(reload_from(store).all()) == 1


class TestEffectiveTag:
    def test_prefers_the_proposal(self):
        track = make_track(title="Old")
        track.match = MatchResult(proposed=TrackTags(title="New"))
        assert effective_tag(track, "title") == "New"

    def test_falls_back_to_the_current_value(self):
        track = make_track(title="Old")
        track.match = MatchResult(proposed=TrackTags(title=None, artist="A"))
        assert effective_tag(track, "title") == "Old"

    def test_no_match_uses_current(self):
        assert effective_tag(make_track(title="Old"), "title") == "Old"

    def test_missing_everywhere_is_none(self):
        assert effective_tag(make_track(), "genre") is None


class TestSorting:
    def _populate(self, state):
        for path, title, track_no in [("C:/M/c.mp3", "Cherry", 3),
                                      ("C:/M/a.mp3", "Apple", 1),
                                      ("C:/M/b.mp3", "Banana", 2)]:
            t = make_track(path, title=title)
            t.match = MatchResult(proposed=TrackTags(title=title, track_no=track_no))
            state.add([t])

    def test_sorts_by_a_tag_field(self, state):
        self._populate(state)
        titles = [t["match"]["proposed"]["title"]
                  for t in state.filtered(sort="title")["tracks"]]
        assert titles == ["Apple", "Banana", "Cherry"]

    def test_descending(self, state):
        self._populate(state)
        titles = [t["match"]["proposed"]["title"]
                  for t in state.filtered(sort="title", desc=True)["tracks"]]
        assert titles == ["Cherry", "Banana", "Apple"]

    def test_numeric_fields_sort_numerically(self, state):
        for path, n in [("C:/M/x.mp3", 10), ("C:/M/y.mp3", 9), ("C:/M/z.mp3", 1)]:
            t = make_track(path)
            t.match = MatchResult(proposed=TrackTags(title=f"T{n}", track_no=n))
            state.add([t])
        numbers = [t["match"]["proposed"]["track_no"]
                   for t in state.filtered(sort="track_no")["tracks"]]
        assert numbers == [1, 9, 10]

    def test_blank_cells_sort_last_in_both_directions(self, state):
        filled = make_track("C:/M/filled.mp3")
        filled.current = TrackTags(title="T", genre="Rock")
        blank = make_track("C:/M/blank.mp3")
        blank.current = TrackTags(title="T")
        state.add([filled, blank])

        for desc in (False, True):
            paths = [t["path"] for t in state.filtered(sort="genre", desc=desc)["tracks"]]
            assert paths[-1] == "C:/M/blank.mp3"

    def test_unknown_sort_key_does_not_raise(self, state):
        self._populate(state)
        assert state.filtered(sort="nonsense")["total"] == 3
