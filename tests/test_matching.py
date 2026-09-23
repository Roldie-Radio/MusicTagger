"""Confidence scoring.

The number this module produces is the whole point of requirement 3, so these
tests pin down the properties that make it trustworthy: it rises with evidence,
falls with disagreement, falls when a runner-up is nearly as good, and is capped
when there was nothing to check the answer against.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from musictag.config import Config
from musictag.matching import (
    WEIGHTS, Matcher, _dedupe_candidates, _duration_gap, _evidence_ceiling,
    _length_s,
)
from musictag.models import AudioProps, Candidate, Signal, Track, TrackTags


# Path parsing has to run against separators the host platform recognises.
# A literal r"C:\Music\Portishead\Dummy\10 - Glory Box.flac" is a single
# filename with no parent directories on POSIX, so the album and artist folders
# the filename fallback reads off the path simply are not there, and it returns
# None for both without failing in any visible way.
MUSIC_ROOT = Path("C:/Music") if os.name == "nt" else Path("/Music")


def music_path(*parts: str) -> str:
    """An absolute path under a notional library root, native to this platform."""
    return str(MUSIC_ROOT.joinpath(*parts))


@pytest.fixture
def cfg() -> Config:
    config = Config()
    config.acoustid_api_key = ""
    config.max_candidates = 5
    return config


@pytest.fixture
def matcher(cfg) -> Matcher:
    return Matcher(cfg)


def make_track(**kwargs) -> Track:
    tags = TrackTags(
        title=kwargs.pop("title", "Glory Box"),
        artist=kwargs.pop("artist", "Portishead"),
        album=kwargs.pop("album", "Dummy"),
        track_no=kwargs.pop("track_no", None),
    )
    track = Track(path=kwargs.pop("path", r"C:\Music\Portishead\Dummy\10 - Glory Box.flac"))
    track.filename = "10 - Glory Box.flac"
    track.current = tags
    track.props = AudioProps(duration_s=kwargs.pop("duration_s", 301.0))
    return track


def _with_id(cand: Candidate, raw_id: str) -> Candidate:
    """A candidate MusicBrainz considers a separate recording."""
    cand.raw_id = raw_id
    return cand


def make_candidate(source="musicbrainz-search", *, title="Glory Box", artist="Portishead",
                   album="Dummy", length_s=301.0, fingerprint=None, track_no=None) -> Candidate:
    cand = Candidate(
        source=source,
        tags=TrackTags(title=title, artist=artist, album=album,
                       album_artist=artist, track_no=track_no),
        length_s=length_s,
        raw_id=f"{title}-{album}",
    )
    if fingerprint is not None:
        cand.signals.append(Signal("fingerprint", f"AcoustID {fingerprint}", fingerprint, 3.0))
    return cand


# ===========================================================================

class TestEvidenceCeiling:
    def test_strong_fingerprint_allows_near_certainty(self):
        cand = make_candidate("acoustid", fingerprint=0.95)
        observed = {"evidence": 0, "from_tags": False}
        assert _evidence_ceiling(observed, cand) == 99.0

    def test_weak_fingerprint_lowers_the_ceiling(self):
        cand = make_candidate("acoustid", fingerprint=0.7)
        assert _evidence_ceiling({"evidence": 0, "from_tags": False}, cand) == 95.0

    def test_filename_only_is_capped_hard(self):
        cand = make_candidate("filename")
        ceiling = _evidence_ceiling({"evidence": 3, "from_tags": False}, cand)
        assert ceiling <= 50, "a guess from a filename must never look confident"

    def test_no_tags_caps_below_auto_apply(self):
        cand = make_candidate()
        ceiling = _evidence_ceiling({"evidence": 2, "from_tags": False}, cand)
        assert ceiling <= 60

    def test_rich_tags_allow_a_high_ceiling(self):
        cand = make_candidate()
        assert _evidence_ceiling({"evidence": 7, "from_tags": True}, cand) == 93.0

    def test_ceiling_rises_with_evidence(self):
        cand = make_candidate()
        thin = _evidence_ceiling({"evidence": 3, "from_tags": True}, cand)
        middling = _evidence_ceiling({"evidence": 5, "from_tags": True}, cand)
        rich = _evidence_ceiling({"evidence": 7, "from_tags": True}, cand)
        assert thin < middling < rich


class TestScoring:
    def test_a_perfect_match_scores_high(self, matcher):
        track = make_track()
        observed = matcher._observations(track)
        score = matcher._score(make_candidate(), observed)
        assert score > 90

    def test_duration_mismatch_is_penalised(self, matcher):
        track = make_track(duration_s=301.0)
        observed = matcher._observations(track)
        close = matcher._score(make_candidate(length_s=302.0), observed)
        far = matcher._score(make_candidate(length_s=312.0), observed)
        assert close > far

    def test_wildly_wrong_duration_is_vetoed(self, matcher):
        track = make_track(duration_s=301.0)
        observed = matcher._observations(track)
        score = matcher._score(make_candidate(length_s=90.0), observed)
        assert score < 45, "a 3.5 minute difference means it is a different recording"

    def test_duration_signals_weight_shrinks_as_the_gap_widens(self, matcher):
        """Not just the score - the *influence* of the duration signal should
        taper off as the difference grows, so a few seconds' drift (a plausible
        remaster/fade edit) does not pull as hard on the average as an exact
        match, letting title/artist/album decide more in that ambiguous zone."""
        track = make_track(duration_s=301.0)
        observed = matcher._observations(track)

        def duration_weight(length_s):
            cand = make_candidate(length_s=length_s)
            matcher._score(cand, observed)
            return next(s.weight for s in cand.signals if s.name == "duration")

        exact = duration_weight(303.0)      # within DURATION_PERFECT_S
        close = duration_weight(346.0)      # mid-ramp
        far = duration_weight(386.0)        # near DURATION_ZERO_S, still under the veto

        assert exact > close > far
        assert far > 0, "a moderate gap should still count for something, not nothing"

    def test_an_identical_length_outweighs_a_merely_close_one(self, matcher):
        """Two unrelated recordings landing on the same length to the second is
        a coincidence; the same recording doing it is the norm. So agreeing
        exactly has to count for more than agreeing within a few seconds -
        which used to be flattened into one full-weight band."""
        track = make_track(duration_s=301.0)
        observed = matcher._observations(track)

        def duration_weight(length_s):
            cand = make_candidate(length_s=length_s)
            matcher._score(cand, observed)
            return next(s.weight for s in cand.signals if s.name == "duration")

        identical = duration_weight(301.0)        # 0.0s apart
        noise = duration_weight(301.05)           # below DURATION_IDENTICAL_S
        sub_second = duration_weight(301.6)       # close, but not the same length
        close = duration_weight(305.0)            # a few seconds off
        edge = duration_weight(311.0)             # at DURATION_PERFECT_S

        assert identical == noise, \
            "a hundredth of a second is the last decimal place, not a difference"
        assert identical > sub_second, \
            "agreeing exactly beats agreeing within a second - it is the better bet"
        assert sub_second > close > edge, "the boost has to taper, not cliff"
        assert identical > WEIGHTS["duration"], "an exact match beats ordinary full weight"
        assert edge == pytest.approx(WEIGHTS["duration"], abs=0.05), \
            "and it decays back to ordinary full weight, not below it"

    def test_an_identical_length_says_so(self, matcher):
        track = make_track(duration_s=301.0)
        cand = make_candidate(length_s=301.0)
        matcher._score(cand, matcher._observations(track))
        detail = next(s.detail for s in cand.signals if s.name == "duration")
        assert "matches" in detail.lower()

    def test_a_wrong_title_is_not_rescued_by_the_rest_of_the_album_matching(self, matcher):
        """Same album, same length, wrong title = a different track on that album."""
        track = make_track()
        observed = matcher._observations(track)
        score = matcher._score(make_candidate(title="Enter Sandman"), observed)
        assert score < 50

    def test_a_strong_fingerprint_overrides_a_title_disagreement(self, matcher):
        """The audio is the evidence; a wrong title tag is what we are fixing."""
        track = make_track()
        observed = matcher._observations(track)
        penalised = matcher._score(make_candidate(title="Actually Roads"), observed)
        trusted = matcher._score(
            make_candidate("acoustid", title="Actually Roads", fingerprint=0.97), observed)
        assert trusted > penalised

    def test_fingerprint_raises_the_confidence_of_a_thinly_tagged_file(self, matcher):
        """On a file with almost no tags, the fingerprint is what lifts the ceiling."""
        track = Track(path=r"C:\Music\track07.mp3")
        track.current = TrackTags()
        track.props = AudioProps(duration_s=301.0)
        observed = matcher._observations(track)

        without = matcher._finish(track, observed, [make_candidate()], [], [])
        with_fp = matcher._finish(
            track, observed, [make_candidate("acoustid", fingerprint=0.96)], [], [])
        assert with_fp.confidence > without.confidence

    def test_signals_are_recorded_for_display(self, matcher):
        track = make_track()
        observed = matcher._observations(track)
        cand = make_candidate()
        matcher._score(cand, observed)
        names = {s.name for s in cand.signals}
        assert {"title", "artist", "album", "duration"} <= names
        for signal in cand.signals:
            assert 0.0 <= signal.score <= 1.0
            assert signal.detail, "every signal must explain itself"

    def test_scoring_is_bounded(self, matcher):
        track = make_track()
        observed = matcher._observations(track)
        for cand in (make_candidate(), make_candidate(title="x", artist="y", album="z"),
                     make_candidate("acoustid", fingerprint=1.0)):
            score = matcher._score(cand, observed)
            assert 0.0 <= score <= 100.0


class TestObservations:
    def test_counts_evidence_from_tags(self, matcher):
        rich = matcher._observations(make_track())
        assert rich["from_tags"] is True
        assert rich["evidence"] >= 6

    def test_untagged_file_has_little_evidence(self, matcher):
        track = Track(path=r"C:\Music\track07.mp3")
        track.current = TrackTags()
        track.props = AudioProps(duration_s=200.0)
        observed = matcher._observations(track)
        assert observed["from_tags"] is False
        assert observed["evidence"] < 5

    def test_falls_back_to_the_filename(self, matcher):
        track = Track(path=music_path("Portishead", "Dummy", "10 - Glory Box.flac"))
        track.current = TrackTags()
        track.props = AudioProps(duration_s=301.0)
        observed = matcher._observations(track)
        assert observed["title"] == "Glory Box"
        assert observed["album"] == "Dummy"
        assert observed["track_no"] == 10


class TestFinish:
    # A file with only a title - no artist or album to compare against - is
    # where near-ties actually happen: with nothing to tell a cover from the
    # original, both score the same. It also keeps raw scores below the
    # evidence ceiling, so the penalty is visible instead of being capped away.
    def _titleless_setup(self, matcher):
        track = Track(path=r"C:\Music\Glory Box.mp3")
        track.filename = "Glory Box.mp3"
        track.current = TrackTags(title="Glory Box")
        track.props = AudioProps(duration_s=301.0)
        return track, matcher._observations(track)

    def _pair(self, matcher, track, observed, **runner_up_tags) -> float:
        """Confidence when a close runner-up with these tags is in play."""
        best = make_candidate(length_s=355.0)
        other = _with_id(make_candidate(length_s=355.0, **runner_up_tags), "other")
        return matcher._finish(track, observed, [best, other], [], []).confidence

    def test_ambiguity_reduces_confidence(self, matcher):
        """A close runner-up that is a genuinely different answer - a cover by
        another artist - is the case this penalty exists for."""
        track, observed = self._titleless_setup(matcher)
        alone = matcher._finish(
            track, observed, [make_candidate(length_s=355.0)], [], []).confidence
        together = self._pair(matcher, track, observed,
                              artist="Vitamin String Quartet",
                              album="VSQ Performs Portishead")
        assert together < alone
        assert any("ambiguity" in n.lower() or "second-best" in n.lower()
                   for n in matcher._finish(
                       track, observed,
                       [make_candidate(length_s=355.0),
                        _with_id(make_candidate(length_s=355.0,
                                                artist="Vitamin String Quartet"), "o")],
                       [], []).notes)

    def test_same_song_on_a_different_release_is_only_mildly_ambiguous(self, matcher):
        """Both candidates name the same artist and song and differ only on
        which release - the song is settled, so this must cost far less than a
        rival that disagrees about who is playing."""
        track, observed = self._titleless_setup(matcher)
        same_song = self._pair(matcher, track, observed,
                               album="Dummy: The Anniversary Edition")
        different_artist = self._pair(matcher, track, observed,
                                      artist="Vitamin String Quartet",
                                      album="VSQ Performs Portishead")
        assert same_song > different_artist

        notes = matcher._finish(
            track, observed,
            [make_candidate(length_s=355.0),
             _with_id(make_candidate(length_s=355.0,
                                     album="Dummy: The Anniversary Edition"), "o")],
            [], []).notes
        assert any("release" in n.lower() for n in notes)

    def test_identical_tags_on_two_recordings_is_not_ambiguity(self, matcher):
        """MusicBrainz often holds one track as several recordings. If applying
        either writes exactly the same tags, there is nothing to be unsure of."""
        track, observed = self._titleless_setup(matcher)
        alone = matcher._finish(
            track, observed, [make_candidate(length_s=355.0)], [], []).confidence
        together = self._pair(matcher, track, observed)      # identical tags
        assert together == alone

    def test_no_candidates_gives_zero_and_says_why(self, matcher):
        track = make_track()
        result = matcher._finish(track, matcher._observations(track), [], [], [])
        assert result.confidence == 0.0
        assert result.notes

    def test_field_confidence_is_populated(self, matcher):
        track = make_track()
        result = matcher._finish(track, matcher._observations(track),
                                 [make_candidate()], [], ["test"])
        assert result.field_confidence["title"] > 0
        assert all(0 <= v <= 100 for v in result.field_confidence.values())

    def test_genre_is_less_trusted_than_title(self, matcher):
        track = make_track()
        cand = make_candidate()
        cand.tags.genre = "Trip Hop"
        result = matcher._finish(track, matcher._observations(track), [cand], [], [])
        assert result.field_confidence["genre"] < result.field_confidence["title"]

    def test_album_artist_is_always_filled(self, matcher):
        """Plex splits albums without it, so it must never come back empty."""
        track = make_track()
        cand = make_candidate()
        cand.tags.album_artist = None
        result = matcher._finish(track, matcher._observations(track), [cand], [], [])
        assert result.proposed.album_artist

    def test_existing_genre_is_kept_when_the_database_has_none(self, matcher):
        track = make_track()
        track.current.genre = "Trip Hop"
        result = matcher._finish(track, matcher._observations(track),
                                 [make_candidate()], [], [])
        assert result.proposed.genre == "Trip Hop"

    def test_preserve_existing_only_fills_blanks(self, cfg):
        cfg.preserve_existing_tags = True
        matcher = Matcher(cfg)
        track = make_track(title="My Own Title")
        cand = make_candidate(title="Glory Box")
        result = matcher._finish(track, matcher._observations(track), [cand], [], [])
        assert result.proposed.title == "My Own Title"


class TestDedupe:
    def test_same_recording_is_collapsed(self):
        a = make_candidate("acoustid", fingerprint=0.9)
        b = make_candidate("musicbrainz-search")
        merged = _dedupe_candidates([a, b])
        assert len(merged) == 1
        assert "acoustid" in merged[0].source and "musicbrainz-search" in merged[0].source

    def test_corroborating_signals_are_merged(self):
        a = make_candidate("acoustid", fingerprint=0.9)
        b = make_candidate("musicbrainz-search")
        b.signals.append(Signal("search_rank", "rank 1", 0.9, 0.5))
        merged = _dedupe_candidates([a, b])
        names = {s.name for s in merged[0].signals}
        assert {"fingerprint", "search_rank"} <= names

    def test_different_recordings_are_kept(self):
        a = make_candidate(title="Glory Box")
        b = make_candidate(title="Roads")
        assert len(_dedupe_candidates([a, b])) == 2

    def test_length_is_carried_over_when_missing(self):
        a = make_candidate(length_s=None)
        b = make_candidate(length_s=301.0)
        merged = _dedupe_candidates([a, b])
        assert merged[0].length_s == 301.0


class TestLengthParsing:
    def test_milliseconds_become_seconds(self):
        assert _length_s({"length": 301000}) == 301.0

    def test_missing_length_is_none(self):
        assert _length_s({}) is None
        assert _length_s(None) is None
        assert _length_s({"length": None}) is None


class TestAlbumConsolidation:
    def _album(self, matcher, release_ids):
        tracks = []
        for index, release_id in enumerate(release_ids, start=1):
            track = make_track(path=rf"C:\Music\Artist\Album\{index:02d} - Song {index}.flac",
                               title=f"Song {index}")
            cand = make_candidate(title=f"Song {index}")
            cand.tags.mb_release_id = release_id
            track.match = matcher._finish(track, matcher._observations(track), [cand], [], [])
            track.match.proposed.mb_release_id = release_id
            tracks.append(track)
        return tracks

    def test_does_nothing_for_a_tiny_folder(self, matcher, monkeypatch):
        called = []
        monkeypatch.setattr(matcher.mb, "full_release_tracklist",
                            lambda mbid: called.append(mbid) or [])
        matcher._consolidate_album(self._album(matcher, ["r1", "r2"]))
        assert not called, "two tracks is not enough evidence to snap an album together"

    def test_ignores_a_folder_with_no_majority(self, matcher, monkeypatch):
        called = []
        monkeypatch.setattr(matcher.mb, "full_release_tracklist",
                            lambda mbid: called.append(mbid) or [])
        matcher._consolidate_album(self._album(matcher, ["r1", "r2", "r3", "r4"]))
        assert not called

    def test_snaps_a_majority_folder_onto_one_release(self, matcher, monkeypatch):
        tracks = self._album(matcher, ["r1", "r1", "r1", "r2"])
        tracklist = [
            {"recording": {"id": f"rec{i}"}, "title": f"Song {i}", "length_ms": 301000,
             "track_no": i, "track_total": 4, "disc_no": 1, "disc_total": 1,
             "artist_credit": None}
            for i in range(1, 5)
        ]
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: tracklist)

        before = tracks[3].match.proposed.mb_release_id
        matcher._consolidate_album(tracks)
        assert before == "r2"
        assert tracks[3].match.proposed.mb_release_id == "r1", \
            "the odd one out should join the release the rest agreed on"
        assert all(t.match.proposed.mb_release_id == "r1" for t in tracks)

    def test_consolidation_assigns_track_numbers(self, matcher, monkeypatch):
        tracks = self._album(matcher, ["r1"] * 4)
        tracklist = [
            {"recording": {"id": f"rec{i}"}, "title": f"Song {i}", "length_ms": 301000,
             "track_no": i, "track_total": 4, "disc_no": 1, "disc_total": 1,
             "artist_credit": None}
            for i in range(1, 5)
        ]
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: tracklist)
        matcher._consolidate_album(tracks)
        numbers = sorted(t.match.proposed.track_no for t in tracks)
        assert numbers == [1, 2, 3, 4]

    def test_confidence_never_exceeds_the_cap(self, matcher, monkeypatch):
        tracks = self._album(matcher, ["r1"] * 6)
        tracklist = [
            {"recording": {"id": f"rec{i}"}, "title": f"Song {i}", "length_ms": 301000,
             "track_no": i, "track_total": 6, "disc_no": 1, "disc_total": 1,
             "artist_credit": None}
            for i in range(1, 7)
        ]
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: tracklist)
        matcher._consolidate_album(tracks)
        assert all(t.match.confidence <= 97.0 for t in tracks)


class TestConsolidationSeating:
    """Only the right file may take a tracklist slot - never a lookalike."""

    TRACKLIST = [
        {"recording": {"id": "rec-mysterons"}, "title": "Mysterons", "length_ms": 306000,
         "track_no": 1, "track_total": 3, "disc_no": 1, "disc_total": 1},
        {"recording": {"id": "rec-strangers"}, "title": "Strangers", "length_ms": 238000,
         "track_no": 2, "track_total": 3, "disc_no": 1, "disc_total": 1},
        {"recording": {"id": "rec-numb"}, "title": "Numb", "length_ms": 239000,
         "track_no": 3, "track_total": 3, "disc_no": 1, "disc_total": 1},
    ]

    def _track(self, matcher, name, title, seconds, release, recording):
        track = make_track(path=rf"C:\Music\Portishead\Dummy\{name}.mp3",
                           title=title, duration_s=seconds)
        cand = make_candidate(title=title, length_s=seconds)
        track.match = matcher._finish(track, matcher._observations(track), [cand], [], [])
        track.match.proposed.mb_release_id = release
        track.match.proposed.mb_recording_id = recording
        return track

    def test_a_different_song_of_the_same_length_is_not_reseated(self, matcher, monkeypatch):
        """Regression: "Sour Times" took "Strangers"' recording ID on a length match alone."""
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: self.TRACKLIST)
        tracks = [
            self._track(matcher, "01", "Mysterons", 306, "r1", "rec-mysterons"),
            self._track(matcher, "03", "Numb", 239, "r1", "rec-numb"),
            self._track(matcher, "xx", "Sour Times", 237, "r2", "rec-sour"),
            self._track(matcher, "02", "Strangers", 238, "r1", "rec-strangers"),
        ]
        matcher._consolidate_album(tracks)
        sour, strangers = tracks[2].match.proposed, tracks[3].match.proposed
        assert sour.mb_recording_id == "rec-sour"
        assert sour.mb_release_id == "r2"
        assert strangers.mb_recording_id == "rec-strangers"
        assert strangers.track_no == 2

    def test_seating_is_best_first_not_folder_order(self, matcher, monkeypatch):
        """An earlier, weaker match must not take the slot from the file it belongs to.

        The first file resembles both entries but fits "Wandering Star" a little
        worse than the second file, which *is* "Wandering Star". Taking files
        in folder order handed the first file that slot and pushed the real
        one onto "Wandering Stars".
        """
        tracklist = [
            {"recording": {"id": "rec-star"}, "title": "Wandering Star", "length_ms": 293000,
             "track_no": 1},
            {"recording": {"id": "rec-stars"}, "title": "Wandering Stars", "length_ms": 250000,
             "track_no": 2},
            {"recording": {"id": "rec-roads"}, "title": "Roads", "length_ms": 305000,
             "track_no": 3},
        ]
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: tracklist)
        tracks = [
            self._track(matcher, "a", "Wandering Stars", 293, "r1", None),
            self._track(matcher, "b", "Wandering Star", 293, "r1", None),
            self._track(matcher, "c", "Roads", 305, "r1", None),
        ]
        matcher._consolidate_album(tracks)
        assert tracks[1].match.proposed.mb_recording_id == "rec-star"
        assert tracks[0].match.proposed.mb_recording_id == "rec-stars"

    def test_an_embedded_recording_id_is_seated_by_identity(self, matcher, monkeypatch):
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: self.TRACKLIST)
        tracks = [
            self._track(matcher, "01", "Mysterons", 306, "r1", "rec-mysterons"),
            self._track(matcher, "03", "Numb", 239, "r1", "rec-numb"),
            # A title nothing like the tracklist's, but the same recording.
            self._track(matcher, "02", "Track 02", 100, "r1", "rec-strangers"),
        ]
        matcher._consolidate_album(tracks)
        assert tracks[2].match.proposed.track_no == 2


class TestIdentifyCancellation:
    def test_cancel_stops_inside_a_single_folder(self, matcher, monkeypatch):
        """A flat folder of downloads is one group; Cancel has to reach into it."""
        matcher.cfg.identify_workers = 1
        tracks = [make_track(path=rf"C:\Music\Dump\{i:03d}.mp3") for i in range(50)]
        seen = []
        monkeypatch.setattr(matcher, "identify",
                            lambda track: seen.append(track.path) or None)
        consolidated = []
        monkeypatch.setattr(matcher, "_consolidate_album",
                            lambda items: consolidated.append(items))

        matcher.identify_album(tracks, cancelled=lambda: len(seen) >= 3)
        assert len(seen) == 3
        assert not consolidated, "a half-identified folder must not be snapped together"


class TestDiscTotals:
    """MusicBrainz search results contain only the matching medium."""

    def test_partial_media_does_not_invent_a_disc_total(self):
        from musictag.providers.musicbrainz import _track_position
        release = {"media": [{"position": 2, "track-count": 12, "track-offset": 4}]}
        number, track_total, disc_no, disc_total = _track_position(release, None)
        assert disc_no == 2
        assert disc_total is None, "disc 2 of 1 is not a coherent answer"
        assert number == 5
        assert track_total == 12

    def test_full_release_reports_a_real_disc_total(self):
        from musictag.providers.musicbrainz import _track_position
        release = {"media": [
            {"position": 1, "track-count": 10, "track": [{"recording": {"id": "x"},
                                                          "position": 3}]},
            {"position": 2, "track-count": 8, "track": []},
        ]}
        number, track_total, disc_no, disc_total = _track_position(release, "x")
        assert (disc_no, disc_total) == (1, 2)
        assert number == 3


class TestSearchLoosening:
    """A hard duration filter must not turn a findable track into "no match"."""

    def _fake_mb(self, matcher, monkeypatch):
        """Record every query; only answer ones that omit the duration."""
        calls = []

        def search_recordings(**kwargs):
            calls.append(kwargs)
            if kwargs.get("duration_s"):
                return []            # nothing at this length, as for an edit
            return [{"id": "rec-1", "title": "Glory Box", "length": 301000,
                     "artist-credit": [{"name": "Portishead",
                                        "artist": {"id": "a1", "name": "Portishead"}}],
                     "releases": [], "score": 95}]

        monkeypatch.setattr(matcher.mb, "search_recordings", search_recordings)
        return calls

    def test_finds_a_match_after_dropping_the_duration_filter(self, matcher, monkeypatch):
        self._fake_mb(matcher, monkeypatch)
        observed = matcher._observations(make_track(duration_s=180.0))
        candidates = matcher._search_candidates(observed, [])
        assert candidates, "a track findable by title and artist must not come back empty"
        assert candidates[0].tags.title == "Glory Box"

    def test_it_says_the_search_was_widened(self, matcher, monkeypatch):
        self._fake_mb(matcher, monkeypatch)
        notes = []
        matcher._search_candidates(matcher._observations(make_track(duration_s=180.0)), notes)
        assert any("duration" in n.lower() for n in notes), \
            "a widened search has to be disclosed, not hidden"

    def test_duration_is_tried_before_it_is_dropped(self, matcher, monkeypatch):
        calls = self._fake_mb(matcher, monkeypatch)
        matcher._search_candidates(matcher._observations(make_track(duration_s=180.0)), [])
        assert calls[0].get("duration_s"), "the precise query must come first"
        assert not calls[-1].get("duration_s")

    def test_no_note_when_the_first_query_works(self, matcher, monkeypatch):
        monkeypatch.setattr(matcher.mb, "search_recordings", lambda **k: [
            {"id": "r", "title": "Glory Box", "length": 301000, "score": 100,
             "artist-credit": [{"name": "Portishead"}], "releases": []}])
        notes = []
        matcher._search_candidates(matcher._observations(make_track()), notes)
        assert notes == []

    def test_loosened_matches_still_score_badly_on_duration(self, matcher, monkeypatch):
        """Widening the search must not launder a bad match into a good score."""
        self._fake_mb(matcher, monkeypatch)
        track = make_track(duration_s=20.0)           # 20s file, 5min recording
        observed = matcher._observations(track)
        candidates = matcher._search_candidates(observed, [])
        result = matcher._finish(track, observed, candidates, [], ["search"])
        assert result.confidence < 60, \
            "a 4-minute length disagreement has to show up in the score"
        assert result.candidates, "but the candidate is still offered, with its reason"

    def test_duplicate_queries_are_not_repeated(self, matcher, monkeypatch):
        """With no album and no artist there is only one distinct query to make."""
        calls = self._fake_mb(matcher, monkeypatch)
        track = Track(path=r"C:\M\x.mp3")
        track.current = TrackTags(title="Solo")
        track.props = AudioProps(duration_s=0)
        matcher._search_candidates(matcher._observations(track), [])
        keys = [tuple(sorted((k, v) for k, v in c.items() if v and k != "limit"))
                for c in calls]
        assert len(keys) == len(set(keys)), f"repeated queries: {keys}"


class TestSearchPoolSize:
    """MusicBrainz's own relevance ranking is a coarse text-similarity
    heuristic, not our scoring - the right answer, especially on a file with
    weak or missing tags, is often not its top guess. The search pool is kept
    wider than max_candidates so a match a few slots down MusicBrainz's own
    order still gets scored instead of being cut before it is ever compared.
    """

    def _recordings(self, count: int) -> list[dict]:
        return [
            {"id": f"rec-{i}", "title": "Glory Box", "length": 301000, "score": 100 - i,
             "artist-credit": [{"name": "Portishead"}], "releases": []}
            for i in range(count)
        ]

    def test_more_than_max_candidates_recordings_can_be_scored(self, matcher, monkeypatch):
        cap = matcher.cfg.max_candidates
        monkeypatch.setattr(matcher.mb, "search_recordings",
                            lambda **k: self._recordings(cap * 3))
        candidates = matcher._search_candidates(matcher._observations(make_track()), [])
        assert len(candidates) > cap, \
            "the search pool must hold more than the final display cap"

    def test_query_limit_scales_with_the_wider_pool(self, matcher, monkeypatch):
        calls = []

        def search_recordings(**kwargs):
            calls.append(kwargs)
            return self._recordings(1)

        monkeypatch.setattr(matcher.mb, "search_recordings", search_recordings)
        matcher._search_candidates(matcher._observations(make_track()), [])
        # The old behaviour fetched max_candidates * 2 - this must ask for
        # more than that, to actually have headroom for the wider pool kept.
        assert calls[0]["limit"] > matcher.cfg.max_candidates * 2


class TestUnverifiedDuration:
    """A widened search must never produce a "Confident" badge.

    The user is simultaneously told "check the length before trusting this".
    A high score alongside that warning is a contradiction.
    """

    def test_ceiling_keeps_a_widened_match_out_of_the_confident_band(self, cfg):
        cand = make_candidate(length_s=None)
        cand.duration_unverified = True
        ceiling = _evidence_ceiling({"evidence": 8, "from_tags": True}, cand)
        assert ceiling < cfg.auto_apply_threshold

    def test_a_precise_match_is_not_capped(self):
        cand = make_candidate()
        assert _evidence_ceiling({"evidence": 8, "from_tags": True}, cand) == 93.0

    def test_a_missing_length_is_scored_as_unchecked_not_ignored(self, matcher):
        """A candidate with no length must not outscore one that matches exactly."""
        track = make_track(duration_s=301.0)
        observed = matcher._observations(track)
        exact = matcher._score(make_candidate(length_s=301.0), observed)
        unknown = matcher._score(make_candidate(length_s=None), observed)
        assert unknown < exact

    def test_the_missing_length_is_explained_in_the_signals(self, matcher):
        track = make_track(duration_s=301.0)
        cand = make_candidate(length_s=None)
        matcher._score(cand, matcher._observations(track))
        duration = [s for s in cand.signals if s.name == "duration"]
        assert duration and "could not be checked" in duration[0].detail

    def test_a_widened_match_lands_below_auto_apply_end_to_end(self, cfg, matcher):
        track = make_track(duration_s=301.0)
        cand = make_candidate(length_s=None)
        cand.duration_unverified = True
        result = matcher._finish(track, matcher._observations(track), [cand], [], ["search"])
        assert result.confidence < cfg.auto_apply_threshold, \
            "a widened search cannot yield a confident result"

    def test_corroboration_from_a_precise_search_clears_the_caveat(self):
        widened = make_candidate("musicbrainz-search")
        widened.duration_unverified = True
        precise = make_candidate("acoustid", fingerprint=0.95)
        precise.duration_unverified = False
        merged = _dedupe_candidates([widened, precise])
        assert merged[0].duration_unverified is False

    def test_the_caveat_stands_when_every_source_needed_widening(self):
        a = make_candidate("musicbrainz-search")
        a.duration_unverified = True
        b = make_candidate("acoustid", fingerprint=0.9)
        b.duration_unverified = True
        assert _dedupe_candidates([a, b])[0].duration_unverified is True


class TestCostOfIdentifying:
    """Identifying a library is almost entirely time spent waiting on
    MusicBrainz, so what matters as much as the answer is how many round trips
    it took to get there. These pin the shortcuts that made it affordable."""

    def _fingerprinted(self, matcher, monkeypatch, score, calls):
        """Wire up a matcher whose fingerprint lookup returns one candidate."""
        cand = make_candidate("acoustid", fingerprint=score)
        monkeypatch.setattr(
            Matcher, "_acoustid_candidates",
            lambda self, path, observed: ([cand], "somefingerprint", 301))
        monkeypatch.setattr(
            type(matcher.acoustid), "available", property(lambda self: True))
        monkeypatch.setattr(
            Matcher, "_search_candidates",
            lambda self, observed, notes=None: (calls.append("search"), [])[1])
        monkeypatch.setattr(Matcher, "_enrich", lambda self, cand: None)
        return cand

    def test_a_strong_fingerprint_skips_the_text_search(self, matcher, monkeypatch):
        calls = []
        self._fingerprinted(matcher, monkeypatch, 0.95, calls)
        matcher.identify(make_track())
        assert calls == [], "the audio was already identified; searching is wasted time"

    def test_a_weak_fingerprint_still_searches(self, matcher, monkeypatch):
        calls = []
        self._fingerprinted(matcher, monkeypatch, 0.3, calls)
        matcher.identify(make_track())
        assert calls == ["search"], "a weak hit is not an identification"

    def test_fingerprint_candidates_cost_no_lookups_to_build(self, matcher, monkeypatch):
        """They are built from AcoustID's own response. Fetching full details
        for every candidate up front was the single most expensive thing the
        matcher did, for data that was about to be thrown away."""
        lookups = []
        monkeypatch.setattr(matcher.mb, "lookup_recording",
                            lambda rec_id: lookups.append(rec_id))
        monkeypatch.setattr(
            matcher.acoustid, "identify",
            lambda path: ([{"score": 0.97, "recordings": [
                {"id": "r1", "title": "Glory Box", "duration": 301.0,
                 "artists": [{"id": "a1", "name": "Portishead"}]},
                {"id": "r2", "title": "Glory Box", "duration": 280.0,
                 "artists": [{"id": "a2", "name": "Covers Band"}]},
            ]}], "fp", 301))

        cands, _fp, _dur = matcher._acoustid_candidates(
            Path(r"C:\M\x.mp3"), matcher._observations(make_track()))

        assert lookups == [], "building candidates must not hit MusicBrainz"
        assert {c.tags.artist for c in cands} == {"Portishead", "Covers Band"}
        assert cands[0].tags.title == "Glory Box"
        assert cands[0].length_s == pytest.approx(301.0), "duration must survive"


class TestFingerprintClusterOrdering:
    def test_recordings_are_tried_closest_length_first(self):
        """AcoustID returns a cluster in no useful order - the original can sit
        below a karaoke version - so length against the file decides."""
        recordings = [
            {"id": "far", "duration": 172.0},
            {"id": "near", "duration": 167.1},
            {"id": "unknown"},
        ]
        order = [r["id"] for r in sorted(recordings, key=lambda r: _duration_gap(r, 168.0))]
        assert order == ["near", "far", "unknown"], \
            "an unknown length must sort last, not count as a match"

    def test_the_artist_credited_most_across_a_cluster_wins_a_tie(self, matcher, monkeypatch):
        """The original artist accumulates MusicBrainz entries - album, single,
        reissues - where a covers band has one. On a file with no tags, that is
        the only thing separating two identical-looking candidates."""
        monkeypatch.setattr(
            matcher.acoustid, "identify",
            lambda path: ([{"score": 0.98, "recordings": [
                {"id": "cover", "title": "All the Small Things", "duration": 168.1,
                 "artists": [{"id": "c", "name": "Covers Band"}]},
                {"id": "orig", "title": "All the Small Things", "duration": 167.1,
                 "artists": [{"id": "b", "name": "blink-182"}]},
                {"id": "orig2", "title": "All the Small Things", "duration": 171.0,
                 "artists": [{"id": "b", "name": "blink-182"}]},
                {"id": "orig3", "title": "All the Small Things (remaster)",
                 "duration": 167.5, "artists": [{"id": "b", "name": "blink-182"}]},
            ]}], "fp", 168))

        track = Track(path=r"C:\M\All the Small Things.mp3")
        track.current = TrackTags(title="All the Small Things")
        track.props = AudioProps(duration_s=168.0)
        observed = matcher._observations(track)
        cands, _fp, _dur = matcher._acoustid_candidates(Path(track.path), observed)

        consensus = {c.raw_id: next(
            (s.score for s in c.signals if s.name == "fingerprint_consensus"), 0.0)
            for c in cands}
        assert consensus["orig"] > consensus["cover"]

        for cand in cands:
            cand.confidence = matcher._score(cand, observed)
        winner = max(cands, key=lambda c: c.confidence)
        assert winner.tags.artist == "blink-182"
