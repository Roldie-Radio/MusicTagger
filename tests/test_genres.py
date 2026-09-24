"""Genres from MusicBrainz.

Genre is a community vote, not a fact about a recording, so these tests pin
down the rules that keep it sensible: one genre per album, a real consensus
rather than a stray vote, and never at the cost of the match itself.
"""

from __future__ import annotations

import pytest

from conftest import REAL_GENRE_FOR
from musictag.config import Config
from musictag.matching import Matcher
from musictag.models import Candidate, Track, TrackTags
from musictag.providers.http import ProviderError
from musictag.providers.musicbrainz import (
    VARIOUS_ARTISTS_MBID, MusicBrainzClient, _genre_display, _top_genre,
)


def genres(*pairs):
    return {"genres": [{"name": name, "count": count} for name, count in pairs]}


@pytest.fixture
def cfg() -> Config:
    config = Config()
    config.acoustid_api_key = ""
    return config


@pytest.fixture
def matcher(cfg) -> Matcher:
    return Matcher(cfg)


@pytest.fixture
def client(cfg, monkeypatch) -> MusicBrainzClient:
    monkeypatch.setattr(MusicBrainzClient, "genre_for", REAL_GENRE_FOR)
    return MusicBrainzClient(cfg)


class TestTopGenre:
    def test_most_votes_wins(self):
        assert _top_genre(genres(("rock", 3), ("trip hop", 9), ("electronic", 5))) == "Trip Hop"

    def test_a_single_vote_is_not_a_genre(self):
        assert _top_genre(genres(("polka", 1))) is None

    def test_ties_break_alphabetically(self):
        assert _top_genre(genres(("rock", 4), ("pop", 4))) == "Pop"

    def test_nothing_to_go_on(self):
        assert _top_genre(None) is None
        assert _top_genre({}) is None
        assert _top_genre({"genres": []}) is None

    @pytest.mark.parametrize("raw, shown", [
        ("alternative rock", "Alternative Rock"),
        ("k-pop", "K-Pop"),
        ("drum and bass", "Drum and Bass"),
        ("r&b", "R&B"),
        ("uk garage", "UK Garage"),
        ("the blues", "The Blues"),
    ])
    def test_display_case(self, raw, shown):
        assert _genre_display(raw) == shown


class TestGenreFor:
    def test_album_genre_comes_first(self, client, monkeypatch):
        monkeypatch.setattr(client, "lookup_release_group", lambda mbid: genres(("trip hop", 5)))
        monkeypatch.setattr(client, "lookup_artist", lambda mbid: pytest.fail("not needed"))
        tags = TrackTags(mb_release_group_id="rg", mb_album_artist_id="a1")
        assert client.genre_for(tags) == "Trip Hop"

    def test_falls_back_to_the_album_artist(self, client, monkeypatch):
        asked = []
        monkeypatch.setattr(client, "lookup_release_group", lambda mbid: genres())
        monkeypatch.setattr(client, "lookup_artist",
                            lambda mbid: asked.append(mbid) or genres(("electronic", 7)))
        tags = TrackTags(mb_release_group_id="rg", mb_album_artist_id="aa", mb_artist_id="a")
        assert client.genre_for(tags) == "Electronic"
        assert asked == ["aa"], "the album artist keeps the whole album on one genre"

    def test_never_uses_various_artists(self, client, monkeypatch):
        monkeypatch.setattr(client, "lookup_release_group", lambda mbid: None)
        monkeypatch.setattr(client, "lookup_artist", lambda mbid: pytest.fail("meaningless"))
        tags = TrackTags(mb_release_group_id="rg", mb_album_artist_id=VARIOUS_ARTISTS_MBID)
        assert client.genre_for(tags) is None

    def test_no_ids_means_no_lookups(self, client, monkeypatch):
        monkeypatch.setattr(client, "lookup_release_group", lambda mbid: pytest.fail("no id"))
        monkeypatch.setattr(client, "lookup_artist", lambda mbid: pytest.fail("no id"))
        assert client.genre_for(TrackTags(title="x")) is None


def _track(genre=None) -> Track:
    track = Track(path="/Music/Portishead/Dummy/10 - Glory Box.flac")
    track.current = TrackTags(title="Glory Box", artist="Portishead", genre=genre)
    return track


def _candidate() -> Candidate:
    return Candidate(source="musicbrainz-search", raw_id="rec",
                     tags=TrackTags(title="Glory Box", artist="Portishead", album="Dummy",
                                    mb_release_group_id="rg"))


class TestMatcher:
    def test_fills_in_the_genre(self, matcher, monkeypatch):
        monkeypatch.setattr(matcher.mb, "genre_for", lambda tags: "Trip Hop")
        cand = _candidate()
        matcher._fill_genre(cand)
        assert matcher._build_proposal(_track(genre="Rock"), cand).genre == "Trip Hop"

    def test_keeps_the_existing_genre_when_musicbrainz_has_none(self, matcher, monkeypatch):
        monkeypatch.setattr(matcher.mb, "genre_for", lambda tags: None)
        cand = _candidate()
        matcher._fill_genre(cand)
        assert matcher._build_proposal(_track(genre="Rock"), cand).genre == "Rock"

    def test_only_fill_blanks_keeps_a_hand_set_genre(self, cfg, matcher, monkeypatch):
        cfg.preserve_existing_tags = True
        monkeypatch.setattr(matcher.mb, "genre_for", lambda tags: "Trip Hop")
        cand = _candidate()
        matcher._fill_genre(cand)
        assert matcher._build_proposal(_track(genre="Rock"), cand).genre == "Rock"

    def test_can_be_switched_off(self, cfg, matcher, monkeypatch):
        cfg.fetch_genres = False
        monkeypatch.setattr(matcher.mb, "genre_for", lambda tags: pytest.fail("switched off"))
        cand = _candidate()
        matcher._fill_genre(cand)
        assert cand.tags.genre is None

    def test_a_failed_lookup_costs_only_the_genre(self, matcher, monkeypatch):
        def boom(tags):
            raise ProviderError("MusicBrainz is down")
        monkeypatch.setattr(matcher.mb, "genre_for", boom)
        cand = _candidate()
        matcher._fill_genre(cand)
        assert cand.tags.genre is None
        assert cand.tags.album == "Dummy"

    def test_identify_writes_the_genre(self, matcher, monkeypatch):
        monkeypatch.setattr(matcher.mb, "genre_for", lambda tags: "Trip Hop")
        track = _track()
        observed = matcher._observations(track)
        result = matcher._finish(track, observed, [_candidate()], [], ["search"])
        assert result.proposed.genre == "Trip Hop"
        assert "genre" in result.field_confidence


class TestAlbumConsolidation:
    def test_reseated_tracks_take_the_albums_genre(self, matcher, monkeypatch):
        monkeypatch.setattr(matcher.mb, "genre_for",
                            lambda tags: "Trip Hop" if tags.mb_release_group_id == "rg1" else "Pop")
        tracks = []
        for i, (rid, rg) in enumerate([("r1", "rg1")] * 3 + [("r2", "rg2")], start=1):
            track = Track(path=f"/Music/Artist/Album/{i:02d} - Song {i}.flac")
            track.current = TrackTags(title=f"Song {i}")
            cand = Candidate(source="musicbrainz-search", raw_id=f"rec{i}", length_s=301.0,
                             tags=TrackTags(title=f"Song {i}", artist="Artist", album="Album",
                                            mb_release_id=rid, mb_release_group_id=rg))
            track.match = matcher._finish(track, matcher._observations(track), [cand], [], [])
            tracks.append(track)
        assert tracks[3].match.proposed.genre == "Pop"
        tracklist = [{"recording": {"id": f"rec{i}"}, "title": f"Song {i}", "length_ms": 301000,
                      "track_no": i, "track_total": 4, "disc_no": 1, "disc_total": 1}
                     for i in range(1, 5)]
        monkeypatch.setattr(matcher.mb, "full_release_tracklist", lambda mbid: tracklist)
        matcher._consolidate_album(tracks)
        assert [t.match.proposed.genre for t in tracks] == ["Trip Hop"] * 4
