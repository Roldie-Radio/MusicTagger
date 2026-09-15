from pathlib import Path

import pytest

from musictag.config import Config
from musictag.models import Track, TrackTags
from musictag.organize import plan_all, plan_path, render


def make_tags(**kwargs) -> TrackTags:
    base = dict(title="Glory Box", artist="Portishead", album="Dummy",
                album_artist="Portishead", track_no=10, track_total=11, year=1994)
    base.update(kwargs)
    return TrackTags(**base)


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.organize_enabled = True
    config.organize_root = str(tmp_path / "Music")
    return config


class TestRender:
    def test_default_folder_template(self):
        parts = render("{album_artist}/{album}{year_suffix}", make_tags())
        assert parts == ["Portishead", "Dummy (1994)"]

    def test_no_year_leaves_no_empty_brackets(self):
        parts = render("{album_artist}/{album}{year_suffix}", make_tags(year=None))
        assert parts == ["Portishead", "Dummy"]

    def test_track_zero_padding(self):
        parts = render("{disc_prefix}{track:02d} - {title}", make_tags())
        assert parts == ["10 - Glory Box"]

    def test_missing_track_does_not_crash_or_leave_separators(self):
        parts = render("{disc_prefix}{track:02d} - {title}", make_tags(track_no=None))
        assert parts == ["Glory Box"]

    def test_disc_prefix_only_for_multi_disc(self):
        single = render("{disc_prefix}{track:02d} - {title}",
                        make_tags(disc_no=1, disc_total=1))
        assert single == ["10 - Glory Box"]

        multi = render("{disc_prefix}{track:02d} - {title}",
                       make_tags(disc_no=2, disc_total=2))
        assert multi == ["2-10 - Glory Box"]

    def test_illegal_characters_are_sanitised(self):
        parts = render("{album_artist}/{album}", make_tags(album_artist="AC/DC", album="Back?"))
        assert parts == ["AC-DC", "Back"]

    def test_typo_in_placeholder_falls_back_to_the_default(self):
        # "{albumartist}" is a plausible typo for "{album_artist}". Filing every
        # album under a blank folder would be worse than ignoring the template.
        parts = render("{albumartist}/{album}", make_tags())
        assert parts == ["Portishead", "Dummy"]

    def test_invalid_format_spec_falls_back_instead_of_raising(self):
        parts = render("{album:d}", make_tags())
        assert parts == ["Portishead", "Dummy"]

    def test_slash_inside_a_tag_does_not_create_a_folder(self):
        parts = render("{album_artist}", make_tags(album_artist="AC/DC"))
        assert parts == ["AC-DC"]


class TestPlanPath:
    def test_builds_plex_shaped_path(self, cfg, tmp_path):
        track = Track(path=str(tmp_path / "junk" / "whatever.flac"), filename="whatever.flac")
        dest = plan_path(track, make_tags(), cfg)
        assert dest == Path(cfg.organize_root) / "Portishead" / "Dummy (1994)" / "10 - Glory Box.flac"

    def test_preserves_extension_lowercased(self, cfg, tmp_path):
        track = Track(path=str(tmp_path / "x.FLAC"), filename="x.FLAC")
        assert plan_path(track, make_tags(), cfg).suffix == ".flac"

    def test_various_artists_groups_under_one_folder(self, cfg, tmp_path):
        tags = make_tags(album_artist="Various Artists", artist="Some Guest",
                         album="Now That's What I Call Music")
        track = Track(path=str(tmp_path / "x.mp3"))
        dest = plan_path(track, tags, cfg)
        # Album artist, not track artist, is what keeps a compilation together.
        assert dest.parent.parent.name == "Various Artists"


class TestPlanAll:
    def test_disambiguates_collisions(self, cfg, tmp_path):
        from musictag.models import MatchResult

        tracks = []
        for name in ("a.mp3", "b.mp3"):
            track = Track(path=str(tmp_path / name), filename=name)
            track.match = MatchResult(confidence=90, proposed=make_tags())
            tracks.append(track)

        planned = plan_all(tracks, cfg)
        destinations = set(planned.values())
        assert len(destinations) == 2, "identical tags must not silently overwrite"
        assert any("(2)" in d for d in destinations)


class TestMultiDisc:
    def test_disc_prefix_when_total_is_unknown_but_disc_is_not_first(self):
        """MusicBrainz search results often omit the real disc count.

        Without a prefix, disc 1 track 1 and disc 2 track 1 both become
        "01 - Title" in the same folder and collide.
        """
        parts = render("{disc_prefix}{track:02d} - {title}",
                       make_tags(disc_no=2, disc_total=None, track_no=1, title="Song"))
        assert parts == ["2-01 - Song"]

    def test_no_prefix_for_disc_one_with_unknown_total(self):
        parts = render("{disc_prefix}{track:02d} - {title}",
                       make_tags(disc_no=1, disc_total=None, track_no=1, title="Song"))
        assert parts == ["01 - Song"]

    def test_two_discs_do_not_collide(self, cfg, tmp_path):
        from musictag.models import MatchResult
        tracks = []
        for disc in (1, 2):
            track = Track(path=str(tmp_path / f"d{disc}.mp3"), filename=f"d{disc}.mp3")
            track.match = MatchResult(
                confidence=95,
                proposed=make_tags(disc_no=disc, disc_total=2, track_no=1, title="Opener"))
            tracks.append(track)
        destinations = set(plan_all(tracks, cfg).values())
        assert len(destinations) == 2
        assert not any("(2)" in d for d in destinations), \
            "the disc prefix should separate them, not a collision suffix"
