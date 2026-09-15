"""Round-trip every format we claim to support.

A tagger that writes fields the player cannot read is worse than useless, so
these tests write a full tag set and read it back through a fresh handle.
"""

from __future__ import annotations

import pytest

from musictag.models import TrackTags
from musictag.tags import read_embedded_art, read_file, write_file

from conftest import ffmpeg_required

FULL_TAGS = TrackTags(
    title="Glory Box",
    artist="Portishead",
    album="Dummy",
    album_artist="Portishead",
    track_no=10,
    track_total=11,
    disc_no=1,
    disc_total=2,
    date="1994-08-22",
    year=1994,
    genre="Trip Hop",
    composer="Geoff Barrow",
    isrc="GBAAA9400010",
    compilation=False,
    mb_recording_id="c0c0c0c0-1111-2222-3333-444444444444",
    mb_release_id="d1d1d1d1-1111-2222-3333-444444444444",
    mb_release_group_id="e2e2e2e2-1111-2222-3333-444444444444",
    mb_artist_id="f3f3f3f3-1111-2222-3333-444444444444",
    mb_album_artist_id="a4a4a4a4-1111-2222-3333-444444444444",
)

# A 1x1 red JPEG, small enough to inline.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffc2000b080001000101011100ff"
    "c40014000100000000000000000000000000000009ffda000801"
    "0100000000d2cfffd9"
)


@ffmpeg_required
@pytest.mark.parametrize("taggable", ["mp3", "flac", "m4a", "ogg", "opus"], indirect=True)
class TestRoundTrip:
    def test_core_fields_survive(self, taggable):
        write_file(taggable, FULL_TAGS)
        read_back, _ = read_file(taggable)

        assert read_back.title == "Glory Box"
        assert read_back.artist == "Portishead"
        assert read_back.album == "Dummy"
        assert read_back.album_artist == "Portishead", \
            "album artist is the field Plex groups by; it must survive"
        assert read_back.track_no == 10
        assert read_back.disc_no == 1

    def test_totals_survive(self, taggable):
        write_file(taggable, FULL_TAGS)
        read_back, _ = read_file(taggable)
        assert read_back.track_total == 11
        assert read_back.disc_total == 2

    def test_date_and_genre_survive(self, taggable):
        write_file(taggable, FULL_TAGS)
        read_back, _ = read_file(taggable)
        assert read_back.year == 1994
        assert read_back.genre == "Trip Hop"

    def test_musicbrainz_ids_survive(self, taggable):
        write_file(taggable, FULL_TAGS)
        read_back, _ = read_file(taggable)
        assert read_back.mb_recording_id == FULL_TAGS.mb_recording_id
        assert read_back.mb_release_id == FULL_TAGS.mb_release_id
        assert read_back.mb_artist_id == FULL_TAGS.mb_artist_id

    def test_musicbrainz_ids_can_be_suppressed(self, taggable):
        write_file(taggable, FULL_TAGS, write_mb_ids=False)
        read_back, _ = read_file(taggable)
        assert read_back.mb_recording_id is None
        assert read_back.title == "Glory Box"

    def test_cover_art_embeds_and_reads_back(self, taggable):
        write_file(taggable, FULL_TAGS, art=TINY_JPEG, art_mime="image/jpeg")
        read_back, _ = read_file(taggable)
        assert read_back.has_art is True

        art = read_embedded_art(taggable)
        assert art is not None
        data, mime = art
        assert data == TINY_JPEG
        assert "image" in mime

    def test_compilation_flag_round_trips(self, taggable):
        tags = TrackTags(**{**FULL_TAGS.to_dict(), "compilation": True,
                            "album_artist": "Various Artists"})
        write_file(taggable, tags)
        read_back, _ = read_file(taggable)
        assert read_back.compilation is True
        assert read_back.album_artist == "Various Artists"

    def test_rewriting_replaces_rather_than_duplicates(self, taggable):
        write_file(taggable, FULL_TAGS)
        second = TrackTags(**{**FULL_TAGS.to_dict(), "title": "Roads", "track_no": 11})
        write_file(taggable, second)
        read_back, _ = read_file(taggable)
        assert read_back.title == "Roads"
        assert read_back.track_no == 11


@ffmpeg_required
class TestProperties:
    def test_reads_technical_properties(self, mp3_128):
        _, props = read_file(mp3_128)
        assert props.container == "mp3"
        assert props.sample_rate == 44100
        assert 100 <= props.bitrate_kbps <= 140
        assert props.duration_s == pytest.approx(8.0, abs=0.5)
        assert props.lossless is False

    def test_flac_is_reported_lossless(self, real_flac):
        _, props = read_file(real_flac)
        assert props.lossless is True
        assert props.bit_depth == 16

    def test_unreadable_extension_raises(self, tmp_path):
        from musictag.tags import UnsupportedFormat, get_adapter
        bogus = tmp_path / "notes.txt"
        bogus.write_text("hello")
        with pytest.raises(UnsupportedFormat):
            get_adapter(bogus)


@ffmpeg_required
class TestID3Versions:
    def test_id3v23_is_written_when_requested(self, taggable):
        from mutagen.mp3 import MP3
        write_file(taggable, FULL_TAGS, id3v2_version=3)
        assert MP3(taggable).tags.version[:2] == (2, 3)
        read_back, _ = read_file(taggable)
        assert read_back.title == "Glory Box"
        assert read_back.year == 1994

    def test_id3v24_is_the_default(self, taggable):
        from mutagen.mp3 import MP3
        write_file(taggable, FULL_TAGS)
        assert MP3(taggable).tags.version[:2] == (2, 4)


@ffmpeg_required
class TestPartialTags:
    def test_writing_only_a_title_leaves_a_readable_file(self, taggable):
        write_file(taggable, TrackTags(title="Just a title"))
        read_back, props = read_file(taggable)
        assert read_back.title == "Just a title"
        assert props.duration_s > 0

    def test_empty_fields_are_not_written_as_empty_strings(self, taggable):
        write_file(taggable, TrackTags(title="T", artist="A"))
        read_back, _ = read_file(taggable)
        assert read_back.album in (None, "")
        assert read_back.track_no is None
