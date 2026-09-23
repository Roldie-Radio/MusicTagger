"""Converting tracks to MP3 320, M4A and WMA.

These run the real ffmpeg, because the things that matter - the bitrate that
comes out, the tags and cover art that survive, the original left untouched -
are properties of the files produced, not of the code that asked for them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from musictag.config import Config
from musictag.convert import FORMATS, convert_tracks
from musictag.journal import Journal
from musictag.models import TrackTags
from musictag.tags import read_embedded_art, read_file, write_file

from conftest import encode, ffmpeg_required

pytestmark = ffmpeg_required

TAGS = TrackTags(title="Glory Box", artist="Portishead", album="Dummy",
                 album_artist="Portishead", track_no=11, track_total=11,
                 date="1994", genre="Trip Hop")
# The smallest valid PNG: enough for every container to accept as cover art.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


@pytest.fixture
def cfg() -> Config:
    config = Config()
    config.ffmpeg_path = shutil.which("ffmpeg") or ""
    return config


@pytest.fixture
def journal(tmp_path, monkeypatch) -> Journal:
    j = Journal(tmp_path / "journal.db")
    monkeypatch.setattr("musictag.convert.get_journal", lambda: j)
    return j


@pytest.fixture
def flac(tmp_path, clean_wav) -> Path:
    (tmp_path / "album").mkdir()
    path = encode(clean_wav, tmp_path / "album" / "11 - Glory Box.flac")
    write_file(path, TAGS, art=PNG, art_mime="image/png")
    return path


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_converts_beside_the_original_with_tags(fmt, flac, cfg, journal):
    before = flac.read_bytes()
    report = convert_tracks([str(flac)], fmt, cfg)

    assert report.converted == 1, report.errors
    out = flac.with_suffix(FORMATS[fmt]["ext"])
    assert report.created == [str(out)]
    assert out.exists()
    assert flac.read_bytes() == before, "the original must not change"
    assert not list(flac.parent.glob("*.converting*")), "no temporary file left"

    tags, props = read_file(out)
    assert tags.title == "Glory Box"
    assert tags.artist == "Portishead"
    assert tags.album == "Dummy"
    assert tags.track_no == 11
    if fmt != "wma":            # the WMA writer does not embed art
        assert read_embedded_art(out) is not None


def test_mp3_is_320_kbps(flac, cfg, journal):
    convert_tracks([str(flac)], "mp3", cfg)
    _tags, props = read_file(flac.with_suffix(".mp3"))
    assert props.bitrate_kbps >= 310


def test_same_format_is_skipped_not_reencoded(tmp_path, clean_wav, cfg, journal):
    mp3 = encode(clean_wav, tmp_path / "song.mp3", "-b:a", "128k")
    report = convert_tracks([str(mp3)], "mp3", cfg)
    assert report.skipped == 1 and report.converted == 0
    assert "Already MP3" in report.skipped_reasons[0]["reason"]


def test_never_overwrites_an_existing_file(flac, cfg, journal):
    existing = flac.with_suffix(".mp3")
    existing.write_bytes(b"someone else's mp3")
    report = convert_tracks([str(flac)], "mp3", cfg)
    assert existing.read_bytes() == b"someone else's mp3"
    assert report.created == [str(flac.with_name(f"{flac.stem} (2).mp3"))]


def test_a_bad_file_fails_alone_and_leaves_nothing_behind(tmp_path, flac, cfg, journal):
    junk = tmp_path / "album" / "junk.flac"
    junk.write_bytes(b"not audio at all")
    report = convert_tracks([str(junk), str(flac)], "m4a", cfg)
    assert report.failed == 1 and report.converted == 1
    assert not list(junk.parent.glob("junk*.m4a"))
    assert not list(junk.parent.glob("*.converting*"))


def test_cancel_stops_between_files(tmp_path, clean_wav, cfg, journal):
    files = []
    for n in range(3):
        files.append(str(encode(clean_wav, tmp_path / f"{n}.flac")))
    seen = []
    report = convert_tracks(files, "mp3", cfg,
                            progress=lambda d, t, n: seen.append(n),
                            cancelled=lambda: len(seen) >= 1)
    assert report.converted == 1


def test_undo_leaves_converted_files_and_says_so(flac, cfg, journal):
    report = convert_tracks([str(flac)], "mp3", cfg)
    result = journal.undo(report.batch_id)
    assert flac.with_suffix(".mp3").exists()
    assert any("Converted file left in place" in m for m in result.messages)


def test_without_ffmpeg_it_reports_instead_of_crashing(flac, journal):
    config = Config()
    config.ffmpeg_path = ""
    config.resolve_tool = lambda name, override: None  # type: ignore[method-assign]
    report = convert_tracks([str(flac)], "mp3", config)
    assert report.failed == 1
    assert "ffmpeg" in report.errors[0]["error"]
