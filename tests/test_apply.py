"""End-to-end: scan a folder, apply tags, reorganise, then undo it all.

This is the test that matters most for trust. The app edits files in place, so
"undo actually restores what was there" has to be demonstrated, not asserted in
a docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musictag.apply import Applier, ApplyOptions
from musictag.config import Config
from musictag.journal import Journal
from musictag.library import iter_audio_files, scan
from musictag.models import MatchResult, Track, TrackTags
from musictag.tags import read_file, write_file

from conftest import encode, ffmpeg_required


@pytest.fixture
def library(tmp_path, clean_wav):
    """A small album laid out the way a messy library usually is."""
    album = tmp_path / "raw" / "some folder"
    album.mkdir(parents=True)
    files = []
    for index in range(1, 4):
        dest = album / f"track{index}.mp3"
        encode(clean_wav, dest, "-codec:a", "libmp3lame", "-b:a", "192k")
        write_file(dest, TrackTags(title=f"Old Title {index}", artist="Old Artist"))
        files.append(dest)
    (album / "cover.jpg").write_bytes(b"\xff\xd8\xff\xe0not-really-a-jpeg")
    return album, files


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.organize_enabled = True
    config.organize_root = str(tmp_path / "Library")
    config.organize_mode = "move"
    config.write_cover_art = False        # no network in tests
    config.write_cover_file = False
    return config


@pytest.fixture
def journal(tmp_path, monkeypatch) -> Journal:
    j = Journal(tmp_path / "journal.db")
    monkeypatch.setattr("musictag.apply.get_journal", lambda: j)
    return j


def proposed_for(index: int) -> TrackTags:
    return TrackTags(
        title=f"New Title {index}",
        artist="Portishead",
        album="Dummy",
        album_artist="Portishead",
        track_no=index,
        track_total=3,
        disc_no=1,
        disc_total=1,
        date="1994",
        year=1994,
    )


def build_tracks(files) -> list[Track]:
    tracks = []
    for index, path in enumerate(files, start=1):
        track = Track(path=str(path), filename=path.name)
        track.current, track.props = read_file(path)
        track.match = MatchResult(confidence=95.0, proposed=proposed_for(index),
                                  candidates=[], method="test")
        tracks.append(track)
    return tracks


# ===========================================================================

@ffmpeg_required
class TestScanning:
    def test_finds_every_audio_file(self, library):
        album, files = library
        found = list(iter_audio_files([album]))
        assert len(found) == len(files)

    def test_reads_existing_tags(self, library):
        album, _ = library
        tracks = scan([album], workers=2)
        assert len(tracks) == 3
        assert all(t.current.artist == "Old Artist" for t in tracks)
        assert all(t.props.duration_s > 0 for t in tracks)

    def test_ignores_non_audio_files(self, library):
        album, _ = library
        assert not any(p.name == "cover.jpg" for p in iter_audio_files([album]))

    def test_missing_path_is_skipped_not_fatal(self, tmp_path):
        assert scan([tmp_path / "does-not-exist"]) == []


@ffmpeg_required
class TestApplyTags:
    def test_dry_run_changes_nothing(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        report = Applier(cfg).apply(tracks, ApplyOptions(dry_run=True, organize=True))

        assert report.dry_run
        assert report.tagged == 3
        for path in files:
            tags, _ = read_file(path)
            assert tags.title.startswith("Old Title"), "dry run must not write"
            assert path.exists()

    def test_dry_run_reports_where_files_would_go(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        report = Applier(cfg).apply(tracks, ApplyOptions(dry_run=True, organize=True))
        assert len(report.planned) == 3
        assert all("Portishead" in move["to"] for move in report.planned)

    def test_tags_are_written(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        Applier(cfg).apply(tracks, ApplyOptions(organize=False))

        for index, path in enumerate(files, start=1):
            tags, _ = read_file(path)
            assert tags.title == f"New Title {index}"
            assert tags.album == "Dummy"
            assert tags.album_artist == "Portishead"
            assert tags.track_no == index

    def test_confidence_floor_skips_weak_matches(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        tracks[0].match.confidence = 40.0

        report = Applier(cfg).apply(tracks, ApplyOptions(min_confidence=70.0))
        assert report.tagged == 2
        assert report.skipped == 1
        tags, _ = read_file(files[0])
        assert tags.title == "Old Title 1", "a low-confidence match must be left alone"

    def test_a_missing_file_is_an_error_not_a_crash(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        files[0].unlink()
        report = Applier(cfg).apply(tracks, ApplyOptions())
        assert report.failed == 1
        assert report.tagged == 2


@ffmpeg_required
class TestOrganize:
    def test_files_land_in_the_plex_layout(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        Applier(cfg).apply(tracks, ApplyOptions(organize=True))

        root = Path(cfg.organize_root)
        expected = root / "Portishead" / "Dummy (1994)"
        assert expected.is_dir()
        landed = sorted(p.name for p in expected.glob("*.mp3"))
        assert landed == ["01 - New Title 1.mp3", "02 - New Title 2.mp3",
                          "03 - New Title 3.mp3"]

    def test_originals_are_gone_after_a_move(self, library, cfg, journal):
        album, files = library
        Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        assert not any(p.exists() for p in files)

    def test_copy_mode_leaves_the_originals(self, library, cfg, journal):
        cfg.organize_mode = "copy"
        album, files = library
        Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        assert all(p.exists() for p in files)
        assert list(Path(cfg.organize_root).rglob("*.mp3"))

    def test_track_path_is_updated_after_a_move(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        Applier(cfg).apply(tracks, ApplyOptions(organize=True))
        for track in tracks:
            assert Path(track.path).exists()
            assert "Portishead" in track.path

    def test_cover_art_travels_with_the_album(self, library, cfg, journal):
        cfg.keep_extra_files = True
        album, files = library
        Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        moved = Path(cfg.organize_root) / "Portishead" / "Dummy (1994)" / "cover.jpg"
        assert moved.exists(), "Plex uses folder art as a fallback; it must come along"


@ffmpeg_required
class TestUndo:
    def test_undo_restores_the_previous_tags(self, library, cfg, journal):
        album, files = library
        tracks = build_tracks(files)
        report = Applier(cfg).apply(tracks, ApplyOptions(organize=False))

        result = journal.undo(report.batch_id)
        assert result.failed == 0
        for index, path in enumerate(files, start=1):
            tags, _ = read_file(path)
            assert tags.title == f"Old Title {index}"
            assert tags.artist == "Old Artist"

    def test_undo_moves_files_back(self, library, cfg, journal):
        album, files = library
        report = Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        assert not files[0].exists()

        journal.undo(report.batch_id)
        for path in files:
            assert path.exists(), f"{path.name} was not returned to where it came from"

    def test_undo_restores_tags_after_a_move(self, library, cfg, journal):
        album, files = library
        report = Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        journal.undo(report.batch_id)
        tags, _ = read_file(files[0])
        assert tags.title == "Old Title 1"

    def test_undo_does_not_delete_copies(self, library, cfg, journal):
        """Deleting a file the user might want is not ours to decide."""
        cfg.organize_mode = "copy"
        album, files = library
        report = Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        copies = list(Path(cfg.organize_root).rglob("*.mp3"))

        result = journal.undo(report.batch_id)
        assert all(c.exists() for c in copies)
        assert any("copy" in m.lower() for m in result.messages)

    def test_undo_is_reported_not_silent(self, library, cfg, journal):
        album, files = library
        report = Applier(cfg).apply(build_tracks(files), ApplyOptions())
        result = journal.undo(report.batch_id)
        assert result.restored > 0


@ffmpeg_required
class TestJournal:
    def test_a_batch_is_recorded(self, library, cfg, journal):
        album, files = library
        report = Applier(cfg).apply(build_tracks(files), ApplyOptions(organize=True))
        batches = journal.list_batches()
        assert batches and batches[0]["id"] == report.batch_id

    def test_entries_capture_the_previous_tags(self, library, cfg, journal):
        album, files = library
        report = Applier(cfg).apply(build_tracks(files), ApplyOptions())
        entries = journal.batch_entries(report.batch_id)
        tag_entries = [e for e in entries if e["op"] == "tags"]
        assert len(tag_entries) == 3
        assert all(e["prev_tags"] for e in tag_entries)

    def test_dry_run_writes_no_journal_entries(self, library, cfg, journal):
        album, files = library
        Applier(cfg).apply(build_tracks(files), ApplyOptions(dry_run=True, organize=True))
        assert journal.list_batches() == []

    def test_prune_keeps_the_most_recent(self, journal):
        for i in range(5):
            journal.finish_batch(journal.start_batch(f"batch {i}"), {})
        removed = journal.prune(keep=2)
        assert removed == 3
        assert len(journal.list_batches()) == 2
