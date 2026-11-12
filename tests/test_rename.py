"""Renaming files in place from their tags, and what happens when names clash."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from musictag.config import Config
from musictag.journal import Journal
from musictag.library import load_track, scan
from musictag.merge import DUPLICATES_DIRNAME
from musictag.models import MatchResult, TrackTags
from musictag.rename import commit_renames, new_name, plan_renames
from musictag.tags import read_file, write_file


def make_file(path: Path, *, seconds: float = 0.1, **tags) -> Path:
    """A tiny valid WAV with the given tags. ``seconds`` sets its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(44100 * seconds), dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(samples.tobytes())
    write_file(path, TrackTags(**tags))
    return path


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.library_paths = [str(tmp_path / "music")]
    config.write_cover_art = False
    return config


@pytest.fixture
def journal(tmp_path, monkeypatch) -> Journal:
    j = Journal(tmp_path / "journal.db")
    monkeypatch.setattr("musictag.rename.get_journal", lambda: j)
    return j


def tracks_in(folder: Path):
    return sorted(scan([folder]), key=lambda t: t.path)


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file())


class TestNewName:
    def test_uses_track_number_and_title(self, tmp_path, cfg):
        track = load_track(make_file(tmp_path / "x.wav", title="Glory Box", track_no=7))
        assert new_name(track, cfg) == "07 - Glory Box.wav"

    def test_without_a_track_number_is_just_the_title(self, tmp_path, cfg):
        track = load_track(make_file(tmp_path / "x.wav", title="Glory Box"))
        assert new_name(track, cfg) == "Glory Box.wav"

    def test_no_title_means_no_name(self, tmp_path, cfg):
        track = load_track(make_file(tmp_path / "x.wav", artist="Portishead", track_no=7))
        assert new_name(track, cfg) is None

    def test_proposal_wins_over_the_tags_on_disk(self, tmp_path, cfg):
        track = load_track(make_file(tmp_path / "x.wav", title="glory bx"))
        track.match = MatchResult(proposed=TrackTags(title="Glory Box", track_no=2))
        assert new_name(track, cfg) == "02 - Glory Box.wav"

    def test_characters_a_filename_cannot_hold_are_replaced(self, tmp_path, cfg):
        track = load_track(make_file(tmp_path / "x.wav", title="What/If: Part 1?"))
        name = new_name(track, cfg)
        assert "/" not in name and ":" not in name and "?" not in name

    def test_follows_the_filename_template(self, tmp_path, cfg):
        cfg.file_template = "{artist} - {title}"
        track = load_track(make_file(tmp_path / "x.wav", title="Roads", artist="Portishead"))
        assert new_name(track, cfg) == "Portishead - Roads.wav"


class TestRename:
    def test_files_are_renamed_in_their_own_folder(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "a" / "track1.wav", title="Mysterons", track_no=1)
        make_file(music / "b" / "zzz.wav", title="Roads", track_no=8)
        tracks = tracks_in(music)

        report = commit_renames(plan_renames(tracks, cfg), cfg)

        assert report.renamed == 2 and report.failed == 0
        assert names(music / "a") == ["01 - Mysterons.wav"]
        assert names(music / "b") == ["08 - Roads.wav"]
        assert sorted(Path(t.path).name for t in tracks) == ["01 - Mysterons.wav", "08 - Roads.wav"]
        assert [t.filename for t in tracks] == [Path(t.path).name for t in tracks]

    def test_planning_touches_nothing(self, tmp_path, cfg):
        music = tmp_path / "music"
        make_file(music / "track1.wav", title="Mysterons", track_no=1)
        plan = plan_renames(tracks_in(music), cfg)
        assert plan.to_dict()["rename_count"] == 1
        assert names(music) == ["track1.wav"]

    def test_correctly_named_and_untitled_files_are_left_alone(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "01 - Mysterons.wav", title="Mysterons", track_no=1)
        make_file(music / "mystery.wav", artist="Somebody")
        plan = plan_renames(tracks_in(music), cfg)
        report = commit_renames(plan, cfg)
        assert (report.renamed, report.unchanged, report.skipped) == (0, 1, 1)
        assert report.batch_id == ""          # nothing to undo, so no empty history entry
        assert names(music) == ["01 - Mysterons.wav", "mystery.wav"]

    def test_undo_puts_the_old_names_back(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "track1.wav", title="Mysterons", track_no=1)
        make_file(music / "track2.wav", title="Sour Times", track_no=2)
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert names(music) == ["01 - Mysterons.wav", "02 - Sour Times.wav"]

        result = journal.undo(report.batch_id)
        assert result.restored == 2 and result.failed == 0
        assert names(music) == ["track1.wav", "track2.wav"]

    def test_a_name_held_by_a_file_being_renamed_away_is_free(self, tmp_path, cfg, journal):
        """"a" -> "b" while "b" -> "c": order must not cost anyone their name."""
        music = tmp_path / "music"
        make_file(music / "Roads.wav", title="Sour Times")
        make_file(music / "old.wav", title="Roads")
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert report.renamed == 2 and report.failed == 0
        assert names(music) == ["Roads.wav", "Sour Times.wav"]
        assert read_file(music / "Roads.wav")[0].title == "Roads"

    def test_two_files_swapping_names(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "Roads.wav", title="Sour Times")
        make_file(music / "Sour Times.wav", title="Roads")
        tracks = tracks_in(music)
        report = commit_renames(plan_renames(tracks, cfg), cfg)

        assert report.renamed == 2 and report.failed == 0
        assert names(music) == ["Roads.wav", "Sour Times.wav"]
        assert read_file(music / "Roads.wav")[0].title == "Roads"
        assert read_file(music / "Sour Times.wav")[0].title == "Sour Times"
        assert all(Path(t.path).exists() for t in tracks)

        journal.undo(report.batch_id)
        assert read_file(music / "Roads.wav")[0].title == "Sour Times"
        assert names(music) == ["Roads.wav", "Sour Times.wav"]


class TestClashes:
    def test_different_songs_with_one_title_both_stay(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "a.wav", title="Intro", artist="Portishead", album="Dummy")
        make_file(music / "b.wav", title="Intro", artist="The xx", album="xx")
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert report.renamed == 2 and report.merged == 0
        assert names(music) == ["Intro (2).wav", "Intro.wav"]
        assert not (music / DUPLICATES_DIRNAME).exists()

    def test_two_copies_of_one_track_are_merged(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy", track_no=8)
        make_file(music / "Dummy" / "roads.wav", seconds=0.1, **tags)
        make_file(music / "Dummy" / "roads copy.wav", seconds=0.2, **tags)   # the bigger file
        tracks = tracks_in(music)
        plan = plan_renames(tracks, cfg)
        assert plan.to_dict()["duplicate_count"] == 1

        report = commit_renames(plan, cfg)

        assert report.renamed == 1 and report.merged == 1 and report.failed == 0
        assert names(music / "Dummy") == ["08 - Roads.wav"]
        aside = music / DUPLICATES_DIRNAME / "Dummy" / "roads.wav"
        assert aside.exists()
        assert report.retired == [str(music / "Dummy" / "roads.wav")]
        assert report.folders == [str(music / DUPLICATES_DIRNAME)]
        # The kept copy is the better (larger) one.
        assert (music / "Dummy" / "08 - Roads.wav").stat().st_size > aside.stat().st_size

    def test_merging_can_be_switched_off(self, tmp_path, cfg, journal):
        cfg.merge_duplicates = False
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "a.wav", **tags)
        make_file(music / "b.wav", **tags)
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert report.merged == 0
        assert names(music) == ["Roads (2).wav", "Roads.wav"]

    def test_kept_copy_gains_the_tags_only_the_other_had(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "big.wav", seconds=0.3, title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "small.wav", seconds=0.1, title="Roads", artist="Portishead",
                  album="Dummy", genre="Trip Hop", date="1994")
        tracks = tracks_in(music)
        # Fewer tags, but the tags are what is being compared here - make the
        # big file win on audio grounds.
        next(t for t in tracks if t.filename == "big.wav").props.bit_depth = 24

        report = commit_renames(plan_renames(tracks, cfg), cfg)

        kept, _props = read_file(music / "Roads.wav")
        assert (music / "Roads.wav").stat().st_size > 20000
        assert kept.genre == "Trip Hop" and kept.date == "1994"

        journal.undo(report.batch_id)
        assert names(music) == ["big.wav", "small.wav"]
        restored, _props = read_file(music / "big.wav")
        assert restored.genre is None and restored.date is None

    def test_a_better_copy_takes_the_name_from_the_file_that_had_it(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "Roads.wav", seconds=0.1, **tags)          # already well named
        make_file(music / "download.wav", seconds=0.3, **tags)       # better copy
        tracks = tracks_in(music)
        plan = plan_renames(tracks, cfg)
        # The file that was "unchanged" is now the duplicate - listed once.
        assert sorted(i.action for i in plan.items) == ["duplicate", "rename"]

        report = commit_renames(plan, cfg)

        assert names(music) == ["Roads.wav"]
        assert (music / "Roads.wav").stat().st_size > 20000
        assert (music / DUPLICATES_DIRNAME / "Roads.wav").exists()
        assert report.retired == [str(music / "Roads.wav")]

    def test_a_worse_copy_is_set_aside_and_the_named_file_stays(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "Roads.wav", seconds=0.3, **tags)
        make_file(music / "download.wav", seconds=0.1, **tags)
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert report.renamed == 0 and report.merged == 1
        assert names(music) == ["Roads.wav"]
        assert (music / "Roads.wav").stat().st_size > 20000
        assert (music / DUPLICATES_DIRNAME / "download.wav").exists()

    def test_a_file_outside_the_selection_is_never_overwritten(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "Intro.wav", title="Intro", artist="The xx", album="xx")
        mine = make_file(music / "a.wav", title="Intro", artist="Portishead", album="Dummy")
        report = commit_renames(plan_renames([load_track(mine)], cfg), cfg)
        assert report.renamed == 1
        assert names(music) == ["Intro (2).wav", "Intro.wav"]
        assert read_file(music / "Intro.wav")[0].artist == "The xx"

    def test_same_song_from_different_albums_is_not_merged(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        make_file(music / "a.wav", title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "b.wav", title="Roads", artist="Portishead", album="Roseland NYC Live")
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert report.merged == 0
        assert names(music) == ["Roads (2).wav", "Roads.wav"]


class TestDuplicatesFolder:
    def test_a_later_scan_does_not_find_the_set_aside_copies(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "a.wav", **tags)
        make_file(music / "b.wav", **tags)
        commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        assert (music / DUPLICATES_DIRNAME).is_dir()
        assert [Path(t.path).name for t in scan([music])] == ["Roads.wav"]

    def test_undo_brings_the_duplicate_back(self, tmp_path, cfg, journal):
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "a.wav", **tags)
        make_file(music / "b.wav", **tags)
        report = commit_renames(plan_renames(tracks_in(music), cfg), cfg)
        journal.undo(report.batch_id)
        assert names(music) == ["a.wav", "b.wav"]
