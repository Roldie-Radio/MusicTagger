"""The ingest-folder -> Plex-folder export flow: planning and committing."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from musictag.config import Config
from musictag.export_plex import commit_export, has_pending_changes, plan_export
from musictag.journal import Journal
from musictag.models import MatchResult, Track, TrackTags
from musictag.tags import write_file


def _write_wav(path: Path) -> Path:
    samples = np.zeros(4410, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(samples.tobytes())
    return path


def make_file(path: Path, **tag_kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(path)
    write_file(path, TrackTags(**tag_kwargs))
    return path


def make_track(path: Path, current: TrackTags, proposed: TrackTags | None = None) -> Track:
    match = MatchResult(confidence=95.0, proposed=proposed) if proposed else None
    return Track(path=str(path), filename=path.name, current=current, match=match)


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.organize_root = str(tmp_path / "Plex")
    config.folder_template = "{album_artist}/{album}"
    config.file_template = "{track:02d} - {title}"
    return config


class TestHasPendingChanges:
    def test_no_match_is_not_pending(self, tmp_path):
        track = Track(path=str(tmp_path / "a.wav"), current=TrackTags(title="X"))
        assert has_pending_changes(track) is False

    def test_proposal_matching_current_is_not_pending(self, tmp_path):
        tags = TrackTags(title="Glory Box", artist="Portishead")
        track = make_track(tmp_path / "a.wav", current=tags, proposed=TrackTags(title="Glory Box"))
        assert has_pending_changes(track) is False

    def test_proposal_that_would_change_a_field_is_pending(self, tmp_path):
        current = TrackTags(title="glory box (bad rip)", artist="Portishead")
        proposed = TrackTags(title="Glory Box", artist="Portishead")
        track = make_track(tmp_path / "a.wav", current=current, proposed=proposed)
        assert has_pending_changes(track) is True

    def test_proposal_field_left_blank_does_not_count_as_a_change(self, tmp_path):
        # A writer never clears a field it did not propose a value for, so an
        # unset proposed field must not be treated as "would change to blank".
        current = TrackTags(title="Glory Box", genre="Trip Hop")
        proposed = TrackTags(title="Glory Box")  # genre untouched
        track = make_track(tmp_path / "a.wav", current=current, proposed=proposed)
        assert has_pending_changes(track) is False


class TestPlanExport:
    def test_requires_a_configured_plex_root(self, tmp_path):
        cfg = Config()
        cfg.organize_root = ""
        track = Track(path=str(tmp_path / "a.wav"), current=TrackTags(title="X", artist="Y"))
        with pytest.raises(ValueError):
            plan_export([track], cfg)

    def test_ready_track_gets_a_destination(self, cfg, tmp_path):
        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead",
                        album="Dummy", track_no=1)
        track = make_track(src, current=TrackTags(title="Glory Box", artist="Portishead",
                                                   album="Dummy", track_no=1))
        plan = plan_export([track], cfg)
        assert len(plan.items) == 1
        assert plan.items[0].duplicate is None
        assert "Portishead" in plan.items[0].dest and "Dummy" in plan.items[0].dest

    def test_track_with_unapplied_changes_is_held_back(self, cfg, tmp_path):
        src = make_file(tmp_path / "in" / "a.wav", title="glory box (bad rip)", artist="Portishead")
        current = TrackTags(title="glory box (bad rip)", artist="Portishead")
        proposed = TrackTags(title="Glory Box", artist="Portishead", album="Dummy")
        track = make_track(src, current=current, proposed=proposed)

        plan = plan_export([track], cfg)
        assert plan.items == []
        assert plan.pending_apply == [str(src)]

    def test_flags_a_track_already_in_the_plex_folder(self, cfg, tmp_path):
        make_file(Path(cfg.organize_root) / "Portishead" / "Dummy" / "01 - Glory Box.wav",
                 title="Glory Box", artist="Portishead", album="Dummy")

        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead",
                        album="Dummy", track_no=1)
        track = make_track(src, current=TrackTags(title="Glory Box", artist="Portishead",
                                                   album="Dummy", track_no=1))
        plan = plan_export([track], cfg)
        assert len(plan.items) == 1
        assert plan.items[0].duplicate is not None
        assert plan.items[0].duplicate.kind == "fuzzy"

    def test_untagged_track_is_ineligible(self, cfg, tmp_path):
        src = make_file(tmp_path / "in" / "a.wav")
        track = make_track(src, current=TrackTags())
        plan = plan_export([track], cfg)
        assert plan.items == []
        assert plan.ineligible == 1


class TestCommitExport:
    def test_export_moves_the_file(self, cfg, tmp_path):
        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        dest = Path(cfg.organize_root) / "Portishead" / "01 - Glory Box.wav"
        report = commit_export(
            [{"path": str(src), "dest": str(dest), "action": "export"}], cfg)
        assert report.exported == 1
        assert not src.exists()
        assert dest.exists()

    def test_skip_leaves_the_file_alone(self, cfg, tmp_path):
        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        dest = Path(cfg.organize_root) / "Portishead" / "01 - Glory Box.wav"
        report = commit_export(
            [{"path": str(src), "dest": str(dest), "action": "skip"}], cfg)
        assert report.skipped == 1
        assert src.exists()
        assert not dest.exists()

    def test_export_disambiguates_an_unrelated_name_collision(self, cfg, tmp_path):
        dest = Path(cfg.organize_root) / "Portishead" / "01 - Glory Box.wav"
        make_file(dest, title="Some Other Song Entirely", artist="Someone Else")

        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        report = commit_export(
            [{"path": str(src), "dest": str(dest), "action": "export"}], cfg)
        assert report.exported == 1
        assert not src.exists()
        assert dest.exists()                       # the original file untouched
        assert (Path(cfg.organize_root) / "Portishead" / "01 - Glory Box (2).wav").exists()

    def test_replace_moves_existing_file_to_trash_not_deleting_it(self, cfg, tmp_path):
        existing = make_file(Path(cfg.organize_root) / "Portishead" / "01 - Glory Box.wav",
                             title="Glory Box (old rip)", artist="Portishead")
        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")

        report = commit_export([{
            "path": str(src), "dest": str(existing), "action": "replace",
            "existing_path": str(existing),
        }], cfg)

        assert report.exported == 1
        assert report.replaced == 1
        assert not src.exists()
        assert existing.exists()                    # new file now lives here
        trash = list((Path(cfg.organize_root) / ".musictagger-trash").glob("*.wav"))
        assert len(trash) == 1                       # old file preserved, not deleted

    def test_replace_never_overwrites_an_unrelated_file_at_dest(self, cfg, tmp_path):
        """The duplicate being replaced can live somewhere other than ``dest``."""
        duplicate = make_file(Path(cfg.organize_root) / "Old Folder" / "glory box.wav",
                              title="Glory Box", artist="Portishead")
        unrelated = make_file(Path(cfg.organize_root) / "Portishead" / "01 - Glory Box.wav",
                              title="Something Else", artist="Someone Else")
        before = unrelated.read_bytes()
        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")

        report = commit_export([{
            "path": str(src), "dest": str(unrelated), "action": "replace",
            "existing_path": str(duplicate),
        }], cfg)

        assert report.exported == 1 and report.replaced == 1
        assert not duplicate.exists()                        # went to the trash
        assert unrelated.read_bytes() == before              # untouched
        assert (unrelated.parent / "01 - Glory Box (2).wav").exists()

    def test_failure_on_one_item_does_not_abort_the_batch(self, cfg, tmp_path):
        missing = tmp_path / "in" / "gone.wav"       # never created
        ok = make_file(tmp_path / "in" / "ok.wav", title="Glory Box", artist="Portishead")
        dest_ok = Path(cfg.organize_root) / "01 - Glory Box.wav"

        report = commit_export([
            {"path": str(missing), "dest": str(Path(cfg.organize_root) / "gone.wav"),
             "action": "export"},
            {"path": str(ok), "dest": str(dest_ok), "action": "export"},
        ], cfg)
        assert report.exported == 1
        assert dest_ok.exists()

    def test_moves_are_journalled_for_undo(self, cfg, tmp_path, monkeypatch):
        journal = Journal(path=tmp_path / "journal.db")
        monkeypatch.setattr("musictag.export_plex.get_journal", lambda: journal)

        src = make_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        dest = Path(cfg.organize_root) / "01 - Glory Box.wav"
        report = commit_export(
            [{"path": str(src), "dest": str(dest), "action": "export"}], cfg)

        entries = journal.batch_entries(report.batch_id)
        assert len(entries) == 1
        assert entries[0]["op"] == "move"
        assert entries[0]["dest"] == str(dest)

        result = journal.undo(report.batch_id)
        assert result.restored == 1
        assert src.exists()
        assert not dest.exists()
