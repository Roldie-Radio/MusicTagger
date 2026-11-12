"""Finding copies of the same track inside the library, and merging them."""

from __future__ import annotations

from pathlib import Path

import pytest

from musictag.apply import Applier, ApplyOptions
from musictag.config import Config
from musictag.duplicates import find_groups, pick_best, same_recording, why_best
from musictag.journal import Journal
from musictag.library import load_track, scan
from musictag.merge import DUPLICATES_DIRNAME, merge_groups
from musictag.models import (AudioProps, MatchResult, QualityIssue, QualityReport,
                             Track, TrackTags)
from musictag.tags import read_embedded_art, read_file, write_file

from conftest import encode, ffmpeg_required
from test_rename import make_file, names


def track(path="/m/a.mp3", *, duration=200.0, lossless=False, bitrate=192,
          quality=None, proposed=None, **tags) -> Track:
    base = dict(title="Roads", artist="Portishead", album="Dummy")
    base.update(tags)
    t = Track(path=path, filename=Path(path).name, current=TrackTags(**base))
    t.props = AudioProps(container="flac" if lossless else "mp3", duration_s=duration,
                         lossless=lossless, bitrate_kbps=bitrate,
                         bit_depth=16 if lossless else 0,
                         sample_rate=44100, filesize=bitrate * 1000)
    if proposed:
        t.match = MatchResult(proposed=TrackTags(**proposed))
    t.quality = quality
    return t


def report(*issues: tuple[str, str], score=100) -> QualityReport:
    return QualityReport(score=score, analysed=True,
                         issues=[QualityIssue(code, sev, code, "") for code, sev in issues])


class TestSameRecording:
    def test_same_musicbrainz_id_whatever_the_tags_say(self):
        a = TrackTags(title="Roads", mb_recording_id="r1")
        b = TrackTags(title="ROADS (live?)", artist="someone else", mb_recording_id="r1")
        assert same_recording(a, b).kind == "mbid"

    def test_different_musicbrainz_ids_are_never_the_same(self):
        a = TrackTags(title="Roads", artist="Portishead", mb_recording_id="r1")
        b = TrackTags(title="Roads", artist="Portishead", mb_recording_id="r2")
        assert same_recording(a, b) is None

    def test_matching_title_and_artist(self):
        a = TrackTags(title="Roads", artist="Portishead", album="Dummy")
        b = TrackTags(title="roads", artist="PORTISHEAD", album="Dummy (Remastered)")
        assert same_recording(a, b).kind == "fuzzy"

    def test_similar_titles_are_different_songs(self):
        a = TrackTags(title="Shine On You Crazy Diamond, Pts. 1-5", artist="Pink Floyd")
        b = TrackTags(title="Shine On You Crazy Diamond, Pts. 6-9", artist="Pink Floyd")
        assert same_recording(a, b) is None

    def test_same_title_by_another_artist(self):
        a = TrackTags(title="Intro", artist="Portishead")
        b = TrackTags(title="Intro", artist="The xx")
        assert same_recording(a, b) is None

    def test_very_different_lengths_are_different_versions(self):
        a = TrackTags(title="Roads", artist="Portishead")
        assert same_recording(a, a, a_duration=305, b_duration=306)
        assert same_recording(a, a, a_duration=305, b_duration=412) is None

    def test_notes_when_the_albums_differ(self):
        a = TrackTags(title="Roads", artist="Portishead", album="Dummy")
        b = TrackTags(title="Roads", artist="Portishead", album="Roseland NYC Live")
        assert same_recording(a, b).same_album is False
        assert same_recording(a, a).same_album is True


class TestPickBest:
    def test_lossless_beats_lossy(self):
        flac, mp3 = track("/m/a.flac", lossless=True, bitrate=900), track("/m/a.mp3", bitrate=320)
        assert pick_best([mp3, flac]) is flac
        assert why_best(flac, [mp3]) == "lossless"

    def test_higher_bitrate_wins_among_lossy(self):
        low, high = track("/m/a.mp3", bitrate=128), track("/m/b.mp3", bitrate=320)
        assert pick_best([low, high]) is high
        assert why_best(high, [low]) == "higher bitrate"

    def test_a_fake_lossless_file_does_not_count_as_lossless(self):
        fake = track("/m/a.flac", lossless=True, bitrate=900,
                     quality=report(("fake_lossless", "high")))
        mp3 = track("/m/a.mp3", bitrate=320, quality=report())
        assert pick_best([fake, mp3]) is mp3

    def test_a_damaged_file_loses_to_a_sound_one(self):
        damaged = track("/m/a.mp3", bitrate=320, quality=report(("truncated", "high")))
        sound = track("/m/b.mp3", bitrate=192, quality=report())
        assert pick_best([damaged, sound]) is sound

    def test_ties_are_stable(self):
        a, b = track("/m/a.mp3"), track("/m/b.mp3")
        assert pick_best([b, a]) is a and pick_best([a, b]) is a


class TestFindGroups:
    def test_no_duplicates(self):
        tracks = [track("/m/1.mp3", title="Roads"), track("/m/2.mp3", title="Glory Box")]
        assert find_groups(tracks) == []

    def test_groups_copies_best_first(self):
        mp3 = track("/m/a/roads.mp3", bitrate=128)
        flac = track("/m/b/roads.flac", lossless=True)
        other = track("/m/a/glory.mp3", title="Glory Box")
        groups = find_groups([mp3, other, flac])
        assert len(groups) == 1
        assert groups[0].tracks == [flac, mp3]
        assert groups[0].keep is flac
        assert groups[0].kind == "fuzzy" and groups[0].same_album
        assert groups[0].to_dict()["keep"] == "/m/b/roads.flac"

    def test_an_unapplied_proposal_counts(self):
        """Identified but not applied: the proposal already names the recording."""
        a = track("/m/a.mp3", title="track 08", artist=None, album=None,
                  proposed=dict(title="Roads", artist="Portishead", mb_recording_id="r1"))
        b = track("/m/b.mp3", title="Roads", artist="Portishead", mb_recording_id="r1")
        groups = find_groups([a, b])
        assert len(groups) == 1 and groups[0].kind == "mbid"

    def test_different_albums_are_grouped_but_flagged(self):
        a = track("/m/a.mp3", album="Dummy")
        b = track("/m/b.mp3", album="Roseland NYC Live")
        groups = find_groups([a, b])
        assert len(groups) == 1 and groups[0].same_album is False

    def test_an_untagged_copy_cannot_join_two_different_recordings(self):
        studio = track("/m/studio.mp3", mb_recording_id="r1")
        live = track("/m/live.mp3", mb_recording_id="r2")
        loose = track("/m/loose.mp3")
        groups = find_groups([studio, live, loose])
        assert len(groups) == 1 and len(groups[0].tracks) == 2
        assert sum(t is studio or t is live for t in groups[0].tracks) == 1

    def test_untitled_files_are_never_grouped(self):
        a = track("/m/a.mp3", title=None)
        b = track("/m/b.mp3", title=None)
        assert find_groups([a, b]) == []


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.library_paths = [str(tmp_path / "music")]
    config.write_cover_art = True
    return config


@pytest.fixture
def journal(tmp_path, monkeypatch) -> Journal:
    j = Journal(tmp_path / "journal.db")
    monkeypatch.setattr("musictag.merge.get_journal", lambda: j)
    monkeypatch.setattr("musictag.apply.get_journal", lambda: j)
    return j


class TestMergeGroups:
    def library(self, tmp_path):
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy")
        make_file(music / "Dummy" / "08 - Roads.wav", seconds=0.3, **tags)
        make_file(music / "Downloads" / "roads.wav", seconds=0.1, genre="Trip Hop", **tags)
        make_file(music / "Dummy" / "01 - Mysterons.wav", title="Mysterons", artist="Portishead")
        tracks = sorted(scan([music]), key=lambda t: t.path)
        # The album copy is the better audio; the download only has more tags.
        next(t for t in tracks if t.filename == "08 - Roads.wav").props.bit_depth = 24
        return music, tracks

    def test_keeps_the_best_and_sets_the_rest_aside(self, tmp_path, cfg, journal):
        music, tracks = self.library(tmp_path)
        (group,) = find_groups(tracks)
        assert Path(group.keep.path).name == "08 - Roads.wav"

        result = merge_groups([(group.keep, group.tracks[1:])], cfg)

        assert (result.groups, result.merged, result.failed) == (1, 1, 0)
        assert names(music / "Dummy") == ["01 - Mysterons.wav", "08 - Roads.wav"]
        assert not (music / "Downloads" / "roads.wav").exists()
        # Same relative path, so it is obvious where each copy came from.
        assert (music / DUPLICATES_DIRNAME / "Downloads" / "roads.wav").exists()
        assert result.retired == [str(music / "Downloads" / "roads.wav")]
        assert result.folders == [str(music / DUPLICATES_DIRNAME)]

    def test_kept_copy_inherits_missing_tags_without_losing_its_own(self, tmp_path, cfg, journal):
        music, tracks = self.library(tmp_path)
        (group,) = find_groups(tracks)
        result = merge_groups([(group.keep, group.tracks[1:])], cfg)
        kept, _props = read_file(music / "Dummy" / "08 - Roads.wav")
        assert result.tags_filled == 1
        assert kept.genre == "Trip Hop"
        assert (kept.title, kept.artist, kept.album) == ("Roads", "Portishead", "Dummy")
        assert group.keep.current.genre == "Trip Hop"

    def test_undo_restores_files_and_tags(self, tmp_path, cfg, journal):
        music, tracks = self.library(tmp_path)
        (group,) = find_groups(tracks)
        result = merge_groups([(group.keep, group.tracks[1:])], cfg)

        undone = journal.undo(result.batch_id)

        assert undone.failed == 0
        assert (music / "Downloads" / "roads.wav").exists()
        assert read_file(music / "Dummy" / "08 - Roads.wav")[0].genre is None

    def test_nothing_is_set_aside_when_the_kept_copy_is_missing(self, tmp_path, cfg, journal):
        music, tracks = self.library(tmp_path)
        (group,) = find_groups(tracks)
        Path(group.keep.path).unlink()
        result = merge_groups([(group.keep, group.tracks[1:])], cfg)
        assert result.merged == 0 and result.failed == 1
        assert (music / "Downloads" / "roads.wav").exists()

    @ffmpeg_required
    def test_kept_copy_inherits_cover_art(self, tmp_path, cfg, journal, clean_wav):
        music = tmp_path / "music"
        music.mkdir()
        tags = TrackTags(title="Roads", artist="Portishead", album="Dummy")
        good = encode(clean_wav, music / "good.mp3", "-codec:a", "libmp3lame", "-b:a", "320k")
        poor = encode(clean_wav, music / "poor.mp3", "-codec:a", "libmp3lame", "-b:a", "96k")
        art = b"\xff\xd8\xff\xe0" + b"art" * 50
        write_file(good, tags)
        write_file(poor, tags, art=art)
        keep, other = load_track(good), load_track(poor)
        assert not keep.current.has_art and other.current.has_art

        merge_groups([(keep, [other])], cfg)

        assert read_embedded_art(good)[0] == art
        assert keep.current.has_art


class TestOrganiseCollision:
    """Organising two copies of one track into the same place merges them."""

    def setup_library(self, tmp_path, cfg):
        cfg.organize_enabled = True
        cfg.organize_root = ""                # organise inside the library folder
        cfg.write_cover_art = False
        cfg.write_cover_file = False
        music = tmp_path / "music"
        tags = dict(title="Roads", artist="Portishead", album="Dummy",
                    album_artist="Portishead", track_no=8)
        make_file(music / "in" / "small.wav", seconds=0.1, **tags)
        make_file(music / "in" / "big.wav", seconds=0.3, **tags)
        tracks = sorted(scan([music]), key=lambda t: t.path)
        for t in tracks:
            t.match = MatchResult(confidence=99, proposed=TrackTags(**tags))
        return music, tracks

    def test_better_copy_ends_up_in_place(self, tmp_path, cfg, journal):
        music, tracks = self.setup_library(tmp_path, cfg)
        by_name = {t.filename: t for t in tracks}
        # Worse copy first, so the better one arrives to find its place taken.
        result = Applier(cfg).apply([by_name["small.wav"], by_name["big.wav"]],
                                    ApplyOptions(organize=True))

        dest = music / "Portishead" / "Dummy" / "08 - Roads.wav"
        assert result.merged == 1 and result.failed == 0
        assert names(dest.parent) == ["08 - Roads.wav"]
        assert dest.stat().st_size > 20000
        assert (music / DUPLICATES_DIRNAME / "Portishead" / "Dummy" / "08 - Roads.wav").exists()
        assert result.retired == [str(dest)]

    def test_worse_copy_arriving_second_is_set_aside(self, tmp_path, cfg, journal):
        music, tracks = self.setup_library(tmp_path, cfg)
        by_name = {t.filename: t for t in tracks}
        result = Applier(cfg).apply([by_name["big.wav"], by_name["small.wav"]],
                                    ApplyOptions(organize=True))

        dest = music / "Portishead" / "Dummy" / "08 - Roads.wav"
        assert result.merged == 1 and result.moved == 1 and result.failed == 0
        assert dest.stat().st_size > 20000
        assert (music / DUPLICATES_DIRNAME / "in" / "small.wav").exists()
        assert result.retired == [str(music / "in" / "small.wav")]

    def test_undo_puts_both_copies_back(self, tmp_path, cfg, journal):
        music, tracks = self.setup_library(tmp_path, cfg)
        result = Applier(cfg).apply(tracks, ApplyOptions(organize=True))
        journal.undo(result.batch_id)
        assert names(music / "in") == ["big.wav", "small.wav"]

    def test_different_songs_still_get_separate_names(self, tmp_path, cfg, journal):
        music, tracks = self.setup_library(tmp_path, cfg)
        tracks[1].match.proposed.artist = "Someone Else"       # same folder + name, other song
        tracks[1].props.duration_s = 400
        result = Applier(cfg).apply(tracks, ApplyOptions(organize=True))
        assert result.merged == 0 and result.moved == 2
        assert names(music / "Portishead" / "Dummy") == ["08 - Roads (2).wav", "08 - Roads.wav"]

    def test_switched_off_keeps_both(self, tmp_path, cfg, journal):
        cfg.merge_duplicates = False
        music, tracks = self.setup_library(tmp_path, cfg)
        result = Applier(cfg).apply(tracks, ApplyOptions(organize=True))
        assert result.merged == 0 and result.moved == 2
