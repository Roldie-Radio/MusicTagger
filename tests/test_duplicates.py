"""Duplicate detection against an existing Plex library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wave

from musictag.duplicates import build_index, find_duplicate
from musictag.models import TrackTags
from musictag.tags import write_file


def _write_wav(path: Path) -> Path:
    """A tiny, cheap, valid (silent) WAV - content does not matter for tag tests."""
    samples = np.zeros(4410, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(samples.tobytes())
    return path


def make_file(path: Path, **tag_kwargs) -> Path:
    _write_wav(path)
    write_file(path, TrackTags(**tag_kwargs))
    return path


def make_tags(**kwargs) -> TrackTags:
    base = dict(title="Glory Box", artist="Portishead", album="Dummy", album_artist="Portishead")
    base.update(kwargs)
    return TrackTags(**base)


class TestBuildIndex:
    def test_empty_folder(self, tmp_path):
        index = build_index(tmp_path)
        assert index.count == 0
        assert index.by_mbid == {}

    def test_missing_folder_is_not_an_error(self, tmp_path):
        index = build_index(tmp_path / "does-not-exist-yet")
        assert index.count == 0

    def test_indexes_by_mbid_and_title(self, tmp_path):
        make_file(tmp_path / "a.wav", title="Glory Box", artist="Portishead",
                  mb_recording_id="abc-123")
        index = build_index(tmp_path)
        assert index.count == 1
        assert "abc-123" in index.by_mbid
        assert "glory box" in index.by_title

    def test_cancelled_before_reading_indexes_nothing(self, tmp_path):
        make_file(tmp_path / "a.wav", title="Glory Box", artist="Portishead")
        index = build_index(tmp_path, cancelled=lambda: True)
        assert index.count == 0

    def test_cancel_stops_between_files(self, tmp_path):
        for n in range(5):
            make_file(tmp_path / f"{n}.wav", title=f"Song {n}", artist="Portishead")
        reads = []

        def progress(done, total, name):
            if total:
                reads.append(name)

        # Let the walk finish, then cancel once two files have been read.
        index = build_index(tmp_path, progress=progress,
                            cancelled=lambda: len(reads) >= 2)
        assert index.count == 2

    def test_unreadable_file_does_not_abort_the_scan(self, tmp_path):
        (tmp_path / "junk.wav").write_bytes(b"not a real wav file")
        make_file(tmp_path / "real.wav", title="Glory Box", artist="Portishead")
        index = build_index(tmp_path)
        assert index.count == 1


class TestFindDuplicate:
    def test_matching_mbid_is_a_duplicate_even_with_different_text(self, tmp_path):
        make_file(tmp_path / "existing.wav", title="GLORY BOX (remaster)",
                  artist="portishead", mb_recording_id="abc-123")
        index = build_index(tmp_path)

        incoming = make_tags(mb_recording_id="abc-123")
        dup = find_duplicate(incoming, index)
        assert dup is not None
        assert dup.kind == "mbid"
        assert dup.score == 1.0

    def test_different_mbid_is_not_a_duplicate_even_with_same_title(self, tmp_path):
        make_file(tmp_path / "existing.wav", title="Glory Box", artist="Portishead",
                  mb_recording_id="abc-123")
        index = build_index(tmp_path)

        # A cover version, or a completely different match - different mbid wins.
        incoming = make_tags(mb_recording_id="xyz-999")
        dup = find_duplicate(incoming, index)
        assert dup is None

    def test_fuzzy_match_on_title_artist_album(self, tmp_path):
        make_file(tmp_path / "existing.wav", title="Glory Box", artist="Portishead",
                  album="Dummy")
        index = build_index(tmp_path)

        incoming = make_tags(title="Glory Box", artist="Portishead", album="Dummy")
        dup = find_duplicate(incoming, index)
        assert dup is not None
        assert dup.kind == "fuzzy"
        assert dup.score >= 0.82

    def test_same_title_different_artist_is_not_a_duplicate(self, tmp_path):
        # "Yesterday" - The Beatles vs. a cover band - a real, common case.
        make_file(tmp_path / "existing.wav", title="Yesterday", artist="The Beatles",
                  album="Help!")
        index = build_index(tmp_path)

        incoming = make_tags(title="Yesterday", artist="Boyce Avenue", album="Cover Sessions")
        dup = find_duplicate(incoming, index)
        assert dup is None

    def test_no_title_no_match(self, tmp_path):
        make_file(tmp_path / "existing.wav", title="Glory Box", artist="Portishead")
        index = build_index(tmp_path)
        dup = find_duplicate(TrackTags(), index)
        assert dup is None

    def test_picks_best_of_several_candidates_sharing_a_title(self, tmp_path):
        make_file(tmp_path / "wrong-artist.wav", title="Yesterday", artist="Boyce Avenue")
        make_file(tmp_path / "right-artist.wav", title="Yesterday", artist="The Beatles",
                  album="Help!")
        index = build_index(tmp_path)

        incoming = make_tags(title="Yesterday", artist="The Beatles", album="Help!")
        dup = find_duplicate(incoming, index)
        assert dup is not None
        assert dup.existing_path.endswith("right-artist.wav")
