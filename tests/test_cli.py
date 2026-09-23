"""CLI behaviour that isn't already covered by exercising the same logic
through the API in test_server.py.
"""

from __future__ import annotations

import types

import pytest

from musictag.cli import cmd_scan
from musictag.config import Config, set_config
from musictag.state import AppState


@pytest.fixture
def cfg(tmp_path):
    config = Config()
    config.library_paths = []
    set_config(config)
    return config


@pytest.fixture
def state(monkeypatch):
    s = AppState()
    s._loaded = True
    monkeypatch.setattr("musictag.state.get_state", lambda: s)
    return s


def args(paths):
    return types.SimpleNamespace(paths=paths)


class TestScanLibraryPathMemory:
    """Mirrors TestLibraryPathMemory in test_server.py - the CLI and the app
    should agree on this: one remembered folder, not an accumulating list."""

    def test_an_explicit_path_replaces_the_remembered_folder(self, cfg, state, tmp_path):
        old_folder = tmp_path / "Old"
        new_folder = tmp_path / "New"
        old_folder.mkdir()
        new_folder.mkdir()
        cfg.library_paths = [str(old_folder)]

        assert cmd_scan(args([str(new_folder)])) == 0
        assert cfg.library_paths == [str(new_folder)]

    def test_bare_scan_does_not_touch_the_remembered_folder(self, cfg, state, tmp_path):
        folder = tmp_path / "Music"
        folder.mkdir()
        cfg.library_paths = [str(folder)]

        assert cmd_scan(args([])) == 0
        assert cfg.library_paths == [str(folder)]

    def test_duplicate_explicit_paths_do_not_pile_up(self, cfg, state, tmp_path):
        folder = tmp_path / "Music"
        folder.mkdir()

        cmd_scan(args([str(folder)]))
        cmd_scan(args([str(folder)]))
        assert cfg.library_paths == [str(folder)]

    def test_no_paths_and_nothing_configured_is_a_clean_error(self, cfg, state, capsys):
        assert cmd_scan(args([])) == 2
        assert "No paths given" in capsys.readouterr().out


class TestRelativePaths:
    """A relative path only means something from the directory it was typed in."""

    def _wav(self, path):
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)

    def test_scanning_a_relative_folder_stores_absolute_paths(self, cfg, state, tmp_path,
                                                             monkeypatch):
        folder = tmp_path / "Music"
        folder.mkdir()
        self._wav(folder / "-odd name.wav")
        monkeypatch.chdir(folder)

        assert cmd_scan(args(["."])) == 0
        assert cfg.library_paths == [str(folder)]
        paths = [t.path for t in state.all()]
        assert paths == [str(folder / "-odd name.wav")]
