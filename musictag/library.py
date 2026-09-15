"""Walking the library and reading what is already on disk."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .models import Track
from .tags import SUPPORTED_EXTENSIONS, read_file

log = logging.getLogger(__name__)

#: Directories we never descend into.
SKIP_DIRS = {
    "$RECYCLE.BIN", "System Volume Information", ".git", ".svn", "__pycache__",
    "@eaDir", ".Trash", ".Trashes", "#recycle", ".plexignore",
}


def iter_audio_files(roots: Iterable[str | Path], *, recursive: bool = True) -> Iterator[Path]:
    """Yield every supported audio file under ``roots``."""
    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            if root_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield root_path
            continue
        if not root_path.is_dir():
            log.warning("Skipping missing path: %s", root_path)
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            if not recursive:
                dirnames[:] = []
            for name in filenames:
                if name.startswith("._"):          # macOS resource forks
                    continue
                if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield Path(dirpath) / name


def load_track(path: Path) -> Track:
    """Build a :class:`Track` from one file, capturing errors rather than raising."""
    track = Track(path=str(path), filename=path.name)
    try:
        stat = path.stat()
        track.mtime = stat.st_mtime
    except OSError as exc:
        track.status = "error"
        track.error = f"Cannot stat file: {exc}"
        return track
    try:
        track.current, track.props = read_file(path)
    except Exception as exc:  # unreadable/corrupt files must not kill the scan
        track.status = "error"
        track.error = f"Could not read tags: {exc}"
        log.debug("Tag read failed for %s", path, exc_info=True)
    return track


def scan(
    roots: Iterable[str | Path],
    *,
    recursive: bool = True,
    workers: int = 8,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> list[Track]:
    """Scan ``roots`` and return one :class:`Track` per audio file.

    ``progress`` is called as ``(done, total, current_path)``.
    """
    paths = list(iter_audio_files(roots, recursive=recursive))
    total = len(paths)
    tracks: list[Track] = []
    if not paths:
        return tracks

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, track in enumerate(pool.map(load_track, paths), start=1):
            tracks.append(track)
            if progress and (done % 25 == 0 or done == total):
                progress(done, total, track.path)
            if cancelled and cancelled():
                break
    return tracks


def group_by_album(tracks: Iterable[Track]) -> dict[tuple[str, str], list[Track]]:
    """Group tracks into albums using the best album/artist info available.

    Album grouping matters twice over: MusicBrainz lookups are far more accurate
    when we can compare a whole tracklist at once, and Plex needs a consistent
    album artist across every file in an album.
    """
    groups: dict[tuple[str, str], list[Track]] = {}
    for track in tracks:
        tags = track.current
        album = (tags.album or "").strip().lower()
        artist = (tags.album_artist or tags.artist or "").strip().lower()
        if not album:
            # Fall back to the containing folder, which is how most rips are laid out.
            album = Path(track.path).parent.name.lower()
            artist = artist or Path(track.path).parent.parent.name.lower()
        groups.setdefault((artist, album), []).append(track)
    return groups
