"""Finding tracks that already exist in a Plex library, before exporting into it.

Two passes, cheapest and most reliable first:

1. **MusicBrainz recording ID.** If the incoming track and something already in
   the Plex folder were both identified against MusicBrainz, an exact ID match
   is about as certain as this gets - flag it regardless of how differently
   the tags happen to be typed or capitalised.
2. **Fuzzy title/artist/album.** Covers the common case where the existing
   Plex copy predates this app (never tagged with an MB id) or came from a
   different source. Narrowed to same-normalised-title candidates first so
   this stays fast against a library with tens of thousands of tracks - a
   completely unrecognisable title (garbled beyond normalisation) will not be
   caught by this pass, only by a shared MusicBrainz ID.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .models import TrackTags
from .tags import SUPPORTED_EXTENSIONS, read_file
from .util import normalize, similarity

log = logging.getLogger(__name__)

#: How similar artist+title+album must be (0..1) to call it a fuzzy duplicate.
FUZZY_THRESHOLD = 0.82


@dataclass
class DuplicateMatch:
    kind: str                     # "mbid" | "fuzzy"
    existing_path: str
    existing_tags: TrackTags
    score: float                  # 1.0 for an mbid match, else the fuzzy score

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "existing_path": self.existing_path,
            "existing_tags": self.existing_tags.to_dict(),
            "score": round(self.score, 3),
        }


@dataclass
class ExistingIndex:
    root: Path
    by_mbid: dict[str, tuple[str, TrackTags]]
    by_title: dict[str, list[tuple[str, TrackTags]]]
    count: int


def build_index(root: Path, progress: Optional[Callable[[int, int, str], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None) -> ExistingIndex:
    """Walk ``root`` and read tags of everything already there.

    Tag-only reads, no network calls - this is the same cost as a library
    Scan, not an Identify.

    ``cancelled`` is checked between files, both while walking the folder and
    while reading tags. On a big library, or one on a network share, either
    step alone can take minutes, and Cancel has to work during both. A
    cancelled walk returns whatever it had indexed so far.
    """
    by_mbid: dict[str, tuple[str, TrackTags]] = {}
    by_title: dict[str, list[tuple[str, TrackTags]]] = {}
    count = 0

    files: list[Path] = []
    if root.exists():
        for path in root.rglob("*"):
            if cancelled and cancelled():
                return ExistingIndex(root=root, by_mbid=by_mbid, by_title=by_title, count=0)
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
                # The total is not known until the walk ends, so report what
                # has been found so far rather than showing nothing at all.
                if progress and len(files) % 200 == 0:
                    progress(0, 0, f"Found {len(files)} files so far")
    total = len(files)
    for done, path in enumerate(files, start=1):
        if cancelled and cancelled():
            break
        try:
            tags, _props = read_file(path)
        except Exception as exc:  # noqa: BLE001 - a bad file must not abort the whole index
            log.debug("Skipping unreadable file %s while indexing: %s", path, exc)
            if progress:
                progress(done, total, path.name)
            continue
        count += 1
        if tags.mb_recording_id:
            by_mbid.setdefault(tags.mb_recording_id, (str(path), tags))
        norm_title = normalize(tags.title, drop_feat=True)
        if norm_title:
            by_title.setdefault(norm_title, []).append((str(path), tags))
        if progress:
            progress(done, total, path.name)

    return ExistingIndex(root=root, by_mbid=by_mbid, by_title=by_title, count=count)


def find_duplicate(tags: TrackTags, index: ExistingIndex,
                    threshold: float = FUZZY_THRESHOLD) -> Optional[DuplicateMatch]:
    """Is ``tags`` already present somewhere in ``index``?"""
    if tags.mb_recording_id:
        hit = index.by_mbid.get(tags.mb_recording_id)
        if hit:
            path, existing = hit
            return DuplicateMatch(kind="mbid", existing_path=path,
                                  existing_tags=existing, score=1.0)

    norm_title = normalize(tags.title, drop_feat=True)
    candidates = index.by_title.get(norm_title, [])
    best: Optional[DuplicateMatch] = None
    for path, existing in candidates:
        if (tags.mb_recording_id and existing.mb_recording_id
                and tags.mb_recording_id != existing.mb_recording_id):
            # Both sides were identified, and confirmed to be different
            # recordings (e.g. a live version vs. the studio one) - text
            # similarity does not get a vote once MusicBrainz has spoken.
            continue
        title_sim = similarity(tags.title, existing.title, drop_feat=True)
        artist_sim = similarity(tags.artist or tags.album_artist,
                                existing.artist or existing.album_artist,
                                drop_feat=True, drop_articles=True)
        album_sim = similarity(tags.album, existing.album, drop_feat=True)
        # Artist agreement matters most for telling two different songs that
        # happen to share a title apart; album is corroborating, not required
        # (a compilation vs. the original studio album is still the same song).
        score = title_sim * 0.5 + artist_sim * 0.4 + album_sim * 0.1
        if score >= threshold and (best is None or score > best.score):
            best = DuplicateMatch(kind="fuzzy", existing_path=path,
                                  existing_tags=existing, score=score)
    return best
