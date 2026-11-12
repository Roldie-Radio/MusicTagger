"""Merging copies of the same track: keep the best, set the rest aside.

"Merge" means two things here, and both are undoable:

* **The kept copy inherits what the others had and it lacked** - a tag it
  was missing, or cover art. Nothing it already has is overwritten.
* **The other copies move into a ``Duplicates`` folder** at the top of the
  library, under the same relative path they had. They are never deleted -
  the journal's "nothing here deletes user data" rule (see
  :mod:`musictag.journal`) holds here too - and the folder is skipped by
  every later scan, so set-aside copies do not come back as new duplicates.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import Config
from .journal import Journal, get_journal
from .library import DUPLICATES_MARKER
from .models import TAG_FIELDS, Track, TrackTags
from .tags import read_embedded_art, write_file
from .util import unique_path

log = logging.getLogger(__name__)

#: The folder set-aside copies go to, inside the library root.
DUPLICATES_DIRNAME = "Duplicates"


def _root_for(path: Path, cfg: Config) -> Path:
    """The library (or Plex) folder ``path`` lives in; failing that, its own folder."""
    absolute = Path(os.path.abspath(path))
    for root in list(cfg.library_paths) + ([cfg.organize_root] if cfg.organize_root else []):
        root_path = Path(os.path.abspath(root))
        try:
            absolute.relative_to(root_path)
            return root_path
        except ValueError:
            continue
    return absolute.parent


def duplicates_folder(path: Path, cfg: Config) -> Path:
    """Where a duplicate of the file at ``path`` would be set aside."""
    return _root_for(path, cfg) / DUPLICATES_DIRNAME


def retire_destination(path: Path, cfg: Config) -> Path:
    """The path ``path`` would be set aside at (before making the name unique)."""
    root = _root_for(path, cfg)
    return root / DUPLICATES_DIRNAME / Path(os.path.abspath(path)).relative_to(root)


def retire(path: Path, cfg: Config, journal: Journal, batch_id: str) -> Path:
    """Move ``path`` into the Duplicates folder. Journalled, so undo restores it."""
    folder = duplicates_folder(path, cfg)
    dest = unique_path(retire_destination(path, cfg))
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = folder / DUPLICATES_MARKER
    if not marker.exists():
        try:
            marker.write_text(
                "MusicTagger moves duplicate copies of tracks into this folder.\n"
                "It is skipped when scanning. Delete the folder whenever you are\n"
                "sure you do not need the copies.\n", encoding="utf-8")
        except OSError as exc:
            log.debug("Could not write %s: %s", marker, exc)
    with journal.step(batch_id, "move", str(path), dest=str(dest)):
        shutil.move(str(path), str(dest))
    return dest


def absorb(keeper: Track, others: Iterable[Track], cfg: Config,
           journal: Journal, batch_id: str) -> list[str]:
    """Fill the kept copy's blank tags (and missing art) from the other copies.

    Returns the fields that were filled. Must run while the other copies are
    still where their tracks say they are - art is read from the files.
    """
    others = list(others)
    current = keeper.current
    filled = TrackTags.from_dict(current.to_dict())
    added: list[str] = []
    for name in TAG_FIELDS:
        if name == "compilation" or getattr(current, name) not in (None, ""):
            continue
        if name.startswith("mb_") and not cfg.write_musicbrainz_ids:
            continue
        for other in others:
            value = getattr(other.current, name)
            if value not in (None, ""):
                setattr(filled, name, value)
                added.append(name)
                break

    art: Optional[tuple[bytes, str]] = None
    if not current.has_art and cfg.write_cover_art:
        for other in others:
            if other.current.has_art:
                art = read_embedded_art(Path(other.path))
                if art:
                    break
    if not added and not art:
        return []

    # ``year`` is derived from ``date`` by every writer, so it is not a tag
    # undo could clear on its own - same rule as Apply (see _written_fields).
    written = [name for name in added if name != "year"] + (["has_art"] if art else [])
    journal.record(batch_id, "tags", keeper.path, prev_tags=current, written=written)
    write_file(
        Path(keeper.path), filled,
        art=art[0] if art else None,
        art_mime=art[1] if art else "image/jpeg",
        id3v2_version=cfg.id3v2_version,
        write_mb_ids=cfg.write_musicbrainz_ids,
    )
    if art:
        filled.has_art = True
        added.append("has_art")
    keeper.current = filled
    return added


# ---------------------------------------------------------------------------
# Merging reviewed groups (the "Find duplicates" button)
# ---------------------------------------------------------------------------

@dataclass
class MergeReport:
    batch_id: str = ""
    groups: int = 0               # groups that had at least one copy set aside
    merged: int = 0               # copies moved to the Duplicates folder
    tags_filled: int = 0          # kept copies that gained a tag or cover art
    failed: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    #: Where each set-aside copy used to be - not serialised; the server
    #: drops exactly these tracks from its state.
    retired: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "groups": self.groups,
            "merged": self.merged,
            "tags_filled": self.tags_filled,
            "failed": self.failed,
            "errors": self.errors,
            "folders": self.folders,
        }


def merge_groups(groups: list[tuple[Track, list[Track]]], cfg: Config,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None) -> MergeReport:
    """For each ``(keep, others)``: enrich ``keep``, then set ``others`` aside.

    ``cancelled`` is checked between groups, never partway through one.
    """
    journal = get_journal()
    report = MergeReport()
    report.batch_id = journal.start_batch(f"Merge {len(groups)} set(s) of duplicates")

    for index, (keeper, others) in enumerate(groups):
        if cancelled and cancelled():
            break
        try:
            if not Path(keeper.path).exists():
                # Without the copy to keep, setting the others aside would
                # leave the track nowhere in the library at all.
                raise FileNotFoundError("The copy to keep no longer exists")
            present = [o for o in others if Path(o.path).exists()]
            try:
                if absorb(keeper, present, cfg, journal, report.batch_id):
                    report.tags_filled += 1
            except Exception as exc:  # noqa: BLE001 - filling tags is a bonus, not the job
                log.warning("Could not copy tags onto %s: %s", keeper.path, exc)
            moved = 0
            for other in present:
                try:
                    dest = retire(Path(other.path), cfg, journal, report.batch_id)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Could not set aside %s", other.path)
                    report.failed += 1
                    report.errors.append({"path": other.path, "error": str(exc)})
                    continue
                report.retired.append(other.path)
                folder = str(duplicates_folder(dest, cfg))
                if folder not in report.folders:
                    report.folders.append(folder)
                moved += 1
            report.merged += moved
            if moved:
                report.groups += 1
        except Exception as exc:  # noqa: BLE001 - one bad group must not abort the batch
            log.exception("Merge failed for %s", keeper.path)
            report.failed += 1
            report.errors.append({"path": keeper.path, "error": str(exc)})
        if progress:
            progress(index + 1, len(groups), Path(keeper.path).name)

    journal.finish_batch(report.batch_id, report.to_dict())
    return report
