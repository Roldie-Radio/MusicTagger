"""Moving reviewed tracks from an ingest folder into a Plex Music library.

Two phases:

* :func:`plan_export` - read what's already in the Plex folder, compute
  where each incoming track would land (reusing the same folder/file
  templates as the general "reorganise into Plex folders" apply-time
  option, see :mod:`musictag.organize`), and flag anything that looks like
  it is already there. Read-only; nothing moves yet.
* :func:`commit_export` - given the plan plus how the user resolved every
  duplicate, actually move the files. Every move is journalled exactly like
  a normal Apply, so Export shows up in Undo history and can be reversed.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config
from .duplicates import DuplicateMatch, build_index, find_duplicate
from .journal import get_journal
from .models import TAG_FIELDS, Track
from .organize import plan_all
from .util import unique_path

log = logging.getLogger(__name__)


def has_pending_changes(track: Track) -> bool:
    """Would applying this track's proposal actually change a tag on disk?

    Export moves the file as-is; it does not write tags. Exporting a track
    with an unapplied proposal would file it under a Plex folder name (and
    leave it holding embedded tags) that disagree with each other, so those
    tracks are held back until Apply has actually run - matching the
    "tagged and reviewed, then exported" workflow this feature is for.
    """
    if not track.match or not track.match.proposed:
        return False
    proposed, current = track.match.proposed, track.current
    for name in TAG_FIELDS:
        after = getattr(proposed, name, None)
        if after in (None, "", False):
            continue          # not touched - every writer skips a null field
        before = getattr(current, name, None) if current else None
        if str(before or "") != str(after or ""):
            return True
    return False

#: Where a replaced file goes instead of being deleted outright - Journal's
#: own "nothing here deletes user data" rule (see journal.py) applies here too.
TRASH_DIRNAME = ".musictagger-trash"


@dataclass
class ExportItem:
    path: str
    dest: str
    duplicate: Optional[DuplicateMatch] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "dest": self.dest,
            "duplicate": self.duplicate.to_dict() if self.duplicate else None,
        }


@dataclass
class ExportPlan:
    plex_root: str
    items: list[ExportItem] = field(default_factory=list)
    ineligible: int = 0          # selected but had no usable title/album at all
    pending_apply: list[str] = field(default_factory=list)  # has unapplied tag changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "plex_root": self.plex_root,
            "items": [i.to_dict() for i in self.items],
            "ineligible": self.ineligible,
            "pending_apply": self.pending_apply,
            "duplicate_count": sum(1 for i in self.items if i.duplicate),
        }


def plan_export(tracks: list[Track], cfg: Config,
                 progress: Optional[Callable[[int, int, str], None]] = None) -> ExportPlan:
    """Compute an export plan. Read-only - nothing on disk changes."""
    if not cfg.organize_root:
        raise ValueError("No Plex Music folder configured. Set one in Settings first.")
    plex_root = Path(cfg.organize_root)
    plex_root.mkdir(parents=True, exist_ok=True)

    plan = ExportPlan(plex_root=str(plex_root))
    plan.pending_apply = [t.path for t in tracks if has_pending_changes(t)]
    pending = set(plan.pending_apply)
    ready = [t for t in tracks if t.path not in pending]

    destinations = plan_all(ready, cfg)
    plan.ineligible = len(ready) - len(destinations)

    index = build_index(plex_root, progress=progress)
    for track in ready:
        dest = destinations.get(track.path)
        if not dest:
            continue
        dup = find_duplicate(track.current, index)
        plan.items.append(ExportItem(path=track.path, dest=dest, duplicate=dup))
    return plan


@dataclass
class ExportReport:
    batch_id: str = ""
    exported: int = 0
    replaced: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "exported": self.exported,
            "replaced": self.replaced,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors,
        }


def commit_export(items: list[dict[str, Any]], cfg: Config,
                   progress: Optional[Callable[[int, int, str], None]] = None) -> ExportReport:
    """Move files according to ``items``.

    Each item is ``{"path", "dest", "action", "existing_path"?}`` where
    ``action`` is one of:

    * ``"export"``   - not a duplicate (or the user chose to keep both
                       copies); move to ``dest``, disambiguating the name if
                       something unrelated is already there.
    * ``"replace"``  - move whatever is at ``existing_path`` into a trash
                       folder inside the Plex root first (never deleted
                       outright), then move the incoming file to ``dest``.
    * ``"skip"``     - leave the incoming file where it is; nothing moves.
    """
    journal = get_journal()
    report = ExportReport()
    report.batch_id = journal.start_batch(f"Export {len(items)} track(s) to Plex")
    plex_root = Path(cfg.organize_root) if cfg.organize_root else None

    for index, item in enumerate(items):
        path, dest, action = item["path"], item["dest"], item.get("action", "export")
        try:
            source = Path(path)
            if action == "skip" or not source.exists():
                report.skipped += 1
                continue

            dest_path = Path(dest)
            if action == "replace" and item.get("existing_path"):
                existing = Path(item["existing_path"])
                if existing.exists():
                    trash = unique_path((plex_root or dest_path.parent) / TRASH_DIRNAME / existing.name)
                    trash.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(existing), str(trash))
                    journal.record(report.batch_id, "move", str(existing), dest=str(trash))
                    report.replaced += 1
            else:
                dest_path = unique_path(dest_path)

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest_path))
            journal.record(report.batch_id, "move", str(source), dest=str(dest_path))
            report.exported += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            log.exception("Export failed for %s", path)
            report.failed += 1
            report.errors.append({"path": path, "error": str(exc)})
        if progress:
            progress(index + 1, len(items), Path(path).name)

    journal.finish_batch(report.batch_id, report.to_dict())
    return report
