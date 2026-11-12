"""Renaming files where they are, from what each song is.

The "Rename files" button. Unlike organising (:mod:`musictag.organize`),
nothing changes folder: every file keeps its place and only its name is
rebuilt from the filename template - ``07 - Paranoid Android.mp3`` rather
than ``track07_final(1).mp3``.

Two phases, like export: :func:`plan_renames` works out every new name
without touching anything, and :func:`commit_renames` carries a plan out.

When two files in one folder come out with the same name they are either
two copies of one track - the better one takes the name and the other is
set aside in the Duplicates folder (:mod:`musictag.merge`) - or two different
songs that happen to share a title, which both stay, as ``name`` and
``name (2)``.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config
from .duplicates import identity_tags, rank_key, same_recording
from .journal import get_journal
from .library import load_track
from .merge import absorb, duplicates_folder, retire, retire_destination
from .models import Track, TrackTags
from .organize import render
from .util import unique_path

log = logging.getLogger(__name__)


def naming_tags(track: Track) -> TrackTags:
    """The tags a file is named from: its proposal if it has one, else its own.

    The same choice :func:`musictag.organize.plan_all` makes, so Rename and
    organise always agree on what a file should be called.
    """
    if track.match and track.match.proposed and not track.match.proposed.is_empty():
        return track.match.proposed
    return track.current


def new_name(track: Track, cfg: Config) -> Optional[str]:
    """The filename the template gives this track, or ``None`` without a title.

    A file with no title has nothing to be named after; the template would
    fall back to the current filename and produce ``07 - 07 - old name``.
    """
    tags = naming_tags(track)
    if not tags.title:
        return None
    source = Path(track.path)
    parts = render(cfg.file_template, tags, track)
    if not parts:
        return None
    return parts[-1] + source.suffix.lower()


def _key(path: Path | str) -> str:
    return os.path.normcase(str(path))


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


@dataclass
class RenameItem:
    track: Track
    #: rename    - gets a new name in its folder
    #: duplicate - a copy of ``keeper``; set aside in the Duplicates folder
    #: unchanged - already has the right name
    #: skip      - no title to name it from
    action: str
    dest: Optional[Path] = None
    keeper: Optional[Track] = None
    note: str = ""
    #: The path the track had when the plan was made.
    path: str = ""

    def __post_init__(self):
        self.path = self.path or self.track.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action,
            "dest": str(self.dest) if self.dest else None,
            "keeper": self.keeper.path if self.keeper else None,
            "note": self.note,
        }


@dataclass
class RenamePlan:
    items: list[RenameItem] = field(default_factory=list)

    def of(self, action: str) -> list[RenameItem]:
        return [i for i in self.items if i.action == action]

    def to_dict(self, limit: int = 500) -> dict[str, Any]:
        renames, duplicates = self.of("rename"), self.of("duplicate")
        return {
            "rename_count": len(renames),
            "duplicate_count": len(duplicates),
            "unchanged": len(self.of("unchanged")),
            "skipped": len(self.of("skip")),
            # A big library renames tens of thousands of files; the review
            # shows the first few hundred and the counts cover the rest.
            "renames": [i.to_dict() for i in renames[:limit]],
            "duplicates": [i.to_dict() for i in duplicates[:limit]],
        }


def plan_renames(tracks: list[Track], cfg: Config,
                 lookup: Optional[Callable[[str], Optional[Track]]] = None) -> RenamePlan:
    """Work out the new name of every track. Read-only.

    ``lookup`` returns the :class:`Track` already known for a path, if any -
    used when a wanted name belongs to a file outside this batch, so what
    is already known about that file (its quality report) is not thrown away.
    """
    plan = RenamePlan()
    batch = {_key(t.path): t for t in tracks}
    wanted: dict[str, list[Track]] = {}     # folder+name -> everyone who wants it
    targets: dict[str, Path] = {}

    for track in tracks:
        source = Path(track.path)
        name = new_name(track, cfg)
        if name is None:
            plan.items.append(RenameItem(track, "skip", note="No title to name it from"))
        elif name == source.name:
            plan.items.append(RenameItem(track, "unchanged"))
        else:
            dest = source.with_name(name)
            wanted.setdefault(_key(dest), []).append(track)
            targets.setdefault(_key(dest), dest)

    movers = {_key(t.path) for group in wanted.values() for t in group}
    taken = set(targets)                    # every name this plan hands out

    def free_name(dest: Path) -> Path:
        stem, suffix = dest.stem, dest.suffix
        for n in range(2, 1000):
            candidate = dest.with_name(f"{stem} ({n}){suffix}")
            if _key(candidate) not in taken and not candidate.exists():
                taken.add(_key(candidate))
                return candidate
        raise FileExistsError(f"Could not find a free name near {dest}")

    def same(a: Track, b: Track) -> bool:
        if not cfg.merge_duplicates:
            return False
        verdict = same_recording(identity_tags(a), identity_tags(b),
                                 a_duration=a.props.duration_s,
                                 b_duration=b.props.duration_s)
        # Same song on two different albums: each belongs where it is.
        return bool(verdict and verdict.same_album)

    for key, group in wanted.items():
        dest = targets[key]
        # Who has this name right now, and is staying put? A file that is
        # itself being renamed away frees the name; one that is only having
        # its capitalisation changed *is* the file at ``dest``.
        owner: Optional[Track] = None
        if dest.exists() and key not in movers \
                and not any(_same_file(dest, Path(t.path)) for t in group):
            owner = batch.get(key) or (lookup(str(dest)) if lookup else None) \
                or load_track(dest)

        # Best copy first, so it is the one that gets the plain name.
        remaining = sorted(sorted(group, key=lambda t: t.path), key=rank_key, reverse=True)
        first = True
        while remaining:
            if first and owner is not None:
                copies = [t for t in remaining if same(t, owner)]
                if copies and rank_key(copies[0]) > rank_key(owner):
                    # A better copy arrives: it takes the name, and the file
                    # that had it is the one set aside.
                    head = copies.pop(0)
                    plan.items.append(RenameItem(head, "rename", dest=dest))
                    plan.items.append(RenameItem(
                        owner, "duplicate", keeper=head,
                        dest=retire_destination(Path(owner.path), cfg)))
                else:
                    head = owner
            else:
                head = remaining[0]
                plan.items.append(RenameItem(
                    head, "rename", dest=dest if first else free_name(dest),
                    note="" if first else "Another song already has this name"))
                copies = [t for t in remaining[1:] if same(t, head)]
            for copy in copies:
                plan.items.append(RenameItem(
                    copy, "duplicate", keeper=head,
                    dest=retire_destination(Path(copy.path), cfg)))
            settled = {id(head)} | {id(c) for c in copies}
            remaining = [t for t in remaining if id(t) not in settled]
            first = False

    # A file that was going to stay as it is, but turned out to be the worse
    # copy of something taking its name, is listed once - as the duplicate.
    set_aside = {id(i.track) for i in plan.items if i.action == "duplicate"}
    plan.items = [i for i in plan.items
                  if i.action == "duplicate" or id(i.track) not in set_aside]

    order = {id(t): n for n, t in enumerate(tracks)}
    plan.items.sort(key=lambda i: order.get(id(i.track), len(order)))
    return plan


@dataclass
class RenameReport:
    batch_id: str = ""
    renamed: int = 0
    merged: int = 0               # duplicate copies set aside
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)      # Duplicates folders used
    #: old path -> track, for every file that now has a new name; and the old
    #: paths of copies that were set aside. Not serialised - the server uses
    #: them to keep its state in step with the disk.
    moved: list[tuple[str, Track]] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "renamed": self.renamed,
            "merged": self.merged,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors,
            "folders": self.folders,
        }


def commit_renames(plan: RenamePlan, cfg: Config,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   cancelled: Optional[Callable[[], bool]] = None) -> RenameReport:
    """Carry out ``plan``. Every step is journalled, so the batch can be undone.

    ``cancelled`` is checked between files; a cancelled run leaves each file
    either fully renamed (and in the journal) or untouched.
    """
    journal = get_journal()
    report = RenameReport()
    report.unchanged = len(plan.of("unchanged"))
    report.skipped = len(plan.of("skip"))
    duplicates, renames = plan.of("duplicate"), plan.of("rename")
    total = len(duplicates) + len(renames)
    if not total:
        return report
    report.batch_id = journal.start_batch(f"Rename {len(renames)} file(s)")
    done = 0

    def tick(name: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total, name)

    def failed(item: RenameItem, exc: Exception) -> None:
        log.exception("Rename failed for %s", item.path)
        report.failed += 1
        report.errors.append({"path": item.path, "error": str(exc)})

    def stop() -> bool:
        return bool(cancelled and cancelled())

    # Duplicates first: setting a copy aside is what frees the name the kept
    # one is about to take, and its tags and art have to be read before it
    # goes.
    for item in duplicates:
        if stop():
            break
        try:
            source = Path(item.track.path)
            if not source.exists() or not Path(item.keeper.path).exists():
                raise FileNotFoundError("File no longer exists")
            try:
                absorb(item.keeper, [item.track], cfg, journal, report.batch_id)
            except Exception as exc:  # noqa: BLE001 - filling tags is a bonus
                log.warning("Could not copy tags onto %s: %s", item.keeper.path, exc)
            dest = retire(source, cfg, journal, report.batch_id)
            report.merged += 1
            report.retired.append(item.path)
            folder = str(duplicates_folder(dest, cfg))
            if folder not in report.folders:
                report.folders.append(folder)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            failed(item, exc)
        tick(Path(item.path).name)

    def move(item: RenameItem, dest: Path) -> None:
        source = Path(item.track.path)
        with journal.step(report.batch_id, "move", str(source), dest=str(dest)):
            shutil.move(str(source), str(dest))
        item.track.path = str(dest)
        item.track.filename = dest.name

    def finish(item: RenameItem) -> None:
        report.renamed += 1
        report.moved.append((item.path, item.track))
        tick(item.dest.name)

    # A name can be held by a file that is itself about to be renamed
    # ("b" -> "c" has to happen before "a" -> "b"), so anything whose name is
    # still occupied waits for a later pass.
    pending = list(renames)
    while pending and not stop():
        waiting: list[RenameItem] = []
        progressed = False
        for item in pending:
            if stop():
                waiting.append(item)
                continue
            source = Path(item.track.path)
            try:
                if not source.exists():
                    raise FileNotFoundError("File no longer exists")
                if item.dest.exists() and not _same_file(source, item.dest):
                    waiting.append(item)
                    continue
                move(item, item.dest)
                finish(item)
            except Exception as exc:  # noqa: BLE001
                failed(item, exc)
                tick(Path(item.path).name)
            progressed = True
        pending = waiting
        if pending and not progressed and not stop():
            # Everything left is waiting. Either on a file that is not part
            # of this run (the name was taken since the plan was made, or
            # its owner failed to move) - then it settles for a free name -
            # or on each other: a swap, "a" <-> "b".
            held = {_key(i.track.path) for i in pending}
            item = next((i for i in pending if _key(i.dest) not in held), None)
            try:
                if item is not None:
                    item.dest = unique_path(item.dest)
                    move(item, item.dest)
                    finish(item)
                    pending.remove(item)
                else:
                    # Step one file aside to break the circle. That frees its
                    # name for whichever file wants it, and it gets its own
                    # real name on a later pass.
                    item = pending[0]
                    source = Path(item.track.path)
                    move(item, unique_path(source.with_name(
                        f"{source.stem}.renaming{source.suffix}")))
            except Exception as exc:  # noqa: BLE001
                failed(item, exc)
                tick(Path(item.path).name)
                pending.remove(item)

    # Cancelled halfway round a swap: a file stepped aside and never got its
    # final name. It still moved, so the state has to follow it.
    for item in pending:
        if item.track.path != item.path:
            report.moved.append((item.path, item.track))

    journal.finish_batch(report.batch_id, report.to_dict())
    return report
