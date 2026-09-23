"""Committing proposed metadata to disk: tags, artwork, then organisation.

Order matters. Tags are written first so that if the move fails the file is
still improved, and the journal entry for the tag write points at a path that
still exists. Every step is recorded for undo before it happens.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import Config
from .journal import get_journal
from .models import Track, TrackTags
from .organize import companion_files, plan_path
from .providers.coverart import CoverArtClient
from .tags import read_embedded_art, write_file
from .util import unique_path

log = logging.getLogger(__name__)


@dataclass
class ApplyOptions:
    write_tags: bool = True
    write_art: bool = True
    organize: bool = False
    dry_run: bool = False
    #: Only apply tracks at or above this confidence. ``None`` means "whatever
    #: the caller selected", which is how the UI's explicit selection works.
    min_confidence: Optional[float] = None


@dataclass
class ApplyReport:
    batch_id: str = ""
    tagged: int = 0
    moved: int = 0
    copied: int = 0
    art_embedded: int = 0
    skipped: int = 0
    failed: int = 0
    dry_run: bool = False
    planned: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "tagged": self.tagged,
            "moved": self.moved,
            "copied": self.copied,
            "art_embedded": self.art_embedded,
            "skipped": self.skipped,
            "failed": self.failed,
            "dry_run": self.dry_run,
            "planned": self.planned,
            "errors": self.errors,
        }


def _written_fields(tags: TrackTags, *, art: bool, mb_ids: bool) -> list[str]:
    """Which fields a write of ``tags`` actually sets - what undo may clear.

    Mirrors the writers in :mod:`musictag.tags`: an empty field is left
    alone, the compilation flag is always written, and art only when some
    was embedded.
    """
    written = [f.name for f in fields(TrackTags)
               if f.name not in ("compilation", "has_art", "year")
               and (mb_ids or not f.name.startswith("mb_"))
               and getattr(tags, f.name) not in (None, "")]
    written.append("compilation")
    if art:
        written.append("has_art")
    return written


class Applier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.coverart = CoverArtClient(cfg)
        #: One artwork download per release, not per track.
        self._art_cache: dict[str, Optional[tuple[bytes, str]]] = {}

    # ------------------------------------------------------------------
    def apply(self, tracks: Iterable[Track], options: ApplyOptions,
              progress: Optional[Callable[[int, int, str], None]] = None) -> ApplyReport:
        tracks = list(tracks)
        report = ApplyReport(dry_run=options.dry_run)
        journal = get_journal()
        if not options.dry_run:
            report.batch_id = journal.start_batch(
                f"Apply to {len(tracks)} track(s)"
                + (" with organise" if options.organize else "")
            )

        for index, track in enumerate(tracks):
            try:
                self._apply_one(track, options, report, journal)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
                log.exception("Apply failed for %s", track.path)
                report.failed += 1
                report.errors.append({"path": track.path, "error": str(exc)})
                track.status = "error"
                track.error = str(exc)
            if progress:
                progress(index + 1, len(tracks), track.path)

        if not options.dry_run:
            journal.finish_batch(report.batch_id, report.to_dict())
            self._move_companions(tracks, report, options)
        return report

    # ------------------------------------------------------------------
    def _apply_one(self, track: Track, options: ApplyOptions,
                   report: ApplyReport, journal) -> None:
        if not track.match or not track.match.proposed:
            report.skipped += 1
            return
        proposed = track.match.proposed
        if proposed.is_empty():
            report.skipped += 1
            return
        if options.min_confidence is not None and track.match.confidence < options.min_confidence:
            report.skipped += 1
            return

        source = Path(track.path)
        if not source.exists():
            report.failed += 1
            report.errors.append({"path": track.path, "error": "File no longer exists"})
            return

        # --- artwork -----------------------------------------------------
        art: Optional[tuple[bytes, str]] = None
        if options.write_art and self.cfg.write_cover_art:
            art = self._artwork_for(track)

        # --- tags --------------------------------------------------------
        if options.write_tags:
            if options.dry_run:
                report.tagged += 1
            else:
                journal.record(report.batch_id, "tags", str(source), prev_tags=track.current,
                               written=_written_fields(proposed, art=bool(art),
                                                       mb_ids=self.cfg.write_musicbrainz_ids))
                write_file(
                    source, proposed,
                    art=art[0] if art else None,
                    art_mime=art[1] if art else "image/jpeg",
                    id3v2_version=self.cfg.id3v2_version,
                    write_mb_ids=self.cfg.write_musicbrainz_ids,
                )
                report.tagged += 1
                if art:
                    report.art_embedded += 1
                track.current = proposed
                track.status = "applied"

        # --- organise ----------------------------------------------------
        if options.organize and self.cfg.organize_enabled:
            dest = plan_path(track, proposed, self.cfg)
            track.planned_path = str(dest)
            if dest.resolve() == source.resolve():
                return
            report.planned.append({"from": str(source), "to": str(dest)})
            if options.dry_run:
                return

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest = unique_path(dest)
            if self.cfg.organize_mode == "copy":
                shutil.copy2(source, dest)
                journal.record(report.batch_id, "copy", str(source), dest=str(dest))
                report.copied += 1
            else:
                shutil.move(str(source), str(dest))
                journal.record(report.batch_id, "move", str(source), dest=str(dest))
                report.moved += 1
                track.path = str(dest)
                track.filename = dest.name

            if self.cfg.write_cover_file and art:
                self._write_cover_file(dest.parent, art, report, journal)

        elif not options.dry_run and self.cfg.write_cover_file and art:
            self._write_cover_file(source.parent, art, report, journal)

    # ------------------------------------------------------------------
    def _artwork_for(self, track: Track) -> Optional[tuple[bytes, str]]:
        """Get cover art, preferring what the file already has."""
        proposed = track.match.proposed
        key = proposed.mb_release_id or proposed.mb_release_group_id or ""

        if track.current.has_art:
            existing = read_embedded_art(Path(track.path))
            if existing:
                return existing

        if not key:
            return None
        if key in self._art_cache:
            return self._art_cache[key]
        art = self.coverart.fetch_front(
            release_id=proposed.mb_release_id,
            release_group_id=proposed.mb_release_group_id,
        )
        self._art_cache[key] = art
        return art

    def _write_cover_file(self, folder: Path, art: tuple[bytes, str],
                          report: ApplyReport, journal) -> None:
        """Drop cover.jpg beside the album - Plex's fallback when tags lack art."""
        data, mime = art
        name = self.cfg.cover_filename
        if mime == "image/png" and name.lower().endswith(".jpg"):
            name = name[:-4] + ".png"
        target = folder / name
        if target.exists():
            return
        try:
            target.write_bytes(data)
            journal.record(report.batch_id, "cover", str(folder), dest=str(target))
        except OSError as exc:
            log.debug("Could not write %s: %s", target, exc)

    def _move_companions(self, tracks: list[Track], report: ApplyReport,
                         options: ApplyOptions) -> None:
        """Carry artwork/cue/log files across when an album folder is reorganised."""
        if not (options.organize and self.cfg.organize_enabled and self.cfg.keep_extra_files):
            return
        moves: dict[Path, Path] = {}
        for entry in report.planned:
            moves.setdefault(Path(entry["from"]).parent, Path(entry["to"]).parent)

        for src_dir, dest_dir in moves.items():
            if src_dir == dest_dir or not src_dir.exists() or not dest_dir.exists():
                continue
            for companion in companion_files(src_dir):
                target = dest_dir / companion.name
                if target.exists():
                    continue
                try:
                    if self.cfg.organize_mode == "copy":
                        shutil.copy2(companion, target)
                    else:
                        shutil.move(str(companion), str(target))
                except OSError as exc:
                    log.debug("Companion file %s: %s", companion, exc)
