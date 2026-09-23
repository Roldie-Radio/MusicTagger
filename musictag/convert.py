"""Converting tracks to another format with the bundled ffmpeg.

The converted file is written *beside* the original, never over it: the
original is the better copy (a lossless source, or at least not re-encoded),
and deleting it is a decision for the user, not a side effect of converting.

Tags and cover art are not left to ffmpeg's metadata mapping, which names
fields differently per container and drops some entirely. The audio is
encoded with metadata stripped, then :func:`musictag.tags.write_file` writes
the original's tags and art - the same writer Apply uses, so a converted file
is tagged exactly the way Plex expects.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import Config
from .journal import get_journal
from .models import TrackTags
from .tags import read_embedded_art, read_file, write_file
from .util import unique_path

log = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: What the app converts to. ``label`` is what the UI shows.
FORMATS: dict[str, dict[str, Any]] = {
    "mp3": {
        "label": "MP3 320 kbps",
        "ext": ".mp3",
        "args": ["-c:a", "libmp3lame", "-b:a", "320k"],
    },
    "m4a": {
        # 256 kbps AAC is what the iTunes Store sells: transparent for
        # almost everyone, and what most players expect an .m4a to be.
        "label": "M4A (AAC 256 kbps)",
        "ext": ".m4a",
        "args": ["-c:a", "aac", "-b:a", "256k"],
    },
    "wma": {
        "label": "WMA (192 kbps)",
        "ext": ".wma",
        "args": ["-c:a", "wmav2", "-b:a", "192k"],
    },
}

#: One file should never take this long; a hung ffmpeg must not hang the job.
CONVERT_TIMEOUT_S = 15 * 60


class ConvertError(Exception):
    pass


@dataclass
class ConvertReport:
    batch_id: str = ""
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped_reasons: list[dict[str, str]] = field(default_factory=list)
    #: New files, for the server to add to the track list - not serialised.
    created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "converted": self.converted,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors[:100],
            "skipped_reasons": self.skipped_reasons[:100],
        }


def _same_format(source: Path, fmt: str) -> bool:
    ext = source.suffix.lower()
    if fmt == "m4a":
        return ext in (".m4a", ".mp4", ".m4b")
    return ext == FORMATS[fmt]["ext"]


def convert_file(source: Path, fmt: str, cfg: Config) -> Path:
    """Convert one file and return the path of the new one."""
    spec = FORMATS[fmt]
    ffmpeg = cfg.ffmpeg
    if not ffmpeg:
        raise ConvertError("ffmpeg was not found. Install it, or set its path in Settings.")

    final = unique_path(source.with_suffix(spec["ext"]))
    # Encode under a temporary name so a half-written file - a crash, a
    # timeout, a full disk - never sits in the library looking real.
    temp = final.with_name(f"{final.stem}.converting{spec['ext']}")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(source),
           "-map", "0:a:0", "-vn", "-map_metadata", "-1", *spec["args"], str(temp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=CONVERT_TIMEOUT_S,
                              creationflags=_NO_WINDOW)
        if proc.returncode != 0:
            message = proc.stderr.decode("utf-8", "ignore").strip()[-300:]
            raise ConvertError(message or "ffmpeg could not convert this file")

        try:
            tags, _props = read_file(source)
        except Exception:  # noqa: BLE001 - an untagged source still converts
            tags = TrackTags()
        art = read_embedded_art(source)
        write_file(temp, tags,
                   art=art[0] if art else None,
                   art_mime=art[1] if art else "image/jpeg",
                   id3v2_version=cfg.id3v2_version,
                   write_mb_ids=cfg.write_musicbrainz_ids)
        os.replace(temp, final)
    except subprocess.TimeoutExpired as exc:
        raise ConvertError("ffmpeg took too long and was stopped") from exc
    except OSError as exc:
        raise ConvertError(f"Could not run ffmpeg: {exc}") from exc
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                log.debug("Could not remove %s", temp)
    return final


def convert_tracks(paths: Iterable[str], fmt: str, cfg: Config, *,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   cancelled: Optional[Callable[[], bool]] = None) -> ConvertReport:
    """Convert every file in ``paths``, stopping between files if cancelled."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}")
    paths = list(paths)
    journal = get_journal()
    report = ConvertReport()
    report.batch_id = journal.start_batch(
        f"Convert {len(paths)} track(s) to {FORMATS[fmt]['label']}")

    for index, path in enumerate(paths):
        if cancelled and cancelled():
            break
        source = Path(path)
        if not source.exists():
            report.failed += 1
            report.errors.append({"path": path, "error": "File no longer exists"})
        elif _same_format(source, fmt):
            # Re-encoding a lossy file into the same format only loses
            # quality; there is nothing a higher bitrate can bring back.
            report.skipped += 1
            report.skipped_reasons.append(
                {"path": path, "reason": f"Already {fmt.upper()}"})
        else:
            try:
                created = convert_file(source, fmt, cfg)
                journal.record(report.batch_id, "convert", str(source), dest=str(created))
                report.converted += 1
                report.created.append(str(created))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
                log.warning("Convert failed for %s: %s", path, exc)
                report.failed += 1
                report.errors.append({"path": path, "error": str(exc)})
        if progress:
            progress(index + 1, len(paths), source.name)

    journal.finish_batch(report.batch_id, report.to_dict())
    return report
