"""Planning the on-disk layout Plex expects.

Plex uses *both* mechanisms, and it is worth being precise about which does what:

* **Embedded tags decide what the library looks like.** Album, album artist,
  track number, disc number and year all come from the tags. Plex's music
  agents match against those, so correct tags are non-negotiable.
* **Folder structure decides what Plex can find and how it groups files it
  cannot match.** The documented layout is ``Artist/Album/track files``. When
  tags are missing or a match fails, Plex falls back to the folder names, so a
  clean tree is the safety net rather than the primary mechanism.

The single most common cause of a mangled Plex music library is a missing or
inconsistent **album artist**, which splits one album into one entry per track
artist. :mod:`musictag.matching` always fills that field, and the default
templates below put it at the top of the tree.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .config import Config
from .models import Track, TrackTags
from .util import sanitize_component

#: Non-audio files worth carrying along when a track moves.
COMPANION_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".cue", ".log", ".m3u", ".m3u8", ".nfo", ".txt"}
COMPANION_STEMS = {"cover", "folder", "front", "album", "albumart", "thumb"}

log = logging.getLogger(__name__)


class _Blank:
    """Renders as an empty string in any format spec, so templates tolerate gaps."""

    def __format__(self, spec: str) -> str:
        return ""

    def __str__(self) -> str:
        return ""

    def __bool__(self) -> bool:
        return False


class _SafeDict(dict):
    """Every documented placeholder is pre-populated, so a missing key is a typo.

    Raising here (rather than silently rendering nothing) is what lets
    :func:`render` notice a bad custom template and fall back, instead of
    quietly filing everything under an empty folder name.
    """

    def __missing__(self, key):
        raise KeyError(key)


def _tidy(text: str) -> str:
    """Clean up separators left behind by fields that rendered empty."""
    text = re.sub(r"\s*-\s*-\s*", " - ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip(" -_.")
    return text


def template_values(tags: TrackTags, track: Optional[Track] = None) -> _SafeDict:
    disc_total = tags.disc_total or 0
    disc_no = tags.disc_no or 0
    # Prefix the disc when there genuinely is more than one - and also when the
    # total is unknown but we are past disc 1, because "disc 2 track 1" and
    # "disc 1 track 1" would otherwise both be filed as "01 - ..." and collide.
    multi_disc = disc_total > 1 or (not disc_total and disc_no > 1)
    disc_prefix = f"{disc_no}-" if (multi_disc and disc_no) else ""

    # Values are sanitised *before* substitution so that a slash inside a tag
    # ("AC/DC") becomes a hyphen rather than an extra directory level. Only the
    # slashes written in the template itself separate path components.
    def clean(value: Optional[str], fallback: str) -> str:
        return sanitize_component(value) if value else fallback

    values = _SafeDict(
        album_artist=clean(tags.album_artist or tags.artist, "Unknown Artist"),
        artist=clean(tags.artist or tags.album_artist, "Unknown Artist"),
        album=clean(tags.album, "Unknown Album"),
        title=clean(tags.title, Path(track.path).stem if track else "Unknown Title"),
        track=tags.track_no if tags.track_no else _Blank(),
        track_total=tags.track_total if tags.track_total else _Blank(),
        disc=disc_no if disc_no else _Blank(),
        disc_total=disc_total if disc_total else _Blank(),
        disc_prefix=disc_prefix,
        year=tags.year if tags.year else _Blank(),
        year_suffix=f" ({tags.year})" if tags.year else "",
        genre=sanitize_component(tags.genre) if tags.genre else "",
    )
    return values


def render(template: str, tags: TrackTags, track: Optional[Track] = None) -> list[str]:
    """Render a ``/``-separated template into sanitised path components."""
    values = template_values(tags, track)
    try:
        rendered = template.format_map(values)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        # A bad custom template must not take the whole run down, and it must
        # not quietly produce nonsense paths either.
        log.warning("Template %r is invalid (%s); using the default instead.", template, exc)
        rendered = "{album_artist}/{album}".format_map(values)
    parts = [p for p in (_tidy(part) for part in rendered.split("/")) if p]
    return [sanitize_component(p) for p in parts]


def plan_path(track: Track, tags: TrackTags, cfg: Config) -> Path:
    """Where this track should live once organised."""
    source = Path(track.path)
    root = Path(cfg.organize_root) if cfg.organize_root else _infer_root(source, cfg)

    folder_parts = render(cfg.folder_template, tags, track)
    name_parts = render(cfg.file_template, tags, track)
    filename = (name_parts[-1] if name_parts else source.stem) + source.suffix.lower()
    return root.joinpath(*folder_parts, filename)


def _infer_root(source: Path, cfg: Config) -> Path:
    """Without an explicit target, reorganise inside the configured library root."""
    for lib in cfg.library_paths:
        lib_path = Path(lib)
        try:
            source.relative_to(lib_path)
            return lib_path
        except ValueError:
            continue
    # Fall back to two levels up, i.e. treat .../Artist/Album/track.mp3 as the shape.
    return source.parent.parent if len(source.parents) >= 2 else source.parent


def plan_all(tracks: list[Track], cfg: Config) -> dict[str, str]:
    """Compute planned destinations for every track, avoiding collisions.

    Two different tracks resolving to the same path is a real signal (usually
    duplicate rips), so we disambiguate rather than silently overwrite.
    """
    planned: dict[str, str] = {}
    taken: set[str] = set()
    for track in tracks:
        tags = track.match.proposed if track.match else track.current
        if not tags or not (tags.title or tags.album):
            continue
        dest = plan_path(track, tags, cfg)
        key = str(dest).lower()
        if key in taken:
            stem, suffix = dest.stem, dest.suffix
            for n in range(2, 100):
                candidate = dest.with_name(f"{stem} ({n}){suffix}")
                if str(candidate).lower() not in taken:
                    dest = candidate
                    break
        taken.add(str(dest).lower())
        planned[track.path] = str(dest)
        track.planned_path = str(dest)
    return planned


def companion_files(album_dir: Path) -> list[Path]:
    """Artwork and sidecar files worth moving with an album."""
    out: list[Path] = []
    try:
        entries = list(album_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in COMPANION_SUFFIXES:
            continue
        if suffix in {".jpg", ".jpeg", ".png", ".webp"} and entry.stem.lower() not in COMPANION_STEMS:
            # Only carry recognised artwork names, not every stray screenshot.
            continue
        out.append(entry)
    return out
