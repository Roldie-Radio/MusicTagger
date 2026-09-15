"""In-memory library state, mirrored to sqlite so a restart does not lose work."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from .cache import state_store
from .config import Config, get_config
from .models import Track


#: Tag fields the table can be sorted by. Sorting uses the *effective* value -
#: what the file would end up with - so the column you see is the column you sort.
TAG_SORT_FIELDS = (
    "title", "artist", "album", "album_artist", "track_no", "track_total",
    "disc_no", "disc_total", "year", "date", "genre", "composer", "isrc",
)

#: Everything else that can be sorted, derived from the file rather than its tags.
FILE_SORT_FIELDS = {
    "path": lambda t: t.path.lower(),
    "filename": lambda t: (t.filename or "").lower(),
    "folder": lambda t: str(Path(t.path).parent).lower(),
    "format": lambda t: t.props.container or "",
    "duration": lambda t: t.props.duration_s or None,
    "bitrate": lambda t: t.props.bitrate_kbps or None,
    "size": lambda t: t.props.filesize or None,
    "confidence": lambda t: t.match.confidence if t.match else None,
    "quality": lambda t: (t.quality.score
                          if t.quality and t.quality.analysed else None),
}


def effective_tag(track: Track, field: str):
    """What this field would become: the proposal if there is one, else what is there.

    The grid shows one value per cell, so it has to agree with what an Apply
    would actually write.
    """
    if track.match and track.match.proposed:
        value = getattr(track.match.proposed, field, None)
        if value not in (None, "", False):
            return value
    return getattr(track.current, field, None)


def confidence_bucket(track: Track, cfg: Config) -> str:
    """Which review bucket a track falls into."""
    if track.match is None:
        return "unidentified"
    conf = track.match.confidence
    if conf >= cfg.auto_apply_threshold:
        return "high"
    if conf >= cfg.review_threshold:
        return "review"
    return "low"


class AppState:
    """Everything the UI reads, guarded by one lock."""

    def __init__(self):
        self._lock = threading.RLock()
        self._tracks: dict[str, Track] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Restore the previous session's scan from sqlite."""
        with self._lock:
            if self._loaded:
                return
            store = state_store()
            for key in store.keys():
                data = store.get(key)
                if not data:
                    continue
                try:
                    track = Track.from_dict(data)
                except Exception:
                    continue
                self._tracks[track.path] = track
            self._loaded = True

    def persist(self, tracks: Optional[Iterable[Track]] = None) -> None:
        """Write tracks to the state store.

        Always serialises the object this state actually holds, not the one
        passed in. A re-scan builds fresh :class:`Track` objects with no match
        or quality results; :meth:`add` keeps the existing object when the file
        is unchanged, so writing the caller's copy would silently erase
        everything already learned about the file. It looked fine until the
        next restart, which is the worst way for a bug like this to behave.
        """
        store = state_store()
        with self._lock:
            if tracks is None:
                targets = list(self._tracks.values())
            else:
                targets = [self._tracks.get(t.path, t) for t in tracks]
        for track in targets:
            data = track.to_dict()
            data["fingerprint"] = track.fingerprint     # kept out of the UI payload
            store.set(track.path, data)

    # ------------------------------------------------------------------
    def add(self, tracks: Iterable[Track]) -> int:
        """Merge scan results in, keeping what we already know where it still holds.

        An unchanged file keeps its match and quality results; a file whose
        mtime moved is treated as new, because any previous analysis of it is
        now describing different bytes.
        """
        added = 0
        with self._lock:
            for track in tracks:
                existing = self._tracks.get(track.path)
                if existing is not None and existing.mtime == track.mtime:
                    existing.current = track.current
                    existing.props = track.props
                    continue
                self._tracks[track.path] = track
                added += 1
        return added

    def get(self, path: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(path)

    def all(self) -> list[Track]:
        with self._lock:
            return list(self._tracks.values())

    def replace_path(self, old_path: str, track: Track) -> None:
        """After an organise move, the track lives at a new key."""
        with self._lock:
            if old_path in self._tracks and old_path != track.path:
                del self._tracks[old_path]
                state_store().delete(old_path)
            self._tracks[track.path] = track

    def remove_missing(self) -> int:
        with self._lock:
            gone = [p for p in self._tracks if not Path(p).exists()]
            for path in gone:
                del self._tracks[path]
                state_store().delete(path)
        return len(gone)

    def remove(self, paths: Iterable[str]) -> int:
        """Drop tracks entirely - for files that just left the ingest folder.

        Unlike :meth:`replace_path`, nothing takes their place: once a track
        is exported to Plex it is no longer this app's to manage, the same
        way :meth:`remove_missing` already forgets a file that vanished out
        from under it.
        """
        removed = 0
        with self._lock:
            for path in paths:
                if path in self._tracks:
                    del self._tracks[path]
                    state_store().delete(path)
                    removed += 1
        return removed

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()
        state_store().clear()

    # ------------------------------------------------------------------
    def select(self, paths: Optional[list[str]] = None, *,
               only_unidentified: bool = False,
               only_unanalysed: bool = False) -> list[Track]:
        with self._lock:
            if paths:
                wanted = set(paths)
                tracks = [t for t in self._tracks.values() if t.path in wanted]
            else:
                tracks = list(self._tracks.values())
        if only_unidentified:
            tracks = [t for t in tracks if t.match is None]
        if only_unanalysed:
            tracks = [t for t in tracks if t.quality is None or not t.quality.analysed]
        return tracks

    def sort_value(self, track: Track, sort: str):
        """The value a given sort column compares on."""
        if sort in TAG_SORT_FIELDS:
            value = effective_tag(track, sort)
            return value.lower() if isinstance(value, str) else value
        getter = FILE_SORT_FIELDS.get(sort)
        return getter(track) if getter else track.path.lower()

    def filtered(self, *, query: str = "", bucket: str = "", status: str = "",
                 issues: str = "", fmt: str = "", sort: str = "path",
                 desc: bool = False, offset: int = 0, limit: int = 200) -> dict[str, Any]:
        cfg = get_config()
        tracks = self.all()

        if query:
            q = query.lower()
            def matches(t: Track) -> bool:
                haystack = " ".join(filter(None, [
                    t.path, t.current.title, t.current.artist, t.current.album,
                    t.match.proposed.title if t.match else None,
                    t.match.proposed.artist if t.match else None,
                    t.match.proposed.album if t.match else None,
                ])).lower()
                return q in haystack
            tracks = [t for t in tracks if matches(t)]

        if bucket:
            tracks = [t for t in tracks if confidence_bucket(t, cfg) == bucket]
        if status:
            tracks = [t for t in tracks if t.status == status]
        if fmt:
            tracks = [t for t in tracks if t.props.container == fmt]
        if issues == "any":
            tracks = [t for t in tracks if t.quality and t.quality.issues]
        elif issues in ("high", "medium", "low"):
            tracks = [t for t in tracks
                      if t.quality and any(i.severity == issues for i in t.quality.issues)]

        # "confidence_desc" predates the `desc` flag; keep it working.
        if sort.endswith("_desc"):
            sort, desc = sort[: -len("_desc")], True

        # Rows with nothing in the sorted column always sink to the bottom,
        # in both directions - an empty cell is not a small value, it is a gap,
        # and gaps at the top would bury the rows you can actually act on.
        present, missing = [], []
        for track in tracks:
            (missing if self.sort_value(track, sort) in (None, "") else present).append(track)
        try:
            present.sort(key=lambda t: self.sort_value(t, sort), reverse=desc)
        except TypeError:
            # Mixed types in one column: fall back to comparing as text.
            present.sort(key=lambda t: str(self.sort_value(t, sort)), reverse=desc)
        missing.sort(key=lambda t: t.path.lower())
        tracks = present + missing

        total = len(tracks)
        page = tracks[offset: offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "tracks": [self._track_payload(t, cfg) for t in page],
        }

    def _track_payload(self, track: Track, cfg: Config) -> dict[str, Any]:
        payload = track.to_dict()
        payload["bucket"] = confidence_bucket(track, cfg)
        return payload

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        cfg = get_config()
        tracks = self.all()
        buckets = {"high": 0, "review": 0, "low": 0, "unidentified": 0}
        formats: dict[str, int] = {}
        issue_counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
        analysed = 0
        applied = 0
        errors = 0
        quality_total = 0

        for track in tracks:
            buckets[confidence_bucket(track, cfg)] += 1
            formats[track.props.container or "?"] = formats.get(track.props.container or "?", 0) + 1
            if track.status == "applied":
                applied += 1
            if track.status == "error":
                errors += 1
            if track.quality and track.quality.analysed:
                analysed += 1
                quality_total += track.quality.score
                for issue in track.quality.issues:
                    issue_counts[issue.severity] = issue_counts.get(issue.severity, 0) + 1

        return {
            "total": len(tracks),
            "buckets": buckets,
            "formats": formats,
            "analysed": analysed,
            "applied": applied,
            "errors": errors,
            "issues": issue_counts,
            "average_quality": round(quality_total / analysed, 1) if analysed else None,
            "thresholds": {
                "auto_apply": cfg.auto_apply_threshold,
                "review": cfg.review_threshold,
            },
        }


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
        _state.load()
    return _state
