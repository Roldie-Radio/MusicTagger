"""Plain data structures shared by every layer of the app.

Everything here is JSON-serialisable via :func:`to_dict` so the web UI and the
sqlite state store can use the exact same shapes the Python code uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Optional


# --------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------

#: Fields we read/write. Order matters: it is the display order in the UI.
TAG_FIELDS = (
    "title",
    "artist",
    "album",
    "album_artist",
    "track_no",
    "track_total",
    "disc_no",
    "disc_total",
    "date",
    "year",
    "genre",
    "composer",
    "isrc",
    "compilation",
    "mb_recording_id",
    "mb_release_id",
    "mb_release_group_id",
    "mb_artist_id",
    "mb_album_artist_id",
)


@dataclass
class TrackTags:
    """Format-agnostic view of the metadata embedded in (or proposed for) a file."""

    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    album_artist: Optional[str] = None
    track_no: Optional[int] = None
    track_total: Optional[int] = None
    disc_no: Optional[int] = None
    disc_total: Optional[int] = None
    date: Optional[str] = None          # ISO-ish: "1997", "1997-06-16"
    year: Optional[int] = None
    genre: Optional[str] = None
    composer: Optional[str] = None
    isrc: Optional[str] = None
    compilation: bool = False
    mb_recording_id: Optional[str] = None
    mb_release_id: Optional[str] = None
    mb_release_group_id: Optional[str] = None
    mb_artist_id: Optional[str] = None
    mb_album_artist_id: Optional[str] = None
    has_art: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackTags":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def is_empty(self) -> bool:
        return not any(getattr(self, f) for f in ("title", "artist", "album"))

    def merged_with(self, other: "TrackTags", only_missing: bool = False) -> "TrackTags":
        """Return a copy of ``self`` overlaid with the non-empty values of ``other``."""
        out = TrackTags(**asdict(self))
        for f in fields(TrackTags):
            new = getattr(other, f.name)
            if new in (None, "", False):
                continue
            if only_missing and getattr(out, f.name) not in (None, "", False):
                continue
            setattr(out, f.name, new)
        return out


# --------------------------------------------------------------------------
# Technical properties
# --------------------------------------------------------------------------

@dataclass
class AudioProps:
    """Container/codec facts read straight off the file."""

    container: str = ""                 # "mp3", "flac", "m4a", ...
    codec: str = ""                     # "mp3", "flac", "aac", "alac", ...
    duration_s: float = 0.0
    bitrate_kbps: int = 0
    bitrate_mode: str = ""              # "cbr", "vbr", "abr", "" (unknown)
    sample_rate: int = 0
    bit_depth: int = 0                  # 0 when not applicable (lossy)
    channels: int = 0
    filesize: int = 0
    lossless: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioProps":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

@dataclass
class Signal:
    """One piece of evidence that fed the confidence score.

    The UI shows these so a low score is explainable rather than mysterious.
    """

    name: str
    detail: str
    score: float                # 0..1, how well this signal agreed
    weight: float               # relative importance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """A possible identification of the file, with its own score."""

    source: str                                  # "acoustid", "musicbrainz-search", "existing-tags"
    confidence: float = 0.0                      # 0..100
    tags: TrackTags = field(default_factory=TrackTags)
    signals: list[Signal] = field(default_factory=list)
    release_summary: str = ""                    # "Album - Artist (1997, CD, GB)"
    cover_art_url: Optional[str] = None
    raw_id: Optional[str] = None                 # MB recording id, for dedupe
    length_s: Optional[float] = None             # MusicBrainz recording length, for duration checks
    #: True when this was only found after dropping the duration filter, i.e.
    #: the database holds no release of this length. That is evidence against
    #: the match, not a neutral absence, so it caps the confidence.
    duration_unverified: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = self.tags.to_dict()
        d["signals"] = [s.to_dict() for s in self.signals]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            source=data.get("source", ""),
            confidence=data.get("confidence", 0.0),
            tags=TrackTags.from_dict(data.get("tags", {})),
            signals=[Signal(**s) for s in data.get("signals", [])],
            release_summary=data.get("release_summary", ""),
            cover_art_url=data.get("cover_art_url"),
            raw_id=data.get("raw_id"),
            length_s=data.get("length_s"),
            duration_unverified=data.get("duration_unverified", False),
        )


@dataclass
class MatchResult:
    """Outcome of identifying one file."""

    confidence: float = 0.0                               # 0..100 for the chosen candidate
    field_confidence: dict[str, float] = field(default_factory=dict)
    proposed: TrackTags = field(default_factory=TrackTags)
    candidates: list[Candidate] = field(default_factory=list)
    chosen_index: int = 0
    method: str = ""                                      # how we got here
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "field_confidence": self.field_confidence,
            "proposed": self.proposed.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "chosen_index": self.chosen_index,
            "method": self.method,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchResult":
        return cls(
            confidence=data.get("confidence", 0.0),
            field_confidence=data.get("field_confidence", {}) or {},
            proposed=TrackTags.from_dict(data.get("proposed", {})),
            candidates=[Candidate.from_dict(c) for c in data.get("candidates", [])],
            chosen_index=data.get("chosen_index", 0),
            method=data.get("method", ""),
            notes=data.get("notes", []) or [],
        )


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}

#: Points deducted from a file's quality score per finding.
SEVERITY_PENALTY = {"high": 30, "medium": 15, "low": 6, "info": 0}

#: How each severity is described to the user. Kept here rather than in the UI
#: so the score, the badge and the help text can never drift apart.
SEVERITY_INFO = {
    "high": {
        "label": "Serious",
        "meaning": "Something is likely wrong with the file itself - a transcode, "
                   "a cut-off ending, audible distortion. Usually worth replacing.",
    },
    "medium": {
        "label": "Moderate",
        "meaning": "Noticeably below par, or a problem that may or may not bother "
                   "you. Worth listening to before deciding.",
    },
    "low": {
        "label": "Minor",
        "meaning": "A small imperfection. Most people will never hear these.",
    },
    "info": {
        "label": "Note",
        "meaning": "An observation rather than a fault - a mono recording, or a "
                   "very short track. Costs no points.",
    },
}


def quality_scale() -> dict[str, Any]:
    """Everything the UI needs to explain a quality badge, from one place."""
    return {
        "max_score": 100,
        "severities": [
            {
                "key": key,
                "label": SEVERITY_INFO[key]["label"],
                "meaning": SEVERITY_INFO[key]["meaning"],
                "penalty": SEVERITY_PENALTY[key],
            }
            # Worst first, which is the order people read them in.
            for key in sorted(SEVERITY_INFO, key=lambda k: -SEVERITY_ORDER[k])
        ],
    }


@dataclass
class QualityIssue:
    code: str                   # machine key, e.g. "low_bitrate"
    severity: str               # info | low | medium | high
    title: str                  # short human label
    detail: str                 # one sentence of explanation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    score: int = 100                                   # 0..100, 100 = no problems found
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    analysed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "analysed": self.analysed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityReport":
        return cls(
            score=data.get("score", 100),
            issues=[QualityIssue(**i) for i in data.get("issues", [])],
            metrics=data.get("metrics", {}) or {},
            analysed=data.get("analysed", False),
            error=data.get("error"),
        )

    @property
    def worst_severity(self) -> str:
        if not self.issues:
            return "info"
        return max((i.severity for i in self.issues), key=lambda s: SEVERITY_ORDER.get(s, 0))


# --------------------------------------------------------------------------
# The unit the UI works with
# --------------------------------------------------------------------------

@dataclass
class Track:
    """One audio file plus everything we know or propose about it."""

    path: str
    filename: str = ""
    mtime: float = 0.0
    props: AudioProps = field(default_factory=AudioProps)
    current: TrackTags = field(default_factory=TrackTags)
    match: Optional[MatchResult] = None
    quality: Optional[QualityReport] = None
    status: str = "scanned"             # scanned | identified | applied | error | skipped
    error: Optional[str] = None
    planned_path: Optional[str] = None  # where organise would put it
    fingerprint: Optional[str] = None   # chromaprint, cached to avoid re-decoding

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "mtime": self.mtime,
            "props": self.props.to_dict(),
            "current": self.current.to_dict(),
            "match": self.match.to_dict() if self.match else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "status": self.status,
            "error": self.error,
            "planned_path": self.planned_path,
            # fingerprints are long and the UI never shows them
            "has_fingerprint": bool(self.fingerprint),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(
            path=data["path"],
            filename=data.get("filename", ""),
            mtime=data.get("mtime", 0.0),
            props=AudioProps.from_dict(data.get("props", {})),
            current=TrackTags.from_dict(data.get("current", {})),
            match=MatchResult.from_dict(data["match"]) if data.get("match") else None,
            quality=QualityReport.from_dict(data["quality"]) if data.get("quality") else None,
            status=data.get("status", "scanned"),
            error=data.get("error"),
            planned_path=data.get("planned_path"),
            fingerprint=data.get("fingerprint"),
        )
