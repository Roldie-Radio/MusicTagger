"""Small shared helpers: string normalisation, filename safety, filename parsing."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# String normalisation (used for comparing our tags against MusicBrainz)
# ---------------------------------------------------------------------------

_PAREN_NOISE = re.compile(
    r"""\s*[\(\[\{]\s*
        (?:official\s+(?:music\s+)?video|official\s+audio|lyric[s]?\s*video|audio|video|
           hd|hq|4k|remaster(?:ed)?(?:\s*\d{4})?|explicit|clean|album\s+version|
           radio\s+edit|single\s+version|bonus\s+track|free\s+download)
        \s*[\)\]\}]""",
    re.IGNORECASE | re.VERBOSE,
)

_FEAT = re.compile(r"\s*(?:\(|\[)?\s*(?:feat\.?|ft\.?|featuring)\s+[^)\]]*(?:\)|\])?", re.IGNORECASE)

_ARTICLES = ("the ", "a ", "an ")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: Optional[str], *, drop_articles: bool = False, drop_feat: bool = False) -> str:
    """Aggressively normalise a string for fuzzy comparison (never for display)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _PAREN_NOISE.sub(" ", s)
    if drop_feat:
        s = _FEAT.sub(" ", s)
    s = strip_accents(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if drop_articles:
        for art in _ARTICLES:
            if s.startswith(art):
                s = s[len(art):]
                break
    return s


def similarity(a: Optional[str], b: Optional[str], **norm_kwargs) -> float:
    """0..1 similarity between two display strings."""
    na, nb = normalize(a, **norm_kwargs), normalize(b, **norm_kwargs)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    try:
        from rapidfuzz import fuzz
        # token_set_ratio copes with reordering and extra words ("feat." leftovers).
        return max(fuzz.ratio(na, nb), fuzz.token_set_ratio(na, nb)) / 100.0
    except ImportError:  # pragma: no cover - rapidfuzz is a hard dependency in practice
        import difflib
        return difflib.SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------------------
# Filesystem-safe names
# ---------------------------------------------------------------------------

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

#: Characters that read better replaced than deleted.
_PRETTY = {
    "/": "-", "\\": "-", ":": " -", "|": "-",
    "<": "(", ">": ")", '"': "'", "?": "", "*": "",
}


def sanitize_component(name: str, *, max_len: int = 120) -> str:
    """Make one path component safe on Windows, macOS and Linux."""
    if not name:
        return "Unknown"
    out = "".join(_PRETTY.get(ch, ch) for ch in name)
    out = _ILLEGAL.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Windows refuses trailing dots/spaces.
    out = out.rstrip(". ")
    if out.upper().split(".")[0] in _WINDOWS_RESERVED:
        out = "_" + out
    if len(out) > max_len:
        out = out[:max_len].rstrip(". ")
    return out or "Unknown"


def unique_path(path: Path) -> Path:
    """Return ``path``, or ``name (2).ext`` etc. if it is already taken."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for n in range(2, 1000):
        cand = parent / f"{stem} ({n}){suffix}"
        if not cand.exists():
            return cand
    raise FileExistsError(f"Could not find a free name near {path}")


# ---------------------------------------------------------------------------
# Guessing metadata from paths, used as a fallback and as a matching signal
# ---------------------------------------------------------------------------

#: Tried in order, so the most specific shapes must come first. In particular
#: "2-05 - Song" is a disc-track prefix, not track 2 by an artist called "05".
_FILENAME_PATTERNS = (
    # 1-01 - Title  /  1.01 Title
    re.compile(r"^(?P<disc>\d{1,2})[-.](?P<track>\d{1,3})\s*[-._)]\s*(?P<title>.+)$"),
    # 01 - Artist - Title
    re.compile(r"^(?P<track>\d{1,3})\s*[-._)]\s*(?P<artist>.+?)\s+-\s+(?P<title>.+)$"),
    # 01 - Title  /  01. Title
    re.compile(r"^(?P<track>\d{1,3})\s*[-._)]\s*(?P<title>.+)$"),
    # Artist - Title
    re.compile(r"^(?P<artist>.+?)\s+-\s+(?P<title>.+)$"),
)


def guess_from_filename(path: Path, root: Optional[Path] = None) -> dict[str, object]:
    """Best-effort artist/album/title/track from the path itself.

    Uses the two parent directories as artist/album hints, which matches the
    ``Artist/Album/track`` layout most libraries already have.

    ``root`` is the folder the user actually pointed the app at. It is what
    tells an album folder apart from a plain drop folder: files sitting
    directly in the root have no album folder above them, and one level down
    there is an album but no artist folder. Without it, a flat ingest folder
    turns its own name - and the name of whatever happens to contain it - into
    an "album" and an "artist", which is not a harmless guess: it becomes a
    search filter, it votes against every correct candidate during scoring,
    and it can be written to the file as real metadata.
    """
    stem = path.stem.strip()
    guess: dict[str, object] = {}

    for pat in _FILENAME_PATTERNS:
        m = pat.match(stem)
        if not m:
            continue
        groups = m.groupdict()
        if groups.get("track"):
            try:
                guess["track_no"] = int(groups["track"])
            except ValueError:
                pass
        if groups.get("disc"):
            try:
                guess["disc_no"] = int(groups["disc"])
            except ValueError:
                pass
        if groups.get("artist"):
            guess["artist"] = groups["artist"].strip(" -_.")
        if groups.get("title"):
            guess["title"] = groups["title"].strip(" -_.")
        break
    else:
        guess["title"] = stem

    parents = path.parents
    # How many folder levels sit between the file and the root the user chose.
    # None means "no root known", i.e. fall back to the old assumption that
    # the layout is Artist/Album/track.
    depth = _depth_below_root(path, root)
    if depth == 0:
        # Straight in the drop folder: the containing folder is the drop
        # folder itself, so there is no album or artist to read off it.
        return guess

    if len(parents) >= 1:
        album_dir = parents[0].name
        # "Album (1997)" / "1997 - Album"
        m = re.match(r"^(?P<album>.+?)\s*[\(\[](?P<year>(19|20)\d{2})[\)\]]\s*$", album_dir)
        if m:
            guess["album"] = m.group("album").strip()
            guess["year"] = int(m.group("year"))
        else:
            m = re.match(r"^(?P<year>(19|20)\d{2})\s*[-._]\s*(?P<album>.+)$", album_dir)
            if m:
                guess["album"] = m.group("album").strip()
                guess["year"] = int(m.group("year"))
            elif album_dir and not re.fullmatch(r"(?i)(cd|disc)\s*\d+", album_dir):
                guess["album"] = album_dir
    if len(parents) >= 2 and depth != 1:
        # depth == 1 means the album folder *is* the top level under the root,
        # so whatever contains the root is not an artist folder.
        parent_name = parents[1].name
        # If the album dir was actually "CD1", step up one more level.
        if re.fullmatch(r"(?i)(cd|disc)\s*\d+", parents[0].name) and len(parents) >= 3:
            guess["album"] = parent_name
            parent_name = parents[2].name
        guess.setdefault("album_artist", parent_name)
    return guess


def _depth_below_root(path: Path, root: Optional[Path]) -> Optional[int]:
    """How many folders sit between ``path``'s file and ``root``.

    0 = the file is directly in the root, 1 = one folder down, and so on.
    ``None`` when there is no root, or the file is not under it.
    """
    if root is None:
        return None
    try:
        relative = path.resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return None
    return len(relative.parts) - 1


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def safe_int(value: object) -> Optional[int]:
    """Parse ints out of tag values like ``"3/12"``, ``"03"``, ``(5,)``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and value:
        return safe_int(value[0])
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def split_pair(value: object) -> tuple[Optional[int], Optional[int]]:
    """Split ``"3/12"`` into ``(3, 12)``."""
    if value is None:
        return None, None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return safe_int(value[0]), safe_int(value[1])
    text = str(value)
    if "/" in text:
        left, _, right = text.partition("/")
        return safe_int(left), safe_int(right)
    return safe_int(text), None


def year_from_date(date: Optional[str]) -> Optional[int]:
    if not date:
        return None
    m = re.search(r"(19|20)\d{2}", str(date))
    return int(m.group()) if m else None


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
