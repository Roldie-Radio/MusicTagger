"""Finding copies of the same track.

Against a Plex library before exporting into it (:func:`build_index`,
:func:`find_duplicate`), and inside the scanned library itself
(:func:`find_groups`), where the copies can then be merged - see
:mod:`musictag.merge`.

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
from typing import Any, Callable, Iterable, Optional

from .library import iter_audio_files
from .models import TAG_FIELDS, Track, TrackTags
from .tags import read_file
from .util import normalize, similarity

log = logging.getLogger(__name__)

#: How similar artist+title+album must be (0..1) to call it a fuzzy duplicate.
FUZZY_THRESHOLD = 0.82

#: Two files this far apart in length are different versions (a live take,
#: an extended mix, a radio edit), however alike their tags read. Rips of the
#: same recording differ by a second or two of silence at most.
DURATION_TOLERANCE_S = 5.0

#: Album names at least this alike (0..1) count as the same album.
ALBUM_THRESHOLD = 0.8


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
        # The scanner's own walk, so the trash and Duplicates folders - copies
        # already set aside - are not reported as "already in Plex".
        for path in iter_audio_files([root]):
            if cancelled and cancelled():
                return ExistingIndex(root=root, by_mbid=by_mbid, by_title=by_title, count=0)
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


def fuzzy_score(a: TrackTags, b: TrackTags) -> float:
    """0..1: how alike two sets of tags are, as descriptions of one song."""
    title_sim = similarity(a.title, b.title, drop_feat=True)
    artist_sim = similarity(a.artist or a.album_artist,
                            b.artist or b.album_artist,
                            drop_feat=True, drop_articles=True)
    album_sim = similarity(a.album, b.album, drop_feat=True)
    # Artist agreement matters most for telling two different songs that
    # happen to share a title apart; album is corroborating, not required
    # (a compilation vs. the original studio album is still the same song).
    return title_sim * 0.5 + artist_sim * 0.4 + album_sim * 0.1


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
        score = fuzzy_score(tags, existing)
        if score >= threshold and (best is None or score > best.score):
            best = DuplicateMatch(kind="fuzzy", existing_path=path,
                                  existing_tags=existing, score=score)
    return best


# ---------------------------------------------------------------------------
# Duplicates inside the scanned library
# ---------------------------------------------------------------------------

@dataclass
class Sameness:
    """Why two files are taken to be the same track."""

    kind: str                     # "mbid" | "fuzzy"
    score: float
    #: False when both name an album and the names differ - the same song on
    #: a studio album and on a compilation. Each copy then belongs to an
    #: album of its own, and removing one leaves that album a track short.
    same_album: bool


def identity_tags(track: Track) -> TrackTags:
    """The best description of what a file is: its tags, plus any proposal.

    A track that has been identified but not applied yet already carries the
    MusicBrainz ID in its proposal, which is far better evidence than the
    text of whatever tags the file happened to arrive with.
    """
    if track.match and track.match.proposed and not track.match.proposed.is_empty():
        return track.current.merged_with(track.match.proposed)
    return track.current


def same_album(a: TrackTags, b: TrackTags) -> bool:
    if not a.album or not b.album:
        return True               # nothing says they differ
    return similarity(a.album, b.album, drop_feat=True) >= ALBUM_THRESHOLD


def same_recording(a: TrackTags, b: TrackTags, *, a_duration: float = 0.0,
                   b_duration: float = 0.0,
                   threshold: float = FUZZY_THRESHOLD) -> Optional[Sameness]:
    """Are these two files copies of one track? ``None`` when they are not."""
    if a.mb_recording_id and b.mb_recording_id:
        # Both identified: MusicBrainz decides, in either direction.
        if a.mb_recording_id != b.mb_recording_id:
            return None
        return Sameness("mbid", 1.0, same_album(a, b))

    title = normalize(a.title, drop_feat=True)
    if not title or title != normalize(b.title, drop_feat=True):
        # Near-identical titles are usually different songs ("Part 1" and
        # "Part 2"), so the fuzzy pass only compares the rest.
        return None
    if (a_duration and b_duration
            and abs(a_duration - b_duration) > DURATION_TOLERANCE_S):
        return None
    score = fuzzy_score(a, b)
    if score < threshold:
        return None
    return Sameness("fuzzy", score, same_album(a, b))


# -- which copy is the one worth keeping ------------------------------------

def _completeness(tags: TrackTags) -> int:
    return sum(1 for name in TAG_FIELDS if getattr(tags, name) not in (None, "", False))


def rank_key(track: Track) -> tuple:
    """Sort key: the larger, the better the copy.

    In order: nothing seriously wrong with the audio (when Check Quality has
    looked), genuinely lossless, resolution, bitrate, quality score, cover
    art, how complete the tags are, and finally size.
    """
    props, quality = track.props, track.quality
    issues = quality.issues if quality and quality.analysed else []
    serious = any(i.severity == "high" for i in issues)
    # A FLAC made from an MP3 is lossless in name only.
    lossless = props.lossless and not any(i.code == "fake_lossless" for i in issues)
    return (
        not serious,
        lossless,
        (props.bit_depth, props.sample_rate) if lossless else (0, 0),
        0 if lossless else props.bitrate_kbps,
        quality.score if quality and quality.analysed else -1,
        track.current.has_art,
        _completeness(track.current),
        props.filesize,
    )


def pick_best(tracks: Iterable[Track]) -> Track:
    """The copy to keep. Ties go to the first path, so the answer is stable."""
    ordered = sorted(tracks, key=lambda t: t.path)
    return max(ordered, key=rank_key)


_RANK_REASONS = (
    "no serious quality problems",
    "lossless",
    "higher resolution",
    "higher bitrate",
    "better quality score",
    "has cover art",
    "more complete tags",
    "larger file",
)


def why_best(best: Track, others: Iterable[Track]) -> str:
    """One short phrase saying what put ``best`` ahead of the rest."""
    others = list(others)
    if not others:
        return ""
    key = rank_key(best)
    rival = max(rank_key(o) for o in others)
    for index, reason in enumerate(_RANK_REASONS):
        if key[index] != rival[index]:
            return reason if key[index] > rival[index] else ""
    return "identical quality"


@dataclass
class DuplicateGroup:
    tracks: list[Track]           # best copy first
    kind: str                     # "mbid" when every copy shares one recording ID
    same_album: bool
    reason: str = ""              # why tracks[0] is the one to keep

    @property
    def keep(self) -> Track:
        return self.tracks[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "same_album": self.same_album,
            "reason": self.reason,
            "keep": self.keep.path,
            "tracks": [_copy_payload(t, self.keep) for t in self.tracks],
        }


def _copy_payload(track: Track, best: Track) -> dict[str, Any]:
    tags = identity_tags(track)
    quality = track.quality if track.quality and track.quality.analysed else None
    return {
        "path": track.path,
        "filename": track.filename or Path(track.path).name,
        "title": tags.title, "artist": tags.artist or tags.album_artist,
        "album": tags.album,
        "container": track.props.container,
        "lossless": track.props.lossless,
        "bitrate_kbps": track.props.bitrate_kbps,
        "duration_s": track.props.duration_s,
        "filesize": track.props.filesize,
        "quality_score": quality.score if quality else None,
        "has_art": track.current.has_art,
        # False for a copy that belongs to another album than the best one:
        # the review leaves those where they are unless told otherwise.
        "same_album": same_album(tags, identity_tags(best)),
    }


def find_groups(tracks: Iterable[Track]) -> list[DuplicateGroup]:
    """Group ``tracks`` into sets of copies of the same track.

    A file joins a group only if it matches *every* copy already in it. With
    looser chaining, an unidentified file could link two that MusicBrainz
    has already said are different recordings.
    """
    clusters: list[list[tuple[Track, TrackTags]]] = []
    by_mbid: dict[str, list[int]] = {}
    by_title: dict[str, list[int]] = {}

    # Best copies first, so each group forms around the one that will be kept.
    ordered = sorted(sorted(tracks, key=lambda t: t.path), key=rank_key, reverse=True)
    for track in ordered:
        if track.status == "error" and track.error:
            continue              # unreadable: nothing trustworthy to compare
        tags = identity_tags(track)
        title = normalize(tags.title, drop_feat=True)
        if not title and not tags.mb_recording_id:
            continue
        candidates = dict.fromkeys(
            by_mbid.get(tags.mb_recording_id or "", []) + by_title.get(title, []))
        home = None
        for index in candidates:
            if all(same_recording(tags, other_tags,
                                  a_duration=track.props.duration_s,
                                  b_duration=other.props.duration_s)
                   for other, other_tags in clusters[index]):
                home = index
                break
        if home is None:
            home = len(clusters)
            clusters.append([])
        clusters[home].append((track, tags))
        if tags.mb_recording_id and home not in by_mbid.setdefault(tags.mb_recording_id, []):
            by_mbid[tags.mb_recording_id].append(home)
        if title and home not in by_title.setdefault(title, []):
            by_title[title].append(home)

    groups = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        members = [t for t, _ in cluster]
        best = pick_best(members)
        rest = sorted((t for t in members if t is not best),
                      key=rank_key, reverse=True)
        best_tags = identity_tags(best)
        ids = {tags.mb_recording_id for _, tags in cluster}
        groups.append(DuplicateGroup(
            tracks=[best] + rest,
            kind="mbid" if len(ids) == 1 and None not in ids and "" not in ids else "fuzzy",
            same_album=all(same_album(best_tags, tags) for _, tags in cluster),
            reason=why_best(best, rest),
        ))
    groups.sort(key=lambda g: g.keep.path.lower())
    return groups
