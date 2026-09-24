"""MusicBrainz Web Service v2 client and result normalisation.

We only use the read-only search/lookup endpoints. Two rules the service asks
of every client and that this module enforces:

* a descriptive ``User-Agent`` including contact details
* at most one request per second (see :class:`RateLimiter`)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import Config
from ..models import TrackTags
from ..util import year_from_date
from .http import HttpClient, shared_limiter

log = logging.getLogger(__name__)

BASE = "https://musicbrainz.org/ws/2"

#: MusicBrainz's own artist entry for compilations by multiple artists.
VARIOUS_ARTISTS_MBID = "89ad4ac3-39f7-470e-963a-56509c546377"

RECORDING_INC = "artists+releases+release-groups+isrcs+media"
RELEASE_INC = "recordings+artist-credits+release-groups+labels"


def _credit_to_name(artist_credit: list[dict[str, Any]] | None) -> tuple[Optional[str], Optional[str]]:
    """Join a MusicBrainz artist-credit into a display string plus the lead MBID."""
    if not artist_credit:
        return None, None
    parts: list[str] = []
    for credit in artist_credit:
        name = credit.get("name") or (credit.get("artist") or {}).get("name") or ""
        parts.append(name)
        join = credit.get("joinphrase") or ""
        if join:
            parts.append(join)
    lead_id = (artist_credit[0].get("artist") or {}).get("id")
    return ("".join(parts).strip() or None), lead_id


class MusicBrainzClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HttpClient(
            user_agent=cfg.user_agent,
            rate_limiter=shared_limiter("musicbrainz", max(1.0, cfg.musicbrainz_rate_limit)),
        )

    # ------------------------------------------------------------------
    # raw endpoints
    # ------------------------------------------------------------------
    def lookup_recording(self, mbid: str) -> Optional[dict[str, Any]]:
        return self.http.get_json(
            f"{BASE}/recording/{mbid}",
            {"inc": RECORDING_INC, "fmt": "json"},
            cache_key=f"mb:recording:{mbid}",
        )

    def lookup_release(self, mbid: str) -> Optional[dict[str, Any]]:
        return self.http.get_json(
            f"{BASE}/release/{mbid}",
            {"inc": RELEASE_INC, "fmt": "json"},
            cache_key=f"mb:release:{mbid}",
        )

    def lookup_release_group(self, mbid: str) -> Optional[dict[str, Any]]:
        return self.http.get_json(
            f"{BASE}/release-group/{mbid}",
            {"inc": "genres", "fmt": "json"},
            cache_key=f"mb:release-group:genres:{mbid}",
        )

    def lookup_artist(self, mbid: str) -> Optional[dict[str, Any]]:
        return self.http.get_json(
            f"{BASE}/artist/{mbid}",
            {"inc": "genres", "fmt": "json"},
            cache_key=f"mb:artist:genres:{mbid}",
        )

    def search_recordings(self, *, title: Optional[str], artist: Optional[str] = None,
                          album: Optional[str] = None, duration_s: Optional[float] = None,
                          limit: int = 10) -> list[dict[str, Any]]:
        if not title:
            return []
        clauses = [f'recording:"{_escape(title)}"']
        if artist:
            clauses.append(f'artist:"{_escape(artist)}"')
        if album:
            clauses.append(f'release:"{_escape(album)}"')
        if duration_s:
            # +/- 5 seconds, in milliseconds - a cheap but very effective filter.
            lo, hi = int((duration_s - 5) * 1000), int((duration_s + 5) * 1000)
            clauses.append(f"dur:[{max(0, lo)} TO {hi}]")
        query = " AND ".join(clauses)
        data = self.http.get_json(
            f"{BASE}/recording",
            {"query": query, "fmt": "json", "limit": limit},
            cache_key=f"mb:search:rec:{query}:{limit}",
            not_found_is_empty=False,   # a search 404 is a transient fault, not "no such song"
        )
        return (data or {}).get("recordings", []) or []

    def search_releases(self, *, album: str, artist: Optional[str] = None,
                        track_count: Optional[int] = None, limit: int = 8) -> list[dict[str, Any]]:
        clauses = [f'release:"{_escape(album)}"']
        if artist:
            clauses.append(f'artist:"{_escape(artist)}"')
        if track_count:
            clauses.append(f"tracks:{track_count}")
        query = " AND ".join(clauses)
        data = self.http.get_json(
            f"{BASE}/release",
            {"query": query, "fmt": "json", "limit": limit},
            cache_key=f"mb:search:rel:{query}:{limit}",
            not_found_is_empty=False,
        )
        return (data or {}).get("releases", []) or []

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------
    def recording_to_tags(self, recording: dict[str, Any],
                          release: Optional[dict[str, Any]] = None) -> tuple[TrackTags, str]:
        """Flatten a recording (and the release it came from) into tags.

        Returns ``(tags, release_summary)`` where the summary is the short
        human string shown next to a candidate in the UI.
        """
        tags = TrackTags()
        tags.title = recording.get("title")
        tags.mb_recording_id = recording.get("id")
        artist_name, artist_id = _credit_to_name(recording.get("artist-credit"))
        tags.artist = artist_name
        tags.mb_artist_id = artist_id

        isrcs = recording.get("isrcs") or []
        if isrcs:
            tags.isrc = isrcs[0]

        release = release or _pick_release(recording)
        summary = ""
        if release:
            tags.album = release.get("title")
            tags.mb_release_id = release.get("id")
            album_artist, album_artist_id = _credit_to_name(release.get("artist-credit"))
            tags.album_artist = album_artist
            tags.mb_album_artist_id = album_artist_id
            tags.compilation = album_artist_id == VARIOUS_ARTISTS_MBID

            rg = release.get("release-group") or {}
            tags.mb_release_group_id = rg.get("id") or None
            # Prefer the release date; fall back to the first release of the group.
            tags.date = release.get("date") or rg.get("first-release-date") or None
            tags.year = year_from_date(tags.date)

            pos = _track_position(release, recording.get("id"))
            if pos:
                tags.track_no, tags.track_total, tags.disc_no, tags.disc_total = pos

            summary = _release_summary(release, tags)

        if not tags.album_artist:
            tags.album_artist = tags.artist
        return tags, summary

    def genre_for(self, tags: TrackTags) -> Optional[str]:
        """The top-voted MusicBrainz genre for a matched track, or ``None``.

        Taken from the release group first, so every track on an album gets
        the same genre and Plex does not show one album under three. Only
        when the album has no genres does it fall back to the album artist -
        never to "Various Artists", whose genres describe nothing. Genres
        are community votes, so a single stray vote is not enough.
        """
        if tags.mb_release_group_id:
            genre = _top_genre(self.lookup_release_group(tags.mb_release_group_id))
            if genre:
                return genre
        artist_id = tags.mb_album_artist_id or tags.mb_artist_id
        if artist_id and artist_id != VARIOUS_ARTISTS_MBID:
            return _top_genre(self.lookup_artist(artist_id))
        return None

    def full_release_tracklist(self, release_mbid: str) -> list[dict[str, Any]]:
        """Every track on a release, flattened across discs.

        Used by album-level matching, which is far more reliable than matching
        tracks one at a time.
        """
        release = self.lookup_release(release_mbid)
        if not release:
            return []
        out: list[dict[str, Any]] = []
        media = release.get("media") or []
        disc_total = len(media)
        for medium in media:
            disc_no = medium.get("position") or 1
            track_total = medium.get("track-count") or len(medium.get("tracks") or [])
            for track in medium.get("tracks") or []:
                rec = track.get("recording") or {}
                out.append({
                    "recording": rec,
                    "title": track.get("title") or rec.get("title"),
                    "length_ms": track.get("length") or rec.get("length"),
                    "track_no": _int_or_none(track.get("position") or track.get("number")),
                    "track_total": track_total,
                    "disc_no": disc_no,
                    "disc_total": disc_total,
                    "artist_credit": track.get("artist-credit") or rec.get("artist-credit"),
                })
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Escape Lucene special characters in a MusicBrainz query term."""
    out = []
    for ch in str(text):
        if ch in '+-&|!(){}[]^"~*?:\\/':
            out.append("\\")
        out.append(ch)
    return "".join(out)


#: Genres below this many votes are one person's opinion, not a consensus.
MIN_GENRE_VOTES = 2

#: Words kept lower case inside a genre name ("Drum and Bass").
_GENRE_SMALL_WORDS = {"and", "of", "the", "n", "'n'", "in", "de"}
#: Written in capitals rather than title case.
_GENRE_ACRONYMS = {"uk": "UK", "us": "US", "edm": "EDM", "idm": "IDM", "ebm": "EBM",
                   "r&b": "R&B", "dj": "DJ", "nwobhm": "NWOBHM", "aor": "AOR"}


def _top_genre(entity: Optional[dict[str, Any]]) -> Optional[str]:
    """The most-voted genre on a MusicBrainz entity, display-cased."""
    genres = [g for g in (entity or {}).get("genres") or []
              if g.get("name") and (g.get("count") or 0) >= MIN_GENRE_VOTES]
    if not genres:
        return None
    # Ties go alphabetically, so the same data always gives the same answer.
    best = min(genres, key=lambda g: (-g["count"], g["name"]))
    return _genre_display(best["name"])


def _genre_display(name: str) -> str:
    """``"alternative rock"`` -> ``"Alternative Rock"``; ``"k-pop"`` -> ``"K-Pop"``.

    MusicBrainz stores genre names in lower case; tags read better, and match
    what most libraries already hold, in title case.
    """
    def word(w: str, first: bool) -> str:
        if w in _GENRE_ACRONYMS:
            return _GENRE_ACRONYMS[w]
        if not first and w in _GENRE_SMALL_WORDS:
            return w
        return "-".join(part[:1].upper() + part[1:] for part in w.split("-"))

    words = name.strip().lower().split()
    return " ".join(word(w, i == 0) for i, w in enumerate(words))


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_release(recording: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Choose the most representative release a recording appears on.

    Preference order: official album > earliest date > anything. This avoids
    tagging everything as a random 2011 Japanese bonus-disc reissue.
    """
    releases = recording.get("releases") or []
    if not releases:
        return None

    def sort_key(rel: dict[str, Any]):
        status = (rel.get("status") or "").lower()
        rg = rel.get("release-group") or {}
        primary = (rg.get("primary-type") or "").lower()
        secondary = [s.lower() for s in (rg.get("secondary-types") or [])]
        return (
            0 if status == "official" else 1,
            0 if primary == "album" else (1 if primary == "ep" else 2),
            1 if ("compilation" in secondary or "live" in secondary) else 0,
            rel.get("date") or "9999",
        )

    return sorted(releases, key=sort_key)[0]


def _track_position(release: dict[str, Any], recording_id: Optional[str]):
    """Pull (track_no, track_total, disc_no, disc_total) out of a release blob.

    Search results only include the medium the match was found on, not the whole
    release, so ``len(media)`` is not the disc count there - trusting it yields
    nonsense like "disc 2 of 1", which then suppresses the disc prefix in
    filenames and lets two discs collide. When the numbers cannot both be true,
    report the disc total as unknown and let album consolidation fill it in from
    a full release lookup.
    """
    media = release.get("media") or []
    media_count = len(media) or None
    for medium in media:
        disc_no = medium.get("position") or 1
        track_total = medium.get("track-count")
        tracks = medium.get("track") or medium.get("tracks") or []

        disc_total = media_count
        if disc_total is not None and disc_no and disc_no > disc_total:
            disc_total = None          # partial view: we only got one medium

        for track in tracks:
            rec_id = (track.get("recording") or {}).get("id")
            if recording_id and rec_id and rec_id != recording_id:
                continue
            number = _int_or_none(track.get("position") or track.get("number"))
            return number, track_total, disc_no, disc_total
        # Search results include track-offset instead of a full tracklist.
        if "track-offset" in medium and not tracks:
            offset = _int_or_none(medium.get("track-offset"))
            if offset is not None:
                return offset + 1, track_total, disc_no, disc_total
    return None


def _release_summary(release: dict[str, Any], tags: TrackTags) -> str:
    bits = [tags.album or "Unknown album"]
    if tags.album_artist:
        bits.append(f"by {tags.album_artist}")
    extras = []
    if tags.year:
        extras.append(str(tags.year))
    media = release.get("media") or []
    if media and media[0].get("format"):
        extras.append(media[0]["format"])
    country = release.get("country")
    if country:
        extras.append(country)
    if extras:
        bits.append("(" + ", ".join(extras) + ")")
    return " ".join(bits)
