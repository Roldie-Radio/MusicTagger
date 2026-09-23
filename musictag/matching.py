"""Identify tracks and put an honest number on how sure we are.

Confidence is a weighted average of independent *signals* - fingerprint score,
duration agreement, title/artist/album similarity, track position - then
adjusted for two things a raw average misses:

* **Evidence**: a file with no tags and no fingerprint gives us nothing to
  verify a guess against, so its ceiling is low no matter how good the guess
  looks.
* **Ambiguity**: if the runner-up candidate scores nearly as well, the top
  score is worth less. Two plausible answers is not the same as one good one.

Every signal is kept on the result so the UI can explain the number rather
than just assert it.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import Config
from .fingerprint import AcoustIDClient, FingerprintUnavailable
from .models import Candidate, MatchResult, Signal, Track, TrackTags
from .providers.coverart import CoverArtClient
from .providers.http import ProviderError
from .providers.musicbrainz import MusicBrainzClient, VARIOUS_ARTISTS_MBID
from .util import guess_from_filename, normalize, similarity, year_from_date

log = logging.getLogger(__name__)

# Relative importance of each signal. These are deliberately blunt: precision
# here is false comfort, the ordering is what matters.
WEIGHTS = {
    "fingerprint": 3.0,
    "existing_mbid": 3.0,
    "duration": 2.0,
    "title": 2.0,
    "artist": 1.5,
    "album": 1.0,
    "track_no": 0.5,
    "search_rank": 0.5,
    # The filename is independent evidence, not merely a fallback for when tags
    # are missing - it's frequently the *more* trustworthy source, since a
    # scrambled, mislabelled or half-edited tag is common and a renamed file is
    # not. Weighted on a par with the embedded title/artist tags themselves,
    # so a candidate that matches the filename but disagrees with a bad tag
    # still gets real credit for it, rather than none at all.
    "filename_title": 2.0,
    "filename_artist": 1.0,
    # Deliberately light. This only exists to break ties inside a fingerprint
    # cluster, where the original and a covers-band version agree on title and
    # length and there is nothing else to go on. It should settle a coin flip,
    # never outvote real evidence.
    "fingerprint_consensus": 0.6,
}

#: How much we trust each field once a candidate is chosen. Track/disc numbers
#: come from a specific release and can be wrong even when the recording is
#: right; genre is not something MusicBrainz gives us directly at all.
FIELD_RELIABILITY = {
    "title": 1.00,
    "artist": 0.98,
    "album": 0.92,
    "album_artist": 0.92,
    "track_no": 0.88,
    "track_total": 0.85,
    "disc_no": 0.90,
    "disc_total": 0.85,
    "date": 0.80,
    "year": 0.80,
    "isrc": 0.95,
    "composer": 0.70,
    "genre": 0.55,
    "mb_recording_id": 1.00,
    "mb_release_id": 0.92,
    "mb_release_group_id": 0.92,
    "mb_artist_id": 0.98,
    "mb_album_artist_id": 0.92,
    "compilation": 0.90,
}

#: How many MusicBrainz search results get scored, as a multiple of
#: max_candidates. Doubled from a straight 1x - see _search_candidates.
MB_SEARCH_POOL_MULTIPLIER = 2

#: An AcoustID score at or above this is direct evidence about the audio
#: itself - the file *is* that recording - which no amount of agreement
#: between two pieces of text can match.
STRONG_FINGERPRINT = 0.8

#: Below this, an AcoustID cluster hit is noise rather than evidence, and the
#: junk recordings hanging off it are not worth offering as candidates.
ACOUSTID_MIN_SCORE = 0.5

#: How far down the candidate list to look for a recording that is actually on
#: a release, when the winner turns out not to be. Each step costs a lookup, so
#: this stays small - it is a rescue, not a search.
ENRICH_FALLBACK_LIMIT = 4

#: How many recordings to take from one AcoustID result, after ordering them
#: by length against the file. A cluster can carry a dozen - the original plus
#: covers, karaoke versions and outright mislinks - but they cost nothing to
#: build now, so this is set wide enough that the right one is never cut and
#: scoring gets to make the call.
ACOUSTID_MAX_RECORDINGS = 8

#: Live versions, extended/12" mixes, radio edits and medley edits are all
#: still "the same song" as far as tagging is concerned, and routinely run
#: minutes longer or shorter than the studio version a search turns up - so
#: these stay generous. DURATION_VETO_S in particular is deliberately far out:
#: it exists to catch a genuinely different song of unrelated length, not to
#: penalise a legitimate alternate version.
DURATION_IDENTICAL_S = 0.1   # below this the two lengths are the same number
DURATION_EXACT_S = 1.0       # within this, close enough to call it a match in words
DURATION_PERFECT_S = 10.0    # within this, close enough to score full marks
DURATION_ZERO_S = 90.0       # beyond this, the signal scores zero
DURATION_VETO_S = 180.0      # beyond this, it is almost certainly the wrong track

#: What an identical length is worth, as a multiple of the duration weight.
#: Being within ten seconds is ordinary - versions, fades and rounding all
#: land there. Agreeing to the second is not: it is the single best
#: corroboration available short of a fingerprint, so it counts for half
#: again as much rather than being lumped in with "close".
DURATION_EXACT_BOOST = 1.5

#: How much of the duration signal's weight survives at the far edge of the
#: 0..DURATION_ZERO_S ramp (see _score). A few seconds' difference is barely
#: worth mentioning - remasters trim silence, rips round differently - so it
#: should not pull as hard on the average as an exact match does; a gap that
#: has drifted all the way out to DURATION_ZERO_S is more telling but still
#: short of DURATION_VETO_S's "different recording" call, so it keeps this
#: floor rather than dropping to zero.
DURATION_WEIGHT_FLOOR = 0.1


class Matcher:
    """Turns a :class:`Track` into a :class:`MatchResult`."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mb = MusicBrainzClient(cfg)
        self.acoustid = AcoustIDClient(cfg)
        self.coverart = CoverArtClient(cfg)

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def identify(self, track: Track) -> MatchResult:
        """Identify a single file."""
        path = Path(track.path)
        observed = self._observations(track)
        candidates: list[Candidate] = []
        notes: list[str] = []
        methods: list[str] = []

        # 1. The file already claims a MusicBrainz recording. Trust but verify.
        if track.current.mb_recording_id:
            try:
                rec = self.mb.lookup_recording(track.current.mb_recording_id)
                if rec:
                    tags, summary = self.mb.recording_to_tags(rec)
                    cand = Candidate(source="existing-mbid", tags=tags, release_summary=summary,
                                     raw_id=tags.mb_recording_id, length_s=_length_s(rec))
                    cand.signals.append(Signal(
                        "existing_mbid",
                        "File already carries a MusicBrainz recording ID that resolved",
                        1.0, WEIGHTS["existing_mbid"]))
                    candidates.append(cand)
                    methods.append("existing MusicBrainz ID")
            except ProviderError as exc:
                notes.append(f"MusicBrainz lookup of the embedded ID failed: {exc}")

        # 2. Acoustic fingerprint - the only signal that reads the audio itself.
        if self.acoustid.available:
            try:
                fp_candidates, fingerprint, fp_duration = self._acoustid_candidates(path, observed)
                track.fingerprint = fingerprint
                candidates.extend(fp_candidates)
                if fp_candidates:
                    methods.append("acoustic fingerprint")
                else:
                    notes.append("Fingerprinted successfully, but AcoustID has no match for this audio.")
            except FingerprintUnavailable as exc:
                notes.append(str(exc))
            except ProviderError as exc:
                notes.append(f"Fingerprint lookup failed: {exc}")
            except Exception as exc:  # noqa: BLE001 - never let one bad file stop a run
                log.debug("Fingerprint failed for %s", path, exc_info=True)
                notes.append(f"Fingerprinting error: {exc}")
        else:
            notes.append("Fingerprinting is off (no AcoustID key or fpcalc); "
                         "matching on tags and filename only.")

        # 3. Text search against MusicBrainz - but only if it can still change
        #    the answer. The search costs several rate-limited round trips, by
        #    far the slowest thing here, and when the audio has already been
        #    identified outright they are round trips spent confirming what a
        #    fingerprint established more reliably than text ever could.
        if _has_strong_fingerprint(candidates):
            notes.append("The audio was identified by fingerprint, so no text search "
                         "was needed - alternatives below come from that match.")
        else:
            try:
                candidates.extend(self._search_candidates(observed, notes))
                if any(c.source == "musicbrainz-search" for c in candidates):
                    methods.append("MusicBrainz search")
            except ProviderError as exc:
                notes.append(f"MusicBrainz search failed: {exc}")

        # 4. Nothing online worked - at least offer what the file/path implies.
        if not candidates:
            fallback = self._filename_candidate(track, observed)
            if fallback:
                candidates.append(fallback)
                methods.append("filename")

        return self._finish(track, observed, candidates, notes, methods)

    def identify_album(self, tracks: list[Track], *, progress=None,
                       cancelled: Optional[Callable[[], bool]] = None) -> None:
        """Identify a folder of tracks, then make them agree with each other.

        Tracks matched independently often scatter across several releases of
        the same album. Plex then shows three half-albums. So: identify each
        file, find the release that most of them point at, and re-seat the rest
        onto that release's tracklist.

        Files are worked on concurrently. Almost all of the elapsed time is
        spent waiting - on the MusicBrainz rate limiter, and on MusicBrainz
        itself - so running them one at a time left the machine idle while
        decoding and fingerprinting, which are local and fast, could have been
        happening in the gaps. The rate limiters are process-wide, so this
        overlaps the waiting without ever raising the request rate.

        ``cancelled`` is polled before each file. One folder can hold
        thousands of files - a flat dump of downloads is a single group - so
        checking only between folders left Cancel doing nothing for an hour.
        Files not reached keep no match, and a cancelled folder is not
        consolidated, since a partial vote is not the folder's answer.
        """
        done = 0
        lock = threading.Lock()

        def identify_one(track: Track) -> None:
            nonlocal done
            if cancelled and cancelled():
                return
            try:
                track.match = self.identify(track)
                track.status = "identified"
            except Exception as exc:  # noqa: BLE001
                log.exception("Identify failed for %s", track.path)
                track.status = "error"
                track.error = str(exc)
            with lock:
                done += 1
                if progress:
                    progress(done, len(tracks), track.path)

        workers = max(1, min(self.cfg.identify_workers, len(tracks)))
        if workers == 1:
            for track in tracks:
                identify_one(track)
        else:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="identify") as pool:
                list(pool.map(identify_one, tracks))

        if cancelled and cancelled():
            return
        self._consolidate_album(tracks)

    # ------------------------------------------------------------------
    # candidate generation
    # ------------------------------------------------------------------
    def _root_for(self, track: Track) -> Optional[Path]:
        """The configured library folder this track came from, if any.

        Lets :func:`guess_from_filename` tell "this folder is an album" apart
        from "this folder is just where the user dropped their files".
        """
        try:
            path = Path(track.path).resolve()
        except OSError:
            return None
        for lib in self.cfg.library_paths:
            candidate = Path(lib)
            try:
                path.relative_to(candidate.resolve())
            except (ValueError, OSError):
                continue
            return candidate
        return None

    def _observations(self, track: Track) -> dict[str, Any]:
        """What we actually know about the file, and how much of it there is."""
        tags = track.current
        guess = guess_from_filename(Path(track.path), self._root_for(track))

        title = tags.title or guess.get("title")
        artist = tags.artist or tags.album_artist or guess.get("artist") or guess.get("album_artist")
        album = tags.album or guess.get("album")

        # Evidence strength drives the confidence ceiling later on.
        evidence = 0
        if tags.title:
            evidence += 2
        if tags.artist:
            evidence += 2
        if tags.album:
            evidence += 1
        if track.props.duration_s > 0:
            evidence += 2
        if guess.get("title") and not tags.title:
            evidence += 1

        return {
            "title": title,
            "artist": artist,
            "album": album,
            # Kept separate from the blended fields above (which prefer tags
            # and only fall back to the filename when a tag is *absent*): a
            # candidate needs to be checked against what the filename itself
            # says regardless of whether a tag also exists, so a wrong tag
            # cannot silently hide disagreement with a good filename.
            "filename_title": guess.get("title"),
            "filename_artist": guess.get("artist") or guess.get("album_artist"),
            # Whether the file has its own embedded value, as opposed to
            # "title"/"artist" above having quietly fallen back to the
            # filename because there was nothing else. Scoring needs this to
            # know when a filename check is genuinely new evidence versus a
            # duplicate of a check it has already made.
            "has_tag_title": bool(tags.title),
            "has_tag_artist": bool(tags.artist),
            "track_no": tags.track_no or guess.get("track_no"),
            "disc_no": tags.disc_no or guess.get("disc_no"),
            "duration_s": track.props.duration_s,
            "from_tags": bool(tags.title or tags.artist),
            "evidence": evidence,
            "guess": guess,
        }

    def _acoustid_candidates(self, path: Path, observed: dict[str, Any]
                             ) -> tuple[list[Candidate], Optional[str], int]:
        results, fingerprint, duration = self.acoustid.identify(path)
        candidates: list[Candidate] = []

        # Built straight from what AcoustID already told us, with no
        # MusicBrainz round trip per candidate. That used to cost one
        # rate-limited lookup for every recording in the cluster - up to ten
        # per file, and the single largest chunk of the time identifying a
        # library took - to fill in album details for candidates that were
        # about to be discarded anyway. The winner gets the full picture
        # later, in :meth:`_enrich`, which is one lookup instead of ten.
        # How often each artist is credited across everything this fingerprint
        # matched. A song's original artist accumulates entries in MusicBrainz -
        # the album, the single, reissues, compilations - while a karaoke or
        # covers-band version usually has exactly one. When the audio matches
        # several recordings that are otherwise indistinguishable (same title,
        # same length, and a file with no tags to separate them), that count is
        # the only evidence available as to which is the real thing, and it
        # costs nothing: it is already in the response.
        cluster = Counter()
        for result in results[: self.cfg.max_candidates]:
            if float(result.get("score") or 0.0) < ACOUSTID_MIN_SCORE:
                continue
            for recording in (result.get("recordings") or []):
                for artist in (recording.get("artists") or []):
                    name = normalize(artist.get("name"), drop_articles=True)
                    if name:
                        cluster[name] += 1
        most_common = max(cluster.values()) if cluster else 0

        for result in results[: self.cfg.max_candidates]:
            fp_score = float(result.get("score") or 0.0)
            if fp_score < ACOUSTID_MIN_SCORE:
                continue        # noise: a weak cluster hit is not evidence
            # AcoustID returns a cluster's recordings in no useful order - the
            # original can sit below a karaoke version and two string-quartet
            # covers. Ordering by how closely each one's length matches this
            # file puts the real recording in reach before the cap, using a
            # number already in hand and costing nothing to check.
            # Recordings credited to the cluster's most-credited artist go
            # first regardless, because MusicBrainz often has no length at all
            # for exactly those entries - and sorting unknown lengths last (the
            # only safe thing to do with them otherwise) would push the real
            # recording below a couple of covers that happen to have lengths.
            ordered = sorted(
                (result.get("recordings") or []),
                key=lambda r: (0 if _credited_to(r, cluster, most_common) else 1,
                               _duration_gap(r, observed["duration_s"])))
            for recording in ordered[:ACOUSTID_MAX_RECORDINGS]:
                rec_id = recording.get("id")
                if not rec_id:
                    continue
                as_mb = _acoustid_to_mb(recording)
                tags, summary = self.mb.recording_to_tags(as_mb)
                if not tags.title:
                    continue
                cand = Candidate(source="acoustid", tags=tags, release_summary=summary,
                                 raw_id=rec_id, length_s=_length_s(as_mb))
                cand.signals.append(Signal(
                    "fingerprint",
                    f"AcoustID match {fp_score * 100:.0f}%",
                    fp_score, WEIGHTS["fingerprint"]))

                credited = cluster.get(normalize(tags.artist, drop_articles=True), 0)
                if most_common and credited:
                    cand.signals.append(Signal(
                        "fingerprint_consensus",
                        f"{credited} of the {sum(cluster.values())} recordings matching "
                        f"this audio are credited to {tags.artist}",
                        credited / most_common, WEIGHTS["fingerprint_consensus"]))
                candidates.append(cand)
        return candidates, fingerprint, duration

    def _enrich(self, cand: Candidate) -> None:
        """Fetch the album, date and track position for one chosen candidate.

        Fingerprint candidates are built from AcoustID's thin metadata - title,
        artist, length - which is enough to score them but not enough to write
        to a file. Only the candidate that actually wins needs the rest, so it
        is fetched here rather than for every candidate up front.
        """
        if not cand.raw_id or cand.tags.album:
            return
        try:
            full = self.mb.lookup_recording(cand.raw_id)
        except ProviderError as exc:
            log.debug("Could not enrich %s: %s", cand.raw_id, exc)
            return
        if not full:
            return
        tags, summary = self.mb.recording_to_tags(full)
        if not tags.title:
            return
        cand.tags = tags
        cand.release_summary = summary or cand.release_summary
        cand.length_s = _length_s(full) or cand.length_s

    def _search_candidates(self, observed: dict[str, Any],
                           notes: Optional[list[str]] = None) -> list[Candidate]:
        """Search MusicBrainz, loosening the query until something comes back.

        The duration filter is the single most effective way to cut false
        matches, but it is a *hard exclusion*: a gapless rip, a radio edit, or a
        file with a wrong length tag would return nothing at all, and the user
        would be told "no match found" when the real answer was one query away.

        So duration is dropped as a last resort. Candidates found that way are
        still scored against the file's real length, which penalises them
        heavily - a low-confidence match with a visible reason beats silence.
        """
        title = observed["title"]
        if not title:
            return []

        artist, album = observed["artist"], observed["album"]
        duration = observed["duration_s"] or None
        # MusicBrainz's own relevance ranking is a coarse text-similarity
        # heuristic, not our scoring - on a file with weak or missing tags the
        # right answer is often not its top guess. Fetch and keep a wider pool
        # here than max_candidates alone would give, so a correct match a few
        # slots down MusicBrainz's own ranking still gets scored rather than
        # being cut before it is ever compared.
        search_pool = self.cfg.max_candidates * MB_SEARCH_POOL_MULTIPLIER
        limit = search_pool * 2

        attempts = [
            # (query kwargs, whether duration was used to filter)
            ({"title": title, "artist": artist, "album": album, "duration_s": duration}, True),
            ({"title": title, "artist": artist, "duration_s": duration}, True),
            ({"title": title, "artist": artist}, False),
            ({"title": title}, False),
        ]

        # "08 The Reason" is a track number glued to a title by a filename with
        # no separator to split on. Guessing that up front would mangle titles
        # that genuinely start with a number ("99 Problems", "50 Ways..."), so
        # the literal title is always tried first and this only runs when
        # MusicBrainz found nothing for it - at which point the number is
        # almost certainly the problem.
        bare = _strip_leading_track_number(title)
        if bare:
            attempts.append(({"title": bare, "artist": artist}, False))
            attempts.append(({"title": bare}, False))

        recordings: list[dict[str, Any]] = []
        dropped_duration = False
        tried: set[tuple] = set()
        for kwargs, used_duration in attempts:
            query = {k: v for k, v in kwargs.items() if v}
            key = tuple(sorted(query.items()))
            if key in tried:
                continue
            tried.add(key)
            recordings = self.mb.search_recordings(limit=limit, **query)
            if recordings:
                dropped_duration = bool(duration) and not used_duration
                break

        if dropped_duration and notes is not None:
            notes.append(
                "No release of this length was found, so the search was widened to "
                "ignore duration. Check the length before trusting the match.")

        candidates: list[Candidate] = []
        for rank, rec in enumerate(recordings[:search_pool]):
            tags, summary = self.mb.recording_to_tags(rec)
            if not tags.title:
                continue
            cand = Candidate(source="musicbrainz-search", tags=tags,
                             release_summary=summary, raw_id=rec.get("id"),
                             length_s=_length_s(rec),
                             duration_unverified=dropped_duration)
            # MusicBrainz's own relevance score, normalised and de-emphasised:
            # it reflects text similarity, which we measure ourselves anyway.
            mb_score = float(rec.get("score") or 0) / 100.0
            rank_score = max(0.0, mb_score * (1.0 - rank * 0.08))
            cand.signals.append(Signal(
                "search_rank",
                f"MusicBrainz search relevance {mb_score * 100:.0f}% (result #{rank + 1})",
                rank_score, WEIGHTS["search_rank"]))
            candidates.append(cand)
        return candidates

    def _filename_candidate(self, track: Track, observed: dict[str, Any]) -> Optional[Candidate]:
        guess = observed["guess"]
        if not guess.get("title"):
            return None
        tags = TrackTags(
            title=guess.get("title"),
            artist=guess.get("artist") or guess.get("album_artist"),
            album=guess.get("album"),
            album_artist=guess.get("album_artist") or guess.get("artist"),
            track_no=guess.get("track_no"),
            disc_no=guess.get("disc_no"),
            year=guess.get("year"),
            date=str(guess["year"]) if guess.get("year") else None,
        )
        cand = Candidate(source="filename", tags=tags,
                         release_summary="Derived from the file path - not verified against any database")
        cand.signals.append(Signal(
            "search_rank", "No online match; values read from the filename and folders", 0.35, 1.0))
        return cand

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _score(self, cand: Candidate, observed: dict[str, Any]) -> float:
        """Score one candidate against the observations. Returns 0..100."""
        tags = cand.tags
        signals = list(cand.signals)          # source-specific signals already attached

        # Duration - the single most useful cross-check we have.
        obs_dur = observed["duration_s"]
        cand_dur = cand.length_s
        veto = False
        if obs_dur and not cand_dur:
            # We have a length and the database does not. Say so at half weight
            # rather than silently dropping the strongest cross-check we have -
            # otherwise a candidate with no length looks *better* than one whose
            # length disagrees, which is exactly backwards.
            signals.append(Signal(
                "duration",
                "MusicBrainz has no length for this recording, so the file's "
                "length could not be checked",
                0.5, WEIGHTS["duration"] * 0.5))
        if obs_dur and cand_dur:
            delta = abs(obs_dur - cand_dur)
            if delta <= DURATION_PERFECT_S:
                score = 1.0
            elif delta >= DURATION_ZERO_S:
                score = 0.0
            else:
                score = 1.0 - (delta - DURATION_PERFECT_S) / (DURATION_ZERO_S - DURATION_PERFECT_S)
            # The signal's *influence*, not just its score, scales with how
            # close the lengths are, across three stretches:
            #
            # - Identical (to the second): two unrelated recordings landing on
            #   the same length to the second is a coincidence; the same
            #   recording doing it is the norm. That is worth more than the
            #   ordinary full weight, so it gets a boost.
            # - Close: normal rounding and fade differences. Full weight, but
            #   no boost - "within ten seconds" is not the same evidence as
            #   "the same length".
            # - Drifting out toward DURATION_ZERO_S: real but genuinely
            #   ambiguous (fade edits, silence trimming), so it is let to
            #   matter less and leave room for title/artist/album to decide.
            #
            # Confirmed way-off cases are still caught hard by the veto below,
            # which applies after and on top of this.
            if delta <= DURATION_IDENTICAL_S:
                # The same number, give or take the last decimal place either
                # side reported. Nothing below this is a real difference.
                factor = DURATION_EXACT_BOOST
            elif delta <= DURATION_PERFECT_S:
                # Taper from there, so the boost peaks at agreeing exactly
                # rather than being shared flat across the first second. Two
                # candidates can sit inside that second - one landing on the
                # same length, one half a second out - and the one that agrees
                # exactly is the better bet, which a flat band could not say.
                span = ((delta - DURATION_IDENTICAL_S)
                        / (DURATION_PERFECT_S - DURATION_IDENTICAL_S))
                factor = DURATION_EXACT_BOOST - span * (DURATION_EXACT_BOOST - 1.0)
            else:
                factor = DURATION_WEIGHT_FLOOR + (1 - DURATION_WEIGHT_FLOOR) * score
            detail = (f"Length matches to within {delta:.1f}s"
                      if delta <= DURATION_EXACT_S
                      else f"Length differs by {delta:.1f}s")
            signals.append(Signal("duration", detail, score,
                                  WEIGHTS["duration"] * factor))
            if delta > DURATION_VETO_S:
                veto = True

        title_similarity: Optional[float] = None
        if observed["title"] and tags.title:
            title_similarity = similarity(observed["title"], tags.title, drop_feat=True)
            signals.append(Signal("title", f'Title "{tags.title}" vs "{observed["title"]}"',
                                  title_similarity, WEIGHTS["title"]))

        # The filename, checked independently of whatever the embedded tag
        # says. Only added when the file *has* its own tag.title - otherwise
        # "title" above is already the filename (observed["title"] fell back
        # to it), and adding this too would weigh the same one piece of
        # evidence twice.
        filename_title_similarity: Optional[float] = None
        if observed["has_tag_title"] and observed["filename_title"] and tags.title:
            filename_title_similarity = similarity(
                observed["filename_title"], tags.title, drop_feat=True)
            signals.append(Signal(
                "filename_title",
                f'Filename says "{observed["filename_title"]}" vs candidate "{tags.title}"',
                filename_title_similarity, WEIGHTS["filename_title"]))

        if observed["artist"] and tags.artist:
            s = similarity(observed["artist"], tags.artist, drop_feat=True, drop_articles=True)
            signals.append(Signal("artist", f'Artist "{tags.artist}" vs "{observed["artist"]}"',
                                  s, WEIGHTS["artist"]))
        if observed["has_tag_artist"] and observed["filename_artist"] and tags.artist:
            s = similarity(observed["filename_artist"], tags.artist,
                           drop_feat=True, drop_articles=True)
            signals.append(Signal(
                "filename_artist",
                f'Filename says "{observed["filename_artist"]}" vs candidate "{tags.artist}"',
                s, WEIGHTS["filename_artist"]))
        if observed["album"] and tags.album:
            s = similarity(observed["album"], tags.album, drop_articles=True)
            signals.append(Signal("album", f'Album "{tags.album}" vs "{observed["album"]}"',
                                  s, WEIGHTS["album"]))
        if observed["track_no"] and tags.track_no:
            s = 1.0 if observed["track_no"] == tags.track_no else 0.0
            signals.append(Signal("track_no",
                                  f"Track number {tags.track_no} vs {observed['track_no']}",
                                  s, WEIGHTS["track_no"]))

        cand.signals = signals
        total_weight = sum(s.weight for s in signals)
        if not total_weight:
            return 0.0
        raw = sum(s.score * s.weight for s in signals) / total_weight * 100.0

        if veto:
            raw *= 0.35
            cand.signals.append(Signal(
                "duration", "Length is wildly different - probably a different recording",
                0.0, 0.0))

        # The title is the track's identity. A plain weighted average lets a
        # matching artist, album and duration outvote a completely wrong title -
        # but that combination is the signature of a *different track on the
        # same album*, which is precisely the mistake worth avoiding.
        #
        # Two things exempt a candidate from this veto, both because they mean
        # the *tag* is the thing that is wrong, not the candidate:
        # - A strong fingerprint, which identifies the audio directly.
        # - Agreeing with the filename even though it disagrees with the tag.
        #   A renamed file is rare; a wrong or scrambled tag is not, so a
        #   candidate the filename backs up should not be punished for a tag
        #   that is plausibly the actual mistake.
        strong_fingerprint = any(
            s.name == "fingerprint" and s.score >= STRONG_FINGERPRINT for s in signals)
        filename_backs_it_up = filename_title_similarity is not None and filename_title_similarity >= 0.8
        if (title_similarity is not None and title_similarity < 0.5
                and not strong_fingerprint and not filename_backs_it_up):
            raw *= 0.55
            cand.signals.append(Signal(
                "title", "Title disagrees strongly - likely a different track", 0.0, 0.0))

        return max(0.0, min(100.0, raw))

    def _finish(self, track: Track, observed: dict[str, Any], candidates: list[Candidate],
                notes: list[str], methods: list[str]) -> MatchResult:
        # Collapse duplicates first: scoring appends signals, so it must run
        # exactly once per surviving candidate.
        candidates = _dedupe_candidates(candidates)
        for cand in candidates:
            cand.confidence = round(self._score(cand, observed), 1)
        # Audio evidence outranks text evidence, always. An AcoustID hit says
        # "this audio *is* this recording"; a search hit only says "a recording
        # with this title exists". Ranking on confidence alone let a cover
        # version that happened to score a flat 1.00 on title and length beat
        # the fingerprinted original - whose own 0.98 fingerprint score pulled
        # its weighted average *down* relative to that perfect 1.00. The
        # strongest evidence available was penalised for not being certain,
        # which is exactly backwards.
        candidates.sort(
            key=lambda c: (_fingerprint_score(c) >= STRONG_FINGERPRINT, c.confidence),
            reverse=True)
        candidates = candidates[: self.cfg.max_candidates]

        result = MatchResult(candidates=candidates, notes=notes,
                             method=", ".join(dict.fromkeys(methods)) or "none")
        if not candidates:
            result.confidence = 0.0
            result.proposed = TrackTags()
            result.notes.append("No candidates found. Try enabling fingerprinting, "
                                "or fix the filename so a text search can work.")
            return result

        best = candidates[0]
        confidence = best.confidence

        # Ambiguity: a close runner-up means the top answer is less certain -
        # but only when it is genuinely a *different* answer. The same
        # recording reissued on a greatest-hits, and a cover version beaten by
        # a fingerprinted original, both used to trip this and quietly knock
        # 20% off a match that was not actually in doubt.
        if len(candidates) > 1:
            runner_up = candidates[1]
            margin = best.confidence - runner_up.confidence
            fingerprint_settles_it = (_fingerprint_score(best) >= STRONG_FINGERPRINT
                                      > _fingerprint_score(runner_up))
            if (margin < 12 and runner_up.raw_id != best.raw_id
                    and not fingerprint_settles_it
                    and not _same_answer(best, runner_up)):
                # How much doubt a near-tie deserves depends on what the two
                # candidates actually disagree about. "The Beatles" on Let It
                # Be versus the same performance on a compilation is a real
                # question, but a far smaller one than The Beatles versus
                # Aretha Franklin - the song is already settled, only the
                # release is open. Charging both the same penalty buried
                # correct matches in the review pile for no good reason.
                same_song = _same_recording(best, runner_up)
                floor = 0.95 if same_song else 0.80
                factor = floor + (margin / 12.0) * (1.0 - floor)
                confidence *= factor
                what = "which release this came from is unclear" if same_song \
                    else "confidence reduced for ambiguity"
                result.notes.append(
                    f"Second-best candidate scores {runner_up.confidence:.0f}% "
                    f"({runner_up.release_summary or 'unnamed'}) - {what}.")

        # Evidence ceiling: we cannot be sure of a guess we could not check.
        ceiling = _evidence_ceiling(observed, best)
        if confidence > ceiling:
            result.notes.append(
                f"Capped at {ceiling:.0f}% - too little existing information to verify the match.")
            confidence = ceiling

        result.confidence = round(max(0.0, min(100.0, confidence)), 1)
        result.chosen_index = 0
        # Only now, once one candidate has actually won, is it worth spending a
        # round trip on the album and track details needed to write tags.
        self._enrich(best)
        if not best.tags.album:
            # MusicBrainz holds some recordings on no release at all. That is a
            # dead end for a library filed by album, so swap to an equivalent
            # recording that is on one - same artist, same song, so this
            # changes which release gets written, not what the track is, and
            # the confidence worked out above still stands.
            for rank, alt in enumerate(candidates[1:ENRICH_FALLBACK_LIMIT], start=1):
                if not _same_recording(alt, best):
                    continue
                self._enrich(alt)
                if alt.tags.album:
                    best = alt
                    result.chosen_index = rank
                    break
        result.proposed = self._build_proposal(track, best)
        result.field_confidence = self._field_confidence(result.confidence, observed, best)
        return result

    def _build_proposal(self, track: Track, cand: Candidate) -> TrackTags:
        """Combine the candidate with what is already on the file."""
        proposed = TrackTags(**cand.tags.to_dict())

        # Genre never comes from our MusicBrainz queries, so keep the existing one.
        if not proposed.genre and track.current.genre:
            proposed.genre = track.current.genre
        if not proposed.composer and track.current.composer:
            proposed.composer = track.current.composer

        # Plex groups by album artist. Never leave it empty.
        if not proposed.album_artist:
            proposed.album_artist = proposed.artist or track.current.album_artist
        if proposed.compilation and proposed.mb_album_artist_id == VARIOUS_ARTISTS_MBID:
            proposed.album_artist = self.cfg.various_artists_name

        if not proposed.year and proposed.date:
            proposed.year = year_from_date(proposed.date)

        if self.cfg.preserve_existing_tags:
            proposed = track.current.merged_with(proposed, only_missing=True)
        return proposed

    def _field_confidence(self, overall: float, observed: dict[str, Any],
                          cand: Candidate) -> dict[str, float]:
        """Per-field confidence: overall score, adjusted for field reliability.

        A field that independently agrees with what was already on the file is
        worth more than one we are taking purely on the database's word.
        """
        out: dict[str, float] = {}
        agreement = {
            "title": observed["title"],
            "artist": observed["artist"],
            "album": observed["album"],
        }
        for field, reliability in FIELD_RELIABILITY.items():
            value = getattr(cand.tags, field, None)
            if value in (None, "", False):
                continue
            score = overall * reliability
            observed_value = agreement.get(field)
            if observed_value:
                sim = similarity(observed_value, str(value), drop_feat=True)
                if sim > 0.92:
                    # Two independent sources agree - pull towards certainty.
                    score = score + (100.0 - score) * 0.5
                elif sim < 0.5:
                    score *= 0.75
            if field in ("track_no", "track_total", "disc_no", "disc_total"):
                if observed.get("track_no") and cand.tags.track_no == observed["track_no"]:
                    score = score + (100.0 - score) * 0.4
            out[field] = round(max(0.0, min(100.0, score)), 1)
        return out

    # ------------------------------------------------------------------
    # album consolidation
    # ------------------------------------------------------------------
    def _consolidate_album(self, tracks: list[Track]) -> None:
        """Snap a folder of tracks onto the single release most of them chose."""
        matched = [t for t in tracks if t.match and t.match.candidates]
        if len(matched) < 3:
            return

        votes = Counter()
        for t in matched:
            rid = t.match.proposed.mb_release_id
            if rid:
                votes[rid] += 1
        if not votes:
            return

        release_id, count = votes.most_common(1)[0]
        share = count / len(matched)
        if share < 0.5:
            return

        try:
            tracklist = self.mb.full_release_tracklist(release_id)
        except ProviderError:
            return
        if not tracklist:
            return

        release_tags_sample = next(
            (t.match.proposed for t in matched if t.match.proposed.mb_release_id == release_id), None)
        if release_tags_sample is None:
            return

        assignment = _seat_on_tracklist(matched, tracklist)

        realigned = 0
        for index, track in enumerate(matched):
            if index not in assignment:
                continue
            entry = tracklist[assignment[index]]

            proposed = track.match.proposed
            already_here = proposed.mb_release_id == release_id
            proposed.album = release_tags_sample.album
            proposed.album_artist = release_tags_sample.album_artist
            proposed.mb_release_id = release_id
            proposed.mb_release_group_id = release_tags_sample.mb_release_group_id
            proposed.mb_album_artist_id = release_tags_sample.mb_album_artist_id
            proposed.compilation = release_tags_sample.compilation
            proposed.date = release_tags_sample.date
            proposed.year = release_tags_sample.year
            proposed.track_no = entry.get("track_no") or proposed.track_no
            proposed.track_total = entry.get("track_total") or proposed.track_total
            proposed.disc_no = entry.get("disc_no") or proposed.disc_no
            proposed.disc_total = entry.get("disc_total") or proposed.disc_total
            if entry.get("recording", {}).get("id"):
                proposed.mb_recording_id = entry["recording"]["id"]

            match = track.match
            if not already_here:
                realigned += 1
                match.notes.append(
                    f"Re-seated onto the release chosen by {count} of {len(matched)} tracks "
                    "in this folder, so the album stays together in Plex.")
            # Agreement across a whole folder is real evidence: raise the floor,
            # but never claim certainty a single track did not earn.
            boost = 8.0 * share
            match.confidence = round(min(97.0, match.confidence + boost), 1)
            match.field_confidence["album"] = round(min(98.0, max(
                match.field_confidence.get("album", 0), 70 + 25 * share)), 1)
            match.field_confidence["album_artist"] = match.field_confidence["album"]
            match.notes.append(
                f"Album agreement: {count}/{len(matched)} tracks in this folder resolved "
                f"to the same MusicBrainz release.")

        if realigned:
            log.info("Album consolidation moved %d tracks onto release %s", realigned, release_id)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Below this title similarity a file is never seated on a tracklist entry,
#: however well its length and position line up. Unrelated song titles
#: routinely score 0.2-0.45 ("Sour Times" vs "Strangers" is 0.42), and an
#: album has plenty of tracks within 3s of each other - so without a floor a
#: bonus track from another release would inherit a neighbour's recording ID.
SEAT_TITLE_FLOOR = 0.6
#: The combined score a seat needs once the title has cleared the floor.
SEAT_MIN_SCORE = 0.55


def _seat_on_tracklist(tracks: list[Track], tracklist: list[dict[str, Any]]) -> dict[int, int]:
    """Pair files with tracklist entries: ``{track index: entry index}``.

    Scored across the whole folder and assigned best-first, not in file
    order: taking each file's best free entry in turn let an early file claim
    the slot that belonged to a later one, which then had nowhere to go.

    A file whose proposal already names a recording on this release is
    seated there outright - that is identity, not resemblance.
    """
    pairs: list[tuple[float, int, int]] = []
    for ti, track in enumerate(tracks):
        rec_id = track.match.proposed.mb_recording_id
        title = track.match.proposed.title or track.current.title or Path(track.path).stem
        for ei, entry in enumerate(tracklist):
            if rec_id and (entry.get("recording") or {}).get("id") == rec_id:
                pairs.append((2.0, ti, ei))
                continue
            similar = similarity(title, entry["title"], drop_feat=True)
            if similar < SEAT_TITLE_FLOOR:
                continue
            score = similar
            if entry.get("length_ms") and track.props.duration_s:
                delta = abs(entry["length_ms"] / 1000.0 - track.props.duration_s)
                score = score * 0.75 + (0.25 if delta <= 3 else 0.0)
            if track.current.track_no and entry.get("track_no") == track.current.track_no:
                score += 0.15
            if score >= SEAT_MIN_SCORE:
                pairs.append((score, ti, ei))

    # Ties fall back to folder order, so the result never depends on luck.
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    seated: dict[int, int] = {}
    taken: set[int] = set()
    for _score, ti, ei in pairs:
        if ti in seated or ei in taken:
            continue
        seated[ti] = ei
        taken.add(ei)
    return seated


def _credited_to(recording: dict[str, Any], cluster: "Counter[str]",
                 most_common: int) -> bool:
    """Is this recording by the artist credited most across the cluster?"""
    if not most_common:
        return False
    return any(cluster.get(normalize(a.get("name"), drop_articles=True), 0) == most_common
               for a in (recording.get("artists") or []))


def _duration_gap(recording: dict[str, Any], observed_s: Optional[float]) -> float:
    """How far this recording's length is from the file's, for ordering.

    Recordings AcoustID has no length for sort last: unknown is not the same
    as matching, and guessing otherwise would let a blank entry outrank a
    recording whose length actually agrees.
    """
    seconds = recording.get("duration")
    if not seconds or not observed_s:
        return float("inf")
    return abs(float(seconds) - float(observed_s))


def _acoustid_to_mb(recording: dict[str, Any]) -> dict[str, Any]:
    """Reshape an AcoustID recording into the shape MusicBrainz returns.

    AcoustID gives ``{"id", "title", "artists": [{"id", "name"}], "duration"}``
    where MusicBrainz gives ``artist-credit`` and a ``length`` in milliseconds.
    Translating here means :meth:`MusicBrainzClient.recording_to_tags` reads
    both without knowing the difference - and means a fingerprint candidate
    costs no network call at all to build.
    """
    out: dict[str, Any] = {"id": recording.get("id"), "title": recording.get("title")}
    artists = recording.get("artists") or []
    if artists:
        out["artist-credit"] = [
            {"name": a.get("name"), "artist": {"id": a.get("id"), "name": a.get("name")},
             # AcoustID does not report how names were joined; assume the usual
             # "A & B" rather than inventing a phrase per artist.
             "joinphrase": " & " if i < len(artists) - 1 else ""}
            for i, a in enumerate(artists)
        ]
    seconds = recording.get("duration")
    if seconds:
        out["length"] = float(seconds) * 1000.0
    return out


def _has_strong_fingerprint(candidates: Iterable[Candidate]) -> bool:
    """Has the audio itself already been identified beyond reasonable doubt?"""
    return any(_fingerprint_score(c) >= STRONG_FINGERPRINT for c in candidates)


def _fingerprint_score(cand: Candidate) -> float:
    """The candidate's AcoustID score, or 0 if it was not fingerprint-derived."""
    fp = next((s for s in cand.signals if s.name == "fingerprint"), None)
    return fp.score if fp else 0.0


def _same_answer(a: Candidate, b: Candidate) -> bool:
    """Would applying either candidate write the same tags?

    MusicBrainz frequently holds the same track as several recordings that
    are indistinguishable once written to a file. A near-tie between two of
    those is not ambiguity - whichever wins, the user gets the same result -
    so it should not cost any confidence.

    Compared literally rather than with :func:`similarity`, because the
    question is what actually gets written to the file, not whether two
    strings mean the same thing. Normalisation deliberately treats "Dummy"
    and "Dummy (Remastered)" as one name for *matching* - but they are two
    different album tags, they file into two different Plex folders, and
    choosing between them is a real decision.
    """
    def same(x: Optional[str], y: Optional[str]) -> bool:
        return (x or "").strip().casefold() == (y or "").strip().casefold()

    return (same(a.tags.title, b.tags.title)
            and same(a.tags.artist, b.tags.artist)
            and same(a.tags.album, b.tags.album))


def _same_recording(a: Candidate, b: Candidate) -> bool:
    """Same song by the same artist, whatever release each one came from.

    Unlike :func:`_same_answer` this compares loosely, because the question
    is "is this the same song?" rather than "would the tags come out byte for
    byte identical?" - so "Jay-Z" and "JAY-Z" count as one artist.
    """
    return (similarity(a.tags.title, b.tags.title, drop_feat=True) >= 0.9
            and similarity(a.tags.artist, b.tags.artist,
                           drop_feat=True, drop_articles=True) >= 0.9)


#: "08 The Reason" -> "The Reason". Two digits or fewer, then whitespace, then
#: something that starts with a letter - so "1979" and "99 Luftballons" are
#: left alone by the shape alone, and the caller only uses this as a fallback.
_LEADING_TRACK_NUMBER = re.compile(r"^\d{1,2}\s+(?P<title>[^\W\d].*)$")


def _strip_leading_track_number(title: str) -> Optional[str]:
    """The title with a leading track number removed, or None if there is none."""
    m = _LEADING_TRACK_NUMBER.match(title.strip())
    return m.group("title").strip() if m else None


def _length_s(recording: dict[str, Any] | None) -> Optional[float]:
    """MusicBrainz stores recording length in milliseconds, and often omits it."""
    if not recording:
        return None
    ms = recording.get("length")
    try:
        return float(ms) / 1000.0 if ms else None
    except (TypeError, ValueError):
        return None


def _dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Collapse candidates that are the same recording, keeping the best signals."""
    by_id: dict[str, Candidate] = {}
    out: list[Candidate] = []
    for cand in candidates:
        key = cand.raw_id or f"{cand.tags.title}|{cand.tags.artist}|{cand.tags.album}"
        existing = by_id.get(key)
        if existing is None:
            by_id[key] = cand
            out.append(cand)
            continue
        # Merge: keep whichever signals are stronger, and remember it was
        # corroborated by more than one source.
        have = {s.name for s in existing.signals}
        for sig in cand.signals:
            if sig.name not in have:
                existing.signals.append(sig)
        if existing.length_s is None:
            existing.length_s = cand.length_s
        # Corroboration from a precise search clears the caveat; if every source
        # needed a widened search, the caveat stands.
        existing.duration_unverified = existing.duration_unverified and cand.duration_unverified
        if cand.source != existing.source:
            existing.source = f"{existing.source}+{cand.source}"
    return out


#: Ceiling for a match found only after the duration filter was dropped. Sits
#: below the default auto-apply threshold so such matches always get a human
#: look, which is what the accompanying note tells the user to do.
UNVERIFIED_DURATION_CEILING = 75.0


def _evidence_ceiling(observed: dict[str, Any], best: Candidate) -> float:
    """The most confidence the available evidence can justify.

    A fingerprint hit is direct evidence about the audio, so it lifts the
    ceiling on its own. Without one, we are only ever comparing text we were
    already given - which cannot exceed the quality of that text.
    """
    fp = next((s for s in best.signals if s.name == "fingerprint"), None)
    if fp and fp.score >= 0.9:
        return 99.0
    if fp and fp.score >= 0.6:
        return 95.0

    if best.source == "existing-mbid":
        return 96.0
    if best.source == "filename":
        return 45.0

    if best.duration_unverified:
        # We asked for a recording of this length and the database had none.
        # Whether our length is wrong or the match is wrong, it is not settled,
        # and a "Confident" badge would contradict the warning we just printed.
        return UNVERIFIED_DURATION_CEILING

    evidence = observed["evidence"]
    if not observed["from_tags"]:
        return 60.0          # filename-derived text checked against a database
    if evidence >= 6:
        return 93.0
    if evidence >= 4:
        return 85.0
    return 72.0
