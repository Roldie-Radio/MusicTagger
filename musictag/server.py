"""FastAPI backend for the desktop UI.

Everything long-running goes through :mod:`musictag.jobs` and is polled, so the
window stays responsive while a 40,000 file library is scanned.
"""

from __future__ import annotations

import logging
import os
import string
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .apply import Applier, ApplyOptions
from .cache import http_cache
from .config import APP_DIR, get_config, set_config
from .convert import FORMATS as CONVERT_FORMATS, convert_tracks
from .export_plex import commit_export, plan_export
from .fingerprint import FPCALC_HOMEPAGE, fpcalc_download_url, install_fpcalc
from .jobs import EDIT_BLOCKING_JOBS, LIBRARY_JOBS, Job, JobConflict, get_jobs
from .journal import get_journal
from .library import load_track, scan as scan_library
from .matching import Matcher
from .models import Track, TrackTags, quality_scale
from .organize import plan_all, plan_path
from .quality import analyze_track
from .state import get_state
from .tags import SUPPORTED_EXTENSIONS, read_embedded_art
from .update import check_for_update

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="MusicTagger", version=__version__, docs_url=None, redoc_url=None)


# ===========================================================================
# Local-only guard
# ===========================================================================

#: Hostnames the UI legitimately reaches this server by. Binding to 127.0.0.1
#: stops other machines connecting, but not other *websites*: a page can point
#: its own domain at 127.0.0.1 (DNS rebinding) and its requests then arrive
#: here same-origin, able to read folders, move files, or set ``ffmpeg_path``
#: to any program and have the next quality scan run it. Such a request still
#: names the attacker's domain in its Host header, which is what this checks.
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _hostname(value: str) -> str:
    """``host[:port]`` -> ``host``, lowercased; IPv6 brackets kept intact."""
    value = value.strip().lower()
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if ":" in value else value


@app.middleware("http")
async def local_only(request: Request, call_next):
    if _hostname(request.headers.get("host", "")) not in ALLOWED_HOSTS:
        return JSONResponse({"error": "Forbidden host."}, status_code=403)
    # A cross-site form POST carries no JSON, but endpoints with no body
    # (clear, cancel, install-fpcalc) would still run. Browsers always send
    # Origin on such requests, so refuse any that come from another site.
    if request.method not in _SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin and (origin == "null"
                       or _hostname(urlsplit(origin).netloc) not in ALLOWED_HOSTS):
            return JSONResponse({"error": "Cross-site request refused."}, status_code=403)
    return await call_next(request)


# ===========================================================================
# Request bodies
# ===========================================================================

class ScanRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    recursive: bool = True
    replace: bool = False


class SelectionRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    only_pending: bool = True


class ApplyRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    organize: bool = False
    dry_run: bool = False
    write_art: bool = True
    min_confidence: Optional[float] = None


class ChooseRequest(BaseModel):
    path: str
    candidate_index: int


class EditRequest(BaseModel):
    path: str
    tags: dict[str, Any]


class ConvertRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    format: str = "mp3"


class UndoRequest(BaseModel):
    batch_id: str


class ExportPlanRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class ExportCommitItem(BaseModel):
    path: str
    dest: str
    action: str = "export"                    # export | replace | skip
    existing_path: Optional[str] = None


class ExportCommitRequest(BaseModel):
    items: list[ExportCommitItem] = Field(default_factory=list)


# ===========================================================================
# Status, config, capabilities
# ===========================================================================

@app.get("/api/status")
def api_status() -> dict[str, Any]:
    cfg = get_config()
    state = get_state()
    jobs = get_jobs()
    return {
        "version": __version__,
        "app_dir": str(APP_DIR),
        "stats": state.stats(),
        "capabilities": cfg.capabilities(),
        "active_jobs": [j.to_dict() for j in jobs.active()],
        "supported_formats": sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
        # So the UI can explain a quality badge without hardcoding the numbers.
        "quality_scale": quality_scale(),
    }


@app.get("/api/update")
def api_update(force: bool = False) -> dict[str, Any]:
    """Is there a newer release? Deliberately not part of /api/status.

    /api/status is polled while jobs run, and is expected to answer instantly
    from local state. This one can go out to the network and block for a few
    seconds on a cold cache, so it stays a separate call the UI makes once,
    after the page is already usable.
    """
    return check_for_update(get_config(), force=force).to_dict()


@app.get("/api/config")
def api_get_config() -> dict[str, Any]:
    cfg = get_config()
    data = cfg.to_dict()
    # Never round-trip the full key to the UI; show only enough to confirm it is set.
    key = data.get("acoustid_api_key") or ""
    data["acoustid_api_key"] = ("*" * max(0, len(key) - 3) + key[-3:]) if key else ""
    data["_capabilities"] = cfg.capabilities()
    data["_fpcalc_url"] = fpcalc_download_url()
    data["_fpcalc_homepage"] = FPCALC_HOMEPAGE
    return data


@app.post("/api/config")
def api_set_config(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = get_config()
    # A masked key means "unchanged" - do not overwrite the real one with asterisks.
    if isinstance(payload.get("acoustid_api_key"), str) and set(payload["acoustid_api_key"]) <= {"*"} | set(
            payload["acoustid_api_key"][-3:]) and "*" in payload["acoustid_api_key"]:
        payload.pop("acoustid_api_key")
    cfg.update(payload)
    cfg.save()
    set_config(cfg)
    return api_get_config()


@app.post("/api/install-fpcalc")
def api_install_fpcalc() -> dict[str, Any]:
    """Download Chromaprint's fpcalc. Only ever reached by an explicit click."""
    jobs = get_jobs()

    def run(job: Job):
        job.log(f"Downloading fpcalc from {fpcalc_download_url()}")
        path = install_fpcalc(progress=job.log)
        cfg = get_config()
        cfg.fpcalc_path = path
        cfg.save()
        return {"path": path}

    job = jobs.submit("install-fpcalc", run, message="Installing fpcalc")
    return job.to_dict()


# ===========================================================================
# Filesystem browsing (for the folder picker)
# ===========================================================================

@app.get("/api/browse")
def api_browse(path: str = "") -> dict[str, Any]:
    """List drives and subdirectories so the UI can offer a folder picker."""
    if not path:
        if os.name == "nt":
            roots = []
            for letter in string.ascii_uppercase:
                try:
                    if Path(f"{letter}:\\").exists():
                        roots.append(f"{letter}:\\")
                except OSError:
                    # A mapped network drive with stale/expired credentials
                    # raises instead of just reporting "not found" - skip it
                    # rather than taking down the whole folder picker.
                    continue
        else:
            roots = ["/"]
        home = str(Path.home())
        return {"path": "", "parent": None,
                "entries": [{"name": r, "path": r, "kind": "drive"} for r in roots]
                           + [{"name": "Home", "path": home, "kind": "dir"}]}

    target = Path(path)
    if not target.is_dir():
        raise HTTPException(404, f"Not a directory: {path}")

    entries = []
    audio_count = 0
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                if child.is_dir():
                    entries.append({"name": child.name, "path": str(child), "kind": "dir"})
                elif child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    audio_count += 1
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")

    parent = str(target.parent) if target.parent != target else ""
    return {"path": str(target), "parent": parent,
            "audio_files": audio_count, "entries": entries[:500]}


# ===========================================================================
# Scanning
# ===========================================================================

@app.post("/api/scan")
def api_scan(req: ScanRequest) -> dict[str, Any]:
    cfg = get_config()
    paths = req.paths or cfg.library_paths
    if not paths:
        raise HTTPException(400, "No library folder selected.")

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise HTTPException(400, f"Path does not exist: {missing[0]}")

    # Remember the folder for next time - but only when the caller explicitly
    # chose one. An explicit selection *replaces* whatever was remembered
    # before: the app tracks one current library folder, not a growing list
    # of every folder ever pointed at. A plain re-scan (paths=[], falling
    # back to cfg.library_paths above) must not re-append what is already
    # there, which is why this only fires when req.paths was actually given.
    if req.paths:
        cfg.library_paths = list(dict.fromkeys(req.paths))
        cfg.save()

    state = get_state()

    def run(job: Job):
        # Inside the job, not before submitting it: a scan refused because
        # another job is running must not have wiped the library first.
        if req.replace:
            state.clear()
        job.log(f"Scanning {len(paths)} folder(s)")
        tracks = scan_library(
            paths, recursive=req.recursive, workers=cfg.scan_workers,
            progress=lambda done, total, path: job.progress(done, total, path),
            cancelled=lambda: job.cancelled,
        )
        added = state.add(tracks)
        state.persist(tracks)
        job.log(f"Found {len(tracks)} audio files ({added} new)")
        return {"found": len(tracks), "added": added}

    return get_jobs().submit("scan", run, message="Scanning library").to_dict()


# ===========================================================================
# Identification
# ===========================================================================

@app.post("/api/identify")
def api_identify(req: SelectionRequest) -> dict[str, Any]:
    cfg = get_config()
    state = get_state()
    tracks = state.select(req.paths or None, only_unidentified=req.only_pending and not req.paths)
    if not tracks:
        raise HTTPException(400, "Nothing to identify. Scan a folder first, "
                                 "or clear the 'only pending' option.")

    def run(job: Job):
        matcher = Matcher(cfg)
        # Group by folder: album-level agreement is what keeps Plex from
        # splitting one album into several.
        groups: dict[str, list[Track]] = {}
        for track in tracks:
            groups.setdefault(str(Path(track.path).parent), []).append(track)

        job.total = len(tracks)
        progress_lock = threading.Lock()
        completed = 0

        def do_group(items: list[Track]):
            if job.cancelled:
                return

            def on_track(_done_in_group: int, _total_in_group: int, path: str) -> None:
                nonlocal completed
                with progress_lock:
                    completed += 1
                    job.progress(completed, len(tracks), Path(path).name)

            matcher.identify_album(items, progress=on_track,
                                   cancelled=lambda: job.cancelled)
            # Save each album as it finishes: a crash or a closed window an
            # hour into a big library should cost one album, not the run.
            state.persist(items)

        workers = max(1, min(cfg.identify_workers, len(groups)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(do_group, groups.values()))
        identified = sum(1 for t in tracks if t.match and t.match.candidates)
        job.log(f"Identified {identified} of {len(tracks)} tracks")
        return {"identified": identified, "total": len(tracks)}

    return get_jobs().submit("identify", run, message="Tagging tracks").to_dict()


# ===========================================================================
# Quality analysis
# ===========================================================================

@app.post("/api/quality")
def api_quality(req: SelectionRequest) -> dict[str, Any]:
    cfg = get_config()
    if not cfg.ffmpeg:
        raise HTTPException(400, "ffmpeg was not found. Install it, or set its path in Settings.")

    state = get_state()
    tracks = state.select(req.paths or None, only_unanalysed=req.only_pending and not req.paths)
    if not tracks:
        raise HTTPException(400, "Nothing to analyse.")

    def run(job: Job):
        job.total = len(tracks)
        done = 0
        progress_lock = threading.Lock()

        def analyse(track: Track):
            nonlocal done
            if job.cancelled:
                return
            try:
                track.quality = analyze_track(track, cfg)
            except Exception as exc:  # noqa: BLE001
                log.exception("Quality analysis failed for %s", track.path)
                from .models import QualityReport
                track.quality = QualityReport(analysed=False, error=str(exc))
            # Decoding a file takes seconds, so saving each result as it lands
            # costs nothing and means an interrupted run keeps its work.
            state.persist([track])
            # ``done += 1`` is a read-modify-write; unguarded, workers
            # finishing together lose counts and the final tally comes up short.
            with progress_lock:
                done += 1
                job.progress(done, len(tracks), track.filename)

        with ThreadPoolExecutor(max_workers=max(1, cfg.quality_workers)) as pool:
            list(pool.map(analyse, tracks))
        flagged = sum(1 for t in tracks if t.quality and t.quality.issues)
        job.log(f"Analysed {done} files, {flagged} with findings")
        return {"analysed": done, "flagged": flagged}

    return get_jobs().submit("quality", run, message="Checking quality").to_dict()


# ===========================================================================
# Applying
# ===========================================================================

def _applyable(track: Track) -> bool:
    """Has something worth writing - either identified, or hand-edited.

    Gating on ``match.candidates`` would exclude a file the user typed tags
    into by hand without ever running Tag: :func:`api_edit` gives it a
    manual :class:`MatchResult` with an empty candidate list, since there was
    never a lookup to record candidates from.
    """
    return bool(track.match and track.match.proposed and not track.match.proposed.is_empty())


@app.post("/api/apply")
def api_apply(req: ApplyRequest) -> dict[str, Any]:
    cfg = get_config()
    state = get_state()
    tracks = [t for t in state.select(req.paths or None) if _applyable(t)]
    if not tracks:
        raise HTTPException(400, "No identified or hand-edited tracks selected.")

    options = ApplyOptions(
        write_tags=True,
        write_art=req.write_art,
        organize=req.organize,
        dry_run=req.dry_run,
        min_confidence=req.min_confidence,
    )
    if req.organize and not cfg.organize_enabled:
        raise HTTPException(400, "Organising is switched off in Settings.")

    def run(job: Job):
        applier = Applier(cfg)
        originals = {id(t): t.path for t in tracks}
        report = applier.apply(
            tracks, options,
            progress=lambda done, total, path: job.progress(done, total, Path(path).name),
            cancelled=lambda: job.cancelled,
        )
        for track in tracks:
            old = originals[id(track)]
            if old != track.path:
                state.replace_path(old, track)
        state.persist(tracks)
        verb = "Would apply" if req.dry_run else "Applied"
        job.log(f"{verb} {report.tagged} tag writes, {report.moved} moves, {report.copied} copies")
        return report.to_dict()

    kind = "preview" if req.dry_run else "apply"
    return get_jobs().submit(kind, run, message="Writing tags").to_dict()


@app.post("/api/preview-organize")
def api_preview_organize(req: SelectionRequest) -> dict[str, Any]:
    """Show where files would land, without touching anything."""
    cfg = get_config()
    state = get_state()
    tracks = [t for t in state.select(req.paths or None) if _applyable(t)]
    planned = plan_all(tracks, cfg)
    moves = [{"from": src, "to": dest} for src, dest in planned.items() if src != dest]
    return {"count": len(moves), "moves": moves[:500],
            "unchanged": len(planned) - len(moves)}


# ===========================================================================
# Converting to another format
# ===========================================================================

@app.get("/api/convert/formats")
def api_convert_formats() -> dict[str, Any]:
    return {"formats": [{"id": key, "label": spec["label"]}
                        for key, spec in CONVERT_FORMATS.items()]}


@app.post("/api/convert")
def api_convert(req: ConvertRequest) -> dict[str, Any]:
    """Convert selected tracks. New files go beside the originals."""
    cfg = get_config()
    if req.format not in CONVERT_FORMATS:
        raise HTTPException(400, f"Unknown format: {req.format}")
    if not cfg.ffmpeg:
        raise HTTPException(400, "ffmpeg was not found. Install it, or set its path in Settings.")
    state = get_state()
    tracks = state.select(req.paths or None)
    if not req.paths or not tracks:
        raise HTTPException(400, "Select the tracks to convert first.")

    def run(job: Job):
        report = convert_tracks(
            [t.path for t in tracks], req.format, cfg,
            progress=lambda done, total, name: job.progress(done, total, name),
            cancelled=lambda: job.cancelled,
        )
        # The new files join the list straight away, so they can be checked,
        # tagged or exported like any other.
        created = [load_track(Path(p)) for p in report.created]
        state.add(created)
        state.persist(created)
        job.log(f"Converted {report.converted}, skipped {report.skipped}, "
                f"failed {report.failed}")
        return report.to_dict()

    label = CONVERT_FORMATS[req.format]["label"]
    return get_jobs().submit("convert", run, message=f"Converting to {label}").to_dict()


# ===========================================================================
# Export to Plex - moving reviewed tracks out of the ingest folder
# ===========================================================================

@app.post("/api/export/plan")
def api_export_plan(req: ExportPlanRequest) -> dict[str, Any]:
    """Where would selected tracks land in the Plex folder, and what already looks like it's there?

    Read-only. Runs as a job because indexing a large existing Plex library
    (tag reads only, no network) is the same cost as a Scan.
    """
    cfg = get_config()
    state = get_state()
    tracks = state.select(req.paths or None)
    if not tracks:
        raise HTTPException(400, "Nothing selected to export.")
    if not cfg.organize_root:
        raise HTTPException(400, "No Plex Music folder configured. Set one in Settings first.")

    def run(job: Job):
        job.log("Reading what's already in the Plex folder")
        plan = plan_export(
            tracks, cfg,
            progress=lambda done, total, name: job.progress(done, total, name),
            cancelled=lambda: job.cancelled,
        )
        if job.cancelled:
            # Half an index would miss duplicates, so there is nothing
            # trustworthy to review. The UI only opens a plan from a job
            # that finished as "done".
            job.log("Export cancelled")
            return None
        dupes = sum(1 for i in plan.items if i.duplicate)
        job.log(f"{len(plan.items)} track(s) planned"
                + (f", {dupes} possible duplicate(s) to resolve" if dupes else ""))
        return plan.to_dict()

    return get_jobs().submit("export-plan", run, message="Scanning Plex folder").to_dict()


@app.post("/api/export/commit")
def api_export_commit(req: ExportCommitRequest) -> dict[str, Any]:
    """Actually move the files, per the per-item resolutions the UI collected."""
    if not req.items:
        raise HTTPException(400, "Nothing to export.")
    cfg = get_config()
    state = get_state()
    items = [i.model_dump() for i in req.items]

    def run(job: Job):
        report = commit_export(
            items, cfg,
            progress=lambda done, total, name: job.progress(done, total, name),
            cancelled=lambda: job.cancelled,
        )
        # Only forget what actually left: a failed or skipped item is still
        # sitting in the ingest folder and must stay visible in the app.
        state.remove(report.exported_paths)
        job.log(f"Exported {report.exported}, replaced {report.replaced}, "
                f"skipped {report.skipped}, failed {report.failed}")
        return report.to_dict()

    return get_jobs().submit("export-commit", run, message="Exporting to Plex").to_dict()


# ===========================================================================
# Track-level edits
# ===========================================================================

@app.get("/api/tracks")
def api_tracks(q: str = "", bucket: str = "", status: str = "", issues: str = "",
               fmt: str = "", sort: str = "path", desc: bool = False,
               offset: int = 0, limit: int = Query(200, le=1000)) -> dict[str, Any]:
    return get_state().filtered(query=q, bucket=bucket, status=status, issues=issues,
                                fmt=fmt, sort=sort, desc=desc,
                                offset=offset, limit=limit)


@app.get("/api/columns")
def api_columns() -> dict[str, Any]:
    """Which fields the grid can show and sort by."""
    from .state import FILE_SORT_FIELDS, TAG_SORT_FIELDS
    return {"tag_fields": list(TAG_SORT_FIELDS),
            "file_fields": sorted(FILE_SORT_FIELDS)}


@app.get("/api/track")
def api_track(path: str) -> dict[str, Any]:
    track = get_state().get(path)
    if not track:
        raise HTTPException(404, "Track not in the current scan.")
    cfg = get_config()
    payload = track.to_dict()
    if track.match:
        payload["planned_path"] = str(plan_path(track, track.match.proposed, cfg)) \
            if cfg.organize_enabled else None
    return payload


def _refuse_edit_while_busy() -> None:
    """A hand edit made mid-Identify or mid-Apply would be lost or half-written."""
    busy = get_jobs().running_any(EDIT_BLOCKING_JOBS)
    if busy:
        raise HTTPException(409, str(JobConflict(busy)))


@app.post("/api/track/choose")
def api_choose(req: ChooseRequest) -> dict[str, Any]:
    """Pick a different candidate for a track."""
    _refuse_edit_while_busy()
    state = get_state()
    track = state.get(req.path)
    if not track or not track.match:
        raise HTTPException(404, "Track has no match to change.")
    if not 0 <= req.candidate_index < len(track.match.candidates):
        raise HTTPException(400, "No such candidate.")

    cfg = get_config()
    matcher = Matcher(cfg)
    chosen = track.match.candidates[req.candidate_index]
    track.match.chosen_index = req.candidate_index
    track.match.confidence = chosen.confidence
    # Fingerprint candidates carry only what AcoustID returned until one is
    # picked, so a runner-up needs its album and track details fetched now -
    # otherwise choosing it would write tags with the album missing.
    matcher._enrich(chosen)
    track.match.proposed = matcher._build_proposal(track, chosen)
    # A human picked it, so the fields are as good as the candidate's own score.
    track.match.field_confidence = {k: chosen.confidence for k in track.match.field_confidence}
    state.persist([track])
    return track.to_dict()


@app.post("/api/track/edit")
def api_edit(req: EditRequest) -> dict[str, Any]:
    """Hand-edit the proposed tags for one track."""
    _refuse_edit_while_busy()
    state = get_state()
    track = state.get(req.path)
    if not track:
        raise HTTPException(404, "Track not in the current scan.")
    if not track.match:
        from .models import MatchResult
        track.match = MatchResult(method="manual", confidence=100.0)
        track.match.proposed = TrackTags.from_dict(track.current.to_dict())

    current = track.match.proposed.to_dict()
    current.update({k: v for k, v in req.tags.items() if k in current})
    track.match.proposed = TrackTags.from_dict(current)
    for key in req.tags:
        track.match.field_confidence[key] = 100.0
    track.match.notes.append("Edited by hand.")
    state.persist([track])
    return track.to_dict()


@app.get("/api/art")
def api_art(path: str):
    """Serve a file's embedded cover art, for the detail panel."""
    target = Path(path)
    if not target.exists():
        raise HTTPException(404, "File not found")
    art = read_embedded_art(target)
    if not art:
        raise HTTPException(404, "No embedded art")
    data, mime = art
    return Response(content=data, media_type=mime,
                    headers={"Cache-Control": "private, max-age=300"})


# ===========================================================================
# Jobs and history
# ===========================================================================

@app.get("/api/jobs")
def api_jobs() -> dict[str, Any]:
    return {"jobs": get_jobs().list()}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict[str, Any]:
    job = get_jobs().get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str) -> dict[str, Any]:
    return {"cancelled": get_jobs().cancel(job_id)}


@app.get("/api/history")
def api_history() -> dict[str, Any]:
    return {"batches": get_journal().list_batches()}


@app.get("/api/history/{batch_id}")
def api_history_detail(batch_id: str) -> dict[str, Any]:
    return {"entries": get_journal().batch_entries(batch_id)}


@app.post("/api/undo")
def api_undo(req: UndoRequest) -> dict[str, Any]:
    cfg = get_config()
    journal = get_journal()
    if journal.is_undone(req.batch_id):
        raise HTTPException(409, "This change has already been undone.")

    def run(job: Job):
        job.log(f"Undoing batch {req.batch_id}")
        result = journal.undo(req.batch_id, id3v2_version=cfg.id3v2_version)
        state = get_state()
        state.remove_missing()
        return {"restored": result.restored, "skipped": result.skipped,
                "failed": result.failed, "messages": result.messages[:100]}

    return get_jobs().submit("undo", run, message="Undoing changes").to_dict()


@app.post("/api/clear")
def api_clear() -> dict[str, Any]:
    # A job still working on these tracks would write them straight back,
    # so the list would look cleared and then reappear.
    busy = get_jobs().running_any(LIBRARY_JOBS)
    if busy:
        raise HTTPException(409, str(JobConflict(busy)))
    get_state().clear()
    return {"ok": True}


@app.get("/api/lookup-cache")
def api_lookup_cache_info() -> dict[str, Any]:
    return {"entries": http_cache().count()}


@app.post("/api/lookup-cache/clear")
def api_lookup_cache_clear() -> dict[str, Any]:
    """Forget every cached MusicBrainz/AcoustID/cover-art response.

    Mainly a manual escape hatch: a negative result (real or, before the 404
    retry fix, spurious) is cached for up to 30 days. If a track's confidence
    looks wrong because of a lookup that should not have failed, clearing the
    cache and re-running Tag forces every lookup fresh rather than
    waiting out the TTL.
    """
    before = http_cache().count()
    http_cache().clear()
    return {"cleared": before}


# ===========================================================================
# Static UI
# ===========================================================================

class NoCacheStatic(StaticFiles):
    """Serve the UI without caching.

    Everything is local, so caching buys nothing and costs a whole class of
    bug: after an update the browser keeps serving the old stylesheet and the
    app looks broken in ways that do not reproduce.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.exception_handler(HTTPException)
def http_error(request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(JobConflict)
def job_conflict(request, exc: JobConflict):
    """Every job endpoint refuses the same way: 409, with what to wait for."""
    return JSONResponse({"error": str(exc), "running": exc.running.to_dict()},
                        status_code=409)


if WEB_DIR.exists():
    app.mount("/static", NoCacheStatic(directory=WEB_DIR), name="static")
