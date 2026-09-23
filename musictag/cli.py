"""Command line entry point.

``musictag ui`` is the normal way in - it starts the local server and opens the
window. The other subcommands do the same work headlessly, which is what you
want for a scheduled re-scan of a growing library.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__
from .config import APP_DIR, get_config
from .util import format_duration

log = logging.getLogger("musictag")


# ===========================================================================
# UI
# ===========================================================================

def _free_port(preferred: int = 8731) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port available")


def cmd_ui(args) -> int:
    import uvicorn
    from .server import app

    port = args.port or _free_port()
    url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the socket to accept before opening a window at it.
    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)

    print(f"MusicTagger {__version__} running at {url}")
    print(f"Settings and cache live in {APP_DIR}")

    if args.no_window:
        print("Press Ctrl+C to stop.")
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            pass
        return 0

    try:
        import webview  # pywebview, optional
        window = webview.create_window("MusicTagger", url, width=1280, height=860,
                                       min_size=(900, 600))
        webview.start()
        _ = window
    except ImportError:
        print("pywebview not installed - opening in your browser instead.")
        webbrowser.open(url)
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


# ===========================================================================
# Headless commands
# ===========================================================================

def _progress(label: str):
    last = [0.0]

    def report(done: int, total: int, detail: str = ""):
        now = time.time()
        if now - last[0] < 0.2 and done != total:
            return
        last[0] = now
        pct = (done / total * 100) if total else 0
        sys.stdout.write(f"\r{label}: {done}/{total} ({pct:.0f}%)   ")
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")
    return report


def cmd_scan(args) -> int:
    from .library import scan
    from .state import get_state

    cfg = get_config()
    paths = args.paths or cfg.library_paths
    if not paths:
        print("No paths given and no library configured. Try: musictag scan \"D:\\Music\"")
        return 2

    tracks = scan(paths, workers=cfg.scan_workers, progress=_progress("Scanning"))
    state = get_state()
    added = state.add(tracks)
    state.persist(tracks)

    # An explicit path on the command line replaces the remembered library,
    # matching the app: one current folder, not an accumulating list. Bare
    # `musictag scan` (falling back to cfg.library_paths above) leaves it as-is.
    if args.paths:
        # Stored absolute, like the tracks, so it still means the same folder
        # when the next command runs from a different directory.
        cfg.library_paths = list(dict.fromkeys(os.path.abspath(p) for p in args.paths))
        cfg.save()

    print(f"Found {len(tracks)} audio files ({added} new).")
    return 0


def cmd_identify(args) -> int:
    from .matching import Matcher
    from .state import get_state

    cfg = get_config()
    state = get_state()
    tracks = state.select(only_unidentified=not args.all)
    if not tracks:
        print("Nothing to identify. Run 'musictag scan' first.")
        return 0

    caps = cfg.capabilities()
    if not caps["fingerprinting"]:
        print("Note: fingerprinting is off (needs an AcoustID key and fpcalc). "
              "Matching on tags and filenames only.")

    matcher = Matcher(cfg)
    groups: dict[str, list] = {}
    for track in tracks:
        groups.setdefault(str(Path(track.path).parent), []).append(track)

    report = _progress("Identifying")
    done = 0
    for items in groups.values():
        matcher.identify_album(items)
        done += len(items)
        report(done, len(tracks))
    state.persist(tracks)

    buckets = {"high": 0, "review": 0, "low": 0, "none": 0}
    for track in tracks:
        if not track.match or not track.match.candidates:
            buckets["none"] += 1
        elif track.match.confidence >= cfg.auto_apply_threshold:
            buckets["high"] += 1
        elif track.match.confidence >= cfg.review_threshold:
            buckets["review"] += 1
        else:
            buckets["low"] += 1

    print(f"Confident: {buckets['high']}   Needs review: {buckets['review']}   "
          f"Uncertain: {buckets['low']}   No match: {buckets['none']}")
    return 0


def cmd_quality(args) -> int:
    from concurrent.futures import ThreadPoolExecutor
    from .quality import analyze_track
    from .state import get_state

    cfg = get_config()
    if not cfg.ffmpeg:
        print("ffmpeg not found. Install it or set ffmpeg_path in the config.")
        return 2

    state = get_state()
    tracks = state.select(only_unanalysed=not args.all)
    if not tracks:
        print("Nothing to analyse.")
        return 0

    report = _progress("Analysing")
    done = 0

    def run(track):
        nonlocal done
        track.quality = analyze_track(track, cfg)
        done += 1
        report(done, len(tracks))

    with ThreadPoolExecutor(max_workers=cfg.quality_workers) as pool:
        list(pool.map(run, tracks))
    state.persist(tracks)

    flagged = [t for t in tracks if t.quality and t.quality.issues]
    print(f"Analysed {len(tracks)} files; {len(flagged)} have findings.")
    for track in sorted(flagged, key=lambda t: t.quality.score)[:20]:
        worst = track.quality.issues[0]
        print(f"  {track.quality.score:3d}  {Path(track.path).name}  — {worst.title}")
    return 0


def cmd_apply(args) -> int:
    from .apply import Applier, ApplyOptions
    from .state import get_state

    cfg = get_config()
    state = get_state()
    # Not just `match.candidates`: a hand-edited file has a manual match with
    # no candidates (there was never a lookup), but it still has a real
    # proposal that deserves to be applied.
    tracks = [t for t in state.select()
             if t.match and t.match.proposed and not t.match.proposed.is_empty()]
    if args.min_confidence is not None:
        tracks = [t for t in tracks if t.match.confidence >= args.min_confidence]
    if not tracks:
        print("Nothing to apply.")
        return 0

    options = ApplyOptions(
        organize=args.organize,
        dry_run=args.dry_run,
        min_confidence=args.min_confidence,
    )
    if args.organize and not cfg.organize_enabled:
        print("Organising is disabled in the config; enable organize_enabled first.")
        return 2

    applier = Applier(cfg)
    result = applier.apply(tracks, options, progress=_progress("Applying"))
    state.persist(tracks)

    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} tags to {result.tagged} files. "
          f"Moved {result.moved}, copied {result.copied}, failed {result.failed}.")
    if args.dry_run and result.planned:
        print("\nFirst 20 planned moves:")
        for move in result.planned[:20]:
            print(f"  {move['from']}\n    -> {move['to']}")
    if result.batch_id:
        print(f"\nUndo with: musictag undo {result.batch_id}")
    return 0


def cmd_undo(args) -> int:
    from .journal import get_journal
    cfg = get_config()
    result = get_journal().undo(args.batch_id, id3v2_version=cfg.id3v2_version)
    print(f"Restored {result.restored}, skipped {result.skipped}, failed {result.failed}.")
    for message in result.messages[:20]:
        print(f"  {message}")
    return 0


def cmd_history(args) -> int:
    from .journal import get_journal
    batches = get_journal().list_batches()
    if not batches:
        print("No changes have been applied yet.")
        return 0
    for batch in batches:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(batch["started"]))
        print(f"{batch['id']}  {when}  {batch['entry_count']:>5} entries  {batch['description']}")
    return 0


def cmd_report(args) -> int:
    from .state import get_state
    state = get_state()
    stats = state.stats()
    if not stats["total"]:
        print("Nothing scanned yet.")
        return 0

    print(f"Tracks:          {stats['total']}")
    print(f"  confident:     {stats['buckets']['high']}")
    print(f"  needs review:  {stats['buckets']['review']}")
    print(f"  uncertain:     {stats['buckets']['low']}")
    print(f"  unidentified:  {stats['buckets']['unidentified']}")
    print("Formats:         " + ", ".join(f"{k} {v}" for k, v in sorted(stats["formats"].items())))
    if stats["analysed"]:
        print(f"\nQuality analysed: {stats['analysed']}  (average score {stats['average_quality']})")
        print(f"  serious findings: {stats['issues']['high']}")
        print(f"  moderate:         {stats['issues']['medium']}")

    total_seconds = sum(t.props.duration_s for t in state.all())
    print(f"\nTotal playing time: {format_duration(total_seconds)}")
    return 0


def cmd_config(args) -> int:
    cfg = get_config()
    if not args.set:
        for key, value in sorted(cfg.to_dict().items()):
            if key == "acoustid_api_key" and value:
                value = "*" * (len(value) - 3) + value[-3:]
            print(f"{key} = {value}")
        return 0

    updates = {}
    for pair in args.set:
        if "=" not in pair:
            print(f"Expected key=value, got: {pair}")
            return 2
        key, _, value = pair.partition("=")
        updates[key.strip()] = value.strip()
    cfg.update(updates)
    cfg.save()
    print(f"Updated {len(updates)} setting(s) in {APP_DIR / 'config.json'}")
    return 0


# ===========================================================================
# Parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musictag",
        description="Automatic music tagging, Plex organisation and audio quality analysis.",
    )
    parser.add_argument("--version", action="version", version=f"MusicTagger {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command")

    p_ui = sub.add_parser("ui", help="open the app window (default)")
    p_ui.add_argument("--port", type=int, default=0)
    p_ui.add_argument("--no-window", action="store_true",
                      help="just serve; do not open a window or browser")
    p_ui.set_defaults(func=cmd_ui)

    p_scan = sub.add_parser("scan", help="find audio files and read their current tags")
    p_scan.add_argument("paths", nargs="*")
    p_scan.set_defaults(func=cmd_scan)

    p_id = sub.add_parser("identify", help="look up metadata and score the confidence")
    p_id.add_argument("--all", action="store_true", help="re-identify tracks that already matched")
    p_id.set_defaults(func=cmd_identify)

    p_q = sub.add_parser("quality", help="analyse audio quality")
    p_q.add_argument("--all", action="store_true", help="re-analyse files already done")
    p_q.set_defaults(func=cmd_quality)

    p_apply = sub.add_parser("apply", help="write the proposed tags to your files")
    p_apply.add_argument("--organize", action="store_true", help="also move files into the Plex tree")
    p_apply.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    p_apply.add_argument("--min-confidence", type=float, default=None,
                         help="only apply tracks at or above this confidence")
    p_apply.set_defaults(func=cmd_apply)

    p_undo = sub.add_parser("undo", help="reverse a batch of changes")
    p_undo.add_argument("batch_id")
    p_undo.set_defaults(func=cmd_undo)

    sub.add_parser("history", help="list applied batches").set_defaults(func=cmd_history)
    sub.add_parser("report", help="summarise the current scan").set_defaults(func=cmd_report)

    p_cfg = sub.add_parser("config", help="show or change settings")
    p_cfg.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    p_cfg.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if not getattr(args, "func", None):
        # Bare `musictag` opens the window, which is what most people want.
        verbose = args.verbose
        args = parser.parse_args(["ui"])
        args.verbose = verbose
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
