"""User configuration, persisted as JSON so the Settings panel can round-trip it."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

from . import __version__, USER_AGENT_TEMPLATE

APP_DIR = Path(os.environ.get("MUSICTAGGER_HOME") or (Path.home() / ".musictagger"))
CONFIG_PATH = APP_DIR / "config.json"
CACHE_DB = APP_DIR / "cache.db"
STATE_DB = APP_DIR / "state.db"
JOURNAL_DB = APP_DIR / "journal.db"
TOOLS_DIR = APP_DIR / "tools"


def builtin_acoustid_key() -> str:
    """The AcoustID application key built into this copy of the app, if any.

    AcoustID issues keys per *application*, not per person, and expects them
    to ship inside the app (MusicBrainz Picard does the same). This one lets a
    fresh install fingerprint straight away instead of every machine needing
    its own key typed into Settings.

    The repository is public, so the key is never committed. The release
    build writes it into ``musictag/_app_key.py`` (gitignored) from the
    ``ACOUSTID_API_KEY`` repository secret. ``MUSICTAGGER_ACOUSTID_KEY`` in
    the environment covers running from source.
    """
    env = os.environ.get("MUSICTAGGER_ACOUSTID_KEY", "").strip()
    if env:
        return env
    try:
        from ._app_key import ACOUSTID_API_KEY
    except ImportError:
        return ""
    return str(ACOUSTID_API_KEY or "").strip()


@dataclass
class Config:
    # --- library ---------------------------------------------------------
    library_paths: list[str] = field(default_factory=list)

    # --- identification --------------------------------------------------
    acoustid_api_key: str = ""
    fpcalc_path: str = ""                     # blank -> look on PATH / TOOLS_DIR
    musicbrainz_contact: str = ""             # email or URL; MB asks for a contact
    musicbrainz_rate_limit: float = 1.0       # seconds between requests (MB asks for >= 1)
    max_candidates: int = 5

    # Confidence thresholds, in percent.
    auto_apply_threshold: int = 92            # at/above this, safe to apply unattended
    review_threshold: int = 70                # below this, flag as "needs review"

    # --- tag writing -----------------------------------------------------
    preserve_existing_tags: bool = False      # True -> only fill in blanks
    write_cover_art: bool = True
    embed_art_max_px: int = 1000
    write_cover_file: bool = True             # also drop cover.jpg next to the album
    cover_filename: str = "cover.jpg"
    id3v2_version: int = 4                    # 3 or 4; Plex reads both, 4 is the modern default
    write_musicbrainz_ids: bool = True
    various_artists_name: str = "Various Artists"

    # --- organisation (opt-in) -------------------------------------------
    organize_enabled: bool = False
    organize_mode: str = "move"               # "move" or "copy"
    organize_root: str = ""                   # blank -> in place, relative to library root
    folder_template: str = "{album_artist}/{album}{year_suffix}"
    file_template: str = "{disc_prefix}{track:02d} - {title}"
    keep_extra_files: bool = True             # move cover.jpg / .cue / .log alongside

    # --- quality analysis ------------------------------------------------
    quality_enabled: bool = True
    quality_workers: int = 4
    quality_max_seconds: int = 0              # 0 = whole file
    ffmpeg_path: str = ""
    ffprobe_path: str = ""

    # --- updates ---------------------------------------------------------
    # The one request the app makes on its own initiative rather than because
    # the user asked for something, so it is switchable off. Nothing is ever
    # downloaded or installed automatically - see musictag/update.py.
    update_check_enabled: bool = True

    # --- misc ------------------------------------------------------------
    scan_workers: int = 8
    identify_workers: int = 4                 # network bound; MB limiter serialises anyway
    theme: str = "system"                     # system | light | dark

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_PATH
        cfg = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            known = {f.name for f in fields(cls)}
            for key, value in data.items():
                if key in known:
                    setattr(cfg, key, value)
        return cfg

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, data: dict[str, Any]) -> None:
        known = {f.name: f.type for f in fields(self)}
        for key, value in (data or {}).items():
            if key not in known:
                continue
            current = getattr(self, key)
            # Light coercion so the UI can send strings for numeric inputs.
            try:
                if isinstance(current, bool):
                    value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int) and not isinstance(value, bool):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
            except (TypeError, ValueError):
                continue
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # derived helpers
    # ------------------------------------------------------------------
    @property
    def user_agent(self) -> str:
        contact = self.musicbrainz_contact.strip() or "https://github.com/local/musictagger"
        return USER_AGENT_TEMPLATE.format(version=__version__, contact=contact)

    def resolve_tool(self, name: str, configured: str) -> str | None:
        """Find an external binary: explicit config -> our tools dir -> PATH."""
        if configured:
            p = Path(configured)
            if p.exists():
                return str(p)
        exe = name + (".exe" if os.name == "nt" else "")
        local = TOOLS_DIR / exe
        if local.exists():
            return str(local)
        found = shutil.which(name)
        return found

    @property
    def ffmpeg(self) -> str | None:
        return self.resolve_tool("ffmpeg", self.ffmpeg_path)

    @property
    def ffprobe(self) -> str | None:
        return self.resolve_tool("ffprobe", self.ffprobe_path)

    @property
    def fpcalc(self) -> str | None:
        return self.resolve_tool("fpcalc", self.fpcalc_path)

    @property
    def acoustid_key(self) -> str:
        """The key lookups actually use: yours if set, else the built-in one."""
        return self.acoustid_api_key.strip() or builtin_acoustid_key()

    def capabilities(self) -> dict[str, Any]:
        """What the app can actually do right now, for the UI to show honestly."""
        return {
            "fingerprinting": bool(self.fpcalc and self.acoustid_key),
            "fpcalc": self.fpcalc,
            "acoustid_key_set": bool(self.acoustid_api_key.strip()),
            # Whether fingerprinting works with no key of your own.
            "acoustid_builtin_key": bool(builtin_acoustid_key()),
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "quality_analysis": bool(self.ffmpeg),
        }


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config.load()
    return _CONFIG


def set_config(cfg: Config) -> None:
    global _CONFIG
    _CONFIG = cfg
