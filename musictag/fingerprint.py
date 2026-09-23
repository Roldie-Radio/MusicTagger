"""Acoustic fingerprinting via Chromaprint (``fpcalc``) and AcoustID lookup.

This is the part that identifies a file from the audio itself, so it works on
``track07.mp3`` with no tags at all. It needs two things:

1. the ``fpcalc`` binary from Chromaprint (a single ~2 MB executable)
2. a free AcoustID API key from https://acoustid.org/new-application

Everything still works without both - :mod:`musictag.matching` falls back to
tag and filename based lookup - just with lower confidence on mystery files.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from .config import Config, TOOLS_DIR
from .providers.http import HttpClient, ProviderError, shared_limiter

log = logging.getLogger(__name__)

ACOUSTID_LOOKUP = "https://api.acoustid.org/v2/lookup"

#: Where to get fpcalc, shown in the UI so nothing is downloaded behind your back.
FPCALC_HOMEPAGE = "https://acoustid.org/chromaprint"
FPCALC_RELEASES = {
    "win32": "https://github.com/acoustid/chromaprint/releases/download/v1.5.1/chromaprint-fpcalc-1.5.1-windows-x86_64.zip",
    "darwin": "https://github.com/acoustid/chromaprint/releases/download/v1.5.1/chromaprint-fpcalc-1.5.1-macos-x86_64.tar.gz",
    "linux": "https://github.com/acoustid/chromaprint/releases/download/v1.5.1/chromaprint-fpcalc-1.5.1-linux-x86_64.tar.gz",
}

#: SHA-256 of each archive above. fpcalc is an executable the app then runs on
#: every file, so the download is checked against these before anything is
#: written - HTTPS proves who served it, this proves it is the build that was
#: reviewed. Measured from the v1.5.1 release assets; a new version needs new
#: hashes, never a skipped check.
FPCALC_SHA256 = {
    "win32": "36b478e16aa69f757f376645db0d436073a42c0097b6bb2677109e7835b59bbc",
    "darwin": "c6c2797c4f087cf139eedd71554bc59ef8f26a783dc00c7f3ad5ae71d3a616fe",
    "linux": "4d7433a7f778e5946d7225230681cbcd634e153316ecac87c538c33ac32387a5",
}

#: fpcalc only needs the first couple of minutes to produce a usable fingerprint.
FINGERPRINT_LENGTH_S = 120


class FingerprintUnavailable(Exception):
    """Raised when fpcalc or the AcoustID key is missing."""


def fingerprint_file(path: Path, cfg: Config, *, length_s: int = FINGERPRINT_LENGTH_S
                     ) -> tuple[int, str]:
    """Run fpcalc and return ``(duration_seconds, fingerprint)``."""
    fpcalc = cfg.fpcalc
    if not fpcalc:
        raise FingerprintUnavailable(
            "fpcalc not found. Install Chromaprint, or set its path in Settings."
        )
    cmd = [fpcalc, "-json", "-length", str(length_s), str(path)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"fpcalc timed out on {path.name}") from exc
    except OSError as exc:
        raise FingerprintUnavailable(f"Could not run fpcalc: {exc}") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        raise ProviderError(f"fpcalc failed on {path.name}: {stderr[:200]}")

    try:
        data = json.loads(proc.stdout.decode("utf-8", "ignore"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"fpcalc returned unparseable output for {path.name}") from exc

    return int(data.get("duration") or 0), data.get("fingerprint") or ""


class AcoustIDClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # AcoustID asks for no more than 3 requests per second.
        self.http = HttpClient(user_agent=cfg.user_agent, rate_limiter=shared_limiter("acoustid", 0.34))

    @property
    def available(self) -> bool:
        return bool(self.cfg.acoustid_api_key.strip() and self.cfg.fpcalc)

    def lookup(self, duration: int, fingerprint: str) -> list[dict[str, Any]]:
        """Return AcoustID results, best score first.

        Each result looks like::

            {"score": 0.94, "id": "...", "recordings": [{"id": "...", ...}]}
        """
        key = self.cfg.acoustid_api_key.strip()
        if not key:
            raise FingerprintUnavailable("No AcoustID API key configured.")
        if not fingerprint:
            return []

        params = {
            "client": key,
            "duration": duration,
            "fingerprint": fingerprint,
            # ONLY "recordings". Asking for anything more - releases,
            # releasegroups, compress - makes AcoustID return results with an
            # empty recordings list, which silently turned every fingerprint
            # hit into "no match" and made fingerprinting useless. Verified
            # against the live API: "recordings" returns 8 recordings for a
            # track that "recordings+releasegroups+releases+compress" returns
            # 0 for.
            #
            # What comes back here matters: :meth:`Matcher._acoustid_candidates`
            # builds its candidates straight out of this response rather than
            # re-fetching each one from MusicBrainz, so the titles, artists and
            # durations below are what scoring actually sees.
            "meta": "recordings",
            "format": "json",
        }
        # Cache on the fingerprint, not the key, so rotating keys keeps the cache.
        #
        # The "v2" is a schema marker, and it matters: the cache key does not
        # include the request's meta parameter, so every response cached while
        # meta was wrong (returning results with an empty recordings list) is
        # still sitting in caches with up to a month left to live, and would go
        # on reporting "no match" for files that fingerprint perfectly. Bumping
        # this retires all of them without anyone having to clear a cache.
        cache_key = f"acoustid:v2:{duration}:{fingerprint[:120]}"
        data = self.http.get_json(ACOUSTID_LOOKUP, params, cache_key=cache_key)
        if not data:
            return []
        if data.get("status") != "ok":
            raise ProviderError(f"AcoustID error: {data.get('error', {}).get('message', 'unknown')}")
        results = data.get("results") or []
        return sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    def identify(self, path: Path) -> tuple[list[dict[str, Any]], Optional[str], int]:
        """Fingerprint ``path`` and look it up.

        Returns ``(results, fingerprint, duration)``.
        """
        duration, fp = fingerprint_file(path, self.cfg)
        return self.lookup(duration, fp), fp, duration


# ---------------------------------------------------------------------------
# Optional installer, only ever run when the user clicks "Install" in Settings
# ---------------------------------------------------------------------------

def _fpcalc_platform() -> Optional[str]:
    import sys
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return None


def fpcalc_download_url() -> Optional[str]:
    platform = _fpcalc_platform()
    return FPCALC_RELEASES[platform] if platform else None


def install_fpcalc(progress=None) -> str:
    """Download the official Chromaprint release and unpack ``fpcalc``.

    Explicitly user-initiated: the UI shows the exact URL and requires a click.
    Returns the path to the installed binary.
    """
    import io
    import os
    import sys
    import tarfile
    import zipfile

    import requests

    url = fpcalc_download_url()
    if not url:
        raise FingerprintUnavailable(f"No prebuilt fpcalc for {sys.platform}; see {FPCALC_HOMEPAGE}")

    if progress:
        progress(f"Downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    import hashlib
    expected = FPCALC_SHA256[_fpcalc_platform()]
    actual = hashlib.sha256(resp.content).hexdigest()
    if actual != expected:
        raise FingerprintUnavailable(
            f"The fpcalc download did not match the expected checksum, so it was not "
            f"installed (got {actual[:12]}..., expected {expected[:12]}...). "
            f"Try again later, or install Chromaprint yourself from {FPCALC_HOMEPAGE}")

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    exe_name = "fpcalc.exe" if os.name == "nt" else "fpcalc"
    target = TOOLS_DIR / exe_name

    payload = io.BytesIO(resp.content)
    if url.endswith(".zip"):
        with zipfile.ZipFile(payload) as zf:
            member = next((n for n in zf.namelist() if n.rsplit("/", 1)[-1] == exe_name), None)
            if member is None:
                raise FingerprintUnavailable("The downloaded archive did not contain fpcalc")
            target.write_bytes(zf.read(member))
    else:
        with tarfile.open(fileobj=payload, mode="r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.rsplit("/", 1)[-1] == exe_name), None)
            if member is None:
                raise FingerprintUnavailable("The downloaded archive did not contain fpcalc")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FingerprintUnavailable("Archive did not contain fpcalc")
            target.write_bytes(extracted.read())

    if os.name != "nt":
        target.chmod(0o755)
    if progress:
        progress(f"Installed {target}")
    return str(target)
