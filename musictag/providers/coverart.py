"""Cover Art Archive lookups.

The archive serves pre-scaled thumbnails, so we pick a size rather than pulling
a 20 MB TIFF and resizing it locally (which would mean depending on Pillow).
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Config
from .http import HttpClient, shared_limiter

log = logging.getLogger(__name__)

BASE = "https://coverartarchive.org"
AVAILABLE_SIZES = (250, 500, 1200)


def _size_for(max_px: int) -> int:
    """Pick the largest offered thumbnail that fits under ``max_px``."""
    usable = [s for s in AVAILABLE_SIZES if s <= max_px]
    return max(usable) if usable else AVAILABLE_SIZES[0]


class CoverArtClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # The archive is friendlier than MusicBrainz proper, but stay polite.
        self.http = HttpClient(user_agent=cfg.user_agent, rate_limiter=shared_limiter("coverart", 0.34))

    def front_url(self, *, release_id: Optional[str] = None,
                  release_group_id: Optional[str] = None,
                  max_px: Optional[int] = None) -> Optional[str]:
        size = _size_for(max_px or self.cfg.embed_art_max_px)
        if release_id:
            return f"{BASE}/release/{release_id}/front-{size}"
        if release_group_id:
            return f"{BASE}/release-group/{release_group_id}/front-{size}"
        return None

    def fetch_front(self, *, release_id: Optional[str] = None,
                    release_group_id: Optional[str] = None,
                    max_px: Optional[int] = None) -> Optional[tuple[bytes, str]]:
        """Return ``(image_bytes, mime)`` for the front cover, or ``None``.

        Tries the specific release first, then the release group, because many
        releases have no art of their own but their group does.
        """
        for kwargs in ({"release_id": release_id}, {"release_group_id": release_group_id}):
            if not any(kwargs.values()):
                continue
            url = self.front_url(max_px=max_px, **kwargs)
            if not url:
                continue
            data = self.http.get_bytes(url)
            if data:
                mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
                return data, mime
        return None
