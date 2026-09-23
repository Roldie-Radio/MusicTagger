"""Installing fpcalc: the download is an executable, so it has to be the right one."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import zipfile

import pytest

from musictag import fingerprint
from musictag.fingerprint import FPCALC_SHA256, FingerprintUnavailable, install_fpcalc


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def _archive(platform: str, *, with_fpcalc: bool = True) -> bytes:
    exe = "fpcalc.exe" if platform == "win32" else "fpcalc"
    name = f"chromaprint-fpcalc-1.5.1/{exe if with_fpcalc else 'README'}"
    buf = io.BytesIO()
    if platform == "win32":
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, b"binary")
    else:
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo(name)
            info.size = len(b"binary")
            tf.addfile(info, io.BytesIO(b"binary"))
    return buf.getvalue()


@pytest.fixture
def platform(monkeypatch, tmp_path):
    key = fingerprint._fpcalc_platform()
    if key is None:
        pytest.skip(f"no prebuilt fpcalc for {sys.platform}")
    monkeypatch.setattr(fingerprint, "TOOLS_DIR", tmp_path / "tools")
    return key


def serve(monkeypatch, content: bytes):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: FakeResponse(content))


def test_a_download_that_does_not_match_the_pinned_hash_is_refused(platform, monkeypatch, tmp_path):
    serve(monkeypatch, _archive(platform))            # a valid archive, but not the real one
    with pytest.raises(FingerprintUnavailable, match="checksum"):
        install_fpcalc()
    assert not (tmp_path / "tools").exists() or not any((tmp_path / "tools").iterdir()), \
        "nothing may be written when the checksum fails"


def test_a_matching_download_is_installed(platform, monkeypatch, tmp_path):
    content = _archive(platform)
    monkeypatch.setitem(FPCALC_SHA256, platform, hashlib.sha256(content).hexdigest())
    serve(monkeypatch, content)
    path = install_fpcalc()
    assert open(path, "rb").read() == b"binary"


def test_an_archive_without_fpcalc_says_so(platform, monkeypatch):
    content = _archive(platform, with_fpcalc=False)
    monkeypatch.setitem(FPCALC_SHA256, platform, hashlib.sha256(content).hexdigest())
    serve(monkeypatch, content)
    with pytest.raises(FingerprintUnavailable, match="did not contain fpcalc"):
        install_fpcalc()


def test_every_download_url_has_a_pinned_hash():
    assert set(fingerprint.FPCALC_RELEASES) == set(FPCALC_SHA256)
    assert all(len(h) == 64 for h in FPCALC_SHA256.values())
