# PyInstaller spec for the MusicTagger backend, bundled into the Electron app.
#
# Build from the repo root:
#   .venv\Scripts\pyinstaller desktop\backend\backend.spec --distpath desktop\dist --workpath desktop\build --noconfirm
#
# Produces desktop/dist/musictagger-backend/musictagger-backend.exe (onedir,
# not onefile - onefile's self-extracting startup cost is noticeable for a UI
# that should feel instant, and onedir is easier to debug when something in
# this list is wrong).

from pathlib import Path

block_cipher = None

# PyInstaller exec()s this file without setting __file__; it injects SPECPATH
# instead (the directory containing this spec) - use that to find the repo
# root regardless of the caller's cwd.
SPEC_DIR = Path(SPECPATH).resolve()  # noqa: F821 - injected by PyInstaller
REPO_ROOT = SPEC_DIR.parent.parent

a = Analysis(
    [str(SPEC_DIR / "backend_main.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=[
        # The web UI's static files. Kept at "musictag/web" inside the bundle
        # so server.py's `Path(__file__).parent / "web"` still resolves.
        (str(REPO_ROOT / "musictag" / "web"), "musictag/web"),
    ],
    hiddenimports=[
        # uvicorn resolves its actual protocol implementations lazily by
        # string name, which PyInstaller's static import scan cannot see.
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "mutagen.mp3", "mutagen.flac", "mutagen.mp4", "mutagen.oggvorbis",
        "mutagen.oggopus", "mutagen.asf", "mutagen.wave", "mutagen.aiff",
        "mutagen.id3",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # This app never opens its own window - drop pywebview and its GUI
        # toolkit backends so they are not silently pulled in and frozen.
        "webview",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="musictagger-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # verified working with console=True; no window needed - Electron owns the UI
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="musictagger-backend",
)
