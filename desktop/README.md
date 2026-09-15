# MusicTagger Desktop (Windows installer)

Packages the existing Python app - unchanged - into a proper Windows
installer. Nothing about `musictag/` changes for this: the frozen backend
runs the exact same FastAPI app the `.bat`-based install runs, so every fix
and every test in the main test suite applies here too.

## Why a Python backend inside Electron, rather than a native Electron app

Tagging (`mutagen`), matching (`rapidfuzz`, the MusicBrainz client) and the
quality analyzer (`numpy`, FFT-based spectral work) are all deeply
Python-and-numpy. Reimplementing that in JavaScript would mean rebuilding
everything the 300+ tests in `../tests/` already validate, calibrated
against real music, from scratch, in a different language, for no functional
benefit. Instead: Electron owns the window and the installer; a PyInstaller-
frozen copy of the same backend does the actual work, exactly as it does
when launched via `MusicTagger.bat`.

```
Electron (main.js)
  -> spawns musictagger-backend.exe (frozen musictag.server:app)
  -> waits for it to answer /api/status
  -> opens a BrowserWindow pointed at http://127.0.0.1:<port>/
```

The web UI itself (`musictag/web/`) is served by the backend unchanged - the
BrowserWindow is just loading the same page a browser tab would.

## Building it yourself

### 1. Freeze the Python backend

From the repo root, with the project's `.venv` active:

```bash
pip install pyinstaller
```

```bash
python -m PyInstaller desktop/backend/backend.spec --distpath desktop/dist --workpath desktop/build --noconfirm
```

Produces `desktop/dist/musictagger-backend/musictagger-backend.exe`. Sanity
check it standalone before going further - point `MUSICTAGGER_HOME` at a
throwaway folder so this never touches your real library data:

```bash
MUSICTAGGER_HOME=/tmp/mt-test desktop/dist/musictagger-backend/musictagger-backend.exe
```

It prints `MUSICTAGGER_PORT=<port>`; `curl http://127.0.0.1:<port>/api/status`
should return real JSON.

### 2. Get the bundled tools

See `build-resources/tools/README.md` - two downloads, ffmpeg essentials
build and fpcalc, both pinned to specific versions.

### 3. Build the icon (only needed if you change it)

```bash
pip install pillow
```

```bash
python desktop/build-resources/make_icon.py
```

### 4. Install Electron dependencies and package

```bash
cd desktop/electron && npm install
```

```bash
npm run dist
```

Produces `desktop/release/MusicTagger Setup <version>.exe`. `npm start` runs
the shell straight from source (against whatever is in `desktop/dist/`)
without a full package, for faster iteration on `main.js`.

If `npm install`'s Electron download silently produces an empty
`node_modules/electron/dist/` (a known `extract-zip` incompatibility with
newer Node versions, unrelated to this project): the zip itself downloads
fine to `%LOCALAPPDATA%\electron\Cache`, it just isn't extracted. Expand it
yourself with `Expand-Archive` into `node_modules/electron/dist/` and write
`electron.exe` into `node_modules/electron/path.txt`, then retry.

## What ships in the installer

Only: the Electron shell (`main.js` + `package.json`, asar-packed), the
frozen backend and its Python library dependencies, and the three bundled
tool binaries. No user data, no library paths, nothing from a development
machine's `~/.musictagger` - verified by listing `app.asar`'s contents and
grepping the whole unpacked output for any local path. Actual app data
(scan results, settings, the undo journal) is created fresh at
`~/.musictagger` the first time the *installed* app runs, on the machine it
is installed on - never anything carried in the installer itself.

## What first launch does

`main.js` copies the three bundled tool binaries into `~/.musictagger/tools/`
if they are not already there (never overwriting a tool you already have)-
this is the same folder `Config.resolve_tool()` in `musictag/config.py`
already checks first, so no backend code has to know it is running inside a
packaged app. ffmpeg, ffprobe and quality analysis work immediately with no
setup. AcoustID fingerprinting still needs your own free API key (Settings →
Identification) - that part was never a bundling problem to solve.

## Uninstalling

Standard NSIS uninstaller (Start Menu → MusicTagger → Uninstall, or Windows
Settings → Apps). Removes the installed files and both shortcuts. Your
`~/.musictagger` app data is untouched by uninstalling, by design - the same
data an update should be able to pick back up.
