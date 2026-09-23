# MusicTagger

Automatic metadata tagging for a music library, organised the way Plex expects,
with an honest confidence score on every match and an audio quality report on
every file.

Nothing is written to your files until you explicitly apply changes, and every
change that *is* made can be undone.

---

## Quick start (Windows)

**Installer** (recommended if you just want to run the app): grab
`MusicTagger Setup <version>.exe` and run it - no Python, no ffmpeg, nothing
else to install first. See [`desktop/README.md`](desktop/README.md) for how
that installer is built, what it bundles, and how to build your own.

**From source**, double-click **`MusicTagger.bat`**. The first run creates a
virtual environment and installs dependencies (about a minute), then opens
the app.

From a terminal:

```bash
py -3 -m venv .venv && .venv\Scripts\pip install -r requirements.txt
```

```bash
.venv\Scripts\python -m musictag ui
```

If `pywebview` is installed you get a native window; otherwise it opens in your
browser. Either way the server is local only, bound to `127.0.0.1`.

### What you need

| | Required? | What it gets you |
|---|---|---|
| **Python 3.10+** | yes | — |
| **ffmpeg** | for quality analysis | bitrate/transcode/clipping/truncation detection |
| **AcoustID API key** | optional | identifies files from the audio itself |
| **fpcalc** (Chromaprint) | optional | generates the fingerprints the key looks up |

ffmpeg: <https://ffmpeg.org/download.html> — or `winget install Gyan.FFmpeg`.
AcoustID key: free, ~2 minutes, at <https://acoustid.org/new-application>.
`fpcalc`: Settings → Identification has a **Download fpcalc** button that shows
you the exact URL before fetching anything, or install Chromaprint yourself.

Without fingerprinting the app still works — it matches on existing tags and
filenames — but mystery files score much lower, on purpose.

---

## The four things it does

### 1. Fills in metadata automatically

For each file it gathers evidence, then asks MusicBrainz:

1. **An embedded MusicBrainz ID**, if the file already has one — verified, not trusted blindly.
2. **An acoustic fingerprint** via Chromaprint → AcoustID. This reads the audio
   itself, so it works on `track07.mp3` with no tags at all.
3. **A text search** built from whatever tags and filename structure exist.
4. **The file path**, as a last resort — clearly labelled as unverified.

Then it does an **album consolidation pass**. Tracks matched one at a time tend
to scatter across several releases of the same album, which makes Plex show you
three half-albums. So after matching a folder, whichever release most of the
tracks agree on wins, and the stragglers are re-seated onto that release's
tracklist.

### 2. Makes the library work in Plex

Plex uses **both** mechanisms, and it is worth being precise about which does what:

- **Embedded tags decide what your library looks like.** Album, album artist,
  track number, disc number and year all come from the tags; Plex's music agents
  match on those.
- **Folder structure is the fallback and the browse structure.** The documented
  layout is `Artist/Album/tracks`. When a match fails or tags are missing, Plex
  reads the folders.

So this writes both.

The single most common cause of a mangled Plex music library is a **missing or
inconsistent album artist** — without it, one compilation explodes into one
"album" per track artist. The matcher never leaves that field empty, and
compilations are set to `Various Artists` with the compilation flag.

Also written: embedded cover art (plus `cover.jpg` in the album folder, which is
Plex's fallback), disc numbers with totals, and MusicBrainz IDs so a future
re-tag is exact rather than another guess.

Default layout, both templates configurable:

```
<root>/<Album Artist>/<Album> (<Year>)/<track##> - <Title>.<ext>
```

Reorganising is **off by default**. When enabled it defaults to a dry run you
can preview, and moves are journalled so they can be reversed.

Formats: MP3, FLAC, M4A/MP4 (AAC and ALAC), OGG Vorbis, Opus, WMA, WAV, AIFF.

### 3. Tells you how confident it is

Every match carries a percentage, and every percentage is explained. It is a
weighted blend of independent signals — fingerprint score, duration agreement,
title/artist/album similarity, track position — with three adjustments that a
plain average would miss:

- **The title is the track's identity.** If artist, album and duration all agree
  but the title does not, that is the signature of *a different track on the same
  album*, not a good match. It gets a hard penalty (unless a strong fingerprint
  says otherwise — then the file's own title tag is simply wrong, which is what
  you are here to fix).
- **Ambiguity is discounted.** A runner-up scoring nearly as well means the top
  answer is worth less. Two plausible answers is not one good answer.
- **Evidence caps the ceiling.** A file with no tags gives nothing to check a
  guess against, so it cannot score above ~60% no matter how clean the guess
  looks. A filename-only match caps at 45%. A strong fingerprint lifts the
  ceiling to 99%, because that is direct evidence about the audio.

Confidence is reported per field too — a title corroborated by two independent
sources is worth more than a genre nobody verified.

#### When a lookup fails

A MusicBrainz *search* (as opposed to a direct lookup by ID) should never 404 —
"nothing matches" comes back as a normal 200 with an empty list. A 404 there is
treated as a transient hiccup and retried, not as proof the song doesn't exist;
only after every retry fails does it give up, and that failure is never cached,
so the very next attempt tries fresh rather than being stuck for the month-long
cache TTL. A genuine timeout is retried the same way, with backoff, before it's
surfaced as a note on that track.

If a track's confidence still looks wrong because of a lookup that should have
found something and didn't — a real outage while it was scanning, say — Settings
→ Identification has a **Clear cached lookups** button. It forgets every cached
MusicBrainz/AcoustID/cover-art response so re-running Tag starts over
rather than waiting out the TTL.

The UI groups tracks into **Confident / Needs review / Uncertain** (thresholds
configurable) and expanding any row shows the signals, the alternatives, and why
the score is what it is. You can pick a different candidate or type a value in
by hand.

#### Two views

A toggle in the toolbar switches between them; your choice is remembered.

- **Review** — current tags on the left, proposed on the right, changes
  highlighted. This is the view for deciding whether a match is right.
- **Columns** — a spreadsheet of the actual tag fields: Title, Artist, Album,
  Album artist, track and disc numbers, Year, Genre, Composer, ISRC, plus file
  facts like format, bitrate, length and folder. This is the view for *tagging*:
  scanning down the Album artist column to spot the blanks that will split an
  album in Plex is something the review view simply cannot show you.

  Click any heading to sort by it (again to reverse). Sorting happens server-side
  on the **effective** value — what the file would actually end up with — so the
  column you see is the column you sort. Empty cells always sink to the bottom in
  both directions, because a blank is a gap to fill, not a small value.

  **Double-click any tag cell to edit it in place.** Pick which columns are shown
  via *Columns…*, and flip the whole grid between proposed and current values
  with the *Values* selector.

#### Current metadata, and before → after

Expanding a row always shows two sections, whether or not the file has been
identified yet:

- **Current metadata** — everything already on the file: title, artist, album,
  album artist, track/disc numbers, date, genre, composer, ISRC, compilation
  flag, and an embedded-art thumbnail if there is one.
- **Proposed changes** — a field-by-field table: current value, an arrow, the
  proposed value, and that field's confidence. `→` means the field would
  change; `=` means it would end up the same. A field the match has no opinion
  on is shown **as its current value with `=`**, not as blank — applying never
  erases a tag the proposal doesn't cover, so the table doesn't pretend it
  will. Before Tag has run, this section is where you type tags in by
  hand instead: click **Edit by hand**, and the diff fills in as soon as you
  save.

This is also where **Apply this file** lives for a single track, and it only
appears once there is at least one real proposed value to apply — including a
value you typed in yourself, not only one a lookup found.

### 4. Finds broken and low-quality audio

Requires ffmpeg. Decodes each file and measures it:

| Finding | How it is detected |
|---|---|
| **Fake lossless** — a FLAC made from an MP3 | Lossy encoders low-pass the signal and that shelf never comes back. A cliff at 15 kHz in a "lossless" file gives it away. |
| **Inflated bitrate** — 96 kbps re-encoded to 320 | Measured roll-off compared against what the claimed bitrate should actually produce. |
| **Cut-off tracks** | Ends *near full level* with no fade or decay (a moderately loud, undecayed ending — common when a song is mixed to close on a chord rather than fade — is deliberately left alone); and/or is meaningfully shorter than the matched recording. |
| **Clipping** | Samples pinned at full scale, weighted by both the *ratio* and *runs* — calibrated against 24 real commercial tracks, where ordinary peak limiting touches full scale for a handful of isolated samples without being audible distortion. Only a genuinely large share of the file (checked: >1.5%) reaches "high," and even then the wording doesn't assert the file is broken — sustained clipping is standard practice in some genres (hip-hop, EDM, industrial). |
| **Clicks and pops** | A sample-to-sample jump, judged against the *local* noise floor (not one number for the whole track) so a loud chorus isn't judged by a quiet intro's standard. Deliberately conservative: checked against 24 real commercial tracks, a hard drum hit can produce the same shape as genuine damage, so this never claims "high" severity and says so when it fires — trust your ears over the count. |
| **Dropouts** | A near-silent gap in the middle of a track, lasting several seconds — short pauses (checked: under 2s) are left alone, since a dramatic pause or beat-drop in the arrangement looks identical to a short one and is far more common than real corruption. |
| **Low bitrate / sample rate** | Per-codec thresholds; 128 kbps Opus is fine, 128 kbps MP3 is not. |
| **Also** | DC offset, crushed dynamic range, mono-stored-as-stereo, long silent lead-ins. |

#### Reading a quality badge

A badge like `70 · Serious` or `94 · Minor` is two separate things, and the app
says so — there is a **?** beside the Quality column header that explains it,
and every expanded row spells it out again.

- **The number is the overall score.** Every file starts at 100; each finding
  deducts points. Nothing wrong means it stays at 100.
- **The word is the worst single problem found.** A file can score well overall
  and still have one fault worth acting on.

| Word | Cost | Meaning |
|---|---|---|
| Serious | −30 | Something is likely wrong with the file itself. Usually worth replacing. |
| Moderate | −15 | Noticeably below par. Worth a listen before deciding. |
| Minor | −6 | A small imperfection most people will never hear. |
| Note | 0 | An observation, not a fault — a mono recording, a very short track. |

So `70 · Serious` is one serious finding (100 − 30), and `94 · Minor` is one
minor one (100 − 6). Expanding the row shows the working — *"Score 64 of 100.
Deducted 30 for 1 serious finding, 6 for 1 minor finding."* — with each finding
tagged by severity and point cost, worst first.

The scale is served from the API rather than hardcoded in the UI, so the badge,
the help text and the score cannot drift apart.

Each finding says what was measured, so you can disagree with a threshold rather
than take it on faith:

> **Spectrum suggests a lower bitrate than 320 kbps** — A 320 kbps MP3 should
> reach about 20.6 kHz, but this rolls off at 15.6 kHz. The file was probably
> re-encoded upward from something smaller: the bitrate is real, the quality is not.

The cutoff detector finds the *edge* — the steepest sustained drop — rather than
comparing against an absolute floor, which is what makes it work across quiet
recordings, loud ones, and every spectral tilt in between.

---

## Safety

- **Nothing is written until you click Apply.** Scan and identify are read-only.
- **Every write is journalled** with the previous tag values. Settings → History
  lists every batch with an Undo button; undo restores the old tags and moves
  files back where they came from.
- **Undo takes back what Apply added.** A tag Apply wrote into a field that was
  blank before is removed again on undo, not left behind; tags Apply never
  touched are left exactly as they were.
- **Undo never deletes.** If you organised in *copy* mode, undo reports the
  copies it made rather than removing files.
- **Preview first.** The Preview button shows every planned tag write and file
  move without touching anything.
- **Only this app can drive the local server.** It answers only requests
  addressed to `127.0.0.1`/`localhost` and refuses state-changing requests from
  other websites, so a page open in your browser cannot reach your files
  through it.
- Nothing leaves your machine except metadata queries to MusicBrainz/AcoustID
  and cover art downloads.

---

## Command line

The UI is the main way in, but everything works headlessly — useful for a
scheduled re-scan as the library grows.

```bash
.venv\Scripts\python -m musictag scan "D:\Music"
```

```bash
.venv\Scripts\python -m musictag identify
```

```bash
.venv\Scripts\python -m musictag quality
```

```bash
.venv\Scripts\python -m musictag apply --dry-run --organize
```

```bash
.venv\Scripts\python -m musictag apply --min-confidence 92
```

```bash
.venv\Scripts\python -m musictag undo <batch-id>
```

`report`, `history` and `config` round it out. `musictag config --set key=value`
edits settings; `musictag config` prints them all.

---

## Where things live

Settings, caches and the undo journal are in `~/.musictagger/`
(`C:\Users\<you>\.musictagger`), not in the project folder. Set
`MUSICTAGGER_HOME` to move them.

| File | Contents |
|---|---|
| `config.json` | your settings |
| `cache.db` | MusicBrainz/AcoustID responses (makes re-scans fast and stays polite) |
| `state.db` | the current scan, so closing the window loses nothing |
| `journal.db` | the undo log |

---

## Layout

```
musictag/
  models.py        shared data shapes
  config.py        settings, tool discovery
  library.py       filesystem scanning
  tags.py          per-format read/write (the Plex compatibility layer)
  fingerprint.py   Chromaprint + AcoustID
  providers/       MusicBrainz, Cover Art Archive, shared HTTP with rate limiting
  matching.py      candidate generation, scoring, confidence
  organize.py      Plex path planning
  apply.py         writing tags, artwork, moves
  journal.py       undo
  quality/         ffmpeg probe + defect analysis
  state.py         in-memory library, persisted
  jobs.py          background job runner
  server.py        FastAPI API
  web/             the UI (no build step)
  cli.py           command line
```

## Updates

The app asks GitHub at most once every six hours whether a newer release
exists, and shows a banner if one does. **Nothing is downloaded or installed
automatically** - the banner links to the release and you decide.

This is the only request MusicTagger makes on its own initiative rather than
because you asked it to look something up, so it is a single switch to turn
off: Settings → Audio analysis → Updates → *Check for new versions*. With it
off, no request is made at all. There is also a *Check now* button there for
an immediate answer that ignores the cache.

Releases are built by
[`.github/workflows/release.yml`](.github/workflows/release.yml), which has two
ways in.

**Push a tag**, if your credentials can write `refs/tags/`:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

**Or run the workflow by hand** from the Actions tab, on any branch. No tag is
needed: the version comes from `musictag/__init__.py`, and publishing the draft
release the build produces is what creates the tag - GitHub's Releases API
mints a tag that does not exist yet when a release naming it is published. This
is the route to use when a token is scoped to `refs/heads/` and is refused on
`refs/tags/`, which is common for CI and app credentials.

Either way the installer is uploaded to a **draft** release, so it can be
checked before anything is public. Until that draft is published,
`/releases/latest` keeps returning 404 and the in-app check keeps correctly
reporting there is nothing to update to.

Bump `__version__` in `musictag/__init__.py` and `version` in
`desktop/electron/package.json` together. A tagged build fails if the tag
disagrees with the first, and `tests/test_update.py` fails if the two files
disagree with each other - an installer that misreports its own version would
tell every user who installs it that an update is available forever.

## Tests

```bash
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

Audio fixtures are synthesised with numpy and ffmpeg rather than checked in, so
every defect the analyser looks for is present in a file where the exact signal
is known. The suite includes a false-positive baseline: a clean file must come
back quiet, or the detectors are not worth having.

Building those fixtures needs `ffmpeg` on your `PATH`. Without it they skip
rather than fail, taking 118 of the 377 tests out of the run - so a green result
on a machine with no ffmpeg is a weaker signal than it looks.
[`desktop/build-resources/tools/README.md`](desktop/build-resources/tools/README.md)
has the download links.
