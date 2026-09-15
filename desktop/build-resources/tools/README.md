# Bundled tool binaries

`ffmpeg.exe`, `ffprobe.exe` and `fpcalc.exe` are downloaded, not authored, and
are gitignored (~200MB combined). Fetch them fresh with:

```bash
curl -L -o ffmpeg-essentials.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
```

Unzip it and copy `bin/ffmpeg.exe` and `bin/ffprobe.exe` here. The
**essentials** build, not **full**, is what you want - it's a third the size
(~200MB vs ~650MB for both binaries) and MusicTagger only ever decodes common
formats and runs the `ebur128` filter, none of which need the full build's
extra encoders, subtitle libraries or hardware-acceleration backends.

```bash
curl -L -o fpcalc.zip https://github.com/acoustid/chromaprint/releases/download/v1.5.1/chromaprint-fpcalc-1.5.1-windows-x86_64.zip
```

Unzip and copy `fpcalc.exe` here. This is the exact same pinned release
`musictag/fingerprint.py`'s in-app "Download fpcalc" button uses, so the
version bundled here always matches what the app would fetch on its own.
