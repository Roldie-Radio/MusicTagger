"""Test fixtures.

Audio fixtures are synthesised with ffmpeg and numpy so the suite has no binary
files checked in and the signals are known exactly - which is the only way to
test a detector honestly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

# Point the app at a throwaway home *before* importing anything that reads it,
# so running the suite can never touch the real config, cache or journal.
# A fresh directory per run, not a fixed one: a fixed path carries the last
# run's cache and state into this one, and tests that expect to start empty
# then pass or fail depending on what ran before.
os.environ["MUSICTAGGER_HOME"] = tempfile.mkdtemp(prefix="musictagger-tests-")

import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402

from musictag.config import Config                                  # noqa: E402
from musictag.config import builtin_acoustid_key                    # noqa: E402

from musictag.providers.musicbrainz import MusicBrainzClient      # noqa: E402

#: The unpatched lookup, for the one test that checks it.
REAL_BUILTIN_ACOUSTID_KEY = builtin_acoustid_key
#: The unpatched genre lookup, for the tests of the lookup itself.
REAL_GENRE_FOR = MusicBrainzClient.genre_for

SR = 44100
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

@pytest.fixture(autouse=True)
def no_builtin_acoustid_key(monkeypatch):
    """Tests decide for themselves whether a key exists.

    A developer with ``musictag/_app_key.py`` or ``MUSICTAGGER_ACOUSTID_KEY``
    set would otherwise get different results - and real network lookups.
    """
    monkeypatch.setattr("musictag.config.builtin_acoustid_key", lambda: "")


@pytest.fixture(autouse=True)
def no_genre_lookups(monkeypatch):
    """Genre lookups are real network calls; tests that want one stub it."""
    monkeypatch.setattr("musictag.providers.musicbrainz.MusicBrainzClient.genre_for",
                        lambda self, tags: None)


ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Config:
    home = tmp_path_factory.mktemp("musictagger-home")
    config = Config()
    config.library_paths = []
    config.quality_max_seconds = 0
    config.organize_root = str(home / "library")
    return config


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _pink(n: int, rng: np.random.Generator, sr: int = SR, exponent: float = 0.9) -> np.ndarray:
    """Noise with a 1/f spectral tilt - roughly how real music is shaped."""
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    freqs[0] = freqs[1]
    spectrum /= freqs ** exponent
    return np.fft.irfft(spectrum, n)


def tone(seconds: float = 6.0, freq: float = 220.0, sr: int = SR,
         amplitude: float = 0.6, seed: int = 7) -> np.ndarray:
    """One channel of music-like audio: pink noise plus a harmonic series.

    The spectral tilt matters. Flat white noise makes a lossy encoder behave
    nothing like it does on music - it barely low-passes at all - so tests
    built on white noise would validate the detector against a signal it will
    never meet in a real library.
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    signal = _pink(n, rng, sr)
    signal /= np.abs(signal).max()
    for harmonic, level in ((1, 0.5), (2, 0.3), (4, 0.18), (8, 0.09), (16, 0.04)):
        signal += level * np.sin(2 * np.pi * freq * harmonic * t)
    signal /= np.abs(signal).max()
    return (signal * amplitude).astype(np.float32)


def stereo(seconds: float = 6.0, **kwargs) -> np.ndarray:
    """Two decorrelated channels, shape (n, 2).

    Encoders allocate bits per channel, so a mono fixture at 96 kbps behaves
    like a stereo one at 192 kbps and hides the low-pass we are testing for.
    """
    left = tone(seconds, seed=7, **kwargs)
    right = tone(seconds, seed=8, **kwargs)
    return np.stack([left, right], axis=1)


def fade_edges(samples: np.ndarray, seconds: float = 0.3, sr: int = SR) -> np.ndarray:
    n = int(seconds * sr)
    ramp = np.linspace(0, 1, n)
    if samples.ndim == 2:
        ramp = ramp[:, None]
    samples[:n] *= ramp
    samples[-n:] *= ramp[::-1]
    return samples


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    """Write float samples as 16-bit PCM WAV. Accepts (n,) or (n, channels)."""
    clipped = np.clip(samples, -1.0, 1.0)
    channels = clipped.shape[1] if clipped.ndim == 2 else 1
    ints = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(ints.reshape(-1).tobytes())
    return path


def encode(src: Path, dest: Path, *args: str) -> Path:
    """Run ffmpeg to transcode a fixture."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(dest)]
    subprocess.run(cmd, check=True, capture_output=True, creationflags=NO_WINDOW)
    return dest


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def clean_wav(tmp_path_factory) -> Path:
    """A well-behaved source: stereo, full spectrum, fades in and out, no clipping."""
    directory = tmp_path_factory.mktemp("audio")
    return write_wav(directory / "clean.wav", fade_edges(stereo(8.0)))


@pytest.fixture(scope="session")
def clipped_wav(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("audio")
    samples = tone(5.0, amplitude=1.0) * 1.8      # drive it hard into the ceiling
    return write_wav(directory / "clipped.wav", samples)


@pytest.fixture(scope="session")
def clicky_wav(tmp_path_factory) -> Path:
    """Clean audio with unmistakable single-sample discontinuities."""
    directory = tmp_path_factory.mktemp("audio")
    samples = tone(6.0, amplitude=0.4)
    for position in (int(1.0 * SR), int(2.5 * SR), int(4.0 * SR), int(5.2 * SR)):
        samples[position] = 0.98
        samples[position + 1] = -0.95
    return write_wav(directory / "clicky.wav", samples)


def _drum_hits(seconds: float, sr: int = SR, bpm: float = 128, seed: int = 3) -> np.ndarray:
    """Kick + snare on a steady grid: fast-attack, exponentially-decaying hits.

    This is what exposed the original click-detector false positive - real
    percussion has exactly the "big jump, sharply reversed" shape a naive
    click test looks for, and no amount of tonal/pink-noise fixture (which is
    all the original suite had) will ever produce it.
    """
    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    beat = 60.0 / bpm
    t_hit = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * 60 * t_hit) * np.exp(-t_hit * 35)
    snare = rng.standard_normal(t_hit.size) * np.exp(-t_hit * 25)
    for beat_i in range(int(seconds / beat)):
        pos = int(beat_i * beat * sr)
        hit = kick if beat_i % 2 == 0 else snare
        end = min(n, pos + hit.size)
        out[pos:end] += hit[: end - pos]
    return out


@pytest.fixture(scope="session")
def percussive_wav(tmp_path_factory) -> Path:
    """A loud, heavily-limited, drum-driven track with no damage at all.

    Modelled on what a real "loudness war" master looks like: a musical bed
    plus a steady kick/snare grid, pushed through a brick-wall-style limiter.
    Real files this shape (hip-hop, EDM, most modern pop/rock) are exactly
    what falsely tripped the original click detector - one commercial track
    tested against it registered 373 "clicks" at high severity despite having
    nothing wrong with it.
    """
    directory = tmp_path_factory.mktemp("audio")
    bed = tone(20.0, amplitude=0.5)
    mix = bed + _drum_hits(20.0) * 1.4
    mastered = np.tanh(mix / np.abs(mix).max() * 6.0) * 0.98
    return write_wav(directory / "percussive.wav", mastered)


@pytest.fixture(scope="session")
def truncated_wav(tmp_path_factory) -> Path:
    """Stops dead at genuinely full level - what a cut-off download looks like.

    The abrupt-end check was tightened to only fire when the file is still
    near its own peak level (RMS close to 0 dBFS) with zero decay, since
    merely "audible but not fading" turned out to flag ordinary songs that
    end on a sustained chord or hit. So the tail here has to actually be
    loud, not just present: a sustained, near-full-scale sine, cut off cold.
    """
    directory = tmp_path_factory.mktemp("audio")
    samples = tone(5.0, amplitude=0.5)
    fade = int(0.3 * SR)
    samples[:fade] *= np.linspace(0, 1, fade)     # normal start
    tail_n = int(1.0 * SR)
    t = np.arange(tail_n) / SR
    samples[-tail_n:] = 0.97 * np.sin(2 * np.pi * 220 * t)   # loud, sustained, no fade
    return write_wav(directory / "truncated.wav", samples)


@pytest.fixture(scope="session")
def gapped_wav(tmp_path_factory) -> Path:
    """Clean audio with a silent hole in the middle."""
    directory = tmp_path_factory.mktemp("audio")
    samples = tone(8.0, amplitude=0.5)
    samples[int(3.0 * SR):int(6.0 * SR)] = 0.0
    return write_wav(directory / "gapped.wav", samples)


@pytest.fixture(scope="session")
def mp3_128(tmp_path_factory, clean_wav) -> Path:
    directory = tmp_path_factory.mktemp("audio")
    return encode(clean_wav, directory / "128.mp3", "-codec:a", "libmp3lame", "-b:a", "128k")


@pytest.fixture(scope="session")
def mp3_320(tmp_path_factory, clean_wav) -> Path:
    directory = tmp_path_factory.mktemp("audio")
    return encode(clean_wav, directory / "320.mp3", "-codec:a", "libmp3lame", "-b:a", "320k")


@pytest.fixture(scope="session")
def fake_lossless_flac(tmp_path_factory, clean_wav) -> Path:
    """A FLAC made from a low-bitrate MP3 - lossless in name only."""
    directory = tmp_path_factory.mktemp("audio")
    lossy = encode(clean_wav, directory / "step.mp3", "-codec:a", "libmp3lame", "-b:a", "96k")
    return encode(lossy, directory / "fake.flac", "-codec:a", "flac")


@pytest.fixture(scope="session")
def real_flac(tmp_path_factory, clean_wav) -> Path:
    directory = tmp_path_factory.mktemp("audio")
    return encode(clean_wav, directory / "real.flac", "-codec:a", "flac")


@pytest.fixture(scope="session")
def upscaled_mp3(tmp_path_factory, clean_wav) -> Path:
    """96 kbps re-encoded to 320 kbps: big file, small sound."""
    directory = tmp_path_factory.mktemp("audio")
    lossy = encode(clean_wav, directory / "small.mp3", "-codec:a", "libmp3lame", "-b:a", "96k")
    return encode(lossy, directory / "upscaled.mp3", "-codec:a", "libmp3lame", "-b:a", "320k")


@pytest.fixture
def taggable(tmp_path, clean_wav, request):
    """A writable copy of a file in each taggable format."""
    fmt = getattr(request, "param", "mp3")
    dest = tmp_path / f"sample.{fmt}"
    codec = {"mp3": "libmp3lame", "flac": "flac", "m4a": "aac",
             "ogg": "libvorbis", "opus": "libopus"}[fmt]
    encode(clean_wav, dest, "-codec:a", codec)
    return dest
