"""Talking to ffmpeg/ffprobe: container facts and decoded PCM."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Config

log = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FfmpegUnavailable(Exception):
    """ffmpeg or ffprobe could not be found or run."""


class DecodeError(Exception):
    """The file could not be decoded - itself a quality finding."""


def ffprobe_info(path: Path, cfg: Config) -> dict[str, Any]:
    """Container/stream facts straight from ffprobe.

    These are more trustworthy than the tag library's view: ffprobe reports the
    *actual* decoded stream, which is how we catch files whose extension lies.
    """
    ffprobe = cfg.ffprobe
    if not ffprobe:
        raise FfmpegUnavailable("ffprobe not found. Install ffmpeg or set its path in Settings.")

    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "a:0", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError(f"ffprobe timed out on {path.name}") from exc
    except OSError as exc:
        raise FfmpegUnavailable(f"Could not run ffprobe: {exc}") from exc

    if proc.returncode != 0:
        raise DecodeError(proc.stderr.decode("utf-8", "ignore").strip()[:300] or "ffprobe failed")

    try:
        data = json.loads(proc.stdout.decode("utf-8", "ignore"))
    except json.JSONDecodeError as exc:
        raise DecodeError("ffprobe returned unparseable output") from exc

    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}

    def _int(value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    return {
        "codec": stream.get("codec_name") or "",
        "codec_long": stream.get("codec_long_name") or "",
        "profile": stream.get("profile") or "",
        "sample_rate": _int(stream.get("sample_rate")),
        "channels": _int(stream.get("channels")),
        "bits_per_raw_sample": _int(stream.get("bits_per_raw_sample")),
        "sample_fmt": stream.get("sample_fmt") or "",
        "stream_bitrate": _int(stream.get("bit_rate")),
        "format_bitrate": _int(fmt.get("bit_rate")),
        "duration": float(fmt.get("duration") or stream.get("duration") or 0.0),
        "format_name": fmt.get("format_name") or "",
        "size": _int(fmt.get("size")),
    }


def decode_samples(path: Path, cfg: Config, *, max_seconds: int = 0,
                   max_channels: int = 2) -> tuple[np.ndarray, int]:
    """Decode to float32 PCM at the file's native sample rate.

    Returns ``(samples, sample_rate)`` where ``samples`` has shape
    ``(n_frames, n_channels)``. The native rate is preserved deliberately:
    resampling would destroy the high-frequency shelf that reveals whether a
    "lossless" file was really transcoded from an MP3.
    """
    ffmpeg = cfg.ffmpeg
    if not ffmpeg:
        raise FfmpegUnavailable("ffmpeg not found. Install ffmpeg or set its path in Settings.")

    info = ffprobe_info(path, cfg)
    sample_rate = info["sample_rate"] or 44100
    channels = min(info["channels"] or 1, max_channels) or 1

    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(int(max_seconds))]
    cmd += [
        "-i", str(path),
        "-map", "0:a:0",
        "-ac", str(channels),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError(f"Decoding timed out on {path.name}") from exc
    except OSError as exc:
        raise FfmpegUnavailable(f"Could not run ffmpeg: {exc}") from exc

    stderr = proc.stderr.decode("utf-8", "ignore").strip()
    if proc.returncode != 0 and not proc.stdout:
        raise DecodeError(stderr[:300] or "ffmpeg failed to decode")

    raw = np.frombuffer(proc.stdout, dtype="<f4")
    if raw.size == 0:
        raise DecodeError("Decoded to zero samples")

    usable = (raw.size // channels) * channels
    samples = raw[:usable].reshape(-1, channels)
    # ffmpeg can emit non-finite values from damaged frames; treat them as silence
    # rather than letting them poison every statistic downstream.
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    return samples, sample_rate


def loudness_ebur128(path: Path, cfg: Config) -> Optional[dict[str, float]]:
    """Integrated loudness and true peak via ffmpeg's EBU R128 filter.

    Optional: returns ``None`` if the filter is unavailable rather than failing
    the whole analysis.
    """
    ffmpeg = cfg.ffmpeg
    if not ffmpeg:
        return None
    cmd = [ffmpeg, "-v", "info", "-nostdin", "-i", str(path),
           "-filter_complex", "ebur128=peak=true", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300, creationflags=_NO_WINDOW)
    except (subprocess.TimeoutExpired, OSError):
        return None

    text = proc.stderr.decode("utf-8", "ignore")
    tail = text[-2000:]
    out: dict[str, float] = {}
    for label, key in (("I:", "lufs"), ("LRA:", "lra"), ("Peak:", "true_peak_db")):
        idx = tail.rfind(label)
        if idx == -1:
            continue
        chunk = tail[idx + len(label):].strip().split()
        if chunk:
            try:
                out[key] = float(chunk[0])
            except ValueError:
                pass
    return out or None
