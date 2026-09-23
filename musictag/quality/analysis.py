"""Detecting the things that make a music file unpleasant to listen to.

What this looks for, roughly in order of how often it actually matters:

* **Transcodes** - a "FLAC" that was made from a 128 kbps MP3, or an MP3
  re-encoded to 320. Lossy encoders low-pass the signal, and that shelf stays
  in the file forever. Finding the shelf and comparing it against what the
  claimed bitrate should produce is the single most reliable tell.
* **Truncated tracks** - songs that stop mid-note, usually from an interrupted
  download or a bad rip.
* **Clipping** - samples pinned at full scale, from over-loud masters or a
  botched volume normalisation.
* **Clicks, pops and dropouts** - single-sample discontinuities from scratched
  discs or damaged files, and runs of digital silence in the middle of a track.
* **Weak source quality** - genuinely low bitrate or sample rate.

Everything is measured, not guessed, and each issue carries the number that
triggered it so you can disagree with the threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Config
from ..models import SEVERITY_PENALTY, QualityIssue, QualityReport, Track
from .probe import DecodeError, FfmpegUnavailable, decode_samples, ffprobe_info

log = logging.getLogger(__name__)

# --- thresholds -------------------------------------------------------------

#: Roughly where each lossy codec's low-pass sits, in kHz, by bitrate.
#: Used to spot files whose spectrum does not match their claimed bitrate.
#: Calibrated against LAME 3.100 / ffmpeg output at 44.1 kHz stereo. Values are
#: the *measured* shelf, deliberately a little generous so honest files are
#: never accused - a false "this is a transcode" is worse than a missed one.
EXPECTED_CUTOFF_KHZ = {
    "mp3":    {64: 11.0, 96: 15.5, 128: 16.5, 160: 17.5, 192: 19.0, 256: 19.8, 320: 20.6},
    "aac":    {64: 13.0, 96: 15.5, 128: 16.5, 192: 18.5, 256: 19.5, 320: 20.0},
    "vorbis": {96: 15.0, 128: 17.0, 192: 19.0, 256: 19.5, 320: 20.0},
    "opus":   {64: 16.0, 96: 18.0, 128: 19.5, 192: 20.0, 256: 20.0},
    "wma":    {64: 12.0, 128: 15.5, 192: 17.5, 320: 19.0},
}

#: Below this, a lossy file is worth flagging regardless of codec.
LOW_BITRATE_KBPS = {"mp3": 160, "aac": 128, "vorbis": 128, "opus": 96, "wma": 160}
VERY_LOW_BITRATE_KBPS = {"mp3": 128, "aac": 96, "vorbis": 96, "opus": 64, "wma": 128}

CLIP_THRESHOLD = 0.9995        # |sample| at or above this counts as clipped

#: Clipping severity, as a fraction of all samples pinned at full scale.
#: Calibrated against 24 real commercial tracks: ordinary loud mastering
#: routinely touches full scale for a handful of isolated samples (ratios of
#: 0.01-0.02%, often with zero sustained runs) without being audible
#: distortion. The old thresholds (0.1% / 0.005%) flagged 14 of 24 real files,
#: including several with essentially no sustained runs at all. Genuinely
#: audible distortion needs a much bigger share of the file pinned down, not
#: a few peaks - and even then, sustained clipping is common, intentional
#: practice in some genres (hip-hop, EDM, industrial), not automatically a
#: broken file, hence the hedged wording below.
CLIP_HIGH_RATIO = 0.015        # >1.5% of all samples clipped
CLIP_MEDIUM_RATIO = 0.003      # >0.3%
CLIP_MEDIUM_RUNS = 500         # or this many sustained (4+ sample) runs
CLIP_LOW_RATIO = 0.0005        # >0.05% ...
CLIP_LOW_RUNS = 20             # ... together with at least this many runs
CLICK_SIGMA = 14.0             # robust-sigma multiplier for click detection
CLICK_BLOCK_S = 0.2            # window for the *local* click threshold, in seconds
#: Fewer events than this is noise, not a finding. Calibrated against 24 real
#: commercial tracks: even clean acoustic material showed 2-3 events from
#: ordinary lossy-encoding artefacts near transients, so the floor sits just
#: above that rather than at zero.
CLICK_MIN_COUNT = 4
CLICK_MEDIUM_PER_MINUTE = 15.0 # rate that separates "a few" from "frequent"
SILENCE_DB = -60.0             # below this is "silence" for edge trimming

#: A silent-seeming gap mid-track is at least as likely to be a real, mixed-in
#: pause (a dramatic beat drop, a breakdown, a bridge) as it is to be a
#: corrupted or interrupted download. Real files this size checked out fine
#: at gaps of 0.28-1.7s. A genuinely broken file - a dropped network
#: connection, a corrupt sector - tends to leave a much longer hole than that.
DROPOUT_MEDIUM_S = 2.0
DROPOUT_HIGH_S = 10.0

#: How loud the very end of the file has to still be, with no decay, to call
#: it an abrupt (possibly cut-off) ending. Checked against 24 real tracks:
#: -35 dBFS flagged two normal songs that simply end on a sustained chord or
#: hit instead of fading out - a completely ordinary mixing choice, not
#: evidence of missing audio. A file that is actually cut off tends to stop
#: while genuinely loud (near its own peak level), not merely audible.
ABRUPT_END_DB = -10.0
ABRUPT_START_DB = -15.0        # same idea, for a track starting already loud

#: Decoding keeps the whole file in memory as float32 - about 21 MB a minute
#: for 44.1 kHz stereo, before any working copies - and several files are
#: analysed at once. That is nothing for a song and ruinous for a two-hour
#: DJ mix or a ten-hour audiobook (over 12 GB). Past this length a file is
#: *sampled* instead: the opening SAMPLE_HEAD_S seconds for everything that
#: works as a rate or a share (clipping, clicks, spectrum, levels, dropouts),
#: plus the final SAMPLE_TAIL_S seconds, decoded separately, for the checks
#: that are about the real ending of the file.
LONG_FILE_S = 15 * 60
SAMPLE_HEAD_S = 8 * 60
SAMPLE_TAIL_S = 30


# ===========================================================================
# Public entry points
# ===========================================================================

def analyze_track(track: Track, cfg: Config, *,
                  reference_duration_s: Optional[float] = None) -> QualityReport:
    """Analyse a :class:`Track`, using its match for a reference duration."""
    if reference_duration_s is None and track.match and track.match.candidates:
        reference_duration_s = track.match.candidates[track.match.chosen_index].length_s
    return analyze_file(Path(track.path), cfg, reference_duration_s=reference_duration_s)


def analyze_file(path: Path, cfg: Config, *,
                 reference_duration_s: Optional[float] = None) -> QualityReport:
    report = QualityReport()
    try:
        info = ffprobe_info(path, cfg)
    except FfmpegUnavailable as exc:
        report.error = str(exc)
        return report
    except DecodeError as exc:
        report.analysed = True
        report.error = str(exc)
        report.issues.append(QualityIssue(
            "undecodable", "high", "File will not decode",
            f"ffprobe could not read this file: {exc}. It is probably corrupt."))
        report.score = 0
        return report

    codec = (info["codec"] or "").lower()
    lossless = codec in ("flac", "alac", "pcm_s16le", "pcm_s24le", "pcm_f32le", "wavpack", "ape")
    bitrate_kbps = int(round((info["stream_bitrate"] or info["format_bitrate"] or 0) / 1000))
    if not bitrate_kbps and info["duration"] and info["size"]:
        bitrate_kbps = int(round(info["size"] * 8 / info["duration"] / 1000))

    report.metrics.update({
        "codec": codec,
        "lossless": lossless,
        "bitrate_kbps": bitrate_kbps,
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "duration_s": round(info["duration"], 2),
        # Only lossless formats have a meaningful bit depth. For a lossy file
        # ffprobe reports the decoder's internal float format, and showing
        # "32-bit" next to a 128 kbps MP3 is actively misleading.
        "bit_depth": (info["bits_per_raw_sample"] or _depth_from_fmt(info["sample_fmt"]))
                     if lossless else 0,
    })

    _check_container(report, info, codec, lossless, bitrate_kbps)

    # --- decode and measure ---------------------------------------------
    # A user-set cap wins; otherwise long files are sampled (see LONG_FILE_S).
    sampled = not cfg.quality_max_seconds and (info["duration"] or 0) > LONG_FILE_S
    max_seconds = cfg.quality_max_seconds or (SAMPLE_HEAD_S if sampled else 0)
    try:
        samples, sample_rate = decode_samples(path, cfg, max_seconds=max_seconds)
    except FfmpegUnavailable as exc:
        report.error = str(exc)
        report.analysed = True
        report.score = _score(report)
        return report
    except DecodeError as exc:
        report.analysed = True
        report.issues.append(QualityIssue(
            "decode_error", "high", "Audio failed to decode cleanly",
            f"ffmpeg reported: {exc}. The file is likely damaged or incomplete."))
        report.score = _score(report)
        return report

    mono = _mono(samples)
    duration = len(mono) / sample_rate
    report.metrics["analysed_seconds"] = round(duration, 2)
    report.metrics["sampled"] = sampled

    tail: Optional[np.ndarray] = None
    if sampled:
        try:
            tail = _mono(decode_samples(path, cfg, from_end_s=SAMPLE_TAIL_S)[0])
        except (DecodeError, FfmpegUnavailable) as exc:
            # The body was analysed fine; only the end checks are lost.
            log.debug("Could not decode the end of %s: %s", path, exc)

    _check_levels(report, samples, mono)
    _check_clipping(report, samples)
    _check_clicks(report, mono, sample_rate)
    _check_silence_and_dropouts(report, mono, sample_rate, sampled=sampled, tail=tail)
    _check_truncation(report, mono, sample_rate, info["duration"], reference_duration_s, cfg,
                      sampled=sampled, tail=tail)
    _check_spectrum(report, mono, sample_rate, codec, lossless, bitrate_kbps)
    _check_channels(report, samples)

    report.analysed = True
    report.score = _score(report)
    return report


# ===========================================================================
# Individual checks
# ===========================================================================

def _mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1) if samples.ndim > 1 and samples.shape[1] > 1 else samples.reshape(-1)


def _quiet_frames(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Which 20 ms frames are below the silence floor, and the frame size."""
    win = max(1, int(0.02 * sample_rate))
    frames = mono[: (mono.size // win) * win].reshape(-1, win)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return frame_rms < _from_db(SILENCE_DB), win


def _check_container(report: QualityReport, info: dict[str, Any], codec: str,
                     lossless: bool, bitrate: int) -> None:
    if not lossless and bitrate:
        very_low = VERY_LOW_BITRATE_KBPS.get(codec, 128)
        low = LOW_BITRATE_KBPS.get(codec, 160)
        if bitrate < very_low:
            report.issues.append(QualityIssue(
                "very_low_bitrate", "high", f"Very low bitrate ({bitrate} kbps)",
                f"{bitrate} kbps {codec.upper()} is well below the {very_low} kbps "
                "where artefacts stop being obvious. Worth re-sourcing."))
        elif bitrate < low:
            report.issues.append(QualityIssue(
                "low_bitrate", "medium", f"Low bitrate ({bitrate} kbps)",
                f"{bitrate} kbps {codec.upper()} is listenable but below the "
                f"{low} kbps that most people consider transparent."))

    sample_rate = info["sample_rate"]
    if sample_rate and sample_rate < 44100:
        report.issues.append(QualityIssue(
            "low_sample_rate", "medium", f"Low sample rate ({sample_rate} Hz)",
            f"{sample_rate} Hz cuts everything above {sample_rate // 2} Hz. "
            "CD quality is 44100 Hz."))

    if info["channels"] == 1:
        report.issues.append(QualityIssue(
            "mono", "info", "Mono audio",
            "This file has a single channel. Fine for old recordings, "
            "unexpected for modern releases."))

    if info["duration"] and info["duration"] < 30:
        report.issues.append(QualityIssue(
            "very_short", "info", f"Very short ({info['duration']:.0f}s)",
            "Under 30 seconds - this may be an intro, a sample, or a partial download."))


def _check_levels(report: QualityReport, samples: np.ndarray, mono: np.ndarray) -> None:
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    peak_db = _db(peak)
    rms_db = _db(rms)
    crest = peak_db - rms_db if rms > 0 else 0.0

    report.metrics.update({
        "peak_dbfs": round(peak_db, 2),
        "rms_dbfs": round(rms_db, 2),
        "crest_factor_db": round(crest, 2),
    })

    dc = float(np.mean(mono)) if mono.size else 0.0
    report.metrics["dc_offset"] = round(dc, 5)
    if abs(dc) > 0.01:
        report.issues.append(QualityIssue(
            "dc_offset", "low", f"DC offset ({dc:+.3f})",
            "The waveform is not centred on zero. This wastes headroom and can "
            "cause a click at the start and end of playback."))

    if rms > 0 and crest < 6.0:
        report.issues.append(QualityIssue(
            "low_dynamic_range", "low", f"Very compressed ({crest:.1f} dB crest factor)",
            "Peaks are barely above the average level, which is the signature of "
            "heavy loudness-war mastering. Not damage, but it can sound fatiguing."))

    if peak_db < -6.0 and rms_db < -30.0:
        report.issues.append(QualityIssue(
            "very_quiet", "low", f"Very quiet (peak {peak_db:.1f} dBFS)",
            "This file is much quieter than a normal master and will need the "
            "volume turned up relative to the rest of your library."))


def _check_clipping(report: QualityReport, samples: np.ndarray) -> None:
    """Flag samples pinned at full scale - but only when there are enough of
    them to plausibly be audible, not merely present.

    See :data:`CLIP_HIGH_RATIO` and friends for the real-world calibration
    behind these numbers. Even at "high", the wording does not assert the
    file is broken: sustained clipping is standard practice in some genres.
    """
    if samples.size == 0:
        return
    flat = np.abs(samples).reshape(-1)
    clipped = flat >= CLIP_THRESHOLD
    count = int(clipped.sum())
    ratio = count / flat.size
    report.metrics["clipped_samples"] = count
    report.metrics["clipped_ratio"] = round(ratio, 8)

    if count == 0:
        return

    # Isolated full-scale samples are normal in a hot master; long *runs* of them
    # are what actually audibly distorts.
    runs = _run_lengths(clipped)
    long_runs = int((runs >= 4).sum()) if runs.size else 0
    report.metrics["clipped_runs"] = long_runs

    if ratio > CLIP_HIGH_RATIO:
        report.issues.append(QualityIssue(
            "clipping", "high", f"Heavy clipping ({ratio * 100:.2f}% of samples)",
            f"{count:,} samples are pinned at full scale, in {long_runs:,} sustained runs - "
            f"{ratio * 100:.1f}% of the whole file. That is enough to be audible as "
            "distortion, though some genres (hip-hop, EDM, industrial) master this way "
            "on purpose."))
    elif ratio > CLIP_MEDIUM_RATIO or long_runs > CLIP_MEDIUM_RUNS:
        report.issues.append(QualityIssue(
            "clipping", "medium", f"Some clipping ({ratio * 100:.2f}% of samples)",
            f"{count:,} samples hit full scale, in {long_runs:,} sustained runs. Could be "
            "audible on some equipment, but a lot of modern, loudly mastered music "
            "looks like this without sounding obviously broken."))
    elif ratio > CLIP_LOW_RATIO and long_runs >= CLIP_LOW_RUNS:
        report.issues.append(QualityIssue(
            "clipping", "low", f"Minor clipping ({ratio * 100:.3f}% of samples)",
            f"{count:,} samples touch full scale, in {long_runs:,} short runs. "
            "Typical of a hot but otherwise normal master; unlikely to be audible."))


def _local_diff_sigma(diff: np.ndarray, sample_rate: int, block_s: float = CLICK_BLOCK_S) -> np.ndarray:
    """Robust noise-floor of ``diff``, estimated separately per short block.

    A single sigma for the whole track is wrong for essentially all real
    music: a quiet verse and a loud chorus do not share a noise floor, so a
    threshold calibrated on the quiet part is far too loose once the chorus
    (or any drum hit) arrives. Splitting into ~200ms blocks lets each moment
    get judged against its own surroundings instead of the whole song.
    """
    block = max(64, int(block_s * sample_rate))
    n_blocks = diff.size // block
    if n_blocks < 1:
        return np.full(diff.size, max(float(np.std(diff)), 1e-7))
    trimmed = diff[: n_blocks * block].reshape(n_blocks, block)
    median = np.median(trimmed, axis=1, keepdims=True)
    mad = np.median(np.abs(trimmed - median), axis=1, keepdims=True)
    sigma = np.maximum(1.4826 * mad, 1e-7)
    out = np.repeat(sigma[:, 0], block)
    if out.size < diff.size:
        out = np.concatenate([out, np.full(diff.size - out.size, sigma[-1, 0])])
    return out


def _check_clicks(report: QualityReport, mono: np.ndarray, sample_rate: int) -> None:
    """Find single-sample discontinuities - the clicks and pops.

    A click is a step the waveform cannot physically have made: a large jump
    immediately reversed by another large jump. That test alone is not enough,
    though - a real drum hit's fast attack and decay produces exactly the same
    shape in the derivative, and no amount of extra cleverness here fully
    tells the two apart. This was checked against 24 real, commercially
    released tracks: with a single global threshold for the whole file, 13 of
    24 were flagged, including a gentle Beatles ballad (22 "clicks") and one
    hot, bass-heavy hip-hop track that hit 373. None of them are damaged.

    Two things are done about that, neither of which claims to be a perfect
    fix:

    1. The threshold is computed *locally* (~200ms blocks) instead of once for
       the whole file, which is a real, measurable improvement on anything
       with dynamic range - the Beatles track above dropped from 22 to 2.
    2. This check can no longer reach "high" severity, and needs considerably
       more evidence before it says anything at all. A loud, heavily limited,
       transient-dense master (a lot of modern hip-hop and EDM) can still
       legitimately clear these thresholds from its ordinary drum hits - that
       case cannot be reliably separated from real damage by this kind of
       heuristic, so the finding is worded to say so rather than asserting
       damage the evidence does not actually support.
    """
    if mono.size < sample_rate:
        return
    diff = np.diff(mono)
    if diff.size < 3:
        return

    sigma = _local_diff_sigma(diff, sample_rate)[:-1]
    threshold = np.maximum(CLICK_SIGMA * sigma, 0.08)

    big = np.abs(diff[:-1]) > threshold
    reversal = (np.sign(diff[:-1]) != np.sign(diff[1:])) & (np.abs(diff[1:]) > threshold * 0.5)
    spikes = np.flatnonzero(big & reversal)

    if spikes.size == 0:
        report.metrics["clicks"] = 0
        return

    # Collapse spikes closer than 20 ms into a single event.
    gap = max(1, int(0.02 * sample_rate))
    events = spikes[np.insert(np.diff(spikes) > gap, 0, True)]
    count = int(events.size)
    report.metrics["clicks"] = count
    report.metrics["click_times_s"] = [round(float(i) / sample_rate, 2) for i in events[:12]]

    if count < CLICK_MIN_COUNT:
        return          # a handful of events is well within normal variation

    per_minute = count / max(1.0, mono.size / sample_rate / 60.0)
    report.metrics["clicks_per_minute"] = round(per_minute, 1)
    if per_minute >= CLICK_MEDIUM_PER_MINUTE:
        report.issues.append(QualityIssue(
            "clicks", "medium", f"Frequent sharp transients ({count} found, {per_minute:.0f}/min)",
            f"{count} sample-level discontinuities, about {per_minute:.0f} per minute. "
            "This can mean a damaged or corrupted file - but a loud, bass-heavy, heavily "
            "limited master produces the same pattern from ordinary drum hits. If the "
            "track sounds fine to you, trust your ears over this number."))
    else:
        report.issues.append(QualityIssue(
            "clicks", "low", f"A few sharp transients ({count} found)",
            f"{count} sample-level discontinuities, first around "
            f"{report.metrics['click_times_s'][0]:.1f}s. Common in percussive or loudly "
            "mastered music and rarely audible as damage."))


def _check_silence_and_dropouts(report: QualityReport, mono: np.ndarray, sample_rate: int,
                                *, sampled: bool = False,
                                tail: Optional[np.ndarray] = None) -> None:
    """Silent lead-in/tail, and gaps inside the track.

    When the file is ``sampled``, ``mono`` stops somewhere mid-file, so its
    end says nothing about the file's end: trailing silence is measured on
    ``tail`` (the real last seconds) instead, or not at all without one.
    """
    if mono.size < sample_rate:
        return
    quiet, win = _quiet_frames(mono, sample_rate)

    lead = int(np.argmax(~quiet)) if (~quiet).any() else len(quiet)
    if sampled:
        trail = 0                         # the head ends mid-file, not at its end
    else:
        trail = int(np.argmax(~quiet[::-1])) if (~quiet).any() else 0
    lead_s = lead * win / sample_rate
    trail_s: Optional[float] = trail * win / sample_rate
    if sampled:
        trail_s = None
        if tail is not None and tail.size >= sample_rate:
            tail_quiet, _ = _quiet_frames(tail, sample_rate)
            tail_trail = (int(np.argmax(~tail_quiet[::-1])) if (~tail_quiet).any()
                          else len(tail_quiet))
            trail_s = tail_trail * win / sample_rate
    report.metrics["leading_silence_s"] = round(lead_s, 2)
    if trail_s is not None:
        report.metrics["trailing_silence_s"] = round(trail_s, 2)

    if lead_s > 3.0:
        report.issues.append(QualityIssue(
            "leading_silence", "low", f"{lead_s:.1f}s of silence at the start",
            "A long silent lead-in usually means a badly split track."))
    if trail_s is not None and trail_s > 10.0:
        report.issues.append(QualityIssue(
            "trailing_silence", "low", f"{trail_s:.1f}s of silence at the end",
            "A long silent tail wastes space and creates an awkward gap in playback."))

    # Silence *inside* the track is the interesting case.
    interior = quiet[lead: len(quiet) - trail] if len(quiet) - trail > lead else np.array([], bool)
    if interior.size:
        runs = _run_lengths(interior)
        if runs.size:
            longest_s = float(runs.max()) * win / sample_rate
            report.metrics["longest_internal_silence_s"] = round(longest_s, 2)
            if longest_s >= DROPOUT_HIGH_S:
                report.issues.append(QualityIssue(
                    "dropout", "high", f"{longest_s:.1f}s gap mid-track",
                    "A long stretch of near-silence in the middle of the song. "
                    "This is what a corrupt or partially downloaded file sounds like."))
            elif longest_s >= DROPOUT_MEDIUM_S:
                report.issues.append(QualityIssue(
                    "dropout", "medium", f"{longest_s:.1f}s quiet gap mid-track",
                    "A stretch of near-silence in the middle of the song. Could be a "
                    "damaged or interrupted file - but could just as easily be a real, "
                    "intentional pause (a breakdown, a dramatic beat drop)."))


def _check_truncation(report: QualityReport, mono: np.ndarray, sample_rate: int,
                      container_duration: float, reference_duration_s: Optional[float],
                      cfg: Config, *, sampled: bool = False,
                      tail: Optional[np.ndarray] = None) -> None:
    """Detect songs that stop before they finish.

    Two independent tests, because either alone produces false positives:

    * the file ends loud, with no fade or decay (a cold stop mid-note)
    * the file is meaningfully shorter than the matched release says it should be
    """
    if mono.size < sample_rate:
        return
    # Only meaningful when we decoded the whole thing - or, for a sampled
    # long file, when its real last seconds were decoded separately.
    partial = bool(cfg.quality_max_seconds and container_duration > cfg.quality_max_seconds)
    ending = mono
    if sampled:
        ending = tail if tail is not None and tail.size >= sample_rate else None

    if not partial and ending is not None:
        tail_n = int(0.05 * sample_rate)
        end_rms = float(np.sqrt(np.mean(np.square(ending[-tail_n:]))))
        end_db = _db(end_rms)
        # Compare the last 50 ms against the half-second before it: real endings decay.
        prev_n = int(0.5 * sample_rate)
        prev_rms = (float(np.sqrt(np.mean(np.square(ending[-prev_n:-tail_n]))))
                    if ending.size > prev_n else end_rms)
        decay_db = _db(prev_rms) - end_db
        report.metrics["end_level_dbfs"] = round(end_db, 2)
        report.metrics["end_decay_db"] = round(decay_db, 2)

        if end_db > ABRUPT_END_DB and decay_db < 3.0:
            report.issues.append(QualityIssue(
                "abrupt_end", "medium", f"Ends abruptly at {end_db:.0f} dBFS",
                "The track is still near full level in its final moments with no fade "
                "or decay. That can mean a cut-off file - but ending on a sustained "
                "note or chord instead of fading out is also a completely normal "
                "mixing choice, so treat this as worth a listen, not a verdict."))

    if not partial:
        start_n = int(0.05 * sample_rate)
        start_db = _db(float(np.sqrt(np.mean(np.square(mono[:start_n])))))
        report.metrics["start_level_dbfs"] = round(start_db, 2)
        if start_db > ABRUPT_START_DB:
            report.issues.append(QualityIssue(
                "abrupt_start", "low", f"Starts abruptly at {start_db:.0f} dBFS",
                "The very first samples are already loud. Could mean the beginning was "
                "clipped off, but plenty of tracks are simply mixed to start right on "
                "the beat with no lead-in."))

    if reference_duration_s and container_duration:
        delta = reference_duration_s - container_duration
        report.metrics["reference_duration_s"] = round(reference_duration_s, 2)
        report.metrics["duration_delta_s"] = round(delta, 2)
        if delta > 20:
            report.issues.append(QualityIssue(
                "truncated", "high", f"{delta:.0f}s shorter than the matched release",
                f"The database says this recording runs {reference_duration_s / 60:.1f} min "
                f"but the file is {container_duration / 60:.1f} min. Likely truncated."))
        elif delta > 6:
            report.issues.append(QualityIssue(
                "short_duration", "medium", f"{delta:.0f}s shorter than expected",
                "Shorter than the matched recording. It may be an edit, or it may be cut off."))
        elif delta < -20:
            report.issues.append(QualityIssue(
                "long_duration", "info", f"{abs(delta):.0f}s longer than expected",
                "Longer than the matched recording - often a hidden track or a "
                "different version than the one matched."))


def _check_spectrum(report: QualityReport, mono: np.ndarray, sample_rate: int,
                    codec: str, lossless: bool, bitrate: int) -> None:
    """Find the low-pass shelf and compare it with what the file claims to be."""
    detail = _spectral_cutoff_detailed(mono, sample_rate)
    if detail is None:
        return
    cutoff, drop_db = detail
    nyquist_khz = sample_rate / 2000.0
    cutoff_khz = cutoff / 1000.0
    report.metrics["spectral_cutoff_khz"] = round(cutoff_khz, 2)
    report.metrics["nyquist_khz"] = round(nyquist_khz, 2)
    report.metrics["shelf_drop_db"] = round(drop_db, 1)

    if lossless:
        # A genuine lossless file should carry content essentially up to Nyquist.
        if sample_rate >= 44100 and cutoff_khz < 19.0:
            report.issues.append(QualityIssue(
                "fake_lossless", "high",
                f"Lossless file with a {cutoff_khz:.1f} kHz ceiling",
                f"A real {codec.upper()} would have content up to about "
                f"{nyquist_khz:.1f} kHz. A hard shelf at {cutoff_khz:.1f} kHz means this "
                "was almost certainly encoded from a lossy source - it is lossless in "
                "name and file size only."))
        elif sample_rate >= 44100 and cutoff_khz < 20.5:
            report.issues.append(QualityIssue(
                "lossless_rolloff", "medium",
                f"Lossless file rolls off at {cutoff_khz:.1f} kHz",
                "Slightly low for true lossless. Could be an old or quiet recording, "
                "could be a high-bitrate transcode."))
        return

    expected = _expected_cutoff_khz(codec, bitrate)
    if expected is None:
        return
    report.metrics["expected_cutoff_khz"] = round(expected, 2)
    shortfall = expected - cutoff_khz

    if shortfall > 3.0:
        report.issues.append(QualityIssue(
            "transcode", "high",
            f"Spectrum suggests a lower bitrate than {bitrate} kbps",
            f"A {bitrate} kbps {codec.upper()} should reach about {expected:.1f} kHz, "
            f"but this rolls off at {cutoff_khz:.1f} kHz. The file was probably "
            "re-encoded upward from something smaller - the bitrate is real, the "
            "quality is not."))
    elif shortfall > 1.5:
        report.issues.append(QualityIssue(
            "possible_transcode", "medium",
            f"Rolls off at {cutoff_khz:.1f} kHz, expected about {expected:.1f} kHz",
            "A little low for the claimed bitrate. Possibly a transcode, possibly "
            "just a conservative encoder or a dull recording."))


def _check_channels(report: QualityReport, samples: np.ndarray) -> None:
    if samples.ndim < 2 or samples.shape[1] != 2:
        return
    left, right = samples[:, 0], samples[:, 1]
    if left.size == 0:
        return
    difference = float(np.abs(left - right).max())
    report.metrics["channel_difference"] = round(difference, 6)
    if difference < 1e-6:
        report.issues.append(QualityIssue(
            "fake_stereo", "info", "Stereo file with identical channels",
            "Both channels are bit-identical, so this is mono stored as stereo. "
            "It sounds the same but takes twice the space."))


# ===========================================================================
# Signal helpers
# ===========================================================================

#: A lossy low-pass is a cliff, not a slope: 70+ dB inside a few hundred Hz.
#: Real music rolls off gradually, so this threshold separates them cleanly.
SHELF_MIN_DROP_DB = 25.0

#: Lossy cutoffs live above this. Searching lower would find musical features.
SHELF_SEARCH_FROM_HZ = 9000.0

SHELF_GAP_HZ = 200.0       # ignore the transition band itself
SHELF_SPAN_HZ = 1500.0     # how much spectrum to average either side


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Moving average, used to stop single noisy bins from moving the answer."""
    if width < 2 or values.size < width:
        return values
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="same")


def _average_spectrum(mono: np.ndarray, sample_rate: int, *, n_fft: int,
                      max_frames: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Power spectrum in dB, averaged over the loud frames only.

    Silence has no high end to measure, so quiet frames would drag the top of
    the spectrum down and invent a shelf that is not there.
    """
    n_frames = mono.size // n_fft
    if n_frames < 4:
        return None

    frames = mono[: n_frames * n_fft].reshape(n_frames, n_fft)
    energies = np.sqrt(np.mean(np.square(frames), axis=1))
    if energies.max() <= 0:
        return None

    loud = np.flatnonzero(energies > energies.max() * 0.15)
    if loud.size < 4:
        loud = np.argsort(energies)[-min(n_frames, max_frames):]
    if loud.size > max_frames:
        loud = loud[np.linspace(0, loud.size - 1, max_frames).astype(int)]

    window = np.hanning(n_fft)
    accum = np.zeros(n_fft // 2 + 1)
    for i in loud:
        accum += np.square(np.abs(np.fft.rfft(frames[i] * window)))
    accum /= max(1, loud.size)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    power_db = 10.0 * np.log10(accum + 1e-20)
    bin_hz = sample_rate / n_fft
    power_db = _smooth(power_db, max(3, int(150 / bin_hz) | 1))   # ~150 Hz
    return freqs, power_db


def _find_cliff(freqs: np.ndarray, power_db: np.ndarray) -> tuple[float, float]:
    """Locate the steepest sustained drop in the top of the spectrum.

    Returns ``(cutoff_hz, drop_db)``. When no cliff clears
    :data:`SHELF_MIN_DROP_DB`, the cutoff is Nyquist - the file was not
    low-passed.

    Measuring the *edge* rather than an absolute level is what makes this work
    across quiet recordings, loud ones and every spectral tilt in between. An
    earlier threshold-based version was fooled by MP3 decoder leakage sitting
    above true digital silence.
    """
    nyquist = float(freqs[-1])
    bin_hz = float(freqs[1] - freqs[0])
    if bin_hz <= 0:
        return nyquist, 0.0

    search_from = SHELF_SEARCH_FROM_HZ if nyquist > SHELF_SEARCH_FROM_HZ + 2000 \
        else nyquist * 0.4

    gap = max(1, int(SHELF_GAP_HZ / bin_hz))
    span = max(gap + 2, int(SHELF_SPAN_HZ / bin_hz))
    step = max(1, int(50.0 / bin_hz))            # 50 Hz resolution is plenty
    start = int(search_from / bin_hz)
    stop = len(freqs) - gap - 2
    if start >= stop:
        return nyquist, 0.0

    best_index, best_drop = -1, 0.0
    for i in range(start, stop, step):
        below = power_db[max(0, i - span): i - gap]
        above = power_db[i + gap: min(len(power_db), i + span)]
        if below.size < 5 or above.size < 5:
            continue
        drop = float(np.median(below) - np.median(above))
        if drop > best_drop:
            best_drop, best_index = drop, i

    if best_index < 0 or best_drop < SHELF_MIN_DROP_DB:
        return nyquist, best_drop
    return float(freqs[best_index]), best_drop


def _spectral_cutoff(mono: np.ndarray, sample_rate: int, *,
                     n_fft: int = 8192, max_frames: int = 300
                     ) -> Optional[float]:
    """Frequency in Hz above which the file has essentially no content."""
    result = _spectral_cutoff_detailed(mono, sample_rate, n_fft=n_fft, max_frames=max_frames)
    return None if result is None else result[0]


def _spectral_cutoff_detailed(mono: np.ndarray, sample_rate: int, *,
                              n_fft: int = 8192, max_frames: int = 300
                              ) -> Optional[tuple[float, float]]:
    """As :func:`_spectral_cutoff`, but also returns the size of the cliff."""
    if mono.size < n_fft * 4:
        return None
    spectrum = _average_spectrum(mono, sample_rate, n_fft=n_fft, max_frames=max_frames)
    if spectrum is None:
        return None
    freqs, power_db = spectrum
    return _find_cliff(freqs, power_db)


def _expected_cutoff_khz(codec: str, bitrate: int) -> Optional[float]:
    table = EXPECTED_CUTOFF_KHZ.get(codec)
    if not table or not bitrate:
        return None
    points = sorted(table.items())
    if bitrate <= points[0][0]:
        return points[0][1]
    if bitrate >= points[-1][0]:
        return points[-1][1]
    for (b0, c0), (b1, c1) in zip(points, points[1:]):
        if b0 <= bitrate <= b1:
            span = b1 - b0
            return c0 + (c1 - c0) * ((bitrate - b0) / span) if span else c0
    return None


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of every run of ``True`` in a boolean array."""
    if mask.size == 0 or not mask.any():
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[::2]


def _db(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-10))


def _from_db(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _depth_from_fmt(sample_fmt: str) -> int:
    return {"s16": 16, "s16p": 16, "s32": 32, "s32p": 32,
            "flt": 32, "fltp": 32, "dbl": 64, "dblp": 64}.get(sample_fmt, 0)


def _score(report: QualityReport) -> int:
    """Roll the issues up into a single 0-100 number."""
    score = 100
    for issue in report.issues:
        score -= SEVERITY_PENALTY.get(issue.severity, 0)
    return max(0, min(100, score))
