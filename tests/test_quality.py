"""Verify the quality detectors against signals with known defects.

Each fixture has exactly one thing wrong with it, so a detector firing here is
evidence it works, and a detector firing on ``clean`` is evidence it does not.
"""

from __future__ import annotations

import numpy as np

from musictag.quality.analysis import (
    _expected_cutoff_khz, _run_lengths, _spectral_cutoff, analyze_file,
)
from musictag.quality.probe import decode_samples, ffprobe_info

from conftest import SR, _drum_hits, ffmpeg_required, tone, write_wav


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


# ===========================================================================
# Pure signal helpers - no ffmpeg needed
# ===========================================================================

class TestSpectralCutoff:
    def test_full_band_noise_reports_near_nyquist(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(SR * 4).astype(np.float32) * 0.3
        cutoff = _spectral_cutoff(noise, SR)
        assert cutoff is not None
        assert cutoff > 20000, "white noise has content all the way up"

    def test_low_passed_signal_reports_the_shelf(self):
        """Build a signal with nothing above 12 kHz and check we find the edge."""
        rng = np.random.default_rng(1)
        n = SR * 4
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1 / SR)
        spectrum[freqs > 12000] = 0
        signal = np.fft.irfft(spectrum, n).astype(np.float32)
        signal /= np.abs(signal).max()

        cutoff = _spectral_cutoff(signal, SR)
        assert cutoff is not None
        assert 11000 < cutoff < 13500, f"expected a shelf near 12 kHz, got {cutoff}"

    def test_too_short_returns_none(self):
        assert _spectral_cutoff(np.zeros(100, dtype=np.float32), SR) is None


class TestExpectedCutoff:
    def test_known_bitrates(self):
        # Calibrated against measured LAME output, not folklore.
        assert _expected_cutoff_khz("mp3", 320) == 20.6
        assert _expected_cutoff_khz("mp3", 128) == 16.5

    def test_interpolates_between_points(self):
        value = _expected_cutoff_khz("mp3", 224)
        assert 19.0 < value < 19.5

    def test_clamps_outside_the_table(self):
        assert _expected_cutoff_khz("mp3", 32) == 11.0
        assert _expected_cutoff_khz("mp3", 500) == 20.6

    def test_unknown_codec_is_none(self):
        assert _expected_cutoff_khz("dsd", 320) is None


class TestRunLengths:
    def test_counts_consecutive_trues(self):
        mask = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        assert sorted(_run_lengths(mask).tolist()) == [2, 3]

    def test_empty_and_all_false(self):
        assert _run_lengths(np.array([], dtype=bool)).size == 0
        assert _run_lengths(np.zeros(10, dtype=bool)).size == 0

    def test_all_true_is_one_run(self):
        assert _run_lengths(np.ones(7, dtype=bool)).tolist() == [7]


# ===========================================================================
# End-to-end analysis
# ===========================================================================

@ffmpeg_required
class TestCleanBaseline:
    """The false-positive check. A good file must come back quiet."""

    def test_clean_flac_scores_well(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        assert report.analysed
        assert report.error is None
        assert report.score >= 85, f"clean file flagged: {[i.title for i in report.issues]}"

    def test_clean_flac_has_no_serious_findings(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        serious = [i for i in report.issues if i.severity in ("high", "medium")]
        assert not serious, f"false positives: {[i.code for i in serious]}"

    def test_clean_flac_is_not_called_a_transcode(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        assert "fake_lossless" not in codes(report)

    def test_clean_file_is_not_reported_as_clipping(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        assert "clipping" not in codes(report)

    def test_clean_file_has_no_clicks(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        assert report.metrics.get("clicks", 0) == 0


@ffmpeg_required
class TestTranscodeDetection:
    def test_flac_made_from_an_mp3_is_caught(self, cfg, fake_lossless_flac):
        report = analyze_file(fake_lossless_flac, cfg)
        assert "fake_lossless" in codes(report), (
            f"cutoff was {report.metrics.get('spectral_cutoff_khz')} kHz; "
            f"issues: {[i.code for i in report.issues]}"
        )

    def test_fake_lossless_is_a_high_severity_finding(self, cfg, fake_lossless_flac):
        report = analyze_file(fake_lossless_flac, cfg)
        issue = next(i for i in report.issues if i.code == "fake_lossless")
        assert issue.severity == "high"
        assert report.score < 80

    def test_the_finding_reports_the_measured_cutoff(self, cfg, fake_lossless_flac):
        report = analyze_file(fake_lossless_flac, cfg)
        cutoff = report.metrics["spectral_cutoff_khz"]
        assert 12 < cutoff < 19, f"96 kbps should roll off well below 19 kHz, got {cutoff}"

    def test_96k_upscaled_to_320_is_caught(self, cfg, upscaled_mp3):
        report = analyze_file(upscaled_mp3, cfg)
        found = codes(report)
        assert "transcode" in found or "possible_transcode" in found, (
            f"cutoff {report.metrics.get('spectral_cutoff_khz')} vs expected "
            f"{report.metrics.get('expected_cutoff_khz')}"
        )

    def test_an_honest_320_mp3_is_not_accused(self, cfg, mp3_320):
        report = analyze_file(mp3_320, cfg)
        assert "transcode" not in codes(report)

    def test_an_honest_128_mp3_is_not_accused_of_transcoding(self, cfg, mp3_128):
        report = analyze_file(mp3_128, cfg)
        assert "transcode" not in codes(report)


@ffmpeg_required
class TestBitrate:
    def test_128k_mp3_is_flagged_as_low(self, cfg, mp3_128):
        report = analyze_file(mp3_128, cfg)
        assert {"low_bitrate", "very_low_bitrate"} & codes(report)

    def test_320k_mp3_is_not_flagged(self, cfg, mp3_320):
        report = analyze_file(mp3_320, cfg)
        assert not ({"low_bitrate", "very_low_bitrate"} & codes(report))

    def test_bitrate_is_reported_in_metrics(self, cfg, mp3_320):
        report = analyze_file(mp3_320, cfg)
        assert 280 <= report.metrics["bitrate_kbps"] <= 340


@ffmpeg_required
class TestClipping:
    def test_hard_clipped_audio_is_detected(self, cfg, clipped_wav):
        report = analyze_file(clipped_wav, cfg)
        assert "clipping" in codes(report)

    def test_clipping_severity_reflects_how_much(self, cfg, clipped_wav):
        report = analyze_file(clipped_wav, cfg)
        issue = next(i for i in report.issues if i.code == "clipping")
        assert issue.severity == "high"
        assert report.metrics["clipped_samples"] > 1000

    def test_clipped_runs_are_counted(self, cfg, clipped_wav):
        report = analyze_file(clipped_wav, cfg)
        assert report.metrics["clipped_runs"] > 0


@ffmpeg_required
class TestClippingCalibration:
    """The regression suite for the second false-positive this session found:
    24 real commercial tracks showed 14 flagged for clipping, most from tiny
    ratios (0.01-0.02% of samples, several with *zero* sustained runs) that
    are just normal peak limiting, not audible distortion. Only one file
    (2.77% clipped) was genuinely extreme.
    """

    def test_a_handful_of_isolated_peaks_is_not_flagged(self, cfg, tmp_path):
        """Matches the real "Cochise"/"Purple Haze"/"Out of Control" case:
        a tiny fraction of samples touch full scale, mostly in isolation."""
        samples = tone(20.0, amplitude=0.6)
        rng = np.random.default_rng(11)
        # ~10 isolated full-scale samples scattered through the track - no
        # sustained runs, matching what ordinary peak limiting looks like.
        for pos in rng.choice(samples.size - 1, size=10, replace=False):
            samples[pos] = 1.0
        report = analyze_file(write_wav(tmp_path / "peaks.wav", samples), cfg)
        assert "clipping" not in codes(report)

    def test_a_justify_my_thug_sized_ratio_is_medium_not_high(self, cfg, tmp_path):
        """~0.8% of samples clipped, in real sustained runs - the exact shape
        of the real track that prompted this fix. Genuinely present, but not
        severe enough to call the file broken outright."""
        samples = tone(20.0, amplitude=0.6)
        n = samples.size
        run_len = 6
        rng = np.random.default_rng(12)
        target = int(n * 0.008)
        starts = rng.choice(n - run_len, size=max(1, target // run_len), replace=False)
        for s in starts:
            samples[s: s + run_len] = 1.0
        report = analyze_file(write_wav(tmp_path / "moderate_clip.wav", samples), cfg)
        issue = next((i for i in report.issues if i.code == "clipping"), None)
        assert issue is not None
        assert issue.severity == "medium"

    def test_only_a_lucifer_sized_ratio_reaches_high(self, cfg, tmp_path):
        """~2.8% clipped - the one real file out of 24 that genuinely
        warranted "high"."""
        samples = tone(20.0, amplitude=0.6)
        n = samples.size
        run_len = 6
        rng = np.random.default_rng(13)
        target = int(n * 0.028)
        starts = rng.choice(n - run_len, size=max(1, target // run_len), replace=False)
        for s in starts:
            samples[s: s + run_len] = 1.0
        report = analyze_file(write_wav(tmp_path / "heavy_clip.wav", samples), cfg)
        issue = next(i for i in report.issues if i.code == "clipping")
        assert issue.severity == "high"

    def test_high_severity_wording_acknowledges_genre_choices(self, cfg, clipped_wav):
        """Even "high" must not flatly assert the file is broken - sustained
        clipping is standard mastering practice in some genres."""
        report = analyze_file(clipped_wav, cfg)
        issue = next(i for i in report.issues if i.code == "clipping")
        assert issue.severity == "high"
        assert "genre" in issue.detail.lower() or "on purpose" in issue.detail.lower()


@ffmpeg_required
class TestClicks:
    def test_inserted_clicks_are_found(self, cfg, clicky_wav):
        report = analyze_file(clicky_wav, cfg)
        assert report.metrics["clicks"] >= 3, \
            f"expected the 4 inserted clicks, found {report.metrics.get('clicks')}"
        assert "clicks" in codes(report)

    def test_click_timestamps_are_reported(self, cfg, clicky_wav):
        report = analyze_file(clicky_wav, cfg)
        times = report.metrics["click_times_s"]
        assert times, "a click finding without a timestamp is not actionable"
        assert any(abs(t - 1.0) < 0.1 for t in times), f"expected one near 1.0s, got {times}"

    def test_a_handful_of_real_clicks_never_reaches_high_severity(self, cfg, clicky_wav):
        """Even genuine damage should not trigger the "likely damaged, replace
        it" framing this check can no longer make - see the module docstring
        for why that claim turned out not to be defensible."""
        report = analyze_file(clicky_wav, cfg)
        issue = next(i for i in report.issues if i.code == "clicks")
        assert issue.severity != "high"

    def test_a_couple_of_events_is_not_worth_a_finding(self):
        """Two or three sample-level discontinuities over a whole track is
        within the noise floor real lossy encoding already produces near
        transients - see the real-track survey behind CLICK_MIN_COUNT."""
        from musictag.models import QualityReport
        from musictag.quality.analysis import _check_clicks

        samples = tone(20.0, amplitude=0.4).astype(np.float32)
        for position in (int(5.0 * SR), int(12.0 * SR)):     # only 2
            samples[position] = 0.98
            samples[position + 1] = -0.95
        report = QualityReport()
        _check_clicks(report, samples, SR)
        assert report.metrics["clicks"] == 2
        assert "clicks" not in {i.code for i in report.issues}


@ffmpeg_required
class TestClicksOnRealMusic:
    """The regression suite for the false-positive this check used to produce.

    Checked directly against 24 real, commercially released tracks spanning
    acoustic, rock, punk and hip-hop: the original detector (one threshold for
    the whole file) flagged 13 of 24, including a Beatles ballad at 22
    "clicks" and a hip-hop track at 373, high severity, "usually means a
    damaged source". None of them are damaged. These fixtures reproduce the
    two failure modes that survey found, without depending on files this repo
    does not ship.
    """

    def test_a_loud_drum_driven_but_undamaged_track_never_reaches_high(self, cfg, percussive_wav):
        report = analyze_file(percussive_wav, cfg)
        click_issues = [i for i in report.issues if i.code == "clicks"]
        assert all(i.severity != "high" for i in click_issues), \
            "ordinary percussion must never be reported as 'likely damaged'"

    def test_a_loud_drum_driven_track_is_at_worst_medium(self, cfg, percussive_wav):
        report = analyze_file(percussive_wav, cfg)
        click_issues = [i for i in report.issues if i.code == "clicks"]
        assert all(i.severity in ("medium", "low") for i in click_issues)

    def test_the_medium_finding_does_not_assert_damage(self, cfg, percussive_wav):
        """It may still fire - a real drum grid does clear the rate threshold -
        but it must not overclaim what that means."""
        report = analyze_file(percussive_wav, cfg)
        issue = next((i for i in report.issues if i.code == "clicks"), None)
        if issue is not None:
            assert "trust your ears" in issue.detail.lower() or "can mean" in issue.detail.lower()

    def test_local_threshold_beats_global_on_dynamic_range(self):
        """The actual defect: one sigma for a whole track is calibrated on
        its quiet parts and is far too loose once a loud section arrives.
        A local (per-block) threshold must catch meaningfully fewer false
        events on the same audio than a single global one would."""
        from musictag.quality.analysis import CLICK_SIGMA, _local_diff_sigma

        quiet = tone(10.0, amplitude=0.15, seed=1)
        loud_with_real_drums = tone(10.0, amplitude=0.6, seed=2) + _drum_hits(10.0, seed=5) * 1.2
        mono = np.concatenate([quiet, loud_with_real_drums]).astype(np.float32)
        diff = np.diff(mono)

        global_sigma = 1.4826 * np.median(np.abs(diff - np.median(diff)))
        global_threshold = max(CLICK_SIGMA * global_sigma, 0.08)
        global_hits = int(np.sum(np.abs(diff[:-1]) > global_threshold))

        local_sigma = _local_diff_sigma(diff, SR)[:-1]
        local_threshold = np.maximum(CLICK_SIGMA * local_sigma, 0.08)
        local_hits = int(np.sum(np.abs(diff[:-1]) > local_threshold))

        assert local_hits < global_hits, (
            f"local threshold found {local_hits} raw crossings, "
            f"global found {global_hits} - local must be stricter on the loud half")


@ffmpeg_required
class TestTruncation:
    def test_a_track_that_stops_at_full_level_is_flagged(self, cfg, truncated_wav):
        report = analyze_file(truncated_wav, cfg)
        assert "abrupt_end" in codes(report)

    def test_an_abrupt_end_is_medium_not_high(self, cfg, truncated_wav):
        """Downgraded deliberately: level alone cannot tell a genuinely
        cut-off file apart from a song that was mixed to end on a hit."""
        report = analyze_file(truncated_wav, cfg)
        issue = next(i for i in report.issues if i.code == "abrupt_end")
        assert issue.severity == "medium"

    def test_a_faded_ending_is_not_flagged(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        assert "abrupt_end" not in codes(report)

    def test_ending_on_a_sustained_but_moderate_note_is_not_flagged(self, cfg, tmp_path):
        """Matches two real songs ("Adam's Song", "Brass Monkey") that end on
        a held chord/hit around -16 to -23 dBFS with little decay, and are not
        actually cut off - they were simply not mixed to fade out."""
        samples = tone(6.0, amplitude=0.5)
        tail_n = int(1.0 * SR)
        t = np.arange(tail_n) / SR
        # ~-20 dBFS RMS tail: audible and undecayed, but well under full scale.
        samples[-tail_n:] = 0.1 * np.sin(2 * np.pi * 220 * t)
        report = analyze_file(write_wav(tmp_path / "hard_stop.wav", samples), cfg)
        assert "abrupt_end" not in codes(report)

    def test_shorter_than_the_reference_is_flagged(self, cfg, real_flac):
        # The file is ~6s; claim the real recording is 60s.
        report = analyze_file(real_flac, cfg, reference_duration_s=60.0)
        assert "truncated" in codes(report)
        assert report.metrics["duration_delta_s"] > 20

    def test_matching_the_reference_is_not_flagged(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg, reference_duration_s=8.0)
        assert "truncated" not in codes(report)
        assert "short_duration" not in codes(report)


@ffmpeg_required
class TestDropouts:
    def test_a_silent_hole_mid_track_is_found(self, cfg, gapped_wav):
        report = analyze_file(gapped_wav, cfg)
        assert "dropout" in codes(report)
        assert report.metrics["longest_internal_silence_s"] > 2.0

    def test_a_three_second_gap_is_medium_not_high(self, cfg, gapped_wav):
        """10s+ is "high"; a few seconds is worth a note, not an alarm."""
        report = analyze_file(gapped_wav, cfg)
        issue = next(i for i in report.issues if i.code == "dropout")
        assert issue.severity == "medium"

    def test_continuous_audio_has_no_dropout(self, cfg, real_flac):
        report = analyze_file(real_flac, cfg)
        assert "dropout" not in codes(report)

    def test_a_short_musical_pause_is_not_flagged(self, cfg, tmp_path):
        """Matches three real songs (gaps of 0.28-0.94s) that were previously
        flagged for what is almost certainly just a dramatic pause in the
        arrangement, not a damaged file."""
        samples = tone(10.0, amplitude=0.5)
        samples[int(4.0 * SR): int(4.9 * SR)] = 0.0     # 0.9s pause
        report = analyze_file(write_wav(tmp_path / "pause.wav", samples), cfg)
        assert "dropout" not in codes(report)


@ffmpeg_required
class TestFailureModes:
    def test_a_corrupt_file_is_reported_not_raised(self, cfg, tmp_path):
        broken = tmp_path / "broken.mp3"
        broken.write_bytes(b"this is definitely not an mp3" * 100)
        report = analyze_file(broken, cfg)
        assert report.error or report.issues
        assert report.score < 100

    def test_missing_ffmpeg_is_reported_cleanly(self, real_flac):
        from musictag.config import Config
        broken_cfg = Config()
        broken_cfg.ffprobe_path = str(real_flac)   # exists, but is not ffprobe
        broken_cfg.ffmpeg_path = str(real_flac)
        report = analyze_file(real_flac, broken_cfg)
        # Either it fails to probe or fails to decode; both must be captured.
        assert report.error is not None or report.issues

    def test_silence_does_not_crash_the_analysers(self, cfg, tmp_path):
        silent = write_wav(tmp_path / "silence.wav", np.zeros(SR * 3, dtype=np.float32))
        report = analyze_file(silent, cfg)
        assert report.analysed


@ffmpeg_required
class TestProbe:
    def test_ffprobe_reports_the_real_codec(self, cfg, mp3_320):
        info = ffprobe_info(mp3_320, cfg)
        assert info["codec"] == "mp3"
        assert info["sample_rate"] == 44100

    def test_decode_returns_the_native_rate(self, cfg, real_flac):
        samples, rate = decode_samples(real_flac, cfg)
        assert rate == 44100, "resampling would destroy the spectral evidence"
        assert samples.shape[0] > SR

    def test_max_seconds_limits_the_decode(self, cfg, real_flac):
        from musictag.config import Config
        limited = Config()
        limited.ffmpeg_path = cfg.ffmpeg_path
        samples, rate = decode_samples(real_flac, limited, max_seconds=2)
        assert samples.shape[0] <= rate * 2.5


class TestScoreExplainability:
    """The UI shows users the arithmetic, so the arithmetic must hold."""

    def _report(self, *severities):
        from musictag.models import QualityIssue, QualityReport
        from musictag.quality.analysis import _score
        report = QualityReport()
        report.issues = [QualityIssue(f"c{i}", sev, "t", "d")
                         for i, sev in enumerate(severities)]
        report.score = _score(report)
        return report

    def test_a_clean_file_scores_full_marks(self):
        assert self._report().score == 100

    def test_each_severity_deducts_its_published_penalty(self):
        from musictag.models import SEVERITY_PENALTY
        for severity, penalty in SEVERITY_PENALTY.items():
            assert self._report(severity).score == 100 - penalty

    def test_penalties_add_up(self):
        from musictag.models import SEVERITY_PENALTY
        expected = 100 - SEVERITY_PENALTY["high"] - SEVERITY_PENALTY["low"]
        assert self._report("high", "low").score == expected

    def test_notes_do_not_change_the_score(self):
        assert self._report("info", "info", "info").score == 100

    def test_score_never_goes_negative(self):
        assert self._report(*(["high"] * 20)).score == 0

    def test_every_issue_the_analyser_raises_is_explainable(self):
        """A severity with no published label would render as a raw key."""
        from musictag.models import SEVERITY_INFO, SEVERITY_PENALTY, quality_scale
        published = {s["key"] for s in quality_scale()["severities"]}
        assert published == set(SEVERITY_PENALTY) == set(SEVERITY_INFO)
