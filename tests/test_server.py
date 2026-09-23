"""API surface the UI depends on.

The front end is plain JavaScript against these routes, so a broken contract
here shows up as a silently empty screen rather than an exception.
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from musictag.models import MatchResult, Track, TrackTags
from musictag.tags import write_file

from conftest import encode, ffmpeg_required

pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")
from fastapi.testclient import TestClient                     # noqa: E402

from musictag import server                                   # noqa: E402
from musictag.jobs import JobManager                           # noqa: E402
from musictag.config import Config, set_config                # noqa: E402
from musictag.state import AppState                           # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A server wired to throwaway state and config."""
    config = Config()
    config.organize_root = str(tmp_path / "Library")
    config.library_paths = []
    set_config(config)

    state = AppState()
    state._loaded = True
    monkeypatch.setattr(server, "get_state", lambda: state)
    monkeypatch.setattr("musictag.state.get_state", lambda: state)
    # Persisting would write to the shared sqlite store; not what we are testing.
    monkeypatch.setattr(state, "persist", lambda tracks=None: None)
    # A job left running by one test must not block the next one's.
    jobs = JobManager()
    monkeypatch.setattr(server, "get_jobs", lambda: jobs)

    # The real UI reaches the server as 127.0.0.1; TestClient defaults to
    # "testserver", which the local-only guard rightly refuses.
    with TestClient(server.app, base_url="http://127.0.0.1") as test_client:
        test_client.state = state
        test_client.config = config
        yield test_client


def add_track(state, path="C:/Music/A/B/01 - Song.mp3", *, confidence=95.0,
              quality=None) -> Track:
    track = Track(path=path, filename=Path(path).name)
    track.current = TrackTags(title="Old", artist="Old Artist", album="Old Album")
    track.props.container = "mp3"
    track.props.duration_s = 200.0
    track.match = MatchResult(
        confidence=confidence,
        proposed=TrackTags(title="Song", artist="Artist", album="Album",
                           album_artist="Artist", track_no=1),
        candidates=[],
        field_confidence={"title": confidence},
    )
    if quality:
        track.quality = quality
    state.add([track])
    return track


# ===========================================================================

class TestStatus:
    def test_status_reports_capabilities(self, client):
        data = client.get("/api/status").json()
        assert "capabilities" in data
        assert "quality_analysis" in data["capabilities"]
        assert data["stats"]["total"] == 0

    def test_supported_formats_are_listed(self, client):
        data = client.get("/api/status").json()
        assert "mp3" in data["supported_formats"]
        assert "flac" in data["supported_formats"]

    def test_stats_bucket_tracks_by_confidence(self, client):
        add_track(client.state, "C:/M/high.mp3", confidence=97)
        add_track(client.state, "C:/M/review.mp3", confidence=80)
        add_track(client.state, "C:/M/low.mp3", confidence=30)
        buckets = client.get("/api/status").json()["stats"]["buckets"]
        assert buckets["high"] == 1
        assert buckets["review"] == 1
        assert buckets["low"] == 1


class TestUpdate:
    """The endpoint only adapts what musictag.update decides - see test_update.py.

    Every test here patches the check out, so the suite never reaches GitHub:
    a test that depends on a live network is a test that fails on a train.
    """

    def test_reports_what_the_checker_found(self, client, monkeypatch):
        from musictag.update import UpdateInfo
        monkeypatch.setattr(
            server, "check_for_update",
            lambda cfg, force=False: UpdateInfo(current="0.1.0", latest="0.2.0",
                                                available=True, checked=True))
        data = client.get("/api/update").json()
        assert data["available"] is True
        assert data["latest"] == "0.2.0"
        assert data["current"] == "0.1.0"

    def test_force_reaches_the_checker(self, client, monkeypatch):
        from musictag.update import UpdateInfo
        seen = {}

        def fake(cfg, force=False):
            seen["force"] = force
            return UpdateInfo(current="0.1.0")

        monkeypatch.setattr(server, "check_for_update", fake)
        client.get("/api/update")
        assert seen["force"] is False
        client.get("/api/update?force=true")
        assert seen["force"] is True

    def test_a_failed_check_is_still_a_200(self, client, monkeypatch):
        """The UI has to be able to tell "no update" from "the call broke".

        A 500 here would surface as a generic error toast on a page that is
        working perfectly well, over a request the user never asked to make.
        """
        from musictag.update import UpdateInfo
        monkeypatch.setattr(
            server, "check_for_update",
            lambda cfg, force=False: UpdateInfo(current="0.1.0", error="403 rate limited"))
        response = client.get("/api/update")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert "403" in body["error"]


class TestConfig:
    def test_get_config_masks_the_api_key(self, client):
        client.config.acoustid_api_key = "supersecret123"
        data = client.get("/api/config").json()
        assert "supersecret" not in data["acoustid_api_key"]
        assert data["acoustid_api_key"].endswith("123")

    def test_settings_round_trip(self, client):
        response = client.post("/api/config", json={"review_threshold": 55,
                                                    "folder_template": "{artist}/{album}"})
        assert response.status_code == 200
        data = response.json()
        assert data["review_threshold"] == 55
        assert data["folder_template"] == "{artist}/{album}"

    def test_numeric_strings_are_coerced(self, client):
        data = client.post("/api/config", json={"quality_workers": "6"}).json()
        assert data["quality_workers"] == 6

    def test_a_masked_key_does_not_overwrite_the_real_one(self, client):
        client.post("/api/config", json={"acoustid_api_key": "realkey12345"})
        masked = client.get("/api/config").json()["acoustid_api_key"]

        client.post("/api/config", json={"acoustid_api_key": masked})
        from musictag.config import get_config
        assert get_config().acoustid_api_key == "realkey12345"

    def test_unknown_keys_are_ignored(self, client):
        response = client.post("/api/config", json={"not_a_setting": 1})
        assert response.status_code == 200


class TestTracks:
    def test_empty_library_returns_an_empty_page(self, client):
        data = client.get("/api/tracks").json()
        assert data["total"] == 0
        assert data["tracks"] == []

    def test_tracks_include_their_bucket(self, client):
        add_track(client.state, confidence=97)
        track = client.get("/api/tracks").json()["tracks"][0]
        assert track["bucket"] == "high"
        assert track["match"]["proposed"]["title"] == "Song"

    def test_search_filters_on_proposed_and_current_tags(self, client):
        add_track(client.state, "C:/M/one.mp3")
        add_track(client.state, "C:/M/two.mp3")
        assert client.get("/api/tracks?q=Song").json()["total"] == 2
        assert client.get("/api/tracks?q=nonexistent").json()["total"] == 0

    def test_bucket_filter(self, client):
        add_track(client.state, "C:/M/a.mp3", confidence=97)
        add_track(client.state, "C:/M/b.mp3", confidence=30)
        assert client.get("/api/tracks?bucket=low").json()["total"] == 1

    def test_sorting_by_least_confident(self, client):
        add_track(client.state, "C:/M/a.mp3", confidence=97)
        add_track(client.state, "C:/M/b.mp3", confidence=30)
        tracks = client.get("/api/tracks?sort=confidence").json()["tracks"]
        assert tracks[0]["match"]["confidence"] == 30

    def test_pagination(self, client):
        for i in range(5):
            add_track(client.state, f"C:/M/{i}.mp3")
        page = client.get("/api/tracks?limit=2&offset=2").json()
        assert page["total"] == 5
        assert len(page["tracks"]) == 2

    def test_unknown_track_is_404(self, client):
        assert client.get("/api/track?path=C:/nope.mp3").status_code == 404


class TestEditing:
    def test_hand_edit_updates_the_proposal(self, client):
        track = add_track(client.state)
        response = client.post("/api/track/edit",
                               json={"path": track.path, "tags": {"title": "Corrected"}})
        assert response.status_code == 200
        assert response.json()["match"]["proposed"]["title"] == "Corrected"

    def test_hand_edit_marks_the_field_certain(self, client):
        track = add_track(client.state)
        data = client.post("/api/track/edit",
                           json={"path": track.path, "tags": {"title": "Corrected"}}).json()
        assert data["match"]["field_confidence"]["title"] == 100.0

    def test_editing_an_unknown_track_is_404(self, client):
        response = client.post("/api/track/edit",
                               json={"path": "C:/nope.mp3", "tags": {"title": "x"}})
        assert response.status_code == 404

    def test_choosing_a_candidate_out_of_range_is_rejected(self, client):
        track = add_track(client.state)
        response = client.post("/api/track/choose",
                               json={"path": track.path, "candidate_index": 9})
        assert response.status_code == 400


class TestBrowse:
    def test_root_listing_offers_somewhere_to_start(self, client):
        data = client.get("/api/browse").json()
        assert data["entries"]

    def test_listing_a_real_directory(self, client, tmp_path):
        (tmp_path / "Music").mkdir()
        data = client.get(f"/api/browse?path={tmp_path}").json()
        assert any(e["name"] == "Music" for e in data["entries"])
        assert data["parent"]

    @ffmpeg_required
    def test_counts_audio_files_in_a_folder(self, client, tmp_path, clean_wav):
        encode(clean_wav, tmp_path / "a.mp3", "-codec:a", "libmp3lame")
        data = client.get(f"/api/browse?path={tmp_path}").json()
        assert data["audio_files"] == 1

    def test_browsing_a_file_is_404(self, client, tmp_path):
        target = tmp_path / "x.txt"
        target.write_text("hi")
        assert client.get(f"/api/browse?path={target}").status_code == 404


class TestJobs:
    def test_scan_without_a_path_is_rejected(self, client):
        response = client.post("/api/scan", json={"paths": []})
        assert response.status_code == 400

    def test_scan_of_a_missing_path_is_rejected(self, client):
        response = client.post("/api/scan", json={"paths": ["C:/definitely/not/here"]})
        assert response.status_code == 400

    @ffmpeg_required
    def test_scan_runs_and_reports_progress(self, client, tmp_path, clean_wav):
        album = tmp_path / "Album"
        album.mkdir()
        for i in range(3):
            encode(clean_wav, album / f"{i}.mp3", "-codec:a", "libmp3lame")

        job = client.post("/api/scan", json={"paths": [str(album)]}).json()
        for _ in range(100):
            status = client.get(f"/api/jobs/{job['id']}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert status["status"] == "done", status.get("error")
        assert status["result"]["found"] == 3

    def test_identify_with_nothing_scanned_is_rejected(self, client):
        response = client.post("/api/identify", json={"paths": [], "only_pending": True})
        assert response.status_code == 400

    def test_identify_progress_ticks_up_per_track_not_per_folder(self, client, monkeypatch):
        """Regression: all these tracks share one folder (one group), so a job
        that only reported progress once per *group* sat at 0% for the whole
        batch, then jumped straight to done - identify_album's own per-track
        progress callback has to actually be wired up to the job."""
        from musictag import matching

        tracks = [add_track(client.state, f"C:/Music/Flat/{i:02d}.mp3") for i in range(6)]

        def fake_identify_album(self, items, *, progress=None, cancelled=None):
            for i, t in enumerate(items):
                t.status = "identified"
                if progress:
                    progress(i + 1, len(items), t.path)
                time.sleep(0.05)

        monkeypatch.setattr(matching.Matcher, "identify_album", fake_identify_album)

        job = client.post("/api/identify", json={"paths": [t.path for t in tracks]}).json()
        jid = job["id"]

        seen_midway = False
        for _ in range(150):
            status = client.get(f"/api/jobs/{jid}").json()
            if 0 < status["done"] < status["total"]:
                seen_midway = True
                break
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.02)

        assert seen_midway, "progress must tick up mid-batch, not jump straight from 0 to done"

    def test_identify_saves_each_album_as_it_finishes(self, client, monkeypatch):
        """A crash an hour in should cost one album, not the whole run."""
        from musictag import matching

        for folder in ("A", "B", "C"):
            for i in range(2):
                add_track(client.state, f"C:/Music/{folder}/{i:02d}.mp3")
        saved: list[list[str]] = []
        monkeypatch.setattr(client.state, "persist",
                            lambda tracks=None: saved.append([t.path for t in tracks]))
        monkeypatch.setattr(matching.Matcher, "identify_album",
                            lambda self, items, progress=None, cancelled=None: None)

        job = client.post("/api/identify", json={"paths": [], "only_pending": False}).json()
        assert wait_for_job(client, job["id"])["status"] == "done"
        assert sorted(len(batch) for batch in saved) == [2, 2, 2]

    def test_quality_counts_every_file_with_parallel_workers(self, client, monkeypatch):
        from musictag.models import QualityReport

        for i in range(40):
            add_track(client.state, f"C:/Music/Q/{i:02d}.mp3")
        client.config.ffmpeg_path = "ffmpeg"
        client.config.quality_workers = 8
        monkeypatch.setattr(type(client.config), "ffmpeg", property(lambda self: "ffmpeg"))
        monkeypatch.setattr(server, "analyze_track",
                            lambda track, cfg: QualityReport(analysed=True))

        job = client.post("/api/quality", json={"paths": [], "only_pending": False}).json()
        status = wait_for_job(client, job["id"])
        assert status["status"] == "done", status.get("error")
        assert status["result"]["analysed"] == 40
        assert status["done"] == 40

    def test_apply_with_nothing_identified_is_rejected(self, client):
        response = client.post("/api/apply", json={"paths": []})
        assert response.status_code == 400

    def test_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/deadbeef").status_code == 404

    def test_organize_is_refused_when_disabled(self, client):
        add_track(client.state)
        client.state.all()[0].match.candidates = [object()]   # looks identified
        response = client.post("/api/apply", json={"paths": [], "organize": True})
        assert response.status_code == 400
        assert "settings" in response.json()["error"].lower()


class TestStaticUI:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "MusicTagger" in response.text

    def test_stylesheet_is_served(self, client):
        assert client.get("/static/styles.css").status_code == 200

    def test_script_is_served(self, client):
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "refreshTracks" in response.text


class TestQualityScale:
    """The UI explains badges from this payload, so it has to be present."""

    def test_status_exposes_the_scale(self, client):
        scale = client.get("/api/status").json()["quality_scale"]
        assert scale["max_score"] == 100
        keys = [s["key"] for s in scale["severities"]]
        assert keys == ["high", "medium", "low", "info"], "worst first, as displayed"

    def test_every_severity_has_a_label_and_an_explanation(self, client):
        scale = client.get("/api/status").json()["quality_scale"]
        for severity in scale["severities"]:
            assert severity["label"]
            assert severity["meaning"]
            assert isinstance(severity["penalty"], int)

    def test_notes_cost_nothing(self, client):
        scale = client.get("/api/status").json()["quality_scale"]
        info = next(s for s in scale["severities"] if s["key"] == "info")
        assert info["penalty"] == 0

    def test_penalties_decrease_with_severity(self, client):
        scale = client.get("/api/status").json()["quality_scale"]
        penalties = [s["penalty"] for s in scale["severities"]]
        assert penalties == sorted(penalties, reverse=True)


class TestGridSorting:
    """The columns view sorts on any tag field, server-side."""

    def _library(self, state):
        specs = [
            ("C:/M/c.mp3", "Zulu", "Beta Band", 3),
            ("C:/M/a.mp3", "Alpha", "Alpha Band", 1),
            ("C:/M/b.mp3", "Mike", "Ceta Band", 2),
        ]
        for path, title, artist, track_no in specs:
            t = add_track(state, path)
            t.match.proposed = TrackTags(title=title, artist=artist,
                                         album="X", album_artist=artist,
                                         track_no=track_no)
        return state

    def test_sort_by_title(self, client):
        self._library(client.state)
        titles = [t["match"]["proposed"]["title"]
                  for t in client.get("/api/tracks?sort=title").json()["tracks"]]
        assert titles == ["Alpha", "Mike", "Zulu"]

    def test_sort_descending(self, client):
        self._library(client.state)
        titles = [t["match"]["proposed"]["title"]
                  for t in client.get("/api/tracks?sort=title&desc=true").json()["tracks"]]
        assert titles == ["Zulu", "Mike", "Alpha"]

    def test_sort_by_artist(self, client):
        self._library(client.state)
        artists = [t["match"]["proposed"]["artist"]
                   for t in client.get("/api/tracks?sort=artist").json()["tracks"]]
        assert artists == ["Alpha Band", "Beta Band", "Ceta Band"]

    def test_sort_by_track_number_is_numeric_not_alphabetical(self, client):
        state = client.state
        for path, n in [("C:/M/x.mp3", 10), ("C:/M/y.mp3", 2), ("C:/M/z.mp3", 1)]:
            t = add_track(state, path)
            t.match.proposed = TrackTags(title=f"T{n}", track_no=n)
        numbers = [t["match"]["proposed"]["track_no"]
                   for t in client.get("/api/tracks?sort=track_no").json()["tracks"]]
        assert numbers == [1, 2, 10], "10 must not sort before 2"

    def test_sorts_on_the_value_that_would_be_written(self, client):
        """A cell shows the proposed value, so it must sort on that too."""
        state = client.state
        t = add_track(state, "C:/M/one.mp3")
        t.current = TrackTags(title="Zzz Old Title")
        t.match.proposed = TrackTags(title="Aaa New Title")
        other = add_track(state, "C:/M/two.mp3")
        other.current = TrackTags(title="Mmm")
        other.match.proposed = TrackTags(title="Mmm")

        first = client.get("/api/tracks?sort=title").json()["tracks"][0]
        assert first["path"] == "C:/M/one.mp3"

    def test_rows_with_an_empty_cell_sink_to_the_bottom(self, client):
        state = client.state
        blank = add_track(state, "C:/M/blank.mp3")
        blank.current = TrackTags()
        blank.match.proposed = TrackTags(title="Has title", genre=None)
        tagged = add_track(state, "C:/M/tagged.mp3")
        tagged.match.proposed = TrackTags(title="Also", genre="Rock")

        for desc in ("false", "true"):
            paths = [t["path"] for t in
                     client.get(f"/api/tracks?sort=genre&desc={desc}").json()["tracks"]]
            assert paths[-1] == "C:/M/blank.mp3", \
                "an empty cell is a gap, not a small value - it belongs last either way"

    def test_legacy_confidence_desc_still_works(self, client):
        add_track(client.state, "C:/M/a.mp3", confidence=30)
        add_track(client.state, "C:/M/b.mp3", confidence=97)
        tracks = client.get("/api/tracks?sort=confidence_desc").json()["tracks"]
        assert tracks[0]["match"]["confidence"] == 97

    def test_unknown_sort_falls_back_instead_of_erroring(self, client):
        add_track(client.state)
        assert client.get("/api/tracks?sort=not_a_field").status_code == 200

    def test_columns_endpoint_lists_sortable_fields(self, client):
        data = client.get("/api/columns").json()
        assert "album_artist" in data["tag_fields"]
        assert "quality" in data["file_fields"]


class TestManualEditThenApply:
    """A file typed in by hand, with no Identify ever run, must still apply."""

    def test_applyable_accepts_a_manual_only_match(self, client):
        track = add_track(client.state, "C:/M/manual.mp3")
        track.match = None
        client.post("/api/track/edit",
                   json={"path": track.path, "tags": {"title": "Hand Typed"}})
        assert server._applyable(client.state.get(track.path)) is True

    def test_applyable_rejects_a_track_with_no_proposal(self, client):
        track = Track(path="C:/M/none.mp3", filename="none.mp3")
        client.state.add([track])
        assert server._applyable(client.state.get(track.path)) is False

    def test_preview_organize_includes_a_hand_edited_file(self, client):
        track = add_track(client.state, "C:/M/manual2.mp3")
        track.match = None
        client.post("/api/track/edit",
                   json={"path": track.path,
                         "tags": {"title": "T", "artist": "A", "album": "Al"}})
        result = client.post("/api/preview-organize", json={"paths": [track.path]}).json()
        assert result["count"] + result["unchanged"] >= 1

    @ffmpeg_required
    def test_apply_route_writes_a_hand_edited_never_identified_file(self, client, tmp_path, clean_wav):
        """The actual regression: this request used to 400 with "No identified
        tracks selected", even though the user had just typed real tags in.
        """
        target = tmp_path / "manual.mp3"
        encode(clean_wav, target, "-codec:a", "libmp3lame", "-b:a", "192k")

        from musictag.tags import read_file

        track = Track(path=str(target), filename=target.name)
        track.current, track.props = read_file(target)
        client.state.add([track])

        edit = client.post("/api/track/edit",
                           json={"path": str(target),
                                 "tags": {"title": "Hand Typed", "artist": "Someone"}})
        assert edit.status_code == 200

        response = client.post("/api/apply", json={"paths": [str(target)], "dry_run": False})
        assert response.status_code == 200, response.json()

        job_id = response.json()["id"]
        for _ in range(100):
            status = client.get(f"/api/jobs/{job_id}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert status["status"] == "done", status.get("error")
        assert status["result"]["tagged"] == 1

        from musictag.tags import read_file
        written, _ = read_file(target)
        assert written.title == "Hand Typed"
        assert written.artist == "Someone"


class TestLookupCache:
    def test_info_reports_a_count(self, client):
        data = client.get("/api/lookup-cache").json()
        assert data["entries"] == 0

    def test_clear_reports_how_much_it_removed(self, client):
        from musictag.cache import http_cache
        http_cache().set("k1", {"x": 1})
        http_cache().set("k2", None)
        assert client.get("/api/lookup-cache").json()["entries"] == 2

        result = client.post("/api/lookup-cache/clear")
        assert result.json()["cleared"] == 2
        assert client.get("/api/lookup-cache").json()["entries"] == 0


class TestLibraryPathMemory:
    """MusicTagger tracks one current library folder, not a growing history
    of every folder it has ever been pointed at - a real complaint from
    actually using the app: switching folders kept accumulating paths."""

    def test_an_explicit_selection_replaces_the_remembered_folder(self, client, tmp_path):
        old_folder = tmp_path / "Old"
        new_folder = tmp_path / "New"
        old_folder.mkdir()
        new_folder.mkdir()
        client.config.library_paths = [str(old_folder)]

        client.post("/api/scan", json={"paths": [str(new_folder)], "recursive": True})
        assert client.config.library_paths == [str(new_folder)], \
            "the old folder must not still be remembered alongside the new one"

    def test_a_plain_rescan_does_not_touch_the_remembered_folder(self, client, tmp_path):
        folder = tmp_path / "Music"
        folder.mkdir()
        client.config.library_paths = [str(folder)]

        client.post("/api/scan", json={"paths": [], "recursive": True})
        assert client.config.library_paths == [str(folder)]

    def test_scanning_the_same_folder_twice_does_not_duplicate_it(self, client, tmp_path):
        folder = tmp_path / "Music"
        folder.mkdir()

        client.post("/api/scan", json={"paths": [str(folder)], "recursive": True})
        client.post("/api/scan", json={"paths": [str(folder)], "recursive": True})
        assert client.config.library_paths == [str(folder)]

    def test_multiple_explicit_paths_are_deduplicated_in_order(self, client, tmp_path):
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir()
        b.mkdir()
        client.post("/api/scan", json={"paths": [str(a), str(b), str(a)], "recursive": True})
        assert client.config.library_paths == [str(a), str(b)]

    def test_replace_flag_still_clears_previously_scanned_tracks(self, client, tmp_path):
        add_track(client.state, "C:/Old/song.mp3")
        assert client.state.stats()["total"] == 1

        folder = tmp_path / "New"
        folder.mkdir()
        client.post("/api/scan", json={"paths": [str(folder)], "recursive": True, "replace": True})
        assert client.state.stats()["total"] == 0

    def test_without_the_replace_flag_old_tracks_survive_a_new_scan(self, client, tmp_path):
        """The flag is opt-in - the server itself never guesses that a new
        folder means "forget everything else"; that decision belongs to the
        caller (the UI asks the user first)."""
        add_track(client.state, "C:/Old/song.mp3")
        folder = tmp_path / "New"
        folder.mkdir()
        client.post("/api/scan", json={"paths": [str(folder)], "recursive": True, "replace": False})
        assert client.state.get("C:/Old/song.mp3") is not None


def _write_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(4410, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(samples.tobytes())
    return path


def make_ready_file(path, **tag_kwargs) -> Track:
    """A real file, already tagged and with no pending (unapplied) proposal."""
    _write_wav(path)
    tags = TrackTags(**tag_kwargs)
    write_file(path, tags)
    return Track(path=str(path), filename=path.name, current=tags)


def wait_for_job(client, job_id: str) -> dict:
    status = {}
    for _ in range(200):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    return status


class TestExportToPlex:
    def test_plan_requires_a_configured_plex_root(self, client, tmp_path):
        client.config.organize_root = ""
        track = make_ready_file(tmp_path / "in" / "a.wav", title="Song", artist="Artist")
        client.state.add([track])

        response = client.post("/api/export/plan", json={"paths": [track.path]})
        assert response.status_code == 400

    def test_plan_requires_a_selection(self, client):
        response = client.post("/api/export/plan", json={"paths": ["C:/nope.mp3"]})
        assert response.status_code == 400

    def test_plan_reports_a_ready_track_with_no_duplicate(self, client, tmp_path):
        track = make_ready_file(tmp_path / "in" / "a.wav", title="Glory Box",
                                artist="Portishead", album="Dummy")
        client.state.add([track])

        job = client.post("/api/export/plan", json={"paths": [track.path]}).json()
        status = wait_for_job(client, job["id"])
        assert status["status"] == "done", status.get("error")
        result = status["result"]
        assert len(result["items"]) == 1
        assert result["items"][0]["duplicate"] is None
        assert result["pending_apply"] == []

    def test_plan_holds_back_a_track_with_unapplied_changes(self, client):
        track = add_track(client.state, "C:/Music/A/B/01 - Song.mp3")  # current != proposed
        job = client.post("/api/export/plan", json={"paths": [track.path]}).json()
        status = wait_for_job(client, job["id"])
        assert status["result"]["items"] == []
        assert status["result"]["pending_apply"] == [track.path]

    def test_plan_flags_a_duplicate_already_in_the_plex_folder(self, client, tmp_path):
        existing = Path(client.config.organize_root) / "Portishead" / "Dummy.wav"
        _write_wav(existing)
        write_file(existing, TrackTags(title="Glory Box", artist="Portishead", album="Dummy"))

        track = make_ready_file(tmp_path / "in" / "a.wav", title="Glory Box",
                                artist="Portishead", album="Dummy")
        client.state.add([track])

        job = client.post("/api/export/plan", json={"paths": [track.path]}).json()
        status = wait_for_job(client, job["id"])
        result = status["result"]
        assert len(result["items"]) == 1
        assert result["items"][0]["duplicate"] is not None
        assert result["duplicate_count"] == 1

    def test_commit_requires_items(self, client):
        response = client.post("/api/export/commit", json={"items": []})
        assert response.status_code == 400

    def test_commit_moves_the_file_and_forgets_the_track(self, client, tmp_path):
        track = make_ready_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        client.state.add([track])
        dest = str(Path(client.config.organize_root) / "Portishead" / "01 - Glory Box.wav")

        job = client.post("/api/export/commit", json={
            "items": [{"path": track.path, "dest": dest, "action": "export"}],
        }).json()
        status = wait_for_job(client, job["id"])
        assert status["status"] == "done", status.get("error")
        assert status["result"]["exported"] == 1
        assert Path(dest).exists()
        assert not Path(track.path).exists()
        # Exported tracks leave the ingest library's tracked state entirely.
        assert client.state.get(track.path) is None

    def test_commit_skip_keeps_the_track_tracked(self, client, tmp_path):
        track = make_ready_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        client.state.add([track])
        dest = str(Path(client.config.organize_root) / "01 - Glory Box.wav")

        job = client.post("/api/export/commit", json={
            "items": [{"path": track.path, "dest": dest, "action": "skip"}],
        }).json()
        status = wait_for_job(client, job["id"])
        assert status["result"]["skipped"] == 1
        assert Path(track.path).exists()
        assert client.state.get(track.path) is not None

    def test_failed_commit_keeps_the_track_tracked(self, client, tmp_path):
        """A file that did not move is still in the ingest folder - keep showing it."""
        track = make_ready_file(tmp_path / "in" / "a.wav", title="Glory Box", artist="Portishead")
        client.state.add([track])
        blocker = Path(client.config.organize_root) / "Portishead"
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text("a file where the artist folder should go")
        dest = str(blocker / "01 - Glory Box.wav")

        job = client.post("/api/export/commit", json={
            "items": [{"path": track.path, "dest": dest, "action": "export"}],
        }).json()
        status = wait_for_job(client, job["id"])
        assert status["result"]["failed"] == 1
        assert Path(track.path).exists()
        assert client.state.get(track.path) is not None


class TestLocalOnlyGuard:
    """Another website must not be able to drive this server (DNS rebinding, CSRF)."""

    def test_foreign_host_is_refused(self, client):
        response = client.get("/api/status", headers={"host": "attacker.example:8731"})
        assert response.status_code == 403

    def test_foreign_host_cannot_change_settings(self, client):
        response = client.post("/api/config", json={"ffmpeg_path": "/tmp/evil"},
                               headers={"host": "attacker.example:8731"})
        assert response.status_code == 403
        assert client.config.ffmpeg_path != "/tmp/evil"

    @pytest.mark.parametrize("host", ["127.0.0.1:8731", "localhost:8731", "127.0.0.1"])
    def test_local_hosts_are_allowed(self, client, host):
        assert client.get("/api/status", headers={"host": host}).status_code == 200

    @pytest.mark.parametrize("origin", ["https://attacker.example", "null"])
    def test_cross_site_post_is_refused(self, client, origin):
        response = client.post("/api/clear", headers={"origin": origin})
        assert response.status_code == 403

    def test_same_origin_post_is_allowed(self, client):
        response = client.post("/api/clear", headers={"origin": "http://127.0.0.1:8731"})
        assert response.status_code == 200


class TestJobConflicts:
    """Jobs that would trample each other's tracks or files are refused, not raced."""

    @pytest.fixture
    def held(self, client):
        """Start a job of a given kind that stays running until the test ends."""
        gate = threading.Event()
        jobs = server.get_jobs()

        def start(kind: str):
            job = jobs.submit(kind, lambda job: gate.wait(10))
            for _ in range(100):
                if job.status == "running":
                    break
                time.sleep(0.01)
            return job

        yield start
        gate.set()

    def test_apply_is_refused_while_tagging(self, client, held):
        add_track(client.state)
        held("identify")
        response = client.post("/api/apply", json={"paths": []})
        assert response.status_code == 409
        assert "tagging" in response.json()["error"]
        assert response.json()["running"]["kind"] == "identify"

    def test_tagging_and_quality_may_run_together(self, client, held):
        held("quality")
        job = server.get_jobs().submit("identify", lambda job: None)
        assert job.kind == "identify"

    def test_same_kind_cannot_run_twice(self, client, held):
        add_track(client.state)
        held("identify")
        response = client.post("/api/identify", json={"paths": [], "only_pending": False})
        assert response.status_code == 409

    def test_refused_replace_scan_does_not_wipe_the_library(self, client, held, tmp_path):
        add_track(client.state)
        held("apply")
        response = client.post("/api/scan", json={"paths": [str(tmp_path)], "replace": True})
        assert response.status_code == 409
        assert len(client.state.all()) == 1

    def test_hand_edit_is_refused_while_tagging(self, client, held):
        track = add_track(client.state)
        held("identify")
        response = client.post("/api/track/edit",
                               json={"path": track.path, "tags": {"title": "Mine"}})
        assert response.status_code == 409
        assert track.match.proposed.title == "Song"

    def test_hand_edit_is_allowed_during_a_quality_check(self, client, held):
        track = add_track(client.state)
        held("quality")
        response = client.post("/api/track/edit",
                               json={"path": track.path, "tags": {"title": "Mine"}})
        assert response.status_code == 200

    def test_finished_job_no_longer_blocks(self, client):
        add_track(client.state)
        job = server.get_jobs().submit("identify", lambda job: None)
        for _ in range(100):
            if job.status == "done":
                break
            time.sleep(0.01)
        response = client.post("/api/apply", json={"paths": [], "dry_run": True})
        assert response.status_code == 200
