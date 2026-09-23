"""Which background jobs may run at the same time."""

from __future__ import annotations

import threading

import pytest

from musictag.jobs import JobConflict, JobManager, jobs_conflict


@pytest.mark.parametrize("a, b, clash", [
    ("apply", "identify", True),
    ("identify", "apply", True),
    ("scan", "quality", True),
    ("undo", "export-commit", True),
    ("apply", "apply", True),
    ("identify", "identify", True),
    ("identify", "quality", False),
    ("quality", "identify", False),
    ("install-fpcalc", "apply", False),
    ("apply", "install-fpcalc", False),
])
def test_conflict_rules(a, b, clash):
    assert jobs_conflict(a, b) is clash


def test_submit_refuses_a_clashing_job_and_allows_it_once_finished():
    jobs = JobManager()
    gate = threading.Event()
    first = jobs.submit("apply", lambda job: gate.wait(10))

    with pytest.raises(JobConflict) as info:
        jobs.submit("undo", lambda job: None)
    assert info.value.running is first

    gate.set()
    for _ in range(200):
        if first.status == "done":
            break
        threading.Event().wait(0.01)
    assert jobs.submit("undo", lambda job: None).kind == "undo"
    jobs.shutdown()
