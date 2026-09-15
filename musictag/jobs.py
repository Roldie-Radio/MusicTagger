"""Background job runner.

Scanning, identifying and analysing all take minutes on a real library, so the
UI never calls them directly. It starts a job, then polls for progress. Jobs are
cancellable, and one failing job never takes down the server.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"          # pending | running | done | error | cancelled
    total: int = 0
    done: int = 0
    message: str = ""
    detail: str = ""
    result: Any = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # -- called from the worker ----------------------------------------
    def progress(self, done: int, total: int, detail: str = "") -> None:
        self.done, self.total = done, total
        if detail:
            self.detail = detail

    def log(self, message: str) -> None:
        self.message = message
        log.info("[%s] %s", self.kind, message)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def to_dict(self) -> dict[str, Any]:
        pct = round(self.done / self.total * 100, 1) if self.total else (
            100.0 if self.status == "done" else 0.0)
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "percent": pct,
            "message": self.message,
            "detail": self.detail,
            "result": self.result,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }


class JobManager:
    def __init__(self, max_concurrent: int = 2):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent,
                                        thread_name_prefix="musictag-job")

    def submit(self, kind: str, fn: Callable[[Job], Any], *, message: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, message=message)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # Keep the list from growing forever in a long-running session.
            while len(self._order) > 40:
                stale = self._order.pop(0)
                stale_job = self._jobs.get(stale)
                if stale_job and stale_job.status in ("done", "error", "cancelled"):
                    self._jobs.pop(stale, None)
                else:
                    self._order.append(stale)
                    break

        def runner():
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "cancelled" if job.cancelled else "done"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                log.error("Job %s (%s) failed:\n%s", job.id, kind, traceback.format_exc())
            finally:
                job.finished = time.time()

        self._pool.submit(runner)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._jobs[j].to_dict() for j in reversed(self._order) if j in self._jobs]

    def active(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.status in ("pending", "running")]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in ("pending", "running"):
            job.cancel()
            return True
        return False

    def shutdown(self) -> None:
        for job in self.active():
            job.cancel()
        self._pool.shutdown(wait=False)


_manager: JobManager | None = None


def get_jobs() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
