"""An undo journal for everything that touches the filesystem.

Automatic tagging edits files in place, so every write records what was there
before. Nothing here deletes user data: undoing a *copy* reports the copy it
made rather than removing it, because guessing which file you meant to keep is
exactly the sort of decision a program should not make on your behalf.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import JOURNAL_DB
from .models import TrackTags
from .tags import write_file
from .util import unique_path

#: Key inside a "tags" entry's ``prev_tags`` JSON listing the fields Apply set.
WRITTEN_KEY = "_written"


@dataclass
class UndoResult:
    restored: int = 0
    skipped: int = 0
    failed: int = 0
    messages: list[str] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


class Journal:
    def __init__(self, path: Path | None = None):
        self.path = path or JOURNAL_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS batches ("
            " id TEXT PRIMARY KEY, started REAL, finished REAL,"
            " description TEXT, counts TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " batch_id TEXT NOT NULL,"
            " ts REAL NOT NULL,"
            " op TEXT NOT NULL,"          # tags | move | copy | cover
            " src TEXT NOT NULL,"
            " dest TEXT,"
            " prev_tags TEXT,"
            " ok INTEGER DEFAULT 1,"
            " error TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_batch ON entries(batch_id)")
        # Added after the first release; older journals get the column here.
        try:
            conn.execute("ALTER TABLE batches ADD COLUMN undone REAL")
        except sqlite3.OperationalError:
            pass
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    def start_batch(self, description: str) -> str:
        batch_id = uuid.uuid4().hex[:12]
        conn = self._conn()
        conn.execute(
            "INSERT INTO batches (id, started, description, counts) VALUES (?, ?, ?, ?)",
            (batch_id, time.time(), description, "{}"),
        )
        conn.commit()
        return batch_id

    def finish_batch(self, batch_id: str, counts: dict[str, Any]) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE batches SET finished = ?, counts = ? WHERE id = ?",
            (time.time(), json.dumps(counts), batch_id),
        )
        conn.commit()

    def record(self, batch_id: str, op: str, src: str, *, dest: Optional[str] = None,
               prev_tags: Optional[TrackTags] = None, ok: bool = True,
               error: Optional[str] = None,
               written: Optional[list[str]] = None) -> int:
        """Log one step and return its entry id.

        ``written`` lists the tag fields a "tags" op sets.

        It rides inside the ``prev_tags`` JSON (``TrackTags.from_dict``
        ignores unknown keys) so journals from older versions stay readable
        without a schema change.
        """
        conn = self._conn()
        snapshot = None
        if prev_tags:
            data = prev_tags.to_dict()
            if written is not None:
                data[WRITTEN_KEY] = list(written)
            snapshot = json.dumps(data)
        cur = conn.execute(
            "INSERT INTO entries (batch_id, ts, op, src, dest, prev_tags, ok, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (batch_id, time.time(), op, src, dest,
             snapshot,
             1 if ok else 0, error),
        )
        conn.commit()
        return cur.lastrowid

    def fail(self, entry_id: int, error: str) -> None:
        """Mark a step recorded up front as not having happened after all.

        Filesystem steps are journalled *before* they run, so a crash midway
        still leaves undo a record of a file that may have moved. When the
        step raises instead, this takes it back out of what undo replays.
        """
        conn = self._conn()
        conn.execute("UPDATE entries SET ok = 0, error = ? WHERE id = ?", (error, entry_id))
        conn.commit()

    def step(self, batch_id: str, op: str, src: str, *, dest: Optional[str] = None):
        """Context manager: journal a file operation, then run it.

        ``with journal.step(batch, "move", src, dest=dest): shutil.move(...)``
        """
        return _JournalledStep(self, batch_id, op, src, dest)

    # ------------------------------------------------------------------
    def list_batches(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT b.*, (SELECT COUNT(*) FROM entries e WHERE e.batch_id = b.id) AS entry_count"
            " FROM batches b ORDER BY b.started DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["counts"] = json.loads(item.get("counts") or "{}")
            except json.JSONDecodeError:
                item["counts"] = {}
            out.append(item)
        return out

    def batch_entries(self, batch_id: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM entries WHERE batch_id = ? ORDER BY id", (batch_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    def is_undone(self, batch_id: str) -> bool:
        row = self._conn().execute(
            "SELECT undone FROM batches WHERE id = ?", (batch_id,)).fetchone()
        return bool(row and row["undone"])

    def undo(self, batch_id: str, *, id3v2_version: int = 4) -> UndoResult:
        """Reverse a batch: files move back first, then tags are restored.

        A batch is undone once. Replaying it a second time would move files
        that have since been put back - and after an export that replaced a
        duplicate, that means moving the old Plex copy over the incoming file.
        """
        result = UndoResult()
        if self.is_undone(batch_id):
            result.skipped += 1
            result.messages.append("This change has already been undone.")
            return result
        entries = [e for e in self.batch_entries(batch_id) if e["ok"]]
        #: Files restored under a different name than they left from, so a
        #: tag restore later in the replay lands on the file, not on whatever
        #: now occupies its old path.
        redirected: dict[str, Path] = {}

        # Reverse order so a move followed by a tag write unwinds cleanly.
        for entry in reversed(entries):
            op, src, dest = entry["op"], entry["src"], entry["dest"]
            try:
                if op == "move" and dest:
                    dest_path, src_path = Path(dest), Path(src)
                    if not dest_path.exists():
                        result.skipped += 1
                        result.messages.append(f"Already gone, not restored: {dest}")
                        continue
                    src_path.parent.mkdir(parents=True, exist_ok=True)
                    # Something new may have been put at the original path
                    # since. Moving onto it would overwrite it, so the file
                    # comes back under a free name next to it instead.
                    target = unique_path(src_path)
                    if target != src_path:
                        result.messages.append(
                            f"{src_path} is taken now, restored as {target.name}")
                    shutil.move(str(dest_path), str(target))
                    if target != src_path:
                        redirected[src] = target
                    result.restored += 1
                elif op == "copy" and dest:
                    # Deliberately not deleted - see module docstring.
                    result.skipped += 1
                    result.messages.append(f"Copy left in place (delete manually if unwanted): {dest}")
                elif op == "tags" and entry["prev_tags"]:
                    target = redirected.get(src, Path(src))
                    if not target.exists():
                        # It may have been moved in the same batch and just restored.
                        result.skipped += 1
                        continue
                    data = json.loads(entry["prev_tags"])
                    prev = TrackTags.from_dict(data)
                    # Fields Apply wrote that were blank before get removed,
                    # not left behind. Entries from before this was recorded
                    # carry no list, so they keep the old overwrite-only undo.
                    write_file(target, prev, id3v2_version=id3v2_version,
                               clear=frozenset(data.get(WRITTEN_KEY) or ()))
                    result.restored += 1
                elif op == "cover" and dest:
                    result.skipped += 1
                    result.messages.append(f"Cover file left in place: {dest}")
            except Exception as exc:  # noqa: BLE001
                result.failed += 1
                result.messages.append(f"{op} undo failed for {src}: {exc}")

        conn = self._conn()
        conn.execute("UPDATE batches SET undone = ? WHERE id = ?", (time.time(), batch_id))
        conn.commit()
        return result

    def prune(self, keep: int = 100) -> int:
        """Drop the oldest batches so the journal does not grow without bound."""
        conn = self._conn()
        old = conn.execute(
            "SELECT id FROM batches ORDER BY started DESC LIMIT -1 OFFSET ?", (keep,)
        ).fetchall()
        ids = [r["id"] for r in old]
        for batch_id in ids:
            conn.execute("DELETE FROM entries WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
        conn.commit()
        return len(ids)


class _JournalledStep:
    def __init__(self, journal: Journal, batch_id: str, op: str, src: str,
                 dest: Optional[str]):
        self.journal, self.args = journal, (batch_id, op, src)
        self.dest = dest
        self.entry_id: Optional[int] = None

    def __enter__(self) -> "_JournalledStep":
        self.entry_id = self.journal.record(*self.args, dest=self.dest)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and self.entry_id is not None:
            self.journal.fail(self.entry_id, str(exc))
        return False


_journal: Journal | None = None


def get_journal() -> Journal:
    global _journal
    if _journal is None:
        _journal = Journal()
    return _journal
