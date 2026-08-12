"""Independent, directory-fd anchored session containers for local memory.

Deletion is limited to files addressed relative to the manager's opened
``sessions`` directory. It does not promise erasure from SSDs, backups,
snapshots, or unknown processes that retain already-open file descriptors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
from typing import Dict, Optional, Tuple

from .store import SQLiteEventStore


@dataclass(frozen=True)
class PurgeReport:
    session_id: str
    container_deleted: bool
    deleted_paths: Tuple[str, ...]
    absent_paths: Tuple[str, ...]
    checkpoint_attempted: bool
    checkpoint_completed: bool
    failure_reason: str = ""
    limitations: Tuple[str, ...] = (
        "Does not prove erasure from SSD wear-leveling, backups, snapshots, or external copies.",
        "Cannot revoke unknown external file descriptors or snapshots; their owners must close them.",
        "Checkpoint status is a best-effort WAL-busy check, not proof of all process handles.",
    )

    def to_dict(self) -> dict:
        return asdict(self)


class SessionMemoryManager:
    """Own one SQLite container per session under a fixed directory FD."""

    def __init__(self, memory_root: str) -> None:
        if not memory_root:
            raise ValueError("memory_root must be explicit")
        requested_root = Path(memory_root).expanduser()
        if requested_root.is_symlink():
            raise ValueError("memory_root must not be a symlink")
        requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise ValueError("memory_root must be a directory, not a symlink")
        self.root = requested_root.resolve()
        os.chmod(str(self.root), 0o700)
        self.sessions_root = self.root / "sessions"
        if self.sessions_root.is_symlink():
            raise ValueError("sessions directory must not be a symlink")
        self.sessions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.sessions_root.is_symlink() or self.sessions_root.resolve().parent != self.root:
            raise ValueError("sessions directory escapes memory root")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        self._sessions_fd = os.open(str(self.sessions_root), flags)
        info = os.fstat(self._sessions_fd)
        if not stat.S_ISDIR(info.st_mode):
            os.close(self._sessions_fd)
            raise ValueError("sessions path is not a directory")
        self._sessions_identity = (info.st_dev, info.st_ino)
        os.fchmod(self._sessions_fd, 0o700)
        self._stores: Dict[str, SQLiteEventStore] = {}

    @staticmethod
    def session_key(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _assert_sessions_fd(self) -> None:
        info = os.fstat(self._sessions_fd)
        if (info.st_dev, info.st_ino) != self._sessions_identity or not stat.S_ISDIR(info.st_mode):
            raise OSError("fixed sessions directory is no longer valid")

    def _names(self, session_id: str) -> Tuple[str, str, str]:
        name = self.session_key(session_id) + ".sqlite"
        return name, name + "-wal", name + "-shm"

    def _assert_path_identity(self) -> None:
        self._assert_sessions_fd()
        try:
            info = os.stat(str(self.sessions_root), follow_symlinks=False)
        except OSError as error:
            raise OSError("sessions path is unavailable") from error
        if stat.S_ISLNK(info.st_mode) or (info.st_dev, info.st_ino) != self._sessions_identity:
            raise OSError("sessions path was replaced")

    def database_path(self, session_id: str) -> Path:
        """Public display path; cleanup is anchored to the fixed directory FD."""
        return self.sessions_root / self._names(session_id)[0]

    def _paths(self, session_id: str) -> Tuple[Path, Path, Path]:
        return tuple(self.sessions_root / name for name in self._names(session_id))  # type: ignore[return-value]

    def _report_labels(self, names: Tuple[str, str, str]) -> Tuple[str, str, str]:
        """Stable labels that cannot become attacker-controlled path displays."""
        anchor = "<sessions-fd:%d:%d>" % self._sessions_identity
        return tuple(anchor + "/" + name for name in names)  # type: ignore[return-value]

    def _entry(self, name: str) -> Optional[os.stat_result]:
        try:
            return os.stat(name, dir_fd=self._sessions_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def _preflight_entries(self, names: Tuple[str, str, str]) -> Optional[str]:
        try:
            self._assert_sessions_fd()
            self._assert_path_identity()
            for name in names:
                entry = self._entry(name)
                if entry is not None and stat.S_ISLNK(entry.st_mode):
                    return "session database path is a symlink"
            return None
        except OSError as error:
            return "session directory preflight failed: %s" % error

    def open_session(self, session_id: str) -> SQLiteEventStore:
        """Open after pre/post inode checks; SQLite lacks an openat-style API on macOS.

        Cleanup remains FD-relative.  Opening uses the normal SQLite path only
        after checking that it resolves to the fixed directory inode, and fails
        closed if it changed by return. This is a portable conservative guard,
        not a claim to defeat a hostile same-user path race inside SQLite.
        """
        existing = self._stores.get(session_id)
        if existing is not None:
            try:
                self._assert_path_identity()
            except OSError:
                self._stores.pop(session_id, None)
                self._safe_close(existing)
                raise
            return existing
        names = self._names(session_id)
        problem = self._preflight_entries(names)
        if problem:
            raise OSError(problem)
        store = None
        try:
            self._assert_path_identity()
            store = SQLiteEventStore(str(self.database_path(session_id)), session_id=session_id)
            self._assert_path_identity()
            entry = self._entry(names[0])
            if entry is None or stat.S_ISLNK(entry.st_mode):
                raise OSError("session database changed during open")
            os.chmod(names[0], 0o600, dir_fd=self._sessions_fd, follow_symlinks=False)
            self._stores[session_id] = store
            return store
        except Exception:
            if store is not None:
                self._safe_close(store)
            raise

    def close_session(self, session_id: str) -> None:
        store = self._stores.pop(session_id, None)
        if store is not None:
            self._safe_close(store)

    @staticmethod
    def _safe_close(store: SQLiteEventStore) -> None:
        try:
            store.close()
        except sqlite3.ProgrammingError:
            pass

    def _checkpoint(self, store: SQLiteEventStore) -> Tuple[bool, str]:
        try:
            result = store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                return False, "WAL checkpoint is busy; retry after external readers close."
            return True, ""
        except sqlite3.Error as error:
            return False, "WAL checkpoint failed: %s" % error

    def _temporary_checkpoint(self, session_id: str, name: str) -> Tuple[bool, str]:
        temporary = None
        try:
            self._assert_path_identity()
            temporary = SQLiteEventStore(str(self.sessions_root / name), session_id=session_id)
            self._assert_path_identity()
            return self._checkpoint(temporary)
        except (OSError, sqlite3.Error) as error:
            return False, "WAL checkpoint open failed: %s" % error
        finally:
            if temporary is not None:
                self._safe_close(temporary)

    def purge_session(self, session_id: str, confirmed: bool = False) -> PurgeReport:
        """Checkpoint then unlink known names relative to the fixed directory FD."""
        if not confirmed:
            raise PermissionError("purge requires explicit confirmation")
        names, labels = self._names(session_id), self._report_labels(self._names(session_id))
        problem = self._preflight_entries(names)
        if problem:
            return PurgeReport(session_id, False, (), (), False, False, problem)
        checkpoint_attempted = False
        checkpoint_completed = False
        failure_reason = ""
        store = self._stores.pop(session_id, None)
        if store is not None:
            checkpoint_attempted = True
            checkpoint_completed, failure_reason = self._checkpoint(store)
            self._safe_close(store)
        if not checkpoint_attempted and self._entry(names[0]) is not None:
            checkpoint_attempted = True
            checkpoint_completed, failure_reason = self._temporary_checkpoint(session_id, names[0])
        elif checkpoint_attempted and not checkpoint_completed and self._entry(names[0]) is not None:
            checkpoint_completed, failure_reason = self._temporary_checkpoint(session_id, names[0])
        if checkpoint_attempted and not checkpoint_completed:
            return PurgeReport(session_id, False, (), (), True, False, failure_reason)
        problem = self._preflight_entries(names)
        if problem:
            return PurgeReport(session_id, False, (), (), checkpoint_attempted, checkpoint_completed, problem)
        deleted, absent = [], []
        for name, label in zip(names, labels):
            try:
                entry = self._entry(name)
                if entry is not None:
                    if stat.S_ISLNK(entry.st_mode):
                        return PurgeReport(session_id, False, tuple(deleted), tuple(absent), checkpoint_attempted, checkpoint_completed,
                                           "session database path is a symlink")
                    os.unlink(name, dir_fd=self._sessions_fd)
                    deleted.append(label)
                if self._entry(name) is None:
                    absent.append(label)
            except OSError as error:
                return PurgeReport(session_id, False, tuple(deleted), tuple(absent), checkpoint_attempted, checkpoint_completed,
                                   "session container deletion failed: %s" % error)
        try:
            self._assert_path_identity()
        except OSError as error:
            return PurgeReport(session_id, False, tuple(deleted), tuple(absent), checkpoint_attempted,
                               checkpoint_completed, "sessions path changed during purge: %s" % error)
        return PurgeReport(session_id, len(absent) == len(names), tuple(deleted), tuple(absent),
                           checkpoint_attempted, checkpoint_completed)

    def close(self) -> None:
        for session_id in tuple(self._stores):
            self.close_session(session_id)
        if getattr(self, "_sessions_fd", None) is not None:
            try:
                os.close(self._sessions_fd)
            except OSError:
                pass
            self._sessions_fd = None
