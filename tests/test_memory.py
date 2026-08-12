import os
import sqlite3
import stat
import tempfile
import unittest

from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind
from strangeloop.memory import SessionMemoryManager


class SessionMemoryManagerTests(unittest.TestCase):
    def event(self, session_id, content):
        return CognitiveEvent(session_id=session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="test", payload={"content": content})

    def test_purge_removes_one_independent_container_without_touching_another(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            a, b = manager.open_session("A/session with spaces"), manager.open_session("B")
            a.append(self.event("A/session with spaces", "A"))
            b.append(self.event("B", "B"))
            a_path, b_path = manager.database_path("A/session with spaces"), manager.database_path("B")
            self.assertNotIn("A/session with spaces", a_path.name)
            self.assertNotEqual(a_path, b_path)
            self.assertEqual(stat.S_IMODE(a_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(manager.sessions_root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(manager.root.stat().st_mode), 0o700)
            report = manager.purge_session("A/session with spaces", confirmed=True)
            self.assertTrue(report.container_deleted)
            self.assertFalse(a_path.exists())
            self.assertTrue(b_path.exists())
            self.assertEqual([event.payload["content"] for event in b.list("B")], ["B"])
            reopened = manager.open_session("A/session with spaces")
            self.assertEqual(reopened.list("A/session with spaces"), [])
            manager.close()

    def test_nonexistent_root_is_created_and_supports_open_then_purge(self):
        with tempfile.TemporaryDirectory() as parent:
            root = os.path.join(parent, "new-memory-root")
            self.assertFalse(os.path.exists(root))
            manager = SessionMemoryManager(root)
            self.assertTrue(os.path.isdir(root))
            store = manager.open_session("first-run")
            store.append(self.event("first-run", "first persistent record"))
            report = manager.purge_session("first-run", confirmed=True)
            self.assertTrue(report.container_deleted)
            self.assertFalse(manager.database_path("first-run").exists())
            manager.close()

    def test_busy_external_reader_prevents_deletion_until_retry(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            store = manager.open_session("reader")
            store.append(self.event("reader", "SECRET"))
            reader = sqlite3.connect(str(manager.database_path("reader")))
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM cognitive_events").fetchall()
            report = manager.purge_session("reader", confirmed=True)
            self.assertFalse(report.container_deleted)
            self.assertFalse(report.checkpoint_completed)
            self.assertTrue(manager.database_path("reader").exists())
            self.assertEqual(reader.execute("SELECT payload_json FROM cognitive_events").fetchone()[0], '{"content":"SECRET"}')
            reader.close()
            retry = manager.purge_session("reader", confirmed=True)
            self.assertTrue(retry.container_deleted)
            manager.close()

    def test_restarted_manager_checks_unknown_reader_before_deleting(self):
        with tempfile.TemporaryDirectory() as root:
            creator = SessionMemoryManager(root)
            creator.open_session("unknown-reader").append(self.event("unknown-reader", "SECRET"))
            database = creator.database_path("unknown-reader")
            reader = sqlite3.connect(str(database))
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM cognitive_events").fetchall()
            creator.open_session("unknown-reader").append(self.event("unknown-reader", "later WAL frame"))
            creator.close()
            restarted = SessionMemoryManager(root)
            blocked = restarted.purge_session("unknown-reader", confirmed=True)
            self.assertFalse(blocked.container_deleted)
            self.assertFalse(blocked.checkpoint_completed)
            self.assertTrue(database.exists())
            reader.close()
            self.assertTrue(restarted.purge_session("unknown-reader", confirmed=True).container_deleted)
            restarted.close()

    def test_rejects_root_symlink_and_recovers_from_closed_known_handle(self):
        with tempfile.TemporaryDirectory() as parent:
            target = os.path.join(parent, "target")
            os.mkdir(target)
            link = os.path.join(parent, "memory-link")
            os.symlink(target, link)
            with self.assertRaises(ValueError):
                SessionMemoryManager(link)
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            store = manager.open_session("closed")
            store.append(self.event("closed", "record"))
            store.close()
            report = manager.purge_session("closed", confirmed=True)
            self.assertTrue(report.container_deleted)
            manager.close()

    def test_purge_requires_confirmation_and_reports_known_limits(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            store = manager.open_session("session")
            store.append(self.event("session", "record"))
            with self.assertRaises(PermissionError):
                manager.purge_session("session")
            report = manager.purge_session("session", confirmed=True)
            self.assertTrue(report.checkpoint_attempted)
            self.assertIn("SSD", report.limitations[0])
            self.assertIn("external file descriptors", report.limitations[1])
            self.assertFalse(manager.database_path("session").exists())
            self.assertFalse(os.path.exists(str(manager.database_path("session")) + "-wal"))
            self.assertFalse(os.path.exists(str(manager.database_path("session")) + "-shm"))
            manager.close()

    def test_replaced_parent_path_cannot_redirect_relative_purge(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            manager = SessionMemoryManager(root)
            store = manager.open_session("race")
            store.append(self.event("race", "inside"))
            name = manager.database_path("race").name
            sentinel = os.path.join(outside, name)
            with open(sentinel, "w") as handle:
                handle.write("OUTSIDE SENTINEL")
            original = os.path.join(root, "sessions")
            held = os.path.join(root, "sessions-held")
            os.rename(original, held)
            os.symlink(outside, original)
            try:
                report = manager.purge_session("race", confirmed=True)
                self.assertFalse(report.container_deleted)
                self.assertTrue(report.failure_reason)
                with open(sentinel) as handle:
                    self.assertEqual(handle.read(), "OUTSIDE SENTINEL")
                self.assertTrue(os.path.exists(os.path.join(held, name)))
            finally:
                if os.path.islink(original):
                    os.unlink(original)
                if os.path.exists(held):
                    os.rename(held, original)
                manager.close()

    def test_unlink_error_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            manager.open_session("error").append(self.event("error", "record"))
            original_unlink = os.unlink
            def failing_unlink(path, *args, **kwargs):
                if kwargs.get("dir_fd") == manager._sessions_fd:
                    raise PermissionError("fixture denial")
                return original_unlink(path, *args, **kwargs)
            try:
                os.unlink = failing_unlink
                report = manager.purge_session("error", confirmed=True)
            finally:
                os.unlink = original_unlink
            self.assertFalse(report.container_deleted)
            self.assertIn("deletion failed", report.failure_reason)
            manager.close()
