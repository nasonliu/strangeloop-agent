import io
import os
import struct
import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from contextlib import redirect_stdout
from unittest.mock import patch

from strangeloop.cli import (BANNER, _resume_expedition_after_wake,
                             _sleep_poll_timeout, _sleep_wait_due, build_parser,
                             main, run_repl)
from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind
from strangeloop.engine import StrangeloopAgent
from strangeloop.memory import PurgeReport, SessionMemoryManager
from strangeloop.store import SQLiteEventStore
from strangeloop.providers.kimi_code import (CallableSecretResolver, KimiCodeRuntime,
                                             KimiCodeSettings)
from strangeloop.quota import QuotaController, QuotaSnapshot, QuotaSource, UsageRecord
from strangeloop.capabilities import CapabilityRegistry
from strangeloop.tools import ControlledToolExecutor
from strangeloop.tool_session import ToolSession
from strangeloop.drives import DualIntrinsicDrives
from strangeloop.sleep import SleepWakeCoordinator, SleepWakePolicy
from strangeloop.kimi_cli import KimiCliUsageResult, ManagedUsageWindow


class CliTests(unittest.TestCase):
    def test_expedition_and_unattended_modes_cannot_compete_for_one_controller(self):
        with self.assertRaises(SystemExit):
            main(["--expedition", "--unattended"])

    def test_five_hour_expedition_config_must_fit_slice_and_k3_call_caps(self):
        with self.assertRaises(SystemExit):
            main(["--expedition", "--expedition-authorization-seconds", "18000",
                  "--expedition-slice-seconds", "30"])
        with self.assertRaises(SystemExit):
            main(["--expedition", "--expedition-authorization-seconds", "18000",
                  "--expedition-slice-seconds", "300",
                  "--expedition-max-calls-per-slice", "16"])

    def test_foreground_wake_resumes_only_an_already_ready_expedition(self):
        calls = []

        class Agent:
            def expedition_status(self):
                return {"state": "ready"}

            def run_expedition_foreground(self, adapter, on_slice=None):
                calls.append(adapter)
                if on_slice:
                    on_slice({"state": "sleeping"})

        output = io.StringIO()
        adapter = object()
        with redirect_stdout(output):
            _resume_expedition_after_wake(Agent(), adapter)
        self.assertEqual([adapter], calls)
        self.assertIn('"state": "sleeping"', output.getvalue())

    def test_unattended_flags_are_bounded_and_command_is_inspectable(self):
        args = build_parser().parse_args(["--unattended", "--unattended-goal", "read public docs",
                                          "--unattended-max-calls", "2", "--unattended-wall-seconds", "5"])
        self.assertTrue(args.unattended)
        self.assertEqual(2, args.unattended_max_calls)
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(StrangeloopAgent(session_id="unattended-cli"),
                     ["/unattended status", "/unattended start public docs", "/quit"])
        self.assertIn('"restart_behavior": "stopped_requires_fresh_user_policy"', output.getvalue())
        self.assertIn("Unattended research refused", output.getvalue())

    def test_sleep_auto_wake_waits_before_reset_not_only_after_it(self):
        now = datetime.now(timezone.utc)
        coordinator = SleepWakeCoordinator(SleepWakePolicy(threshold=.10), clock=lambda: now)
        window = ManagedUsageWindow("rolling_5h", 10, 1, now + timedelta(seconds=20))
        self.assertTrue(coordinator.prepare_sleep((window,), now, user_auto_wake=True))
        agent = StrangeloopAgent(session_id="sleep-wait", sleep_coordinator=coordinator)
        self.assertTrue(_sleep_wait_due(agent))
        self.assertGreater(_sleep_poll_timeout(agent), 0.0)
        self.assertLessEqual(_sleep_poll_timeout(agent), 1.0)

    def test_grant_feedback_and_improvement_commands_are_public_and_bounded(self):
        agent = StrangeloopAgent(
            session_id="interactive",
            tool_session=ToolSession("interactive", "workspace", CapabilityRegistry(),
                                     ControlledToolExecutor(os.getcwd())),
            drives=DualIntrinsicDrives())
        result = agent.run_turn("outcome")
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/grant repo.status 1 60", "/grants",
                             "/feedback %s accept" % result.event_ids[-1], "/drives",
                             "/improvement", "/quit"])
        text = output.getvalue()
        self.assertIn("repo.status", text)
        self.assertIn("user_alignment", text)
        self.assertIn("configured", text)

    def test_repl_banner_and_commands(self):
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(StrangeloopAgent(session_id="cli"), ["/state", "/quit"])
        self.assertIn(BANNER, output.getvalue())
        self.assertIn('"event_count": 0', output.getvalue())

    def test_provider_status_discloses_k3_egress_without_resolving_credentials(self):
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(StrangeloopAgent(session_id="provider"), ["/provider", "/quit"])
        status = json.loads(output.getvalue().splitlines()[1])
        self.assertIsNone(status["model"])
        self.assertFalse(status["configured"])
        self.assertIn("api.kimi.com", status["egress"])
        self.assertEqual("not_enabled", status["tools"])

    def test_quota_command_keeps_unknown_and_local_usage_distinct_from_plan_balance(self):
        controller = QuotaController()
        runtime = KimiCodeRuntime(KimiCodeSettings(), CallableSecretResolver(lambda: "test-key"),
                                  quota_controller=controller)
        agent = StrangeloopAgent(session_id="quota", runtime=runtime)
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/quota", "/quit"])
        status = json.loads(output.getvalue().splitlines()[1])
        self.assertEqual("unknown", status["authority"])
        self.assertEqual("unknown", status["quota_telemetry"])
        self.assertFalse(status["local_ledger"]["is_code_plan_balance"])
        self.assertIn("not a Code Plan balance", status["note"])

    def test_quota_command_projects_trusted_snapshot_and_observed_ledger(self):
        controller = QuotaController()
        now = datetime.now(timezone.utc)
        controller.ingest_snapshot(
            QuotaSnapshot(100, 80, now + timedelta(hours=1), now, 1.0, False),
            QuotaSource.PROVIDER_USAGE, "system")
        reservation = controller.reserve_call()
        self.assertIsNotNone(reservation)
        controller.commit(reservation, UsageRecord(prompt_tokens=3, completion_tokens=2))
        runtime = KimiCodeRuntime(KimiCodeSettings(), CallableSecretResolver(lambda: "test-key"),
                                  quota_controller=controller)
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(StrangeloopAgent(session_id="quota-known", runtime=runtime), ["/quota", "/quit"])
        status = json.loads(output.getvalue().splitlines()[1])
        self.assertEqual("provider_usage", status["authority"])
        self.assertEqual("current", status["freshness"])
        self.assertEqual(5, status["local_ledger"]["observed_tokens"])
        self.assertEqual(1, status["local_ledger"]["observed_calls"])
        self.assertNotIn("test-key", output.getvalue())

    def test_main_attaches_a_quota_controller_without_resolving_a_secret(self):
        captured = []

        def capture(agent, **kwargs):
            captured.append(agent)

        with patch("strangeloop.cli.run_repl", side_effect=capture), \
             patch("strangeloop.cli.KimiCliManagedUsageAdapter.refresh_controller") as refresh:
            self.assertEqual(0, main(["--session", "quota-main"]))
        self.assertEqual(1, len(captured))
        self.assertIsInstance(captured[0].runtime._quota_controller, QuotaController)
        self.assertEqual(90.0, captured[0].runtime.settings.timeout_seconds)
        refresh.assert_called_once()

    def test_auto_wake_flag_is_explicit_auditable_and_enables_foreground_sleep_wait(self):
        now = datetime.now(timezone.utc)
        snapshot = QuotaSnapshot(10, 1, now + timedelta(hours=1), now, 1.0, False)
        usage = KimiCliUsageResult(snapshot, None,
                                   windows=(ManagedUsageWindow("rolling_5h", 10, 1,
                                                               now + timedelta(hours=1)),))
        observed = {}

        def refresh(controller):
            controller.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE, "system")
            return usage

        def capture(agent, **kwargs):
            observed["status"] = agent.sleep_status()
            observed["wait"] = _sleep_wait_due(agent)
            observed["policies"] = [event for event in agent.event_store.list(agent.session_id)
                                    if event.kind == EventKind.AUTO_WAKE_POLICY]

        with patch("strangeloop.cli.KimiCliOAuthUsageBridge.refresh_controller", side_effect=refresh), \
             patch("strangeloop.cli.run_repl", side_effect=capture):
            self.assertEqual(0, main(["--session", "auto-wake", "--auto-wake"]))
        self.assertEqual("sleeping", observed["status"]["state"])
        self.assertTrue(observed["status"]["auto_wake_user_approved"])
        self.assertTrue(observed["wait"])
        self.assertEqual(1, len(observed["policies"]))
        self.assertEqual(SourceKind.USER, observed["policies"][0].source_kind)
        self.assertEqual("enable", observed["policies"][0].payload["action"])

    def test_default_cli_does_not_create_auto_wake_policy_or_enter_poll_wait(self):
        observed = {}

        def capture(agent, **kwargs):
            observed["status"] = agent.sleep_status()
            observed["wait"] = _sleep_wait_due(agent)
            observed["policies"] = [event for event in agent.event_store.list(agent.session_id)
                                    if event.kind == EventKind.AUTO_WAKE_POLICY]

        with patch("strangeloop.cli.KimiCliManagedUsageAdapter.refresh_controller",
                   return_value=KimiCliUsageResult(None, "unknown")), \
             patch("strangeloop.cli.run_repl", side_effect=capture) as repl:
            self.assertEqual(0, main(["--session", "default-no-auto-wake"]))
        self.assertFalse(observed["status"]["auto_wake_user_approved"])
        self.assertFalse(observed["wait"])
        self.assertEqual([], observed["policies"])
        repl.assert_called_once()

    def test_monitor_commands_start_status_and_stop_local_surface(self):
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(StrangeloopAgent(session_id="monitor"),
                     ["/monitor start", "/monitor status", "/monitor stop", "/quit"])
        text = output.getvalue()
        self.assertIn("Monitor: http://127.0.0.1:", text)
        self.assertIn('"running": true', text)
        self.assertIn("Monitor stopped.", text)

    def test_default_operation_is_not_persistent(self):
        first = SQLiteEventStore()
        StrangeloopAgent(session_id="default", event_store=first).run_turn("one")
        second = SQLiteEventStore()
        self.assertEqual([], second.list("default"))
        first.close()
        second.close()

    def test_memory_root_persists_session_across_manager_restart(self):
        with tempfile.TemporaryDirectory() as root:
            first = SessionMemoryManager(root)
            store = first.open_session("durable")
            StrangeloopAgent(session_id="durable", event_store=store).run_turn("one")
            first.close()
            second = SessionMemoryManager(root)
            reopened = second.open_session("durable")
            # A completed turn now includes one bounded public mirror record.
            self.assertEqual(5, len(reopened.list("durable")))
            second.close()

    def test_purge_requires_exact_current_session_confirmation(self):
        agent = StrangeloopAgent(session_id="current")
        agent.run_turn("keep until confirmed")
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/purge", "/purge another", "/state", "/purge current", "/state", "/quit"])
        text = output.getvalue()
        self.assertIn("Confirm with: /purge current", text)
        self.assertIn("Refused: session id must exactly match", text)
        self.assertIn("Logical session data deleted.", text)
        self.assertIn('"event_count": 5', text)
        self.assertIn('"event_count": 0', text)

    def test_export_returns_current_session_records(self):
        agent = StrangeloopAgent(session_id="export")
        agent.run_turn("one")
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/export", "/quit"])
        self.assertIn('"session_id": "export"', output.getvalue())
        self.assertIn('"events"', output.getvalue())

    def test_events_command_uses_grouped_view(self):
        agent = StrangeloopAgent(session_id="events")
        agent.run_turn("one")
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/events", "/quit"])
        self.assertIn('"observations"', output.getvalue())
        self.assertIn('"actions_and_decisions"', output.getvalue())

    def test_memory_root_purge_deletes_container_and_ends_repl(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            store = manager.open_session("managed")
            agent = StrangeloopAgent(session_id="managed", event_store=store)
            agent.run_turn("one")
            path = manager.database_path("managed")
            output = io.StringIO()
            with redirect_stdout(output):
                run_repl(agent, ["/purge managed", "should-not-run"], memory_manager=manager)
            self.assertFalse(path.exists())
            self.assertIn('"container_deleted": true', output.getvalue())
            self.assertNotIn("should-not-run", output.getvalue())
            manager.close()

    def test_busy_managed_purge_shows_failure_and_keeps_repl_available(self):
        class BusyManager:
            def __init__(self, store):
                self.store = store

            def purge_session(self, session_id, confirmed=False):
                self.called = (session_id, confirmed)
                return PurgeReport(session_id=session_id, container_deleted=False,
                                   deleted_paths=(), absent_paths=(), checkpoint_attempted=True,
                                   checkpoint_completed=False, failure_reason="WAL checkpoint is busy")

            def open_session(self, session_id):
                return self.store

        agent = StrangeloopAgent(session_id="busy")
        agent.run_turn("one")
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/purge busy", "/state", "/quit"], BusyManager(agent.event_store))
        text = output.getvalue()
        self.assertIn('"container_deleted": false', text)
        self.assertIn('"failure_reason": "WAL checkpoint is busy"', text)
        self.assertIn('"event_count": 5', text)

    def test_correct_command_records_a_bounded_correction(self):
        agent = StrangeloopAgent(session_id="correct")
        result = agent.run_turn("initial")
        counterevidence = agent.event_store.append(CognitiveEvent(
            session_id="correct", kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER,
            source_ref="user", payload={"content": "counter"},
        ))
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(agent, ["/correct %s %s" % (result.event_ids[2], counterevidence.event_id), "/quit"])
        self.assertIn("Correction recorded:", output.getvalue())
        self.assertEqual("correction", agent.event_store.list("correct")[-1].kind.value)

    def test_loop_value_reward_and_media_commands_use_bounded_public_records(self):
        agent = StrangeloopAgent(session_id="runtime-cli")
        result = agent.run_turn("input")
        media = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" +
                 struct.pack(">II", 2, 3) + b"\x08\x02\x00\x00\x00")
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            handle.write(media)
            handle.close()
            output = io.StringIO()
            with redirect_stdout(output):
                run_repl(agent, ["/media " + handle.name, "/loop start 1", "/loop run",
                                 "/value %s respond" % result.event_ids[-1],
                                 "/reward %s 0.5" % result.event_ids[-1], "/events", "/quit"])
            text = output.getvalue()
            self.assertIn("Only metadata and bounded percepts were recorded.", text)
            self.assertNotIn(handle.name, repr(agent.export_session()))
            self.assertIn("Reward and TD update recorded:", text)
            self.assertIn('"media_observations"', text)
            self.assertIn('"autonomy"', text)
            self.assertTrue(agent.event_store.verify_chain("runtime-cli"))
        finally:
            if not handle.closed:
                handle.close()
            os.unlink(handle.name)
