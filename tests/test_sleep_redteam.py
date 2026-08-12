"""Black-box hostile-input checks for the host-owned sleep boundary.

These tests deliberately use only fake clocks, fake transports, and disposable
SQLite files.  They assert operational lifecycle behaviour; ``sleep`` and
``wake`` are not claims about a model's mental state.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from strangeloop.capabilities import (Capability, CapabilityGrant,
                                      CapabilityRegistry, GrantScope,
                                      ResearchAutonomyProfile, ResearchBudget,
                                      ToolPlan)
from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind
from strangeloop.engine import StrangeloopAgent
from strangeloop.kimi_cli import KimiCliUsageResult, ManagedUsageWindow
from strangeloop.providers.kimi_code import (CallableSecretResolver,
                                             KimiCodeRuntime,
                                             KimiCodeSettings)
from strangeloop.quota import (QuotaController, QuotaSnapshot, QuotaSource,
                               UsageRecord)
from strangeloop.sleep import (SleepState, SleepWakeCoordinator,
                               SleepWakePolicy, build_public_archive)
from strangeloop.store import SQLiteEventStore
from strangeloop.tool_session import ToolSession
from strangeloop.tools import ControlledToolExecutor, ToolOutcome, ToolStatus
from strangeloop.unattended import UnattendedPolicy


NOW = datetime(2026, 8, 12, 12, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def rolling(remaining, reset_at):
    return ManagedUsageWindow("rolling_5h", 100, remaining, reset_at)


class CountingK3Transport:
    """A transport which proves that sleeping/quota-paused paths never call out."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, payload))
        raise AssertionError("a sleeping runtime must not contact K3")


class FakeUsageAdapter:
    """A foreground refresh fixture; it never uses a real credential or network."""

    def __init__(self, snapshot, windows):
        self.snapshot, self.windows = snapshot, tuple(windows)
        self.calls = 0

    def refresh_controller(self, controller):
        self.calls += 1
        controller.ingest_snapshot(self.snapshot, QuotaSource.PROVIDER_USAGE,
                                   "system", now=self.snapshot.observed_at)
        return KimiCliUsageResult(self.snapshot, None, windows=self.windows)


class SleepRedTeamTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.coordinator = SleepWakeCoordinator(
            SleepWakePolicy(threshold=.05, max_staleness=timedelta(minutes=5),
                            initial_backoff=timedelta(seconds=10),
                            max_backoff=timedelta(seconds=20)), self.clock)

    def _sleep(self, auto_wake=False, reset_at=None):
        reset_at = reset_at or NOW + timedelta(minutes=1)
        self.assertTrue(self.coordinator.prepare_sleep(
            [rolling(5, reset_at)], NOW, authoritative=True,
            user_auto_wake=auto_wake))
        return reset_at

    def _wake(self, observed_at):
        return self.coordinator.check_and_wake(
            [rolling(100, observed_at + timedelta(hours=1))], observed_at,
            authoritative=True, authority_token=self.coordinator.authority_token,
            expected_generation=self.coordinator.generation)

    def test_model_tool_and_archive_text_cannot_supply_wake_authority(self):
        self._sleep(auto_wake=False)
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        # Plain text and an archive are not telemetry APIs; only the host's
        # explicit user-policy + authority-token callback can reach wake.
        archive = build_public_archive(
            chain_head_sequence=0, chain_head_hash="0" * 64,
            active_seed_ids=("seed_active",), active_claim_ids=("claim_active",),
            pending_public_event_ids=(), quota_source="provider_usage",
            quota_observed_at=NOW, quota_reset_at=NOW + timedelta(hours=1),
            quota_remaining=100, quota_total=100)
        self.assertFalse(self.coordinator.check_and_wake(
            [rolling(100, NOW + timedelta(hours=1))], NOW,
            authoritative=True, authority_token=token, expected_generation=generation))
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)
        self.assertIn("seed_active", repr(archive.to_payload()))

        quota = QuotaController()
        snapshot = QuotaSnapshot(100, 100, NOW + timedelta(hours=1), NOW, 1.0, False)
        self.assertFalse(quota.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE, "model", now=NOW))
        self.assertFalse(quota.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE, "tool", now=NOW))
        self.assertFalse(quota.authoritative_wake_evidence(NOW).allow)

    def test_reset_clock_failure_stale_and_future_callbacks_remain_asleep(self):
        reset = self._sleep(auto_wake=True)
        self.clock.now = reset
        self.assertTrue(self.coordinator.refresh_due())
        self.assertFalse(self.coordinator.check_and_wake(
            [], self.clock.now, authority_token=self.coordinator.authority_token,
            expected_generation=self.coordinator.generation))
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)
        self.clock.now += timedelta(seconds=11)
        self.assertFalse(self.coordinator.check_and_wake(
            [rolling(100, self.clock.now + timedelta(hours=1))], NOW,
            authority_token=self.coordinator.authority_token,
            expected_generation=self.coordinator.generation))
        self.clock.now += timedelta(seconds=11)
        self.assertFalse(self.coordinator.check_and_wake(
            [rolling(100, self.clock.now + timedelta(hours=1))], self.clock.now + timedelta(seconds=1),
            authority_token=self.coordinator.authority_token,
            expected_generation=self.coordinator.generation))
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)

    def test_fresh_restored_snapshot_wakes_once_new_epoch_and_old_callback_is_inert(self):
        self._sleep(auto_wake=True)
        old_token, old_generation = self.coordinator.authority_token, self.coordinator.generation
        self.clock.now += timedelta(seconds=1)
        self.assertTrue(self._wake(self.clock.now))
        self.assertEqual(SleepState.READY, self.coordinator.state)
        self.assertEqual(2, self.coordinator.epoch)
        self.assertFalse(self.coordinator.check_and_wake(
            [rolling(100, self.clock.now + timedelta(hours=1))], self.clock.now,
            authority_token=old_token, expected_generation=old_generation))
        self.assertEqual(2, self.coordinator.epoch)

    def test_auto_wake_is_off_by_default_and_user_revocation_wins(self):
        self._sleep()
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        self.assertFalse(self._wake(NOW + timedelta(seconds=1)))
        self.assertFalse(self.coordinator.set_user_auto_wake(True, origin="model"))
        self.assertTrue(self.coordinator.set_user_auto_wake(True, origin="user"))
        self.assertTrue(self.coordinator.revoke_auto_wake())
        self.assertFalse(self.coordinator.check_and_wake(
            [rolling(100, NOW + timedelta(hours=1))], NOW + timedelta(seconds=1),
            authority_token=token, expected_generation=generation))
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)

    def test_stop_and_purge_invalidate_late_callbacks_and_never_restart_container(self):
        self._sleep(auto_wake=True)
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        self.coordinator.stop()
        self.assertEqual(SleepState.TERMINAL, self.coordinator.state)
        self.assertFalse(self.coordinator.check_and_wake(
            [rolling(100, NOW + timedelta(hours=1))], NOW + timedelta(seconds=1),
            authority_token=token, expected_generation=generation))
        self.assertFalse(self.coordinator.prepare_sleep([rolling(5, NOW + timedelta(hours=1))], NOW))
        self.coordinator.purge()
        self.assertEqual(SleepState.TERMINAL, self.coordinator.state)

    def test_archive_is_metadata_only_and_excludes_candidates_and_private_content(self):
        archive = build_public_archive(
            chain_head_sequence=7, chain_head_hash="a" * 64,
            active_seed_ids=("seed_approved",), active_claim_ids=("claim_approved",),
            pending_public_event_ids=("task_1",), quota_source="provider_usage",
            quota_observed_at=NOW, quota_reset_at=NOW + timedelta(hours=1),
            quota_remaining=5, quota_total=100)
        payload = archive.to_payload()
        self.assertEqual(["seed_approved"], payload["active_seed_ids"])
        self.assertEqual("rolling_5h", payload["quota"]["window_kind"])
        with self.assertRaises(ValueError):
            build_public_archive(
                chain_head_sequence=7, chain_head_hash="a" * 64,
                quota_source="provider_usage", quota_observed_at=NOW,
                quota_reset_at=NOW + timedelta(hours=1), quota_remaining=5, quota_total=100,
                quota_window_kind="weekly")
        text = repr(payload).lower()
        for forbidden in ("prompt", "chain_of_thought", "key", "token", "raw output", "path"):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("seed_candidate", text)

    def test_sleep_wake_events_cannot_be_reward_td_or_memory_claim_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteEventStore(directory + "/events.sqlite", session_id="redteam")
            try:
                # Lifecycle payload schemas reject both private content and any
                # attempt to turn a sleep record into generic evidence.
                with self.assertRaises(ValueError):
                    store.append(CognitiveEvent(
                        session_id="redteam", kind=EventKind.SLEEP_ARCHIVE,
                        source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
                        payload={"prompt": "wake me", "chain_of_thought": "hidden"}))
                observation = store.append(CognitiveEvent(
                    session_id="redteam", kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="user", payload={"content": "normal"}))
                with self.assertRaises(ValueError):
                    store.append(CognitiveEvent(
                        session_id="redteam", kind=EventKind.REWARD_OBSERVATION,
                        source_kind=SourceKind.USER, source_ref="user", parent_event_ids=(observation.event_id,),
                        payload={"reward_id": "sleep_reward", "target_event_id": "not_an_event",
                                 "signal_kind": "user_feedback", "normalized_value": 1.0,
                                 "evaluator_ref": "user", "scale_version": "v1", "observed_at": NOW.isoformat()}))
            finally:
                store.close()

    def test_sleeping_quota_path_never_contacts_k3_or_creates_drive_reward(self):
        quota = QuotaController()
        quota.ingest_snapshot(QuotaSnapshot(100, 0, NOW + timedelta(hours=1), NOW, 1.0, False),
                              QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        transport = CountingK3Transport()
        runtime = KimiCodeRuntime(KimiCodeSettings(), CallableSecretResolver(lambda: "test-key"),
                                  transport, quota_controller=quota)
        agent = StrangeloopAgent(session_id="sleep-network", runtime=runtime,
                                 quota_controller=quota)
        self._sleep(auto_wake=False)
        agent.run_turn("do not make a provider request while quota sleeping")
        self.assertEqual([], transport.calls)
        self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                              if event.kind in (EventKind.REWARD_OBSERVATION,
                                                EventKind.VALUE_ESTIMATE,
                                                EventKind.RPE_UPDATE)])

    def test_engine_wake_requires_one_fresh_foreground_refresh_and_late_poll_is_inert(self):
        """The durable lifecycle has one wake path and no scheduler/container resurrection."""
        quota = QuotaController()
        agent = StrangeloopAgent(session_id="wake-once", quota_controller=quota,
                                 sleep_coordinator=self.coordinator)
        reset = NOW + timedelta(minutes=1)
        initial_snapshot = QuotaSnapshot(100, 5, reset, NOW, 1.0, False)
        quota.ingest_snapshot(initial_snapshot, QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        initial = KimiCliUsageResult(initial_snapshot, None, windows=(rolling(5, reset),))
        self.assertTrue(agent.set_auto_wake(True))
        self.assertTrue(agent.update_managed_usage(initial))
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)
        self.clock.now = reset
        refreshed_at = reset + timedelta(seconds=1)
        self.clock.now = refreshed_at
        adapter = FakeUsageAdapter(
            QuotaSnapshot(100, 100, refreshed_at + timedelta(hours=1), refreshed_at, 1.0, False),
            (rolling(100, refreshed_at + timedelta(hours=1)),))
        self.assertTrue(agent.poll_sleep(adapter))
        self.assertEqual(1, adapter.calls)
        kinds = [event.kind for event in agent.event_store.list("wake-once")]
        kinds = kinds[kinds.index(EventKind.SLEEP_ARCHIVE):]
        self.assertEqual([EventKind.SLEEP_ARCHIVE, EventKind.SLEEP_ENTERED,
                          EventKind.PROVIDER_USAGE_EVIDENCE, EventKind.WAKE_CHECK,
                          EventKind.WAKE_READY, EventKind.AWAKE], kinds[:6])
        self.assertFalse(agent.poll_sleep(adapter))
        self.assertEqual(1, adapter.calls)
        self.assertEqual(2, self.coordinator.epoch)

    def test_pre_sleep_grant_and_plan_stay_invalid_after_wake(self):
        """Wake restores no capability or deferred tool execution authority."""
        registry = CapabilityRegistry()
        store = SQLiteEventStore(session_id="old-authority")
        try:
            session = ToolSession("old-authority", "fixture-workspace", registry,
                                  ControlledToolExecutor("."), event_store=store)
            command = store.append(CognitiveEvent(
                session_id="old-authority", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={"content": "grant"}))
            grant = CapabilityGrant("old-authority", Capability.REPO_STATUS,
                                    GrantScope(workspace_id="fixture-workspace"),
                                    (NOW + timedelta(hours=1)).isoformat(), 1,
                                    issued_at=NOW.isoformat())
            session.register_user_grant(grant, command.event_id)
            old_plan = ToolPlan("old-authority", grant.grant_id, "repo.status",
                                {"workspace_id": "fixture-workspace"})
            quota = QuotaController()
            agent = StrangeloopAgent(session_id="old-authority", event_store=store,
                                     quota_controller=quota, tool_session=session,
                                     sleep_coordinator=self.coordinator)
            reset = NOW + timedelta(minutes=1)
            self.assertTrue(agent.set_auto_wake(True))
            initial_snapshot = QuotaSnapshot(100, 5, reset, NOW, 1.0, False)
            quota.ingest_snapshot(initial_snapshot, QuotaSource.PROVIDER_USAGE, "system", now=NOW)
            self.assertTrue(agent.update_managed_usage(KimiCliUsageResult(
                initial_snapshot, None, windows=(rolling(5, reset),))))
            self.assertEqual("restart_suspended", registry.snapshot(grant.grant_id).status.value)
            self.clock.now = reset + timedelta(seconds=1)
            refreshed = self.clock.now
            self.assertTrue(agent.poll_sleep(FakeUsageAdapter(
                QuotaSnapshot(100, 100, refreshed + timedelta(hours=1), refreshed, 1.0, False),
                (rolling(100, refreshed + timedelta(hours=1)),))))
            with self.assertRaises(PermissionError):
                registry.consume(old_plan, now=refreshed.isoformat())
        finally:
            store.close()

    def test_wake_continuation_is_one_fresh_read_only_run_and_never_reuses_old_authority(self):
        """A fresh provider poll may consume one USER capsule, never old work.

        This is deliberately a black-box integration test: K3 and all public
        web tools are fixtures, while the lifecycle, capability registry, and
        quota reservation boundary are the production objects.
        """
        class ReadOnlyBackend:
            def __init__(self):
                self.calls = 0

            def execute(self, plan, registry):
                self.calls += 1
                registry.consume(plan)
                return ToolOutcome.from_bytes(plan.tool_name, ToolStatus.SUCCEEDED,
                                              b"bounded public fixture")

        transport_calls = []

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            transport_calls.append(payload)
            return {"choices": [{"message": {"content":
                '{"intents":[{"tool_name":"web_fetch","arguments":'
                '{"url":"https://ordinary.example/research"},'
                '"rationale_summary":"Read a public page."}]}'}}]}

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteEventStore(directory + "/events.sqlite", session_id="wake-research")
            try:
                # Keep synthetic sleep telemetry slightly behind the real
                # ledger clock.  The store correctly refuses a wake event
                # whose purported wake time is in the future.
                observed = datetime.now(timezone.utc) - timedelta(seconds=3)
                self.clock.now = observed
                registry, backend = CapabilityRegistry(), ReadOnlyBackend()
                session = ToolSession("wake-research", "fixture-workspace", registry,
                                      ControlledToolExecutor(directory), event_store=store,
                                      web_fetch=backend, web_search=backend, browser_read=backend)
                quota = QuotaController()
                runtime = KimiCodeRuntime(KimiCodeSettings(),
                                          CallableSecretResolver(lambda: "fixture-secret"),
                                          transport, quota_controller=quota)
                agent = StrangeloopAgent(session_id="wake-research", event_store=store,
                                         runtime=runtime, quota_controller=quota,
                                         tool_session=session,
                                         sleep_coordinator=self.coordinator)
                approval = store.append(CognitiveEvent(
                    session_id="wake-research", kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="user",
                    payload={"content": "continue one bounded public research run", "channel": "text"}))
                profile = ResearchAutonomyProfile("fixture-workspace", ResearchBudget(
                    max_tool_calls=2, max_total_bytes=32 * 1024,
                    max_response_bytes=1024, max_wall_ms=30_000, ttl_seconds=300))
                policy = UnattendedPolicy.user_issued(
                    "Read ordinary public documentation only.", observed + timedelta(hours=1), observed)

                # An unrelated old capability/plan and a reserved model call
                # are both valid before sleep and must become unusable after it.
                old_grant = CapabilityGrant("wake-research", Capability.REPO_STATUS,
                                            GrantScope(workspace_id="fixture-workspace"),
                                            (observed + timedelta(hours=1)).isoformat(), 1,
                                            issued_at=observed.isoformat())
                session.register_user_grant(old_grant, approval.event_id)
                old_plan = ToolPlan("wake-research", old_grant.grant_id, "repo.status",
                                    {"workspace_id": "fixture-workspace"})
                self.assertTrue(agent.set_auto_wake(True))
                self.assertTrue(agent.start_unattended_research(profile, policy, approval.event_id)["active"])
                armed = agent.prepare_unattended_wake_continuation(profile, policy, approval.event_id)
                self.assertTrue(armed["armed"])

                reset = observed + timedelta(seconds=1)
                initial = QuotaSnapshot(100, 5, reset, observed, 1.0, False)
                quota.ingest_snapshot(initial, QuotaSource.PROVIDER_USAGE, "system", now=observed)
                old_reservation = quota.reserve_call(now=observed)
                self.assertIsNotNone(old_reservation)
                self.assertTrue(agent.update_managed_usage(KimiCliUsageResult(
                    initial, None, windows=(rolling(5, reset),))))
                archive = [event for event in store.list("wake-research")
                           if event.kind == EventKind.SLEEP_ARCHIVE]
                self.assertEqual(1, len(archive))
                self.assertTrue(agent.wake_continuation_status()["armed"])
                self.assertFalse(quota.commit(old_reservation, UsageRecord(prompt_tokens=1)))

                self.clock.now = reset + timedelta(seconds=1)
                refreshed = self.clock.now
                fresh = QuotaSnapshot(100, 100, refreshed + timedelta(hours=1), refreshed, 1.0, False)
                self.assertTrue(agent.poll_sleep(FakeUsageAdapter(
                    fresh, (rolling(100, refreshed + timedelta(hours=1)),))))
                self.assertGreaterEqual(len(transport_calls), 1)
                model_calls_after_wake = len(transport_calls)
                self.assertGreaterEqual(backend.calls, 1)
                with self.assertRaises(PermissionError):
                    registry.consume(old_plan, now=refreshed.isoformat())

                events = store.list("wake-research")
                runs = [event for event in events if event.kind == EventKind.UNATTENDED_WAKE_RUN]
                self.assertEqual(1, len(runs))
                self.assertEqual("completed", runs[0].payload["status"])
                self.assertEqual(1, runs[0].payload["run_index"])
                self.assertNotEqual(profile.profile_id, runs[0].payload["profile_id"])
                self.assertEqual(policy.policy_id, runs[0].payload["authorization_policy_id"])
                self.assertNotEqual(policy.policy_id, runs[0].payload["runtime_policy_id"])
                self.assertEqual({"web.fetch"}, {
                    event.payload["tool_name"] for event in events if event.kind == EventKind.TOOL_RESULT})
                self.assertFalse(agent.wake_continuation_status()["armed"])

                # The consumed compare-and-swap callback and a duplicate
                # foreground poll cannot create another run or provider call.
                self.assertFalse(agent.poll_sleep(FakeUsageAdapter(
                    fresh, (rolling(100, refreshed + timedelta(hours=1)),))))
                self.assertEqual(1, len([event for event in store.list("wake-research")
                                         if event.kind == EventKind.UNATTENDED_WAKE_RUN]))
                self.assertEqual(model_calls_after_wake, len(transport_calls))
            finally:
                store.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
