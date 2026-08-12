"""Black-box interruption and authorization tests for public-web expeditions.

The tests use only public engine/CLI/provider entry points.  They do not make
network requests and never inspect an expedition scheduler's private state.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from strangeloop.capabilities import (CapabilityRegistry, ResearchAutonomyProfile,
                                      ResearchBudget)
from strangeloop.cli import _resume_expedition_after_wake, run_repl
from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind, WorkspaceFrame
from strangeloop.engine import StrangeloopAgent
from strangeloop.kimi_usage_bridge import (BridgeHttpResponse, BridgeReadiness,
                                           KimiCliOAuthUsageBridge)
from strangeloop.kimi_cli import (KimiCliUsageResult, ManagedUsageWindow,
                                  normalize_managed_usage, normalize_managed_usage_windows)
from strangeloop.providers.kimi_code import (CallableSecretResolver, KimiCodeRuntime,
                                             KimiCodeSettings, ToolIntent)
from strangeloop.quota import (ForegroundRefreshStatus, QuotaController,
                               QuotaSnapshot, QuotaSource)
from strangeloop.store import SQLiteEventStore
from strangeloop.sleep import SleepWakeCoordinator, SleepState
from strangeloop.tool_session import ToolSession
from strangeloop.tools import (ControlledToolExecutor, PublicWebFetch,
                               SafePublicWebSearch, StatelessBrowserRead)
from strangeloop.unattended import UnattendedPolicy


NOW = datetime(2026, 8, 12, 12, tzinfo=timezone.utc)


class _NoNetworkTransport:
    def __call__(self, url, headers, payload, timeout):
        del url, headers, payload, timeout
        raise AssertionError("red-team fixture must not contact K3")


class _FreshQuotaAdapter:
    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        return ForegroundRefreshStatus(
            True, True, True, False, "authoritative_quota_available", None,
            observed, observed + timedelta(seconds=30),
            "provider_plan_units_not_currency", None)


def _agent(session_id="expedition-redteam", sleep_coordinator=None):
    store = SQLiteEventStore(session_id=session_id)
    registry = CapabilityRegistry()
    tools = ToolSession(
        session_id, "workspace", registry, ControlledToolExecutor(os.getcwd()), event_store=store,
        web_fetch=PublicWebFetch(()), web_search=SafePublicWebSearch(PublicWebFetch(())),
        browser_read=StatelessBrowserRead(PublicWebFetch(())),
        )
    runtime = KimiCodeRuntime(
        KimiCodeSettings(max_calls=1000), CallableSecretResolver(lambda: "fixture-key"),
        _NoNetworkTransport())
    quota = QuotaController()
    moment = datetime.now(timezone.utc)
    quota.ingest_snapshot(QuotaSnapshot(100, 100, moment + timedelta(hours=1), moment,
                                        1.0, False), QuotaSource.PROVIDER_USAGE,
                          "system", now=moment)
    return StrangeloopAgent(session_id=session_id, event_store=store, runtime=runtime,
                             quota_controller=quota, tool_session=tools,
                             sleep_coordinator=sleep_coordinator), tools


class _TransientThenFreshQuotaAdapter:
    """Foreground-only bridge double: no cached result authorizes a slice."""

    def __init__(self, retry_at):
        self.retry_at = retry_at
        self.calls = 0

    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        self.calls += 1
        if self.calls == 1:
            return ForegroundRefreshStatus(
                True, False, False, False, "quota_refresh_transient_retry_pending",
                "bridge_start_failed", observed, self.retry_at,
                "unknown_provider_plan_cost", None, degraded=True, cache_used=True,
                next_retry_at=self.retry_at)
        return ForegroundRefreshStatus(
            True, True, True, False, "authoritative_quota_available", None,
            observed, observed + timedelta(seconds=30),
            "provider_plan_units_not_currency", None)


class _UnknownQuotaAdapter:
    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        return ForegroundRefreshStatus(True, False, False, False,
                                       "quota_refresh_unknown_fail_closed", None,
                                       observed, None, "unknown_provider_plan_cost", None)


class _PermanentAuthQuotaAdapter:
    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        return ForegroundRefreshStatus(True, False, False, False,
                                       "quota_refresh_unknown_fail_closed", "oauth_unauthorized",
                                       observed, None, "unknown_provider_plan_cost", None)


class _PermanentSchemaQuotaAdapter:
    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        return ForegroundRefreshStatus(True, False, False, False,
                                       "quota_refresh_unknown_fail_closed", "invalid_json",
                                       observed, None, "unknown_provider_plan_cost", None)


class _BridgeExceptionAdapter:
    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, observed, force
        raise RuntimeError("untrusted bridge detail")


class _ThresholdUsageAdapter:
    """Returns a threshold flag with explicit, independently inspectable windows."""

    def __init__(self, snapshot, windows):
        self.snapshot, self.windows = snapshot, tuple(windows)

    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del gate, force
        quota.ingest_snapshot(self.snapshot, QuotaSource.PROVIDER_USAGE, "system",
                              now=self.snapshot.observed_at)
        return ForegroundRefreshStatus(
            True, True, False, True, "quota_archive_threshold_reached", None,
            observed, observed + timedelta(seconds=30), "provider_plan_units_not_currency",
            KimiCliUsageResult(self.snapshot, None, windows=self.windows))


class _WakeUsageAdapter:
    def __init__(self, snapshot, windows):
        self.snapshot, self.windows = snapshot, tuple(windows)

    def refresh_controller(self, quota):
        quota.ingest_snapshot(self.snapshot, QuotaSource.PROVIDER_USAGE, "system",
                              now=self.snapshot.observed_at)
        return KimiCliUsageResult(self.snapshot, None, windows=self.windows)


def _window(kind, remaining, reset_at):
    return ManagedUsageWindow(kind, 100, remaining, reset_at)


class _AcceptedThenIntervalThenFreshQuotaAdapter:
    """A real bridge's cadence shape: success, skipped interval, new success."""

    def __init__(self, next_refresh_at):
        self.next_refresh_at = next_refresh_at
        self.calls = 0

    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        self.calls += 1
        if self.calls == 2:
            return ForegroundRefreshStatus(
                False, False, False, False, "quota_refresh_interval_waiting", None,
                observed, self.next_refresh_at, "provider_plan_units_not_currency", None)
        return ForegroundRefreshStatus(
            True, True, True, False, "authoritative_quota_available", None,
            observed, self.next_refresh_at, "provider_plan_units_not_currency", None)


class _ResetBoundaryThenFreshQuotaAdapter:
    """A reset boundary requires a new provider observation before work resumes."""

    def __init__(self, retry_at):
        self.retry_at = retry_at
        self.calls = 0

    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del gate, force
        self.calls += 1
        if self.calls == 1:
            # The quota controller still exposes the accepted pre-reset
            # snapshot, but that snapshot cannot authorize a post-reset slice.
            return ForegroundRefreshStatus(
                True, False, False, False, "quota_reset_requires_fresh_telemetry", None,
                observed, self.retry_at, "unknown_provider_plan_cost", None,
                next_retry_at=self.retry_at)
        fresh = QuotaSnapshot(100, 90, observed + timedelta(hours=1), observed,
                              1.0, False)
        assert quota.ingest_snapshot(fresh, QuotaSource.PROVIDER_USAGE, "system", now=observed)
        return ForegroundRefreshStatus(
            True, True, True, False, "authoritative_quota_available", None,
            observed, observed + timedelta(seconds=30),
            "provider_plan_units_not_currency", None)


class _BridgeProcess:
    """Minimal one-shot child seam for the real OAuth bridge test path."""

    def __init__(self):
        self.running = True

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.running = False

    def kill(self):
        self.running = False

    def wait(self, timeout=None):
        del timeout
        return 0


def _expedition_approval(agent, goal="public source comparison", seed="host-seed",
                          authorization_seconds=60, slice_seconds=30, max_calls=2,
                          issued_at=None):
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="user", payload={
            "content": agent.expedition_authorization_content(
                goal, seed, authorization_seconds, slice_seconds, max_calls),
            "channel": "expedition_authorization",
        }))
    # The authorization is a subsequent USER record, so preserve the ledger's
    # temporal lineage even when a test supplies a deterministic clock.
    minimum_issued = datetime.fromisoformat(approval.created_at)
    issued = max(issued_at or minimum_issued, minimum_issued)
    return agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.EXPEDITION_AUTHORIZATION,
        source_kind=SourceKind.USER, source_ref="user", created_at=issued.isoformat(), payload={
            "authorization_id": "expeditionauth_fixture", "approval_event_id": approval.event_id,
            "nonce": "nonce_fixture", "issued_at": issued.isoformat(),
            "expires_at": (issued + timedelta(seconds=authorization_seconds)).isoformat(),
            "goal_digest": __import__("hashlib").sha256(goal.encode("utf-8")).hexdigest(),
            "host_seed_digest": __import__("hashlib").sha256(seed.encode("utf-8")).hexdigest(),
            "max_calls_per_slice": max_calls, "slice_seconds": slice_seconds,
            "authorization_seconds": authorization_seconds, "profile": "public_web_only_v1",
            "version": "expedition_authorization_v1",
        }, parent_event_ids=(approval.event_id,)))


def _active_research(agent, tools):
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="user", payload={
            "content": "one bounded public read-only research profile", "channel": "text",
        }))
    profile = ResearchAutonomyProfile(
        "workspace", ResearchBudget(max_tool_calls=1, max_total_bytes=4096,
        max_response_bytes=1024, max_wall_ms=10_000, ttl_seconds=60), public_web_only=True)
    policy = UnattendedPolicy.user_issued(
        "Read public documentation only.", datetime.now(timezone.utc) + timedelta(seconds=60))
    agent.start_unattended_research(profile, policy, approval.event_id)
    assert tools.research_autonomy_status()["active"]


class ExpeditionRedTeamTests(unittest.TestCase):
    def test_same_user_authorization_cannot_be_replayed_after_stop(self):
        agent, _ = _agent()
        try:
            approval = _expedition_approval(agent)
            agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                   approval.event_id)
            self.assertEqual("stopped", agent.stop_expedition("user_stop")["state"])
            # Stopping consumes the one user authorization.  A new expedition
            # needs a newly recorded, exact USER approval rather than replaying
            # the old event ID.
            with self.assertRaises((RuntimeError, ValueError, PermissionError)):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       approval.event_id)
        finally:
            agent.event_store.close()

    def test_expired_authorization_window_runs_no_slice_and_mints_no_grant(self):
        agent, tools = _agent("expedition-expiry")
        try:
            approval = _expedition_approval(agent, authorization_seconds=1, slice_seconds=1)
            with patch.object(agent, "_sleep_now", return_value=datetime.now(timezone.utc) + timedelta(seconds=2)):
                with self.assertRaisesRegex(RuntimeError, "authorization has expired"):
                    agent.start_expedition("public source comparison", "host-seed", 1, 1, 2,
                                           approval.event_id)
            self.assertIsNone(tools.research_autonomy_status())
            self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                  if event.kind == EventKind.CAPABILITY_GRANTED])
        finally:
            agent.event_store.close()

    def test_delayed_start_keeps_the_original_absolute_authorization_window(self):
        """Starting near expiry must not move the scheduler's wall-clock anchor.

        A foreground caller is allowed to arm the run one second before its
        already-issued authorization expires.  Once the next slice begins
        after the recorded expiry, it must stop before quota refresh, planning,
        or capability-grant creation.
        """
        agent, tools = _agent("expedition-delayed-start")
        try:
            authorization = _expedition_approval(
                agent, authorization_seconds=60, slice_seconds=30)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": issued + timedelta(seconds=59)}
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                clock["now"] = issued + timedelta(seconds=61)
                status = agent.expedition_slice(_FreshQuotaAdapter())
            self.assertEqual("completed", status["state"])
            self.assertEqual("authorization_window_elapsed", status["stop_reason"])
            self.assertIsNone(tools.research_autonomy_status())
            self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                  if event.kind == EventKind.CAPABILITY_GRANTED])
        finally:
            agent.event_store.close()

    def test_fresh_transient_waits_with_zero_k3_or_grants_then_fresh_retry_continues(self):
        """A cached quota observation may schedule retry, never run a slice itself."""
        agent, tools = _agent("expedition-quota-retry")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            # The fixture seeds a controller snapshot while constructing the
            # agent.  Make the bridge observation strictly newer than it,
            # while remaining inside the authorization's absolute window.
            clock = {"now": max(issued + timedelta(seconds=1),
                                  datetime.now(timezone.utc) + timedelta(seconds=1))}
            retry_at = clock["now"] + timedelta(seconds=5)
            adapter = _TransientThenFreshQuotaAdapter(retry_at)
            k3_calls = []
            agent.runtime.plan_tools = lambda prompt, context: (
                k3_calls.append((prompt, context)) or
                (ToolIntent("retry_done", "respond", {"message": "bounded complete"}, "done"),))
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                pending = agent.expedition_slice(adapter)
                self.assertEqual("waiting_quota_retry", pending["state"])
                self.assertEqual(1, pending["quota_retry"]["attempts"])
                self.assertEqual(retry_at.isoformat(), pending["quota_retry"]["retry_at"])
                self.assertEqual("quota_refresh_transient_retry_pending",
                                 pending["quota_retry"]["reason"])
                self.assertEqual(0, len(k3_calls))
                self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                      if event.kind == EventKind.CAPABILITY_GRANTED])
                # Polling before the bounded retry deadline must not make a
                # second bridge attempt or materialize any model/tool work.
                clock["now"] = retry_at - timedelta(seconds=1)
                early = agent.expedition_slice(adapter)
                self.assertEqual("waiting_quota_retry", early["state"])
                self.assertEqual(1, adapter.calls)
                self.assertEqual(0, len(k3_calls))
                self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                      if event.kind == EventKind.CAPABILITY_GRANTED])
                # Only a new accepted authoritative snapshot at retry time
                # lets this foreground loop resume planning.
                clock["now"] = retry_at
                resumed = agent.expedition_slice(adapter)
            self.assertNotEqual("waiting_quota_retry", resumed["state"])
            self.assertEqual(2, adapter.calls)
            self.assertEqual(1, len(k3_calls))
        finally:
            agent.event_store.close()

    def test_unknown_refresh_waits_with_active_coordinator_and_no_k3_or_grant(self):
        coordinator = SleepWakeCoordinator()
        agent, tools = _agent("expedition-unknown-waits", coordinator)
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1), datetime.now(timezone.utc) + timedelta(seconds=1))}
            k3_calls = []
            agent.runtime.plan_tools = lambda prompt, context: k3_calls.append((prompt, context)) or ()
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                status = agent.expedition_slice(_UnknownQuotaAdapter())
            self.assertEqual("waiting_quota_retry", status["state"])
            self.assertEqual(SleepState.ACTIVE, coordinator.state)
            self.assertEqual([], k3_calls)
            self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                  if event.kind == EventKind.CAPABILITY_GRANTED])
        finally:
            agent.event_store.close()

    def test_bridge_exception_waits_without_exposing_detail_or_granting_work(self):
        agent, tools = _agent("expedition-bridge-exception")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1), datetime.now(timezone.utc) + timedelta(seconds=1))}
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                status = agent.expedition_slice(_BridgeExceptionAdapter())
            self.assertEqual("waiting_quota_retry", status["state"])
            self.assertNotIn("untrusted bridge detail", repr(status))
            self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                  if event.kind == EventKind.CAPABILITY_GRANTED])
        finally:
            agent.event_store.close()

    def test_permanent_authorization_refresh_error_stops_immediately(self):
        agent, tools = _agent("expedition-permanent-auth")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1), datetime.now(timezone.utc) + timedelta(seconds=1))}
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                status = agent.expedition_slice(_PermanentAuthQuotaAdapter())
            self.assertEqual("stopped", status["state"])
            self.assertEqual("quota_refresh_permanent_authorization_error", status["stop_reason"])
            self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                  if event.kind == EventKind.CAPABILITY_GRANTED])
        finally:
            agent.event_store.close()

    def test_permanent_schema_refresh_error_stops_without_retry_or_grant(self):
        agent, tools = _agent("expedition-permanent-schema")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1), datetime.now(timezone.utc) + timedelta(seconds=1))}
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                status = agent.expedition_slice(_PermanentSchemaQuotaAdapter())
            self.assertEqual("stopped", status["state"])
            self.assertEqual("quota_refresh_permanent_schema_error", status["stop_reason"])
            self.assertEqual(0, status["quota_retry"]["attempts"])
            self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                  if event.kind == EventKind.CAPABILITY_GRANTED])
        finally:
            agent.event_store.close()

    def test_repeated_unknown_refreshes_stop_without_archive_k3_or_grants(self):
        coordinator = SleepWakeCoordinator()
        agent, tools = _agent("expedition-unknown-exhaustion", coordinator)
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1), datetime.now(timezone.utc) + timedelta(seconds=1))}
            adapter = _UnknownQuotaAdapter()
            k3_calls = []
            agent.runtime.plan_tools = lambda prompt, context: k3_calls.append((prompt, context)) or ()
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                for _ in range(4):
                    status = agent.expedition_slice(adapter)
                    retry = status["quota_retry"]["retry_at"]
                    if retry is not None:
                        clock["now"] = datetime.fromisoformat(retry)
            self.assertEqual("stopped", status["state"])
            self.assertEqual("quota_refresh_retry_exhausted", status["stop_reason"])
            self.assertEqual(SleepState.ACTIVE, coordinator.state)
            self.assertEqual([], k3_calls)
            kinds = [event.kind for event in agent.event_store.list(agent.session_id)]
            self.assertNotIn(EventKind.SLEEP_ARCHIVE, kinds)
            self.assertNotIn(EventKind.CAPABILITY_GRANTED, kinds)
        finally:
            agent.event_store.close()

    def test_weekly_primary_with_low_rolling_window_formally_archives_before_sleep(self):
        clock = {"now": datetime.now(timezone.utc) + timedelta(seconds=2)}
        coordinator = SleepWakeCoordinator(clock=lambda: clock["now"])
        agent, tools = _agent("expedition-weekly-primary", coordinator)
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock["now"] = max(clock["now"], issued + timedelta(seconds=1))
            observed = clock["now"]
            reset = observed + timedelta(hours=1)
            payload = {
                "usage": {"limit": 100, "used": 95, "resetAt": reset.isoformat()},
                "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                            "limit": 100, "used": 90, "resetAt": reset.isoformat()}],
            }
            snapshot = normalize_managed_usage(payload, observed)
            windows = normalize_managed_usage_windows(payload, observed)
            self.assertEqual("weekly", snapshot.primary_window_kind)
            adapter = _ThresholdUsageAdapter(snapshot, windows)
            with patch.object(agent, "_sleep_now", side_effect=lambda: observed):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                status = agent.expedition_slice(adapter)
            self.assertEqual("sleeping", status["state"])
            self.assertEqual(SleepState.SLEEPING, coordinator.state)
            events = agent.event_store.list(agent.session_id)
            archive = next(event for event in events if event.kind == EventKind.SLEEP_ARCHIVE)
            self.assertEqual("rolling_5h", archive.payload["quota_window_kind"])
            self.assertIn(EventKind.SLEEP_ENTERED, [event.kind for event in events])
        finally:
            agent.event_store.close()

    def test_rolling_primary_threshold_archives_then_wakes_with_v2_evidence(self):
        clock = {"now": datetime.now(timezone.utc) + timedelta(seconds=2)}
        coordinator = SleepWakeCoordinator(clock=lambda: clock["now"])
        agent, tools = _agent("expedition-rolling-sleep-wake", coordinator)
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock["now"] = max(clock["now"], issued + timedelta(seconds=1))
            # The fixture starts with a legacy bootstrap snapshot; keep this
            # managed window's reset later so quota monotonicity accepts the
            # new explicit primary identity.
            reset = clock["now"] + timedelta(hours=2)
            snapshot = QuotaSnapshot(100, 5, reset, clock["now"], 1.0, False,
                                     primary_window_kind="rolling_5h")
            adapter = _ThresholdUsageAdapter(snapshot, (_window("rolling_5h", 5, reset),))
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.set_auto_wake(True)
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                sleeping = agent.expedition_slice(adapter)
            self.assertEqual("sleeping", sleeping["state"])
            self.assertEqual(SleepState.SLEEPING, coordinator.state)
            clock["now"] = reset + timedelta(seconds=1)
            fresh_reset = clock["now"] + timedelta(hours=1)
            fresh = QuotaSnapshot(100, 100, fresh_reset, clock["now"], 1.0, False,
                                  primary_window_kind="rolling_5h")
            self.assertTrue(agent.poll_sleep(_WakeUsageAdapter(
                fresh, (_window("rolling_5h", 100, fresh_reset),))))
            self.assertEqual(SleepState.ACTIVE, coordinator.state)
            self.assertFalse(agent.poll_sleep(_WakeUsageAdapter(
                fresh, (_window("rolling_5h", 100, fresh_reset),))))
            events = agent.event_store.list(agent.session_id)
            archive = next(event for event in events if event.kind == EventKind.SLEEP_ARCHIVE)
            evidence = next(event for event in events if event.kind == EventKind.PROVIDER_USAGE_EVIDENCE)
            self.assertEqual("sleep_archive_v3", archive.payload["archive_version"])
            self.assertEqual("rolling_5h", archive.payload["quota_window_kind"])
            self.assertEqual("provider_usage_v2", evidence.payload["evidence_version"])
            self.assertEqual("rolling_5h", evidence.payload["window_kind"])
            ready = next(event for event in events if event.kind == EventKind.WAKE_READY)
            self.assertEqual("sleep_wake_v2", ready.payload["protocol_version"])
        finally:
            agent.event_store.close()

    def test_reset_boundary_waits_for_fresh_telemetry_then_resumes_without_sleeping(self):
        """An accepted pre-reset snapshot never authorizes post-reset work.

        The provider balance is deliberately still inspectable at the reset
        boundary.  It must be treated as non-authoritative until a newer
        provider observation arrives at the bounded foreground retry.
        """
        agent, _ = _agent("expedition-reset-fresh-telemetry")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1),
                                  datetime.now(timezone.utc) + timedelta(seconds=1))}
            reset_at = clock["now"] + timedelta(seconds=1)
            # Replace the generic fixture's future-window measurement so the
            # accepted provider record below is the only quota epoch under
            # test.
            agent.quota_controller = QuotaController()
            pre_reset = QuotaSnapshot(100, 0, reset_at,
                                      clock["now"] - timedelta(seconds=1), 1.0, False)
            self.assertTrue(agent.quota_controller.ingest_snapshot(
                pre_reset, QuotaSource.PROVIDER_USAGE, "system", now=clock["now"]))
            retry_at = reset_at + timedelta(seconds=5)
            adapter = _ResetBoundaryThenFreshQuotaAdapter(retry_at)
            k3_calls = []
            agent.runtime.plan_tools = lambda prompt, context: (
                k3_calls.append((prompt, context)) or
                (ToolIntent("post_reset_done", "respond", {"message": "bounded complete"}, "done"),))

            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                clock["now"] = reset_at
                waiting = agent.expedition_slice(adapter)
                self.assertEqual("waiting_quota_retry", waiting["state"], waiting)
                self.assertEqual("quota_reset_requires_fresh_telemetry",
                                 waiting["quota_retry"]["reason"])
                self.assertEqual(retry_at.isoformat(), waiting["quota_retry"]["retry_at"])
                self.assertEqual(0, len(k3_calls))
                self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                                      if event.kind == EventKind.CAPABILITY_GRANTED])

                clock["now"] = retry_at
                resumed = agent.expedition_slice(adapter)
            self.assertNotEqual("waiting_quota_retry", resumed["state"])
            self.assertNotEqual("sleeping", resumed["state"], resumed)
            self.assertEqual(2, adapter.calls)
            self.assertEqual(1, len(k3_calls))
            self.assertEqual(3, len([event for event in agent.event_store.list(agent.session_id)
                                     if event.kind == EventKind.CAPABILITY_GRANTED]))
        finally:
            agent.event_store.close()

    def test_interval_not_due_waits_without_reusing_cached_quota_for_a_new_slice(self):
        """An accepted refresh never authorizes a second immediate K3 slice."""
        agent, _ = _agent("expedition-interval-wait")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1),
                                  datetime.now(timezone.utc) + timedelta(seconds=1))}
            next_refresh_at = clock["now"] + timedelta(seconds=5)
            adapter = _AcceptedThenIntervalThenFreshQuotaAdapter(next_refresh_at)
            k3_calls = []
            agent.runtime.plan_tools = lambda prompt, context: (
                k3_calls.append((prompt, context)) or
                (ToolIntent("interval_done", "respond", {"message": "bounded complete"}, "done"),))
            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                agent.expedition_slice(adapter)
                grants_after_first = [event for event in agent.event_store.list(agent.session_id)
                                      if event.kind == EventKind.CAPABILITY_GRANTED]
                self.assertEqual(1, len(k3_calls))
                self.assertEqual(3, len(grants_after_first))

                clock["now"] = next_refresh_at - timedelta(seconds=1)
                waiting = agent.expedition_slice(adapter)
                self.assertEqual("waiting_quota_retry", waiting["state"], waiting)
                self.assertEqual("quota_refresh_interval_waiting", waiting["quota_retry"]["reason"])
                self.assertEqual(next_refresh_at.isoformat(), waiting["quota_retry"]["retry_at"])
                self.assertEqual(2, adapter.calls)
                self.assertEqual(1, len(k3_calls))
                self.assertEqual(3, len([event for event in agent.event_store.list(agent.session_id)
                                         if event.kind == EventKind.CAPABILITY_GRANTED]))

                clock["now"] = next_refresh_at
                resumed = agent.expedition_slice(adapter)
            self.assertNotEqual("waiting_quota_retry", resumed["state"])
            self.assertEqual(3, adapter.calls)
            self.assertEqual(2, len(k3_calls))
            self.assertEqual(6, len([event for event in agent.event_store.list(agent.session_id)
                                     if event.kind == EventKind.CAPABILITY_GRANTED]))
        finally:
            agent.event_store.close()

    def test_real_bridge_cadence_skip_waits_then_fresh_provider_refresh_resumes(self):
        """The production bridge's interval result cannot authorize cached work."""
        agent, _ = _agent("expedition-real-bridge-cadence")
        try:
            authorization = _expedition_approval(agent)
            issued = datetime.fromisoformat(authorization.payload["issued_at"])
            clock = {"now": max(issued + timedelta(seconds=1),
                                  datetime.now(timezone.utc) + timedelta(seconds=1))}

            def usage_reply(used):
                # The agent fixture has a pre-existing provider measurement;
                # this bridge observation must establish a non-regressing
                # provider window before cadence behavior is relevant.
                reset = clock["now"] + timedelta(hours=2)
                return {"usage": {"limit": 100, "used": used, "resetAt": reset.isoformat()},
                        "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                                    "detail": {"limit": 100, "remaining": 90,
                                               "reset_at": reset.isoformat()}}]}

            responses = [BridgeHttpResponse(200, usage_reply(5)),
                         BridgeHttpResponse(200, usage_reply(6))]
            bridge = KimiCliOAuthUsageBridge(
                process_factory=lambda args: _BridgeProcess(),
                readiness_reader=lambda process, timeout: BridgeReadiness(45687, "ephemeral-token"),
                transport=lambda *args: responses.pop(0))
            bridge_statuses = []
            bridge_refresh = bridge.refresh_foreground_slice
            def tracked_refresh(*args, **kwargs):
                status = bridge_refresh(*args, **kwargs)
                bridge_statuses.append(status)
                return status
            bridge.refresh_foreground_slice = tracked_refresh
            k3_calls = []
            agent.runtime.plan_tools = lambda prompt, context: (
                k3_calls.append((prompt, context)) or
                (ToolIntent("real_bridge_done", "respond", {"message": "bounded complete"}, "done"),))

            with patch.object(agent, "_sleep_now", side_effect=lambda: clock["now"]):
                agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)
                first = agent.expedition_slice(bridge)
                self.assertNotEqual("waiting_quota_retry", first["state"])
                first_grants = [event for event in agent.event_store.list(agent.session_id)
                                if event.kind == EventKind.CAPABILITY_GRANTED]
                self.assertEqual((1, 3), (len(k3_calls), len(first_grants)),
                                 (first, bridge_statuses))

                clock["now"] += timedelta(seconds=1)
                waiting = agent.expedition_slice(bridge)
                self.assertEqual("waiting_quota_retry", waiting["state"])
                self.assertEqual("quota_refresh_interval_waiting", waiting["quota_retry"]["reason"])
                retry_at = datetime.fromisoformat(waiting["quota_retry"]["retry_at"])
                self.assertEqual(1, len(k3_calls))
                self.assertEqual(3, len([event for event in agent.event_store.list(agent.session_id)
                                         if event.kind == EventKind.CAPABILITY_GRANTED]))
                self.assertEqual(1, len(responses))

                clock["now"] = retry_at
                resumed = agent.expedition_slice(bridge)
            self.assertNotEqual("waiting_quota_retry", resumed["state"])
            self.assertEqual(2, len(k3_calls))
            self.assertEqual(6, len([event for event in agent.event_store.list(agent.session_id)
                                     if event.kind == EventKind.CAPABILITY_GRANTED]))
            self.assertEqual([], responses)
        finally:
            agent.event_store.close()

    def test_repl_run_keyboard_interrupt_stops_expedition_and_active_grants(self):
        agent, tools = _agent("expedition-repl-interrupt")
        try:
            approval = _expedition_approval(agent)
            agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                   approval.event_id)
            _active_research(agent, tools)
            output = io.StringIO()
            with patch.object(agent, "run_expedition_foreground", side_effect=KeyboardInterrupt):
                with redirect_stdout(output):
                    run_repl(agent, ["/expedition run", "/quit"])
            self.assertEqual("stopped", agent.expedition_status()["state"])
            self.assertFalse(tools.research_autonomy_status()["active"])
            self.assertIn("user_stop", output.getvalue())
        finally:
            agent.event_store.close()

    def test_cross_session_authorization_and_concurrent_consume_fail_closed(self):
        # Separate SQLite connections model independent foreground callers;
        # sharing an in-memory connection across threads would only test
        # sqlite's thread-affinity guard, not authorization consumption.
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.sqlite")
            owner_store = SQLiteEventStore(path, session_id="expedition-owner")
            agent, _ = _agent("expedition-owner")
            agent.event_store.close()
            agent.event_store = owner_store
            authorization = _expedition_approval(agent)
            other_store = SQLiteEventStore(path, session_id="expedition-other")
            other = StrangeloopAgent(session_id="expedition-other", event_store=other_store,
                                     runtime=agent.runtime, quota_controller=agent.quota_controller,
                                     tool_session=ToolSession("expedition-other", "workspace",
                                         CapabilityRegistry(), ControlledToolExecutor(os.getcwd()),
                                         event_store=other_store, web_fetch=PublicWebFetch(()),
                                         web_search=SafePublicWebSearch(PublicWebFetch(())),
                                         browser_read=StatelessBrowserRead(PublicWebFetch(()))))
            with self.assertRaises(ValueError):
                other.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                       authorization.event_id)

            outcomes = []
            gate = threading.Barrier(3)

            def start():
                gate.wait()
                store = SQLiteEventStore(path, session_id="expedition-owner")
                thread_agent = StrangeloopAgent(
                    session_id="expedition-owner", event_store=store, runtime=agent.runtime,
                    quota_controller=agent.quota_controller, tool_session=ToolSession(
                        "expedition-owner", "workspace", CapabilityRegistry(),
                        ControlledToolExecutor(os.getcwd()), event_store=store,
                        web_fetch=PublicWebFetch(()), web_search=SafePublicWebSearch(PublicWebFetch(())),
                        browser_read=StatelessBrowserRead(PublicWebFetch(()))))
                try:
                    thread_agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                                  authorization.event_id)
                    outcomes.append("started")
                except (RuntimeError, ValueError, PermissionError):
                    outcomes.append("refused")
                finally:
                    store.close()

            threads = [threading.Thread(target=start), threading.Thread(target=start)]
            for item in threads: item.start()
            gate.wait()
            for item in threads: item.join()
            self.assertEqual(["refused", "started"], sorted(outcomes))
            consumed = [event for event in owner_store.list(agent.session_id)
                        if event.kind == EventKind.EXPEDITION_AUTHORIZATION_CONSUMED]
            self.assertEqual(1, len(consumed))
            other_store.close()
            owner_store.close()

    def test_wake_resume_keyboard_interrupt_stops_expedition_and_active_grants(self):
        agent, tools = _agent("expedition-wake-interrupt")
        try:
            approval = _expedition_approval(agent)
            agent.start_expedition("public source comparison", "host-seed", 60, 30, 2,
                                   approval.event_id)
            _active_research(agent, tools)
            output = io.StringIO()
            with patch.object(agent, "run_expedition_foreground", side_effect=KeyboardInterrupt):
                with redirect_stdout(output):
                    _resume_expedition_after_wake(agent, _FreshQuotaAdapter())
            self.assertEqual("stopped", agent.expedition_status()["state"])
            self.assertFalse(tools.research_autonomy_status()["active"])
            self.assertIn("user_stop", output.getvalue())
        finally:
            agent.event_store.close()

    def test_provider_keyboard_interrupt_releases_quota_reservation(self):
        quota = QuotaController()
        now = datetime.now(timezone.utc)
        quota.ingest_snapshot(QuotaSnapshot(100, 100, now + timedelta(hours=1), now,
                                            1.0, False), QuotaSource.PROVIDER_USAGE,
                              "system", now=now)

        def interrupted(url, headers, payload, timeout):
            del url, headers, payload, timeout
            raise KeyboardInterrupt

        runtime = KimiCodeRuntime(KimiCodeSettings(), CallableSecretResolver(lambda: "fixture-key"),
                                  interrupted, quota_controller=quota)
        with self.assertRaises(KeyboardInterrupt):
            runtime.complete_deliberation("bounded public prompt",
                WorkspaceFrame("turn", ("evt_1",), (), (), (), ()))
        self.assertEqual(0, quota.export_telemetry(now)["reserved_calls"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
