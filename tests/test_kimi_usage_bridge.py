from datetime import datetime, timedelta, timezone
import io
import unittest

from strangeloop.kimi_usage_bridge import (BridgeHttpResponse,
                                           BridgeReadiness,
                                           KimiCliOAuthUsageBridge,
                                           _usage_url, _validate_usage_url,
                                           default_readiness_reader)
from strangeloop.quota import ForegroundRefreshGate, ForegroundRefreshPolicy, QuotaController
from strangeloop.engine import StrangeloopAgent
from strangeloop.sleep import SleepWakeCoordinator


NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def payload():
    return {
        "usage": {"limit": 100, "used": 20, "resetAt": "2026-08-19T12:00:00Z"},
        "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                    "detail": {"limit": 10, "remaining": 1,
                               "reset_at": "2026-08-12T14:00:00Z"}}],
    }


def loopback_payload():
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "kind": "ok",
            "summary": {"label": "Weekly limit", "used": 1, "limit": 100,
                        "reset_hint": "resets in 1d 16h 6m"},
            "limits": [{"label": "5h limit", "used": 4, "limit": 100,
                        "reset_hint": "resets in 3h 6m"}],
            "extra_usage": None,
        },
        "request_id": "opaque-request-id",
    }


class FakeProcess:
    def __init__(self, output=("Local: http://127.0.0.1:45678/#token=temporary-bridge-bearer\n"
                               "Token: temporary-bridge-bearer\n")):
        self.stdout = io.StringIO(output)
        self.stderr = None
        self.terminated = False
        self.killed = False
        self.waited = False
        self.running = True

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False

    def wait(self, timeout=None):
        self.waited = True
        return 0


class KimiUsageBridgeTests(unittest.TestCase):
    def _bridge_for_payload(self, response_payload):
        return KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45682, "ephemeral-token"),
            transport=lambda *args: BridgeHttpResponse(200, response_payload))

    def test_one_fetch_uses_fixed_loopback_route_and_stops_process(self):
        process = FakeProcess()
        commands, calls = [], []

        def launch(args):
            commands.append(tuple(args))
            return process

        def ready(child, timeout):
            self.assertIs(process, child)
            self.assertEqual(3.0, timeout)
            return BridgeReadiness(45678, "temporary-bridge-bearer")

        def transport(url, token, timeout):
            calls.append((url, token, timeout))
            return BridgeHttpResponse(200, payload())

        result = KimiCliOAuthUsageBridge(process_factory=launch, readiness_reader=ready,
                                         transport=transport,
                                         timeout_seconds=3).fetch(NOW)
        self.assertTrue(result.known)
        self.assertEqual(("kimi", "web", "--no-open", "--host", "127.0.0.1", "--port", "0"),
                         commands[0])
        self.assertEqual("http://127.0.0.1:45678/api/v1/oauth/usage?provider=managed%3Akimi-code",
                         calls[0][0])
        self.assertEqual(1, len(calls))
        self.assertTrue(process.terminated)
        self.assertNotIn("temporary-bridge-bearer", repr(result))

    def test_start_failure_and_login_requirement_are_stable_and_redacted(self):
        started = FakeProcess()
        fail = KimiCliOAuthUsageBridge(process_factory=lambda args: None).fetch(NOW)
        self.assertEqual("bridge_start_failed", fail.error_category)
        unauth = KimiCliOAuthUsageBridge(process_factory=lambda args: started,
                                         readiness_reader=lambda p, t: BridgeReadiness(45679, "sensitive-token"),
                                         transport=lambda *args: BridgeHttpResponse(401, {"detail": "secret"}),
                                         ).fetch(NOW)
        self.assertEqual("oauth_login_required", unauth.error_category)
        self.assertTrue(started.terminated)
        self.assertNotIn("sensitive", repr(unauth).lower())
        self.assertNotIn("secret", repr(unauth).lower())

    def test_non_loopback_or_altered_endpoint_is_rejected_before_transport(self):
        for url in ("http://localhost:41234/api/v1/oauth/usage?provider=managed%3Akimi-code",
                    "http://127.0.0.1:41234/other?provider=managed%3Akimi-code",
                    "https://127.0.0.1:41234/api/v1/oauth/usage?provider=managed%3Akimi-code",
                    "http://127.0.0.1:41234/api/v1/oauth/usage?provider=other"):
            with self.assertRaises(ValueError):
                _validate_usage_url(url)
        self.assertEqual("http://127.0.0.1:1/api/v1/oauth/usage?provider=managed%3Akimi-code", _usage_url(1))

    def test_readiness_failure_still_cleans_up_and_does_not_leak_output(self):
        process = FakeProcess("Local: http://127.0.0.1:45680/#token=never-report-this\n")
        result = KimiCliOAuthUsageBridge(
            process_factory=lambda args: process,
            readiness_reader=lambda p, t: (_ for _ in ()).throw(Exception("never-report-this")),
            ).fetch(NOW)
        self.assertEqual("bridge_start_failed", result.error_category)
        self.assertTrue(process.terminated)
        self.assertNotIn("never-report-this", repr(result))

    def test_refresh_controller_only_ingests_redacted_normalized_snapshot(self):
        controller = QuotaController()
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45681, "ephemeral-token"),
            transport=lambda *args: BridgeHttpResponse(200, payload()))
        result = bridge.refresh_controller(controller, NOW)
        self.assertTrue(result.known)
        telemetry = controller.export_telemetry()
        self.assertEqual("provider_usage", telemetry["source"])
        self.assertNotIn("ephemeral-token", repr(telemetry))

    def test_controller_canonical_reset_is_returned_for_bridge_and_engine_usage_update(self):
        first = payload()
        first["usage"] = {"limit": 100, "used": 1,
                          "resetAt": (NOW + timedelta(days=1)).isoformat()}
        first["limits"][0]["detail"] = {"limit": 100, "remaining": 94,
                                         "reset_at": (NOW + timedelta(hours=1, seconds=7)).isoformat()}
        second = payload()
        second["usage"] = {"limit": 100, "used": 1,
                           "resetAt": (NOW + timedelta(days=1)).isoformat()}
        second["limits"][0]["detail"] = {"limit": 100, "remaining": 93,
                                          "reset_at": (NOW + timedelta(hours=1)).isoformat()}
        replies = [first, second]
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45687, "ephemeral-token"),
            transport=lambda *args: BridgeHttpResponse(200, replies.pop(0)))
        controller = QuotaController()
        initial = bridge.refresh_controller(controller, NOW)
        canonical = bridge.refresh_controller(controller, NOW + timedelta(seconds=30))
        self.assertTrue(initial.known)
        self.assertTrue(canonical.known)
        self.assertEqual(initial.snapshot.reset_at, canonical.snapshot.reset_at)
        rolling = next(item for item in canonical.windows if item.kind == "rolling_5h")
        self.assertEqual(initial.snapshot.reset_at, rolling.reset_at)
        agent = StrangeloopAgent(session_id="canonical-reset", quota_controller=controller,
                                 sleep_coordinator=SleepWakeCoordinator())
        self.assertFalse(agent.update_managed_usage(canonical))
        accepted_rolling = next(item for item in agent._managed_usage_windows if item.kind == "rolling_5h")
        self.assertEqual(initial.snapshot.reset_at, accepted_rolling.reset_at)

    def test_foreground_primary_window_switch_keeps_each_window_reset_identity(self):
        """A weekly-primary sample cannot canonicalize the rolling window.

        These are two complete managed-usage provider payloads.  The first
        has the weekly window tighter; the second is a tie resolved toward
        rolling_5h by its earlier reset.  Both are fresh provider observations
        and therefore must be accepted as separate primary-window epochs.
        """
        first_observed = NOW
        second_observed = NOW + timedelta(seconds=2)
        weekly_reset = NOW + timedelta(days=1)
        rolling_reset = NOW + timedelta(hours=2)

        def provider_payload(weekly_remaining, rolling_remaining):
            return {
                "usage": {"limit": 100, "used": 100 - weekly_remaining,
                          "resetAt": weekly_reset.isoformat()},
                "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                            "detail": {"limit": 100, "remaining": rolling_remaining,
                                       "reset_at": rolling_reset.isoformat()}}],
            }

        # Weekly is tighter in the first reading; then both are equally
        # tight and rolling_5h wins the normalizer's earlier-reset tie-break.
        responses = [BridgeHttpResponse(200, provider_payload(20, 21)),
                     BridgeHttpResponse(200, provider_payload(20, 20))]
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45688, "ephemeral-token"),
            transport=lambda *args: responses.pop(0))
        controller = QuotaController()
        gate = ForegroundRefreshGate(ForegroundRefreshPolicy(minimum_interval=timedelta(seconds=1)))

        first = bridge.refresh_foreground_slice(controller, gate, first_observed)
        self.assertEqual((True, True, True),
                         (first.attempted, first.accepted, first.automatic_allowed))
        self.assertEqual("weekly", first.usage_result.snapshot.primary_window_kind)
        first_windows = {window.kind: window for window in first.usage_result.windows}
        self.assertEqual(weekly_reset, first_windows["weekly"].reset_at)
        self.assertEqual(rolling_reset, first_windows["rolling_5h"].reset_at)
        self.assertEqual("weekly", controller.export_telemetry(first_observed)["snapshot"]
                         ["primary_window_kind"])

        second = bridge.refresh_foreground_slice(controller, gate, second_observed)
        self.assertEqual((True, True, True),
                         (second.attempted, second.accepted, second.automatic_allowed))
        self.assertEqual("rolling_5h", second.usage_result.snapshot.primary_window_kind)
        second_windows = {window.kind: window for window in second.usage_result.windows}
        self.assertEqual(weekly_reset, second_windows["weekly"].reset_at)
        self.assertEqual(rolling_reset, second_windows["rolling_5h"].reset_at)
        telemetry = controller.export_telemetry(second_observed)["snapshot"]
        self.assertEqual("rolling_5h", telemetry["primary_window_kind"])
        self.assertEqual(rolling_reset.isoformat(), telemetry["reset_at"])

        # The controller, returned primary snapshot, and matching window
        # describe exactly the same primary quota epoch in each refresh.
        self.assertEqual(second.usage_result.snapshot.reset_at,
                         second_windows[second.usage_result.snapshot.primary_window_kind].reset_at)

    def test_foreground_slice_refresh_is_synchronous_rate_limited_and_flags_archive(self):
        calls = []
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45684, "ephemeral-token"),
            transport=lambda *args: (calls.append(args) or BridgeHttpResponse(200, payload())))
        controller = QuotaController()
        gate = ForegroundRefreshGate(ForegroundRefreshPolicy(minimum_interval=timedelta(seconds=30),
                                                              archive_threshold=.10))
        first = bridge.refresh_foreground_slice(controller, gate, NOW)
        self.assertEqual((True, True, True, True),
                         (first.attempted, first.accepted, first.automatic_allowed,
                          first.archive_threshold_reached))
        self.assertEqual("quota_archive_threshold_reached", first.reason)
        self.assertEqual("provider_plan_units_not_currency", first.cost_status)
        self.assertIsNotNone(first.usage_result)
        second = bridge.refresh_foreground_slice(controller, gate, NOW + timedelta(seconds=1))
        self.assertFalse(second.attempted)
        self.assertFalse(second.automatic_allowed)
        self.assertFalse(second.archive_threshold_reached)
        self.assertEqual("quota_refresh_interval_waiting", second.reason)
        self.assertIsNone(second.usage_result)
        self.assertEqual(1, len(calls))

    def test_foreground_slice_unknown_bridge_is_fail_closed_without_leaking_error_body(self):
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45683, "ephemeral-token"),
            transport=lambda *args: BridgeHttpResponse(401, {"detail": "never-export"}))
        status = bridge.refresh_foreground_slice(QuotaController(), ForegroundRefreshGate(), NOW)
        self.assertEqual((True, False, False),
                         (status.attempted, status.accepted, status.automatic_allowed))
        self.assertEqual("quota_refresh_unknown_fail_closed", status.reason)
        self.assertEqual("oauth_login_required", status.error_category)
        self.assertEqual("unknown_provider_plan_cost", status.cost_status)
        self.assertNotIn("never-export", repr(status))

    def test_single_transient_after_fresh_evidence_waits_for_bounded_retry_then_recovers(self):
        """A one-off bridge failure is neither permission to run nor a permanent sleep.

        The last authoritative snapshot stays inspectable, but it must never
        become authority for another automatic slice.  The next foreground
        poll is bounded by the gate and a fresh successful bridge observation
        is required before automatic work can resume.
        """
        healthy = payload()
        healthy["limits"][0]["detail"]["remaining"] = 9
        calls = []
        replies = [BridgeHttpResponse(200, healthy),
                   BridgeHttpResponse(503, {"detail": "transient-only"}),
                   BridgeHttpResponse(200, healthy)]
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45685, "ephemeral-token"),
            transport=lambda *args: (calls.append(args) or replies.pop(0)))
        controller = QuotaController()
        gate = ForegroundRefreshGate(ForegroundRefreshPolicy(
            minimum_interval=timedelta(seconds=5), archive_threshold=.10))

        first = bridge.refresh_foreground_slice(controller, gate, NOW)
        self.assertTrue(first.automatic_allowed)
        transient_at = NOW + timedelta(seconds=5)
        transient = bridge.refresh_foreground_slice(controller, gate, transient_at)
        self.assertEqual((True, False, False),
                         (transient.attempted, transient.accepted, transient.automatic_allowed))
        self.assertEqual("quota_refresh_transient_retry_pending", transient.reason)
        self.assertEqual(transient_at + timedelta(seconds=5), transient.next_refresh_at)
        # Fresh evidence remains visible to the host's retry policy, but no
        # cached usage result is handed back as permission to execute.
        self.assertTrue(controller.authoritative_wake_evidence(transient_at).allow)
        self.assertIsNone(transient.usage_result)

        recovered = bridge.refresh_foreground_slice(controller, gate, transient.next_refresh_at)
        self.assertEqual((True, True, True),
                         (recovered.attempted, recovered.accepted, recovered.automatic_allowed))
        self.assertEqual("authoritative_quota_available", recovered.reason)
        self.assertEqual(3, len(calls))

    def test_newer_same_window_snapshot_tolerates_small_reset_clock_regression(self):
        """The bridge accepts fresh provider evidence without shortening a known window."""
        first_observed = datetime(2026, 8, 12, 12, 32, 37, tzinfo=timezone.utc)
        first_reset = datetime(2026, 8, 12, 13, 40, 37, tzinfo=timezone.utc)
        second_observed = datetime(2026, 8, 12, 12, 35, 54, tzinfo=timezone.utc)
        regressed_reset = datetime(2026, 8, 12, 13, 39, 54, tzinfo=timezone.utc)

        def provider_response(used, reset):
            return {"usage": {"limit": 100, "used": used, "resetAt": reset.isoformat()},
                    "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                                "detail": {"limit": 100, "remaining": 93,
                                           "reset_at": reset.isoformat()}}]}

        responses = [BridgeHttpResponse(200, provider_response(6, first_reset)),
                     BridgeHttpResponse(200, provider_response(7, regressed_reset))]
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45686, "ephemeral-token"),
            transport=lambda *args: responses.pop(0))
        controller = QuotaController()
        gate = ForegroundRefreshGate(ForegroundRefreshPolicy(minimum_interval=timedelta(seconds=1)))
        first = bridge.refresh_foreground_slice(controller, gate, first_observed)
        second = bridge.refresh_foreground_slice(controller, gate, second_observed)
        self.assertEqual((True, True, True), (first.accepted, second.accepted, second.automatic_allowed))
        measured = controller.export_telemetry(second_observed)["snapshot"]
        self.assertEqual(93, measured["remaining"])
        self.assertEqual(second_observed.isoformat(), measured["observed_at"])
        self.assertEqual(first_reset.isoformat(), measured["reset_at"])
        self.assertEqual("rolling_5h", measured["primary_window_kind"])

    def test_transient_refresh_uses_matching_cache_only_for_one_bounded_retry(self):
        mode = {"value": "success"}
        calls = []

        def transport(*args):
            calls.append(args)
            if mode["value"] == "success":
                return BridgeHttpResponse(200, payload())
            return BridgeHttpResponse(408, {"detail": "never-export"})

        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45685, "ephemeral-token"),
            transport=transport)
        controller = QuotaController()
        gate = ForegroundRefreshGate(ForegroundRefreshPolicy(
            minimum_interval=timedelta(seconds=30), retry_delay=timedelta(seconds=2),
            max_transient_retries=1))
        first = bridge.refresh_foreground_slice(controller, gate, NOW)
        self.assertTrue(first.automatic_allowed)
        mode["value"] = "timeout"
        failed = bridge.refresh_foreground_slice(controller, gate, NOW + timedelta(seconds=1), force=True)
        self.assertEqual("quota_refresh_transient_retry_pending", failed.reason)
        self.assertEqual((False, True, True),
                         (failed.automatic_allowed, failed.degraded, failed.cache_used))
        self.assertEqual(NOW + timedelta(seconds=3), failed.next_retry_at)
        self.assertIsNone(failed.usage_result)
        pending = bridge.refresh_foreground_slice(controller, gate, NOW + timedelta(seconds=2))
        self.assertEqual("quota_refresh_retry_pending", pending.reason)
        self.assertFalse(pending.automatic_allowed)
        self.assertFalse(pending.cache_used)
        exhausted = bridge.refresh_foreground_slice(controller, gate, NOW + timedelta(seconds=3))
        self.assertEqual("quota_refresh_unknown_fail_closed", exhausted.reason)
        self.assertFalse(exhausted.cache_used)
        self.assertEqual(3, len(calls))

    def test_auth_failure_clears_matching_cache_without_retry(self):
        mode = {"value": "success"}
        bridge = KimiCliOAuthUsageBridge(
            process_factory=lambda args: FakeProcess(),
            readiness_reader=lambda p, t: BridgeReadiness(45686, "ephemeral-token"),
            transport=lambda *args: (BridgeHttpResponse(200, payload()) if mode["value"] == "success"
                                      else BridgeHttpResponse(401, {"detail": "never-export"})))
        controller, gate = QuotaController(), ForegroundRefreshGate()
        self.assertTrue(bridge.refresh_foreground_slice(controller, gate, NOW).automatic_allowed)
        mode["value"] = "auth"
        denied = bridge.refresh_foreground_slice(controller, gate, NOW + timedelta(seconds=1), force=True)
        self.assertEqual("oauth_login_required", denied.error_category)
        self.assertFalse(denied.cache_used)
        self.assertFalse(denied.degraded)
        self.assertIsNotNone(denied.usage_result)

    def test_real_loopback_usage_shape_is_strictly_unwrapped_without_metadata(self):
        result = self._bridge_for_payload(loopback_payload()).fetch(NOW)
        self.assertTrue(result.known)
        self.assertEqual("kimi_cli_loopback_oauth_managed_usage", result.source)
        self.assertEqual({"weekly", "rolling_5h"}, {window.kind for window in result.windows})
        windows = {window.kind: window for window in result.windows}
        self.assertEqual(99, windows["weekly"].remaining)
        self.assertEqual(96, windows["rolling_5h"].remaining)
        self.assertEqual("2026-08-14T04:06:00+00:00", windows["weekly"].reset_at.isoformat())
        self.assertEqual("2026-08-12T15:06:00+00:00", windows["rolling_5h"].reset_at.isoformat())
        self.assertNotIn("opaque-request-id", repr(result))

    def test_loopback_usage_rejects_ambiguous_unknown_or_out_of_bounds_records(self):
        malformed = []

        bad_code = loopback_payload()
        bad_code["code"] = 1
        malformed.append(bad_code)

        duplicate_five_hour = loopback_payload()
        duplicate_five_hour["data"]["limits"].append(
            {"label": "5h limit", "used": 1, "limit": 100, "reset_hint": "resets in 1h"})
        malformed.append(duplicate_five_hour)

        unknown_field = loopback_payload()
        unknown_field["data"]["summary"]["unexpected"] = True
        malformed.append(unknown_field)

        noncanonical_hint = loopback_payload()
        noncanonical_hint["data"]["limits"][0]["reset_hint"] = "resets in 6m 3h"
        malformed.append(noncanonical_hint)

        out_of_bounds = loopback_payload()
        out_of_bounds["data"]["limits"][0]["reset_hint"] = "resets in 6h 1m"
        malformed.append(out_of_bounds)

        negative = loopback_payload()
        negative["data"]["summary"]["used"] = -1
        malformed.append(negative)

        for response_payload in malformed:
            with self.subTest(response_payload=response_payload):
                result = self._bridge_for_payload(response_payload).fetch(NOW)
                self.assertFalse(result.known)
                self.assertEqual("unrecognized_usage_payload", result.error_category)

    def test_default_reader_extracts_cli_assigned_port_and_matching_bearer(self):
        readiness = default_readiness_reader(FakeProcess(
            "Local: http://127.0.0.1:51234/#token=matching-token\n"
            "Token: matching-token\n"), 1)
        self.assertEqual(51234, readiness.port)
        self.assertNotIn("matching-token", repr(readiness))

    def test_default_reader_rejects_non_loopback_or_conflicting_advertisements(self):
        invalid_outputs = (
            "Local: http://localhost:51234/#token=secret-token\nToken: secret-token\n",
            "Local: http://127.0.0.1:51234/#token=secret-token\nToken: other-token\n",
            ("Local: http://127.0.0.1:51234/#token=secret-token\n"
             "Local: http://127.0.0.1:51235/#token=secret-token\nToken: secret-token\n"),
        )
        for output in invalid_outputs:
            with self.assertRaises(Exception):
                default_readiness_reader(FakeProcess(output), 1)

    def test_invalid_injected_readiness_is_rejected_and_process_is_cleaned_up(self):
        process = FakeProcess()
        result = KimiCliOAuthUsageBridge(
            process_factory=lambda args: process,
            readiness_reader=lambda p, t: BridgeReadiness(0, "not-reported"),
            transport=lambda *args: self.fail("transport must not run")).fetch(NOW)
        self.assertEqual("bridge_start_failed", result.error_category)
        self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
