from datetime import datetime, timedelta, timezone
import threading
import unittest

from strangeloop.quota import (QuotaController, QuotaPolicy, QuotaSnapshot,
                               ForegroundRefreshGate, ForegroundRefreshPolicy,
                               QuotaSource, ReservationInvalidationReason, UsageRecord,
                               quota_snapshot_from_usage_payload,
                               usage_record_from_provider_response)


NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def snapshot(remaining=80, total=100, calls=None, now=NOW):
    return QuotaSnapshot(total, remaining, now + timedelta(hours=1), now, 1.0, False,
                         call_total=calls, call_remaining=calls)


class QuotaControllerTests(unittest.TestCase):
    def test_foreground_refresh_gate_is_caller_driven_and_rate_limited(self):
        gate = ForegroundRefreshGate(ForegroundRefreshPolicy(minimum_interval=timedelta(seconds=30)))
        self.assertTrue(gate.refresh_due(NOW))
        gate.record_attempt(NOW)
        self.assertFalse(gate.refresh_due(NOW + timedelta(seconds=29)))
        self.assertTrue(gate.refresh_due(NOW + timedelta(seconds=30)))
        gate.record_success(NOW, True)
        self.assertTrue(gate.archive_threshold_reached())
        self.assertEqual(NOW + timedelta(seconds=30), gate.next_refresh_at())

    def test_foreground_refresh_policy_rejects_invalid_cadence_and_threshold(self):
        with self.assertRaises(ValueError):
            ForegroundRefreshPolicy(minimum_interval=timedelta(0))
        with self.assertRaises(ValueError):
            ForegroundRefreshPolicy(archive_threshold=1.1)

    def test_thresholds_are_deterministic_and_never_route_to_none(self):
        controller = QuotaController(QuotaPolicy(soft_threshold=.2, hard_threshold=.05,
                                                  normal_max_completion_tokens=100, normal_max_tool_steps=8))
        self.assertTrue(controller.ingest_snapshot(snapshot(20), QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        soft = controller.decision(NOW)
        self.assertEqual((True, "low", 50, 4, False),
                         (soft.allow_call, soft.reasoning_effort, soft.max_completion_tokens, soft.max_tool_steps, soft.must_pause_loop))
        controller.ingest_snapshot(snapshot(5), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        hard = controller.decision(NOW)
        self.assertEqual(("low", 25, 2), (hard.reasoning_effort, hard.max_completion_tokens, hard.max_tool_steps))
        self.assertTrue(controller.ingest_snapshot(snapshot(0), QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertEqual((False, True), (controller.decision(NOW).allow_call, controller.decision(NOW).must_pause_loop))

    def test_staleness_and_reset_need_fresh_telemetry(self):
        controller = QuotaController(QuotaPolicy(max_staleness=timedelta(minutes=5)))
        stale = QuotaSnapshot(100, 90, NOW + timedelta(hours=1), NOW - timedelta(minutes=6), 1, False)
        controller.ingest_snapshot(stale, QuotaSource.AUTHORITATIVE_HEADER, "system", now=NOW)
        self.assertEqual("quota_telemetry_stale", controller.decision(NOW).reason)
        controller = QuotaController(QuotaPolicy(max_staleness=timedelta(minutes=5)))
        expired = QuotaSnapshot(100, 0, NOW - timedelta(seconds=1), NOW, 1, False)
        controller.ingest_snapshot(expired, QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        self.assertEqual("quota_reset_requires_fresh_telemetry", controller.decision(NOW).reason)
        controller.ingest_snapshot(snapshot(100, 100), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        self.assertTrue(controller.decision(NOW).allow_call)

    def test_authoritative_snapshot_reconciles_local_ledger(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(100), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        reservation = controller.reserve_call(NOW)
        self.assertTrue(controller.commit(reservation, UsageRecord(prompt_tokens=20, completion_tokens=10, cached_tokens=5)))
        self.assertEqual(75, controller._effective()["remaining"])
        controller.ingest_snapshot(snapshot(70), QuotaSource.AUTHORITATIVE_HEADER, "system", now=NOW)
        self.assertEqual(70, controller._effective()["remaining"])
        self.assertEqual(0, controller.export_telemetry(NOW)["ledger_tokens"])

    def test_provider_primary_units_never_mix_with_local_token_ledger(self):
        controller = QuotaController()
        managed = QuotaSnapshot(100, 96, NOW + timedelta(hours=1), NOW, 1.0, False,
                                primary_unit="provider_units")
        self.assertTrue(controller.ingest_snapshot(managed, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        reservation = controller.reserve_call(NOW)
        self.assertIsNotNone(reservation)
        self.assertTrue(controller.commit(reservation, UsageRecord(prompt_tokens=3000)))
        self.assertEqual(96, controller._effective()["remaining"])
        self.assertTrue(controller.decision(NOW).allow_call)

    def test_explicit_token_dimension_still_tracks_local_token_usage(self):
        controller = QuotaController()
        managed = QuotaSnapshot(100, 96, NOW + timedelta(hours=1), NOW, 1.0, False,
                                token_total=100, token_remaining=40,
                                primary_unit="provider_units")
        self.assertTrue(controller.ingest_snapshot(managed, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        reservation = controller.reserve_call(NOW)
        self.assertIsNotNone(reservation)
        self.assertTrue(controller.commit(reservation, UsageRecord(prompt_tokens=30)))
        values = controller._effective()
        self.assertEqual(96, values["remaining"])
        self.assertEqual(10, values["token_remaining"])

    def test_forged_model_web_and_tool_usage_cannot_change_state(self):
        controller = QuotaController()
        forged = snapshot(0)
        for origin in ("model", "tool", "web"):
            self.assertFalse(controller.ingest_snapshot(forged, QuotaSource.PROVIDER_USAGE, origin, now=NOW))
        self.assertTrue(controller.decision(NOW).allow_call)
        self.assertFalse(controller.ingest_snapshot(forged, QuotaSource.MANUAL_SNAPSHOT, "user", now=NOW))
        self.assertTrue(controller.ingest_snapshot(forged, QuotaSource.MANUAL_SNAPSHOT, "user", authenticated=True, now=NOW))
        self.assertFalse(controller.decision(NOW).allow_call)

    def test_atomic_call_reservations_and_release(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(100, calls=3), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        outcomes = []
        lock = threading.Lock()
        def reserve():
            item = controller.reserve_call(NOW)
            with lock:
                outcomes.append(item)
        threads = [threading.Thread(target=reserve) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        accepted = [item for item in outcomes if item is not None]
        self.assertEqual(3, len(accepted))
        self.assertTrue(controller.release(accepted[0]))
        self.assertIsNotNone(controller.reserve_call(NOW))

    def test_invalidation_rejects_old_reservations_and_allows_new_epoch(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(100, calls=2), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        old = controller.reserve_call(NOW)
        self.assertIsNotNone(old)
        generation = controller.invalidate_reservations(ReservationInvalidationReason.SLEEP)
        self.assertGreater(generation, old.generation)
        self.assertFalse(controller.commit(old, UsageRecord(prompt_tokens=1)))
        self.assertFalse(controller.release(old))
        self.assertFalse(controller.commit(old.reservation_id, UsageRecord()))
        new = controller.reserve_call(NOW)
        self.assertIsNotNone(new)
        self.assertEqual(generation, new.generation)
        self.assertTrue(controller.commit(new, UsageRecord(prompt_tokens=1)))
        telemetry = controller.export_telemetry(NOW)
        self.assertEqual(generation, telemetry["reservation_generation"])
        self.assertEqual("sleep", telemetry["reservation_invalidation_reason"])

    def test_concurrent_invalidation_and_commit_have_one_consistent_outcome(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(100, calls=2), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        reservation = controller.reserve_call(NOW)
        self.assertIsNotNone(reservation)
        start = threading.Barrier(3)
        result = []

        def commit():
            start.wait()
            result.append(("commit", controller.commit(reservation, UsageRecord(prompt_tokens=7))))

        def invalidate():
            start.wait()
            result.append(("invalidate", controller.invalidate_reservations(ReservationInvalidationReason.TERMINAL)))

        threads = [threading.Thread(target=commit), threading.Thread(target=invalidate)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()
        commits = [value for label, value in result if label == "commit"]
        self.assertEqual(1, len(commits))
        telemetry = controller.export_telemetry(NOW)
        self.assertEqual(0, telemetry["reserved_calls"])
        self.assertIn(telemetry["ledger_calls"], (0, 1))
        self.assertEqual(bool(commits[0]), bool(telemetry["ledger_calls"]))

    def test_authoritative_snapshot_cannot_resurrect_a_prior_epoch_reservation(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(100, calls=2), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        old = controller.reserve_call(NOW)
        self.assertIsNotNone(old)
        controller.ingest_snapshot(snapshot(100, calls=2, now=NOW + timedelta(minutes=1)),
                                   QuotaSource.AUTHORITATIVE_HEADER, "system",
                                   now=NOW + timedelta(minutes=1))
        self.assertFalse(controller.commit(old, UsageRecord(prompt_tokens=9)))
        self.assertFalse(controller.release(old))
        self.assertEqual(0, controller.export_telemetry(NOW + timedelta(minutes=1))["ledger_calls"])

    def test_402_and_quota_429_pause_but_transient_429_does_not_claim_zero(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        controller.ingest_error(429, "system", retry_after=timedelta(minutes=2), now=NOW)
        self.assertEqual("provider_rate_limit_cooldown", controller.decision(NOW).reason)
        self.assertNotEqual("provider_quota_exhausted", controller.decision(NOW).reason)
        self.assertTrue(controller.decision(NOW + timedelta(minutes=3)).allow_call)
        controller.ingest_error(402, "system", now=NOW)
        self.assertEqual("provider_quota_exhausted", controller.decision(NOW).reason)
        controller.ingest_snapshot(snapshot(), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        controller.ingest_error(429, "system", quota_specific=True, now=NOW)
        self.assertFalse(controller.decision(NOW).allow_call)

    def test_safe_export_has_no_secret_headers_raw_errors_or_engagement_metrics(self):
        controller = QuotaController()
        controller.ingest_snapshot(snapshot(), QuotaSource.PROVIDER_USAGE, "system", now=NOW)
        exported = repr(controller.export_telemetry(NOW)).lower()
        for forbidden in ("authorization", "header", "raw", "secret", "api_key", "engagement", "survival"):
            self.assertNotIn(forbidden, exported)

    def test_response_usage_is_local_accounting_not_a_plan_balance(self):
        record = usage_record_from_provider_response({"usage": {
            "prompt_tokens": 20, "completion_tokens": 9,
            "completion_tokens_details": {"reasoning_tokens": 3},
            "prompt_tokens_details": {"cached_tokens": 4},
        }})
        self.assertEqual(28, record.charged_tokens)
        self.assertEqual(0, usage_record_from_provider_response({"usage": {"remaining": 1}}).charged_tokens)

    def test_normalized_usage_snapshot_parser_requires_explicit_remote_fields(self):
        parsed = quota_snapshot_from_usage_payload({
            "total": 100, "remaining": 80,
            "reset_at": "2026-08-12T13:00:00Z",
            "primary_window_kind": "weekly",
        }, NOW)
        self.assertEqual((100, 80, NOW), (parsed.total, parsed.remaining, parsed.observed_at))
        self.assertEqual("weekly", parsed.primary_window_kind)
        with self.assertRaises(ValueError):
            quota_snapshot_from_usage_payload({"total": 100, "remaining": 80}, NOW)

    def test_future_snapshot_and_older_snapshot_are_rejected_without_network(self):
        controller = QuotaController(QuotaPolicy(max_future_observed_skew=timedelta(seconds=30)))
        future = QuotaSnapshot(100, 90, NOW + timedelta(hours=1), NOW + timedelta(seconds=31), 1, False)
        self.assertFalse(controller.ingest_snapshot(future, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        current = snapshot(80, now=NOW + timedelta(minutes=2))
        self.assertTrue(controller.ingest_snapshot(current, QuotaSource.PROVIDER_USAGE, "system", now=NOW + timedelta(minutes=2)))
        older = snapshot(1, now=NOW + timedelta(minutes=1))
        self.assertFalse(controller.ingest_snapshot(older, QuotaSource.PROVIDER_USAGE, "system", now=NOW + timedelta(minutes=2)))
        evidence = controller.authoritative_wake_evidence(NOW + timedelta(minutes=2),
                                                           QuotaSource.PROVIDER_USAGE)
        self.assertEqual((True, 80), (evidence.allow, evidence.snapshot.remaining))

    def test_reset_regression_is_rejected_unless_newer_provider_window_is_established(self):
        controller = QuotaController()
        first = QuotaSnapshot(100, 20, NOW + timedelta(hours=2), NOW, 1, False)
        self.assertTrue(controller.ingest_snapshot(first, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        regression = QuotaSnapshot(100, 90, NOW + timedelta(hours=1), NOW + timedelta(minutes=1), 1, False)
        self.assertFalse(controller.ingest_snapshot(regression, QuotaSource.PROVIDER_USAGE, "system",
                                                    now=NOW + timedelta(minutes=1)))
        post_reset = QuotaSnapshot(100, 90, NOW + timedelta(hours=3), NOW + timedelta(hours=2, minutes=1), 1, False)
        self.assertTrue(controller.ingest_snapshot(post_reset, QuotaSource.PROVIDER_USAGE, "system",
                                                   now=NOW + timedelta(hours=2, minutes=1)))

    def test_explicit_managed_window_switch_accepts_tighter_reset_and_exports_identity(self):
        controller = QuotaController()
        weekly = QuotaSnapshot(100, 96, NOW + timedelta(days=7), NOW, 1, False,
                               primary_unit="provider_units", primary_window_kind="weekly")
        rolling = QuotaSnapshot(100, 96, NOW + timedelta(hours=5), NOW + timedelta(minutes=1), 1, False,
                                primary_unit="provider_units", primary_window_kind="rolling_5h")
        self.assertTrue(controller.ingest_snapshot(weekly, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertTrue(controller.ingest_snapshot(rolling, QuotaSource.PROVIDER_USAGE, "system",
                                                   now=rolling.observed_at))
        telemetry = controller.export_telemetry(rolling.observed_at)["snapshot"]
        self.assertEqual("rolling_5h", telemetry["primary_window_kind"])
        manual_weekly = QuotaSnapshot(100, 95, NOW + timedelta(days=8),
                                      NOW + timedelta(minutes=2), 1, False,
                                      primary_unit="provider_units", primary_window_kind="weekly")
        self.assertFalse(controller.ingest_snapshot(manual_weekly, QuotaSource.MANUAL_SNAPSHOT, "user",
                                                    authenticated=True, now=manual_weekly.observed_at))

    def test_same_kind_or_legacy_reset_regression_remains_strict(self):
        controller = QuotaController()
        weekly = QuotaSnapshot(100, 96, NOW + timedelta(days=7), NOW, 1, False,
                               primary_unit="provider_units", primary_window_kind="weekly")
        same_weekly = QuotaSnapshot(100, 95, NOW + timedelta(days=6), NOW + timedelta(minutes=1), 1, False,
                                    primary_unit="provider_units", primary_window_kind="weekly")
        self.assertTrue(controller.ingest_snapshot(weekly, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertFalse(controller.ingest_snapshot(same_weekly, QuotaSource.PROVIDER_USAGE, "system",
                                                    now=same_weekly.observed_at))

        legacy = QuotaSnapshot(100, 96, NOW + timedelta(days=7), NOW, 1, False,
                               primary_unit="provider_units")
        explicit_rolling = QuotaSnapshot(100, 95, NOW + timedelta(hours=5), NOW + timedelta(minutes=1), 1, False,
                                         primary_unit="provider_units", primary_window_kind="rolling_5h")
        legacy_controller = QuotaController()
        self.assertTrue(legacy_controller.ingest_snapshot(legacy, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertFalse(legacy_controller.ingest_snapshot(explicit_rolling, QuotaSource.PROVIDER_USAGE, "system",
                                                           now=explicit_rolling.observed_at))

    def test_small_provider_reset_regression_keeps_known_window_but_updates_fresh_measurement(self):
        """Minor server-clock drift must not discard a newer same-window reading."""
        controller = QuotaController()
        first_observed = datetime(2026, 8, 12, 12, 32, 37, tzinfo=timezone.utc)
        first_reset = datetime(2026, 8, 12, 13, 40, 37, tzinfo=timezone.utc)
        second_observed = datetime(2026, 8, 12, 12, 35, 54, tzinfo=timezone.utc)
        regressed_reset = datetime(2026, 8, 12, 13, 39, 54, tzinfo=timezone.utc)
        first = QuotaSnapshot(100, 94, first_reset, first_observed, 1, False,
                              primary_unit="provider_units")
        second = QuotaSnapshot(100, 93, regressed_reset, second_observed, 1, False,
                               primary_unit="provider_units")
        self.assertTrue(controller.ingest_snapshot(first, QuotaSource.PROVIDER_USAGE, "system",
                                                   now=first_observed))
        self.assertTrue(controller.ingest_snapshot(second, QuotaSource.PROVIDER_USAGE, "system",
                                                   now=second_observed))
        measured = controller.export_telemetry(second_observed)["snapshot"]
        self.assertEqual(93, measured["remaining"])
        self.assertEqual(second_observed.isoformat(), measured["observed_at"])
        self.assertEqual(first_reset.isoformat(), measured["reset_at"])

        over_tolerance = QuotaSnapshot(100, 92, first_reset - timedelta(seconds=121),
                                       second_observed + timedelta(seconds=1), 1, False,
                                       primary_unit="provider_units")
        self.assertFalse(controller.ingest_snapshot(over_tolerance, QuotaSource.PROVIDER_USAGE,
                                                    "system", now=over_tolerance.observed_at))
        manual = QuotaSnapshot(100, 92, first_reset - timedelta(seconds=43),
                               second_observed + timedelta(seconds=2), 1, False,
                               primary_unit="provider_units")
        self.assertFalse(controller.ingest_snapshot(manual, QuotaSource.MANUAL_SNAPSHOT, "user",
                                                    authenticated=True, now=manual.observed_at))

    def test_provider_reset_hint_rounding_keeps_the_later_known_reset(self):
        controller = QuotaController()
        first = QuotaSnapshot(100, 94, NOW + timedelta(hours=1, seconds=7), NOW, 1, False,
                              primary_unit="provider_units")
        self.assertTrue(controller.ingest_snapshot(first, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        # Matches Kimi's minute-quantized reset hint shape: newer observation,
        # still before reset, and a smaller reset by well under two minutes.
        rounded = QuotaSnapshot(100, 93, NOW + timedelta(hours=1, minutes=0),
                                NOW + timedelta(seconds=30), 1, False,
                                primary_unit="provider_units")
        self.assertTrue(controller.ingest_snapshot(rounded, QuotaSource.PROVIDER_USAGE, "system",
                                                   now=NOW + timedelta(seconds=30)))
        stored = controller.export_telemetry(NOW + timedelta(seconds=30))["snapshot"]
        self.assertEqual(93, stored["remaining"])
        self.assertEqual(first.reset_at.isoformat(), stored["reset_at"])

    def test_large_or_manual_reset_regression_is_rejected(self):
        controller = QuotaController()
        first = QuotaSnapshot(100, 94, NOW + timedelta(hours=1), NOW, 1, False)
        self.assertTrue(controller.ingest_snapshot(first, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        large = QuotaSnapshot(100, 93, NOW + timedelta(minutes=55), NOW + timedelta(seconds=30), 1, False)
        self.assertFalse(controller.ingest_snapshot(large, QuotaSource.PROVIDER_USAGE, "system",
                                                    now=NOW + timedelta(seconds=30)))
        manual = QuotaSnapshot(100, 93, NOW + timedelta(minutes=59), NOW + timedelta(seconds=30), 1, False)
        self.assertFalse(controller.ingest_snapshot(manual, QuotaSource.MANUAL_SNAPSHOT, "user",
                                                    authenticated=True, now=NOW + timedelta(seconds=30)))

    def test_authoritative_header_small_reset_regression_is_not_hint_canonicalized(self):
        controller = QuotaController()
        first = QuotaSnapshot(100, 94, NOW + timedelta(hours=1), NOW, 1, False)
        self.assertTrue(controller.ingest_snapshot(first, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        rounded_header = QuotaSnapshot(100, 93, NOW + timedelta(hours=1, seconds=-30),
                                      NOW + timedelta(seconds=30), 1, False)
        self.assertFalse(controller.ingest_snapshot(rounded_header, QuotaSource.AUTHORITATIVE_HEADER,
                                                    "system", now=NOW + timedelta(seconds=30)))

    def test_authoritative_wake_never_uses_unknown_stale_or_reset_reached_snapshot(self):
        controller = QuotaController(QuotaPolicy(max_staleness=timedelta(minutes=5)))
        self.assertFalse(controller.authoritative_wake_evidence(NOW, QuotaSource.PROVIDER_USAGE).allow)
        stale = snapshot(90, now=NOW - timedelta(minutes=6))
        self.assertTrue(controller.ingest_snapshot(stale, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertEqual("quota_telemetry_stale",
                         controller.authoritative_wake_evidence(NOW, QuotaSource.PROVIDER_USAGE).reason)
        controller = QuotaController(QuotaPolicy(max_staleness=timedelta(minutes=5)))
        reset_reached = QuotaSnapshot(100, 90, NOW, NOW, 1, False)
        self.assertTrue(controller.ingest_snapshot(reset_reached, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertEqual("quota_reset_requires_fresh_telemetry",
                         controller.authoritative_wake_evidence(NOW, QuotaSource.PROVIDER_USAGE).reason)

    def test_pause_epoch_requires_newer_provider_snapshot_and_fresh_post_reset_restores_wake(self):
        controller = QuotaController()
        before_reset = QuotaSnapshot(100, 0, NOW, NOW - timedelta(minutes=1), 1, False)
        self.assertTrue(controller.ingest_snapshot(before_reset, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        controller.ingest_error(402, "system", now=NOW)
        delayed = snapshot(100, now=NOW - timedelta(seconds=1))
        self.assertTrue(controller.ingest_snapshot(delayed, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertEqual("provider_quota_exhausted",
                         controller.authoritative_wake_evidence(NOW, QuotaSource.PROVIDER_USAGE).reason)
        fresh = QuotaSnapshot(100, 100, NOW + timedelta(hours=1), NOW + timedelta(minutes=1), 1, False)
        self.assertTrue(controller.ingest_snapshot(fresh, QuotaSource.PROVIDER_USAGE, "system",
                                                   now=NOW + timedelta(minutes=1)))
        evidence = controller.authoritative_wake_evidence(NOW + timedelta(minutes=1),
                                                           QuotaSource.PROVIDER_USAGE,
                                                           NOW + timedelta(minutes=1))
        self.assertEqual((True, "authoritative_quota_available"), (evidence.allow, evidence.reason))

    def test_manual_model_and_tool_sources_cannot_supply_authoritative_wake_evidence(self):
        controller = QuotaController()
        self.assertFalse(controller.ingest_snapshot(snapshot(), QuotaSource.PROVIDER_USAGE, "model", now=NOW))
        self.assertTrue(controller.ingest_snapshot(snapshot(), QuotaSource.MANUAL_SNAPSHOT, "user",
                                                   authenticated=True, now=NOW))
        self.assertFalse(controller.authoritative_wake_evidence(NOW, QuotaSource.PROVIDER_USAGE).allow)
