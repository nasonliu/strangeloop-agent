from datetime import datetime, timedelta, timezone
import threading
import unittest

from strangeloop.kimi_cli import ManagedUsageWindow
from strangeloop.sleep import (SleepState, SleepWakeCoordinator, SleepWakePolicy,
                               build_public_archive)


NOW = datetime(2026, 8, 12, 12, tzinfo=timezone.utc)


class Clock:
    def __init__(self): self.now = NOW
    def __call__(self): return self.now


def window(remaining=1, total=10, reset=None):
    return ManagedUsageWindow("rolling_5h", total, remaining, reset or NOW + timedelta(hours=1))


class SleepWakeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.coordinator = SleepWakeCoordinator(SleepWakePolicy(threshold=.1, max_staleness=timedelta(minutes=5),
                                                                 initial_backoff=timedelta(seconds=10), max_backoff=timedelta(seconds=40)), self.clock)

    def test_threshold_only_uses_fresh_authoritative_rolling_window(self):
        self.assertEqual(.10, SleepWakePolicy().threshold)
        self.assertFalse(self.coordinator.prepare_sleep([], NOW, user_auto_wake=True))
        self.assertFalse(self.coordinator.prepare_sleep([window()], NOW - timedelta(minutes=6), user_auto_wake=True))
        self.assertFalse(self.coordinator.prepare_sleep([window()], NOW, authoritative=False, user_auto_wake=True))
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)

    def test_stale_future_and_older_callbacks_do_not_wake(self):
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        self.assertFalse(self.coordinator.check_and_wake([window(10)], NOW - timedelta(minutes=6), authority_token=token, expected_generation=generation))
        self.clock.now += timedelta(seconds=11)
        self.assertFalse(self.coordinator.check_and_wake([window(10)], self.clock.now + timedelta(seconds=1), authority_token=self.coordinator.authority_token, expected_generation=self.coordinator.generation))
        self.clock.now += timedelta(seconds=21)
        self.assertFalse(self.coordinator.check_and_wake([window(10)], NOW, authority_token=self.coordinator.authority_token, expected_generation=self.coordinator.generation))

    def test_reset_due_never_wakes_but_fresh_restored_window_does(self):
        reset = NOW + timedelta(minutes=1)
        self.assertTrue(self.coordinator.prepare_sleep([window(reset=reset)], NOW, user_auto_wake=True))
        self.clock.now = reset
        self.assertTrue(self.coordinator.refresh_due())
        self.assertEqual(SleepState.SLEEPING, self.coordinator.state)
        self.assertTrue(self.coordinator.check_and_wake([window(10, reset=NOW + timedelta(hours=2))], self.clock.now,
                                                        authority_token=self.coordinator.authority_token, expected_generation=self.coordinator.generation))
        self.assertEqual(SleepState.READY, self.coordinator.state)

    def test_revoke_stop_and_duplicate_callbacks_are_safe(self):
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        self.assertTrue(self.coordinator.revoke_auto_wake())
        self.assertFalse(self.coordinator.check_and_wake([window(10)], NOW + timedelta(seconds=1), authority_token=token, expected_generation=generation))
        self.coordinator.stop()
        self.assertEqual(SleepState.TERMINAL, self.coordinator.state)
        self.assertFalse(self.coordinator.prepare_sleep([window()], NOW + timedelta(seconds=2)))

    def test_successful_callback_cannot_be_replayed(self):
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        self.clock.now = NOW + timedelta(seconds=1)
        self.assertTrue(self.coordinator.check_and_wake([window(10)], NOW + timedelta(seconds=1),
                                                        authority_token=token, expected_generation=generation))
        self.assertFalse(self.coordinator.check_and_wake([window(10)], NOW + timedelta(seconds=2),
                                                         authority_token=token, expected_generation=generation))

    def test_mark_awake_is_one_shot_ready_to_active_cas(self):
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        self.clock.now += timedelta(seconds=1)
        self.assertTrue(self.coordinator.check_and_wake(
            [window(10)], self.clock.now, authority_token=self.coordinator.authority_token,
            expected_generation=self.coordinator.generation))
        ready_epoch = self.coordinator.epoch
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        self.assertTrue(token.startswith("ready:"))
        self.assertFalse(self.coordinator.mark_awake(
            authority_token="ready:wrong:token", expected_generation=generation))
        self.assertEqual(SleepState.READY, self.coordinator.state)
        self.assertTrue(self.coordinator.mark_awake(
            authority_token=token, expected_generation=generation))
        self.assertEqual((SleepState.ACTIVE, ready_epoch, generation + 1),
                         (self.coordinator.state, self.coordinator.epoch,
                          self.coordinator.generation))
        self.assertFalse(self.coordinator.mark_awake(
            authority_token=token, expected_generation=generation))
        self.assertFalse(self.coordinator.to_payload()["auto_wake_user_approved"])

    def test_restore_only_resumes_safe_states_and_invalidates_old_process_authority(self):
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        payload = self.coordinator.to_payload()
        old_token, old_generation = self.coordinator.authority_token, self.coordinator.generation
        restored = SleepWakeCoordinator.from_payload(payload, self.coordinator.policy, self.clock)
        self.assertEqual(SleepState.SLEEPING, restored.state)
        self.assertEqual(old_generation + 1, restored.generation)
        self.assertNotEqual(old_token, restored.authority_token)
        self.assertFalse(restored.to_payload()["auto_wake_user_approved"])
        self.clock.now += timedelta(seconds=1)
        self.assertFalse(restored.check_and_wake(
            [window(10)], self.clock.now, authority_token=restored.authority_token,
            expected_generation=restored.generation))

        self.coordinator.stop()
        terminal = SleepWakeCoordinator.restore(self.coordinator.to_payload(),
                                                self.coordinator.policy, self.clock)
        self.assertEqual(SleepState.TERMINAL, terminal.state)
        active = SleepWakeCoordinator(clock=self.clock).to_payload()
        with self.assertRaises(ValueError):
            SleepWakeCoordinator.replay(active, clock=self.clock)
        with self.assertRaises(ValueError):
            SleepWakeCoordinator.from_payload(dict(payload, hidden_text="forbidden"),
                                              clock=self.clock)

    def test_invalid_prepare_is_atomic_and_concurrent_mark_has_one_winner(self):
        before = self.coordinator.to_payload()
        with self.assertRaises(ValueError):
            self.coordinator.prepare_sleep(
                [window(reset=datetime(2026, 8, 12, 13))], NOW,
                user_auto_wake=True)
        self.assertEqual(before, self.coordinator.to_payload())

        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        self.clock.now += timedelta(seconds=1)
        self.assertTrue(self.coordinator.check_and_wake(
            [window(10)], self.clock.now, authority_token=self.coordinator.authority_token,
            expected_generation=self.coordinator.generation))
        token, generation = self.coordinator.authority_token, self.coordinator.generation
        outcomes = []
        outcome_lock = threading.Lock()

        def mark():
            result = self.coordinator.mark_awake(
                authority_token=token, expected_generation=generation)
            with outcome_lock:
                outcomes.append(result)

        workers = [threading.Thread(target=mark) for _ in range(12)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(1, sum(outcomes))
        self.assertEqual(SleepState.ACTIVE, self.coordinator.state)

    def test_backoff_and_archive_are_canonical_and_content_free(self):
        self.assertTrue(self.coordinator.prepare_sleep([window()], NOW, user_auto_wake=True))
        self.assertFalse(self.coordinator.check_and_wake([], NOW + timedelta(seconds=1), authority_token=self.coordinator.authority_token, expected_generation=self.coordinator.generation))
        first = self.coordinator.to_payload()
        self.assertEqual(1, first["failure_count"])
        archive = build_public_archive(chain_head_sequence=3, chain_head_hash="a" * 64,
            active_seed_ids=("seed-b", "seed-a"), active_claim_ids=("claim-1",), pending_public_event_ids=("evt-2", "evt-1"),
            quota_source="provider_usage", quota_observed_at=NOW, quota_reset_at=NOW + timedelta(hours=1), quota_remaining=1, quota_total=10)
        again = build_public_archive(chain_head_sequence=3, chain_head_hash="a" * 64,
            active_seed_ids=("seed-a", "seed-b"), active_claim_ids=("claim-1",), pending_public_event_ids=("evt-1", "evt-2"),
            quota_source="provider_usage", quota_observed_at=NOW, quota_reset_at=NOW + timedelta(hours=1), quota_remaining=1, quota_total=10)
        self.assertEqual(archive.to_payload(), again.to_payload())
        self.assertEqual(3, archive.schema_version)
        self.assertEqual("rolling_5h", archive.to_payload()["quota"]["window_kind"])
        legacy = build_public_archive(chain_head_sequence=3, chain_head_hash="a" * 64,
            quota_source="provider_usage", quota_observed_at=NOW, quota_reset_at=NOW + timedelta(hours=1),
            quota_remaining=1, quota_total=10, schema_version=2)
        self.assertNotIn("window_kind", legacy.to_payload()["quota"])
        self.assertNotEqual(archive.digest, legacy.digest)
        text = repr(archive.to_payload()).lower()
        for forbidden in ("path", "bytes", "reasoning", "chain_of_thought", "secret"):
            self.assertNotIn(forbidden, text)
