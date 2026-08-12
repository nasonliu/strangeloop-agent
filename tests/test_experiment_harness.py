from __future__ import annotations

from hashlib import sha256
import time
import unittest

from strangeloop.experiment_harness import (ExperimentAuthority, ExperimentBudget,
                                            ExperimentHarness, ExperimentKind, ExperimentRegistryPolicy,
                                            ExperimentSpec, ExperimentStatus)


def digest(value):
    return sha256(value.encode("ascii")).hexdigest()


class ExperimentHarnessTests(unittest.TestCase):
    def setUp(self):
        self.harness = ExperimentHarness("host-fixture-seed")
        self.policy = ExperimentRegistryPolicy(tuple(ExperimentKind), max_experiments=16)
        self.authority = self.harness.issue_authority(self.policy, digest("approval"), time.monotonic() + 10)

    def spec(self, kind=ExperimentKind.FRONTIER_REPLAY, budget=None, seed=None, experiment_id=None):
        return ExperimentSpec(experiment_id or ("fixture-" + kind.value), kind, digest("hypothesis"), digest("baseline"),
                              digest("treatment"), digest("input"),
                              seed or self.harness.host_seed_digest, budget or ExperimentBudget())

    def test_each_fixed_handler_is_deterministic_and_exposes_no_seed(self):
        for kind in ExperimentKind:
            first = self.harness.run(self.spec(kind), self.authority)
            # A replay is a separate, independently pre-registered execution.
            other = ExperimentHarness("host-fixture-seed")
            other_authority = other.issue_authority(self.policy, digest("approval"), time.monotonic() + 10)
            second = other.run(self.spec(kind), other_authority)
            self.assertEqual(ExperimentStatus.PASSED, first.status)
            self.assertEqual(first, second)
            self.assertNotIn("host-fixture-seed", repr(first))

    def test_result_digest_is_cross_instance_measurement_signature(self):
        first = self.harness.run(self.spec(experiment_id="first-replay"), self.authority)
        second = self.harness.run(self.spec(experiment_id="second-replay"), self.authority)

        self.assertNotEqual(first.experiment_id, second.experiment_id)
        self.assertNotEqual(first.spec_digest, second.spec_digest)
        self.assertNotEqual(first.reproduction_digest, second.reproduction_digest)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.invariant_codes, second.invariant_codes)
        self.assertEqual(first.result_digest, second.result_digest)
        self.assertNotIn("host-fixture-seed", repr(first))
        self.assertNotIn(self.harness.host_seed_digest, repr(first))

    def test_result_digest_changes_with_measurement_or_kind(self):
        replay = self.spec(ExperimentKind.FRONTIER_REPLAY, experiment_id="replay-signature")
        changed_measurement = self.spec(ExperimentKind.FRONTIER_REPLAY,
                                        experiment_id="changed-measurement")
        changed_kind = self.spec(ExperimentKind.TD_INVARIANTS, experiment_id="changed-kind")

        baseline = self.harness._result(replay, ExperimentStatus.PASSED,
                                        (("measurement", 1.0),), ("invariant",))
        different_measurement = self.harness._result(changed_measurement, ExperimentStatus.PASSED,
                                                     (("measurement", 0.0),), ("invariant",))
        different_kind = self.harness._result(changed_kind, ExperimentStatus.PASSED,
                                              (("measurement", 1.0),), ("invariant",))

        self.assertNotEqual(baseline.result_digest, different_measurement.result_digest)
        self.assertNotEqual(baseline.result_digest, different_kind.result_digest)

    def test_no_model_or_missing_authority_can_self_authorize(self):
        with self.assertRaises(PermissionError):
            self.harness.run(self.spec(), None)
        restricted_policy = ExperimentRegistryPolicy((ExperimentKind.FRONTIER_REPLAY,))
        restricted = self.harness.issue_authority(restricted_policy, digest("approval2"), time.monotonic() + 10)
        with self.assertRaises(PermissionError):
            self.harness.run(self.spec(ExperimentKind.TD_INVARIANTS), restricted)
        expired = self.harness.issue_authority(self.policy, digest("approval3"), time.monotonic() - 1)
        with self.assertRaises(PermissionError):
            self.harness.run(self.spec(), expired)

    def test_registry_digest_binds_sorted_policy_and_enforces_one_shot_and_caps(self):
        reordered = ExperimentRegistryPolicy(tuple(reversed(tuple(ExperimentKind))), max_experiments=1)
        canonical = ExperimentRegistryPolicy(tuple(ExperimentKind), max_experiments=1)
        self.assertEqual(reordered.registry_digest, canonical.registry_digest)
        with self.assertRaises(TypeError):
            ExperimentAuthority(digest("wrong"), canonical, digest("other"), time.monotonic() + 10)
        authority = self.harness.issue_authority(canonical, digest("single"), time.monotonic() + 10)
        first = self.spec(ExperimentKind.FRONTIER_REPLAY)
        self.assertEqual(ExperimentStatus.PASSED, self.harness.run(first, authority).status)
        with self.assertRaises(PermissionError):
            self.harness.run(first, authority)
        second = ExperimentSpec("a-second-experiment", ExperimentKind.FRONTIER_REPLAY,
                                digest("hypothesis"), digest("baseline"), digest("treatment"),
                                digest("input"), self.harness.host_seed_digest, ExperimentBudget())
        with self.assertRaises(PermissionError):
            self.harness.run(second, authority)
        capped = ExperimentRegistryPolicy((ExperimentKind.FRONTIER_REPLAY,), max_trials=2,
                                          max_steps=20, max_wall_ms=20)
        capped_authority = self.harness.issue_authority(capped, digest("cap"), time.monotonic() + 10)
        with self.assertRaises(PermissionError):
            self.harness.run(self.spec(budget=ExperimentBudget(max_trials=3)), capped_authority)

    def test_authority_is_host_opaque_process_local_and_not_serializable(self):
        with self.assertRaises(TypeError):
            ExperimentAuthority()
        self.assertEqual("<ExperimentAuthority host-issued>", repr(self.authority))
        other = ExperimentHarness("host-fixture-seed")
        with self.assertRaises(PermissionError):
            other.run(self.spec(), self.authority)
        import pickle
        with self.assertRaises(TypeError):
            pickle.dumps(self.authority)

    def test_rejects_seed_mismatch_and_unbounded_parameters(self):
        with self.assertRaises(PermissionError):
            self.harness.run(self.spec(seed=digest("other")), self.authority)
        with self.assertRaises(ValueError):
            ExperimentBudget(max_trials=65)
        with self.assertRaises(ValueError):
            ExperimentBudget(max_steps=10001)
        with self.assertRaises(ValueError):
            ExperimentBudget(max_wall_ms=2001)

    def test_no_free_code_path_url_or_callback_surface_exists(self):
        with self.assertRaises(TypeError):
            ExperimentSpec("x", ExperimentKind.FRONTIER_REPLAY, digest("h"), digest("b"), digest("t"),
                           digest("i"), self.harness.host_seed_digest, ExperimentBudget(), code="x")
        with self.assertRaises(TypeError):
            ExperimentSpec("x", ExperimentKind.FRONTIER_REPLAY, digest("h"), digest("b"), digest("t"),
                           digest("i"), self.harness.host_seed_digest, ExperimentBudget(), path="/tmp/x")
        with self.assertRaises(TypeError):
            self.harness.run(self.spec(), self.authority, callback=lambda: None)

    def test_result_contains_measurements_not_reward_run_count_or_exit_code(self):
        result = self.harness.run(self.spec(ExperimentKind.DUPLICATE_SUPPRESSION), self.authority)
        self.assertNotIn("reward", result.__dict__)
        self.assertNotIn("run_count", result.__dict__)
        self.assertNotIn("exit_code", result.__dict__)
        self.assertEqual(1.0, result.metric_map()["duplicate_blocked"])


if __name__ == "__main__":
    unittest.main()
