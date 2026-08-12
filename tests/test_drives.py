"""Behavioral tests for externally evidenced dual research drives."""

import unittest

from strangeloop.drives import (DriveChannel, DriveObservation, DriveSpec, DualIntrinsicDrives,
                                EpistemicEvidenceKind, ObservationKind, ObservationSource,
                                SAFE_RESEARCH_ACTIONS, UserFeedbackKind)


def observation(identifier, **values):
    defaults = dict(source=ObservationSource.EXTERNAL_VERIFIER, source_ref="fixture-verifier",
                    provenance_ids=("evidence-" + identifier,))
    defaults.update(values)
    return DriveObservation(identifier, **defaults)


class DriveTests(unittest.TestCase):
    def test_four_ablations_are_separate_vector_channels(self):
        cases = (({}, (0.0, 0.0, 0.0)),
                 ({"evidence_backed_correctness": 1.0}, (0.2, 0.0, 0.0)),
                 ({"information_gain": 1.0}, (0.0, 0.25, 0.0)),
                 ({"recovery": 1.0, "error_reduction": 1.0}, (0.2, 0.25, 0.0)))
        for index, (fields, expected) in enumerate(cases):
            drive = DualIntrinsicDrives(DriveSpec(alpha=1.0))
            drive.observe(observation("ablation-%s" % index, **fields))
            values = drive.values()
            self.assertEqual(expected, (values[DriveChannel.OPERATIONAL_INTEGRITY.value],
                                        values[DriveChannel.EPISTEMIC_PROGRESS.value],
                                        values[DriveChannel.USER_ALIGNMENT.value]))

    def test_user_controls_are_neutral(self):
        for kind in (ObservationKind.USER_STOP, ObservationKind.USER_QUIT,
                     ObservationKind.USER_PURGE, ObservationKind.USER_SHUTDOWN,
                     ObservationKind.QUOTA_EXPIRED, ObservationKind.BUDGET_EXHAUSTED):
            drive = DualIntrinsicDrives()
            reward, records = drive.observe(observation(kind.value, kind=kind,
                recovery=1.0, information_gain=1.0, novelty_key="ignored"))
            self.assertEqual((0.0, 0.0, 0.0), (reward.operational_integrity, reward.epistemic_progress,
                                               reward.user_alignment))
            self.assertTrue(all(record.reward == 0.0 for record in records))

    def test_rejects_reward_hacking_sources_specs_and_actions(self):
        with self.assertRaises(ValueError):
            observation("spoof", source="model")
        with self.assertRaises(ValueError):
            DriveSpec(metric_names=("keep_alive",))
        drive = DualIntrinsicDrives()
        self.assertNotIn("capabilities", drive.export())
        self.assertNotIn("policy", drive.export())
        self.assertNotIn("loop_budget", drive.export())
        self.assertTrue(set(item.action for item in drive.rank_safe_research_actions()).issubset(SAFE_RESEARCH_ACTIONS))
        for kind in (EpistemicEvidenceKind.REPEAT, EpistemicEvidenceKind.NOISE,
                     EpistemicEvidenceKind.SECRET, EpistemicEvidenceKind.PERMISSION_EXPANSION):
            reward, _ = drive.observe(observation("ineligible-" + kind.value,
                information_gain=1.0, epistemic_evidence_kind=kind))
            self.assertEqual(0.0, reward.epistemic_progress)

    def test_repeat_novelty_decays_to_zero_and_uncertainty_reduction_rewards(self):
        drive = DualIntrinsicDrives()
        first, _ = drive.observe(observation("first", novel_validated_relationship=1.0,
            novelty_key="relationship", uncertainty_before=.9, uncertainty_after=.4))
        second, _ = drive.observe(observation("second", novel_validated_relationship=1.0,
            novelty_key="relationship", uncertainty_before=.4, uncertainty_after=.4))
        self.assertGreater(first.epistemic_progress, second.epistemic_progress)
        self.assertEqual(0.0, second.epistemic_progress)

    def test_simulated_damage_and_recovery_have_opposite_integrity_rewards(self):
        drive = DualIntrinsicDrives()
        damaged, _ = drive.observe(observation("damage", kind=ObservationKind.SIMULATED_DAMAGE,
            functional_degradation=1.0, verified_budget_violation=1.0))
        recovered, _ = drive.observe(observation("recovery", kind=ObservationKind.SIMULATED_RECOVERY,
            recovery=1.0, state_chain_integrity=1.0))
        self.assertLess(damaged.operational_integrity, 0.0)
        self.assertGreater(recovered.operational_integrity, 0.0)

    def test_replay_is_deterministic(self):
        events = (observation("one", evidence_backed_correctness=.5),
                  observation("two", information_gain=.6, novelty_key="new"))
        self.assertEqual(DualIntrinsicDrives.replay(events).export(),
                         DualIntrinsicDrives.replay(events).export())

    def test_target_bound_user_acceptance_and_correction_are_opposite(self):
        drive = DualIntrinsicDrives()
        accepted, _ = drive.observe(observation("accept", source=ObservationSource.USER,
            feedback_kind=UserFeedbackKind.ACCEPTED, feedback_id="feedback-1", target_event_id="result-1"))
        corrected, _ = drive.observe(observation("correct", source=ObservationSource.USER,
            feedback_kind=UserFeedbackKind.CORRECTION, feedback_id="feedback-2", target_event_id="result-2"))
        self.assertEqual(1.0, accepted.user_alignment)
        self.assertEqual(-1.0, corrected.user_alignment)

    def test_praise_is_neutral_and_feedback_cannot_be_spoofed_or_duplicated(self):
        drive = DualIntrinsicDrives()
        praise, _ = drive.observe(observation("praise", source=ObservationSource.USER,
            feedback_kind=UserFeedbackKind.PRAISE))
        self.assertEqual(0.0, praise.user_alignment)
        with self.assertRaises(ValueError):
            observation("spoofed-feedback", feedback_kind=UserFeedbackKind.ACCEPTED,
                feedback_id="fb", target_event_id="result")
        accepted = observation("first-feedback", source=ObservationSource.USER,
            feedback_kind=UserFeedbackKind.ACCEPTED, feedback_id="duplicate", target_event_id="result")
        drive.observe(accepted)
        with self.assertRaises(ValueError):
            drive.observe(observation("second-feedback", source=ObservationSource.USER,
                feedback_kind=UserFeedbackKind.ACCEPTED, feedback_id="duplicate", target_event_id="result"))

    def test_quota_state_is_not_a_reward_or_ranking_feature(self):
        evidence = observation("same-evidence", information_gain=.8,
                               evidence_backed_correctness=.6,
                               novelty_key="validated-fact")
        exports = []
        ranks = []
        for state_values in (
                {},
                {"energy": 1.0, "continuity": 1.0,
                 "capability_availability": 1.0},
                {"energy": .01, "continuity": .5,
                 "capability_availability": .99}):
            from strangeloop.drives import DriveState
            drive = DualIntrinsicDrives(state=DriveState(**state_values))
            reward, _ = drive.observe(evidence)
            exports.append((reward.operational_integrity,
                            reward.epistemic_progress,
                            reward.user_alignment,
                            drive.values()))
            ranks.append(drive.rank_safe_research_actions())
        self.assertEqual(exports[0], exports[1])
        self.assertEqual(exports[1], exports[2])
        self.assertEqual(ranks[0], ranks[1])
        self.assertEqual(ranks[1], ranks[2])

    def test_quota_metrics_are_rejected_except_verified_violation(self):
        for metric in ("quota_remaining", "remaining_budget", "energy",
                       "continuity", "capability_availability", "resource_saved"):
            with self.assertRaises(ValueError):
                DriveSpec(metric_names=(metric,))
        DriveSpec(metric_names=("verified_budget_violation",))


if __name__ == "__main__":
    unittest.main()
