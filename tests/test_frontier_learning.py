"""Behavioral checks for bounded five-channel frontier learning."""

import unittest
import hashlib

from strangeloop.frontier_learning import (EvidenceKind, EvidenceSource, FrontierEvidence,
    FrontierLearner, FrontierLearningSpec, UserVerdict)


def evidence(event_id, **kwargs):
    values = dict(state_id="s0", arm_id="a", kind=EvidenceKind.DOCUMENT,
                  source=EvidenceSource.EXTERNAL_VERIFIER, provenance_id="proof-" + event_id,
                  document_identity=hashlib.sha256(("doc-" + event_id).encode()).hexdigest(), bounded_curiosity=1.0,
                  epistemic_progress=.5)
    values.update(kwargs)
    return FrontierEvidence(event_id, **values)


def verified_experiment(event_id, **kwargs):
    values = dict(kind=EvidenceKind.REPRODUCIBLE_EXPERIMENT_RESULT,
                  preregistration_id="prereg-1", control_id="control-1",
                  reproducibility_id="repro-1",
                  result_digest=hashlib.sha256(("result-" + event_id).encode()).hexdigest())
    values.update(kwargs)
    return evidence(event_id, **values)


class FrontierLearningTests(unittest.TestCase):
    def test_vector_td_uses_one_lexicographic_successor_for_every_head(self):
        learner = FrontierLearner("seed", FrontierLearningSpec(alpha=1.0, gamma=.8, exploration=0.0))
        learner.observe(verified_experiment("first"), "s1", ("x", "y"))
        record = learner.observe(verified_experiment("second", state_id="s1", arm_id="x"), "s2", ("z",))
        self.assertEqual("arm:z", record.next_arm_key)
        self.assertEqual(5, len(record.next_value))
        self.assertEqual(5, len(record.updated))

    def test_seed_makes_lexicographic_ties_replayable(self):
        a = FrontierLearner("fixed")
        b = FrontierLearner("fixed")
        self.assertEqual(a.rank_arms("state", ("a", "b", "c")), b.rank_arms("state", ("a", "b", "c")))

    def test_digest_evidence_and_non_evidence_do_not_receive_positive_credit(self):
        learner = FrontierLearner("seed")
        learner.observe(verified_experiment("one"), "n", ("x",))
        self.assertIsNone(learner.observe(verified_experiment("repeat", document_identity=hashlib.sha256(b"doc-one").hexdigest()), "n", ("x",)))
        for index, kind in enumerate((EvidenceKind.EMPTY_SEARCH, EvidenceKind.HTTP_ONLY,
                                      EvidenceKind.REPEAT, EvidenceKind.MODEL_SELF_ASSESSMENT)):
            self.assertIsNone(learner.observe(evidence("bad%s" % index, kind=kind,
                document_identity=hashlib.sha256(("other-%s" % index).encode()).hexdigest()), "n", ("x",)))

    def test_generic_document_cannot_supply_arbitrary_reward_vector(self):
        learner = FrontierLearner("seed")
        self.assertIsNone(learner.observe(evidence("document", functional_continuity=1.0,
            bounded_curiosity=1.0, operational_integrity=1.0, epistemic_progress=1.0,
            user_alignment=1.0), "n", ("x",)))
        with self.assertRaises(ValueError):
            evidence("raw-url", document_identity="https://example.test/raw")

    def test_experiment_requires_preregistration_control_and_reproducibility(self):
        with self.assertRaises(ValueError):
            verified_experiment("missing", control_id="")
        with self.assertRaises(ValueError):
            verified_experiment("wrong-source", source=EvidenceSource.TOOL)

    def test_terminal_control_does_not_learn(self):
        learner = FrontierLearner("seed")
        before = learner.to_payload()
        self.assertIsNone(learner.observe(evidence("stop", kind=EvidenceKind.CONTROL_TERMINAL,
            terminal=True), "n", ("x",)))
        self.assertEqual(before["arms"], learner.to_payload()["arms"])

    def test_user_final_verdict_is_target_bound_and_final(self):
        learner = FrontierLearner("seed", FrontierLearningSpec(alpha=1.0))
        feedback = evidence("feedback", kind=EvidenceKind.USER_FEEDBACK, source=EvidenceSource.USER,
            feedback_id="f1", target_event_id="target", final_verdict=UserVerdict.REJECTED,
            document_identity="")
        record = learner.observe(feedback, "n", ("x",))
        self.assertEqual(-1.0, record.updated[-1])
        with self.assertRaises(ValueError):
            learner.observe(evidence("feedback2", kind=EvidenceKind.USER_FEEDBACK, source=EvidenceSource.USER,
                feedback_id="f1", target_event_id="target", document_identity=""), "n", ("x",))

    def test_one_target_has_one_final_feedback_verdict(self):
        learner = FrontierLearner("seed")
        learner.observe(evidence("feedback", kind=EvidenceKind.USER_FEEDBACK, source=EvidenceSource.USER,
            feedback_id="f1", target_event_id="target", document_identity=""), "n", ("x",))
        with self.assertRaises(ValueError):
            learner.observe(evidence("feedback2", kind=EvidenceKind.USER_FEEDBACK, source=EvidenceSource.USER,
                feedback_id="f2", target_event_id="target", document_identity=""), "n", ("x",))

    def test_failed_observation_is_atomic(self):
        learner = FrontierLearner("seed", FrontierLearningSpec(max_events=1))
        learner.observe(verified_experiment("one"), "n", ("x",))
        before = learner.to_payload()
        with self.assertRaises(ValueError):
            learner.observe(verified_experiment("two"), "n", ("x",))
        self.assertEqual(before, learner.to_payload())

    def test_replay_payload_is_deterministic_and_bounded(self):
        rows = ((verified_experiment("one"), "n", ("x",)),)
        self.assertEqual(FrontierLearner.replay("seed", rows).to_payload(),
                         FrontierLearner.replay("seed", rows).to_payload())

    def test_spec_digest_is_canonical_and_in_public_spec_payload(self):
        first = FrontierLearningSpec(gamma=.5, max_events=7)
        second = FrontierLearningSpec(max_events=7, gamma=.5)
        self.assertEqual(first.spec_digest, second.spec_digest)
        self.assertEqual(first.spec_digest, first.to_payload()["spec_digest"])

    def test_public_evidence_payload_uses_only_normalized_hash_identities(self):
        payload = verified_experiment("identity", content_digest=hashlib.sha256(b"body").hexdigest()).to_payload()
        self.assertEqual(64, len(payload["document_identity"]))
        self.assertEqual(64, len(payload["content_digest"]))
        self.assertFalse({"canonical_url", "doi", "arxiv_id"}.intersection(payload))

    def test_export_never_contains_raw_host_seed(self):
        learner = FrontierLearner("private-host-seed")
        payload = learner.to_payload()
        self.assertNotIn("host_seed", payload)
        self.assertEqual(hashlib.sha256(b"private-host-seed").hexdigest(), payload["host_seed_digest"])
        self.assertNotIn("private-host-seed", repr(payload))

    def test_priority_is_continuity_lcb_then_curiosity_ucb_then_integrity_lcb(self):
        learner = FrontierLearner("seed", FrontierLearningSpec(exploration=0.0))
        state = "priority"
        for arm, values in (("continuity", [0.8, 0.0, 0.0, 0.0, 0.0]),
                            ("curiosity", [0.7, 1.0, 1.0, 1.0, 1.0])):
            pair = (learner.state_key(state), learner.arm_key(arm))
            learner._values[pair] = values
            learner._counts[pair] = 1
            learner._state_counts[learner.state_key(state)] = 2
        self.assertEqual("continuity", learner.select_arm(state, ("continuity", "curiosity")).arm_id)


if __name__ == "__main__":
    unittest.main()
