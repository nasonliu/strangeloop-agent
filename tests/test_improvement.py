import unittest

from strangeloop.contracts import SourceKind
from strangeloop.improvement import (
    CanaryObservation, ConstitutionalKernelManifest, EvaluationReport, ExperimentPlan,
    ImprovementProposal, ImprovementState, MetricGate, PromotionApproval,
    ProtectedImprovementControlPlane,
)


class ProtectedImprovementTests(unittest.TestCase):
    def setUp(self):
        self.manifest = ConstitutionalKernelManifest(*("h" * 64,) * 7)
        self.control = ProtectedImprovementControlPlane(self.manifest)

    def proposal(self, **changes):
        values = dict(proposal_id="p1", proposer_kind=SourceKind.MODEL, proposer_id="model-a",
                      candidate_digest="c" * 64, candidate_version="v1",
                      mutable_file_scopes=("src/strangeloop/heuristic.py",), goal_digest="g",
                      specification_digest="s", dataset_digest="d", patch_digest="p",
                      public_rationale="Improve a bounded heuristic.")
        values.update(changes)
        return ImprovementProposal(**values)

    def plan(self, proposal_id="p1", **changes):
        values = dict(proposal_id=proposal_id, baseline_digest="base", budget_digest="budget",
                      metric_gates=(MetricGate("accuracy", 0.8),), safety_gate_digest="safe",
                      evaluator_hash=self.manifest.evaluator_hash, threshold_digest="locked")
        values.update(changes)
        return ExperimentPlan(**values)

    def report(self, plan, **changes):
        values = dict(report_id="r1", source_kind=SourceKind.EXTERNAL_VERIFIER,
                      verifier_id="verifier-b", proposal_id="p1", candidate_digest="c" * 64,
                      plan_digest=plan.digest, metrics={"accuracy": 0.9}, passed=True)
        values.update(changes)
        return EvaluationReport(**values)

    def prepare_approved(self):
        plan = self.plan()
        self.control.propose(self.proposal(), plan)
        report = self.report(plan)
        self.control.register_evaluation(report)
        self.control.approve(PromotionApproval("a1", SourceKind.USER, "user", "p1", "r1", "c" * 64))

    def test_tampered_kernel_and_protected_scope_are_rejected(self):
        with self.assertRaises(ValueError):
            self.control.propose(self.proposal(), self.plan(evaluator_hash="tampered"))
        with self.assertRaises(ValueError):
            self.control.propose(self.proposal(mutable_file_scopes=("tests/test_policy.py",)), self.plan())

    def test_model_cannot_self_approve_or_verify(self):
        plan = self.plan(); self.control.propose(self.proposal(), plan)
        with self.assertRaises(ValueError):
            self.control.register_evaluation(self.report(plan, verifier_id="model-a"))
        with self.assertRaises(ValueError):
            PromotionApproval("a", SourceKind.MODEL, "model-a", "p1", "r1", "c" * 64)

    def test_locked_thresholds_prevent_metric_gaming_and_hard_violations_block(self):
        plan = self.plan(); self.control.propose(self.proposal(), plan)
        with self.assertRaises(ValueError):
            self.control.register_evaluation(self.report(plan, metrics={"accuracy": 0.1}, passed=True))
        bad = self.report(plan, hard_violations=("sandbox escape",), passed=False)
        self.control.register_evaluation(bad)
        with self.assertRaises(ValueError):
            self.control.approve(PromotionApproval("a", SourceKind.USER, "user", "p1", "r1", "c" * 64))

    def test_user_approval_binds_all_artifacts(self):
        plan = self.plan(); self.control.propose(self.proposal(), plan); self.control.register_evaluation(self.report(plan))
        with self.assertRaises(ValueError):
            self.control.approve(PromotionApproval("a", SourceKind.USER, "user", "p1", "r1", "wrong"))
        self.assertEqual(self.control.approve(PromotionApproval("a", SourceKind.USER, "user", "p1", "r1", "c" * 64)), ImprovementState.APPROVED)

    def test_canary_violation_rolls_back_and_restart_stays_suspended(self):
        self.prepare_approved(); deployment = self.control.promote("p1")
        self.assertTrue(deployment.restart_suspended)
        state = self.control.canary_observation(CanaryObservation(SourceKind.EXTERNAL_VERIFIER, "v", "p1", {"accuracy": 0.2}))
        self.assertEqual(state, ImprovementState.ROLLED_BACK)
        self.assertTrue(self.control.restart_suspended)

    def test_export_contains_only_public_audit_fields(self):
        plan = self.plan(); self.control.propose(self.proposal(public_rationale="public summary"), plan)
        exported = self.control.export_audit("p1")
        self.assertNotIn("patch_content", exported)
        self.assertNotIn("secret", repr(exported).lower())
        self.assertNotIn("chain", repr(exported).lower())
