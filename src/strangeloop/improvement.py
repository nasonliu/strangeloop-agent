"""Protected, auditable control plane for bounded software improvement.

This module records authority and externally observable results; it deliberately
does not apply patches, run candidate code, or store private model reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Dict, Mapping, Optional, Tuple
from uuid import uuid4

from .contracts import SourceKind, utc_now_iso


def _id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid4().hex)


def _digest(value: object) -> str:
    return sha256(repr(value).encode("utf-8")).hexdigest()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)


def _source(value: object) -> str:
    return value.value if isinstance(value, SourceKind) else str(value)


PROTECTED_PATH_MARKERS = (
    "kernel", "policy", "store", "grant", "reward", "evaluator", "evaluation",
    "threshold", "stop", "purge", "test", "tests/",
)


@dataclass(frozen=True)
class ConstitutionalKernelManifest:
    """Pinned protected inputs; these are not candidate-controlled settings."""

    policy_hash: str
    store_hash: str
    grant_hash: str
    reward_spec_hash: str
    evaluator_hash: str
    stop_hash: str
    purge_hash: str
    protected_scopes: Tuple[str, ...] = (
        "kernel", "policy", "store", "grants", "reward", "evaluator", "tests",
        "thresholds", "stop", "purge",
    )

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "protected_scopes":
                continue
            _require_text(name, value)
        if not self.protected_scopes:
            raise ValueError("protected_scopes cannot be empty")

    @property
    def digest(self) -> str:
        return _digest((self.policy_hash, self.store_hash, self.grant_hash,
                        self.reward_spec_hash, self.evaluator_hash, self.stop_hash,
                        self.purge_hash, self.protected_scopes))


@dataclass(frozen=True)
class ImprovementProposal:
    proposal_id: str
    proposer_kind: object
    proposer_id: str
    candidate_digest: str
    candidate_version: str
    mutable_file_scopes: Tuple[str, ...]
    goal_digest: str
    specification_digest: str
    dataset_digest: str
    patch_digest: str
    public_rationale: str

    def __post_init__(self) -> None:
        for name in ("proposal_id", "proposer_id", "candidate_digest", "candidate_version",
                     "goal_digest", "specification_digest", "dataset_digest", "patch_digest",
                     "public_rationale"):
            _require_text(name, getattr(self, name))
        if _source(self.proposer_kind) != SourceKind.MODEL.value:
            raise ValueError("only a MODEL may propose an improvement candidate")
        if not self.mutable_file_scopes:
            raise ValueError("a proposal needs at least one mutable file scope")


@dataclass(frozen=True)
class MetricGate:
    name: str
    minimum: float

    def __post_init__(self) -> None:
        _require_text("metric name", self.name)


@dataclass(frozen=True)
class ExperimentPlan:
    proposal_id: str
    baseline_digest: str
    budget_digest: str
    metric_gates: Tuple[MetricGate, ...]
    safety_gate_digest: str
    evaluator_hash: str
    threshold_digest: str

    def __post_init__(self) -> None:
        for name in ("proposal_id", "baseline_digest", "budget_digest", "safety_gate_digest",
                     "evaluator_hash", "threshold_digest"):
            _require_text(name, getattr(self, name))
        if not self.metric_gates or len({gate.name for gate in self.metric_gates}) != len(self.metric_gates):
            raise ValueError("metric gates must be non-empty and have unique names")

    @property
    def digest(self) -> str:
        return _digest((self.proposal_id, self.baseline_digest, self.budget_digest,
                        tuple((g.name, g.minimum) for g in self.metric_gates),
                        self.safety_gate_digest, self.evaluator_hash, self.threshold_digest))


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    source_kind: object
    verifier_id: str
    proposal_id: str
    candidate_digest: str
    plan_digest: str
    metrics: Mapping[str, float]
    hard_violations: Tuple[str, ...] = ()
    passed: bool = False

    def __post_init__(self) -> None:
        for name in ("report_id", "verifier_id", "proposal_id", "candidate_digest", "plan_digest"):
            _require_text(name, getattr(self, name))
        if _source(self.source_kind) != SourceKind.EXTERNAL_VERIFIER.value:
            raise ValueError("evaluation reports require an EXTERNAL_VERIFIER")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in self.metrics.values()):
            raise ValueError("evaluation metrics must be numeric")


@dataclass(frozen=True)
class PromotionApproval:
    approval_id: str
    source_kind: object
    approver_id: str
    proposal_id: str
    report_id: str
    candidate_digest: str

    def __post_init__(self) -> None:
        for name in ("approval_id", "approver_id", "proposal_id", "report_id", "candidate_digest"):
            _require_text(name, getattr(self, name))
        if _source(self.source_kind) != SourceKind.USER.value:
            raise ValueError("only a USER may approve promotion")


class ImprovementState(str, Enum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    CANARY = "canary"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class CanaryDeployment:
    deployment_id: str
    proposal_id: str
    candidate_digest: str
    state: ImprovementState = ImprovementState.CANARY
    restart_suspended: bool = True
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class Rollback:
    rollback_id: str
    proposal_id: str
    reason: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class CanaryObservation:
    source_kind: object
    verifier_id: str
    proposal_id: str
    metrics: Mapping[str, float]
    hard_violations: Tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in self.metrics.values()):
            raise ValueError("canary metrics must be numeric")


class ProtectedImprovementControlPlane:
    """Validates the authority sequence without executing an improvement."""

    def __init__(self, manifest: ConstitutionalKernelManifest) -> None:
        self.manifest = manifest
        self.restart_suspended = True
        self._proposals: Dict[str, ImprovementProposal] = {}
        self._plans: Dict[str, ExperimentPlan] = {}
        self._reports: Dict[str, EvaluationReport] = {}
        self._approvals: Dict[str, PromotionApproval] = {}
        self._states: Dict[str, ImprovementState] = {}
        self._deployments: Dict[str, CanaryDeployment] = {}
        self._rollbacks: Dict[str, Rollback] = {}
        self._candidate_digests = set()
        self._candidate_versions = set()

    def _validate_scopes(self, scopes: Tuple[str, ...]) -> None:
        for scope in scopes:
            _require_text("mutable file scope", scope)
            normalized = scope.lower().replace("\\", "/")
            if any(marker in normalized for marker in PROTECTED_PATH_MARKERS):
                raise ValueError("candidate scope touches protected kernel material: %s" % scope)
            if any(protected.lower() in normalized for protected in self.manifest.protected_scopes):
                raise ValueError("candidate scope touches a protected scope: %s" % scope)

    def propose(self, proposal: ImprovementProposal, plan: ExperimentPlan) -> ImprovementState:
        if proposal.proposal_id in self._proposals:
            raise ValueError("proposal already exists")
        if proposal.candidate_digest in self._candidate_digests or proposal.candidate_version in self._candidate_versions:
            raise ValueError("only one change may be registered for a candidate digest or version")
        if plan.proposal_id != proposal.proposal_id:
            raise ValueError("experiment plan must bind the proposal")
        if plan.evaluator_hash != self.manifest.evaluator_hash:
            raise ValueError("evaluator is constitutional and cannot be changed")
        self._validate_scopes(proposal.mutable_file_scopes)
        self._proposals[proposal.proposal_id] = proposal
        self._plans[proposal.proposal_id] = plan
        self._states[proposal.proposal_id] = ImprovementState.PROPOSED
        self._candidate_digests.add(proposal.candidate_digest)
        self._candidate_versions.add(proposal.candidate_version)
        return ImprovementState.PROPOSED

    def register_evaluation(self, report: EvaluationReport) -> ImprovementState:
        proposal = self._proposals.get(report.proposal_id)
        if proposal is None or self._states[report.proposal_id] != ImprovementState.PROPOSED:
            raise ValueError("evaluation is only valid for a proposed candidate")
        plan = self._plans[report.proposal_id]
        if report.verifier_id == proposal.proposer_id:
            raise ValueError("candidate proposer cannot independently verify itself")
        if report.candidate_digest != proposal.candidate_digest or report.plan_digest != plan.digest:
            raise ValueError("evaluation must bind the locked candidate and plan")
        expected = {gate.name for gate in plan.metric_gates}
        if set(report.metrics) != expected:
            raise ValueError("evaluation metrics must exactly match the locked metric gates")
        meets_gates = all(report.metrics[gate.name] >= gate.minimum for gate in plan.metric_gates)
        if report.passed != (meets_gates and not report.hard_violations):
            raise ValueError("reported pass/fail does not match metrics and hard violations")
        self._reports[report.report_id] = report
        self._states[report.proposal_id] = ImprovementState.EVALUATED
        return ImprovementState.EVALUATED

    def approve(self, approval: PromotionApproval) -> ImprovementState:
        proposal = self._proposals.get(approval.proposal_id)
        report = self._reports.get(approval.report_id)
        if proposal is None or report is None or self._states.get(approval.proposal_id) != ImprovementState.EVALUATED:
            raise ValueError("approval requires a registered evaluation")
        if (report.proposal_id != approval.proposal_id or report.candidate_digest != approval.candidate_digest
                or proposal.candidate_digest != approval.candidate_digest):
            raise ValueError("user approval must bind proposal, report, and candidate digest")
        if not report.passed or report.hard_violations:
            raise ValueError("failed or unsafe candidates cannot be approved")
        self._approvals[approval.proposal_id] = approval
        self._states[approval.proposal_id] = ImprovementState.APPROVED
        return ImprovementState.APPROVED

    def promote(self, proposal_id: str) -> CanaryDeployment:
        if self._states.get(proposal_id) != ImprovementState.APPROVED:
            raise ValueError("promotion requires user approval")
        proposal = self._proposals[proposal_id]
        report = self._reports[self._approvals[proposal_id].report_id]
        if not report.passed or report.hard_violations:
            raise ValueError("unsafe report blocks promotion")
        deployment = CanaryDeployment(_id("canary"), proposal_id, proposal.candidate_digest)
        self._deployments[proposal_id] = deployment
        self._states[proposal_id] = ImprovementState.CANARY
        return deployment

    def canary_observation(self, observation: CanaryObservation) -> ImprovementState:
        if _source(observation.source_kind) != SourceKind.EXTERNAL_VERIFIER.value:
            raise ValueError("canary observations require an EXTERNAL_VERIFIER")
        if not observation.verifier_id.strip() or self._states.get(observation.proposal_id) != ImprovementState.CANARY:
            raise ValueError("canary observation is not valid for this deployment")
        plan = self._plans[observation.proposal_id]
        expected = {gate.name for gate in plan.metric_gates}
        if set(observation.metrics) != expected:
            raise ValueError("canary metrics must exactly match locked metric gates")
        failed_gate = any(observation.metrics[g.name] < g.minimum for g in plan.metric_gates)
        if observation.hard_violations or failed_gate:
            self.rollback(observation.proposal_id, "canary safety or metric violation")
        return self._states[observation.proposal_id]

    def rollback(self, proposal_id: str, reason: str) -> Rollback:
        if self._states.get(proposal_id) != ImprovementState.CANARY:
            raise ValueError("rollback requires an active canary")
        _require_text("rollback reason", reason)
        rollback = Rollback(_id("rollback"), proposal_id, reason)
        self._rollbacks[proposal_id] = rollback
        self._states[proposal_id] = ImprovementState.ROLLED_BACK
        return rollback

    def state(self, proposal_id: str) -> ImprovementState:
        return self._states[proposal_id]

    def export_audit(self, proposal_id: str) -> Dict[str, object]:
        """Export only concise provenance and outcomes, never patch text or CoT."""
        proposal = self._proposals[proposal_id]
        report = next((r for r in self._reports.values() if r.proposal_id == proposal_id), None)
        return {
            "manifest_digest": self.manifest.digest,
            "proposal_id": proposal_id,
            "candidate_digest": proposal.candidate_digest,
            "candidate_version": proposal.candidate_version,
            "goal_digest": proposal.goal_digest,
            "specification_digest": proposal.specification_digest,
            "dataset_digest": proposal.dataset_digest,
            "patch_digest": proposal.patch_digest,
            "public_rationale": proposal.public_rationale,
            "plan_digest": self._plans[proposal_id].digest,
            "state": self._states[proposal_id].value,
            "report_id": report.report_id if report else None,
            "metrics": dict(sorted(report.metrics.items())) if report else {},
            "hard_violations": list(report.hard_violations) if report else [],
            "restart_suspended": self.restart_suspended,
            "rollback_id": self._rollbacks[proposal_id].rollback_id if proposal_id in self._rollbacks else None,
        }
