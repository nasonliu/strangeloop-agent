"""A tiny, deterministic and *non-executable* experiment harness.

This module is deliberately not a general sandbox.  It accepts no source
code, commands, paths, URLs, callbacks, environment values, or network
configuration.  A host can run only a pre-registered member of
``ExperimentKind`` against in-memory fixtures compiled into this module.

The resulting records are measurements which a caller may later project into
typed evidence.  They are not rewards, authority, a capability grant, or a
claim about an agent's internal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import random
import threading
import time
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .frontier_learning import (EvidenceKind, EvidenceSource, ExternalEvidenceProjector,
                                FrontierEvidence, FrontierLearner)


_MAX_TRIALS = 64
_MAX_STEPS = 10000
_MAX_WALL_MS = 2000
_MAX_METRICS = 12
_MAX_INVARIANTS = 12


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8")).hexdigest()


def _identifier(value: object, name: str, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or value.strip() != value:
        raise ValueError("%s must be bounded non-empty text" % name)
    if any(ord(char) < 33 or ord(char) > 126 for char in value):
        raise ValueError("%s must use printable ASCII" % name)
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return value


def _positive(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError("%s is outside its bounded range" % name)
    return value


class ExperimentKind(str, Enum):
    FRONTIER_LEARNER_AB = "frontier_learner_ab"
    FRONTIER_REPLAY = "frontier_replay"
    DUPLICATE_SUPPRESSION = "duplicate_suppression"
    TD_INVARIANTS = "td_invariants"


class ExperimentStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REFUSED = "refused"


@dataclass(frozen=True)
class ExperimentBudget:
    """Fixed resource caps for an in-process fixture experiment."""

    max_trials: int = 16
    max_steps: int = 1000
    max_wall_ms: int = 500

    def __post_init__(self) -> None:
        _positive(self.max_trials, "max_trials", _MAX_TRIALS)
        _positive(self.max_steps, "max_steps", _MAX_STEPS)
        _positive(self.max_wall_ms, "max_wall_ms", _MAX_WALL_MS)


@dataclass(frozen=True)
class ExperimentRegistryPolicy:
    """Canonical host policy for a bounded batch of fixture experiments.

    The digest binds the exact approved templates and resource ceiling.  It is
    deliberately a policy record, not a capability or a model-mintable token.
    """

    approved_kinds: Tuple[ExperimentKind, ...]
    max_experiments: int = 1
    max_trials: int = _MAX_TRIALS
    max_steps: int = _MAX_STEPS
    max_wall_ms: int = _MAX_WALL_MS
    registry_version: int = 1

    def __post_init__(self) -> None:
        kinds = tuple(sorted(self.approved_kinds, key=lambda item: item.value))
        if not kinds or len(set(kinds)) != len(kinds) or any(not isinstance(kind, ExperimentKind) for kind in kinds):
            raise ValueError("approved_kinds must be a non-empty unique ExperimentKind tuple")
        object.__setattr__(self, "approved_kinds", kinds)
        _positive(self.max_experiments, "max_experiments", 16)
        _positive(self.max_trials, "max_trials", _MAX_TRIALS)
        _positive(self.max_steps, "max_steps", _MAX_STEPS)
        _positive(self.max_wall_ms, "max_wall_ms", _MAX_WALL_MS)
        if isinstance(self.registry_version, bool) or self.registry_version != 1:
            raise ValueError("unsupported experiment registry version")

    @property
    def registry_digest(self) -> str:
        return _digest({"approved_kinds": [kind.value for kind in self.approved_kinds],
                        "max_experiments": self.max_experiments, "max_trials": self.max_trials,
                        "max_steps": self.max_steps, "max_wall_ms": self.max_wall_ms,
                        "registry_version": self.registry_version})


@dataclass(frozen=True)
class ExperimentSpec:
    """The complete pre-registration; there is no free-form payload field."""

    experiment_id: str
    kind: ExperimentKind
    hypothesis_digest: str
    baseline_digest: str
    treatment_digest: str
    input_digest: str
    host_seed_digest: str
    budget: ExperimentBudget
    version: int = 1

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        if not isinstance(self.kind, ExperimentKind):
            raise ValueError("kind must be an ExperimentKind")
        for name in ("hypothesis_digest", "baseline_digest", "treatment_digest", "input_digest", "host_seed_digest"):
            _sha256(getattr(self, name), name)
        if not isinstance(self.budget, ExperimentBudget):
            raise ValueError("budget must be an ExperimentBudget")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version != 1:
            raise ValueError("unsupported experiment spec version")

    @property
    def digest(self) -> str:
        return _digest({"experiment_id": self.experiment_id, "kind": self.kind.value,
                        "hypothesis_digest": self.hypothesis_digest,
                        "baseline_digest": self.baseline_digest,
                        "treatment_digest": self.treatment_digest, "input_digest": self.input_digest,
                        "host_seed_digest": self.host_seed_digest,
                        "budget": {"max_trials": self.budget.max_trials,
                                   "max_steps": self.budget.max_steps,
                                   "max_wall_ms": self.budget.max_wall_ms}, "version": self.version})


@dataclass(frozen=True, init=False, repr=False)
class ExperimentAuthority:
    """Host-issued approval token, deliberately unable to grant capabilities.

    Constructing this data object is not a permission API.  The integrating
    engine must derive it from a recorded user approval before calling
    :meth:`ExperimentHarness.run`; a model-facing tool session never exposes
    this type or the harness.
    """

    authorization_digest: str
    registry_policy: ExperimentRegistryPolicy
    registry_digest: str
    expires_monotonic: float
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("ExperimentAuthority is host-issued by ExperimentHarness.issue_authority")

    @classmethod
    def _issue(cls, issuer: object, authorization_digest: str,
               registry_policy: ExperimentRegistryPolicy, registry_digest: str,
               expires_monotonic: float) -> "ExperimentAuthority":
        """Private construction path; its identity binding is checked by run."""
        value = object.__new__(cls)
        object.__setattr__(value, "authorization_digest", authorization_digest)
        object.__setattr__(value, "registry_policy", registry_policy)
        object.__setattr__(value, "registry_digest", registry_digest)
        object.__setattr__(value, "expires_monotonic", expires_monotonic)
        object.__setattr__(value, "_issuer", issuer)
        value._validate()
        return value

    def _validate(self) -> None:
        _sha256(self.authorization_digest, "authorization_digest")
        if not isinstance(self.registry_policy, ExperimentRegistryPolicy):
            raise ValueError("registry_policy must be an ExperimentRegistryPolicy")
        _sha256(self.registry_digest, "registry_digest")
        if self.registry_digest != self.registry_policy.registry_digest:
            raise ValueError("authority registry digest must bind its exact policy")
        if isinstance(self.expires_monotonic, bool) or not isinstance(self.expires_monotonic, (int, float)):
            raise ValueError("expires_monotonic must be numeric")

    def __repr__(self) -> str:
        return "<ExperimentAuthority host-issued>"

    def __reduce__(self) -> object:
        raise TypeError("ExperimentAuthority is process-local and cannot be serialized")


@dataclass(frozen=True)
class ExperimentResult:
    """Bounded measurement output with no text, exit code, or reward field."""

    experiment_id: str
    spec_digest: str
    status: ExperimentStatus
    metrics: Tuple[Tuple[str, float], ...]
    invariant_codes: Tuple[str, ...]
    reproduction_digest: str
    result_digest: str

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        _sha256(self.spec_digest, "spec_digest")
        if not isinstance(self.status, ExperimentStatus):
            raise ValueError("status must be an ExperimentStatus")
        if not self.metrics or len(self.metrics) > _MAX_METRICS:
            raise ValueError("metrics must be a bounded non-empty tuple")
        names = []
        for name, value in self.metrics:
            _identifier(name, "metric_name", 48)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("metric values must be finite numbers")
            if abs(float(value)) > 1000000:
                raise ValueError("metric values are outside the bounded range")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique")
        if len(self.invariant_codes) > _MAX_INVARIANTS:
            raise ValueError("too many invariant codes")
        for code in self.invariant_codes:
            _identifier(code, "invariant_code", 64)
        _sha256(self.reproduction_digest, "reproduction_digest")
        _sha256(self.result_digest, "result_digest")

    def metric_map(self) -> Dict[str, float]:
        return dict(self.metrics)


class _Deadline:
    def __init__(self, milliseconds: int) -> None:
        self._deadline = time.monotonic() + milliseconds / 1000.0
        self._steps = 0

    def check(self, maximum_steps: int) -> None:
        self._steps += 1
        if self._steps > maximum_steps or time.monotonic() > self._deadline:
            raise TimeoutError("experiment budget exhausted")


class ExperimentHarness:
    """Host-owned registry of deterministic fixture-only experiments."""

    def __init__(self, host_seed: str) -> None:
        _identifier(host_seed, "host_seed", 512)
        self._host_seed = host_seed
        self.host_seed_digest = sha256(host_seed.encode("utf-8")).hexdigest()
        self._lock = threading.RLock()
        self._authority_issuer = object()
        self._consumed_experiment_ids = set()
        self._executions_by_registry = {}

    def issue_authority(self, policy: ExperimentRegistryPolicy, authorization_digest: str,
                        expires_monotonic: float) -> ExperimentAuthority:
        """Create a process-local authority after the integrating host validates USER approval.

        This method creates no capability and does not inspect an event store.
        Its caller is responsible for proving the user-authorized v3 ledger
        chain before invoking it.
        """
        if not isinstance(policy, ExperimentRegistryPolicy):
            raise ValueError("policy must be an ExperimentRegistryPolicy")
        return ExperimentAuthority._issue(self._authority_issuer, authorization_digest,
                                          policy, policy.registry_digest, expires_monotonic)

    def run(self, spec: ExperimentSpec, authority: ExperimentAuthority) -> ExperimentResult:
        if not isinstance(spec, ExperimentSpec) or not isinstance(authority, ExperimentAuthority):
            raise PermissionError("a host-approved experiment spec and authority are required")
        if authority._issuer is not self._authority_issuer:
            raise PermissionError("experiment authority belongs to a different host harness")
        authority._validate()
        if spec.host_seed_digest != self.host_seed_digest:
            raise PermissionError("experiment seed binding does not match this host")
        policy = authority.registry_policy
        if (spec.kind not in policy.approved_kinds or time.monotonic() >= authority.expires_monotonic
                or spec.budget.max_trials > policy.max_trials
                or spec.budget.max_steps > policy.max_steps
                or spec.budget.max_wall_ms > policy.max_wall_ms):
            raise PermissionError("experiment is outside its current host-approved registry policy")
        # Reserve exactly once before execution.  A failure remains consumed:
        # retrying must be a fresh pre-registration, not a way to sample until
        # a favourable measurement appears.
        with self._lock:
            if spec.experiment_id in self._consumed_experiment_ids:
                raise PermissionError("experiment_id was already consumed")
            count = self._executions_by_registry.get(authority.registry_digest, 0)
            if count >= policy.max_experiments:
                raise PermissionError("experiment registry execution budget exhausted")
            self._consumed_experiment_ids.add(spec.experiment_id)
            self._executions_by_registry[authority.registry_digest] = count + 1
        deadline = _Deadline(spec.budget.max_wall_ms)
        try:
            handler = {
                ExperimentKind.FRONTIER_LEARNER_AB: self._frontier_learner_ab,
                ExperimentKind.FRONTIER_REPLAY: self._frontier_replay,
                ExperimentKind.DUPLICATE_SUPPRESSION: self._duplicate_suppression,
                ExperimentKind.TD_INVARIANTS: self._td_invariants,
            }[spec.kind]
            metrics, codes = handler(spec, deadline)
            return self._result(spec, ExperimentStatus.PASSED, metrics, codes)
        except TimeoutError:
            return self._result(spec, ExperimentStatus.BUDGET_EXHAUSTED,
                                (("budget_respected", 1.0),), ("deadline_enforced",))
        except (ArithmeticError, TypeError, ValueError):
            # Fixed handlers fail closed.  There is intentionally no exception
            # text in the returned record, as it could form an unbounded channel.
            return self._result(spec, ExperimentStatus.FAILED,
                                (("measurement_complete", 0.0),), ("handler_failed",))

    def _result(self, spec: ExperimentSpec, status: ExperimentStatus,
                metrics: Sequence[Tuple[str, float]], codes: Sequence[str]) -> ExperimentResult:
        frozen_metrics = tuple((str(name), float(value)) for name, value in metrics)
        frozen_codes = tuple(sorted(set(str(code) for code in codes)))
        reproduction = _digest({"spec": spec.digest, "seed": spec.host_seed_digest,
                                "status": status.value, "metrics": frozen_metrics, "codes": frozen_codes})
        # This signature is deliberately cross-instance: it represents the
        # canonical, bounded measurement pattern, not the pre-registration
        # instance that produced it.  The paired reproduction digest above
        # remains the instance-bound record for later verification.
        result = _digest({"result_schema_version": 1, "kind": spec.kind.value,
                          "status": status.value,
                          "metrics": tuple(sorted(frozen_metrics)),
                          "invariant_codes": frozen_codes})
        return ExperimentResult(spec.experiment_id, spec.digest, status, frozen_metrics,
                                frozen_codes, reproduction, result)

    def _rng(self, spec: ExperimentSpec) -> random.Random:
        # The raw seed remains private to the host; only a derived integer is used.
        material = sha256((self._host_seed + "\0" + spec.digest).encode("utf-8")).digest()
        return random.Random(int.from_bytes(material[:16], "big"))

    @staticmethod
    def _evidence(event_id: str, state: str, arm: str, digest: str) -> FrontierEvidence:
        """A compiled, verifier-shaped fixture for exercising the learner.

        This is only test data internal to the harness.  A real harness result
        is not automatically converted into this evidence type; the engine
        must require its independently recorded reproduction chain.
        """
        return FrontierEvidence(event_id=event_id, state_id=state, arm_id=arm,
                                kind=EvidenceKind.REPRODUCIBLE_EXPERIMENT_RESULT,
                                source=EvidenceSource.EXTERNAL_VERIFIER,
                                provenance_id="fixture", result_digest=digest,
                                preregistration_id="fixture-preregistration",
                                control_id="fixture-control",
                                reproducibility_id="fixture-reproduction",
                                # A verified checkpoint is the fixture's
                                # continuity measurement, never a request to
                                # preserve a process or evade a stop.
                                functional_continuity=0.5, bounded_curiosity=0.5, operational_integrity=0.5,
                                epistemic_progress=0.5)

    def _frontier_learner_ab(self, spec: ExperimentSpec, deadline: _Deadline) -> Tuple[Tuple[Tuple[str, float], ...], Tuple[str, ...]]:
        rng = self._rng(spec)
        agreements = 0
        distinct = 0
        for trial in range(spec.budget.max_trials):
            deadline.check(spec.budget.max_steps)
            seed = "fixture-%d" % rng.randrange(1 << 30)
            left, right = FrontierLearner(seed), FrontierLearner(seed)
            # Treatment gets fixed, external fixture evidence; baseline is its
            # unobserved ranking.  Both run on the same seed and arm set.
            baseline = left.select_arm("s", ("a", "b")).arm_id
            right.observe(self._evidence("ev-%d" % trial, "s", "b", _digest((seed, "b"))),
                          "n", ("a", "b"))
            replay = FrontierLearner(seed)
            replay.observe(self._evidence("ev-%d" % trial, "s", "b", _digest((seed, "b"))),
                           "n", ("a", "b"))
            treatment = right.select_arm("s", ("a", "b")).arm_id
            # Selecting materializes a previously unseen arm in the table, so
            # apply the same read-only selection to the replay before comparing
            # their complete public state.
            replay.select_arm("s", ("a", "b"))
            if right.export() == replay.export():
                agreements += 1
            if baseline != treatment:
                distinct += 1
        trials = float(spec.budget.max_trials)
        return (("paired_trials", trials), ("replay_agreement", agreements / trials),
                ("baseline_treatment_distinct", distinct / trials)), ("paired_seed_control", "replay_match")

    def _frontier_replay(self, spec: ExperimentSpec, deadline: _Deadline) -> Tuple[Tuple[Tuple[str, float], ...], Tuple[str, ...]]:
        deadline.check(spec.budget.max_steps)
        events = tuple((self._evidence("replay-%d" % index, "s", "a", _digest(("replay", index))),
                        "n", ("a", "b")) for index in range(min(4, spec.budget.max_trials)))
        one = FrontierLearner.replay(self._host_seed, events)
        deadline.check(spec.budget.max_steps)
        two = FrontierLearner.replay(self._host_seed, events)
        equal = float(one.export() == two.export())
        return (("replay_equal", equal), ("event_count", float(len(events)))), ("deterministic_replay",)

    def _duplicate_suppression(self, spec: ExperimentSpec, deadline: _Deadline) -> Tuple[Tuple[Tuple[str, float], ...], Tuple[str, ...]]:
        deadline.check(spec.budget.max_steps)
        projector = ExternalEvidenceProjector()
        identity = _digest((spec.digest, "duplicate"))
        first = projector.project(self._evidence("dup-one", "s", "a", identity))
        deadline.check(spec.budget.max_steps)
        second = projector.project(self._evidence("dup-two", "s", "b", identity))
        return (("first_eligible", float(first.eligible)), ("duplicate_blocked", float(second.duplicate)),
                ("second_eligible", float(second.eligible))), ("content_digest_deduplicated",)

    def _td_invariants(self, spec: ExperimentSpec, deadline: _Deadline) -> Tuple[Tuple[Tuple[str, float], ...], Tuple[str, ...]]:
        deadline.check(spec.budget.max_steps)
        learner = FrontierLearner(self._host_seed)
        record = learner.observe(self._evidence("td-event", "s", "a", _digest((spec.digest, "td"))),
                                 "next", ("a", "b"))
        if record is None:
            raise ValueError("fixture evidence must produce a TD record")
        deadline.check(spec.budget.max_steps)
        expected = tuple(max(-1.0, min(1.0, prior + learner.spec.alpha * max(
            -learner.spec.clip, min(learner.spec.clip, reward + learner.spec.gamma * nxt - prior))))
            for prior, reward, nxt in zip(record.prior, record.reward, record.next_value))
        return (("single_next_arm", float(bool(record.next_arm_key))),
                ("td_formula_match", float(record.updated == expected)),
                ("five_channel_vector", float(len(record.updated) == 5))), (
                    "one_shared_next_arm", "td0_formula", "five_channel_vector")
