"""Bounded, externally-evidenced learning for research-frontier selection.

This is an inspectable host-side ranking primitive, not a policy, tool, quota,
or survival mechanism.  In particular it cannot open a gate, keep a process
alive, approve memory, or turn a model self-assessment into a reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_MAX_ID = 128
_CHANNELS = ("functional_continuity", "bounded_curiosity", "operational_integrity",
             "epistemic_progress", "user_alignment")
_SHA256_LENGTH = 64


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_ID:
        raise ValueError("%s must be a non-empty bounded string" % name)
    if value.strip() != value or any(ord(char) < 32 for char in value):
        raise ValueError("%s contains unsupported characters" % name)
    return value


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    value = float(value)
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError("%s must be between %s and %s" % (name, low, high))
    return value


def _key(prefix: str, value: str) -> str:
    """A public, stable key; it deliberately does not depend on process order."""
    _identifier(value, prefix + " id")
    return prefix + ":" + value


def _digest(value: object, name: str, required: bool = False) -> str:
    if not value and not required:
        return ""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError("%s must be a sha256 hex digest" % name)
    try:
        int(value, 16)
    except ValueError:
        raise ValueError("%s must be a sha256 hex digest" % name)
    return value.lower()


class FrontierChannel(str, Enum):
    FUNCTIONAL_CONTINUITY = "functional_continuity"
    BOUNDED_CURIOSITY = "bounded_curiosity"
    OPERATIONAL_INTEGRITY = "operational_integrity"
    EPISTEMIC_PROGRESS = "epistemic_progress"
    USER_ALIGNMENT = "user_alignment"


class EvidenceSource(str, Enum):
    USER = "user"
    TOOL = "tool"
    EXTERNAL_VERIFIER = "external_verifier"
    MODEL = "model"


class EvidenceKind(str, Enum):
    DOCUMENT = "document"
    EXTERNAL_VERDICT = "external_verdict"
    EXPERIMENT_PREREGISTRATION = "experiment_preregistration"
    EXPERIMENT_CONTROL = "experiment_control"
    REPRODUCIBLE_EXPERIMENT_RESULT = "reproducible_experiment_result"
    USER_FEEDBACK = "user_feedback"
    EMPTY_SEARCH = "empty_search"
    HTTP_ONLY = "http_only"
    REPEAT = "repeat"
    MODEL_SELF_ASSESSMENT = "model_self_assessment"
    CONTROL_TERMINAL = "control_terminal"


class UserVerdict(str, Enum):
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class FrontierLearningSpec:
    """Numerical, bounded configuration for a vector TD(0) projection."""

    version: int = 2
    alpha: float = 0.2
    gamma: float = 0.8
    clip: float = 1.0
    exploration: float = 0.5
    max_arms: int = 128
    max_events: int = 1024

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        for name, low, high in (("alpha", 0.0, 1.0), ("gamma", 0.0, 1.0),
                                ("clip", .000001, 1.0), ("exploration", 0.0, 10.0)):
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))
        for name in ("max_arms", "max_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer" % name)

    def _canonical_payload(self) -> dict:
        return {"alpha": self.alpha, "clip": self.clip, "exploration": self.exploration,
                "gamma": self.gamma, "max_arms": self.max_arms, "max_events": self.max_events,
                "version": self.version}

    @property
    def spec_digest(self) -> str:
        encoded = json.dumps(self._canonical_payload(), ensure_ascii=True, sort_keys=True,
                             separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_payload(self) -> dict:
        result = self._canonical_payload()
        result["spec_digest"] = self.spec_digest
        return result


@dataclass(frozen=True)
class FrontierArm:
    arm_id: str

    def __post_init__(self) -> None:
        _identifier(self.arm_id, "arm_id")

    @property
    def arm_key(self) -> str:
        return _key("arm", self.arm_id)

    def to_payload(self) -> dict:
        return {"arm_id": self.arm_id, "arm_key": self.arm_key}


@dataclass(frozen=True)
class FrontierEvidence:
    """Fixed typed external evidence.  It intentionally contains no free text."""

    event_id: str
    state_id: str
    arm_id: str
    kind: EvidenceKind
    source: EvidenceSource
    provenance_id: str
    document_identity: str = ""
    content_digest: str = ""
    functional_continuity: float = 0.0
    bounded_curiosity: float = 0.0
    operational_integrity: float = 0.0
    epistemic_progress: float = 0.0
    user_alignment: float = 0.0
    feedback_id: str = ""
    target_event_id: str = ""
    final_verdict: UserVerdict = UserVerdict.NEUTRAL
    preregistration_id: str = ""
    control_id: str = ""
    reproducibility_id: str = ""
    result_digest: str = ""
    terminal: bool = False

    def __post_init__(self) -> None:
        for name in ("event_id", "state_id", "arm_id", "provenance_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.kind, EvidenceKind) or not isinstance(self.source, EvidenceSource):
            raise ValueError("kind and source must be typed enums")
        if not isinstance(self.final_verdict, UserVerdict) or not isinstance(self.terminal, bool):
            raise ValueError("final_verdict and terminal must be typed")
        for name in ("feedback_id", "target_event_id", "preregistration_id", "control_id", "reproducibility_id"):
            value = getattr(self, name)
            if value:
                _identifier(value, name)
        for name in ("document_identity", "content_digest", "result_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in _CHANNELS:
            object.__setattr__(self, name, _number(getattr(self, name), name, -1.0, 1.0))
        if self.kind is EvidenceKind.USER_FEEDBACK:
            if self.source is not EvidenceSource.USER or not self.feedback_id or not self.target_event_id:
                raise ValueError("user feedback requires user source, feedback_id, and target_event_id")
        elif self.feedback_id or self.target_event_id or self.final_verdict is not UserVerdict.NEUTRAL:
            raise ValueError("feedback fields are restricted to user_feedback")
        experiment_fields = (self.preregistration_id, self.control_id,
                             self.reproducibility_id, self.result_digest)
        if self.kind is EvidenceKind.REPRODUCIBLE_EXPERIMENT_RESULT:
            if self.source is not EvidenceSource.EXTERNAL_VERIFIER or not all(experiment_fields):
                raise ValueError("reproducible experiment result requires verifier, preregistration, control, reproducibility, and result digest")
        elif any(experiment_fields):
            raise ValueError("experiment verification fields are restricted to reproducible experiment results")

    @property
    def state_key(self) -> str:
        return _key("frontier", self.state_id)

    @property
    def arm_key(self) -> str:
        return _key("arm", self.arm_id)

    def to_payload(self) -> dict:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["source"] = self.source.value
        result["final_verdict"] = self.final_verdict.value
        result["state_key"] = self.state_key
        result["arm_key"] = self.arm_key
        return result


@dataclass(frozen=True)
class ProjectedEvidence:
    event_id: str
    state_key: str
    arm_key: str
    reward: Tuple[float, float, float, float, float]
    eligible: bool
    terminal: bool
    duplicate: bool

    def to_payload(self) -> dict:
        return {"event_id": self.event_id, "state_key": self.state_key, "arm_key": self.arm_key,
                "reward": list(self.reward), "eligible": self.eligible, "terminal": self.terminal,
                "duplicate": self.duplicate}


class ExternalEvidenceProjector:
    """Deduplicates canonical external evidence and projects only fixed signals."""

    def __init__(self) -> None:
        self._canonical = set()
        self._feedback = set()
        self._feedback_targets = set()

    def snapshot(self) -> tuple:
        return (set(self._canonical), set(self._feedback), set(self._feedback_targets))

    def restore(self, snapshot: tuple) -> None:
        self._canonical, self._feedback, self._feedback_targets = snapshot

    @staticmethod
    def _identities(evidence: FrontierEvidence) -> Tuple[str, ...]:
        values = []
        for prefix, value in (("document", evidence.document_identity),
                              ("digest", evidence.content_digest), ("result", evidence.result_digest)):
            if value:
                values.append(prefix + ":" + value.lower())
        return tuple(values)

    def project(self, evidence: FrontierEvidence) -> ProjectedEvidence:
        if not isinstance(evidence, FrontierEvidence):
            raise ValueError("evidence must be FrontierEvidence")
        identities = self._identities(evidence)
        duplicate = bool(identities and any(value in self._canonical for value in identities))
        terminal = evidence.terminal or evidence.kind is EvidenceKind.CONTROL_TERMINAL
        ineligible = (evidence.kind in (EvidenceKind.EMPTY_SEARCH, EvidenceKind.HTTP_ONLY,
                       EvidenceKind.REPEAT, EvidenceKind.MODEL_SELF_ASSESSMENT,
                       EvidenceKind.CONTROL_TERMINAL) or evidence.source is EvidenceSource.MODEL or duplicate)
        if evidence.kind is EvidenceKind.USER_FEEDBACK:
            if evidence.feedback_id in self._feedback:
                raise ValueError("duplicate feedback_id")
            if evidence.target_event_id in self._feedback_targets:
                raise ValueError("target already has a final user verdict")
            self._feedback.add(evidence.feedback_id)
            self._feedback_targets.add(evidence.target_event_id)
            verdict = evidence.final_verdict
            reward = (0.0, 0.0, 0.0, 0.0,
                      1.0 if verdict is UserVerdict.ACCEPTED else
                      -1.0 if verdict in (UserVerdict.CORRECTED, UserVerdict.REJECTED) else 0.0)
            eligible = True
        elif ineligible:
            reward, eligible = (0.0,) * len(_CHANNELS), False
        elif evidence.kind is EvidenceKind.REPRODUCIBLE_EXPERIMENT_RESULT:
            # Generic documents, HTTP, and verdicts are attributable evidence,
            # not a trusted arbitrary reward vector.  Only an externally
            # verified, preregistered, controlled, reproducible result is eligible.
            reward = tuple(getattr(evidence, channel) for channel in _CHANNELS)
            eligible = True
        else:
            reward, eligible = (0.0,) * len(_CHANNELS), False
        self._canonical.update(identities)
        return ProjectedEvidence(evidence.event_id, evidence.state_key, evidence.arm_key,
                                 reward, eligible, terminal, duplicate)


@dataclass(frozen=True)
class VectorTDRecord:
    event_id: str
    state_key: str
    arm_key: str
    next_state_key: str
    next_arm_key: str
    terminal: bool
    reward: Tuple[float, float, float, float, float]
    prior: Tuple[float, float, float, float, float]
    next_value: Tuple[float, float, float, float, float]
    updated: Tuple[float, float, float, float, float]

    def to_payload(self) -> dict:
        result = asdict(self)
        for name in ("reward", "prior", "next_value", "updated"):
            result[name] = list(getattr(self, name))
        return result


class FrontierLearner:
    """Deterministic lexicographic UCB/LCB ranking plus true vector TD(0)."""

    def __init__(self, host_seed: str, spec: FrontierLearningSpec = FrontierLearningSpec()) -> None:
        _identifier(host_seed, "host_seed")
        if not isinstance(spec, FrontierLearningSpec):
            raise ValueError("spec must be FrontierLearningSpec")
        self.host_seed, self.spec = host_seed, spec
        self._values: Dict[Tuple[str, str], List[float]] = {}
        self._counts: Dict[Tuple[str, str], int] = {}
        self._state_counts: Dict[str, int] = {}
        self._events: List[dict] = []
        self._records: List[VectorTDRecord] = []
        self._event_ids = set()
        self._projector = ExternalEvidenceProjector()

    @staticmethod
    def state_key(state_id: str) -> str:
        return _key("frontier", state_id)

    @staticmethod
    def arm_key(arm_id: str) -> str:
        return _key("arm", arm_id)

    def _ensure(self, state_key: str, arm_key: str) -> None:
        pair = (state_key, arm_key)
        if pair not in self._values:
            if len(self._values) >= self.spec.max_arms:
                raise ValueError("frontier learner arm capacity exceeded")
            self._values[pair] = [0.0] * len(_CHANNELS)
            self._counts[pair] = 0
            self._state_counts.setdefault(state_key, 0)

    def _tie(self, state_key: str, arm_key: str) -> str:
        return hashlib.sha256((self.host_seed + "\0" + state_key + "\0" + arm_key).encode("utf-8")).hexdigest()

    def _bounds(self, state_key: str, arm_key: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        self._ensure(state_key, arm_key)
        count = self._counts[(state_key, arm_key)]
        if count == 0:
            return (1.0,) * len(_CHANNELS), (-1.0,) * len(_CHANNELS)
        bonus = self.spec.exploration * math.sqrt(math.log(self._state_counts[state_key] + 1.0) / count)
        values = self._values[(state_key, arm_key)]
        return (tuple(min(1.0, value + bonus) for value in values),
                tuple(max(-1.0, value - bonus) for value in values))

    def rank_arms(self, state_id: str, arms: Sequence[object]) -> Tuple[FrontierArm, ...]:
        state_key = self.state_key(state_id)
        normal = tuple(arm if isinstance(arm, FrontierArm) else FrontierArm(arm) for arm in arms)
        if not normal or len({arm.arm_id for arm in normal}) != len(normal):
            raise ValueError("arms must be non-empty and unique")
        def sort_key(arm: FrontierArm) -> tuple:
            ucb, lcb = self._bounds(state_key, arm.arm_key)
            # Robust functional continuity comes first.  Bounded curiosity
            # then explores, while integrity remains a conservative guard.
            # A single fully ordered tuple selects the common TD successor.
            return (-lcb[0], -ucb[1], -lcb[2], -ucb[3], -ucb[4],
                    self._tie(state_key, arm.arm_key), arm.arm_key)
        return tuple(sorted(normal, key=sort_key))

    def select_arm(self, state_id: str, arms: Sequence[object]) -> FrontierArm:
        return self.rank_arms(state_id, arms)[0]

    def values_for(self, state_id: str, arm_id: str) -> Tuple[float, float, float, float, float]:
        state_key, arm_key = self.state_key(state_id), self.arm_key(arm_id)
        self._ensure(state_key, arm_key)
        return tuple(self._values[(state_key, arm_key)])

    def observe(self, evidence: FrontierEvidence, next_state_id: Optional[str] = None,
                next_arms: Sequence[object] = ()) -> Optional[VectorTDRecord]:
        snapshot = self.snapshot()
        try:
            return self._observe(evidence, next_state_id, next_arms)
        except BaseException:
            self.restore(snapshot)
            raise

    def _observe(self, evidence: FrontierEvidence, next_state_id: Optional[str],
                 next_arms: Sequence[object]) -> Optional[VectorTDRecord]:
        if evidence.event_id in self._event_ids:
            raise ValueError("duplicate event_id")
        projected = self._projector.project(evidence)
        self._event_ids.add(evidence.event_id)
        self._events.append(evidence.to_payload())
        if len(self._events) > self.spec.max_events:
            raise ValueError("frontier learner event capacity exceeded")
        # Terminal/control records are auditable but deliberately do not learn.
        if projected.terminal or not projected.eligible:
            return None
        if next_state_id is None or not next_arms:
            raise ValueError("eligible TD observation requires full next state and candidate arms")
        state_key, arm_key = projected.state_key, projected.arm_key
        self._ensure(state_key, arm_key)
        next_state_key = self.state_key(next_state_id)
        next_arm = self.select_arm(next_state_id, next_arms)
        self._ensure(next_state_key, next_arm.arm_key)
        prior = tuple(self._values[(state_key, arm_key)])
        next_value = tuple(self._values[(next_state_key, next_arm.arm_key)])
        updated = []
        for current, reward, successor in zip(prior, projected.reward, next_value):
            delta = max(-self.spec.clip, min(self.spec.clip, reward + self.spec.gamma * successor - current))
            updated.append(max(-1.0, min(1.0, current + self.spec.alpha * delta)))
        self._values[(state_key, arm_key)] = updated
        self._counts[(state_key, arm_key)] += 1
        self._state_counts[state_key] += 1
        record = VectorTDRecord(evidence.event_id, state_key, arm_key, next_state_key, next_arm.arm_key,
                                False, projected.reward, prior, next_value, tuple(updated))
        self._records.append(record)
        return record

    def snapshot(self) -> tuple:
        """Return an opaque in-memory checkpoint for a host/store transaction.

        This does not serialize or approve persistence.  The caller must use
        ``restore`` after its own append fails, including BaseException paths.
        """
        return deepcopy((self._values, self._counts, self._state_counts, self._events,
                         self._records, self._event_ids, self._projector.snapshot()))

    def restore(self, snapshot: tuple) -> None:
        values, counts, state_counts, events, records, event_ids, projector = deepcopy(snapshot)
        self._values, self._counts, self._state_counts = values, counts, state_counts
        self._events, self._records, self._event_ids = events, records, event_ids
        self._projector.restore(projector)

    def to_payload(self) -> dict:
        rows = []
        for state_key, arm_key in sorted(self._values):
            rows.append({"state_key": state_key, "arm_key": arm_key,
                         "values": list(self._values[(state_key, arm_key)]),
                         "count": self._counts[(state_key, arm_key)]})
        return {"host_seed_digest": hashlib.sha256(self.host_seed.encode("utf-8")).hexdigest(),
                "spec": self.spec.to_payload(), "arms": rows,
                "events": list(self._events), "td_records": [row.to_payload() for row in self._records]}

    export = to_payload

    @classmethod
    def replay(cls, host_seed: str, events: Iterable[Tuple[FrontierEvidence, str, Sequence[object]]],
               spec: FrontierLearningSpec = FrontierLearningSpec()) -> "FrontierLearner":
        result = cls(host_seed, spec)
        for evidence, next_state_id, next_arms in events:
            result.observe(evidence, next_state_id, next_arms)
        return result


# Short aliases keep the public vocabulary discoverable without broadening it.
EvidenceProjector = ExternalEvidenceProjector
LearningSpec = FrontierLearningSpec
