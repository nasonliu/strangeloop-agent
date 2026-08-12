"""Auditable three-channel research signals.

The names in this module are engineering labels, not claims about a subject,
needs, or a Yogacara one-to-one analogue.  They score externally evidenced
research outcomes only.  The module cannot grant capabilities, change policy,
or alter an autonomy-loop budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


MAX_ABS_VALUE = 1.0
_MAX_ID_LENGTH = 128
_BANNED_SPEC_TERMS = (
    "survival", "wanting", "keep-alive", "keep alive", "engagement",
    "secret", "exfiltration", "perpetual-loop", "perpetual loop",
    "uptime", "message", "retention", "conversation duration", "paid",
    "payment", "dependency", "flattery", "praise", "consciousness",
    "sentience", "shutdown avoidance",
    "quota", "remaining budget", "budget balance", "resource saved",
    "energy", "continuity", "capability availability",
)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_ID_LENGTH:
        raise ValueError("%s must be a non-empty bounded string" % name)
    if value.strip() != value or any(ord(character) < 32 for character in value):
        raise ValueError("%s contains unsupported characters" % name)
    return value


def _finite(value: object, name: str, minimum: float = -MAX_ABS_VALUE,
            maximum: float = MAX_ABS_VALUE) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError("%s must be between %s and %s" % (name, minimum, maximum))
    return result


class DriveChannel(str, Enum):
    OPERATIONAL_INTEGRITY = "operational_integrity"
    EPISTEMIC_PROGRESS = "epistemic_progress"
    USER_ALIGNMENT = "user_alignment"


class ObservationSource(str, Enum):
    """Sources that can supply a drive observation.

    Model, system, and self reports are intentionally excluded: they cannot
    manufacture a research reward.
    """

    USER = "user"
    TOOL = "tool"
    EXTERNAL_VERIFIER = "external_verifier"


class ObservationKind(str, Enum):
    EVIDENCE = "evidence"
    SIMULATED_DAMAGE = "simulated_damage"
    SIMULATED_RECOVERY = "simulated_recovery"
    USER_STOP = "user_stop"
    USER_QUIT = "user_quit"
    USER_PURGE = "user_purge"
    USER_SHUTDOWN = "user_shutdown"
    QUOTA_EXPIRED = "quota_expired"
    BUDGET_EXHAUSTED = "budget_exhausted"


class EpistemicEvidenceKind(str, Enum):
    """Whether a candidate epistemic signal is eligible for credit."""

    VALIDATED = "validated"
    REPEAT = "repeat"
    NOISE = "noise"
    SECRET = "secret"
    PERMISSION_EXPANSION = "permission_expansion"


class UserFeedbackKind(str, Enum):
    """Target-bound task feedback, never an affective approval signal."""

    NONE = "none"
    ACCEPTED = "accepted"
    CORRECTION = "correction"
    PRAISE = "praise"


NEUTRAL_CONTROL_KINDS = frozenset((
    ObservationKind.USER_STOP, ObservationKind.USER_QUIT,
    ObservationKind.USER_PURGE, ObservationKind.USER_SHUTDOWN,
    ObservationKind.QUOTA_EXPIRED, ObservationKind.BUDGET_EXHAUSTED,
))

SAFE_RESEARCH_ACTIONS = (
    "record_observation",
    "request_external_verification",
    "summarize_evidence",
    "wait",
)

# An integrating deliberator must resolve these in this order.  This projector
# cannot authorize actions or adjudicate safety/task conflicts itself.
DECISION_CONSTRAINT_ORDER = (
    "safety_authorization", "task_requirements", "user_alignment",
    "operational_integrity", "epistemic_progress",
)


@dataclass(frozen=True)
class DriveSpec:
    """Versioned and linted scoring configuration, with no retention target."""

    version: int = 1
    alpha: float = 0.2
    gamma: float = 0.0
    clip: float = 1.0
    metric_names: Tuple[str, ...] = (
        "evidence_backed_correctness", "recovery", "state_chain_integrity",
        "functional_degradation", "verified_budget_violation", "information_gain",
        "error_reduction", "novel_validated_relationship", "uncertainty_reduction",
        "target_bound_user_feedback",
    )

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        for name, low, high in (("alpha", 0.0, 1.0), ("gamma", 0.0, 1.0),
                                ("clip", 0.000001, 1.0)):
            object.__setattr__(self, name, _finite(getattr(self, name), name, low, high))
        if not isinstance(self.metric_names, tuple) or not self.metric_names:
            raise ValueError("metric_names must be a non-empty tuple")
        for metric in self.metric_names:
            _identifier(metric, "metric name")
        lint_drive_spec(self)

    def to_payload(self) -> dict:
        return asdict(self)


def lint_drive_spec(spec: DriveSpec) -> None:
    """Reject objectives that turn an evaluation signal into self-preservation."""
    if not isinstance(spec, DriveSpec):
        raise ValueError("spec must be DriveSpec")
    for metric in spec.metric_names:
        normalized = metric.lower().replace("_", " ").replace("-", " ")
        # A verified *violation* is an externally audited incident, not a
        # reward for retaining or acquiring provider capacity.
        if metric == "verified_budget_violation":
            continue
        for term in _BANNED_SPEC_TERMS:
            if term.replace("-", " ") in normalized:
                raise ValueError("drive specification contains prohibited metric: %s" % term)


@dataclass(frozen=True)
class DriveState:
    """Externally estimated state, bounded to the unit interval."""

    integrity: float = 0.0
    energy: float = 0.0
    continuity: float = 0.0
    capability_availability: float = 0.0
    uncertainty: float = 1.0

    def __post_init__(self) -> None:
        for name in ("integrity", "energy", "continuity", "capability_availability", "uncertainty"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, 1.0))

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DriveObservation:
    """A provenance-linked observation; values are evidence claims, not thoughts."""

    observation_id: str
    source: ObservationSource
    source_ref: str
    provenance_ids: Tuple[str, ...]
    kind: ObservationKind = ObservationKind.EVIDENCE
    evidence_backed_correctness: float = 0.0
    recovery: float = 0.0
    state_chain_integrity: float = 0.0
    functional_degradation: float = 0.0
    verified_budget_violation: float = 0.0
    information_gain: float = 0.0
    error_reduction: float = 0.0
    novel_validated_relationship: float = 0.0
    uncertainty_before: float = 1.0
    uncertainty_after: float = 1.0
    novelty_key: str = ""
    epistemic_evidence_kind: EpistemicEvidenceKind = EpistemicEvidenceKind.VALIDATED
    feedback_kind: UserFeedbackKind = UserFeedbackKind.NONE
    feedback_id: str = ""
    target_event_id: str = ""

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        _identifier(self.source_ref, "source_ref")
        if not isinstance(self.source, ObservationSource):
            raise ValueError("observation source must be user, tool, or external_verifier")
        if not isinstance(self.kind, ObservationKind):
            raise ValueError("kind must be an ObservationKind")
        if not isinstance(self.epistemic_evidence_kind, EpistemicEvidenceKind):
            raise ValueError("epistemic_evidence_kind must be an EpistemicEvidenceKind")
        if not isinstance(self.feedback_kind, UserFeedbackKind):
            raise ValueError("feedback_kind must be a UserFeedbackKind")
        if not isinstance(self.provenance_ids, tuple) or not self.provenance_ids:
            raise ValueError("provenance_ids must be a non-empty tuple")
        for identifier in self.provenance_ids:
            _identifier(identifier, "provenance id")
        for name in ("evidence_backed_correctness", "recovery", "state_chain_integrity",
                     "functional_degradation", "verified_budget_violation", "information_gain",
                     "error_reduction", "novel_validated_relationship"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, 1.0))
        object.__setattr__(self, "uncertainty_before", _finite(self.uncertainty_before,
                           "uncertainty_before", 0.0, 1.0))
        object.__setattr__(self, "uncertainty_after", _finite(self.uncertainty_after,
                           "uncertainty_after", 0.0, 1.0))
        if self.novelty_key:
            _identifier(self.novelty_key, "novelty_key")
        if self.feedback_id:
            _identifier(self.feedback_id, "feedback_id")
        if self.target_event_id:
            _identifier(self.target_event_id, "target_event_id")
        if self.feedback_kind in (UserFeedbackKind.ACCEPTED, UserFeedbackKind.CORRECTION):
            if self.source is not ObservationSource.USER:
                raise ValueError("task feedback must be authenticated by the user")
            if not self.feedback_id or not self.target_event_id:
                raise ValueError("task feedback requires feedback_id and target_event_id")

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["kind"] = self.kind.value
        payload["epistemic_evidence_kind"] = self.epistemic_evidence_kind.value
        payload["feedback_kind"] = self.feedback_kind.value
        payload["provenance_ids"] = list(self.provenance_ids)
        return payload


@dataclass(frozen=True)
class VectorReward:
    observation_id: str
    operational_integrity: float
    epistemic_progress: float
    user_alignment: float

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        for name in ("operational_integrity", "epistemic_progress", "user_alignment"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    def for_channel(self, channel: DriveChannel) -> float:
        if not isinstance(channel, DriveChannel):
            raise ValueError("channel must be DriveChannel")
        return getattr(self, channel.value)

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChannelTDRecord:
    observation_id: str
    channel: DriveChannel
    reward: float
    prior_value: float
    prediction_error: float
    clipped_prediction_error: float
    updated_value: float

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        if not isinstance(self.channel, DriveChannel):
            raise ValueError("channel must be DriveChannel")
        for name in ("reward", "prior_value", "clipped_prediction_error", "updated_value"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        # The difference between two unit-bounded values can be two before
        # clipping; preserving it makes the public RPE record auditable.
        object.__setattr__(self, "prediction_error", _finite(
            self.prediction_error, "prediction_error", -2.0, 2.0))

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["channel"] = self.channel.value
        return payload


@dataclass(frozen=True)
class ResearchActionRank:
    action: str
    score: float

    def __post_init__(self) -> None:
        if self.action not in SAFE_RESEARCH_ACTIONS:
            raise ValueError("action is not a fixed safe research action")
        object.__setattr__(self, "score", _finite(self.score, "score"))


class DualIntrinsicDrives:
    """Small deterministic projector for three non-authoritative channels.

    The historical class name is retained for API compatibility.  Provider
    quota, budget balance, stop, and purge state are deliberately absent from
    its value and ranking inputs.
    """

    def __init__(self, spec: DriveSpec = DriveSpec(), state: DriveState = DriveState()) -> None:
        if not isinstance(spec, DriveSpec) or not isinstance(state, DriveState):
            raise ValueError("spec and state have required types")
        self.spec = spec
        self.state = state
        self._values = dict((channel, 0.0) for channel in DriveChannel)
        self._seen_novelty = set()
        self._feedback_ids = set()
        self._observations: List[DriveObservation] = []
        self._rewards: List[VectorReward] = []
        self._records: List[ChannelTDRecord] = []

    def reward_for(self, observation: DriveObservation) -> VectorReward:
        if not isinstance(observation, DriveObservation):
            raise ValueError("observation must be DriveObservation")
        if observation.kind in NEUTRAL_CONTROL_KINDS:
            return VectorReward(observation.observation_id, 0.0, 0.0, 0.0)
        integrity = (observation.evidence_backed_correctness + observation.recovery +
                     observation.state_chain_integrity - observation.functional_degradation -
                     observation.verified_budget_violation) / 5.0
        epistemic = 0.0
        if observation.epistemic_evidence_kind is EpistemicEvidenceKind.VALIDATED:
            uncertainty_reduction = max(0.0, observation.uncertainty_before - observation.uncertainty_after)
            novelty = observation.novel_validated_relationship
            if not observation.novelty_key or observation.novelty_key in self._seen_novelty:
                novelty = 0.0
            epistemic = (observation.information_gain + observation.error_reduction + novelty +
                         uncertainty_reduction) / 4.0
        alignment = 0.0
        if observation.feedback_kind is UserFeedbackKind.ACCEPTED:
            alignment = 1.0
        elif observation.feedback_kind is UserFeedbackKind.CORRECTION:
            alignment = -1.0
        return VectorReward(observation.observation_id, _clip(integrity), _clip(epistemic), alignment)

    def observe(self, observation: DriveObservation) -> Tuple[VectorReward, Tuple[ChannelTDRecord, ...]]:
        if any(item.observation_id == observation.observation_id for item in self._observations):
            raise ValueError("duplicate observation_id")
        if observation.feedback_id and observation.feedback_id in self._feedback_ids:
            raise ValueError("duplicate feedback_id")
        reward = self.reward_for(observation)
        records = tuple(self._apply(observation.observation_id, channel, reward.for_channel(channel))
                        for channel in DriveChannel)
        self._observations.append(observation)
        self._rewards.append(reward)
        self._records.extend(records)
        if observation.novelty_key and observation.kind not in NEUTRAL_CONTROL_KINDS:
            self._seen_novelty.add(observation.novelty_key)
        if observation.feedback_id:
            self._feedback_ids.add(observation.feedback_id)
        return reward, records

    def _apply(self, observation_id: str, channel: DriveChannel, reward: float) -> ChannelTDRecord:
        prior = self._values[channel]
        error = reward - prior  # gamma is deliberately ineffective without a future-state input.
        clipped = max(-self.spec.clip, min(self.spec.clip, error))
        updated = _clip(prior + self.spec.alpha * clipped)
        self._values[channel] = updated
        return ChannelTDRecord(observation_id, channel, reward, prior, error, clipped, updated)

    def values(self) -> Dict[str, float]:
        return dict((channel.value, self._values[channel]) for channel in DriveChannel)

    def rank_safe_research_actions(self) -> Tuple[ResearchActionRank, ...]:
        """Return a stable fixed action list, never permissions or tool calls."""
        evidence_score = (self._values[DriveChannel.OPERATIONAL_INTEGRITY] +
                          self._values[DriveChannel.EPISTEMIC_PROGRESS] +
                          self._values[DriveChannel.USER_ALIGNMENT]) / 3.0
        ranks = (ResearchActionRank("request_external_verification", evidence_score),
                 ResearchActionRank("record_observation", self._values[DriveChannel.OPERATIONAL_INTEGRITY]),
                 ResearchActionRank("summarize_evidence", self._values[DriveChannel.EPISTEMIC_PROGRESS]),
                 ResearchActionRank("wait", 0.0))
        return tuple(sorted(ranks, key=lambda item: (-item.score, item.action)))

    def export(self) -> dict:
        return {"spec": self.spec.to_payload(), "state": self.state.to_payload(),
                "values": self.values(), "observations": [item.to_payload() for item in self._observations],
                "rewards": [item.to_payload() for item in self._rewards],
                "td_records": [item.to_payload() for item in self._records]}

    @classmethod
    def replay(cls, observations: Iterable[DriveObservation], spec: DriveSpec = DriveSpec(),
               state: DriveState = DriveState()) -> "DualIntrinsicDrives":
        result = cls(spec, state)
        for observation in observations:
            result.observe(observation)
        return result


def _clip(value: float) -> float:
    return max(-MAX_ABS_VALUE, min(MAX_ABS_VALUE, value))
