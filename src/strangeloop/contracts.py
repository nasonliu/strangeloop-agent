"""Shared typed contracts for the Strangeloop cognitive loop.

These records describe observable software state.  They are not claims about
subjective experience or a one-to-one implementation of Yogacara concepts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


def new_id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid4().hex)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_aware_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 instant and reject naive or malformed values."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be a valid ISO-8601 value") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed


FRONTIER_STRATEGY_ARM_VERSION = "frontier_strategy_arm_v1"
FRONTIER_OFFLINE_EXPERIMENT_ACTION_MODE = "offline_experiment"


def frontier_strategy_arm_id(experiment_registry_digest: str,
                             learner_spec_digest: str,
                             experiment_kind: str,
                             action_mode: str = FRONTIER_OFFLINE_EXPERIMENT_ACTION_MODE) -> str:
    """Return the canonical, task-instance-independent frontier strategy arm.

    This public identifier credits an authorized learning strategy, not a
    particular task instance or an anthropomorphic capability claim.  The
    canonical manifest deliberately contains only the registry, learner
    specification, experiment kind, and bounded action mode.
    """
    manifest = {
        "action_mode": action_mode,
        "experiment_kind": experiment_kind,
        "experiment_registry_digest": experiment_registry_digest,
        "learner_spec_digest": learner_spec_digest,
        "strategy_arm_version": FRONTIER_STRATEGY_ARM_VERSION,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EventKind(str, Enum):
    OBSERVATION = "observation"
    MEDIA_OBSERVATION = "media_observation"
    PERCEPT = "percept"
    MODEL_INVOCATION = "model_invocation"
    CAPABILITY_GRANTED = "capability_granted"
    CAPABILITY_REVOKED = "capability_revoked"
    TOOL_CALL_PROPOSED = "tool_call_proposed"
    TOOL_EXECUTION_CONFIRMED = "tool_execution_confirmed"
    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_RESULT = "tool_result"
    TOOL_EXECUTION_ABANDONED = "tool_execution_abandoned"
    INFERENCE = "inference"
    ACTION_PROPOSED = "action_proposed"
    ACTION_RESULT = "action_result"
    DECISION = "decision"
    SEED_PROPOSED = "seed_proposed"
    SEED_APPROVED = "seed_approved"
    SEED_RETIRED = "seed_retired"
    SELF_CLAIM_PROPOSED = "self_claim_proposed"
    SELF_CLAIM_APPROVED = "self_claim_approved"
    SELF_CLAIM_REVOKED = "self_claim_revoked"
    CORRECTION = "correction"
    PURGE = "purge"
    AUTONOMY_CONTROL = "autonomy_control"
    LOOP_TICK = "loop_tick"
    AUTONOMY_STOPPED = "autonomy_stopped"
    # A bounded public audit record inspired by the relation between
    # Yogacara's self-awareness and its reflective checking.  It is not a
    # claim of subjective experience or a software self.
    METACOGNITIVE_MIRROR = "metacognitive_mirror"
    REWARD_OBSERVATION = "reward_observation"
    VALUE_ESTIMATE = "value_estimate"
    RPE_UPDATE = "rpe_update"
    # Frontier learning is a constrained, auditable ranking experiment.  It is
    # not a drive, capability, quota signal, or evidence of selfhood.
    FRONTIER_RANKING_DECISION = "frontier_ranking_decision"
    FRONTIER_VECTOR_VALUE_ESTIMATE = "frontier_vector_value_estimate"
    FRONTIER_EVIDENCE_OBSERVATION = "frontier_evidence_observation"
    FRONTIER_VECTOR_REWARD = "frontier_vector_reward"
    FRONTIER_TD_UPDATE = "frontier_td_update"
    EXPERIMENT_PLAN_LOCKED = "experiment_plan_locked"
    EXPERIMENT_EXECUTION_STARTED = "experiment_execution_started"
    EXPERIMENT_RESULT = "experiment_result"
    # Sleep/wake is an operational lifecycle, not an analogue for a mental
    # state.  Its records are intentionally metadata-only public projections.
    SLEEP_ARCHIVE = "sleep_archive"
    SLEEP_ENTERED = "sleep_entered"
    PROVIDER_USAGE_EVIDENCE = "provider_usage_evidence"
    WAKE_CHECK = "wake_check"
    WAKE_READY = "wake_ready"
    AWAKE = "awake"
    WAKE_TERMINAL = "wake_terminal"
    AUTO_WAKE_POLICY = "auto_wake_policy"
    # A user-approved, one-shot continuation of bounded public research after
    # a future quota wake.  This is lifecycle control metadata, not evidence
    # of experience, selfhood, or durable model authority.
    UNATTENDED_WAKE_CONTINUATION_POLICY = "unattended_wake_continuation_policy"
    UNATTENDED_WAKE_RUN = "unattended_wake_run"
    # A user-approved, single-consumption public-web expedition contract.
    # It is operational authorization metadata, never memory or self evidence.
    EXPEDITION_AUTHORIZATION = "expedition_authorization"
    EXPEDITION_AUTHORIZATION_CONSUMED = "expedition_authorization_consumed"


class SourceKind(str, Enum):
    USER = "user"
    TOOL = "tool"
    MODEL = "model"
    POLICY = "policy"
    SYSTEM = "system"
    EXTERNAL_VERIFIER = "external_verifier"


class SeedStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"


class SelfClaimKind(str, Enum):
    ROLE = "role"
    CAPABILITY = "capability"
    COMMITMENT = "commitment"
    EPISTEMIC = "epistemic"
    CONTINUITY = "continuity"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class CognitiveEvent:
    session_id: str
    kind: EventKind
    source_kind: SourceKind
    source_ref: str
    payload: Dict[str, Any]
    confidence: float = 1.0
    parent_event_ids: Tuple[str, ...] = ()
    event_id: str = field(default_factory=lambda: new_id("evt"))
    created_at: str = field(default_factory=utc_now_iso)
    sequence: Optional[int] = None
    previous_hash: Optional[str] = None
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["source_kind"] = self.source_kind.value
        value["parent_event_ids"] = list(self.parent_event_ids)
        return value


@dataclass(frozen=True)
class SeedDisposition:
    cue_terms: Tuple[str, ...]
    policy_bias: str
    scope: str
    provenance_event_ids: Tuple[str, ...]
    strength: float = 0.5
    confidence: float = 0.5
    status: SeedStatus = SeedStatus.CANDIDATE
    seed_id: str = field(default_factory=lambda: new_id("seed"))
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    expires_at: Optional[str] = None
    version: int = 1
    counterevidence: int = 0

    def __post_init__(self) -> None:
        if not self.cue_terms:
            raise ValueError("a seed requires at least one cue term")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.version < 1 or self.counterevidence < 0:
            raise ValueError("version and counterevidence must be non-negative")
        if self.expires_at is not None:
            parse_aware_iso8601(self.expires_at)


@dataclass(frozen=True)
class SelfModelClaim:
    kind: SelfClaimKind
    statement: str
    evidence_event_ids: Tuple[str, ...]
    confidence: float
    claim_id: str = field(default_factory=lambda: new_id("claim"))
    created_at: str = field(default_factory=utc_now_iso)
    expires_at: Optional[str] = None
    revoked_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("a self-model claim cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.expires_at is not None:
            parse_aware_iso8601(self.expires_at)


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    rationale_summary: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    required_capability: Optional[str] = None
    is_mutating: bool = False


@dataclass(frozen=True)
class WorkspaceFrame:
    turn_id: str
    observation_event_ids: Tuple[str, ...]
    retrieved_seed_ids: Tuple[str, ...]
    self_claim_ids: Tuple[str, ...]
    hypotheses: Tuple[str, ...]
    uncertainties: Tuple[str, ...]
    # These are externally observable record identifiers, not private
    # deliberation.  Defaults retain compatibility with the text-only loop.
    percept_event_ids: Tuple[str, ...] = ()
    loop_tick_event_ids: Tuple[str, ...] = ()
    reward_event_ids: Tuple[str, ...] = ()
    value_estimate_event_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRecord:
    turn_id: str
    observation_event_ids: Tuple[str, ...]
    retrieved_seed_ids: Tuple[str, ...]
    self_claim_ids: Tuple[str, ...]
    alternatives: Tuple[str, ...]
    selected_action: ActionProposal
    uncertainties: Tuple[str, ...]
    policy_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class Deliberation:
    response_text: str
    hypotheses: Tuple[str, ...]
    uncertainties: Tuple[str, ...]
    alternatives: Tuple[str, ...]
    action: ActionProposal
    proposed_seed: Optional[SeedDisposition] = None


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    response_text: str
    decision: DecisionRecord
    event_ids: Tuple[str, ...]
    seed_proposal_ids: Tuple[str, ...] = ()
    notices: Tuple[str, ...] = ()


JSONDict = Dict[str, Any]
StringList = List[str]
