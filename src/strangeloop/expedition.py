"""Deterministic, host-only scheduling for bounded research expeditions.

This module deliberately does not call a model, make a network request, mint a
capability, or retain page/model content.  It turns an explicit host seed into
small, public *web-only* research targets which an integrating runtime may
present to its existing policy-enforcing planner.

``sleep`` and ``wake`` are resource-lifecycle labels only.  They do not imply
experience, persistence, or authority, and a wake never restores an old tool
grant or action plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .contracts import (FRONTIER_OFFLINE_EXPERIMENT_ACTION_MODE,
                        FRONTIER_STRATEGY_ARM_VERSION, frontier_strategy_arm_id)


_PERSONA_ORDER = (
    "forager", "diver", "contrarian", "prospector", "bridge", "curator",
)
_TOPICS = (
    "agent_exploration", "intrinsic_motivation", "quality_diversity",
    "agent_evaluation", "persona_conditioning", "open_ended_learning",
)
_SOURCE_TYPES = ("primary_paper", "institutional_report", "author_project")
_STANCES = ("support", "counterevidence", "limitation", "replication")
_ACTION_MODES = ("web_search", "direct_fetch", "citation_trace", "source_triangulation")
_EXPERIMENT_ACTION_MODE = FRONTIER_OFFLINE_EXPERIMENT_ACTION_MODE
_STRATEGY_VERSION = FRONTIER_STRATEGY_ARM_VERSION
_EXPERIMENT_KINDS = (
    "frontier_learner_ab", "frontier_replay", "duplicate_suppression", "td_invariants",
)
_EXPERIMENT_QUALITY_SIGNALS = frozenset((
    "paired_baseline_treatment", "independent_reproduction",
))
_QUALITY_SIGNALS = frozenset((
    "novel_domain", "primary_source", "counterevidence", "prediction_check",
    "cross_topic_link", "method_detail",
))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("%s must be timezone-aware" % name)
    return value.astimezone(timezone.utc)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _sha256_digest(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return value


def _web_strategy_arm_id(task_id: str, action_mode: str) -> str:
    """Return an explicitly per-instance arm for non-transferable web work."""
    material = {"strategy_arm_version": FRONTIER_STRATEGY_ARM_VERSION,
                "action_mode": action_mode, "task_id": task_id}
    return _digest(json.dumps(material, ensure_ascii=True, sort_keys=True,
                              separators=(",", ":"), allow_nan=False))


class Persona(str, Enum):
    """Public strategy labels, not claims about an agent's identity."""

    FORAGER = "forager"
    DIVER = "diver"
    CONTRARIAN = "contrarian"
    PROSPECTOR = "prospector"
    BRIDGE = "bridge"
    CURATOR = "curator"


class ExpeditionState(str, Enum):
    READY = "ready"
    ACTIVE = "active"
    WAITING_QUOTA_RETRY = "waiting_quota_retry"
    SLEEPING = "sleeping"
    COMPLETED = "completed"
    STOPPED = "stopped"


class FrontierStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DEFERRED = "deferred"
    COMPLETE = "complete"
    DROPPED = "dropped"


class ExpeditionOutcome(str, Enum):
    """Small public result labels supplied by an integrating host."""

    BRANCH_COMPLETE = "branch_complete"
    PLANNER_TRANSIENT_FAILURE = "planner_transient_failure"
    EMPTY_SEARCH = "empty_search"
    LOW_VALUE = "low_value"
    QUALITY_PROGRESS = "quality_progress"
    EXPERIMENT_SUPPORTED = "supported"
    EXPERIMENT_REFUTED = "refuted"
    EXPERIMENT_INCONCLUSIVE = "inconclusive"
    EXPERIMENT_INVALID = "invalid"


class FrontierRanker(Protocol):
    """A powerless ordering advisor for public frontier descriptors.

    Implementations receive no seed, prompt, page content, capability, store,
    or quota object.  They may only propose a complete ordering of the supplied
    task IDs; the scheduler retains every state transition and final choice.
    """

    def rank(self, candidates: Sequence[Mapping[str, object]],
             state_context: Mapping[str, object]) -> Sequence[str]:
        """Return every supplied candidate ID exactly once, in preferred order."""


@dataclass(frozen=True)
class SeedGuidanceSnapshot:
    """A bounded host projection of active seed authority.

    This is deliberately *not* a seed record.  It carries only opaque authority
    references plus a digest verified by the integrating runtime; cues, source
    observations, model text, and any private provenance are excluded.  The
    sole v1 directive expresses a review preference and cannot grant a tool,
    change a budget, or create a frontier task.
    """

    seed_ids: Tuple[str, ...]
    authority_event_ids: Tuple[str, ...]
    digest: str
    directive: str = "request_human_review"
    version: str = "seed_guidance_v1"

    def __post_init__(self) -> None:
        if (not self.seed_ids or len(self.seed_ids) > 2
                or len(self.seed_ids) != len(self.authority_event_ids)):
            raise ValueError("seed guidance must contain one or two aligned authority references")
        for collection, name in ((self.seed_ids, "seed_ids"),
                                 (self.authority_event_ids, "authority_event_ids")):
            if (len(set(collection)) != len(collection)
                    or any(not isinstance(item, str) or not item or len(item) > 160
                           for item in collection)):
                raise ValueError("%s must contain unique bounded opaque identifiers" % name)
        _sha256_digest(self.digest, "seed guidance digest")
        if self.directive != "request_human_review" or self.version != "seed_guidance_v1":
            raise ValueError("seed guidance must use the fixed v1 review directive")

    def public_snapshot(self) -> Dict[str, object]:
        """Redacted monitor/runtime projection; never exposes seed cue content."""
        return {"digest": self.digest, "seed_count": len(self.seed_ids),
                "directive": self.directive, "version": self.version}


@dataclass(frozen=True)
class ExpeditionConfig:
    """Bounds for a user-authorized foreground expedition."""

    total_authorization_seconds: int = 5 * 60 * 60
    max_slice_seconds: int = 300
    max_slices: int = 60
    max_task_attempts: int = 4
    max_planner_failures: int = 3
    max_empty_searches: int = 2
    max_low_value_results: int = 3
    # This is a bounded in-memory candidate working set, not an execution
    # allowance.  A full user-authorized expedition may schedule up to 360
    # slices, so 48 could prematurely exhaust the descriptor pool while the
    # stronger authorization, time, quota, and tool limits still allowed work.
    max_frontier_tasks: int = 1000
    max_quota_retry_attempts: int = 3
    minimum_quality_score: float = 0.60
    # v3 is the only authorization format that can schedule host-only,
    # preregistered experiments.  Older authorizations remain web-only.
    authorization_version: str = "expedition_authorization_v1"
    allowed_experiment_kinds: Tuple[str, ...] = ()
    max_experiments: int = 0
    experiment_registry_digest: Optional[str] = None
    learner_spec_digest: Optional[str] = None

    def __post_init__(self) -> None:
        integer_bounds = (
            (self.total_authorization_seconds, "total_authorization_seconds", 1, 5 * 60 * 60),
            (self.max_slice_seconds, "max_slice_seconds", 1, 300),
            (self.max_slices, "max_slices", 1, 360),
            (self.max_task_attempts, "max_task_attempts", 1, 12),
            (self.max_planner_failures, "max_planner_failures", 1, 12),
            (self.max_empty_searches, "max_empty_searches", 1, 12),
            (self.max_low_value_results, "max_low_value_results", 1, 12),
            (self.max_frontier_tasks, "max_frontier_tasks", 6, 1000),
            (self.max_quota_retry_attempts, "max_quota_retry_attempts", 1, 3),
        )
        for value, name, minimum, maximum in integer_bounds:
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise ValueError("%s is outside its bounded range" % name)
        if not isinstance(self.minimum_quality_score, (int, float)) or not 0.0 < float(self.minimum_quality_score) <= 1.0:
            raise ValueError("minimum_quality_score must be in (0, 1]")
        if self.authorization_version not in {
                "expedition_authorization_v1", "expedition_authorization_v2",
                "expedition_authorization_v3"}:
            raise ValueError("authorization_version is not supported")
        kinds = tuple(self.allowed_experiment_kinds)
        if (not all(isinstance(item, str) for item in kinds)
                or len(set(kinds)) != len(kinds)
                or any(item not in _EXPERIMENT_KINDS for item in kinds)):
            raise ValueError("allowed_experiment_kinds must use the fixed public vocabulary")
        if self.authorization_version == "expedition_authorization_v3":
            if not kinds or not isinstance(self.max_experiments, int) or isinstance(self.max_experiments, bool) \
                    or not 1 <= self.max_experiments <= 16:
                raise ValueError("v3 requires non-empty experiments bounded to 1..16")
            _sha256_digest(self.experiment_registry_digest, "experiment_registry_digest")
            _sha256_digest(self.learner_spec_digest, "learner_spec_digest")
        elif (kinds or self.max_experiments != 0 or self.experiment_registry_digest is not None
              or self.learner_spec_digest is not None):
            raise ValueError("only v3 authorizations may schedule experiments")


@dataclass
class FrontierTask:
    """A compact target descriptor.  It intentionally contains no URL/query/body."""

    task_id: str
    persona: Persona
    topic_cluster: str
    source_type: str
    evidence_stance: str
    action_mode: str
    status: FrontierStatus = FrontierStatus.PENDING
    attempts: int = 0
    planner_failures: int = 0
    empty_searches: int = 0
    low_value_results: int = 0
    not_before_slice: int = 1
    quality_score: float = 0.0
    quality_signals: Tuple[str, ...] = ()
    experiment_kind: Optional[str] = None
    strategy_arm_id: str = ""
    strategy_version: str = _STRATEGY_VERSION

    def public_snapshot(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "persona": self.persona.value,
            "topic_cluster": self.topic_cluster,
            "source_type": self.source_type,
            "evidence_stance": self.evidence_stance,
            "action_mode": self.action_mode,
            "status": self.status.value,
            "attempts": self.attempts,
            "planner_failures": self.planner_failures,
            "empty_searches": self.empty_searches,
            "low_value_results": self.low_value_results,
            "not_before_slice": self.not_before_slice,
            "quality_score": round(self.quality_score, 3),
            "quality_signals": list(self.quality_signals),
            "experiment_kind": self.experiment_kind,
            "strategy_arm_id": self.strategy_arm_id,
            "strategy_version": self.strategy_version,
            "target": {"tool_family": "web_only_read", "network_execution": "not_performed"},
        }


@dataclass(frozen=True)
class SliceDecision:
    slice_number: int
    persona: Persona
    task: FrontierTask
    max_seconds: int
    selection_reason: str = "legacy_persona_rotation"
    seed_guidance: Optional[Dict[str, object]] = None

    def public_snapshot(self) -> Dict[str, object]:
        result = {"slice_number": self.slice_number, "persona": self.persona.value,
                  "max_seconds": self.max_seconds, "selection_reason": self.selection_reason,
                  "task": self.task.public_snapshot()}
        if self.seed_guidance is not None:
            result["seed_guidance"] = dict(self.seed_guidance)
        return result


class ExpeditionScheduler:
    """Host-controlled, deterministic frontier scheduling with finite retries.

    The scheduler has no capability registry and no provider dependency.  A
    caller must explicitly decide when a slice starts, report a small outcome,
    and separately handle quota lifecycle transitions.
    """

    def __init__(self, host_seed: str, config: Optional[ExpeditionConfig] = None,
                 now: Optional[datetime] = None, ranker: Optional[FrontierRanker] = None,
                 ranker_active: bool = True) -> None:
        if not isinstance(host_seed, str) or not host_seed.strip() or len(host_seed) > 512:
            raise ValueError("host_seed must be bounded non-empty text")
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            raise ValueError("ranker must provide a callable rank method")
        if not isinstance(ranker_active, bool):
            raise ValueError("ranker_active must be a bool")
        self.config = config or ExpeditionConfig()
        self._seed_digest = _digest(host_seed)
        self._started_at = _require_aware(now or _utc_now(), "now")
        self._state = ExpeditionState.READY
        self._stop_reason: Optional[str] = None
        self._slice_number = 0
        self._active_task_id: Optional[str] = None
        self._quota_retry_attempts = 0
        self._quota_retry_at: Optional[datetime] = None
        self._quota_retry_reason: Optional[str] = None
        self._task_sequence = 0
        self._tasks: Dict[str, FrontierTask] = {}
        self._history: List[Dict[str, object]] = []
        self._ranker = ranker
        self._ranker_active = ranker_active
        self._persona_last_selected_slice = {Persona(item): 0 for item in _PERSONA_ORDER}
        self._last_candidate_snapshot: Dict[str, object] = {
            "candidates": [], "state_context": {}, "recommendation_status": "not_configured",
        }
        self._last_selection_reason: Optional[str] = None
        self._last_experiment_selected_slice = 0
        self._seed_guidance: Optional[SeedGuidanceSnapshot] = None
        self._create_initial_frontier()

    @property
    def state(self) -> ExpeditionState:
        return self._state

    def set_seed_guidance(self, snapshot: Optional[SeedGuidanceSnapshot]) -> None:
        """Install the host-verified guidance projection for a future slice.

        The scheduler accepts no seed text and refuses a mid-slice update, so a
        changed seed can influence a later choice but never mutate an already
        authorized action.  ``None`` removes the projection and restores the
        legacy trajectory.
        """
        if snapshot is not None and not isinstance(snapshot, SeedGuidanceSnapshot):
            raise ValueError("seed guidance must be a SeedGuidanceSnapshot or None")
        if self._active_task_id is not None:
            raise RuntimeError("cannot change seed guidance during an active slice")
        self._seed_guidance = snapshot

    def seed_guidance_snapshot(self) -> Optional[Dict[str, object]]:
        """Return the redacted active guidance projection, if any."""
        return None if self._seed_guidance is None else self._seed_guidance.public_snapshot()

    def begin_slice(self, now: Optional[datetime] = None) -> Optional[SliceDecision]:
        """Select one bounded target.  It never performs the selected action."""
        current = _require_aware(now or _utc_now(), "now")
        self._expire_if_due(current)
        if self._state == ExpeditionState.WAITING_QUOTA_RETRY:
            return None
        if self._state in (ExpeditionState.SLEEPING, ExpeditionState.COMPLETED, ExpeditionState.STOPPED):
            return None
        if self._active_task_id is not None:
            raise RuntimeError("a slice is already active")
        if self._slice_number >= self.config.max_slices:
            self._complete("max_slices")
            return None
        self._state = ExpeditionState.ACTIVE
        self._slice_number += 1
        scheduled_persona = Persona(_PERSONA_ORDER[(self._slice_number - 1) % len(_PERSONA_ORDER)])
        task, selection_reason = self._select_task(scheduled_persona)
        if task is None:
            self._complete("frontier_exhausted")
            return None
        task.status, task.attempts = FrontierStatus.ACTIVE, task.attempts + 1
        if task.experiment_kind is not None:
            self._last_experiment_selected_slice = self._slice_number
        self._active_task_id = task.task_id
        self._persona_last_selected_slice[task.persona] = self._slice_number
        self._last_selection_reason = selection_reason
        return SliceDecision(self._slice_number, task.persona, task, self.config.max_slice_seconds,
                             selection_reason, self.seed_guidance_snapshot())

    def defer_quota_retry(self, reason: str, retry_at: datetime,
                          now: Optional[datetime] = None) -> Dict[str, object]:
        """Fail closed between foreground quota refreshes; it never runs work."""
        current = _require_aware(now or _utc_now(), "now")
        retry = _require_aware(retry_at, "retry_at")
        if (not isinstance(reason, str) or reason not in {
                "quota_refresh_unknown_fail_closed", "quota_refresh_rejected_fail_closed",
                "quota_refresh_old_or_unknown_fail_closed", "quota_refresh_transient_retry_pending",
                "quota_refresh_interval_waiting", "quota_reset_requires_fresh_telemetry"}):
            raise ValueError("quota retry reason is not allowed")
        self._expire_if_due(current)
        if self._state in (ExpeditionState.COMPLETED, ExpeditionState.STOPPED, ExpeditionState.SLEEPING):
            return self.public_snapshot()
        if retry <= current:
            raise ValueError("quota retry_at must be in the future")
        if self._active_task_id is not None:
            raise RuntimeError("cannot defer quota retry during an active slice")
        # Cadence waiting and explicitly classified transient bridge failures
        # are not a terminal resource condition.  They remain fail-closed for
        # work, but a foreground owner may keep retrying at its bounded gate
        # cadence until fresh provider evidence arrives or authorization ends.
        if reason not in ("quota_refresh_interval_waiting", "quota_reset_requires_fresh_telemetry",
                          "quota_refresh_transient_retry_pending"):
            self._quota_retry_attempts += 1
        self._quota_retry_at, self._quota_retry_reason = retry, reason
        if self._quota_retry_attempts > self.config.max_quota_retry_attempts:
            # Retry exhaustion is a bounded foreground-refresh failure, not a
            # resource lifecycle transition.  Only the sleep coordinator (or
            # an explicit quota_sleep observation) may put this scheduler to
            # sleep.
            self._state, self._stop_reason = ExpeditionState.STOPPED, "quota_refresh_retry_exhausted"
        else:
            self._state, self._stop_reason = ExpeditionState.WAITING_QUOTA_RETRY, None
        return self.public_snapshot()

    def quota_retry_due(self, now: Optional[datetime] = None) -> bool:
        current = _require_aware(now or _utc_now(), "now")
        self._expire_if_due(current)
        return bool(self._state == ExpeditionState.WAITING_QUOTA_RETRY and self._quota_retry_at is not None
                    and current >= self._quota_retry_at)

    def resume_after_fresh_quota(self, now: Optional[datetime] = None) -> bool:
        """External caller asserts fresh authority; this does not fetch or grant."""
        current = _require_aware(now or _utc_now(), "now")
        self._expire_if_due(current)
        if not self.quota_retry_due(current):
            return False
        self._state, self._stop_reason = ExpeditionState.READY, None
        self._quota_retry_at, self._quota_retry_reason = None, None
        return True

    def record_outcome(self, task_id: str, outcome: ExpeditionOutcome,
                       quality_score: float = 0.0,
                       quality_signals: Sequence[str] = (),
                       now: Optional[datetime] = None) -> Dict[str, object]:
        """Accept a public outcome label and schedule finite, deterministic recovery."""
        current = _require_aware(now or _utc_now(), "now")
        self._expire_if_due(current)
        if self._state != ExpeditionState.ACTIVE or self._active_task_id != task_id:
            raise RuntimeError("outcome does not belong to the active slice")
        if not isinstance(outcome, ExpeditionOutcome):
            raise ValueError("outcome must be an ExpeditionOutcome")
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError("unknown frontier task")
        signals = tuple(sorted(set(quality_signals)))
        allowed_signals = (_EXPERIMENT_QUALITY_SIGNALS if task.experiment_kind is not None
                           else _QUALITY_SIGNALS)
        if any(item not in allowed_signals for item in signals):
            raise ValueError("quality_signals must use the fixed public vocabulary")
        if not isinstance(quality_score, (int, float)) or not 0.0 <= float(quality_score) <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")

        effective = outcome
        if task.experiment_kind is not None:
            if outcome == ExpeditionOutcome.QUALITY_PROGRESS:
                # Execution count, HTTP success, or a scheduler assertion is
                # never experimental progress.
                effective = ExpeditionOutcome.LOW_VALUE
            elif outcome in (ExpeditionOutcome.EXPERIMENT_SUPPORTED,
                             ExpeditionOutcome.EXPERIMENT_REFUTED):
                # Both a paired comparison and an independent repeat are
                # required. A completed execution, or one receipt alone, is
                # not evidence of experimental progress.
                if (float(quality_score) < self.config.minimum_quality_score
                        or set(signals) != _EXPERIMENT_QUALITY_SIGNALS):
                    effective = ExpeditionOutcome.LOW_VALUE
                else:
                    task.quality_score = max(task.quality_score, float(quality_score))
                    task.quality_signals = signals
                    task.status = FrontierStatus.COMPLETE
            elif outcome in (ExpeditionOutcome.EXPERIMENT_INCONCLUSIVE,
                             ExpeditionOutcome.EXPERIMENT_INVALID):
                # These records remain auditable but cannot improve ranking.
                effective = ExpeditionOutcome.LOW_VALUE
        elif outcome in (ExpeditionOutcome.EXPERIMENT_SUPPORTED,
                         ExpeditionOutcome.EXPERIMENT_REFUTED,
                         ExpeditionOutcome.EXPERIMENT_INCONCLUSIVE,
                         ExpeditionOutcome.EXPERIMENT_INVALID):
            raise ValueError("experiment outcomes require an experiment task")
        if task.experiment_kind is None and outcome == ExpeditionOutcome.QUALITY_PROGRESS:
            # A successful HTTP request alone cannot establish progress.
            if float(quality_score) < self.config.minimum_quality_score or not signals:
                effective = ExpeditionOutcome.LOW_VALUE
            else:
                task.quality_score = max(task.quality_score, float(quality_score))
                task.quality_signals = signals
                task.status = FrontierStatus.COMPLETE
        if effective == ExpeditionOutcome.BRANCH_COMPLETE:
            task.status = FrontierStatus.COMPLETE
        elif effective == ExpeditionOutcome.PLANNER_TRANSIENT_FAILURE:
            task.planner_failures += 1
            self._retry_or_drop(task, task.planner_failures, self.config.max_planner_failures, "planner")
        elif effective == ExpeditionOutcome.EMPTY_SEARCH:
            task.empty_searches += 1
            task.action_mode = self._next_action_mode(task.action_mode)
            self._retry_or_drop(task, task.empty_searches, self.config.max_empty_searches, "empty_search")
        elif effective == ExpeditionOutcome.LOW_VALUE:
            task.low_value_results += 1
            task.evidence_stance = self._next_stance(task.evidence_stance)
            self._retry_or_drop(task, task.low_value_results, self.config.max_low_value_results, "low_value")
        self._history.append({"slice_number": self._slice_number, "task_id": task.task_id,
                              "outcome": effective.value, "quality_score": round(float(quality_score), 3),
                              "quality_signal_count": len(signals),
                              "experiment_kind": task.experiment_kind})
        self._active_task_id = None
        if self._state == ExpeditionState.ACTIVE:
            self._state = ExpeditionState.READY
        self._expire_if_due(current)
        return self.public_snapshot()

    def sleep(self, reason: str = "quota_unavailable") -> Dict[str, object]:
        """Enter a resource-gated state; no task/grant is resumed by this call."""
        if self._state in (ExpeditionState.COMPLETED, ExpeditionState.STOPPED):
            return self.public_snapshot()
        if not isinstance(reason, str) or not reason or len(reason) > 80:
            raise ValueError("sleep reason must be bounded text")
        if self._active_task_id is not None:
            self._tasks[self._active_task_id].status = FrontierStatus.DEFERRED
            self._tasks[self._active_task_id].not_before_slice = self._slice_number + 1
            self._active_task_id = None
        self._state, self._stop_reason = ExpeditionState.SLEEPING, reason
        return self.public_snapshot()

    def wake(self, fresh_authoritative_quota: bool, now: Optional[datetime] = None) -> bool:
        """Return to ready only when an external host has fresh quota evidence."""
        current = _require_aware(now or _utc_now(), "now")
        self._expire_if_due(current)
        if self._state != ExpeditionState.SLEEPING or not fresh_authoritative_quota:
            return False
        self._state, self._stop_reason = ExpeditionState.READY, None
        return True

    def stop(self, reason: str = "user_stop") -> Dict[str, object]:
        if not isinstance(reason, str) or not reason or len(reason) > 80:
            raise ValueError("stop reason must be bounded text")
        self._active_task_id = None
        self._state, self._stop_reason = ExpeditionState.STOPPED, reason
        return self.public_snapshot()

    def complete_if_expired(self, now: Optional[datetime] = None) -> bool:
        """Close an elapsed user authorization without treating it as failure."""
        current = _require_aware(now or _utc_now(), "now")
        self._expire_if_due(current)
        return self._state == ExpeditionState.COMPLETED

    def public_snapshot(self) -> Dict[str, object]:
        """A safe monitor projection; no raw seed, prompts, URLs, or page text."""
        counts = {status.value: 0 for status in FrontierStatus}
        for task in self._tasks.values():
            counts[task.status.value] += 1
        result = {
            "state": self._state.value,
            "stop_reason": self._stop_reason,
            "seed_digest": self._seed_digest,
            "slice_number": self._slice_number,
            "active_task_id": self._active_task_id,
            "authorization_seconds": self.config.total_authorization_seconds,
            "max_slice_seconds": self.config.max_slice_seconds,
            "frontier_counts": counts,
            "history_count": len(self._history),
            "selection": {"last_reason": self._last_selection_reason,
                          "ranker_configured": self._ranker is not None,
                          "ranker_mode": ("active" if self._ranker_active else "shadow")
                          if self._ranker is not None else "disabled"},
            "experiments": {"authorization_version": self.config.authorization_version,
                            "allowed_kinds": list(self.config.allowed_experiment_kinds),
                            "max_experiments": self.config.max_experiments,
                            "selected_count": sum(1 for item in self._history
                                                  if item.get("experiment_kind") is not None)},
            "candidate_snapshot": self.candidate_snapshot(),
            "quota_retry": {"attempts": self._quota_retry_attempts,
                            "retry_at": None if self._quota_retry_at is None else self._quota_retry_at.isoformat(),
                            "reason": self._quota_retry_reason},
            "tasks": [task.public_snapshot() for task in self._tasks.values()],
            "limitations": [
                "host_scheduler_only", "no_network_execution", "no_capability_or_grant_changes",
                "no_raw_seed_urls_queries_page_bodies_or_hidden_reasoning",
            ],
        }
        if self._seed_guidance is not None:
            result["seed_guidance"] = self.seed_guidance_snapshot()
        return result

    def candidate_snapshot(self) -> Dict[str, object]:
        """Latest ranker input, limited to public task descriptors and state."""
        return {
            "candidates": [dict(item) for item in self._last_candidate_snapshot["candidates"]],
            "state_context": dict(self._last_candidate_snapshot["state_context"]),
            "recommendation_status": self._last_candidate_snapshot["recommendation_status"],
        }

    def _create_initial_frontier(self) -> None:
        for index, raw_persona in enumerate(_PERSONA_ORDER):
            self._add_task(Persona(raw_persona), index)
        for index in range(self.config.max_experiments):
            self._add_experiment_task(index)

    def _add_task(self, persona: Persona, index: int) -> Optional[FrontierTask]:
        if len(self._tasks) >= self.config.max_frontier_tasks:
            self._reclaim_terminal_task_slot()
        if len(self._tasks) >= self.config.max_frontier_tasks:
            return None
        self._task_sequence += 1
        material = "%s:%s:%s:%s" % (self._seed_digest, persona.value, index, self._task_sequence)
        number = int(_digest(material)[:8], 16)
        task = FrontierTask(
            # The ID is reproducible from host-owned scheduling inputs.  The
            # original seed remains private; only its digest is embedded.
            task_id="frontier_%s" % _digest("task:%s" % material)[:24],
            persona=persona,
            topic_cluster=_TOPICS[number % len(_TOPICS)],
            source_type=_SOURCE_TYPES[(number // 7) % len(_SOURCE_TYPES)],
            evidence_stance=_STANCES[(number // 13) % len(_STANCES)],
            action_mode=_ACTION_MODES[(number // 17) % len(_ACTION_MODES)],
        )
        # Web work has no transferable offline fixture: each concrete frontier
        # instance is its own stable arm.
        task.strategy_arm_id = _web_strategy_arm_id(task.task_id, task.action_mode)
        self._tasks[task.task_id] = task
        return task

    def _add_experiment_task(self, index: int) -> Optional[FrontierTask]:
        if len(self._tasks) >= self.config.max_frontier_tasks:
            self._reclaim_terminal_task_slot()
        if len(self._tasks) >= self.config.max_frontier_tasks:
            return None
        kind = self.config.allowed_experiment_kinds[index % len(self.config.allowed_experiment_kinds)]
        persona = Persona(_PERSONA_ORDER[index % len(_PERSONA_ORDER)])
        self._task_sequence += 1
        material = "%s:experiment:%s:%s:%s" % (self._seed_digest, kind, index, self._task_sequence)
        task = FrontierTask(
            task_id="frontier_%s" % _digest("task:%s" % material)[:24],
            persona=persona, topic_cluster="frontier_experiment",
            source_type="host_fixture", evidence_stance="replication",
            action_mode=_EXPERIMENT_ACTION_MODE, experiment_kind=kind,
        )
        # The fixed offline handler does not vary by persona or scheduling
        # instance.  Equivalent kinds therefore intentionally share one arm.
        task.strategy_arm_id = frontier_strategy_arm_id(
            self.config.experiment_registry_digest, self.config.learner_spec_digest,
            kind, _EXPERIMENT_ACTION_MODE)
        self._tasks[task.task_id] = task
        return task

    def _reclaim_terminal_task_slot(self) -> bool:
        """Release one completed scheduler cell while retaining its audit trail.

        ``max_frontier_tasks`` bounds the mutable candidate working set, not
        the total number of task instances an already-authorized episode may
        attempt.  Outcomes remain in ``_history`` and in the external event
        ledger before a terminal descriptor is released.  Active, pending,
        and deferred tasks are never reclaimed, so this cannot cancel work or
        bypass retries, authorization, quota, or slice limits.
        """
        terminal = [item for item in self._tasks.values()
                    if item.status in (FrontierStatus.COMPLETE, FrontierStatus.DROPPED)]
        if not terminal:
            return False
        # Stable ordering keeps replay deterministic without retaining an
        # unbounded in-memory archive of already-recorded task descriptors.
        retired = min(terminal, key=lambda item: item.task_id)
        del self._tasks[retired.task_id]
        return True

    def _select_task(self, persona: Persona) -> Tuple[Optional[FrontierTask], str]:
        baseline = self._eligible_for_persona(persona)
        if not baseline:
            # A new cell preserves role rotation after an earlier branch closed.
            self._add_task(persona, self._slice_number)
            baseline = self._eligible_for_persona(persona)
        experiments = [item for item in self._eligible_tasks() if item.experiment_kind is not None]
        experiment_due = bool(experiments and self._experiment_due())
        if self._ranker is None:
            self._record_candidate_snapshot(baseline, persona, "not_configured")
            if experiment_due:
                return self._baseline_task(experiments), "experiment_cadence_due"
            guided = self._seed_guided_task(baseline)
            if guided is not None:
                return guided, "seed_guidance_request_human_review"
            return self._baseline_task(baseline), "legacy_persona_rotation"

        # In active mode all eligible frontier arms are visible to the ranker;
        # persona is a scheduling context, not a hard candidate restriction.
        candidates = self._eligible_tasks() if self._ranker_active else baseline
        recommendation = self._rank_candidates(candidates, persona)
        if not self._ranker_active:
            reason = ("ranker_shadow_observed_baseline" if recommendation is not None
                      else "ranker_shadow_invalid_baseline")
            if experiment_due:
                return self._baseline_task(experiments), "experiment_cadence_due_shadow"
            guided = self._seed_guided_task(baseline)
            if guided is not None:
                return guided, "seed_guidance_request_human_review"
            return self._baseline_task(baseline), reason
        if recommendation is None:
            if experiment_due:
                return self._baseline_task(experiments), "experiment_cadence_due"
            guided = self._seed_guided_task(baseline)
            if guided is not None:
                return guided, "seed_guidance_request_human_review"
            return self._baseline_task(baseline), "ranker_invalid_fallback"
        # A ranker can influence priority but cannot indefinitely starve the
        # persona whose bounded turn is due.  The full ranking is still
        # observed above, so shadow analysis remains comparable to active.
        if baseline and self._persona_last_selected_slice[persona] + len(_PERSONA_ORDER) < self._slice_number:
            return self._baseline_task(baseline), "fairness_persona_due"
        if experiment_due:
            ranked_experiment_ids = set(item.task_id for item in experiments)
            experiment_id = next((task_id for task_id in recommendation
                                  if task_id in ranked_experiment_ids), None)
            if experiment_id is not None:
                return self._tasks[experiment_id], "experiment_cadence_due_ranker"
            # A valid full ranking must include every eligible experiment;
            # retain this closed fallback for future eligibility changes.
            return self._baseline_task(experiments), "experiment_cadence_due"
        guided = self._seed_guided_task([self._tasks[task_id] for task_id in recommendation])
        if guided is not None:
            return guided, "seed_guidance_request_human_review"
        return self._tasks[recommendation[0]], "ranker_active_recommendation"

    def _seed_guided_task(self, candidates: Sequence[FrontierTask]) -> Optional[FrontierTask]:
        """Soft-rank existing web candidates under the fixed review directive.

        This runs only after all host hard gates.  Experiments are excluded,
        and the sequence passed in already represents either persona baseline
        order or a valid ranker recommendation.  Without a matching preferred
        descriptor this returns ``None`` rather than perturbing the legacy
        ordering.
        """
        if self._seed_guidance is None:
            return None
        preferred_stances = frozenset(("counterevidence", "limitation"))
        preferred_actions = frozenset(("citation_trace", "source_triangulation"))
        eligible = [item for item in candidates if item.experiment_kind is None]
        decorated = []
        for index, item in enumerate(eligible):
            stance = item.evidence_stance in preferred_stances
            action = item.action_mode in preferred_actions
            # A pair best embodies the bounded review directive.  The input
            # order deterministically resolves ties, preserving ranker or
            # baseline order among equally suitable candidates.
            preference = 0 if stance and action else 1 if stance else 2 if action else 3
            decorated.append((preference, index, item))
        if not decorated or min(item[0] for item in decorated) == 3:
            return None
        return min(decorated, key=lambda item: (item[0], item[1]))[2]

    def _experiment_due(self) -> bool:
        """At most one offline experiment per persona cycle; at least one in it."""
        return self._last_experiment_selected_slice + len(_PERSONA_ORDER) <= self._slice_number

    def _eligible_for_persona(self, persona: Persona) -> List[FrontierTask]:
        return [item for item in self._eligible_tasks() if item.persona == persona]

    def _eligible_tasks(self) -> List[FrontierTask]:
        return [item for item in self._tasks.values()
                if item.status in (FrontierStatus.PENDING, FrontierStatus.DEFERRED)
                and item.not_before_slice <= self._slice_number]

    @staticmethod
    def _baseline_task(eligible: Sequence[FrontierTask]) -> Optional[FrontierTask]:
        if not eligible:
            return None
        return sorted(eligible, key=lambda item: (item.attempts, item.not_before_slice, item.task_id))[0]

    def _rank_candidates(self, candidates: Sequence[FrontierTask], persona: Persona) -> Optional[Tuple[str, ...]]:
        descriptors = tuple(self._ranker_descriptor(item) for item in candidates)
        context = {
            "state": self._state.value,
            "slice_number": self._slice_number,
            "scheduled_persona": persona.value,
            "candidate_count": len(descriptors),
            "max_slice_seconds": self.config.max_slice_seconds,
        }
        try:
            ranked = self._ranker.rank(descriptors, context)  # type: ignore[union-attr]
        except Exception:
            self._last_candidate_snapshot = {"candidates": list(descriptors), "state_context": context,
                                             "recommendation_status": "ranker_error_fallback"}
            return None
        candidate_ids = tuple(item.task_id for item in candidates)
        if (isinstance(ranked, (str, bytes)) or not isinstance(ranked, Sequence)
                or any(not isinstance(task_id, str) for task_id in ranked)
                or tuple(sorted(ranked)) != tuple(sorted(candidate_ids))
                or len(set(ranked)) != len(candidate_ids)):
            self._last_candidate_snapshot = {"candidates": list(descriptors), "state_context": context,
                                             "recommendation_status": "ranker_invalid_fallback"}
            return None
        self._last_candidate_snapshot = {"candidates": list(descriptors), "state_context": context,
                                         "recommendation_status": "ranker_valid"}
        return tuple(ranked)

    @staticmethod
    def _ranker_descriptor(task: FrontierTask) -> Dict[str, object]:
        """Stable, content-free arm fields exposed to a recommendation-only ranker."""
        return {
            "task_id": task.task_id,
            "persona": task.persona.value,
            "topic_cluster": task.topic_cluster,
            "source_type": task.source_type,
            "evidence_stance": task.evidence_stance,
            "action_mode": task.action_mode,
            "status": task.status.value,
            "attempts": task.attempts,
            "planner_failures": task.planner_failures,
            "empty_searches": task.empty_searches,
            "low_value_results": task.low_value_results,
            "not_before_slice": task.not_before_slice,
            "quality_score": round(task.quality_score, 3),
            "quality_signals": list(task.quality_signals),
            "experiment_kind": task.experiment_kind,
            "strategy_arm_id": task.strategy_arm_id,
            "strategy_version": task.strategy_version,
            "experiment_descriptor": (None if task.experiment_kind is None else {
                "strategy_arm_id": task.strategy_arm_id, "strategy_version": task.strategy_version,
                "network_execution": "not_performed"}),
        }

    def _record_candidate_snapshot(self, candidates: Sequence[FrontierTask], persona: Persona,
                                   recommendation_status: str) -> None:
        context = {"state": self._state.value, "slice_number": self._slice_number,
                   "scheduled_persona": persona.value, "candidate_count": len(candidates),
                   "max_slice_seconds": self.config.max_slice_seconds}
        if self._seed_guidance is not None:
            context["seed_guidance"] = self.seed_guidance_snapshot()
        self._last_candidate_snapshot = {
            "candidates": [self._ranker_descriptor(item) for item in candidates],
            "state_context": context,
            "recommendation_status": recommendation_status,
        }

    def _retry_or_drop(self, task: FrontierTask, failures: int, maximum: int, label: str) -> None:
        if task.attempts >= self.config.max_task_attempts or failures >= maximum:
            task.status = FrontierStatus.DROPPED
            return
        # One, two, then four role rotations: bounded exponential backoff.
        task.status = FrontierStatus.DEFERRED
        task.not_before_slice = self._slice_number + min(4, 2 ** max(0, failures - 1))

    @staticmethod
    def _next_action_mode(current: str) -> str:
        if current == _EXPERIMENT_ACTION_MODE:
            return current
        index = _ACTION_MODES.index(current)
        return _ACTION_MODES[(index + 1) % len(_ACTION_MODES)]

    @staticmethod
    def _next_stance(current: str) -> str:
        index = _STANCES.index(current)
        return _STANCES[(index + 1) % len(_STANCES)]

    def _expire_if_due(self, now: datetime) -> None:
        if self._state in (ExpeditionState.COMPLETED, ExpeditionState.STOPPED):
            return
        if now - self._started_at >= timedelta(seconds=self.config.total_authorization_seconds):
            self._complete("authorization_window_elapsed")

    def _complete(self, reason: str) -> None:
        self._active_task_id = None
        self._state, self._stop_reason = ExpeditionState.COMPLETED, reason
