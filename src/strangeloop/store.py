"""SQLite-backed, append-only storage for auditable cognitive events."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import math
import re
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional

from .contracts import (CognitiveEvent, EventKind, SourceKind,
                        FRONTIER_OFFLINE_EXPERIMENT_ACTION_MODE,
                        FRONTIER_STRATEGY_ARM_VERSION, frontier_strategy_arm_id,
                        parse_aware_iso8601)


MAX_PAYLOAD_BYTES = 65536
MAX_TEXT = 2048
MAX_SHORT_TEXT = 512
MAX_LIST = 32
WAKE_USAGE_MAX_AGE_SECONDS = 15 * 60
FORBIDDEN_PAYLOAD_KEYS = frozenset((
    "chain_of_thought", "hidden_reasoning", "private_reasoning", "scratchpad",
))
ROOT_EVENT_KINDS = frozenset((EventKind.OBSERVATION, EventKind.MEDIA_OBSERVATION,
                              EventKind.AUTONOMY_CONTROL, EventKind.SLEEP_ARCHIVE,
                              EventKind.AUTO_WAKE_POLICY))
SLEEP_WAKE_EVENT_KINDS = frozenset((
    EventKind.SLEEP_ARCHIVE, EventKind.SLEEP_ENTERED,
    EventKind.PROVIDER_USAGE_EVIDENCE, EventKind.WAKE_CHECK,
    EventKind.WAKE_READY, EventKind.AWAKE, EventKind.WAKE_TERMINAL,
    EventKind.AUTO_WAKE_POLICY,
    EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY,
    EventKind.UNATTENDED_WAKE_RUN,
    EventKind.EXPEDITION_AUTHORIZATION,
    EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
))
NON_EVIDENCE_LIFECYCLE_EVENT_KINDS = SLEEP_WAKE_EVENT_KINDS | frozenset((
    EventKind.EXPEDITION_AUTHORIZATION,
    EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
))
FRONTIER_CHANNEL_ORDER = (
    "functional_continuity", "bounded_curiosity", "operational_integrity",
    "epistemic_progress", "user_alignment",
)
FRONTIER_LEARNING_EVENT_KINDS = frozenset((
    EventKind.FRONTIER_RANKING_DECISION,
    EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE,
    EventKind.FRONTIER_EVIDENCE_OBSERVATION,
    EventKind.FRONTIER_VECTOR_REWARD,
    EventKind.FRONTIER_TD_UPDATE,
))
# Operational/lifecycle controls are never evidence for, nor targets of,
# frontier learning.  This includes quota and sleep records by construction.
FRONTIER_FORBIDDEN_EVIDENCE_KINDS = NON_EVIDENCE_LIFECYCLE_EVENT_KINDS | frozenset((
    EventKind.CAPABILITY_GRANTED, EventKind.CAPABILITY_REVOKED,
    EventKind.AUTONOMY_CONTROL, EventKind.AUTONOMY_STOPPED,
    EventKind.SEED_PROPOSED, EventKind.SEED_APPROVED, EventKind.SEED_RETIRED,
    EventKind.SELF_CLAIM_PROPOSED, EventKind.SELF_CLAIM_APPROVED,
    EventKind.SELF_CLAIM_REVOKED, EventKind.PURGE,
    EventKind.EXPERIMENT_PLAN_LOCKED, EventKind.EXPERIMENT_EXECUTION_STARTED,
))
EXPERIMENT_KINDS = frozenset((
    "frontier_learner_ab", "frontier_replay", "duplicate_suppression", "td_invariants",
))
EXPERIMENT_INVARIANT_CODES = frozenset((
    "paired_seed_control", "replay_match", "deterministic_replay",
    "content_digest_deduplicated", "one_shared_next_arm", "td0_formula",
    "five_channel_vector", "deadline_enforced", "handler_failed",
))
EXPERIMENT_EVENT_KINDS = frozenset((
    EventKind.EXPERIMENT_PLAN_LOCKED, EventKind.EXPERIMENT_EXECUTION_STARTED,
    EventKind.EXPERIMENT_RESULT,
))


def frontier_experiment_reward_vector(experiment_kind: str, status: str,
                                      is_novel_result: bool) -> List[float]:
    """Return the sole v1 reward projection for a verified offline result.

    This is an engineering score for bounded frontier ranking, never a quota,
    capability, self-preservation, or general-document signal.  Keeping the
    mapping here makes the runtime and append validator use identical values.
    """
    if experiment_kind not in EXPERIMENT_KINDS:
        raise ValueError("experiment kind is not eligible for frontier reward")
    if status not in {"supported", "refuted", "inconclusive"}:
        raise ValueError("experiment status is not eligible for frontier reward")
    continuity = .4 if experiment_kind in {"frontier_replay", "td_invariants"} else .1
    epistemic = .8 if status in {"supported", "refuted"} else .3
    return [continuity, .7 if is_novel_result else .0, 1.0, epistemic, .0]
ALLOWED_EVENT_SOURCES = {
    EventKind.OBSERVATION: frozenset((
        SourceKind.USER, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.MEDIA_OBSERVATION: frozenset((
        SourceKind.USER, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.PERCEPT: frozenset((SourceKind.SYSTEM, SourceKind.TOOL)),
    EventKind.MODEL_INVOCATION: frozenset((SourceKind.SYSTEM,)),
    EventKind.CAPABILITY_GRANTED: frozenset((SourceKind.USER,)),
    EventKind.CAPABILITY_REVOKED: frozenset((SourceKind.USER,)),
    EventKind.TOOL_CALL_PROPOSED: frozenset((SourceKind.MODEL,)),
    EventKind.TOOL_EXECUTION_CONFIRMED: frozenset((SourceKind.USER,)),
    EventKind.TOOL_EXECUTION_STARTED: frozenset((SourceKind.SYSTEM,)),
    EventKind.TOOL_RESULT: frozenset((
        SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.TOOL_EXECUTION_ABANDONED: frozenset((SourceKind.SYSTEM,)),
    EventKind.ACTION_PROPOSED: frozenset((SourceKind.MODEL,)),
    EventKind.ACTION_RESULT: frozenset((
        SourceKind.SYSTEM, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.DECISION: frozenset((SourceKind.MODEL, SourceKind.POLICY)),
    EventKind.SEED_PROPOSED: frozenset((SourceKind.MODEL,)),
    EventKind.SEED_APPROVED: frozenset((SourceKind.USER,)),
    EventKind.SEED_RETIRED: frozenset((SourceKind.USER,)),
    EventKind.SELF_CLAIM_PROPOSED: frozenset((SourceKind.MODEL, SourceKind.USER)),
    EventKind.SELF_CLAIM_APPROVED: frozenset((SourceKind.USER,)),
    EventKind.SELF_CLAIM_REVOKED: frozenset((SourceKind.USER,)),
    EventKind.CORRECTION: frozenset((
        SourceKind.USER, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.PURGE: frozenset((SourceKind.USER, SourceKind.SYSTEM)),
    EventKind.AUTONOMY_CONTROL: frozenset((SourceKind.USER, SourceKind.SYSTEM)),
    EventKind.LOOP_TICK: frozenset((SourceKind.SYSTEM,)),
    EventKind.AUTONOMY_STOPPED: frozenset((SourceKind.SYSTEM, SourceKind.USER)),
    EventKind.METACOGNITIVE_MIRROR: frozenset((SourceKind.SYSTEM,)),
    EventKind.REWARD_OBSERVATION: frozenset((
        SourceKind.USER, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.VALUE_ESTIMATE: frozenset((SourceKind.MODEL, SourceKind.SYSTEM)),
    EventKind.RPE_UPDATE: frozenset((SourceKind.SYSTEM,)),
    EventKind.FRONTIER_RANKING_DECISION: frozenset((SourceKind.SYSTEM,)),
    EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE: frozenset((SourceKind.SYSTEM,)),
    EventKind.FRONTIER_EVIDENCE_OBSERVATION: frozenset((
        SourceKind.USER, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
    )),
    EventKind.FRONTIER_VECTOR_REWARD: frozenset((SourceKind.SYSTEM,)),
    EventKind.FRONTIER_TD_UPDATE: frozenset((SourceKind.SYSTEM,)),
    EventKind.EXPERIMENT_PLAN_LOCKED: frozenset((SourceKind.SYSTEM,)),
    EventKind.EXPERIMENT_EXECUTION_STARTED: frozenset((SourceKind.SYSTEM,)),
    EventKind.EXPERIMENT_RESULT: frozenset((SourceKind.EXTERNAL_VERIFIER,)),
    EventKind.SLEEP_ARCHIVE: frozenset((SourceKind.SYSTEM,)),
    EventKind.SLEEP_ENTERED: frozenset((SourceKind.SYSTEM,)),
    EventKind.PROVIDER_USAGE_EVIDENCE: frozenset((SourceKind.EXTERNAL_VERIFIER,)),
    EventKind.WAKE_CHECK: frozenset((SourceKind.SYSTEM,)),
    EventKind.WAKE_READY: frozenset((SourceKind.SYSTEM,)),
    EventKind.AWAKE: frozenset((SourceKind.SYSTEM,)),
    EventKind.WAKE_TERMINAL: frozenset((SourceKind.SYSTEM,)),
    EventKind.AUTO_WAKE_POLICY: frozenset((SourceKind.USER,)),
    EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY: frozenset((SourceKind.USER,)),
    EventKind.UNATTENDED_WAKE_RUN: frozenset((SourceKind.SYSTEM,)),
    EventKind.EXPEDITION_AUTHORIZATION: frozenset((SourceKind.USER,)),
    EventKind.EXPEDITION_AUTHORIZATION_CONSUMED: frozenset((SourceKind.SYSTEM,)),
}

# Persisted records are intentionally small public projections.  This is a
# schema boundary, not merely a list of forbidden reasoning-key substrings.
PAYLOAD_KEYS = {
    EventKind.OBSERVATION: frozenset(("content", "channel", "approval", "seed_id", "claim_id", "proposal_event_id")),
    EventKind.MEDIA_OBSERVATION: frozenset(("artifact_id", "modality", "sha256", "mime_type", "byte_length", "received_at", "duration_ms", "retention_scope")),
    EventKind.PERCEPT: frozenset(("percept_id", "artifact_id", "artifact_sha256", "modality", "span_start_ms", "span_end_ms", "percept_kind", "value", "confidence", "adapter_id", "adapter_version")),
    EventKind.MODEL_INVOCATION: frozenset(("invocation_id", "run_id", "trigger_event_id", "provider", "model", "role", "outcome", "latency_ms", "context_scope", "public_summary")),
    EventKind.CAPABILITY_GRANTED: frozenset(("grant_id", "capability", "tool_name", "scope_digest", "not_before", "expires_at", "max_uses", "max_input_bytes", "max_output_bytes", "max_wall_ms", "allow_mutating", "grant_version")),
    EventKind.CAPABILITY_REVOKED: frozenset(("revocation_id", "grant_id", "reason", "revoked_at")),
    EventKind.TOOL_CALL_PROPOSED: frozenset(("call_id", "run_id", "grant_id", "tool_name", "operation", "plan_digest", "input_digest", "requested_input_bytes", "requested_output_bytes", "requested_wall_ms", "mutating", "public_summary")),
    EventKind.TOOL_EXECUTION_CONFIRMED: frozenset(("confirmation_id", "call_id", "run_id", "grant_id", "plan_digest", "confirmed_at")),
    EventKind.TOOL_EXECUTION_STARTED: frozenset(("execution_id", "call_id", "run_id", "grant_id", "plan_digest", "confirmation_event_id", "started_at", "deadline_at")),
    # The first four fields are the legacy schema.  The execution-bound schema
    # below is deliberately separate so existing, non-executing ledger users
    # remain readable while K3 cannot omit its authorization lineage.
    EventKind.TOOL_RESULT: frozenset(("tool_name", "outcome", "result_ref", "summary", "execution_id", "call_id", "run_id", "grant_id", "plan_digest", "result_digest", "result_bytes", "latency_ms")),
    EventKind.TOOL_EXECUTION_ABANDONED: frozenset(("execution_id", "call_id", "run_id", "grant_id", "plan_digest", "reason", "abandoned_at", "public_summary")),
    EventKind.ACTION_PROPOSED: frozenset(("action_type", "required_capability", "is_mutating", "public_summary")),
    EventKind.ACTION_RESULT: frozenset(("action_type", "outcome", "response_text")),
    EventKind.DECISION: frozenset(("turn_id", "observation_event_ids", "retrieved_seed_ids", "self_claim_ids",
                                  "selected_action", "public_summary", "policy_reasons")),
    EventKind.SEED_PROPOSED: frozenset(("seed",)),
    EventKind.SEED_APPROVED: frozenset(("seed_id", "proposal_event_id", "approval_event_id")),
    EventKind.SEED_RETIRED: frozenset(("seed_id", "reason")),
    EventKind.SELF_CLAIM_PROPOSED: frozenset(("claim", "state")),
    EventKind.SELF_CLAIM_APPROVED: frozenset(("claim", "proposal_event_id", "approval_event_id")),
    EventKind.SELF_CLAIM_REVOKED: frozenset(("claim_id", "reason")),
    EventKind.CORRECTION: frozenset(("target_event_id", "counterevidence_event_id", "disposition", "public_summary")),
    EventKind.PURGE: frozenset(("session_id", "scope")),
    EventKind.AUTONOMY_CONTROL: frozenset(("run_id", "action", "config_version", "max_ticks", "max_wall_seconds", "max_events_per_tick", "max_no_progress", "min_interval_seconds", "next_tick_index")),
    EventKind.LOOP_TICK: frozenset(("run_id", "tick_index", "trigger", "phase", "focus_event_ids", "retrieved_event_ids", "progress", "salience_reason", "budget_remaining", "public_summary")),
    EventKind.AUTONOMY_STOPPED: frozenset(("run_id", "reason", "tick_count", "stopped_at")),
    EventKind.METACOGNITIVE_MIRROR: frozenset((
        "mirror_id", "episode_id", "target_event_id", "judgment_event_id",
        "evidence_event_ids", "self_status", "self_confidence",
        "self_uncertainty", "meta_status", "meta_confidence_cap",
        "check_codes", "disposition", "method_version", "public_summary",
    )),
    EventKind.REWARD_OBSERVATION: frozenset(("reward_id", "target_event_id", "signal_kind", "normalized_value", "evaluator_ref", "scale_version", "observed_at")),
    EventKind.VALUE_ESTIMATE: frozenset(("estimate_id", "transition_id", "target_event_id", "state_key", "action_key", "next_state_key", "terminal", "value", "confidence", "estimator_id", "estimator_version", "alpha", "gamma", "clip", "max_entries", "max_events", "formula_version")),
    EventKind.RPE_UPDATE: frozenset(("update_id", "transition_id", "reward_event_id", "prior_value_event_id", "state_key", "action_key", "reward", "prior_value", "next_value", "alpha", "gamma", "raw_delta", "clipped_delta", "clip", "updated_value", "formula_version", "scope")),
    EventKind.FRONTIER_RANKING_DECISION: frozenset((
        "ranking_id", "transition_id", "frontier_task_id", "candidate_digest",
        "learner_spec_digest", "channel_order", "scope", "ranking_version",
        "strategy_arm_id", "strategy_arm_version",
    )),
    EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE: frozenset((
        "estimate_id", "transition_id", "ranking_event_id", "learner_spec_digest",
        "channel_order", "value_vector", "alpha", "gamma", "clip", "scope",
        "formula_version", "strategy_arm_id", "strategy_arm_version",
    )),
    EventKind.FRONTIER_EVIDENCE_OBSERVATION: frozenset((
        "evidence_id", "transition_id", "ranking_event_id", "source_event_id",
        "evidence_digest", "evidence_kind", "learner_spec_digest", "scope",
        "evidence_version", "strategy_arm_id", "strategy_arm_version",
    )),
    EventKind.FRONTIER_VECTOR_REWARD: frozenset((
        "reward_id", "transition_id", "ranking_event_id", "evidence_event_id",
        "value_event_id", "learner_spec_digest", "channel_order", "reward_vector",
        "scope", "reward_version", "strategy_arm_id", "strategy_arm_version",
    )),
    EventKind.FRONTIER_TD_UPDATE: frozenset((
        "update_id", "transition_id", "ranking_event_id", "reward_event_id",
        "prior_value_event_id", "learner_spec_digest", "channel_order",
        "reward_vector", "prior_value_vector", "next_value_vector", "raw_delta_vector",
        "clipped_delta_vector", "updated_value_vector", "alpha", "gamma", "clip",
        "scope", "formula_version", "strategy_arm_id", "strategy_arm_version",
    )),
    EventKind.EXPERIMENT_PLAN_LOCKED: frozenset((
        "experiment_id", "authorization_id", "learner_spec_digest", "experiment_registry_digest", "experiment_instance_digest",
        "experiment_kind", "hypothesis_digest", "baseline_digest", "treatment_digest",
        "input_digest", "evaluator_digest", "host_seed_digest", "budget_digest", "max_trials",
        "max_steps", "max_wall_ms", "no_external_side_effects", "claim_scope", "template_version",
    )),
    EventKind.EXPERIMENT_EXECUTION_STARTED: frozenset((
        "execution_id", "experiment_id", "plan_event_id", "experiment_registry_digest", "experiment_instance_digest",
        "execution_nonce", "started_at", "max_trials", "max_steps", "max_wall_ms", "execution_version",
    )),
    EventKind.EXPERIMENT_RESULT: frozenset((
        "result_id", "execution_id", "experiment_id", "plan_event_id", "authorization_id",
        "learner_spec_digest", "experiment_registry_digest", "experiment_instance_digest", "status", "complete_run_set",
        "baseline_control_valid", "reproducible", "metric_digest", "metric_values", "effect_values",
        "result_digest", "reproduction_digest", "invariant_codes", "claim_scope", "result_version",
    )),
    EventKind.SLEEP_ARCHIVE: frozenset(("archive_id", "epoch_id", "schema_version",
        "chain_head_sequence", "chain_head_hash", "approved_seed_ids", "approved_claim_ids",
        "pending_task_ids", "quota_source", "quota_observed_at", "quota_reset_at",
        "quota_remaining", "quota_total", "quota_window_kind", "archive_digest", "event_count", "archive_version")),
    EventKind.SLEEP_ENTERED: frozenset(("sleep_id", "archive_event_id", "epoch_id",
        "auto_wake_policy_event_id", "continuation_policy_event_id", "slept_at", "protocol_version")),
    EventKind.PROVIDER_USAGE_EVIDENCE: frozenset(("usage_evidence_id", "provider",
        "quota_source", "observed_at", "reset_at", "total", "remaining",
        "window_kind", "evidence_digest", "evidence_version")),
    EventKind.WAKE_CHECK: frozenset(("wake_check_id", "archive_event_id", "sleep_event_id",
        "epoch_id", "usage_evidence_event_id", "checked_at", "check_version")),
    EventKind.WAKE_READY: frozenset(("wake_ready_id", "wake_check_event_id", "epoch_id",
        "ready_at", "protocol_version")),
    EventKind.AWAKE: frozenset(("awake_id", "wake_ready_event_id", "epoch_id",
        "awakened_at", "protocol_version")),
    EventKind.WAKE_TERMINAL: frozenset(("wake_terminal_id", "lifecycle_event_id", "epoch_id",
        "terminal_at", "outcome", "protocol_version")),
    EventKind.AUTO_WAKE_POLICY: frozenset(("policy_id", "action", "scope", "max_auto_runs",
        "max_ticks", "expires_at", "issued_at", "policy_version")),
    EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY: frozenset((
        "continuation_id", "action", "scope", "approval_event_id",
        "unattended_policy_id", "focus_digest", "profile_digest",
        "max_auto_runs", "expires_at", "issued_at", "policy_version",
    )),
    EventKind.UNATTENDED_WAKE_RUN: frozenset((
        "continuation_id", "awake_event_id", "profile_id", "authorization_policy_id", "runtime_policy_id",
        "run_index", "status", "protocol_version",
    )),
    EventKind.EXPEDITION_AUTHORIZATION: frozenset((
        "authorization_id", "approval_event_id", "nonce", "issued_at", "expires_at",
        "goal_digest", "host_seed_digest", "max_calls_per_slice", "slice_seconds",
        "authorization_seconds", "profile", "version", "learning_mode", "learner_spec_digest",
        "experiment_kinds", "experiment_registry_digest", "max_experiments",
        "experiment_max_trials", "experiment_max_steps", "experiment_max_wall_ms",
    )),
    EventKind.EXPEDITION_AUTHORIZATION_CONSUMED: frozenset((
        "consumption_id", "authorization_id", "consumed_at", "run_id", "version",
    )),
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIME = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
TD_SAFE_ACTION_CLASSES = frozenset((
    "ask_clarifying_question", "record_observation", "respond",
    "retrieve_user_approved_memory", "summarize", "wait",
))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class SQLiteEventStore:
    """Session-local hash chains with transactional projection support.

    Only observations are permitted as root events.  Other event kinds must
    name prior, same-session event IDs as parents.
    """

    def __init__(self, path: str = ":memory:", session_id: Optional[str] = None) -> None:
        self.path = path
        self.session_id = session_id
        self._is_file = path != ":memory:"
        self.connection = sqlite3.connect(path, timeout=0.1, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.execute("PRAGMA secure_delete = ON")
        if self._is_file:
            self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS cognitive_events (
                event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL, kind TEXT NOT NULL,
                source_kind TEXT NOT NULL, source_ref TEXT NOT NULL,
                payload_json TEXT NOT NULL, confidence REAL NOT NULL,
                parent_event_ids_json TEXT NOT NULL, created_at TEXT NOT NULL,
                previous_hash TEXT, content_hash TEXT NOT NULL,
                UNIQUE(session_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_events_session_sequence
                ON cognitive_events(session_id, sequence);
        """)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a write transaction, retrying lock acquisition for file stores."""
        if self.connection.in_transaction:
            yield
            return
        last_error = None
        for attempt in range(8):
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as error:
                last_error = error
                if "locked" not in str(error).lower() or attempt == 7:
                    raise
                time.sleep(0.01 * (attempt + 1))
        else:  # pragma: no cover - defensive; loop either breaks or raises
            raise last_error  # type: ignore[misc]
        try:
            yield
            self.connection.commit()
        except BaseException:
            # KeyboardInterrupt/SystemExit are just as capable of leaving a
            # partial write and lock behind as ordinary exceptions.  Preserve
            # the original failure even if the best-effort rollback itself
            # encounters a damaged/closed connection.
            try:
                self.connection.rollback()
            except BaseException:
                pass
            raise

    @staticmethod
    def _has_forbidden_key(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in FORBIDDEN_PAYLOAD_KEYS or SQLiteEventStore._has_forbidden_key(nested):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(SQLiteEventStore._has_forbidden_key(item) for item in value)
        return False

    @staticmethod
    def _hash_material(event: CognitiveEvent) -> Dict[str, Any]:
        return {"event_id": event.event_id, "session_id": event.session_id,
                "sequence": event.sequence, "kind": event.kind.value,
                "source": {"kind": event.source_kind.value, "ref": event.source_ref},
                "payload": event.payload, "confidence": event.confidence,
                "parent_event_ids": list(event.parent_event_ids),
                "created_at": event.created_at, "previous_hash": event.previous_hash}

    @classmethod
    def _content_hash(cls, event: CognitiveEvent) -> str:
        return hashlib.sha256(canonical_json(cls._hash_material(event)).encode("utf-8")).hexdigest()

    def _validate_event(self, event: CognitiveEvent, sequence: int) -> None:
        if event.sequence is not None or event.previous_hash is not None or event.content_hash is not None:
            raise ValueError("append accepts an unpersisted CognitiveEvent")
        if event.kind == EventKind.INFERENCE:
            raise ValueError("inference persistence is not implemented")
        if self.session_id is not None and event.session_id != self.session_id:
            raise ValueError("event session does not match this session-bound store")
        if event.source_kind not in ALLOWED_EVENT_SOURCES[event.kind]:
            raise ValueError("event source is not allowed for kind %s" % event.kind.value)
        encoded = canonical_json(event.payload).encode("utf-8")
        if len(encoded) > MAX_PAYLOAD_BYTES:
            raise ValueError("payload exceeds the structured-record size limit")
        if self._has_forbidden_key(event.payload):
            raise ValueError("payload contains a prohibited hidden-reasoning field")
        self._validate_payload_schema(event)
        self._validate_protocol_uniqueness(event, sequence)
        if not event.parent_event_ids:
            if event.kind not in ROOT_EVENT_KINDS:
                raise ValueError("only observation events may be roots")
            if event.kind == EventKind.AUTONOMY_CONTROL and event.source_kind != SourceKind.USER:
                raise ValueError("only user autonomy controls may be root events")
            if event.kind == EventKind.SLEEP_ARCHIVE:
                if event.source_ref != "SleepWakeCoordinator":
                    raise ValueError("sleep archive source_ref must be SleepWakeCoordinator")
                self._validate_sleep_archive_head(event, sequence)
            return
        if len(set(event.parent_event_ids)) != len(event.parent_event_ids):
            raise ValueError("parent event IDs must be unique")
        placeholders = ",".join("?" for _ in event.parent_event_ids)
        parents = self.connection.execute(
            "SELECT event_id, session_id, sequence FROM cognitive_events "
            "WHERE event_id IN (" + placeholders + ")", tuple(event.parent_event_ids)
        ).fetchall()
        if len(parents) != len(event.parent_event_ids):
            raise ValueError("all parent event IDs must exist")
        for parent in parents:
            if parent["session_id"] != event.session_id or int(parent["sequence"]) >= sequence:
                raise ValueError("parents must be prior events in the same session")
        parent_events = {row["event_id"]: self._from_row(row) for row in self.connection.execute(
            "SELECT * FROM cognitive_events WHERE event_id IN (" + placeholders + ")",
            tuple(event.parent_event_ids),
        ).fetchall()}
        self._validate_lineage(event, parent_events)

    def _validate_protocol_uniqueness(self, event: CognitiveEvent,
                                      sequence: int) -> None:
        """Validate the one-prediction/one-reward/one-update v1 protocol.

        ``append`` calls this while holding ``BEGIN IMMEDIATE``.  Scanning the
        small public payload projection avoids relying on SQLite's optional
        JSON extension and keeps Python 3.9 deployments portable.
        """
        def records(kind: EventKind):
            rows = self.connection.execute(
                "SELECT sequence, payload_json FROM cognitive_events "
                "WHERE session_id = ? AND kind = ?",
                (event.session_id, kind.value),
            ).fetchall()
            return [(int(row["sequence"]), json.loads(row["payload_json"]))
                    for row in rows]

        unique_fields = {
            EventKind.MODEL_INVOCATION: "invocation_id",
            EventKind.CAPABILITY_GRANTED: "grant_id",
            EventKind.CAPABILITY_REVOKED: "grant_id",
            EventKind.TOOL_CALL_PROPOSED: "call_id",
            EventKind.TOOL_EXECUTION_CONFIRMED: "call_id",
            EventKind.TOOL_EXECUTION_STARTED: "execution_id",
            EventKind.TOOL_EXECUTION_ABANDONED: "execution_id",
            EventKind.METACOGNITIVE_MIRROR: "mirror_id",
            EventKind.SLEEP_ARCHIVE: "archive_id",
            EventKind.SLEEP_ENTERED: "sleep_id",
            EventKind.PROVIDER_USAGE_EVIDENCE: "usage_evidence_id",
            EventKind.WAKE_CHECK: "wake_check_id",
            EventKind.WAKE_READY: "wake_ready_id",
            EventKind.AWAKE: "awake_id",
            EventKind.WAKE_TERMINAL: "wake_terminal_id",
            EventKind.AUTO_WAKE_POLICY: "policy_id",
            EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY: "continuation_id",
            EventKind.UNATTENDED_WAKE_RUN: "continuation_id",
            EventKind.EXPEDITION_AUTHORIZATION: "authorization_id",
            EventKind.EXPEDITION_AUTHORIZATION_CONSUMED: "consumption_id",
            EventKind.FRONTIER_RANKING_DECISION: "ranking_id",
            EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE: "estimate_id",
            EventKind.FRONTIER_EVIDENCE_OBSERVATION: "evidence_id",
            EventKind.FRONTIER_VECTOR_REWARD: "reward_id",
    EventKind.FRONTIER_TD_UPDATE: "update_id",
            EventKind.EXPERIMENT_PLAN_LOCKED: "experiment_id",
            EventKind.EXPERIMENT_EXECUTION_STARTED: "execution_id",
            EventKind.EXPERIMENT_RESULT: "result_id",
        }
        if event.kind == EventKind.TOOL_RESULT and "execution_id" in event.payload:
            if any(payload.get("execution_id") == event.payload["execution_id"]
                   for _, payload in records(EventKind.TOOL_RESULT)):
                raise ValueError("tool execution already has a result")
        elif event.kind in unique_fields:
            field = unique_fields[event.kind]
            if any(payload.get(field) == event.payload[field]
                   for _, payload in records(event.kind)):
                raise ValueError("K3 protocol identifier has already been recorded")
            if event.kind == EventKind.METACOGNITIVE_MIRROR:
                if any(payload.get("episode_id") == event.payload["episode_id"]
                       for _, payload in records(event.kind)):
                    raise ValueError("metacognitive mirror episode_id has already been recorded")
            if event.kind == EventKind.SLEEP_ARCHIVE:
                if any(payload.get("epoch_id") == event.payload["epoch_id"]
                       for _, payload in records(EventKind.SLEEP_ARCHIVE)):
                    raise ValueError("sleep epoch has already been archived in this session")
            if event.kind == EventKind.WAKE_CHECK:
                if any(payload.get("archive_event_id") == event.payload["archive_event_id"]
                       for _, payload in records(EventKind.WAKE_CHECK)):
                    raise ValueError("sleep archive already has a wake check")
            if event.kind == EventKind.UNATTENDED_WAKE_RUN:
                # A continuation produces exactly one terminal public record;
                # there is intentionally no separate "started" record whose
                # logical ID could collide with its terminal outcome.
                if any(payload.get("continuation_id") == event.payload["continuation_id"]
                       for _, payload in records(EventKind.UNATTENDED_WAKE_RUN)):
                    raise ValueError("continuation already has an unattended wake run")
            if event.kind == EventKind.EXPEDITION_AUTHORIZATION:
                if any(payload.get("nonce") == event.payload["nonce"]
                       for _, payload in records(EventKind.EXPEDITION_AUTHORIZATION)):
                    raise ValueError("expedition authorization nonce already exists in this session")
            if event.kind == EventKind.EXPEDITION_AUTHORIZATION_CONSUMED:
                if any(payload.get("authorization_id") == event.payload["authorization_id"]
                       for _, payload in records(EventKind.EXPEDITION_AUTHORIZATION_CONSUMED)):
                    raise ValueError("expedition authorization has already been consumed")
            if event.kind == EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE:
                if any(payload.get("ranking_event_id") == event.payload["ranking_event_id"]
                       for _, payload in records(EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE)):
                    raise ValueError("frontier ranking already has a vector value estimate")
            if event.kind == EventKind.FRONTIER_EVIDENCE_OBSERVATION:
                if any(payload.get("source_event_id") == event.payload["source_event_id"]
                       for _, payload in records(EventKind.FRONTIER_EVIDENCE_OBSERVATION)):
                    raise ValueError("frontier source evidence already has an observation")
            if event.kind == EventKind.FRONTIER_VECTOR_REWARD:
                if any(payload.get("evidence_event_id") == event.payload["evidence_event_id"]
                       for _, payload in records(EventKind.FRONTIER_VECTOR_REWARD)):
                    raise ValueError("frontier evidence already has a vector reward")
            if event.kind == EventKind.FRONTIER_TD_UPDATE:
                if any(payload.get("reward_event_id") == event.payload["reward_event_id"]
                       for _, payload in records(EventKind.FRONTIER_TD_UPDATE)):
                    raise ValueError("frontier reward has already been consumed by a TD update")
            if event.kind == EventKind.EXPERIMENT_RESULT:
                if any(payload.get("execution_id") == event.payload["execution_id"]
                       for _, payload in records(EventKind.EXPERIMENT_RESULT)):
                    raise ValueError("experiment execution already has a verifier result")
            successor_fields = {
                EventKind.WAKE_READY: "wake_check_event_id",
                EventKind.AWAKE: "wake_ready_event_id",
                EventKind.WAKE_TERMINAL: "lifecycle_event_id",
            }
            predecessor = successor_fields.get(event.kind)
            if predecessor and any(payload.get(predecessor) == event.payload[predecessor]
                                  for _, payload in records(event.kind)):
                raise ValueError("lifecycle predecessor already has a successor")
        elif event.kind == EventKind.VALUE_ESTIMATE:
            target = event.payload["target_event_id"]
            existing_values = records(EventKind.VALUE_ESTIMATE)
            if any(payload.get("target_event_id") == target
                   for _, payload in existing_values):
                raise ValueError("target already has a value estimate in this session")
            if existing_values:
                first = existing_values[0][1]
                if (event.payload["max_entries"] != first["max_entries"]
                        or event.payload["max_events"] != first["max_events"]):
                    raise ValueError("value table capacity must match the first estimate in this session")
        elif event.kind == EventKind.REWARD_OBSERVATION:
            rewards = records(EventKind.REWARD_OBSERVATION)
            if any(payload.get("reward_id") == event.payload["reward_id"]
                   for _, payload in rewards):
                raise ValueError("reward_id has already been recorded in this session")
            target = event.payload["target_event_id"]
            if any(payload.get("target_event_id") == target
                   for _, payload in rewards):
                raise ValueError("target already has a reward observation in this session")
            predictions = [(stored_sequence, payload) for stored_sequence, payload
                           in records(EventKind.VALUE_ESTIMATE)
                           if payload.get("target_event_id") == target]
            if len(predictions) != 1 or predictions[0][0] >= sequence:
                raise ValueError("reward requires exactly one prior value estimate for its target")
        elif event.kind == EventKind.RPE_UPDATE:
            if any(payload.get("reward_event_id") == event.payload["reward_event_id"]
                   for _, payload in records(EventKind.RPE_UPDATE)):
                raise ValueError("reward event has already been consumed by an RPE update")

    @staticmethod
    def _validate_payload_schema(event: CognitiveEvent) -> None:
        keys = set(event.payload)
        allowed = PAYLOAD_KEYS[event.kind]
        if not keys.issubset(allowed):
            raise ValueError("payload contains keys not allowed for %s" % event.kind.value)
        if event.kind == EventKind.OBSERVATION:
            if event.payload.get("approval") == "seed":
                required = {"approval", "seed_id", "proposal_event_id"}
                if keys != required:
                    raise ValueError("seed approval observation must use its fixed schema")
                SQLiteEventStore._strings(event.payload, required, MAX_SHORT_TEXT)
            elif event.payload.get("approval") == "self_claim":
                required = {"approval", "claim_id", "proposal_event_id"}
                if keys != required:
                    raise ValueError("self-claim approval observation must use its fixed schema")
                SQLiteEventStore._strings(event.payload, required, MAX_SHORT_TEXT)
            elif "approval" in keys:
                raise ValueError("observation approval type is not recognized")
            elif keys not in ({"content"}, {"content", "channel"}):
                raise ValueError("observation must contain only content and optional channel")
            else:
                SQLiteEventStore._strings(event.payload, keys, MAX_TEXT)
        elif event.kind == EventKind.MODEL_INVOCATION:
            required = {"invocation_id", "run_id", "trigger_event_id", "provider", "model",
                        "role", "outcome", "latency_ms", "context_scope", "public_summary"}
            if keys != required:
                raise ValueError("model invocation must use its fixed public schema")
            for key in ("invocation_id", "trigger_event_id", "provider", "model"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["run_id"] is not None:
                SQLiteEventStore._id(event.payload["run_id"], "run_id")
            if event.payload["role"] not in {"deliberation", "tool_planning", "perception",
                                               "response_generation", "summarization"}:
                raise ValueError("model invocation role is invalid")
            if event.payload["outcome"] not in {"started", "completed", "refused", "failed",
                                                 "timed_out", "budget_exhausted"}:
                raise ValueError("model invocation outcome is invalid")
            if event.payload["context_scope"] not in {"text_turn", "image_analysis", "loop_tick",
                                                        "tool_planning", "tool_followup"}:
                raise ValueError("model invocation context_scope is invalid")
            SQLiteEventStore._integer(event.payload["latency_ms"], "latency_ms", 0, 24 * 60 * 60 * 1000)
            SQLiteEventStore._bounded_text(event.payload["public_summary"], "public_summary", MAX_TEXT)
            SQLiteEventStore._id(event.source_ref, "model invocation source_ref")
        elif event.kind == EventKind.CAPABILITY_GRANTED:
            required = {"grant_id", "capability", "tool_name", "scope_digest", "not_before",
                        "expires_at", "max_uses", "max_input_bytes", "max_output_bytes",
                        "max_wall_ms", "allow_mutating", "grant_version"}
            if keys != required:
                raise ValueError("capability grant must use its fixed schema")
            for key in ("grant_id", "capability", "tool_name", "grant_version"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["scope_digest"])
            SQLiteEventStore._timestamp(event.payload["not_before"], "not_before")
            SQLiteEventStore._timestamp(event.payload["expires_at"], "expires_at")
            if parse_aware_iso8601(event.payload["expires_at"]) <= parse_aware_iso8601(event.payload["not_before"]):
                raise ValueError("capability grant expiry must be after not_before")
            SQLiteEventStore._integer(event.payload["max_uses"], "max_uses", 1, 10000)
            SQLiteEventStore._integer(event.payload["max_input_bytes"], "max_input_bytes", 0, 10 * 1024 * 1024)
            SQLiteEventStore._integer(event.payload["max_output_bytes"], "max_output_bytes", 0, 10 * 1024 * 1024)
            SQLiteEventStore._integer(event.payload["max_wall_ms"], "max_wall_ms", 1, 24 * 60 * 60 * 1000)
            if not isinstance(event.payload["allow_mutating"], bool):
                raise ValueError("capability grant allow_mutating must be a boolean")
        elif event.kind == EventKind.CAPABILITY_REVOKED:
            required = {"revocation_id", "grant_id", "reason", "revoked_at"}
            if keys != required:
                raise ValueError("capability revocation must use its fixed schema")
            for key in ("revocation_id", "grant_id"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["reason"] not in {"user_requested", "scope_changed", "expired", "safety_stop"}:
                raise ValueError("capability revocation reason is invalid")
            SQLiteEventStore._timestamp(event.payload["revoked_at"], "revoked_at")
        elif event.kind == EventKind.TOOL_CALL_PROPOSED:
            required = {"call_id", "run_id", "grant_id", "tool_name", "operation", "plan_digest",
                        "input_digest", "requested_input_bytes", "requested_output_bytes",
                        "requested_wall_ms", "mutating", "public_summary"}
            if keys != required:
                raise ValueError("tool proposal must use its fixed schema")
            for key in ("call_id", "run_id", "grant_id", "tool_name", "operation"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["plan_digest"])
            SQLiteEventStore._sha256(event.payload["input_digest"])
            SQLiteEventStore._integer(event.payload["requested_input_bytes"], "requested_input_bytes", 0, 10 * 1024 * 1024)
            SQLiteEventStore._integer(event.payload["requested_output_bytes"], "requested_output_bytes", 0, 10 * 1024 * 1024)
            SQLiteEventStore._integer(event.payload["requested_wall_ms"], "requested_wall_ms", 1, 24 * 60 * 60 * 1000)
            if not isinstance(event.payload["mutating"], bool):
                raise ValueError("tool proposal mutating must be a boolean")
            SQLiteEventStore._bounded_text(event.payload["public_summary"], "public_summary", MAX_TEXT)
        elif event.kind == EventKind.TOOL_EXECUTION_CONFIRMED:
            required = {"confirmation_id", "call_id", "run_id", "grant_id", "plan_digest", "confirmed_at"}
            if keys != required:
                raise ValueError("tool execution confirmation must use its fixed schema")
            for key in ("confirmation_id", "call_id", "run_id", "grant_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["plan_digest"])
            SQLiteEventStore._timestamp(event.payload["confirmed_at"], "confirmed_at")
        elif event.kind == EventKind.TOOL_EXECUTION_STARTED:
            required = {"execution_id", "call_id", "run_id", "grant_id", "plan_digest",
                        "confirmation_event_id", "started_at", "deadline_at"}
            if keys != required:
                raise ValueError("tool execution start must use its fixed schema")
            for key in ("execution_id", "call_id", "run_id", "grant_id"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["confirmation_event_id"] is not None:
                SQLiteEventStore._id(event.payload["confirmation_event_id"], "confirmation_event_id")
            SQLiteEventStore._sha256(event.payload["plan_digest"])
            SQLiteEventStore._timestamp(event.payload["started_at"], "started_at")
            SQLiteEventStore._timestamp(event.payload["deadline_at"], "deadline_at")
            if parse_aware_iso8601(event.payload["deadline_at"]) <= parse_aware_iso8601(event.payload["started_at"]):
                raise ValueError("tool execution deadline must be after start")
        elif event.kind == EventKind.TOOL_RESULT:
            execution_required = {"execution_id", "call_id", "run_id", "grant_id", "plan_digest",
                                  "tool_name", "outcome", "result_digest", "result_bytes",
                                  "latency_ms", "summary"}
            if keys == execution_required:
                for key in ("execution_id", "call_id", "run_id", "grant_id", "tool_name"):
                    SQLiteEventStore._id(event.payload[key], key)
                SQLiteEventStore._sha256(event.payload["plan_digest"])
                SQLiteEventStore._sha256(event.payload["result_digest"])
                if event.payload["outcome"] not in {"succeeded", "failed", "timed_out", "cancelled"}:
                    raise ValueError("tool execution result outcome is invalid")
                SQLiteEventStore._integer(event.payload["result_bytes"], "result_bytes", 0, 10 * 1024 * 1024)
                SQLiteEventStore._integer(event.payload["latency_ms"], "latency_ms", 0, 24 * 60 * 60 * 1000)
                SQLiteEventStore._bounded_text(event.payload["summary"], "summary", MAX_TEXT)
                return
            if keys not in ({"tool_name", "outcome"}, {"tool_name", "outcome", "result_ref"},
                            {"tool_name", "outcome", "summary"}, {"tool_name", "outcome", "result_ref", "summary"}):
                raise ValueError("tool result must use its fixed schema")
            SQLiteEventStore._strings(event.payload, keys, MAX_TEXT)
        elif event.kind == EventKind.TOOL_EXECUTION_ABANDONED:
            required = {"execution_id", "call_id", "run_id", "grant_id", "plan_digest",
                        "reason", "abandoned_at", "public_summary"}
            if keys != required:
                raise ValueError("tool execution abandonment must use its fixed schema")
            for key in ("execution_id", "call_id", "run_id", "grant_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["plan_digest"])
            if event.payload["reason"] not in {"cancelled", "timed_out", "budget_exhausted", "shutdown", "failed"}:
                raise ValueError("tool execution abandonment reason is invalid")
            SQLiteEventStore._timestamp(event.payload["abandoned_at"], "abandoned_at")
            SQLiteEventStore._bounded_text(event.payload["public_summary"], "public_summary", MAX_TEXT)
        elif event.kind in (EventKind.SEED_PROPOSED,):
            if keys != {"seed"}:
                raise ValueError("seed proposal must use its fixed schema")
            SQLiteEventStore._seed(event.payload["seed"])
        elif event.kind == EventKind.SEED_APPROVED:
            if keys != {"seed_id", "proposal_event_id", "approval_event_id"}:
                raise ValueError("seed approval must use its fixed schema")
            SQLiteEventStore._strings(event.payload, keys, MAX_SHORT_TEXT)
        elif event.kind == EventKind.SEED_RETIRED:
            if keys != {"seed_id", "reason"}:
                raise ValueError("seed retirement must use its fixed schema")
            SQLiteEventStore._strings(event.payload, keys, MAX_SHORT_TEXT)
        elif event.kind == EventKind.SELF_CLAIM_PROPOSED:
            if keys != {"claim", "state"} or event.payload.get("state") != "candidate":
                raise ValueError("self-claim proposal must use its fixed candidate schema")
            SQLiteEventStore._claim(event.payload["claim"])
        elif event.kind == EventKind.SELF_CLAIM_APPROVED:
            if keys != {"claim", "proposal_event_id", "approval_event_id"}:
                raise ValueError("self-claim approval must use its fixed schema")
            SQLiteEventStore._claim(event.payload["claim"])
            SQLiteEventStore._strings(event.payload, {"proposal_event_id", "approval_event_id"}, MAX_SHORT_TEXT)
        elif event.kind == EventKind.SELF_CLAIM_REVOKED:
            if keys != {"claim_id", "reason"}:
                raise ValueError("self-claim revocation must use its fixed schema")
            SQLiteEventStore._strings(event.payload, keys, MAX_SHORT_TEXT)
        elif event.kind == EventKind.CORRECTION:
            if keys != {"target_event_id", "counterevidence_event_id", "disposition", "public_summary"}:
                raise ValueError("correction must use its fixed public schema")
            SQLiteEventStore._strings(event.payload, keys, MAX_SHORT_TEXT)
            if event.payload["disposition"] != "review_required":
                raise ValueError("correction disposition must be review_required")
            if event.payload["public_summary"] != "A later observable record conflicts with an earlier record.":
                raise ValueError("correction public_summary must use the fixed template")
        elif event.kind == EventKind.ACTION_PROPOSED:
            if keys != {"action_type", "required_capability", "is_mutating", "public_summary"}:
                raise ValueError("action proposal must use server projection schema")
            if (not isinstance(event.payload["action_type"], str) or event.payload["action_type"] not in {"response", "read", "write", "other"}
                    or event.payload["required_capability"] not in (None, "declared")
                    or not isinstance(event.payload["is_mutating"], bool)
                    or event.payload["public_summary"] != "Action proposal recorded for policy review."):
                raise ValueError("invalid action proposal projection")
        elif event.kind == EventKind.DECISION:
            required = {"turn_id", "observation_event_ids", "retrieved_seed_ids", "self_claim_ids", "selected_action", "public_summary", "policy_reasons"}
            if keys != required or event.payload["public_summary"] != "Policy decision recorded for this turn.":
                raise ValueError("decision must use server projection schema")
            SQLiteEventStore._strings(event.payload, {"turn_id", "public_summary"}, MAX_SHORT_TEXT)
            for key in ("observation_event_ids", "retrieved_seed_ids", "self_claim_ids", "policy_reasons"):
                SQLiteEventStore._string_list(event.payload[key])
            SQLiteEventStore._action(event.payload["selected_action"])
        elif event.kind == EventKind.ACTION_RESULT:
            if keys != {"action_type", "outcome", "response_text"}:
                raise ValueError("action result must use its fixed schema")
            SQLiteEventStore._strings(event.payload, keys, MAX_TEXT)
        elif event.kind == EventKind.MEDIA_OBSERVATION:
            required = {"artifact_id", "modality", "sha256", "mime_type", "byte_length", "received_at", "duration_ms", "retention_scope"}
            if keys != required:
                raise ValueError("media observation must use its fixed metadata schema")
            SQLiteEventStore._id(event.payload["artifact_id"], "artifact_id")
            if event.payload["modality"] not in {"image", "audio"}:
                raise ValueError("media modality must be image or audio")
            SQLiteEventStore._sha256(event.payload["sha256"])
            if not isinstance(event.payload["mime_type"], str) or not _MIME.fullmatch(event.payload["mime_type"]):
                raise ValueError("media mime_type is invalid")
            SQLiteEventStore._integer(event.payload["byte_length"], "byte_length", 0, 100 * 1024 * 1024)
            SQLiteEventStore._timestamp(event.payload["received_at"], "received_at")
            duration = event.payload["duration_ms"]
            if duration is not None:
                SQLiteEventStore._integer(duration, "duration_ms", 0, 24 * 60 * 60 * 1000)
            if event.payload["retention_scope"] not in {"ephemeral", "session", "user_approved"}:
                raise ValueError("media retention_scope is invalid")
        elif event.kind == EventKind.PERCEPT:
            required = {"percept_id", "artifact_id", "artifact_sha256", "modality", "span_start_ms", "span_end_ms", "percept_kind", "value", "confidence", "adapter_id", "adapter_version"}
            if keys != required:
                raise ValueError("percept must use its fixed schema")
            for key in ("percept_id", "artifact_id", "adapter_id", "adapter_version"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["artifact_sha256"])
            if event.payload["modality"] not in {"image", "audio"}:
                raise ValueError("percept modality is invalid")
            SQLiteEventStore._integer(event.payload["span_start_ms"], "span_start_ms", 0, 24 * 60 * 60 * 1000)
            SQLiteEventStore._integer(event.payload["span_end_ms"], "span_end_ms", 0, 24 * 60 * 60 * 1000)
            if event.payload["span_end_ms"] < event.payload["span_start_ms"]:
                raise ValueError("percept span must be ordered")
            SQLiteEventStore._enum_text(event.payload["percept_kind"], "percept_kind")
            SQLiteEventStore._bounded_text(event.payload["value"], "value", MAX_TEXT)
            SQLiteEventStore._number(event.payload["confidence"], "confidence", 0.0, 1.0)
        elif event.kind == EventKind.AUTONOMY_CONTROL:
            required = {"run_id", "action", "config_version", "max_ticks", "max_wall_seconds", "max_events_per_tick", "max_no_progress", "min_interval_seconds", "next_tick_index"}
            if keys != required:
                raise ValueError("autonomy control must use its fixed schema")
            SQLiteEventStore._id(event.payload["run_id"], "run_id")
            if event.payload["action"] not in {"start", "pause", "resume", "stop"}:
                raise ValueError("autonomy control action is invalid")
            SQLiteEventStore._id(event.payload["config_version"], "config_version")
            SQLiteEventStore._integer(event.payload["max_ticks"], "max_ticks", 1, 10000)
            SQLiteEventStore._integer(event.payload["max_wall_seconds"], "max_wall_seconds", 1, 86400)
            SQLiteEventStore._integer(event.payload["max_events_per_tick"], "max_events_per_tick", 1, 256)
            SQLiteEventStore._integer(event.payload["max_no_progress"], "max_no_progress", 0, 10000)
            SQLiteEventStore._number(event.payload["min_interval_seconds"], "min_interval_seconds", 0.0, 3600.0)
            SQLiteEventStore._integer(event.payload["next_tick_index"], "next_tick_index", 0, 1000000)
            if event.source_kind == SourceKind.SYSTEM:
                if (event.payload["action"] != "start"
                        or event.payload["next_tick_index"] != 0
                        or event.payload["max_ticks"] > 4
                        or event.payload["max_wall_seconds"] > 30
                        or event.payload["max_events_per_tick"] != 1
                        or event.payload["max_no_progress"] > 2):
                    raise ValueError("system wake-start control exceeds the fixed auto-wake bounds")
        elif event.kind == EventKind.LOOP_TICK:
            required = {"run_id", "tick_index", "trigger", "phase", "focus_event_ids", "retrieved_event_ids", "progress", "salience_reason", "budget_remaining", "public_summary"}
            if keys != required:
                raise ValueError("loop tick must use its fixed schema")
            SQLiteEventStore._id(event.payload["run_id"], "run_id")
            SQLiteEventStore._integer(event.payload["tick_index"], "tick_index", 0, 1000000)
            if event.payload["trigger"] not in {"timer", "media", "text", "control"}:
                raise ValueError("loop tick trigger is invalid")
            if event.payload["phase"] not in {"observe", "orient", "retrieve", "deliberate", "act", "reflect", "idle", "quota_paused"}:
                raise ValueError("loop tick phase is invalid")
            SQLiteEventStore._event_ids(event.payload["focus_event_ids"], "focus_event_ids")
            SQLiteEventStore._event_ids(event.payload["retrieved_event_ids"], "retrieved_event_ids")
            if event.payload["progress"] not in {"made_progress", "no_progress", "blocked"}:
                raise ValueError("loop tick progress is invalid")
            SQLiteEventStore._bounded_text(event.payload["salience_reason"], "salience_reason", MAX_SHORT_TEXT)
            SQLiteEventStore._integer(event.payload["budget_remaining"], "budget_remaining", 0, 1000000)
            SQLiteEventStore._bounded_text(event.payload["public_summary"], "public_summary", MAX_TEXT)
        elif event.kind == EventKind.AUTONOMY_STOPPED:
            required = {"run_id", "reason", "tick_count", "stopped_at"}
            if keys != required:
                raise ValueError("autonomy stopped must use its fixed schema")
            SQLiteEventStore._id(event.payload["run_id"], "run_id")
            if event.payload["reason"] not in {"user_requested", "budget_exhausted", "no_progress", "completed", "error"}:
                raise ValueError("autonomy stop reason is invalid")
            SQLiteEventStore._integer(event.payload["tick_count"], "tick_count", 0, 1000000)
            SQLiteEventStore._timestamp(event.payload["stopped_at"], "stopped_at")
        elif event.kind == EventKind.METACOGNITIVE_MIRROR:
            required = {"mirror_id", "episode_id", "target_event_id", "judgment_event_id",
                        "evidence_event_ids", "self_status", "self_confidence",
                        "self_uncertainty", "meta_status", "meta_confidence_cap",
                        "check_codes", "disposition", "method_version", "public_summary"}
            if keys != required:
                raise ValueError("metacognitive mirror must use its fixed public schema")
            for key in ("mirror_id", "episode_id", "target_event_id", "judgment_event_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._event_ids(event.payload["evidence_event_ids"], "evidence_event_ids")
            if len(event.payload["evidence_event_ids"]) < 2:
                raise ValueError("metacognitive mirror requires target and judgment evidence")
            if event.payload["self_status"] not in {"supported", "insufficient", "conflicted"}:
                raise ValueError("metacognitive mirror self_status is invalid")
            if event.payload["self_uncertainty"] not in {"low", "medium", "high"}:
                raise ValueError("metacognitive mirror self_uncertainty is invalid")
            if event.payload["meta_status"] not in {"confirmed", "limited", "conflicted"}:
                raise ValueError("metacognitive mirror meta_status is invalid")
            if event.payload["disposition"] not in {"provisional", "review_required", "abstain"}:
                raise ValueError("metacognitive mirror disposition is invalid")
            if event.payload["method_version"] != "mirror_v1":
                raise ValueError("metacognitive mirror method_version must be mirror_v1")
            SQLiteEventStore._number(event.payload["self_confidence"], "self_confidence", 0.0, 1.0)
            SQLiteEventStore._number(event.payload["meta_confidence_cap"], "meta_confidence_cap", 0.0, 1.0)
            if event.payload["meta_confidence_cap"] > event.payload["self_confidence"]:
                raise ValueError("metacognitive mirror confidence cap cannot exceed self confidence")
            if event.confidence > event.payload["meta_confidence_cap"]:
                raise ValueError("metacognitive mirror event confidence exceeds its cap")
            SQLiteEventStore._check_codes(event.payload["check_codes"])
            SQLiteEventStore._bounded_text(event.payload["public_summary"], "public_summary", MAX_TEXT)
        elif event.kind == EventKind.REWARD_OBSERVATION:
            required = {"reward_id", "target_event_id", "signal_kind", "normalized_value", "evaluator_ref", "scale_version", "observed_at"}
            if keys != required:
                raise ValueError("reward observation must use its fixed schema")
            for key in ("reward_id", "target_event_id", "evaluator_ref", "scale_version"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["signal_kind"] not in {"user_feedback", "task_outcome", "external_score"}:
                raise ValueError("reward signal_kind is invalid")
            SQLiteEventStore._number(event.payload["normalized_value"], "normalized_value", -1.0, 1.0)
            SQLiteEventStore._timestamp(event.payload["observed_at"], "observed_at")
        elif event.kind == EventKind.VALUE_ESTIMATE:
            required = {"estimate_id", "transition_id", "target_event_id", "state_key", "action_key", "next_state_key", "terminal", "value", "confidence", "estimator_id", "estimator_version", "alpha", "gamma", "clip", "max_entries", "max_events", "formula_version"}
            if keys != required:
                raise ValueError("value estimate must use its fixed schema")
            for key in ("estimate_id", "transition_id", "target_event_id", "state_key", "action_key", "next_state_key", "estimator_id", "estimator_version"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["action_key"] not in TD_SAFE_ACTION_CLASSES:
                raise ValueError("value estimate action_key is not a safe TD action class")
            if not isinstance(event.payload["terminal"], bool):
                raise ValueError("value estimate terminal must be a boolean")
            # Match td.ValueTable's bounded, inspectable value range.  A
            # value is only a research ranking scalar, never an authority.
            SQLiteEventStore._number(event.payload["value"], "value", -1000.0, 1000.0)
            SQLiteEventStore._number(event.payload["confidence"], "confidence", 0.0, 1.0)
            SQLiteEventStore._number(event.payload["alpha"], "alpha", 0.0, 1.0)
            SQLiteEventStore._number(event.payload["gamma"], "gamma", 0.0, 1.0)
            SQLiteEventStore._number(event.payload["clip"], "clip", 0.000000001, 1000.0)
            SQLiteEventStore._integer(event.payload["max_entries"], "max_entries", 1, 100000)
            SQLiteEventStore._integer(event.payload["max_events"], "max_events", 1, 1000000)
            if event.payload["formula_version"] != "td0_v1":
                raise ValueError("value estimate formula_version must be td0_v1")
        elif event.kind == EventKind.RPE_UPDATE:
            required = {"update_id", "transition_id", "reward_event_id", "prior_value_event_id", "state_key", "action_key", "reward", "prior_value", "next_value", "alpha", "gamma", "raw_delta", "clipped_delta", "clip", "updated_value", "formula_version", "scope"}
            if keys != required:
                raise ValueError("RPE update must use its fixed schema")
            for key in ("update_id", "transition_id", "reward_event_id", "prior_value_event_id", "state_key", "action_key"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["action_key"] not in TD_SAFE_ACTION_CLASSES:
                raise ValueError("RPE action_key is not a safe TD action class")
            for key, lower, upper in (("reward", -1.0, 1.0), ("prior_value", -1000.0, 1000.0), ("next_value", -1000.0, 1000.0), ("alpha", 0.0, 1.0), ("gamma", 0.0, 1.0), ("raw_delta", -2001.0, 2001.0), ("clipped_delta", -1000.0, 1000.0), ("clip", 0.000000001, 1000.0), ("updated_value", -1000.0, 1000.0)):
                SQLiteEventStore._number(event.payload[key], key, lower, upper)
            if event.payload["formula_version"] != "td0_v1" or event.payload["scope"] != "research_ranking_only":
                raise ValueError("RPE formula version or scope is invalid")
        elif event.kind == EventKind.FRONTIER_RANKING_DECISION:
            required_v1 = {"ranking_id", "transition_id", "frontier_task_id", "candidate_digest",
                           "learner_spec_digest", "channel_order", "scope", "ranking_version"}
            required_v2 = required_v1 | {"strategy_arm_id", "strategy_arm_version"}
            if keys not in (required_v1, required_v2):
                raise ValueError("frontier ranking decision must use its fixed schema")
            for key in ("ranking_id", "transition_id", "frontier_task_id"):
                SQLiteEventStore._id(event.payload[key], key)
            for key in ("candidate_digest", "learner_spec_digest"):
                SQLiteEventStore._sha256(event.payload[key])
            SQLiteEventStore._frontier_channel_order(event.payload["channel_order"])
            if (event.payload["scope"] != "frontier_ranking_only"
                    or event.payload["ranking_version"] not in {"frontier_ranking_v1", "frontier_ranking_v2"}
                    or (("strategy_arm_id" in keys) != (event.payload["ranking_version"] == "frontier_ranking_v2"))):
                raise ValueError("frontier ranking scope or version is invalid")
            if "strategy_arm_id" in keys:
                SQLiteEventStore._sha256(event.payload["strategy_arm_id"])
                if event.payload["strategy_arm_version"] != FRONTIER_STRATEGY_ARM_VERSION:
                    raise ValueError("frontier ranking strategy arm version is invalid")
        elif event.kind == EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE:
            required_v1 = {"estimate_id", "transition_id", "ranking_event_id", "learner_spec_digest",
                           "channel_order", "value_vector", "alpha", "gamma", "clip", "scope",
                           "formula_version"}
            required_v2 = required_v1 | {"strategy_arm_id", "strategy_arm_version"}
            if keys not in (required_v1, required_v2):
                raise ValueError("frontier vector value estimate must use its fixed schema")
            for key in ("estimate_id", "transition_id", "ranking_event_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["learner_spec_digest"])
            SQLiteEventStore._frontier_channel_order(event.payload["channel_order"])
            SQLiteEventStore._frontier_vector(event.payload["value_vector"], "value_vector", -1.0, 1.0)
            for key, low, high in (("alpha", 0.0, 1.0), ("gamma", 0.0, 1.0),
                                   ("clip", 0.000000001, 1.0)):
                SQLiteEventStore._number(event.payload[key], key, low, high)
            if (event.payload["scope"] != "frontier_ranking_only"
                    or event.payload["formula_version"] not in {"frontier_td0_vector_v1", "frontier_td0_vector_v2"}
                    or (("strategy_arm_id" in keys) != (event.payload["formula_version"] == "frontier_td0_vector_v2"))):
                raise ValueError("frontier value scope or formula version is invalid")
            if "strategy_arm_id" in keys:
                SQLiteEventStore._sha256(event.payload["strategy_arm_id"])
                if event.payload["strategy_arm_version"] != FRONTIER_STRATEGY_ARM_VERSION:
                    raise ValueError("frontier value strategy arm version is invalid")
        elif event.kind == EventKind.FRONTIER_EVIDENCE_OBSERVATION:
            required_v1 = {"evidence_id", "transition_id", "ranking_event_id", "source_event_id",
                           "evidence_digest", "evidence_kind", "learner_spec_digest", "scope",
                           "evidence_version"}
            required_v2 = required_v1 | {"strategy_arm_id", "strategy_arm_version"}
            if keys not in (required_v1, required_v2):
                raise ValueError("frontier evidence observation must use its fixed schema")
            for key in ("evidence_id", "transition_id", "ranking_event_id", "source_event_id"):
                SQLiteEventStore._id(event.payload[key], key)
            for key in ("evidence_digest", "learner_spec_digest"):
                SQLiteEventStore._sha256(event.payload[key])
            if event.payload["evidence_kind"] not in {"validated_external_evidence", "user_correction", "task_outcome"}:
                raise ValueError("frontier evidence_kind is invalid")
            if (event.payload["scope"] != "frontier_ranking_only"
                    or event.payload["evidence_version"] not in {"frontier_evidence_v1", "frontier_evidence_v2"}
                    or (("strategy_arm_id" in keys) != (event.payload["evidence_version"] == "frontier_evidence_v2"))):
                raise ValueError("frontier evidence scope or version is invalid")
            if "strategy_arm_id" in keys:
                SQLiteEventStore._sha256(event.payload["strategy_arm_id"])
                if event.payload["strategy_arm_version"] != FRONTIER_STRATEGY_ARM_VERSION:
                    raise ValueError("frontier evidence strategy arm version is invalid")
        elif event.kind == EventKind.FRONTIER_VECTOR_REWARD:
            required_v1 = {"reward_id", "transition_id", "ranking_event_id", "evidence_event_id",
                           "value_event_id", "learner_spec_digest", "channel_order", "reward_vector",
                           "scope", "reward_version"}
            required_v2 = required_v1 | {"strategy_arm_id", "strategy_arm_version"}
            if keys not in (required_v1, required_v2):
                raise ValueError("frontier vector reward must use its fixed schema")
            for key in ("reward_id", "transition_id", "ranking_event_id", "evidence_event_id", "value_event_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["learner_spec_digest"])
            SQLiteEventStore._frontier_channel_order(event.payload["channel_order"])
            SQLiteEventStore._frontier_vector(event.payload["reward_vector"], "reward_vector", -1.0, 1.0)
            if (event.payload["scope"] != "frontier_ranking_only"
                    or event.payload["reward_version"] not in {"frontier_reward_v1", "frontier_reward_v2"}
                    or (("strategy_arm_id" in keys) != (event.payload["reward_version"] == "frontier_reward_v2"))):
                raise ValueError("frontier reward scope or version is invalid")
            if "strategy_arm_id" in keys:
                SQLiteEventStore._sha256(event.payload["strategy_arm_id"])
                if event.payload["strategy_arm_version"] != FRONTIER_STRATEGY_ARM_VERSION:
                    raise ValueError("frontier reward strategy arm version is invalid")
        elif event.kind == EventKind.FRONTIER_TD_UPDATE:
            required_v1 = {"update_id", "transition_id", "ranking_event_id", "reward_event_id",
                           "prior_value_event_id", "learner_spec_digest", "channel_order",
                           "reward_vector", "prior_value_vector", "next_value_vector", "raw_delta_vector",
                           "clipped_delta_vector", "updated_value_vector", "alpha", "gamma", "clip",
                           "scope", "formula_version"}
            required_v2 = required_v1 | {"strategy_arm_id", "strategy_arm_version"}
            if keys not in (required_v1, required_v2):
                raise ValueError("frontier TD update must use its fixed schema")
            for key in ("update_id", "transition_id", "ranking_event_id", "reward_event_id", "prior_value_event_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["learner_spec_digest"])
            SQLiteEventStore._frontier_channel_order(event.payload["channel_order"])
            for key, low, high in (("reward_vector", -1.0, 1.0), ("prior_value_vector", -1.0, 1.0),
                                   ("next_value_vector", -1.0, 1.0), ("raw_delta_vector", -2.0, 2.0),
                                   ("clipped_delta_vector", -1.0, 1.0), ("updated_value_vector", -1.0, 1.0)):
                SQLiteEventStore._frontier_vector(event.payload[key], key, low, high)
            for key, low, high in (("alpha", 0.0, 1.0), ("gamma", 0.0, 1.0),
                                   ("clip", 0.000000001, 1.0)):
                SQLiteEventStore._number(event.payload[key], key, low, high)
            if (event.payload["scope"] != "frontier_ranking_only"
                    or event.payload["formula_version"] not in {"frontier_td0_vector_v1", "frontier_td0_vector_v2"}
                    or (("strategy_arm_id" in keys) != (event.payload["formula_version"] == "frontier_td0_vector_v2"))):
                raise ValueError("frontier TD scope or formula version is invalid")
            if "strategy_arm_id" in keys:
                SQLiteEventStore._sha256(event.payload["strategy_arm_id"])
                if event.payload["strategy_arm_version"] != FRONTIER_STRATEGY_ARM_VERSION:
                    raise ValueError("frontier TD strategy arm version is invalid")
        elif event.kind == EventKind.EXPERIMENT_PLAN_LOCKED:
            if event.source_ref != "ExperimentCoordinator":
                raise ValueError("experiment plan source_ref must be ExperimentCoordinator")
            required = {"experiment_id", "authorization_id", "learner_spec_digest", "experiment_registry_digest", "experiment_instance_digest",
                        "experiment_kind", "hypothesis_digest", "baseline_digest", "treatment_digest",
                        "input_digest", "evaluator_digest", "host_seed_digest", "budget_digest", "max_trials",
                        "max_steps", "max_wall_ms", "no_external_side_effects", "claim_scope", "template_version"}
            if keys != required:
                raise ValueError("experiment plan must use its fixed locked schema")
            for key in ("experiment_id", "authorization_id"):
                SQLiteEventStore._id(event.payload[key], key)
            for key in ("learner_spec_digest", "experiment_registry_digest", "experiment_instance_digest", "hypothesis_digest", "baseline_digest",
                        "treatment_digest", "input_digest", "evaluator_digest", "host_seed_digest", "budget_digest"):
                SQLiteEventStore._sha256(event.payload[key])
            if event.payload["experiment_kind"] not in EXPERIMENT_KINDS:
                raise ValueError("experiment kind is not approved")
            SQLiteEventStore._integer(event.payload["max_trials"], "max_trials", 1, 64)
            SQLiteEventStore._integer(event.payload["max_steps"], "max_steps", 1, 10000)
            SQLiteEventStore._integer(event.payload["max_wall_ms"], "max_wall_ms", 1, 2000)
            if event.payload["no_external_side_effects"] is not True:
                raise ValueError("experiment plan must prohibit external side effects")
            if event.payload["claim_scope"] != "offline_simulation" or event.payload["template_version"] != "experiment_template_v1":
                raise ValueError("experiment plan scope or template version is invalid")
        elif event.kind == EventKind.EXPERIMENT_EXECUTION_STARTED:
            if event.source_ref != "ExperimentHarness":
                raise ValueError("experiment execution source_ref must be ExperimentHarness")
            required = {"execution_id", "experiment_id", "plan_event_id", "experiment_registry_digest", "experiment_instance_digest",
                        "execution_nonce", "started_at", "max_trials", "max_steps", "max_wall_ms", "execution_version"}
            if keys != required:
                raise ValueError("experiment execution must use its fixed schema")
            for key in ("execution_id", "experiment_id", "plan_event_id", "execution_nonce"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._sha256(event.payload["experiment_registry_digest"])
            SQLiteEventStore._sha256(event.payload["experiment_instance_digest"])
            SQLiteEventStore._timestamp(event.payload["started_at"], "started_at")
            for key, maximum in (("max_trials", 64), ("max_steps", 10000), ("max_wall_ms", 2000)):
                SQLiteEventStore._integer(event.payload[key], key, 1, maximum)
            if event.payload["execution_version"] != "experiment_execution_v1":
                raise ValueError("experiment execution version is invalid")
        elif event.kind == EventKind.EXPERIMENT_RESULT:
            if event.source_ref != "IndependentExperimentVerifier":
                raise ValueError("experiment result source_ref must be IndependentExperimentVerifier")
            required = {"result_id", "execution_id", "experiment_id", "plan_event_id", "authorization_id",
                        "learner_spec_digest", "experiment_registry_digest", "experiment_instance_digest", "status", "complete_run_set",
                        "baseline_control_valid", "reproducible", "metric_digest", "metric_values", "effect_values",
                        "result_digest", "reproduction_digest", "invariant_codes", "claim_scope", "result_version"}
            if keys != required:
                raise ValueError("experiment result must use its fixed verifier schema")
            for key in ("result_id", "execution_id", "experiment_id", "plan_event_id", "authorization_id"):
                SQLiteEventStore._id(event.payload[key], key)
            for key in ("learner_spec_digest", "experiment_registry_digest", "experiment_instance_digest", "metric_digest", "result_digest", "reproduction_digest"):
                SQLiteEventStore._sha256(event.payload[key])
            if event.payload["status"] not in {"supported", "refuted", "inconclusive", "invalid"}:
                raise ValueError("experiment result status is invalid")
            for key in ("complete_run_set", "baseline_control_valid", "reproducible"):
                if not isinstance(event.payload[key], bool):
                    raise ValueError("experiment result %s must be boolean" % key)
            SQLiteEventStore._fixed_number_vector(event.payload["metric_values"], "metric_values", -1000000.0, 1000000.0)
            SQLiteEventStore._fixed_number_vector(event.payload["effect_values"], "effect_values", -1.0, 1.0)
            if (not isinstance(event.payload["invariant_codes"], list) or len(event.payload["invariant_codes"]) > 12
                    or len(set(event.payload["invariant_codes"])) != len(event.payload["invariant_codes"])
                    or any(code not in EXPERIMENT_INVARIANT_CODES for code in event.payload["invariant_codes"])):
                raise ValueError("experiment result invariant_codes are invalid")
            if event.payload["claim_scope"] != "offline_simulation" or event.payload["result_version"] != "experiment_result_v1":
                raise ValueError("experiment result scope or version is invalid")
        elif event.kind == EventKind.SLEEP_ARCHIVE:
            legacy_required = {"archive_id", "epoch_id", "schema_version", "chain_head_sequence",
                        "chain_head_hash", "approved_seed_ids", "approved_claim_ids",
                        "pending_task_ids", "quota_source", "quota_observed_at", "quota_reset_at",
                        "quota_remaining", "quota_total", "archive_digest", "event_count", "archive_version"}
            required = legacy_required | {"quota_window_kind"}
            if keys not in (legacy_required, required):
                raise ValueError("sleep archive must use its fixed metadata schema")
            for key in ("archive_id", "epoch_id"):
                SQLiteEventStore._id(event.payload[key], key)
            for key in ("chain_head_hash", "archive_digest"):
                SQLiteEventStore._sha256(event.payload[key])
            for key in ("approved_seed_ids", "approved_claim_ids", "pending_task_ids"):
                SQLiteEventStore._identifier_list(event.payload[key], key, sorted_required=True)
            SQLiteEventStore._quota_timestamps(event.payload, "quota_observed_at", "quota_reset_at")
            if event.payload["quota_source"] not in {"provider_usage", "authoritative_header"}:
                raise ValueError("sleep archive quota_source is not trustworthy")
            schema_version = event.payload["schema_version"]
            if keys == legacy_required:
                SQLiteEventStore._integer(schema_version, "schema_version", 2, 2)
                if event.payload["archive_version"] != "sleep_archive_v2":
                    raise ValueError("sleep archive version is invalid")
            else:
                SQLiteEventStore._integer(schema_version, "schema_version", 3, 3)
                if (event.payload["quota_window_kind"] != "rolling_5h"
                        or event.payload["archive_version"] != "sleep_archive_v3"):
                    raise ValueError("sleep archive must bind the rolling_5h quota window")
            SQLiteEventStore._integer(event.payload["chain_head_sequence"], "chain_head_sequence", 0, 1000000)
            SQLiteEventStore._integer(event.payload["event_count"], "event_count", 0, 1000000)
            SQLiteEventStore._integer(event.payload["quota_remaining"], "quota_remaining", 0, 1000000000)
            SQLiteEventStore._integer(event.payload["quota_total"], "quota_total", 0, 1000000000)
            if event.payload["quota_remaining"] > event.payload["quota_total"]:
                raise ValueError("sleep archive quota remaining cannot exceed total")
            if event.payload["archive_digest"] != SQLiteEventStore._archive_digest(event.payload):
                raise ValueError("sleep archive digest does not match its canonical public manifest")
        elif event.kind == EventKind.SLEEP_ENTERED:
            required = {"sleep_id", "archive_event_id", "epoch_id", "auto_wake_policy_event_id",
                        "continuation_policy_event_id", "slept_at", "protocol_version"}
            with_auto = required - {"continuation_policy_event_id"}
            legacy = with_auto - {"auto_wake_policy_event_id"}
            if keys not in (required, with_auto, legacy):
                raise ValueError("sleep entered must use its fixed metadata schema")
            for key in ("sleep_id", "archive_event_id", "epoch_id"):
                SQLiteEventStore._id(event.payload[key], key)
            if "auto_wake_policy_event_id" in keys and event.payload["auto_wake_policy_event_id"] is not None:
                SQLiteEventStore._id(event.payload["auto_wake_policy_event_id"], "auto_wake_policy_event_id")
            if "continuation_policy_event_id" in keys and event.payload["continuation_policy_event_id"] is not None:
                SQLiteEventStore._id(event.payload["continuation_policy_event_id"], "continuation_policy_event_id")
                if event.payload.get("auto_wake_policy_event_id") is None:
                    raise ValueError("continuation sleep requires an auto-wake policy")
            SQLiteEventStore._timestamp(event.payload["slept_at"], "slept_at")
            if event.payload["protocol_version"] != "sleep_wake_v1":
                raise ValueError("sleep protocol version is invalid")
        elif event.kind == EventKind.AUTO_WAKE_POLICY:
            required = {"policy_id", "action", "scope", "max_auto_runs", "max_ticks",
                        "expires_at", "issued_at", "policy_version"}
            if keys != required:
                raise ValueError("auto-wake policy must use its fixed schema")
            SQLiteEventStore._id(event.payload["policy_id"], "policy_id")
            if event.payload["action"] not in {"enable", "disable"} or event.payload["scope"] != "next_sleep_epoch":
                raise ValueError("auto-wake policy action or scope is invalid")
            SQLiteEventStore._integer(event.payload["max_auto_runs"], "max_auto_runs", 1, 1)
            SQLiteEventStore._integer(event.payload["max_ticks"], "max_ticks", 1, 4)
            SQLiteEventStore._timestamp(event.payload["issued_at"], "issued_at")
            SQLiteEventStore._timestamp(event.payload["expires_at"], "expires_at")
            if parse_aware_iso8601(event.payload["expires_at"]) <= parse_aware_iso8601(event.payload["issued_at"]):
                raise ValueError("auto-wake policy must expire after issuance")
            if event.payload["policy_version"] != "auto_wake_policy_v1":
                raise ValueError("auto-wake policy version is invalid")
        elif event.kind == EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY:
            required = {"continuation_id", "action", "scope", "approval_event_id",
                        "unattended_policy_id", "focus_digest", "profile_digest",
                        "max_auto_runs", "expires_at", "issued_at", "policy_version"}
            if keys != required:
                raise ValueError("unattended wake continuation policy must use its fixed schema")
            for key in ("continuation_id", "approval_event_id", "unattended_policy_id"):
                SQLiteEventStore._id(event.payload[key], key)
            if (event.payload["action"] != "enable"
                    or event.payload["scope"] != "next_sleep_epoch_read_only_research"):
                raise ValueError("continuation policy action or scope is invalid")
            SQLiteEventStore._sha256(event.payload["focus_digest"])
            SQLiteEventStore._sha256(event.payload["profile_digest"])
            SQLiteEventStore._integer(event.payload["max_auto_runs"], "max_auto_runs", 1, 1)
            SQLiteEventStore._timestamp(event.payload["issued_at"], "issued_at")
            SQLiteEventStore._timestamp(event.payload["expires_at"], "expires_at")
            if parse_aware_iso8601(event.payload["expires_at"]) <= parse_aware_iso8601(event.payload["issued_at"]):
                raise ValueError("continuation policy must expire after issuance")
            if event.payload["policy_version"] != "unattended_wake_continuation_policy_v1":
                raise ValueError("continuation policy version is invalid")
        elif event.kind == EventKind.UNATTENDED_WAKE_RUN:
            required = {"continuation_id", "awake_event_id", "profile_id", "authorization_policy_id", "runtime_policy_id",
                        "run_index", "status", "protocol_version"}
            if keys != required:
                raise ValueError("unattended wake run must use its fixed terminal schema")
            for key in ("continuation_id", "awake_event_id", "profile_id", "authorization_policy_id", "runtime_policy_id"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["runtime_policy_id"] == event.payload["authorization_policy_id"]:
                raise ValueError("unattended wake run requires a fresh runtime policy ID")
            SQLiteEventStore._integer(event.payload["run_index"], "run_index", 1, 1)
            if event.payload["status"] not in {"completed", "failed"}:
                raise ValueError("unattended wake run must have a terminal status")
            if event.payload["protocol_version"] != "unattended_wake_run_v1":
                raise ValueError("unattended wake run protocol version is invalid")
        elif event.kind == EventKind.EXPEDITION_AUTHORIZATION:
            required_v1 = {"authorization_id", "approval_event_id", "nonce", "issued_at", "expires_at",
                        "goal_digest", "host_seed_digest", "max_calls_per_slice", "slice_seconds",
                        "authorization_seconds", "profile", "version"}
            required_v2 = required_v1 | {"learning_mode", "learner_spec_digest"}
            required_v3 = required_v2 | {"experiment_kinds", "experiment_registry_digest", "max_experiments",
                                         "experiment_max_trials", "experiment_max_steps", "experiment_max_wall_ms"}
            if keys not in (required_v1, required_v2, required_v3):
                raise ValueError("expedition authorization must use its fixed schema")
            for key in ("authorization_id", "approval_event_id", "nonce"):
                SQLiteEventStore._id(event.payload[key], key)
            for key in ("goal_digest", "host_seed_digest"):
                SQLiteEventStore._sha256(event.payload[key])
            SQLiteEventStore._timestamp(event.payload["issued_at"], "issued_at")
            SQLiteEventStore._timestamp(event.payload["expires_at"], "expires_at")
            if parse_aware_iso8601(event.payload["expires_at"]) <= parse_aware_iso8601(event.payload["issued_at"]):
                raise ValueError("expedition authorization must expire after issuance")
            SQLiteEventStore._integer(event.payload["max_calls_per_slice"], "max_calls_per_slice", 1, 100)
            SQLiteEventStore._integer(event.payload["slice_seconds"], "slice_seconds", 1, 300)
            SQLiteEventStore._integer(event.payload["authorization_seconds"], "authorization_seconds", 1, 5 * 60 * 60)
            if (parse_aware_iso8601(event.payload["expires_at"])
                    - parse_aware_iso8601(event.payload["issued_at"]) != timedelta(
                        seconds=event.payload["authorization_seconds"])):
                raise ValueError("expedition authorization expiry must equal authorization_seconds")
            if parse_aware_iso8601(event.created_at) < parse_aware_iso8601(event.payload["issued_at"]):
                raise ValueError("expedition authorization event cannot predate issuance")
            if event.payload["profile"] != "public_web_only_v1":
                raise ValueError("expedition authorization profile or version is invalid")
            if keys == required_v1 and event.payload["version"] != "expedition_authorization_v1":
                raise ValueError("expedition authorization v1 is invalid")
            if keys == required_v2:
                if event.payload["version"] != "expedition_authorization_v2":
                    raise ValueError("expedition authorization v2 is invalid")
                if event.payload["learning_mode"] not in {"off", "shadow", "active"}:
                    raise ValueError("expedition learning_mode is invalid")
                SQLiteEventStore._sha256(event.payload["learner_spec_digest"])
            if keys == required_v3:
                if event.payload["version"] != "expedition_authorization_v3":
                    raise ValueError("expedition authorization v3 is invalid")
                if event.payload["learning_mode"] not in {"active", "shadow"}:
                    raise ValueError("expedition v3 learning_mode is invalid")
                SQLiteEventStore._sha256(event.payload["learner_spec_digest"])
                SQLiteEventStore._sha256(event.payload["experiment_registry_digest"])
                if (not isinstance(event.payload["experiment_kinds"], list) or not event.payload["experiment_kinds"]
                        or len(event.payload["experiment_kinds"]) > len(EXPERIMENT_KINDS)
                        or len(set(event.payload["experiment_kinds"])) != len(event.payload["experiment_kinds"])
                        or tuple(sorted(event.payload["experiment_kinds"])) != tuple(event.payload["experiment_kinds"])
                        or any(kind not in EXPERIMENT_KINDS for kind in event.payload["experiment_kinds"])):
                    raise ValueError("expedition v3 experiment_kinds are invalid")
                SQLiteEventStore._integer(event.payload["max_experiments"], "max_experiments", 1, 16)
                SQLiteEventStore._integer(event.payload["experiment_max_trials"], "experiment_max_trials", 1, 64)
                SQLiteEventStore._integer(event.payload["experiment_max_steps"], "experiment_max_steps", 1, 10000)
                SQLiteEventStore._integer(event.payload["experiment_max_wall_ms"], "experiment_max_wall_ms", 1, 2000)
        elif event.kind == EventKind.EXPEDITION_AUTHORIZATION_CONSUMED:
            required = {"consumption_id", "authorization_id", "consumed_at", "run_id", "version"}
            if keys != required:
                raise ValueError("expedition authorization consumption must use its fixed schema")
            for key in ("consumption_id", "authorization_id", "run_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._timestamp(event.payload["consumed_at"], "consumed_at")
            if event.payload["version"] != "expedition_authorization_consumed_v1":
                raise ValueError("expedition authorization consumption version is invalid")
        elif event.kind == EventKind.PROVIDER_USAGE_EVIDENCE:
            legacy_required = {"usage_evidence_id", "provider", "quota_source", "observed_at", "reset_at",
                        "total", "remaining", "evidence_digest", "evidence_version"}
            required = legacy_required | {"window_kind"}
            if keys not in (legacy_required, required):
                raise ValueError("provider usage evidence must use its fixed metadata schema")
            for key in ("usage_evidence_id", "provider"):
                SQLiteEventStore._id(event.payload[key], key)
            if event.payload["quota_source"] not in {"provider_usage", "authoritative_header"}:
                raise ValueError("provider usage evidence quota_source is not trustworthy")
            SQLiteEventStore._quota_timestamps(event.payload, "observed_at", "reset_at")
            SQLiteEventStore._integer(event.payload["total"], "total", 0, 1000000000)
            SQLiteEventStore._integer(event.payload["remaining"], "remaining", 0, 1000000000)
            if event.payload["remaining"] > event.payload["total"]:
                raise ValueError("provider usage remaining cannot exceed total")
            SQLiteEventStore._sha256(event.payload["evidence_digest"])
            if keys == legacy_required:
                if event.payload["evidence_version"] != "provider_usage_v1":
                    raise ValueError("provider usage evidence version is invalid")
            elif (event.payload["window_kind"] != "rolling_5h"
                  or event.payload["evidence_version"] != "provider_usage_v2"
                  or event.payload["evidence_digest"] != SQLiteEventStore._provider_usage_digest_v2(event.payload)):
                raise ValueError("provider usage evidence version is invalid")
        elif event.kind == EventKind.WAKE_CHECK:
            required = {"wake_check_id", "archive_event_id", "sleep_event_id", "epoch_id",
                        "usage_evidence_event_id", "checked_at", "check_version"}
            if keys != required:
                raise ValueError("wake check must use its fixed metadata schema")
            for key in ("wake_check_id", "archive_event_id", "sleep_event_id", "epoch_id",
                        "usage_evidence_event_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._timestamp(event.payload["checked_at"], "checked_at")
            if event.payload["check_version"] != "wake_check_v1":
                raise ValueError("wake check version is invalid")
        elif event.kind == EventKind.WAKE_READY:
            required = {"wake_ready_id", "wake_check_event_id", "epoch_id", "ready_at", "protocol_version"}
            if keys != required:
                raise ValueError("wake ready must use its fixed metadata schema")
            for key in ("wake_ready_id", "wake_check_event_id", "epoch_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._timestamp(event.payload["ready_at"], "ready_at")
            if event.payload["protocol_version"] not in {"sleep_wake_v1", "sleep_wake_v2"}:
                raise ValueError("wake ready protocol version is invalid")
        elif event.kind == EventKind.AWAKE:
            required = {"awake_id", "wake_ready_event_id", "epoch_id", "awakened_at", "protocol_version"}
            if keys != required:
                raise ValueError("awake must use its fixed metadata schema")
            for key in ("awake_id", "wake_ready_event_id", "epoch_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._timestamp(event.payload["awakened_at"], "awakened_at")
            if event.payload["protocol_version"] != "sleep_wake_v1":
                raise ValueError("awake protocol version is invalid")
        elif event.kind == EventKind.WAKE_TERMINAL:
            required = {"wake_terminal_id", "lifecycle_event_id", "epoch_id", "terminal_at", "outcome", "protocol_version"}
            if keys != required:
                raise ValueError("wake terminal must use its fixed metadata schema")
            for key in ("wake_terminal_id", "lifecycle_event_id", "epoch_id"):
                SQLiteEventStore._id(event.payload[key], key)
            SQLiteEventStore._timestamp(event.payload["terminal_at"], "terminal_at")
            if event.payload["outcome"] not in {"completed", "stopped", "purged", "failed"}:
                raise ValueError("wake terminal outcome is invalid")
            if event.payload["protocol_version"] != "sleep_wake_v1":
                raise ValueError("wake terminal protocol version is invalid")

    @staticmethod
    def _strings(value: Dict[str, Any], keys, limit: int) -> None:
        for key in keys:
            if not isinstance(value[key], str) or not value[key] or len(value[key]) > limit:
                raise ValueError("%s must be a bounded non-empty string" % key)

    @staticmethod
    def _id(value: Any, name: str) -> None:
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
            raise ValueError("%s must be a bounded safe identifier" % name)

    @staticmethod
    def _sha256(value: Any) -> None:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be a lowercase 64-character digest")

    @staticmethod
    def _timestamp(value: Any, name: str) -> None:
        if not isinstance(value, str) or len(value) > MAX_SHORT_TEXT:
            raise ValueError("%s must be a bounded ISO-8601 timestamp" % name)
        try:
            parse_aware_iso8601(value)
        except ValueError as error:
            raise ValueError("%s must be an aware ISO-8601 timestamp" % name) from error

    @staticmethod
    def _number(value: Any, name: str, lower: float, upper: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("%s must be a finite number" % name)
        if value < lower or value > upper:
            raise ValueError("%s is out of range" % name)

    @staticmethod
    def _integer(value: Any, name: str, lower: int, upper: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < lower or value > upper:
            raise ValueError("%s must be an integer in range" % name)

    @staticmethod
    def _bounded_text(value: Any, name: str, limit: int) -> None:
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError("%s must be bounded non-empty text" % name)

    @staticmethod
    def _enum_text(value: Any, name: str) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
            raise ValueError("%s must be a stable enum token" % name)

    @staticmethod
    def _event_ids(value: Any, name: str) -> None:
        if not isinstance(value, list) or len(value) > MAX_LIST:
            raise ValueError("%s must be a bounded event ID list" % name)
        if len(set(value)) != len(value):
            raise ValueError("%s must not repeat IDs" % name)
        for item in value:
            SQLiteEventStore._id(item, name)

    @staticmethod
    def _frontier_channel_order(value: Any) -> None:
        if not isinstance(value, list) or tuple(value) != FRONTIER_CHANNEL_ORDER:
            raise ValueError("frontier channel_order must use the fixed v1 sequence")

    @staticmethod
    def _frontier_vector(value: Any, name: str, lower: float, upper: float) -> None:
        if not isinstance(value, list) or len(value) != len(FRONTIER_CHANNEL_ORDER):
            raise ValueError("%s must contain one value per fixed frontier channel" % name)
        for index, item in enumerate(value):
            SQLiteEventStore._number(item, "%s[%s]" % (name, index), lower, upper)

    @staticmethod
    def _fixed_number_vector(value: Any, name: str, lower: float, upper: float) -> None:
        if not isinstance(value, list) or not value or len(value) > 12:
            raise ValueError("%s must be a bounded non-empty numeric vector" % name)
        for index, item in enumerate(value):
            SQLiteEventStore._number(item, "%s[%s]" % (name, index), lower, upper)

    @staticmethod
    def _identifier_list(value: Any, name: str, sorted_required: bool = False) -> None:
        if not isinstance(value, list) or len(value) > MAX_LIST or len(set(value)) != len(value):
            raise ValueError("%s must be a bounded unique identifier list" % name)
        if sorted_required and value != sorted(value):
            raise ValueError("%s must use canonical sorted ordering" % name)
        for item in value:
            SQLiteEventStore._id(item, name)

    @staticmethod
    def _quota_timestamps(value: Dict[str, Any], observed_key: str, reset_key: str) -> None:
        SQLiteEventStore._timestamp(value[observed_key], observed_key)
        SQLiteEventStore._timestamp(value[reset_key], reset_key)
        if parse_aware_iso8601(value[reset_key]) <= parse_aware_iso8601(value[observed_key]):
            raise ValueError("quota reset must be after its observation")

    def _validate_sleep_archive_head(self, event: CognitiveEvent, sequence: int) -> None:
        previous = self.connection.execute(
            "SELECT content_hash FROM cognitive_events WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
            (event.session_id,),
        ).fetchone()
        expected_head = "0" * 64 if previous is None else str(previous["content_hash"])
        if (event.payload["chain_head_hash"] != expected_head
                or event.payload["chain_head_sequence"] != sequence - 1
                or event.payload["event_count"] != sequence - 1):
            raise ValueError("sleep archive must bind the current session chain head and count")

    @staticmethod
    def _archive_digest_v2(payload: Dict[str, Any]) -> str:
        """Match sleep.build_public_archive's content-free canonical manifest."""
        manifest = {
            "schema_version": payload["schema_version"],
            "chain_head_sequence": payload["chain_head_sequence"],
            "chain_head_hash": payload["chain_head_hash"],
            "active_seed_ids": payload["approved_seed_ids"],
            "active_claim_ids": payload["approved_claim_ids"],
            "pending_public_event_ids": payload["pending_task_ids"],
            "quota": {"source": payload["quota_source"], "observed_at": payload["quota_observed_at"],
                      "reset_at": payload["quota_reset_at"], "remaining": payload["quota_remaining"],
                      "total": payload["quota_total"]},
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _archive_digest(payload: Dict[str, Any]) -> str:
        """Canonical public archive digest for both historical and v3 records."""
        manifest = {
            "schema_version": payload["schema_version"],
            "chain_head_sequence": payload["chain_head_sequence"],
            "chain_head_hash": payload["chain_head_hash"],
            "active_seed_ids": payload["approved_seed_ids"],
            "active_claim_ids": payload["approved_claim_ids"],
            "pending_public_event_ids": payload["pending_task_ids"],
            "quota": {"source": payload["quota_source"], "observed_at": payload["quota_observed_at"],
                      "reset_at": payload["quota_reset_at"], "remaining": payload["quota_remaining"],
                      "total": payload["quota_total"]},
        }
        if payload["schema_version"] >= 3:
            manifest["quota"]["window_kind"] = payload["quota_window_kind"]
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _provider_usage_digest_v2(payload: Dict[str, Any]) -> str:
        """Digest only public authoritative quota facts; never raw provider output."""
        manifest = {
            "provider": payload["provider"], "quota_source": payload["quota_source"],
            "observed_at": payload["observed_at"], "reset_at": payload["reset_at"],
            "total": payload["total"], "remaining": payload["remaining"],
            "window_kind": payload["window_kind"],
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _check_codes(value: Any) -> None:
        allowed = frozenset((
            "provenance_complete", "scope_bounded", "confidence_capped",
            "counterevidence_clear", "evidence_missing", "source_conflict",
        ))
        if not isinstance(value, list) or not value or len(value) > MAX_LIST:
            raise ValueError("check_codes must be a non-empty bounded list")
        if len(set(value)) != len(value) or any(code not in allowed for code in value):
            raise ValueError("check_codes contains an unsupported or repeated code")

    def _validate_lineage(self, event: CognitiveEvent,
                          parents: Dict[str, CognitiveEvent]) -> None:
        """Validate cross-event bindings for media, autonomy, TD, and lifecycle records."""
        ordered = [parents[parent_id] for parent_id in event.parent_event_ids]
        payload = event.payload
        prior_events = self.list(event.session_id)

        def prior_matching(kind: EventKind, field: str, value: Any) -> List[CognitiveEvent]:
            return [candidate for candidate in prior_events
                    if candidate.kind == kind and candidate.payload.get(field) == value]

        def stopped_run(run_id: str) -> bool:
            return any(candidate.kind == EventKind.AUTONOMY_STOPPED
                       and candidate.payload.get("run_id") == run_id
                       for candidate in prior_events)

        def no_sleep_wake_evidence(ids: List[str], label: str) -> None:
            records = {candidate.event_id: candidate for candidate in prior_events}
            if any(candidate is None for candidate in (records.get(item) for item in ids)):
                raise ValueError("%s must name prior same-session records" % label)
            if any(records[item].kind in NON_EVIDENCE_LIFECYCLE_EVENT_KINDS | EXPERIMENT_EVENT_KINDS for item in ids):
                raise ValueError("lifecycle or experiment records cannot support seeds or self-model claims")

        def epoch_terminated(epoch_id: str) -> bool:
            return any(candidate.kind == EventKind.WAKE_TERMINAL
                       and candidate.payload.get("epoch_id") == epoch_id
                       for candidate in prior_events)

        epoch_event_kinds = frozenset((EventKind.SLEEP_ENTERED, EventKind.WAKE_CHECK,
                                       EventKind.WAKE_READY, EventKind.AWAKE))
        if event.kind in epoch_event_kinds:
            if epoch_terminated(payload["epoch_id"]):
                raise ValueError("sleep/wake lifecycle epoch has already terminated")
        elif event.kind == EventKind.WAKE_TERMINAL and epoch_terminated(payload["epoch_id"]):
            raise ValueError("sleep/wake lifecycle epoch already has a terminal record")

        if event.kind == EventKind.SEED_PROPOSED:
            no_sleep_wake_evidence(event.payload["seed"]["provenance_event_ids"], "seed provenance")
            if any(parent.kind in NON_EVIDENCE_LIFECYCLE_EVENT_KINDS | EXPERIMENT_EVENT_KINDS for parent in ordered):
                raise ValueError("lifecycle or experiment records cannot support seed approval")
        if event.kind in (EventKind.SELF_CLAIM_PROPOSED, EventKind.SELF_CLAIM_APPROVED):
            claim = payload["claim"]
            mirror_ids = {candidate.event_id for candidate in prior_events
                          if candidate.kind == EventKind.METACOGNITIVE_MIRROR}
            if mirror_ids.intersection(claim["evidence_event_ids"]):
                raise ValueError("metacognitive mirrors cannot support self-model approval")
            no_sleep_wake_evidence(claim["evidence_event_ids"], "self-model evidence")
            if any(parent.kind in NON_EVIDENCE_LIFECYCLE_EVENT_KINDS | EXPERIMENT_EVENT_KINDS for parent in ordered):
                raise ValueError("lifecycle or experiment records cannot support self-model approval")

        if event.kind in (EventKind.SEED_APPROVED, EventKind.SEED_RETIRED,
                          EventKind.SELF_CLAIM_REVOKED):
            if any(parent.kind in NON_EVIDENCE_LIFECYCLE_EVENT_KINDS | EXPERIMENT_EVENT_KINDS for parent in ordered):
                raise ValueError("lifecycle or experiment records cannot approve or revoke durable records")

        if event.kind == EventKind.MODEL_INVOCATION:
            trigger_kinds = {EventKind.OBSERVATION, EventKind.MEDIA_OBSERVATION,
                             EventKind.PERCEPT, EventKind.LOOP_TICK,
                             EventKind.TOOL_CALL_PROPOSED, EventKind.TOOL_EXECUTION_STARTED,
                             EventKind.AWAKE}
            if len(ordered) != 1 or ordered[0].kind not in trigger_kinds:
                raise ValueError("model invocation requires exactly one observable trigger parent")
            trigger = ordered[0]
            if payload["trigger_event_id"] != trigger.event_id:
                raise ValueError("model invocation must bind its triggering event")
            trigger_run_id = trigger.payload.get("run_id")
            # Text/image turns can be assigned a host turn ID even though
            # their observation has no autonomous run.  Once a trigger itself
            # belongs to a run, however, the invocation cannot switch runs.
            if trigger_run_id is not None and payload["run_id"] != trigger_run_id:
                raise ValueError("model invocation run_id must match its triggering event")
            if payload["run_id"] is not None and stopped_run(payload["run_id"]):
                raise ValueError("model invocation cannot occur after its run stopped")
            if trigger.kind == EventKind.AWAKE:
                if (payload["role"] != "tool_planning"
                        or payload["context_scope"] != "tool_planning"):
                    raise ValueError("awake may trigger only continuation tool planning")
                records_by_id = {candidate.event_id: candidate for candidate in prior_events}
                ready = records_by_id.get(trigger.payload["wake_ready_event_id"])
                check = None if ready is None else records_by_id.get(ready.payload.get("wake_check_event_id"))
                sleep = None if check is None else records_by_id.get(check.payload.get("sleep_event_id"))
                continuation = None if sleep is None else records_by_id.get(
                    sleep.payload.get("continuation_policy_event_id"))
                if (ready is None or ready.kind != EventKind.WAKE_READY
                        or check is None or check.kind != EventKind.WAKE_CHECK
                        or sleep is None or sleep.kind != EventKind.SLEEP_ENTERED
                        or continuation is None
                        or continuation.kind != EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY
                        or epoch_terminated(trigger.payload["epoch_id"])
                        or parse_aware_iso8601(event.created_at)
                        >= parse_aware_iso8601(continuation.payload["expires_at"])):
                    raise ValueError("awake model planning requires an unexpired continuation epoch")
        elif event.kind == EventKind.CAPABILITY_GRANTED:
            if len(ordered) != 1 or ordered[0].kind not in (
                    EventKind.OBSERVATION,
                    EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY,
                    EventKind.EXPEDITION_AUTHORIZATION):
                raise ValueError("capability grant requires one user observation, continuation policy, or expedition authorization parent")
            parent = ordered[0]
            if parent.kind == EventKind.OBSERVATION:
                if parent.source_kind != SourceKind.USER:
                    raise ValueError("capability grant observation parent must be user-sourced")
            elif parent.kind == EventKind.EXPEDITION_AUTHORIZATION:
                consumed = [candidate for candidate in prior_events
                            if candidate.kind == EventKind.EXPEDITION_AUTHORIZATION_CONSUMED
                            and candidate.payload.get("authorization_id") == parent.payload["authorization_id"]]
                if (len(consumed) != 1 or parent.payload.get("profile") != "public_web_only_v1"
                        or event.payload["allow_mutating"]
                        or parse_aware_iso8601(event.created_at) >= parse_aware_iso8601(parent.payload["expires_at"])):
                    raise ValueError("expedition capability grant requires a consumed, unexpired read-only public-web authorization")
            else:
                # The policy is itself a user-authored, one-shot authorization.
                # It may materialize a new bounded read-only grant after its
                # exact sleeping epoch wakes; no SYSTEM record impersonates a
                # user observation.
                sleeps = [candidate for candidate in prior_events
                          if candidate.kind == EventKind.SLEEP_ENTERED
                          and candidate.payload.get("continuation_policy_event_id") == parent.event_id]
                if len(sleeps) != 1:
                    raise ValueError("continuation capability grant requires its bound sleeping epoch")
                sleep = sleeps[0]
                awake_events = [candidate for candidate in prior_events
                                if candidate.kind == EventKind.AWAKE
                                and candidate.payload.get("epoch_id") == sleep.payload["epoch_id"]]
                if len(awake_events) != 1 or epoch_terminated(sleep.payload["epoch_id"]):
                    raise ValueError("continuation capability grant requires one non-terminal awakened epoch")
                if event.payload["allow_mutating"]:
                    raise ValueError("continuation capability grant must be read-only")
        elif event.kind == EventKind.CAPABILITY_REVOKED:
            if len(ordered) != 1 or ordered[0].kind != EventKind.CAPABILITY_GRANTED:
                raise ValueError("capability revocation requires exactly one grant parent")
            grant = ordered[0]
            if payload["grant_id"] != grant.payload["grant_id"]:
                raise ValueError("capability revocation grant_id must match its parent")
            if parse_aware_iso8601(payload["revoked_at"]) < parse_aware_iso8601(grant.created_at):
                raise ValueError("capability revocation cannot predate its grant")
        elif event.kind == EventKind.TOOL_CALL_PROPOSED:
            if len(ordered) != 2:
                raise ValueError("tool proposal requires model invocation and capability grant parents")
            invocation = next((candidate for candidate in ordered
                               if candidate.kind == EventKind.MODEL_INVOCATION), None)
            grant = next((candidate for candidate in ordered
                          if candidate.kind == EventKind.CAPABILITY_GRANTED), None)
            if invocation is None or grant is None:
                raise ValueError("tool proposal parents must be model invocation and capability grant")
            if (payload["run_id"] != invocation.payload["run_id"]
                    or payload["grant_id"] != grant.payload["grant_id"]
                    or payload["tool_name"] != grant.payload["tool_name"]):
                raise ValueError("tool proposal must bind invocation run and matching grant")
            if (payload["requested_input_bytes"] > grant.payload["max_input_bytes"]
                    or payload["requested_output_bytes"] > grant.payload["max_output_bytes"]
                    or payload["requested_wall_ms"] > grant.payload["max_wall_ms"]
                    or (payload["mutating"] and not grant.payload["allow_mutating"])):
                raise ValueError("tool proposal exceeds its capability grant")
            now = parse_aware_iso8601(event.created_at)
            if not (parse_aware_iso8601(grant.payload["not_before"]) <= now
                    < parse_aware_iso8601(grant.payload["expires_at"])):
                raise ValueError("tool proposal is outside capability grant TTL")
            if prior_matching(EventKind.CAPABILITY_REVOKED, "grant_id", payload["grant_id"]):
                raise ValueError("tool proposal cannot use a revoked capability grant")
            if stopped_run(payload["run_id"]):
                raise ValueError("tool proposal cannot occur after its run stopped")
        elif event.kind == EventKind.TOOL_EXECUTION_CONFIRMED:
            if len(ordered) != 1 or ordered[0].kind != EventKind.TOOL_CALL_PROPOSED:
                raise ValueError("tool confirmation requires exactly one proposal parent")
            proposal = ordered[0]
            if not proposal.payload["mutating"]:
                raise ValueError("tool confirmation is only valid for mutating proposals")
            for key in ("call_id", "run_id", "grant_id", "plan_digest"):
                if payload[key] != proposal.payload[key]:
                    raise ValueError("tool confirmation must match its proposal")
        elif event.kind == EventKind.TOOL_EXECUTION_STARTED:
            if len(ordered) not in (2, 3):
                raise ValueError("tool execution start requires proposal, grant, and optional confirmation")
            proposal = next((candidate for candidate in ordered
                             if candidate.kind == EventKind.TOOL_CALL_PROPOSED), None)
            grant = next((candidate for candidate in ordered
                          if candidate.kind == EventKind.CAPABILITY_GRANTED), None)
            confirmation = next((candidate for candidate in ordered
                                 if candidate.kind == EventKind.TOOL_EXECUTION_CONFIRMED), None)
            if proposal is None or grant is None or (len(ordered) == 3 and confirmation is None):
                raise ValueError("tool execution start has invalid parent kinds")
            for key in ("call_id", "run_id", "grant_id", "plan_digest"):
                if payload[key] != proposal.payload[key]:
                    raise ValueError("tool execution start must match its proposal")
            if payload["grant_id"] != grant.payload["grant_id"]:
                raise ValueError("tool execution start grant must match its parent")
            if proposal.payload["mutating"]:
                if confirmation is None or payload["confirmation_event_id"] != confirmation.event_id:
                    raise ValueError("mutating tool execution requires matching user confirmation")
            elif confirmation is not None or payload["confirmation_event_id"] is not None:
                raise ValueError("non-mutating tool execution cannot carry confirmation")
            started_at = parse_aware_iso8601(payload["started_at"])
            deadline_at = parse_aware_iso8601(payload["deadline_at"])
            if not (parse_aware_iso8601(grant.payload["not_before"]) <= started_at
                    < parse_aware_iso8601(grant.payload["expires_at"])):
                raise ValueError("tool execution is outside capability grant TTL")
            if (deadline_at - started_at).total_seconds() * 1000 > proposal.payload["requested_wall_ms"]:
                raise ValueError("tool execution deadline exceeds proposed wall budget")
            if stopped_run(payload["run_id"]):
                raise ValueError("tool execution cannot start after its run stopped")
            uses = prior_matching(EventKind.TOOL_EXECUTION_STARTED, "grant_id", payload["grant_id"])
            if len(uses) >= grant.payload["max_uses"]:
                raise ValueError("capability grant use budget exhausted")
            if prior_matching(EventKind.CAPABILITY_REVOKED, "grant_id", payload["grant_id"]):
                raise ValueError("tool execution cannot use a revoked capability grant")
        elif event.kind == EventKind.TOOL_RESULT and "execution_id" in payload:
            if event.source_kind != SourceKind.TOOL or len(ordered) != 1 or ordered[0].kind != EventKind.TOOL_EXECUTION_STARTED:
                raise ValueError("execution-bound tool result requires one system start parent and tool source")
            execution = ordered[0]
            for key in ("execution_id", "call_id", "run_id", "grant_id", "plan_digest"):
                if payload[key] != execution.payload[key]:
                    raise ValueError("tool result must match its execution")
            # call_id is not an event ID, so find the proposal in the current session.
            proposals = prior_matching(EventKind.TOOL_CALL_PROPOSED, "call_id", payload["call_id"])
            if len(proposals) != 1:
                raise ValueError("tool result execution proposal is unavailable")
            proposal = proposals[0]
            grant_events = prior_matching(EventKind.CAPABILITY_GRANTED, "grant_id", payload["grant_id"])
            if len(grant_events) != 1:
                raise ValueError("tool result capability grant is unavailable")
            grant = grant_events[0]
            if (payload["tool_name"] != proposal.payload["tool_name"]
                    or payload["result_bytes"] > grant.payload["max_output_bytes"]
                    or payload["latency_ms"] > proposal.payload["requested_wall_ms"]):
                raise ValueError("tool result exceeds or mismatches its authorization")
            if prior_matching(EventKind.TOOL_EXECUTION_ABANDONED, "execution_id", payload["execution_id"]):
                raise ValueError("abandoned tool execution cannot record a result")
            if stopped_run(payload["run_id"]):
                raise ValueError("tool result after run stop must be recorded as abandonment")
        elif event.kind == EventKind.TOOL_EXECUTION_ABANDONED:
            if len(ordered) != 1 or ordered[0].kind != EventKind.TOOL_EXECUTION_STARTED:
                raise ValueError("tool abandonment requires exactly one execution start parent")
            execution = ordered[0]
            for key in ("execution_id", "call_id", "run_id", "grant_id", "plan_digest"):
                if payload[key] != execution.payload[key]:
                    raise ValueError("tool abandonment must match its execution")
            if prior_matching(EventKind.TOOL_RESULT, "execution_id", payload["execution_id"]):
                raise ValueError("completed tool execution cannot be abandoned")
        elif event.kind == EventKind.PERCEPT:
            if len(ordered) != 1 or ordered[0].kind != EventKind.MEDIA_OBSERVATION:
                raise ValueError("percept must have exactly one media observation parent")
            media = ordered[0].payload
            if (payload["artifact_id"] != media["artifact_id"]
                    or payload["artifact_sha256"] != media["sha256"]
                    or payload["modality"] != media["modality"]):
                raise ValueError("percept artifact binding does not match media parent")
        elif event.kind == EventKind.AUTONOMY_CONTROL:
            if event.source_kind == SourceKind.USER:
                # User commands are explicit roots so a halt remains available
                # even when a prior autonomous run is malformed or unavailable.
                if ordered:
                    raise ValueError("user autonomy controls must be root events")
            elif len(ordered) != 1:
                raise ValueError("system autonomy control requires exactly one parent")
            elif ordered[0].kind == EventKind.AWAKE:
                awake = ordered[0]
                if event.source_ref != "SleepWakeCoordinator":
                    raise ValueError("system wake-start source_ref must be SleepWakeCoordinator")
                if (payload["action"] != "start" or payload["next_tick_index"] != 0
                        or epoch_terminated(awake.payload["epoch_id"])):
                    raise ValueError("system autonomy start requires a non-terminal awake epoch")
                existing_runs = [candidate for candidate in prior_events
                                 if candidate.kind == EventKind.AUTONOMY_CONTROL
                                 and candidate.payload.get("run_id") == payload["run_id"]]
                if existing_runs:
                    raise ValueError("system wake-start run_id must be new in its session")
                if any(candidate.kind == EventKind.AUTONOMY_CONTROL
                       and candidate.source_kind == SourceKind.SYSTEM
                       and candidate.parent_event_ids == (awake.event_id,)
                       for candidate in prior_events):
                    raise ValueError("awake event already has a system autonomy start")
                records_by_id = {candidate.event_id: candidate for candidate in prior_events}
                ready = records_by_id.get(awake.payload["wake_ready_event_id"])
                check = None if ready is None else records_by_id.get(ready.payload.get("wake_check_event_id"))
                sleep = None if check is None else records_by_id.get(check.payload.get("sleep_event_id"))
                policy = None if sleep is None else records_by_id.get(sleep.payload.get("auto_wake_policy_event_id"))
                if (ready is None or ready.kind != EventKind.WAKE_READY
                        or check is None or check.kind != EventKind.WAKE_CHECK
                        or sleep is None or sleep.kind != EventKind.SLEEP_ENTERED
                        or policy is None or policy.kind != EventKind.AUTO_WAKE_POLICY
                        or policy.payload["action"] != "enable"
                        or payload["max_ticks"] > policy.payload["max_ticks"]):
                    raise ValueError("system autonomy start requires its bound user auto-wake policy")
            elif ordered[0].kind in (EventKind.AUTONOMY_CONTROL, EventKind.LOOP_TICK):
                if ordered[0].payload.get("run_id") != payload["run_id"]:
                    raise ValueError("autonomy control run_id must match its lineage")
            else:
                raise ValueError("system autonomy control requires a control, tick, or awake parent")
        elif event.kind == EventKind.LOOP_TICK:
            if len(ordered) != 1 or ordered[0].kind not in (EventKind.AUTONOMY_CONTROL, EventKind.LOOP_TICK):
                raise ValueError("loop tick requires exactly one control or tick parent")
            parent = ordered[0]
            if parent.payload.get("run_id") != payload["run_id"]:
                raise ValueError("loop tick run_id must match its parent")
            if parent.kind == EventKind.AUTONOMY_CONTROL:
                action = parent.payload["action"]
                if action == "start":
                    if parent.payload["next_tick_index"] != 0 or payload["tick_index"] != 0:
                        raise ValueError("start control and first tick must both use index zero")
                elif action == "resume":
                    if payload["tick_index"] != parent.payload["next_tick_index"]:
                        raise ValueError("resumed loop tick must use the control's next_tick_index")
                else:
                    raise ValueError("loop tick cannot follow a pause or stop control")
            elif payload["tick_index"] != parent.payload["tick_index"] + 1:
                raise ValueError("loop tick indexes must be consecutive")
        elif event.kind == EventKind.AUTONOMY_STOPPED:
            if len(ordered) != 1 or ordered[0].kind not in (EventKind.AUTONOMY_CONTROL, EventKind.LOOP_TICK):
                raise ValueError("autonomy stopped requires exactly one control or tick parent")
            if ordered[0].payload.get("run_id") != payload["run_id"]:
                raise ValueError("autonomy stop run_id must match its parent")
            parent = ordered[0]
            if parent.kind == EventKind.LOOP_TICK:
                if payload["tick_count"] != parent.payload["tick_index"] + 1:
                    raise ValueError("autonomy stop tick_count must match its final tick")
            elif parent.payload["action"] != "stop":
                raise ValueError("autonomy stopped may only follow a stop control or final tick")
            elif payload["tick_count"] != parent.payload["next_tick_index"]:
                raise ValueError("autonomy stop tick_count must match stop control next_tick_index")
        elif event.kind == EventKind.METACOGNITIVE_MIRROR:
            if event.source_ref != "MirrorAuditor":
                raise ValueError("metacognitive mirror source_ref must be MirrorAuditor")
            if len(ordered) != 2:
                raise ValueError("metacognitive mirror requires target and judgment parents")
            target, judgment = ordered
            if target.event_id != payload["target_event_id"] or judgment.event_id != payload["judgment_event_id"]:
                raise ValueError("metacognitive mirror parents must bind target and judgment in order")
            if target.event_id == judgment.event_id:
                raise ValueError("metacognitive mirror target and judgment must differ")
            allowed_evidence = frozenset((
                EventKind.OBSERVATION, EventKind.MEDIA_OBSERVATION, EventKind.PERCEPT,
                EventKind.TOOL_RESULT, EventKind.ACTION_RESULT, EventKind.CORRECTION,
                EventKind.MODEL_INVOCATION,
            ))
            if target.kind not in allowed_evidence - frozenset((EventKind.MODEL_INVOCATION,)):
                raise ValueError("metacognitive mirror target is not public evidence")
            if judgment.kind not in {EventKind.DECISION, EventKind.ACTION_RESULT, EventKind.LOOP_TICK}:
                raise ValueError("metacognitive mirror judgment has an invalid kind")
            evidence_ids = payload["evidence_event_ids"]
            if not {target.event_id, judgment.event_id}.issubset(set(evidence_ids)):
                raise ValueError("metacognitive mirror evidence must include target and judgment")
            evidence = {candidate.event_id: candidate for candidate in prior_events
                        if candidate.event_id in evidence_ids}
            if len(evidence) != len(evidence_ids):
                raise ValueError("metacognitive mirror evidence must be prior same-session records")
            for candidate in evidence.values():
                if candidate.kind not in allowed_evidence and candidate.event_id != judgment.event_id:
                    raise ValueError("metacognitive mirror evidence kind is not allowed")
                if candidate.kind == EventKind.MODEL_INVOCATION and candidate.payload["outcome"] != "completed":
                    raise ValueError("metacognitive mirror model evidence must be completed")
        elif event.kind == EventKind.SLEEP_ARCHIVE:
            raise ValueError("sleep archive must be a root event")
        elif event.kind == EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY:
            if len(ordered) != 1 or ordered[0].kind != EventKind.OBSERVATION:
                raise ValueError("continuation policy requires exactly one user observation approval parent")
            approval = ordered[0]
            if approval.source_kind != SourceKind.USER or payload["approval_event_id"] != approval.event_id:
                raise ValueError("continuation policy must bind a same-session user approval observation")
            if parse_aware_iso8601(payload["issued_at"]) < parse_aware_iso8601(approval.created_at):
                raise ValueError("continuation policy cannot predate its user approval")
            if any(candidate.kind == EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY
                   and candidate.payload.get("approval_event_id") == approval.event_id
                   for candidate in prior_events):
                raise ValueError("user approval already has a continuation policy")
        elif event.kind == EventKind.EXPEDITION_AUTHORIZATION:
            if len(ordered) != 1 or ordered[0].kind != EventKind.OBSERVATION:
                raise ValueError("expedition authorization requires exactly one user observation approval parent")
            approval = ordered[0]
            if approval.source_kind != SourceKind.USER or payload["approval_event_id"] != approval.event_id:
                raise ValueError("expedition authorization must bind its same-session user approval")
            if parse_aware_iso8601(payload["issued_at"]) < parse_aware_iso8601(approval.created_at):
                raise ValueError("expedition authorization cannot predate its user approval")
            if any(candidate.kind == EventKind.EXPEDITION_AUTHORIZATION
                   and candidate.payload.get("approval_event_id") == approval.event_id
                   for candidate in prior_events):
                raise ValueError("user approval already has an expedition authorization")
        elif event.kind == EventKind.EXPEDITION_AUTHORIZATION_CONSUMED:
            if len(ordered) != 1 or ordered[0].kind != EventKind.EXPEDITION_AUTHORIZATION:
                raise ValueError("expedition consumption requires exactly one authorization parent")
            authorization = ordered[0]
            if (payload["authorization_id"] != authorization.payload["authorization_id"]
                    or parse_aware_iso8601(payload["consumed_at"]) < parse_aware_iso8601(authorization.payload["issued_at"])
                    or parse_aware_iso8601(payload["consumed_at"]) >= parse_aware_iso8601(authorization.payload["expires_at"])):
                raise ValueError("expedition consumption must occur in its exact authorization window")
        elif event.kind == EventKind.SLEEP_ENTERED:
            if event.source_ref != "SleepWakeCoordinator" or len(ordered) not in (1, 2, 3):
                raise ValueError("sleep entered requires archive and optional user policy parents")
            archive = ordered[0]
            if archive.kind != EventKind.SLEEP_ARCHIVE:
                raise ValueError("sleep entered must follow a sleep archive")
            if (payload["archive_event_id"] != archive.event_id
                    or payload["epoch_id"] != archive.payload["epoch_id"]
                    or parse_aware_iso8601(payload["slept_at"]) < parse_aware_iso8601(archive.created_at)):
                raise ValueError("sleep entered must bind its archive and epoch")
            policy_id = payload.get("auto_wake_policy_event_id")
            continuation_id = payload.get("continuation_policy_event_id")
            if policy_id is None:
                if len(ordered) != 1 or continuation_id is not None:
                    raise ValueError("sleep without auto-wake policy may only parent its archive")
            else:
                expected_parents = 3 if continuation_id is not None else 2
                if len(ordered) != expected_parents or ordered[1].kind != EventKind.AUTO_WAKE_POLICY:
                    raise ValueError("auto-wake sleep requires its user policy as second parent")
                policy = ordered[1]
                policies = [candidate for candidate in prior_events if candidate.kind == EventKind.AUTO_WAKE_POLICY]
                if (policy.event_id != policy_id or not policies or policies[-1].event_id != policy.event_id
                        or policy.payload["action"] != "enable"
                        or parse_aware_iso8601(policy.payload["expires_at"]) <= parse_aware_iso8601(payload["slept_at"])
                        or any(candidate.kind == EventKind.SLEEP_ENTERED
                               and candidate.payload.get("auto_wake_policy_event_id") == policy_id
                               for candidate in prior_events)):
                    raise ValueError("sleep auto-wake policy must be latest, unexpired, enabled, and unused")
                if continuation_id is not None:
                    continuation = ordered[2]
                    continuations = [candidate for candidate in prior_events
                                     if candidate.kind == EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY]
                    if (continuation.kind != EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY
                            or continuation.event_id != continuation_id
                            or not continuations or continuations[-1].event_id != continuation.event_id
                            or parse_aware_iso8601(continuation.payload["expires_at"])
                            <= parse_aware_iso8601(payload["slept_at"])
                            or any(candidate.kind == EventKind.SLEEP_ENTERED
                                   and candidate.payload.get("continuation_policy_event_id") == continuation_id
                                   for candidate in prior_events)):
                        raise ValueError("continuation policy must be latest, unexpired, and unused for this epoch")
        elif event.kind == EventKind.PROVIDER_USAGE_EVIDENCE:
            if event.source_ref != "ProviderUsageVerifier" or len(ordered) != 1:
                raise ValueError("provider usage evidence requires its verifier and one sleep parent")
            sleep = ordered[0]
            if sleep.kind != EventKind.SLEEP_ENTERED:
                raise ValueError("provider usage evidence must follow a sleeping record")
            archive = next((candidate for candidate in prior_events
                            if candidate.event_id == sleep.payload["archive_event_id"]), None)
            if archive is None or epoch_terminated(archive.payload["epoch_id"]):
                raise ValueError("provider usage evidence cannot follow a terminated epoch")
            if payload["evidence_version"] == "provider_usage_v2":
                if (archive.payload.get("archive_version") != "sleep_archive_v3"
                        or archive.payload.get("quota_window_kind") != "rolling_5h"):
                    raise ValueError("rolling provider usage evidence requires a rolling sleep archive")
            elif archive.payload.get("archive_version") == "sleep_archive_v3":
                # v1 is retained solely to verify old epochs.  A v3 archive
                # cannot downgrade its later provider fact to an unbound window.
                raise ValueError("rolling sleep archive requires provider usage evidence v2")
        elif event.kind == EventKind.WAKE_CHECK:
            if event.source_ref != "SleepWakeCoordinator" or len(ordered) != 2:
                raise ValueError("wake check requires the coordinator, sleep, and provider evidence")
            sleep, evidence = ordered
            if sleep.kind != EventKind.SLEEP_ENTERED or evidence.kind != EventKind.PROVIDER_USAGE_EVIDENCE:
                raise ValueError("wake check parents must be sleep followed by provider usage evidence")
            archive = next((candidate for candidate in prior_events
                            if candidate.event_id == sleep.payload["archive_event_id"]), None)
            if archive is None or archive.kind != EventKind.SLEEP_ARCHIVE:
                raise ValueError("wake check sleep archive is unavailable")
            for field, value in (("archive_event_id", archive.event_id),
                                 ("sleep_event_id", sleep.event_id),
                                 ("epoch_id", archive.payload["epoch_id"]),
                                 ("usage_evidence_event_id", evidence.event_id)):
                if payload[field] != value:
                    raise ValueError("wake check must bind its exact lifecycle lineage")
            observed = parse_aware_iso8601(evidence.payload["observed_at"])
            checked = parse_aware_iso8601(payload["checked_at"])
            if (observed <= parse_aware_iso8601(archive.payload["quota_observed_at"])
                    or checked < observed
                    or (checked - observed).total_seconds() > WAKE_USAGE_MAX_AGE_SECONDS):
                raise ValueError("wake check requires new, fresh provider usage evidence")
        elif event.kind == EventKind.WAKE_READY:
            if event.source_ref != "SleepWakeCoordinator" or len(ordered) != 1 or ordered[0].kind != EventKind.WAKE_CHECK:
                raise ValueError("wake ready requires the coordinator and one wake check")
            check = ordered[0]
            evidence = next((candidate for candidate in prior_events
                             if candidate.event_id == check.payload["usage_evidence_event_id"]), None)
            if (evidence is None or evidence.kind != EventKind.PROVIDER_USAGE_EVIDENCE
                    or evidence.payload["total"] <= 0 or evidence.payload["remaining"] <= 0):
                raise ValueError("wake ready requires positive provider usage evidence")
            if (payload["wake_check_event_id"] != check.event_id
                    or payload["epoch_id"] != check.payload["epoch_id"]
                    or parse_aware_iso8601(payload["ready_at"]) < parse_aware_iso8601(check.payload["checked_at"])):
                raise ValueError("wake ready must bind a checked epoch")
            sleep = next((candidate for candidate in prior_events
                          if candidate.event_id == check.payload["sleep_event_id"]), None)
            archive = None if sleep is None else next((candidate for candidate in prior_events
                if candidate.event_id == sleep.payload.get("archive_event_id")), None)
            rolling_lineage = (evidence.payload.get("evidence_version") == "provider_usage_v2"
                               and evidence.payload.get("window_kind") == "rolling_5h"
                               and archive is not None
                               and archive.payload.get("archive_version") == "sleep_archive_v3"
                               and archive.payload.get("quota_window_kind") == "rolling_5h")
            if payload["protocol_version"] == "sleep_wake_v2":
                if not rolling_lineage:
                    raise ValueError("wake ready v2 requires rolling_5h archive and usage lineage")
            elif rolling_lineage:
                raise ValueError("rolling_5h lineage requires wake ready v2")
        elif event.kind == EventKind.AWAKE:
            if event.source_ref != "SleepWakeCoordinator" or len(ordered) != 1 or ordered[0].kind != EventKind.WAKE_READY:
                raise ValueError("awake requires the coordinator and one wake-ready parent")
            ready = ordered[0]
            if (payload["wake_ready_event_id"] != ready.event_id
                    or payload["epoch_id"] != ready.payload["epoch_id"]
                    or parse_aware_iso8601(payload["awakened_at"]) < parse_aware_iso8601(ready.payload["ready_at"])):
                raise ValueError("awake must bind a ready epoch")
        elif event.kind == EventKind.UNATTENDED_WAKE_RUN:
            if event.source_ref != "UnattendedResearchController" or len(ordered) != 2:
                raise ValueError("unattended wake run requires awake and continuation policy parents")
            awake, continuation = ordered
            if (awake.kind != EventKind.AWAKE
                    or continuation.kind != EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY):
                raise ValueError("unattended wake run parents must be awake followed by continuation policy")
            if (payload["awake_event_id"] != awake.event_id
                    or payload["continuation_id"] != continuation.payload["continuation_id"]
                    or payload["authorization_policy_id"] != continuation.payload["unattended_policy_id"]):
                raise ValueError("unattended wake run must bind its awake event and continuation policy")
            records_by_id = {candidate.event_id: candidate for candidate in prior_events}
            ready = records_by_id.get(awake.payload["wake_ready_event_id"])
            check = None if ready is None else records_by_id.get(ready.payload.get("wake_check_event_id"))
            sleep = None if check is None else records_by_id.get(check.payload.get("sleep_event_id"))
            if (ready is None or ready.kind != EventKind.WAKE_READY
                    or check is None or check.kind != EventKind.WAKE_CHECK
                    or sleep is None or sleep.kind != EventKind.SLEEP_ENTERED
                    or sleep.payload.get("continuation_policy_event_id") != continuation.event_id
                    or epoch_terminated(awake.payload["epoch_id"])
                    or parse_aware_iso8601(event.created_at) < parse_aware_iso8601(awake.payload["awakened_at"])
                    or parse_aware_iso8601(event.created_at) >= parse_aware_iso8601(continuation.payload["expires_at"])):
                raise ValueError("unattended wake run requires its exact non-terminal continuation epoch")
        elif event.kind == EventKind.WAKE_TERMINAL:
            allowed = {EventKind.SLEEP_ENTERED: "slept_at", EventKind.WAKE_CHECK: "checked_at",
                       EventKind.WAKE_READY: "ready_at", EventKind.AWAKE: "awakened_at"}
            if (event.source_ref != "SleepWakeCoordinator" or len(ordered) != 1
                    or ordered[0].kind not in allowed):
                raise ValueError("wake terminal requires one current lifecycle parent")
            lifecycle = ordered[0]
            if (payload["lifecycle_event_id"] != lifecycle.event_id
                    or payload["epoch_id"] != lifecycle.payload["epoch_id"]
                    or parse_aware_iso8601(payload["terminal_at"]) < parse_aware_iso8601(lifecycle.payload[allowed[lifecycle.kind]])):
                raise ValueError("wake terminal must bind its current lifecycle epoch")
        elif event.kind == EventKind.REWARD_OBSERVATION:
            if len(ordered) != 1 or ordered[0].kind not in (EventKind.ACTION_RESULT, EventKind.LOOP_TICK):
                raise ValueError("reward must bind exactly one action result or loop tick")
            if payload["target_event_id"] != ordered[0].event_id:
                raise ValueError("reward target_event_id must name its parent")
        elif event.kind == EventKind.VALUE_ESTIMATE:
            if len(ordered) != 1 or ordered[0].kind not in (EventKind.ACTION_RESULT, EventKind.LOOP_TICK):
                raise ValueError("value estimate must bind exactly one action result or loop tick")
            if payload["target_event_id"] != ordered[0].event_id:
                raise ValueError("value target_event_id must name its parent")
        elif event.kind == EventKind.RPE_UPDATE:
            if len(ordered) != 2:
                raise ValueError("RPE update requires reward and prior value parents")
            reward_parent = parents.get(payload["reward_event_id"])
            value_parent = parents.get(payload["prior_value_event_id"])
            if reward_parent is None or value_parent is None:
                raise ValueError("RPE parent IDs must be listed as parents")
            if reward_parent.kind != EventKind.REWARD_OBSERVATION or value_parent.kind != EventKind.VALUE_ESTIMATE:
                raise ValueError("RPE parents must be reward observation and value estimate")
            if (value_parent.sequence is None or reward_parent.sequence is None
                    or value_parent.sequence >= reward_parent.sequence):
                raise ValueError("value estimate must be recorded before reward observation")
            reward = reward_parent.payload
            value = value_parent.payload
            if (payload["reward"] != reward["normalized_value"]
                    or payload["prior_value"] != value["value"]
                    or payload["transition_id"] != value["transition_id"]
                    or payload["state_key"] != value["state_key"]
                    or payload["action_key"] != value["action_key"]
                    or reward["target_event_id"] != value["target_event_id"]):
                raise ValueError("RPE update fields must match its reward and value inputs")
            if (payload["alpha"] != value["alpha"]
                    or payload["gamma"] != value["gamma"]
                    or payload["clip"] != value["clip"]
                    or payload["formula_version"] != value["formula_version"]):
                raise ValueError("RPE config must match the pinned value estimate config")
            if value["terminal"] and payload["next_value"] != 0.0:
                raise ValueError("terminal TD transitions require next_value zero")
            expected_delta = payload["reward"] + payload["gamma"] * payload["next_value"] - payload["prior_value"]
            expected_clipped = max(-payload["clip"], min(payload["clip"], expected_delta))
            expected_updated = payload["prior_value"] + payload["alpha"] * expected_clipped
            if not math.isclose(payload["raw_delta"], expected_delta, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("RPE raw_delta must equal reward + gamma * next_value - prior_value")
            if not math.isclose(payload["clipped_delta"], expected_clipped, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("RPE clipped_delta must equal clip(raw_delta, -clip, clip)")
            if not math.isclose(payload["updated_value"], expected_updated, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("RPE updated_value must equal prior_value + alpha * clipped_delta")
        elif event.kind == EventKind.FRONTIER_RANKING_DECISION:
            if len(ordered) != 1 or ordered[0].kind != EventKind.EXPEDITION_AUTHORIZATION_CONSUMED:
                raise ValueError("frontier ranking requires exactly one consumed expedition authorization parent")
            consumed = ordered[0]
            authorization = next((candidate for candidate in prior_events
                                  if candidate.event_id == consumed.parent_event_ids[0]), None)
            if (authorization is None or authorization.kind != EventKind.EXPEDITION_AUTHORIZATION
                    or authorization.payload.get("version") not in {"expedition_authorization_v2", "expedition_authorization_v3"}
                    or authorization.payload.get("learning_mode") != "active"
                    or authorization.payload.get("learner_spec_digest") != payload["learner_spec_digest"]):
                raise ValueError("frontier ranking requires an active v2 or v3 authorization bound to its learner spec")
        elif event.kind == EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE:
            if len(ordered) != 1 or ordered[0].kind != EventKind.FRONTIER_RANKING_DECISION:
                raise ValueError("frontier value estimate requires one ranking decision parent")
            ranking = ordered[0]
            for key in ("transition_id", "learner_spec_digest", "scope"):
                if payload[key] != ranking.payload[key]:
                    raise ValueError("frontier value estimate must match its ranking binding")
            if payload["ranking_event_id"] != ranking.event_id:
                raise ValueError("frontier value estimate must name its ranking parent")
            if (payload.get("strategy_arm_id") != ranking.payload.get("strategy_arm_id")
                    or payload.get("strategy_arm_version") != ranking.payload.get("strategy_arm_version")):
                raise ValueError("frontier value estimate strategy arm must match its ranking parent")
        elif event.kind == EventKind.FRONTIER_EVIDENCE_OBSERVATION:
            if len(ordered) != 2 or ordered[0].kind != EventKind.FRONTIER_RANKING_DECISION:
                raise ValueError("frontier evidence requires ranking then external source parents")
            ranking, source = ordered
            experiment_result_valid = (source.kind == EventKind.EXPERIMENT_RESULT
                and source.source_kind == SourceKind.EXTERNAL_VERIFIER
                and source.payload["status"] in {"supported", "refuted", "inconclusive"}
                and source.payload["complete_run_set"]
                and source.payload["baseline_control_valid"]
                and source.payload["reproducible"])
            if (source.kind in FRONTIER_FORBIDDEN_EVIDENCE_KINDS
                    or source.kind in FRONTIER_LEARNING_EVENT_KINDS
                    or (source.kind not in {EventKind.OBSERVATION, EventKind.TOOL_RESULT, EventKind.ACTION_RESULT, EventKind.CORRECTION}
                        and not experiment_result_valid)
                    or (not experiment_result_valid and source.source_kind not in {SourceKind.USER, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER})):
                raise ValueError("frontier evidence source is not eligible external evidence")
            if (payload["ranking_event_id"] != ranking.event_id
                    or payload["source_event_id"] != source.event_id
                    or payload["transition_id"] != ranking.payload["transition_id"]
                    or payload["learner_spec_digest"] != ranking.payload["learner_spec_digest"]
                    or payload["scope"] != ranking.payload["scope"]):
                raise ValueError("frontier evidence must bind its ranking and external source")
            if (payload.get("strategy_arm_id") != ranking.payload.get("strategy_arm_id")
                    or payload.get("strategy_arm_version") != ranking.payload.get("strategy_arm_version")):
                raise ValueError("frontier evidence strategy arm must match its ranking parent")
            if experiment_result_valid:
                ranking_consumed = next((candidate for candidate in prior_events
                                         if candidate.event_id == ranking.parent_event_ids[0]), None)
                ranking_authorization = None if ranking_consumed is None or not ranking_consumed.parent_event_ids else next(
                    (candidate for candidate in prior_events if candidate.event_id == ranking_consumed.parent_event_ids[0]), None)
                if (source.payload["learner_spec_digest"] != ranking.payload["learner_spec_digest"]
                        or ranking_authorization is None
                        or source.payload["authorization_id"] != ranking_authorization.payload.get("authorization_id")):
                    raise ValueError("frontier experiment result must bind the ranking authorization and learner spec")
        elif event.kind == EventKind.EXPERIMENT_PLAN_LOCKED:
            if len(ordered) != 1 or ordered[0].kind != EventKind.EXPEDITION_AUTHORIZATION_CONSUMED:
                raise ValueError("experiment plan requires exactly one consumed v3 authorization parent")
            consumed = ordered[0]
            authorization = next((candidate for candidate in prior_events
                                  if candidate.event_id == consumed.parent_event_ids[0]), None)
            if (authorization is None or authorization.kind != EventKind.EXPEDITION_AUTHORIZATION
                    or authorization.payload.get("version") != "expedition_authorization_v3"
                    or authorization.payload.get("learning_mode") not in {"active", "shadow"}
                    or payload["authorization_id"] != authorization.payload["authorization_id"]
                    or payload["learner_spec_digest"] != authorization.payload["learner_spec_digest"]
                    or payload["experiment_registry_digest"] != authorization.payload["experiment_registry_digest"]
                    or payload["experiment_kind"] not in authorization.payload["experiment_kinds"]):
                raise ValueError("experiment plan requires its authorized v3 template and digests")
            if (payload["max_trials"] > authorization.payload["experiment_max_trials"]
                    or payload["max_steps"] > authorization.payload["experiment_max_steps"]
                    or payload["max_wall_ms"] > authorization.payload["experiment_max_wall_ms"]):
                raise ValueError("experiment plan exceeds its authorized resource caps")
            if sum(1 for candidate in prior_events
                   if candidate.kind == EventKind.EXPERIMENT_PLAN_LOCKED
                   and candidate.payload.get("authorization_id") == payload["authorization_id"]) >= authorization.payload["max_experiments"]:
                raise ValueError("experiment authorization has reached max_experiments")
        elif event.kind == EventKind.EXPERIMENT_EXECUTION_STARTED:
            if len(ordered) != 1 or ordered[0].kind != EventKind.EXPERIMENT_PLAN_LOCKED:
                raise ValueError("experiment execution requires exactly one locked plan parent")
            plan = ordered[0]
            for key in ("experiment_id", "experiment_registry_digest", "experiment_instance_digest", "max_trials", "max_steps", "max_wall_ms"):
                if payload[key] != plan.payload[key]:
                    raise ValueError("experiment execution must match its locked plan")
            if (payload["plan_event_id"] != plan.event_id
                    or parse_aware_iso8601(payload["started_at"]) < parse_aware_iso8601(plan.created_at)):
                raise ValueError("experiment execution must bind and follow its locked plan")
        elif event.kind == EventKind.EXPERIMENT_RESULT:
            if len(ordered) != 1 or ordered[0].kind != EventKind.EXPERIMENT_EXECUTION_STARTED:
                raise ValueError("experiment result requires exactly one execution parent")
            execution = ordered[0]
            plan = next((candidate for candidate in prior_events if candidate.event_id == execution.payload["plan_event_id"]), None)
            if plan is None or plan.kind != EventKind.EXPERIMENT_PLAN_LOCKED:
                raise ValueError("experiment result locked plan is unavailable")
            for key in ("execution_id", "experiment_id", "experiment_registry_digest", "experiment_instance_digest"):
                expected = execution.payload[key]
                if payload[key] != expected:
                    raise ValueError("experiment result must match its execution")
            for key in ("plan_event_id", "authorization_id", "learner_spec_digest", "claim_scope"):
                expected = plan.event_id if key == "plan_event_id" else plan.payload[key]
                if payload[key] != expected:
                    raise ValueError("experiment result must match its locked plan")
            if payload["status"] == "invalid" and (payload["complete_run_set"] or payload["baseline_control_valid"] or payload["reproducible"]):
                raise ValueError("invalid experiment result cannot assert reward eligibility")
        elif event.kind == EventKind.FRONTIER_VECTOR_REWARD:
            if len(ordered) != 2 or ordered[0].kind != EventKind.FRONTIER_EVIDENCE_OBSERVATION or ordered[1].kind != EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE:
                raise ValueError("frontier vector reward requires evidence then value parents")
            evidence, value = ordered
            if (payload["evidence_event_id"] != evidence.event_id or payload["value_event_id"] != value.event_id
                    or payload["ranking_event_id"] != evidence.payload["ranking_event_id"]
                    or payload["ranking_event_id"] != value.payload["ranking_event_id"]):
                raise ValueError("frontier vector reward must bind its evidence and value parents")
            for key in ("transition_id", "learner_spec_digest", "scope"):
                if payload[key] != evidence.payload[key] or payload[key] != value.payload[key]:
                    raise ValueError("frontier vector reward must match its input binding")
            if (payload.get("strategy_arm_id") != evidence.payload.get("strategy_arm_id")
                    or payload.get("strategy_arm_id") != value.payload.get("strategy_arm_id")
                    or payload.get("strategy_arm_version") != evidence.payload.get("strategy_arm_version")
                    or payload.get("strategy_arm_version") != value.payload.get("strategy_arm_version")):
                raise ValueError("frontier vector reward strategy arm must match its input parents")
            source = next((candidate for candidate in prior_events
                           if candidate.event_id == evidence.payload["source_event_id"]), None)
            if source is None or source.kind != EventKind.EXPERIMENT_RESULT:
                raise ValueError("frontier vector rewards require a qualifying experiment result source")
            plan = next((candidate for candidate in prior_events
                         if candidate.event_id == source.payload["plan_event_id"]), None)
            if plan is None or plan.kind != EventKind.EXPERIMENT_PLAN_LOCKED:
                raise ValueError("frontier experiment result plan is unavailable")
            if "strategy_arm_id" in payload:
                authorization = next((candidate for candidate in prior_events
                                      if candidate.event_id == plan.parent_event_ids[0]), None)
                if authorization is None or authorization.kind != EventKind.EXPEDITION_AUTHORIZATION_CONSUMED:
                    raise ValueError("frontier experiment result authorization lineage is unavailable")
                authorization = next((candidate for candidate in prior_events
                                      if candidate.event_id == authorization.parent_event_ids[0]), None)
                if (authorization is None or authorization.kind != EventKind.EXPEDITION_AUTHORIZATION
                        or authorization.payload.get("version") != "expedition_authorization_v3"
                        or authorization.payload.get("experiment_registry_digest") != plan.payload["experiment_registry_digest"]):
                    raise ValueError("frontier experiment result must use its authorized registry")
                expected_arm = frontier_strategy_arm_id(
                    authorization.payload["experiment_registry_digest"],
                    plan.payload["learner_spec_digest"], plan.payload["experiment_kind"],
                    FRONTIER_OFFLINE_EXPERIMENT_ACTION_MODE)
                if payload["strategy_arm_id"] != expected_arm:
                    raise ValueError("frontier vector reward strategy arm must equal its authorized experiment strategy")
            duplicate = any(candidate.kind == EventKind.EXPERIMENT_RESULT
                            and candidate.event_id != source.event_id
                            and candidate.payload.get("result_digest") == source.payload["result_digest"]
                            for candidate in prior_events)
            expected_vector = frontier_experiment_reward_vector(
                plan.payload["experiment_kind"], source.payload["status"], not duplicate)
            if payload["reward_vector"] != expected_vector:
                raise ValueError("frontier reward vector must equal the fixed experiment projection")
        elif event.kind == EventKind.FRONTIER_TD_UPDATE:
            if len(ordered) != 2 or ordered[0].kind != EventKind.FRONTIER_VECTOR_REWARD or ordered[1].kind != EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE:
                raise ValueError("frontier TD update requires reward then prior value parents")
            reward, value = ordered
            if (payload["reward_event_id"] != reward.event_id or payload["prior_value_event_id"] != value.event_id
                    or payload["ranking_event_id"] != reward.payload["ranking_event_id"]
                    or payload["ranking_event_id"] != value.payload["ranking_event_id"]):
                raise ValueError("frontier TD update must bind its reward and value parents")
            for key in ("transition_id", "learner_spec_digest", "scope"):
                if payload[key] != reward.payload[key] or payload[key] != value.payload[key]:
                    raise ValueError("frontier TD update must match its input binding")
            if (payload.get("strategy_arm_id") != reward.payload.get("strategy_arm_id")
                    or payload.get("strategy_arm_id") != value.payload.get("strategy_arm_id")
                    or payload.get("strategy_arm_version") != reward.payload.get("strategy_arm_version")
                    or payload.get("strategy_arm_version") != value.payload.get("strategy_arm_version")):
                raise ValueError("frontier TD update strategy arm must match its input parents")
            if (payload["reward_vector"] != reward.payload["reward_vector"]
                    or payload["prior_value_vector"] != value.payload["value_vector"]
                    or payload["alpha"] != value.payload["alpha"]
                    or payload["gamma"] != value.payload["gamma"]
                    or payload["clip"] != value.payload["clip"]):
                raise ValueError("frontier TD update must match its reward and value inputs")
            for index in range(len(FRONTIER_CHANNEL_ORDER)):
                expected_delta = (payload["reward_vector"][index] + payload["gamma"] * payload["next_value_vector"][index]
                                  - payload["prior_value_vector"][index])
                expected_clipped = max(-payload["clip"], min(payload["clip"], expected_delta))
                expected_updated = payload["prior_value_vector"][index] + payload["alpha"] * expected_clipped
                if (not math.isclose(payload["raw_delta_vector"][index], expected_delta, rel_tol=0.0, abs_tol=1e-12)
                        or not math.isclose(payload["clipped_delta_vector"][index], expected_clipped, rel_tol=0.0, abs_tol=1e-12)
                        or not math.isclose(payload["updated_value_vector"][index], expected_updated, rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError("frontier TD vector arithmetic is invalid")

    @staticmethod
    def _string_list(value: Any) -> None:
        if not isinstance(value, list) or len(value) > MAX_LIST or any(not isinstance(item, str) or len(item) > MAX_SHORT_TEXT for item in value):
            raise ValueError("value must be a bounded string list")

    @staticmethod
    def _seed(seed: Any) -> None:
        required = {"cue_terms", "policy_bias", "scope", "provenance_event_ids", "strength", "confidence", "status", "seed_id", "created_at", "updated_at", "expires_at", "version", "counterevidence"}
        if not isinstance(seed, dict) or set(seed) != required or seed.get("status") != "candidate":
            raise ValueError("seed must use the bounded candidate schema")
        SQLiteEventStore._strings(seed, {"policy_bias", "scope", "seed_id", "created_at", "updated_at"}, MAX_SHORT_TEXT)
        SQLiteEventStore._string_list(seed["cue_terms"]); SQLiteEventStore._string_list(seed["provenance_event_ids"])
        if seed["expires_at"] is not None and (not isinstance(seed["expires_at"], str) or len(seed["expires_at"]) > MAX_SHORT_TEXT): raise ValueError("invalid seed expiry")
        if (not all(isinstance(seed[key], (int, float)) and not isinstance(seed[key], bool) for key in ("strength", "confidence"))
                or not all(isinstance(seed[key], int) and not isinstance(seed[key], bool) for key in ("version", "counterevidence"))): raise ValueError("invalid seed numeric fields")

    @staticmethod
    def _claim(claim: Any) -> None:
        required = {"claim_id", "kind", "statement", "evidence_event_ids", "confidence", "created_at", "expires_at"}
        if not isinstance(claim, dict) or set(claim) != required or claim["kind"] not in {"role", "capability", "commitment", "epistemic", "continuity", "boundary"}:
            raise ValueError("claim must use the bounded schema")
        SQLiteEventStore._strings(claim, {"claim_id", "kind", "statement", "created_at"}, MAX_TEXT)
        SQLiteEventStore._string_list(claim["evidence_event_ids"])
        if not isinstance(claim["confidence"], (int, float)) or isinstance(claim["confidence"], bool): raise ValueError("invalid claim confidence")
        if claim["expires_at"] is not None and (not isinstance(claim["expires_at"], str) or len(claim["expires_at"]) > MAX_SHORT_TEXT): raise ValueError("invalid claim expiry")

    @staticmethod
    def _action(action: Any) -> None:
        required = {"action_type", "required_capability", "is_mutating", "public_summary"}
        if not isinstance(action, dict) or set(action) != required: raise ValueError("invalid decision action projection")
        if (action.get("action_type") not in {"response", "read", "write", "other"} or action.get("required_capability") not in (None, "declared")
                or not isinstance(action.get("is_mutating"), bool) or action.get("public_summary") != "Action proposal recorded for policy review."):
            raise ValueError("invalid decision action projection")

    def _append(self, event: CognitiveEvent) -> CognitiveEvent:
        row = self.connection.execute(
            "SELECT sequence, content_hash FROM cognitive_events WHERE session_id = ? "
            "ORDER BY sequence DESC LIMIT 1", (event.session_id,)
        ).fetchone()
        sequence = 1 if row is None else int(row["sequence"]) + 1
        self._validate_event(event, sequence)
        stored = replace(event, sequence=sequence,
                         previous_hash=None if row is None else str(row["content_hash"]))
        stored = replace(stored, content_hash=self._content_hash(stored))
        self.connection.execute("""INSERT INTO cognitive_events
            (event_id, session_id, sequence, kind, source_kind, source_ref,
             payload_json, confidence, parent_event_ids_json, created_at,
             previous_hash, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (stored.event_id, stored.session_id, stored.sequence, stored.kind.value,
             stored.source_kind.value, stored.source_ref, canonical_json(stored.payload),
             stored.confidence, canonical_json(list(stored.parent_event_ids)), stored.created_at,
             stored.previous_hash, stored.content_hash))
        return stored

    def append(self, event: CognitiveEvent, commit: bool = True) -> CognitiveEvent:
        """Append and return the filled event; ``commit=False`` requires transaction()."""
        if commit:
            with self.transaction():
                return self._append(event)
        if not self.connection.in_transaction:
            raise RuntimeError("append(commit=False) requires an active store transaction")
        return self._append(event)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> CognitiveEvent:
        return CognitiveEvent(event_id=row["event_id"], session_id=row["session_id"],
            sequence=int(row["sequence"]), kind=EventKind(row["kind"]),
            source_kind=SourceKind(row["source_kind"]), source_ref=row["source_ref"],
            payload=json.loads(row["payload_json"]), confidence=float(row["confidence"]),
            parent_event_ids=tuple(json.loads(row["parent_event_ids_json"])),
            created_at=row["created_at"], previous_hash=row["previous_hash"], content_hash=row["content_hash"])

    def list(self, session_id: str) -> List[CognitiveEvent]:
        rows = self.connection.execute("SELECT * FROM cognitive_events WHERE session_id = ? ORDER BY sequence", (session_id,)).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        row = self.connection.execute("SELECT * FROM cognitive_events WHERE event_id = ?", (event_id,)).fetchone()
        return None if row is None else self._from_row(row)

    def verify_chain(self, session_id: str) -> bool:
        previous_hash = None
        events = self.list(session_id)
        by_id = {event.event_id: event for event in events}
        for expected, event in enumerate(events, 1):
            if event.sequence != expected or event.previous_hash != previous_hash:
                return False
            if event.content_hash != self._content_hash(replace(event, content_hash=None)):
                return False
            if not event.parent_event_ids:
                if event.kind not in ROOT_EVENT_KINDS:
                    return False
            else:
                if len(set(event.parent_event_ids)) != len(event.parent_event_ids):
                    return False
                for parent_id in event.parent_event_ids:
                    parent = by_id.get(parent_id)
                    if parent is None or parent.sequence >= event.sequence:
                        return False
            previous_hash = event.content_hash
        return True

    def purge_session(self, session_id: str) -> Dict[str, Any]:
        """Delete logical records; secure deletion is best-effort, not an SSD guarantee."""
        with self.transaction():
            event_count = self.connection.execute("DELETE FROM cognitive_events WHERE session_id = ?", (session_id,)).rowcount
            has_seeds = self.connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='seeds'").fetchone()
            seed_count = 0 if has_seeds is None else self.connection.execute("DELETE FROM seeds WHERE session_id = ?", (session_id,)).rowcount
        report = {"logical_deletion": True, "events_deleted": event_count, "seeds_deleted": seed_count,
                  "secure_delete": self.connection.execute("PRAGMA secure_delete").fetchone()[0] == 1,
                  "vacuum_attempted": self._is_file, "vacuum_completed": False}
        if self._is_file:
            try:
                self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.connection.execute("VACUUM")
                report["vacuum_completed"] = True
            except sqlite3.OperationalError:
                # Concurrent readers can prevent compaction; logical deletion still completed.
                pass
        return report

    def close(self) -> None:
        self.connection.close()
