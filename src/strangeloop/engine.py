"""The bounded, event-sourced Strangeloop turn loop."""

from __future__ import annotations

import re
import json
import math
import time
from hashlib import sha256
from datetime import datetime, timezone, timedelta
from typing import Any, BinaryIO, Dict, Iterable, Optional, Sequence, Tuple
from urllib.parse import urlsplit
from uuid import uuid4

from .autoloop import (CycleResult, FunctionalLoopController, LoopConfig,
                       LoopState, StopReason, TickContext, TickRecord,
                       TickTrigger)

from .contracts import (
    ActionProposal, CognitiveEvent, DecisionRecord, Deliberation, EventKind,
    FRONTIER_STRATEGY_ARM_VERSION, SeedDisposition, SeedGuidance, SeedStatus,
    SourceKind, TurnResult, WorkspaceFrame,
)
from .deliberation import Deliberator, TransparentHeuristicDeliberator
from .media import (BasicMetadataPerceptor, MediaArtifact, MediaPerceptor,
                    Percept, ingest_media, validate_percepts)
from .metacognition import (EvidencePolarity, MirrorAuditor, PublicEvidence)
from .policy import PolicyGate
from .seeds import (SQLiteSeedStore, SeedStandingPolicy,
                    canonical_seed_identity)
from .memory_graph import MemoryGraph
from .self_model import EventSourcedSelfModel
from .store import SQLiteEventStore
from .store import frontier_experiment_reward_vector
from .td import (RewardObservation, RewardSource, SAFE_ACTION_CLASSES,
                 SafeActionPreference, TDConfig, Transition, ValueTable)
from .providers.kimi_code import (KeychainSecretResolver, KimiCodeProviderError,
                                  KimiCodeRuntime, KimiCodeSettings,
                                  KimiDeliberator, KimiLoopReflector, KimiToolPlanner)
from .quota import QuotaController, ReservationInvalidationReason
from .drives import (DriveObservation, DualIntrinsicDrives, ObservationSource,
                     UserFeedbackKind)
from .improvement import ProtectedImprovementControlPlane
from .tool_session import PreparedToolCall, ToolSession
from .sleep import SleepState, SleepWakeCoordinator, build_public_archive
from .quota import QuotaSource, ForegroundRefreshGate, ForegroundRefreshPolicy
from .kimi_cli import KimiCliUsageResult
from .capabilities import ResearchAutonomyProfile
from .expedition import (ExpeditionConfig, ExpeditionOutcome, ExpeditionScheduler,
                         SeedGuidanceSnapshot)
from .frontier_learning import (EvidenceKind, EvidenceSource, FrontierEvidence,
                                FrontierLearner, FrontierLearningSpec)
from .experiment_harness import (ExperimentAuthority, ExperimentBudget, ExperimentHarness,
                                 ExperimentKind, ExperimentRegistryPolicy, ExperimentSpec,
                                 ExperimentStatus)
from .unattended import (CancellationToken, PreparedResearchAction, ResearchExecution,
                         ResearchProposal, UnattendedContext, UnattendedPolicy,
                         UnattendedResearchController, UnattendedStopReason)


class _FrontierLearnerRanker:
    """Content-free adapter from scheduler descriptors to the local learner."""

    def __init__(self, learner: FrontierLearner, state_id: str) -> None:
        self._learner = learner
        self._state_id = state_id

    def rank(self, candidates: Sequence[Dict[str, object]],
             state_context: Dict[str, object]) -> Sequence[str]:
        # The scheduler deliberately supplies only bounded descriptors.  Do
        # not add a goal, URL, query, tool result, or provider signal here.
        del state_context
        by_arm: Dict[str, list[str]] = {}
        for candidate in candidates:
            by_arm.setdefault(str(candidate["strategy_arm_id"]), []).append(str(candidate["task_id"]))
        # Learn a transferable strategy identity, then deterministically
        # expand it back into the scheduler's exact task-instance IDs.
        ranked_arms = self._learner.rank_arms(self._state_id, tuple(sorted(by_arm)))
        return tuple(task_id for arm in ranked_arms for task_id in sorted(by_arm[arm.arm_id]))


class StrangeloopAgent:
    """An auditable task agent inspired by Yogacara cognitive theory.

    The self-model is functional and revocable.  This class does not claim
    subjective experience, sentience, an intrinsic self, or religious insight.
    """

    def __init__(self, session_id: Optional[str] = None,
                 event_store: Optional[SQLiteEventStore] = None,
                 seed_store: Optional[SQLiteSeedStore] = None,
                 self_model: Optional[EventSourcedSelfModel] = None,
                 deliberator: Optional[Deliberator] = None,
                 policy_gate: Optional[PolicyGate] = None,
                 value_table: Optional[ValueTable] = None,
                 runtime: Optional[KimiCodeRuntime] = None,
                 quota_controller: Optional[QuotaController] = None,
                 tool_session: Optional[ToolSession] = None,
                 drives: Optional[DualIntrinsicDrives] = None,
                 improvement_control: Optional[ProtectedImprovementControlPlane] = None,
                 sleep_coordinator: Optional[SleepWakeCoordinator] = None,
                 memory_graph: Optional[MemoryGraph] = None,
                 loop_remote_call_budget: int = 1) -> None:
        self.session_id = session_id or "session_%s" % uuid4().hex
        self.event_store = event_store or SQLiteEventStore()
        self.seed_store = seed_store or SQLiteSeedStore(self.event_store)
        # The graph is a cache-like projection of the same event ledger.  It
        # is not consulted for approval, policy, capability, reward, quota,
        # sleep, or lifecycle decisions.
        self.memory_graph = memory_graph or MemoryGraph(self.event_store)
        # This cache is intentionally updated only by the agent's caller
        # thread.  The monitor serves it read-only from its HTTP thread, so it
        # never touches the thread-affine SQLite connection.
        self._memory_graph_public_status = self.memory_graph.ensure_current(
            self.session_id).to_dict()
        self.self_model = self_model or EventSourcedSelfModel(self.event_store)
        if (not isinstance(loop_remote_call_budget, int)
                or isinstance(loop_remote_call_budget, bool)
                or not 0 <= loop_remote_call_budget <= 32):
            raise ValueError("loop_remote_call_budget must be an integer between 0 and 32")
        # Library construction remains entirely offline.  The CLI explicitly
        # supplies K3; embedded callers and tests get no Keychain or network
        # activity unless they opt in with a runtime.
        self.runtime = runtime
        self.quota_controller = quota_controller or getattr(runtime, "_quota_controller", None)
        self.tool_session = tool_session
        self.drives = drives
        self.improvement_control = improvement_control
        self.sleep_coordinator = sleep_coordinator
        self._managed_usage_windows = ()
        self._managed_usage_observed_at = None
        self._sleep_archive_event_id = None
        self._sleep_event_id = None
        self._sleep_epoch_id = None
        self._auto_wake_policy_event_id = None
        # An unattended research controller is an explicitly user-authorized,
        # caller-driven read-only profile.  It is not the DMN loop and does
        # not survive sleep, restart, quota exhaustion, or purge.
        self._unattended: Optional[UnattendedResearchController] = None
        self._unattended_policy_event_id: Optional[str] = None
        self._unattended_prepared: Dict[str, PreparedToolCall] = {}
        self._unattended_proposals: Dict[str, Tuple[Any, str, str]] = {}
        self._unattended_findings: list[Dict[str, Any]] = []
        self._unattended_allowed_tools: Tuple[str, ...] = ()
        self._last_unattended_planner_outcome = "not_started"
        # This descriptor is deliberately process-local.  It contains no tool
        # plan, grant, page content, provider response, or model reasoning.
        # It is consumed at most once after a fresh foreground wake and is
        # never reconstructed from the event store after a restart.
        self._wake_continuation: Optional[Dict[str, Any]] = None
        self._wake_continuation_last: Dict[str, Any] = {"armed": False, "reason": "not_armed"}
        self._instance_marker = "instance_%s" % uuid4().hex
        self._expedition: Optional[ExpeditionScheduler] = None
        self._expedition_goal: Optional[str] = None
        self._expedition_goal_digest: Optional[str] = None
        self._expedition_refresh_gate: Optional[ForegroundRefreshGate] = None
        self._expedition_approval_event_id: Optional[str] = None
        self._expedition_current_persona: Optional[str] = None
        self._expedition_domains: set[str] = set()
        self._expedition_useful_findings = 0
        # The frontier learner is a host-side, bounded ranking experiment.  It
        # is deliberately separate from the existing safe-action TD table and
        # receives no policy, grant, quota, or persistence authority.
        self._expedition_learning_mode = "off"
        self._expedition_learner: Optional[FrontierLearner] = None
        self._expedition_learner_spec: Optional[FrontierLearningSpec] = None
        self._expedition_learner_spec_digest: Optional[str] = None
        self._expedition_frontier_state_id: Optional[str] = None
        self._expedition_consumption_event_id: Optional[str] = None
        self._expedition_active_transition: Optional[Dict[str, str]] = None
        self._expedition_experiment_policy: Optional[ExperimentRegistryPolicy] = None
        self._expedition_experiment_harness: Optional[ExperimentHarness] = None
        self._expedition_experiment_completed = 0
        self._expedition_experiment_last: Dict[str, Any] = {}
        self._expedition_host_seed: Optional[str] = None
        self._expedition_rewarded_strategy_arms: set[str] = set()
        self._expedition_post_reward_selection_count = 0
        self._expedition_last_post_reward_selection_reason = "none"
        self._expedition_duplicate_experiment_result_count = 0
        self._expedition_last_strategy_arm_id: Optional[str] = None
        self._expedition_last_strategy_arm_version: Optional[str] = None
        # The durable context event is an audit edge only.  The current
        # guidance snapshot is recomputed before each slice, so a bounded
        # user-originated seed update can affect a later slice but never an
        # already selected task.
        self._expedition_seed_context_event_id: Optional[str] = None
        if tool_session is not None and tool_session.session_id != self.session_id:
            raise ValueError("tool_session must belong to the agent session")
        self._heuristic_deliberator = TransparentHeuristicDeliberator()
        self.deliberator = deliberator or (KimiDeliberator(runtime)
                                           if runtime is not None else self._heuristic_deliberator)
        self._uses_kimi_deliberator = isinstance(self.deliberator, KimiDeliberator)
        self._loop_reflector = KimiLoopReflector(runtime) if runtime is not None else None
        self._loop_remote_call_budget = loop_remote_call_budget
        self._loop_remote_calls_remaining = 0
        self.policy_gate = policy_gate or PolicyGate()
        # This table ranks a fixed, safe action vocabulary only.  It is never
        # passed to PolicyGate and has no authority over tools or persistence.
        self.value_table = value_table or ValueTable()
        self._loop: Optional[FunctionalLoopController] = None
        self._loop_run_id: Optional[str] = None
        self._loop_parent_event_id: Optional[str] = None
        self._loop_stop_recorded = False
        self._processed_focus_event_ids = set()
        self._td_transitions: Dict[str, Transition] = {}
        self._td_replay_head_sequence = 0
        # The mirror is a deterministic, local audit only.  It deliberately
        # receives no runtime, tool, capability, drive, TD, quota, seed, or
        # self-model authority.
        self._mirror_auditor = MirrorAuditor()
        self._mirror_status = "ready"
        self._mirror_count = 0
        self._restore_td_history(value_table is not None)

    def update_managed_usage(self, result: Any) -> bool:
        """Accept an adapter result as host telemetry, then possibly pre-sleep.

        Results are only retained as public window counters/timestamps.  This
        method never retains an OAuth credential, raw response, or CLI output.
        """
        if not isinstance(result, KimiCliUsageResult):
            return False
        snapshot = result.snapshot
        windows = getattr(result, "windows", ())
        if snapshot is None or not windows:
            return False
        # A normalized managed snapshot may select weekly as its primary
        # capacity dimension while the independent rolling_5h window still
        # governs the formal sleep threshold.  Accept only a declared managed
        # primary (or legacy None) plus exactly one valid rolling window.
        if (getattr(snapshot, "primary_window_kind", None) not in (None, "weekly", "rolling_5h")
                or sum(1 for window in windows if getattr(window, "kind", None) == "rolling_5h") != 1):
            return False
        if self.quota_controller is None or snapshot.is_estimate:
            return False
        # A low or exhausted authoritative window is expected to make
        # ``allow`` false, but it still must be retained so the checkpoint and
        # sleep transition can be audited.  Accept only the exact normalized
        # snapshot that the trusted host has already ingested as provider data.
        telemetry = self.quota_controller.export_telemetry(now=snapshot.observed_at)
        current = telemetry.get("snapshot")
        if (telemetry.get("source") != QuotaSource.PROVIDER_USAGE.value
                or not isinstance(current, dict)
                or current.get("observed_at") != snapshot.observed_at.isoformat()
                or current.get("reset_at") != snapshot.reset_at.isoformat()
                or current.get("remaining") != snapshot.remaining
                or current.get("total") != snapshot.total
                or current.get("is_estimate") is not False):
            return False
        self._managed_usage_windows = tuple(windows)
        self._managed_usage_observed_at = snapshot.observed_at
        return self._maybe_enter_sleep()

    def sleep_status(self) -> Dict[str, Any]:
        if self.sleep_coordinator is None:
            return {"configured": False, "state": "not_configured", "foreground_only": True}
        result = self.sleep_coordinator.to_payload()
        result.update({"configured": True, "foreground_only": True,
                       "terminal_closed_session_will_not_wake": True,
                       "archive_event_id": self._sleep_archive_event_id,
                       "sleep_event_id": self._sleep_event_id})
        return result

    def set_auto_wake(self, approved: bool) -> bool:
        if self.sleep_coordinator is None:
            raise RuntimeError("sleep coordination is not configured")
        issued = self._sleep_now()
        event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.AUTO_WAKE_POLICY,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "policy_id": "wakepolicy_%s" % uuid4().hex,
                "action": "enable" if approved else "disable", "scope": "next_sleep_epoch",
                "max_auto_runs": 1, "max_ticks": 4,
                "expires_at": (issued + __import__("datetime").timedelta(hours=6)).isoformat(),
                "issued_at": issued.isoformat(), "policy_version": "auto_wake_policy_v1"}))
        accepted = (self.sleep_coordinator.set_user_auto_wake(True, origin="user") if approved
                    else self.sleep_coordinator.revoke_auto_wake())
        self._auto_wake_policy_event_id = event.event_id if approved and accepted else None
        if not approved:
            self._clear_wake_continuation("auto_wake_disabled")
        return accepted

    @staticmethod
    def _profile_digest(profile: ResearchAutonomyProfile) -> str:
        budget = profile.budget
        public = {"workspace_id": profile.workspace_id, "budget": {
            "max_tool_calls": budget.max_tool_calls, "max_total_bytes": budget.max_total_bytes,
            "max_response_bytes": budget.max_response_bytes, "max_wall_ms": budget.max_wall_ms,
            "ttl_seconds": budget.ttl_seconds},
            "capabilities": tuple(item.value for item in profile.capabilities)}
        return sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def prepare_unattended_wake_continuation(self, profile: ResearchAutonomyProfile,
                                              policy: UnattendedPolicy,
                                              approval_event_id: str) -> Dict[str, Any]:
        """Record one explicit USER continuation policy before a quota refresh.

        The opaque process-local capsule is a fixed descriptor, not a grant.
        A later wake creates a new profile and new grants rather than reviving
        any of the pre-sleep authority.
        """
        if self.sleep_coordinator is None or not self.sleep_coordinator.to_payload().get("auto_wake_user_approved"):
            raise RuntimeError("wake continuation requires explicit auto-wake approval")
        approval = self.event_store.get(approval_event_id)
        if (approval is None or approval.session_id != self.session_id or approval.kind != EventKind.OBSERVATION
                or approval.source_kind != SourceKind.USER):
            raise ValueError("wake continuation requires its original USER observation")
        auto = self.event_store.get(self._auto_wake_policy_event_id) if self._auto_wake_policy_event_id else None
        if auto is None:
            raise RuntimeError("wake continuation requires a current auto-wake policy")
        issued = self._sleep_now()
        auto_expiry = datetime.fromisoformat(auto.payload["expires_at"])
        expiry = min(issued + timedelta(hours=6), auto_expiry)
        if expiry <= issued:
            raise RuntimeError("auto-wake policy has expired")
        continuation_id = "continuation_%s" % uuid4().hex
        focus_digest = sha256(policy.focus.encode("utf-8")).hexdigest()
        profile_digest = self._profile_digest(profile)
        event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "continuation_id": continuation_id, "action": "enable",
                "scope": "next_sleep_epoch_read_only_research", "approval_event_id": approval_event_id,
                "unattended_policy_id": policy.policy_id, "focus_digest": focus_digest,
                "profile_digest": profile_digest, "max_auto_runs": 1,
                "expires_at": expiry.isoformat(), "issued_at": issued.isoformat(),
                "policy_version": "unattended_wake_continuation_policy_v1"},
            parent_event_ids=(approval_event_id,)))
        self._wake_continuation = {
            "continuation_id": continuation_id, "policy_event_id": event.event_id,
            "approval_event_id": approval_event_id, "auto_wake_policy_event_id": self._auto_wake_policy_event_id,
            "unattended_policy_id": policy.policy_id,
            "focus": policy.focus, "focus_digest": focus_digest, "profile_digest": profile_digest,
            "workspace_id": profile.workspace_id, "budget": profile.budget, "expires_at": expiry,
            "used": False, "reason": "armed", "instance_marker": self._instance_marker,
        }
        self._wake_continuation_last = {"armed": True, "used": False,
            "continuation_digest": sha256(continuation_id.encode("utf-8")).hexdigest(),
            "focus_digest": focus_digest, "profile_digest": profile_digest,
            "expires_at": expiry.isoformat(), "reason": "armed"}
        return self.wake_continuation_status()

    def _clear_wake_continuation(self, reason: str) -> None:
        if self._wake_continuation is not None:
            self._wake_continuation["reason"] = reason
            self._wake_continuation_last = {"armed": False, "used": bool(self._wake_continuation["used"]),
                "continuation_digest": sha256(self._wake_continuation["continuation_id"].encode("utf-8")).hexdigest(),
                "focus_digest": self._wake_continuation["focus_digest"],
                "profile_digest": self._wake_continuation["profile_digest"],
                "expires_at": self._wake_continuation["expires_at"].isoformat(), "reason": reason}
            self._wake_continuation = None

    def wake_continuation_status(self) -> Dict[str, Any]:
        capsule = self._wake_continuation
        if capsule is None:
            return dict(self._wake_continuation_last)
        return {"armed": not capsule["used"], "used": bool(capsule["used"]),
                "continuation_digest": sha256(capsule["continuation_id"].encode("utf-8")).hexdigest(),
                "focus_digest": capsule["focus_digest"], "profile_digest": capsule["profile_digest"],
                "expires_at": capsule["expires_at"].isoformat(), "reason": capsule["reason"]}

    def sleep_now(self) -> bool:
        if self.sleep_coordinator is None:
            raise RuntimeError("sleep coordination is not configured")
        return self._maybe_enter_sleep(force=True)

    def poll_sleep(self, usage_adapter: Any, allow_manual: bool = False) -> bool:
        """Foreground-only wake poll; callers invoke it at REPL boundaries."""
        if self.sleep_coordinator is None or self.sleep_coordinator.state != SleepState.SLEEPING:
            return False
        if not self.sleep_coordinator.to_payload().get("auto_wake_user_approved") and not allow_manual:
            return False
        if not self.sleep_coordinator.refresh_due() or usage_adapter is None:
            return False
        token, generation = self.sleep_coordinator.authority_token, self.sleep_coordinator.generation
        result = usage_adapter.refresh_controller(self.quota_controller) if self.quota_controller else usage_adapter.fetch()
        if getattr(result, "snapshot", None) is None or not getattr(result, "windows", ()):
            self.sleep_coordinator.check_and_wake((), self._sleep_now(), authoritative=False,
                                                  authority_token=token, expected_generation=generation)
            return False
        snapshot = result.snapshot
        evidence = self.quota_controller.authoritative_wake_evidence(
            now=snapshot.observed_at, required_source=QuotaSource.PROVIDER_USAGE,
            minimum_observed_at=self._managed_usage_observed_at) if self.quota_controller else None
        if evidence is None or not evidence.allow:
            self.sleep_coordinator.check_and_wake((), snapshot.observed_at, authoritative=False,
                                                  authority_token=token, expected_generation=generation)
            return False
        rolling = tuple(window for window in result.windows if window.kind == "rolling_5h")
        if (len(rolling) != 1
                or getattr(snapshot, "primary_window_kind", None) not in (None, "weekly", "rolling_5h")):
            self.sleep_coordinator.check_and_wake((), snapshot.observed_at, authoritative=False,
                                                  authority_token=token, expected_generation=generation)
            return False
        usage_event = self._append_provider_usage_evidence(snapshot, rolling[0])
        woke = self.sleep_coordinator.check_and_wake(
            rolling, snapshot.observed_at, authoritative=True,
            authority_token=token, expected_generation=generation)
        check = self._append_wake_check(usage_event, snapshot.observed_at)
        self._managed_usage_windows, self._managed_usage_observed_at = tuple(result.windows), snapshot.observed_at
        if not woke:
            return False
        ready = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.WAKE_READY, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", payload={"wake_ready_id": "ready_%s" % uuid4().hex,
                "wake_check_event_id": check.event_id, "epoch_id": self._sleep_epoch_id,
                "ready_at": self._sleep_now().isoformat(), "protocol_version": "sleep_wake_v2"},
            parent_event_ids=(check.event_id,)))
        mark_awake = getattr(self.sleep_coordinator, "mark_awake", None)
        # No post-wake work, including the deterministic local loop, is
        # permitted until the READY -> ACTIVE compare-and-swap succeeds.
        if not callable(mark_awake) or not mark_awake(
                authority_token=self.sleep_coordinator.authority_token,
                expected_generation=self.sleep_coordinator.generation):
            return False
        awake = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.AWAKE, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", payload={"awake_id": "awake_%s" % uuid4().hex,
                "wake_ready_event_id": ready.event_id, "epoch_id": self._sleep_epoch_id,
                "awakened_at": self._sleep_now().isoformat(), "protocol_version": "sleep_wake_v1"},
            parent_event_ids=(ready.event_id,)))
        self._run_wake_continuation(awake)
        if self._expedition is not None:
            self._expedition.wake(True, self._sleep_now())
        self._start_wake_loop(awake)
        # Any pre-sleep tool authority remains suspended; a wake creates no grant.
        self._pending_tool_plans = {}
        return True

    def _maybe_enter_sleep(self, force: bool = False) -> bool:
        """Fail terminally if a checkpoint cannot be recorded atomically."""
        try:
            return self._maybe_enter_sleep_impl(force)
        except BaseException:
            if self.sleep_coordinator is not None:
                self.sleep_coordinator.stop()
            if self._expedition is not None:
                self._expedition.stop("archive_store_failure")
            self.stop_research_autonomy("terminal")
            raise

    def _maybe_enter_sleep_impl(self, force: bool = False) -> bool:
        coordinator = self.sleep_coordinator
        if coordinator is None or coordinator.state in (SleepState.SLEEPING, SleepState.TERMINAL):
            return False
        windows, observed = self._managed_usage_windows, self._managed_usage_observed_at
        if not windows or observed is None:
            return False
        entered = coordinator.prepare_sleep(windows, observed, authoritative=True,
                                            user_auto_wake=coordinator.to_payload()["auto_wake_user_approved"])
        if not entered and force:
            # "now" is deliberately conservative: it refuses rather than
            # inventing a quota window or bypassing authoritative telemetry.
            return False
        if not entered:
            return False
        if self._expedition is not None:
            self._expedition.sleep("quota_sleep")
        self._close_loop_for_sleep()
        self.stop_research_autonomy("sleep")
        if self.tool_session is not None:
            self.tool_session.registry.suspend_after_restart()
        if self.quota_controller is not None:
            self.quota_controller.invalidate_reservations(ReservationInvalidationReason.SLEEP,
                                                          epoch=coordinator.epoch)
        self._pending_tool_plans = {}
        rolling = next(item for item in windows if getattr(item, "kind", None) == "rolling_5h")
        events = self.event_store.list(self.session_id)
        head = events[-1] if events else None
        self._sleep_epoch_id = "epoch_%d" % coordinator.epoch
        archive = build_public_archive(
            chain_head_sequence=head.sequence if head else 0,
            chain_head_hash=head.content_hash if head else "0" * 64,
            active_seed_ids=(item.seed_id for item in self.seed_store.list(self.session_id) if item.status.value == "active"),
            active_claim_ids=(item.claim_id for item in self.self_model.current_claims(self.session_id)),
            pending_public_event_ids=(), quota_source="provider_usage", quota_observed_at=observed,
            quota_reset_at=rolling.reset_at, quota_remaining=rolling.remaining, quota_total=rolling.total,
            schema_version=3)
        # The public checkpoint and its lifecycle edge are one SQLite commit.
        # A process interruption or validation/commit failure cannot leave a
        # durable archive that falsely appears to have entered sleep.
        with self.event_store.transaction():
            archive_event = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.SLEEP_ARCHIVE, source_kind=SourceKind.SYSTEM,
                source_ref="SleepWakeCoordinator", payload={"archive_id": "archive_%s" % uuid4().hex,
                    "epoch_id": self._sleep_epoch_id, "schema_version": 3,
                    "chain_head_sequence": archive.chain_head_sequence, "chain_head_hash": archive.chain_head_hash,
                    "approved_seed_ids": list(archive.active_seed_ids), "approved_claim_ids": list(archive.active_claim_ids),
                    "pending_task_ids": list(archive.pending_public_event_ids), "quota_source": "provider_usage",
                    "quota_observed_at": archive.quota_observed_at, "quota_reset_at": archive.quota_reset_at,
                    "quota_remaining": archive.quota_remaining, "quota_total": archive.quota_total,
                    "quota_window_kind": "rolling_5h", "archive_digest": archive.digest,
                    "event_count": len(events), "archive_version": "sleep_archive_v3"}),
                commit=False)
            sleep_payload = {"sleep_id": "sleep_%s" % uuid4().hex,
                "archive_event_id": archive_event.event_id, "epoch_id": self._sleep_epoch_id,
                "auto_wake_policy_event_id": self._auto_wake_policy_event_id,
                "slept_at": self._sleep_now().isoformat(), "protocol_version": "sleep_wake_v1"}
            if self._wake_continuation is not None:
                sleep_payload["continuation_policy_event_id"] = self._wake_continuation["policy_event_id"]
            sleep_event = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.SLEEP_ENTERED, source_kind=SourceKind.SYSTEM,
                source_ref="SleepWakeCoordinator", payload=sleep_payload,
                parent_event_ids=tuple(item for item in (
                    archive_event.event_id, self._auto_wake_policy_event_id,
                    (self._wake_continuation or {}).get("policy_event_id")) if item)),
                commit=False)
        self._sleep_archive_event_id, self._sleep_event_id = archive_event.event_id, sleep_event.event_id
        return True

    def _run_wake_continuation(self, awake: CognitiveEvent) -> None:
        """Consume a same-process capsule and make exactly one fresh run.

        It runs only after ``poll_sleep`` has installed fresh provider usage
        evidence and the sleep coordinator has atomically become ACTIVE.
        All original research grants were already suspended on sleep.
        """
        capsule = self._wake_continuation
        if capsule is None or capsule.get("used"):
            return
        now = self._sleep_now()
        if (capsule.get("instance_marker") != self._instance_marker or now >= capsule["expires_at"]
                or self.sleep_coordinator is None or self.sleep_coordinator.state != SleepState.ACTIVE
                or self._quota_denied() or capsule.get("auto_wake_policy_event_id") != self._auto_wake_policy_event_id):
            self._clear_wake_continuation("expired_or_gated")
            return
        # Consume before creating any mutable authority, so a duplicate wake
        # callback or a failure cannot retry or resume the old controller.
        capsule["used"] = True
        capsule["reason"] = "consumed"
        try:
            authority = self._wake_continuation_authority(capsule)
            profile = ResearchAutonomyProfile(capsule["workspace_id"], budget=capsule["budget"])
            # Create a new short-lived policy id and entirely fresh
            # profile/grants.  The immutable continuation event remains its
            # direct parent provenance; no old policy object or grant is
            # resumed, and no synthetic USER event is created.
            policy = UnattendedPolicy.user_issued(
                capsule["focus"], now + timedelta(seconds=capsule["budget"].ttl_seconds), now=now)
            self._start_unattended_research(profile, policy, authority.event_id,
                                            allow_wake_continuation=True,
                                            model_trigger_event_id=awake.event_id)
            self.run_unattended_research()
            status = "completed"
        except (RuntimeError, ValueError, PermissionError):
            profile = None
            policy = None
            status = "failed"
        self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.UNATTENDED_WAKE_RUN,
            source_kind=SourceKind.SYSTEM, source_ref="UnattendedResearchController", payload={
                "continuation_id": capsule["continuation_id"], "awake_event_id": awake.event_id,
                "profile_id": profile.profile_id if profile is not None else "unavailable",
                "authorization_policy_id": capsule["unattended_policy_id"],
                "runtime_policy_id": policy.policy_id if policy is not None else "unavailable",
                "run_index": 1, "status": status, "protocol_version": "unattended_wake_run_v1"},
            parent_event_ids=(awake.event_id, capsule["policy_event_id"])))
        self._clear_wake_continuation("completed" if status == "completed" else "failed")

    def _wake_continuation_authority(self, capsule: Dict[str, Any]) -> CognitiveEvent:
        """Validate the sole non-observation authority for a wake run.

        The process-local capsule is not authority by itself.  It must still
        name the exact USER continuation policy that was appended before the
        matching sleep epoch, and that policy must remain tied to the original
        USER observation.  This helper is private so ordinary start paths
        cannot turn arbitrary policy/model/system records into authority.
        """
        event = self.event_store.get(capsule.get("policy_event_id"))
        if (event is None or event.event_id != capsule.get("policy_event_id")
                or event.session_id != self.session_id
                or event.kind != EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY
                or event.source_kind != SourceKind.USER
                or datetime.fromisoformat(event.payload["expires_at"]) <= self._sleep_now()
                or event.payload.get("continuation_id") != capsule.get("continuation_id")
                or event.payload.get("approval_event_id") != capsule.get("approval_event_id")
                or event.payload.get("unattended_policy_id") != capsule.get("unattended_policy_id")
                or event.payload.get("focus_digest") != capsule.get("focus_digest")
                or event.payload.get("profile_digest") != capsule.get("profile_digest")
                or len(event.parent_event_ids) != 1):
            raise ValueError("wake continuation authority is invalid")
        approval = self.event_store.get(event.parent_event_ids[0])
        if (approval is None or approval.event_id != capsule.get("approval_event_id")
                or approval.session_id != self.session_id
                or approval.kind != EventKind.OBSERVATION
                or approval.source_kind != SourceKind.USER):
            raise ValueError("wake continuation authority lacks its original user approval")
        return event

    def plan_tools(self, task: str) -> Tuple[PreparedToolCall, ...]:
        """Ask K3 for one bounded tool plan, without granting or executing it.

        The returned calls are host-mapped through ``ToolSession``.  The user
        observation and completed model-invocation record are created first so
        every eventual execution has public, non-reasoning provenance.
        """
        self._maybe_enter_sleep()
        if self.sleep_coordinator is not None and self.sleep_coordinator.state == SleepState.SLEEPING:
            return ()
        if self.tool_session is None or self.runtime is None:
            raise RuntimeError("controlled tools are not configured")
        if not isinstance(task, str) or not task.strip() or len(task) > 12000:
            raise ValueError("task must be bounded non-empty text")
        observation = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user_agent_task",
            payload={"content": task.strip(), "channel": "agent_task"},
        ))
        run_id = "toolrun_%s" % uuid4().hex
        invocation_id = "inv_%s" % uuid4().hex
        started_at = time.monotonic()
        self._append_model_invocation(invocation_id + ".start", run_id, observation.event_id,
                                      "started", 0, "K3 tool planning request started.", (),
                                      role="tool_planning", context_scope="tool_planning")
        try:
            intents = KimiToolPlanner(self.runtime).plan(task.strip(), {
                "session_id": self.session_id,
                "tool_limit": self.tool_session.limits.max_tool_calls,
                "public_capability_summary": list(self.tool_session.grant_snapshots()),
            })
        except KimiCodeProviderError as error:
            elapsed = int((time.monotonic() - started_at) * 1000)
            timed_out = error.category == "provider_timeout"
            outcome = "timed_out" if timed_out else ("budget_exhausted" if self._quota_denied() else "failed")
            self._append_model_invocation(invocation_id + ".finish", run_id, observation.event_id,
                                          outcome, elapsed,
                                          ("K3 tool planning timed out; no tool was executed."
                                           if timed_out else "K3 tool planning unavailable; no tool was executed."), (),
                                          role="tool_planning", context_scope="tool_planning")
            return ()
        except Exception:
            elapsed = int((time.monotonic() - started_at) * 1000)
            self._append_model_invocation(invocation_id + ".finish", run_id, observation.event_id,
                                          "failed", elapsed,
                                          "K3 tool planning unavailable; no tool was executed.", (),
                                          role="tool_planning", context_scope="tool_planning")
            return ()
        elapsed = int((time.monotonic() - started_at) * 1000)
        invocation = self._append_model_invocation(
            invocation_id + ".finish", run_id, observation.event_id, "completed", elapsed,
            "K3 returned bounded tool intents; host authorization is still required.", (),
            role="tool_planning", context_scope="tool_planning")
        return tuple(self.tool_session.prepare(intent, run_id, invocation.event_id) for intent in intents)

    def start_unattended_research(self, profile: ResearchAutonomyProfile,
                                  policy: UnattendedPolicy,
                                  approval_event_id: str) -> Dict[str, Any]:
        """Activate a fresh, bounded public-web research run.

        ``approval_event_id`` must be a recorded USER observation supplied by
        the host/CLI.  The model supplies only typed proposals after this
        point; it cannot create, extend, or restore this profile.
        """
        return self._start_unattended_research(profile, policy, approval_event_id)

    def _start_unattended_research(self, profile: ResearchAutonomyProfile,
                                   policy: UnattendedPolicy, authority_event_id: str,
                                   allow_wake_continuation: bool = False,
                                   model_trigger_event_id: Optional[str] = None) -> Dict[str, Any]:
        """Internal start path with a narrowly scoped continuation exception."""
        if self.tool_session is None or self.runtime is None:
            raise RuntimeError("unattended research requires K3 and controlled tool backends")
        if not isinstance(profile, ResearchAutonomyProfile) or not isinstance(policy, UnattendedPolicy):
            raise ValueError("unattended research requires a profile and explicit user policy")
        authority = self.event_store.get(authority_event_id)
        normal_user_observation = (authority is not None and authority.session_id == self.session_id
                                   and authority.kind == EventKind.OBSERVATION
                                   and authority.source_kind == SourceKind.USER)
        continuation_policy = (allow_wake_continuation and authority is not None
                               and authority.session_id == self.session_id
                               and authority.kind == EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY
                               and authority.source_kind == SourceKind.USER)
        expedition_authorization = (authority is not None and authority.session_id == self.session_id
                                    and authority.kind == EventKind.EXPEDITION_AUTHORIZATION
                                    and authority.source_kind == SourceKind.USER)
        if not normal_user_observation and not continuation_policy and not expedition_authorization:
            raise ValueError("unattended research requires a recorded USER observation")
        if continuation_policy:
            trigger = self.event_store.get(model_trigger_event_id)
            if (trigger is None or trigger.session_id != self.session_id
                    or trigger.kind != EventKind.AWAKE):
                raise ValueError("wake continuation requires its exact awake model trigger")
        elif expedition_authorization:
            trigger = self.event_store.get(model_trigger_event_id)
            if (trigger is None or trigger.session_id != self.session_id
                    or trigger.kind != EventKind.OBSERVATION or trigger.source_kind != SourceKind.USER
                    or trigger.event_id != authority.payload.get("approval_event_id")):
                raise ValueError("expedition authorization requires its original user observation as model trigger")
        elif model_trigger_event_id is not None:
            raise ValueError("ordinary unattended research has no alternate model trigger")
        if self.sleep_coordinator is not None and self.sleep_coordinator.state != SleepState.ACTIVE:
            raise RuntimeError("unattended research cannot start while sleep coordination is inactive")
        if self._quota_denied():
            raise RuntimeError("unattended research cannot start while K3 quota is unavailable")
        self.stop_research_autonomy("replaced")
        self.tool_session.register_research_autonomy(profile, authority_event_id)
        self._unattended_allowed_tools = tuple(capability.value.replace(".", "_")
                                              for capability in profile.capabilities)
        # The continuation policy is authority for grants only.  Its planning
        # invocation is triggered by the exact observed AWAKE transition;
        # ordinary starts retain the original USER observation as both.
        self._unattended_policy_event_id = (
            model_trigger_event_id if (continuation_policy or expedition_authorization) else authority_event_id)
        self._unattended_prepared, self._unattended_proposals = {}, {}
        self._unattended_findings = []
        controller = UnattendedResearchController(
            self._unattended_plan, self._unattended_prepare, self._unattended_execute,
            sleeping=lambda: self.sleep_coordinator is not None and self.sleep_coordinator.state != SleepState.ACTIVE,
            quota_available=lambda: not self._quota_denied(),
        )
        # The profile narrows, but can never widen, the foreground controller's
        # absolute resource caps.
        controller.MAX_CALLS = min(controller.MAX_CALLS, profile.budget.max_tool_calls)
        controller.MAX_WALL_SECONDS = min(controller.MAX_WALL_SECONDS, profile.budget.max_wall_ms // 1000)
        controller.MAX_OUTPUT_BYTES = min(controller.MAX_OUTPUT_BYTES, profile.budget.max_total_bytes)
        controller.start(policy)
        self._unattended = controller
        return self.unattended_status()

    def start_expedition(self, goal: str, host_seed: str, authorization_seconds: int,
                         slice_seconds: int, max_calls_per_slice: int,
                         authorization_event_id: str, learning_mode: str = "off",
                         learner_spec: Optional[FrontierLearningSpec] = None,
                         experiment_policy: Optional[ExperimentRegistryPolicy] = None) -> Dict[str, Any]:
        """Arm a foreground-only public-web expedition from explicit USER input."""
        authorization = self.event_store.get(authorization_event_id)
        if (authorization is None or authorization.session_id != self.session_id
                or authorization.kind != EventKind.EXPEDITION_AUTHORIZATION):
            raise ValueError("expedition requires an unconsumed USER authorization")
        if self.tool_session is None:
            raise RuntimeError("expedition requires a controlled tool session")
        if self.runtime is None:
            raise RuntimeError("expedition requires an explicitly configured K3 runtime")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
            raise ValueError("expedition goal must be bounded non-empty text")
        if (not isinstance(max_calls_per_slice, int) or isinstance(max_calls_per_slice, bool)
                or not 1 <= max_calls_per_slice <= 100):
            raise ValueError("max_calls_per_slice must be between 1 and 100")
        required_slices = max(1, int(math.ceil(float(authorization_seconds) / slice_seconds)))
        if required_slices > 360:
            raise ValueError("expedition duration cannot be covered by at most 360 slices")
        maximum_slices = min(360, self.runtime.settings.max_calls // (max_calls_per_slice + 1))
        if required_slices > maximum_slices:
            raise ValueError("expedition configuration exceeds the bounded K3 call allowance")
        if learning_mode not in ("off", "shadow", "active"):
            raise ValueError("expedition learning_mode must be off, shadow, or active")
        spec = learner_spec or FrontierLearningSpec()
        if not isinstance(spec, FrontierLearningSpec):
            raise ValueError("learner_spec must be a FrontierLearningSpec")
        spec_digest = self.frontier_learning_spec_digest(spec)
        payload = authorization.payload
        if (payload["goal_digest"] != sha256(goal.encode("utf-8")).hexdigest()
                or payload["host_seed_digest"] != sha256(host_seed.encode("utf-8")).hexdigest()
                or payload["authorization_seconds"] != authorization_seconds
                or payload["slice_seconds"] != slice_seconds
                or payload["max_calls_per_slice"] != max_calls_per_slice
                or payload["profile"] != "public_web_only_v1"):
            raise ValueError("expedition approval does not bind the exact goal and budgets")
        if payload.get("version") == "expedition_authorization_v1":
            if learning_mode != "off":
                raise ValueError("frontier learning requires a v2 USER authorization")
            if experiment_policy is not None:
                raise ValueError("offline experiments require a v3 USER authorization")
        elif payload.get("version") == "expedition_authorization_v2":
            if experiment_policy is not None:
                raise ValueError("offline experiments require a v3 USER authorization")
            if (payload.get("learning_mode") != learning_mode
                    or payload.get("learner_spec_digest") != spec_digest):
                raise ValueError("expedition approval does not bind the learning mode and learner spec")
        elif (payload.get("version") != "expedition_authorization_v3"
              or payload.get("learning_mode") != learning_mode
              or payload.get("learner_spec_digest") != spec_digest):
            raise ValueError("expedition approval does not bind the learning mode and learner spec")
        if payload.get("version") == "expedition_authorization_v3":
            if experiment_policy is None or learning_mode not in ("active", "shadow"):
                raise ValueError("expedition v3 requires an explicit active or shadow experiment policy")
            if (payload.get("experiment_registry_digest") != experiment_policy.registry_digest
                    or payload.get("experiment_kinds") != [item.value for item in experiment_policy.approved_kinds]
                    or payload.get("max_experiments") != experiment_policy.max_experiments
                    or payload.get("experiment_max_trials") != experiment_policy.max_trials
                    or payload.get("experiment_max_steps") != experiment_policy.max_steps
                    or payload.get("experiment_max_wall_ms") != experiment_policy.max_wall_ms):
                raise ValueError("expedition approval does not bind the exact experiment registry")
        now = self._sleep_now()
        if now >= datetime.fromisoformat(payload["expires_at"]):
            raise RuntimeError("expedition authorization has expired")
        run_id = "expedition_%s" % uuid4().hex
        with self.event_store.transaction():
            consumption = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
                source_kind=SourceKind.SYSTEM, source_ref="ExpeditionRuntime", payload={
                    "consumption_id": "consume_%s" % uuid4().hex,
                    "authorization_id": payload["authorization_id"], "consumed_at": now.isoformat(),
                    "run_id": run_id, "version": "expedition_authorization_consumed_v1"},
                parent_event_ids=(authorization.event_id,)), commit=False)
        config = ExpeditionConfig(total_authorization_seconds=authorization_seconds,
                                  max_slice_seconds=slice_seconds,
                                  # Slices often close early.  Use every slice
                                  # that remains within both the host's 360-slice
                                  # cap and the provider call budget, while
                                  # still proving the wall-clock window is
                                  # coverable at the configured maximum length.
                                  max_slices=maximum_slices,
                                  authorization_version=payload["version"],
                                  allowed_experiment_kinds=(tuple(payload.get("experiment_kinds", ()))
                                                            if experiment_policy is not None else ()),
                                  max_experiments=(experiment_policy.max_experiments
                                                   if experiment_policy is not None else 0),
                                  experiment_registry_digest=(experiment_policy.registry_digest
                                                              if experiment_policy is not None else None),
                                  learner_spec_digest=(spec_digest if learning_mode != "off" else None))
        state_id = "expedition_%s" % sha256(goal.encode("utf-8")).hexdigest()[:24]
        learner = (FrontierLearner(host_seed, spec) if learning_mode != "off" else None)
        ranker = (_FrontierLearnerRanker(learner, state_id) if learner is not None else None)
        self._expedition = ExpeditionScheduler(host_seed, config,
                                               datetime.fromisoformat(payload["issued_at"]),
                                               ranker=ranker,
                                               ranker_active=(learning_mode == "active"))
        self._expedition_goal_digest = sha256(goal.encode("utf-8")).hexdigest()
        self._expedition_approval_event_id = authorization.event_id
        self._expedition_model_trigger_event_id = payload["approval_event_id"]
        self._expedition_expires_at = datetime.fromisoformat(payload["expires_at"])
        self._expedition_refresh_gate = ForegroundRefreshGate(ForegroundRefreshPolicy())
        self._expedition_max_calls = max_calls_per_slice
        self._expedition_goal = goal
        self._expedition_current_persona = None
        self._expedition_domains = set()
        self._expedition_useful_findings = 0
        self._expedition_learning_mode = learning_mode
        self._expedition_learner = learner
        self._expedition_learner_spec = spec if learner is not None else None
        self._expedition_learner_spec_digest = spec_digest if learner is not None else None
        self._expedition_frontier_state_id = state_id if learner is not None else None
        self._expedition_consumption_event_id = consumption.event_id
        self._expedition_active_transition = None
        self._expedition_experiment_policy = experiment_policy
        self._expedition_experiment_harness = (ExperimentHarness(host_seed)
                                               if experiment_policy is not None else None)
        self._expedition_experiment_completed = 0
        self._expedition_experiment_last = {}
        self._expedition_host_seed = host_seed
        self._expedition_rewarded_strategy_arms = set()
        self._expedition_post_reward_selection_count = 0
        self._expedition_last_post_reward_selection_reason = "none"
        self._expedition_duplicate_experiment_result_count = 0
        self._expedition_last_strategy_arm_id = None
        self._expedition_last_strategy_arm_version = None
        guidance = self._expedition_seed_guidance()
        self._expedition_seed_context_event_id = self._record_expedition_seed_context(
            run_id, consumption.event_id, guidance).event_id
        if guidance:
            self._expedition.set_seed_guidance(SeedGuidanceSnapshot(
                seed_ids=tuple(item.seed_id for item in guidance),
                authority_event_ids=tuple(item.current_authority_event_id for item in guidance),
                digest=sha256(json.dumps([
                    {"seed_id": item.seed_id,
                     "authority_event_id": item.current_authority_event_id,
                     "snapshot_digest": item.snapshot_digest,
                     "priority_band": item.priority_band}
                    for item in guidance], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()))
        return self.expedition_status()

    @staticmethod
    def expedition_authorization_content(goal: str, host_seed: str,
                                         authorization_seconds: int,
                                         slice_seconds: int,
                                         max_calls_per_slice: int,
                                         learning_mode: str = "off",
                                         learner_spec: Optional[FrontierLearningSpec] = None,
                                         experiment_policy: Optional[ExperimentRegistryPolicy] = None) -> str:
        """Canonical public USER contract without storing the goal or seed."""
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
            raise ValueError("expedition goal must be bounded non-empty text")
        if not isinstance(host_seed, str) or not host_seed.strip() or len(host_seed) > 512:
            raise ValueError("expedition host seed must be bounded non-empty text")
        if learning_mode not in ("off", "shadow", "active"):
            raise ValueError("expedition learning_mode must be off, shadow, or active")
        contract = {
            "authorization_seconds": authorization_seconds,
            "goal_digest": sha256(goal.encode("utf-8")).hexdigest(),
            "host_seed_digest": sha256(host_seed.encode("utf-8")).hexdigest(),
            "max_calls_per_slice": max_calls_per_slice,
            "profile": "public_web_only_v1",
            "slice_seconds": slice_seconds,
            "version": "expedition_authorization_v1",
        }
        if learning_mode != "off":
            spec = learner_spec or FrontierLearningSpec()
            if not isinstance(spec, FrontierLearningSpec):
                raise ValueError("learner_spec must be a FrontierLearningSpec")
            contract.update({"learning_mode": learning_mode,
                             "learner_spec_digest": StrangeloopAgent.frontier_learning_spec_digest(spec),
                             "version": "expedition_authorization_v2"})
        if experiment_policy is not None:
            if learning_mode not in ("active", "shadow"):
                raise ValueError("offline experiments require active or shadow frontier learning")
            contract.update({"experiment_kinds": [item.value for item in experiment_policy.approved_kinds],
                             "experiment_registry_digest": experiment_policy.registry_digest,
                             "max_experiments": experiment_policy.max_experiments,
                             "experiment_max_trials": experiment_policy.max_trials,
                             "experiment_max_steps": experiment_policy.max_steps,
                             "experiment_max_wall_ms": experiment_policy.max_wall_ms,
                             "version": "expedition_authorization_v3"})
        return json.dumps(contract, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def frontier_learning_spec_digest(spec: FrontierLearningSpec) -> str:
        if not isinstance(spec, FrontierLearningSpec):
            raise ValueError("learner_spec must be a FrontierLearningSpec")
        return spec.spec_digest

    def expedition_status(self) -> Dict[str, Any]:
        if self._expedition is None:
            return {"state": "not_configured"}
        raw = self._expedition.public_snapshot()
        counts = raw.get("frontier_counts", {})
        quota = self.quota_status()
        task_terminal_coverage = (float(counts.get("complete", 0)) /
                                  max(1, sum(counts.values())))
        return {"state": raw["state"], "seed_digest": raw["seed_digest"],
                "slice": raw["slice_number"], "authorization_seconds": raw["authorization_seconds"],
                "max_slice_seconds": raw["max_slice_seconds"], "goal_digest": self._expedition_goal_digest,
                # The expedition goal is an explicit user-supplied research
                # contract.  It is intentionally distinct from model output
                # and is safe for the localhost monitor only when accompanied
                # by this fixed visibility marker.
                "public_goal": self._expedition_goal,
                "goal_visibility": "user_public_research_goal_v1",
                "persona": self._expedition_current_persona,
                # Completion coverage is scheduler-terminal bookkeeping, not
                # a quality/progress claim about an experiment's findings.
                "frontier_coverage": task_terminal_coverage,
                "task_terminal_coverage": task_terminal_coverage,
                "domain_count": len(self._expedition_domains),
                "useful_findings": self._expedition_useful_findings,
                "planner_failures": sum(item.get("planner_failures", 0) for item in raw.get("tasks", [])),
                "empty_searches": sum(item.get("empty_searches", 0) for item in raw.get("tasks", [])),
                "quota_retry": raw.get("quota_retry"),
                "selection": {"last_reason": raw.get("selection", {}).get("last_reason")},
                "quota": quota.get("reason"), "sleep": self.sleep_status().get("state"),
                "stop_reason": raw.get("stop_reason"),
                "frontier_learning": {"mode": self._expedition_learning_mode,
                                      "spec_digest": self._expedition_learner_spec_digest,
                                      "state": self._expedition_frontier_state_id,
                                      "transition_active": self._expedition_active_transition is not None,
                                      "strategy_arm_id": self._expedition_last_strategy_arm_id,
                                      "strategy_arm_version": self._expedition_last_strategy_arm_version,
                                      "post_reward_selection_count": self._expedition_post_reward_selection_count,
                                      "last_post_reward_selection_reason": self._expedition_last_post_reward_selection_reason},
                "experiment": {"mode": self._expedition_learning_mode,
                               "approved_kinds": ([] if self._expedition_experiment_policy is None else
                                                  [item.value for item in self._expedition_experiment_policy.approved_kinds]),
                               "completed_count": self._expedition_experiment_completed,
                               "duplicate_experiment_result_count": self._expedition_duplicate_experiment_result_count,
                               **self._expedition_experiment_last}}

    def _frontier_active(self) -> bool:
        return (self._expedition_learning_mode == "active"
                and self._expedition is not None
                and self._expedition_learner is not None
                and self._expedition_learner_spec is not None
                and self._expedition_learner_spec_digest is not None
                and self._expedition_frontier_state_id is not None
                and self._expedition_consumption_event_id is not None
                and self._expedition.state.value == "active")

    @staticmethod
    def _frontier_channels() -> list[str]:
        return ["functional_continuity", "bounded_curiosity", "operational_integrity",
                "epistemic_progress", "user_alignment"]

    def _frontier_fail_closed(self, reason: str) -> None:
        """Do not leave an unrecorded selection with mutable learner state."""
        self._expedition_active_transition = None
        if self._expedition is not None:
            self._expedition.stop(reason)
        self.stop_research_autonomy("terminal")

    def _frontier_begin_transition(self, decision: Any) -> None:
        """Persist an active v2 choice before a tool can run under it."""
        if not self._frontier_active():
            self._expedition_active_transition = None
            return
        assert self._expedition is not None and self._expedition_learner is not None
        assert self._expedition_learner_spec is not None
        assert self._expedition_learner_spec_digest is not None
        assert self._expedition_frontier_state_id is not None
        assert self._expedition_consumption_event_id is not None
        transition_id = "frontiertransition_%s" % uuid4().hex
        candidate_snapshot = self._expedition.candidate_snapshot()
        candidate_digest = sha256(json.dumps(candidate_snapshot, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
        ranking_id, estimate_id = ("frontierranking_%s" % uuid4().hex,
                                   "frontierestimate_%s" % uuid4().hex)
        channels = self._frontier_channels()
        strategy_arm_id = decision.task.strategy_arm_id
        strategy_arm_version = FRONTIER_STRATEGY_ARM_VERSION
        self._expedition_last_strategy_arm_id = strategy_arm_id
        self._expedition_last_strategy_arm_version = strategy_arm_version
        if self._expedition_rewarded_strategy_arms:
            self._expedition_post_reward_selection_count += 1
            self._expedition_last_post_reward_selection_reason = (
                "strategy_reward_preferred" if strategy_arm_id in self._expedition_rewarded_strategy_arms
                else "strategy_ranked")
        prior = list(self._expedition_learner.values_for(
            self._expedition_frontier_state_id, strategy_arm_id))
        try:
            with self.event_store.transaction():
                ranking_parents = (self._expedition_consumption_event_id,)
                # The optional context is recomputed immediately before this
                # choice.  Binding it makes a later replay able to distinguish
                # an unguided ranking from one whose already-eligible tasks
                # were softly reordered by current, non-authoritative seed
                # guidance.  It never widens the authorization parent.
                if self._expedition_seed_context_event_id is not None:
                    ranking_parents += (self._expedition_seed_context_event_id,)
                ranking = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.FRONTIER_RANKING_DECISION,
                    source_kind=SourceKind.SYSTEM, source_ref="FrontierLearningHost",
                    parent_event_ids=ranking_parents, payload={
                        "ranking_id": ranking_id, "transition_id": transition_id,
                        "frontier_task_id": decision.task.task_id, "candidate_digest": candidate_digest,
                        "strategy_arm_id": strategy_arm_id, "strategy_arm_version": strategy_arm_version,
                        "learner_spec_digest": self._expedition_learner_spec_digest,
                        "channel_order": channels, "scope": "frontier_ranking_only",
                        "ranking_version": "frontier_ranking_v2"}), commit=False)
                estimate = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE,
                    source_kind=SourceKind.SYSTEM, source_ref="FrontierLearningHost",
                    parent_event_ids=(ranking.event_id,), payload={
                        "estimate_id": estimate_id, "transition_id": transition_id,
                        "ranking_event_id": ranking.event_id,
                        "strategy_arm_id": strategy_arm_id, "strategy_arm_version": strategy_arm_version,
                        "learner_spec_digest": self._expedition_learner_spec_digest,
                        "channel_order": channels, "value_vector": prior,
                        "alpha": self._expedition_learner_spec.alpha,
                        "gamma": self._expedition_learner_spec.gamma,
                        "clip": self._expedition_learner_spec.clip,
                        "scope": "frontier_ranking_only", "formula_version": "frontier_td0_vector_v2"},
                    ), commit=False)
        except BaseException:
            self._frontier_fail_closed("frontier_store_failure")
            raise
        self._expedition_active_transition = {"transition_id": transition_id,
                                               "ranking_event_id": ranking.event_id,
                                               "value_event_id": estimate.event_id,
                                               "task_id": decision.task.task_id,
                                               "strategy_arm_id": strategy_arm_id,
                                               "strategy_arm_version": strategy_arm_version}

    def _frontier_next_arms(self) -> Tuple[str, ...]:
        if self._expedition is None:
            return ()
        snapshot = self._expedition.public_snapshot()
        arms = sorted(str(item["strategy_arm_id"]) for item in snapshot.get("tasks", ())
                      if item.get("status") in ("pending", "deferred"))
        return tuple(dict.fromkeys(arms))

    def _frontier_record_document(self, report: Dict[str, Any]) -> None:
        """Persist attributable web evidence, deliberately without a reward.

        The host selects the receipt and fixed reward projection.  K3 output,
        URLs, queries, raw pages, quota, and lifecycle controls never enter the
        learner or frontier event payloads.
        """
        if not self._frontier_active() or self._expedition_active_transition is None:
            return
        assert self._expedition_learner is not None and self._expedition_learner_spec is not None
        assert self._expedition_learner_spec_digest is not None and self._expedition_frontier_state_id is not None
        direct = [item for item in report.get("findings", ()) if isinstance(item, dict)
                  and item.get("status") == "succeeded"
                  and item.get("tool") in ("web.fetch", "browser.read")]
        if not direct:
            return
        finding = next((item for item in direct if isinstance(item.get("result_event_id"), str)), None)
        if finding is None:
            return
        source = self.event_store.get(finding["result_event_id"])
        if source is None or source.kind != EventKind.TOOL_RESULT or source.source_kind != SourceKind.TOOL:
            return
        result_digest = source.payload.get("result_digest")
        if not isinstance(result_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", result_digest):
            return
        transition = dict(self._expedition_active_transition)
        evidence_id = "frontierevidence_%s" % uuid4().hex
        evidence = FrontierEvidence(
            event_id=evidence_id, state_id=self._expedition_frontier_state_id,
            arm_id=transition["strategy_arm_id"], kind=EvidenceKind.DOCUMENT,
            source=EvidenceSource.TOOL, provenance_id=source.event_id,
            content_digest=result_digest,
            # Fixed host projection from controlled direct-document success;
            # user alignment is intentionally only a user-feedback channel.
            bounded_curiosity=.50, operational_integrity=.25, epistemic_progress=.75)
        try:
            evidence_digest = sha256(json.dumps({"kind": evidence.kind.value,
                "source": evidence.source.value, "content_digest": result_digest},
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            with self.event_store.transaction():
                evidence_event = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.FRONTIER_EVIDENCE_OBSERVATION,
                    # The source is a direct, controlled TOOL_RESULT; this is
                    # not an inference or a model-generated receipt.
                    source_kind=SourceKind.TOOL, source_ref="FrontierLearningHost",
                    parent_event_ids=(transition["ranking_event_id"], source.event_id), payload={
                        "evidence_id": evidence_id, "transition_id": transition["transition_id"],
                        "ranking_event_id": transition["ranking_event_id"], "source_event_id": source.event_id,
                        "evidence_digest": evidence_digest, "evidence_kind": "validated_external_evidence",
                        "strategy_arm_id": transition["strategy_arm_id"],
                        "strategy_arm_version": transition["strategy_arm_version"],
                        "learner_spec_digest": self._expedition_learner_spec_digest,
                        "scope": "frontier_ranking_only", "evidence_version": "frontier_evidence_v2"},
                    ), commit=False)
        except BaseException:
            self._frontier_fail_closed("frontier_store_failure")
            raise

    def _run_frontier_experiment(self, decision: Any) -> Dict[str, Any]:
        """Run one host-only preregistered fixture experiment, never a tool/K3 call."""
        if (self._expedition_experiment_policy is None or self._expedition_experiment_harness is None
                or self._expedition_consumption_event_id is None or self._expedition_learner_spec_digest is None):
            return {}
        authorization = self.event_store.get(self._expedition_approval_event_id)
        if (authorization is None or authorization.payload.get("version") != "expedition_authorization_v3"
                or self._sleep_now() >= self._expedition_expires_at):
            self._frontier_fail_closed("experiment_authorization_invalid")
            return {"last_status": "invalid", "reward_qualified": False}
        kind_text = getattr(decision.task, "experiment_kind", None)
        if not isinstance(kind_text, str):
            return {}
        try:
            kind = ExperimentKind(kind_text)
        except ValueError:
            self._frontier_fail_closed("experiment_kind_invalid")
            return {"last_status": "invalid", "reward_qualified": False}
        if kind not in self._expedition_experiment_policy.approved_kinds:
            self._frontier_fail_closed("experiment_kind_unauthorized")
            return {"last_status": "invalid", "reward_qualified": False}
        ident = "experiment_%s" % uuid4().hex
        host_digest = authorization.payload["host_seed_digest"]
        digest = lambda label: sha256((label + "\0" + decision.task.task_id + "\0" +
                                       authorization.payload["experiment_registry_digest"]).encode("utf-8")).hexdigest()
        budget = ExperimentBudget(self._expedition_experiment_policy.max_trials,
                                  self._expedition_experiment_policy.max_steps,
                                  self._expedition_experiment_policy.max_wall_ms)
        spec = ExperimentSpec(ident, kind, digest("hypothesis"), digest("baseline"), digest("treatment"),
                              digest("input"), host_digest, budget)
        authority_digest = sha256(authorization.payload["authorization_id"].encode("utf-8")).hexdigest()
        expiry_monotonic = time.monotonic() + max(0.0, (self._expedition_expires_at - self._sleep_now()).total_seconds())
        authority = self._expedition_experiment_harness.issue_authority(
            self._expedition_experiment_policy, authority_digest, expiry_monotonic)
        plan_id, execution_id = "experimentplan_%s" % uuid4().hex, "experimentexec_%s" % uuid4().hex
        transition = self._expedition_active_transition
        # Shadow has audit-only experiment records; no learner state or reward chain.
        try:
            with self.event_store.transaction():
                plan = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.EXPERIMENT_PLAN_LOCKED,
                    source_kind=SourceKind.SYSTEM, source_ref="ExperimentCoordinator",
                    parent_event_ids=(self._expedition_consumption_event_id,), payload={
                        "experiment_id": ident, "authorization_id": authorization.payload["authorization_id"],
                        "learner_spec_digest": self._expedition_learner_spec_digest,
                        "experiment_registry_digest": self._expedition_experiment_policy.registry_digest,
                        "experiment_instance_digest": spec.digest, "experiment_kind": kind.value,
                        "hypothesis_digest": spec.hypothesis_digest, "baseline_digest": spec.baseline_digest,
                        "treatment_digest": spec.treatment_digest, "input_digest": spec.input_digest,
                        "evaluator_digest": digest("independent_verifier"), "host_seed_digest": spec.host_seed_digest,
                        "budget_digest": sha256(json.dumps({"trials": budget.max_trials, "steps": budget.max_steps,
                            "wall": budget.max_wall_ms}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                        "max_trials": budget.max_trials, "max_steps": budget.max_steps,
                        "max_wall_ms": budget.max_wall_ms, "no_external_side_effects": True,
                        "claim_scope": "offline_simulation", "template_version": "experiment_template_v1"},
                    ), commit=False)
                execution = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.EXPERIMENT_EXECUTION_STARTED,
                    source_kind=SourceKind.SYSTEM, source_ref="ExperimentHarness", parent_event_ids=(plan.event_id,),
                    payload={"execution_id": execution_id, "experiment_id": ident, "plan_event_id": plan.event_id,
                        "experiment_registry_digest": self._expedition_experiment_policy.registry_digest,
                        "experiment_instance_digest": spec.digest, "execution_nonce": "nonce_%s" % uuid4().hex,
                        "started_at": self._sleep_now().isoformat(), "max_trials": budget.max_trials,
                        "max_steps": budget.max_steps, "max_wall_ms": budget.max_wall_ms,
                        "execution_version": "experiment_execution_v1"}), commit=False)
            first = self._expedition_experiment_harness.run(spec, authority)
            # A fresh harness has no shared reservation state; equality is an
            # independent deterministic reproduction check, not model scoring.
            if self._expedition_host_seed is None:
                raise RuntimeError("experiment host seed is unavailable")
            verifier_harness = ExperimentHarness(self._expedition_host_seed)
            verifier_authority = verifier_harness.issue_authority(self._expedition_experiment_policy,
                                                                    authority_digest, expiry_monotonic)
            second = verifier_harness.run(spec, verifier_authority)
            reproducible = first == second and first.status is ExperimentStatus.PASSED
            status = "supported" if reproducible else ("inconclusive" if first.status is ExperimentStatus.BUDGET_EXHAUSTED else "invalid")
            metric_values = [value for _, value in first.metrics]
            effect_values = [0.0 for _ in metric_values]
            # Classify against only records that preceded this occurrence.
            # The result is then appended before any evidence/reward chain.
            is_novel_result = not any(
                event.kind == EventKind.EXPERIMENT_RESULT
                and event.payload.get("result_digest") == first.result_digest
                for event in self.event_store.list(self.session_id))
            with self.event_store.transaction():
                result = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.EXPERIMENT_RESULT,
                    source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="IndependentExperimentVerifier",
                    parent_event_ids=(execution.event_id,), payload={
                        "result_id": "experimentresult_%s" % uuid4().hex, "execution_id": execution_id,
                        "experiment_id": ident, "plan_event_id": plan.event_id,
                        "authorization_id": authorization.payload["authorization_id"],
                        "learner_spec_digest": self._expedition_learner_spec_digest,
                        "experiment_registry_digest": self._expedition_experiment_policy.registry_digest,
                        "experiment_instance_digest": spec.digest, "status": status,
                        "complete_run_set": reproducible, "baseline_control_valid": reproducible,
                        "reproducible": reproducible,
                        "metric_digest": sha256(json.dumps(metric_values, separators=(",", ":")).encode("utf-8")).hexdigest(),
                        "metric_values": metric_values, "effect_values": effect_values,
                        "result_digest": first.result_digest, "reproduction_digest": first.reproduction_digest,
                        "invariant_codes": list(first.invariant_codes), "claim_scope": "offline_simulation",
                        "result_version": "experiment_result_v1"}), commit=False)
            qualified = (reproducible and is_novel_result and transition is not None
                         and self._frontier_active())
            if reproducible and not is_novel_result:
                self._expedition_duplicate_experiment_result_count += 1
            if qualified:
                reward_vector = frontier_experiment_reward_vector(kind.value, status, is_novel_result)
                evidence = FrontierEvidence(event_id="frontierevidence_%s" % uuid4().hex,
                    state_id=self._expedition_frontier_state_id, arm_id=transition["strategy_arm_id"],
                    kind=EvidenceKind.REPRODUCIBLE_EXPERIMENT_RESULT, source=EvidenceSource.EXTERNAL_VERIFIER,
                    provenance_id=result.event_id, result_digest=first.result_digest,
                    preregistration_id=plan.event_id, control_id=execution.event_id,
                    reproducibility_id=first.reproduction_digest,
                    functional_continuity=reward_vector[0], bounded_curiosity=reward_vector[1],
                    operational_integrity=reward_vector[2], epistemic_progress=reward_vector[3],
                    user_alignment=reward_vector[4])
                self._frontier_record_verified_evidence(evidence, result.event_id, transition)
            self._expedition_experiment_completed += 1
            self._expedition_experiment_last = {"last_kind": kind.value, "last_status": status,
                "reproducible": reproducible, "control_valid": reproducible,
                "result_digest": first.result_digest, "is_novel_result": is_novel_result,
                "reward_qualified": qualified}
        except BaseException:
            self._frontier_fail_closed("experiment_store_or_verifier_failure")
            raise
        return dict(self._expedition_experiment_last)

    def _frontier_record_verified_evidence(self, evidence: FrontierEvidence, source_event_id: str,
                                           transition: Dict[str, str]) -> None:
        """Commit verified experiment evidence, reward and TD update atomically."""
        assert self._expedition_learner is not None and self._expedition_learner_spec is not None
        assert self._expedition_learner_spec_digest is not None and self._expedition_frontier_state_id is not None
        checkpoint = self._expedition_learner.snapshot()
        record = self._expedition_learner.observe(evidence, self._expedition_frontier_state_id,
            self._frontier_next_arms() or (transition["strategy_arm_id"],))
        if record is None:
            self._expedition_learner.restore(checkpoint)
            return
        channels, reward = self._frontier_channels(), list(record.reward)
        digest = sha256(json.dumps({"result": evidence.result_digest,
            "reproduction": evidence.reproducibility_id}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        try:
            with self.event_store.transaction():
                receipt = self.event_store.append(CognitiveEvent(session_id=self.session_id,
                    kind=EventKind.FRONTIER_EVIDENCE_OBSERVATION, source_kind=SourceKind.EXTERNAL_VERIFIER,
                    source_ref="FrontierLearningHost", parent_event_ids=(transition["ranking_event_id"], source_event_id),
                    payload={"evidence_id": evidence.event_id, "transition_id": transition["transition_id"],
                        "ranking_event_id": transition["ranking_event_id"], "source_event_id": source_event_id,
                        "evidence_digest": digest, "evidence_kind": "validated_external_evidence",
                        "strategy_arm_id": transition["strategy_arm_id"],
                        "strategy_arm_version": transition["strategy_arm_version"],
                        "learner_spec_digest": self._expedition_learner_spec_digest,
                        "scope": "frontier_ranking_only", "evidence_version": "frontier_evidence_v2"}), commit=False)
                reward_event = self.event_store.append(CognitiveEvent(session_id=self.session_id,
                    kind=EventKind.FRONTIER_VECTOR_REWARD, source_kind=SourceKind.SYSTEM,
                    source_ref="FrontierLearningHost", parent_event_ids=(receipt.event_id, transition["value_event_id"]),
                    payload={"reward_id": "frontierreward_%s" % uuid4().hex,
                        "transition_id": transition["transition_id"], "ranking_event_id": transition["ranking_event_id"],
                        "evidence_event_id": receipt.event_id, "value_event_id": transition["value_event_id"],
                        "strategy_arm_id": transition["strategy_arm_id"],
                        "strategy_arm_version": transition["strategy_arm_version"],
                        "learner_spec_digest": self._expedition_learner_spec_digest, "channel_order": channels,
                        "reward_vector": reward, "scope": "frontier_ranking_only", "reward_version": "frontier_reward_v2"}), commit=False)
                raw = [r + self._expedition_learner_spec.gamma * n - p
                       for r, n, p in zip(record.reward, record.next_value, record.prior)]
                clipped = [max(-self._expedition_learner_spec.clip, min(self._expedition_learner_spec.clip, value)) for value in raw]
                self.event_store.append(CognitiveEvent(session_id=self.session_id, kind=EventKind.FRONTIER_TD_UPDATE,
                    source_kind=SourceKind.SYSTEM, source_ref="FrontierLearningHost",
                    parent_event_ids=(reward_event.event_id, transition["value_event_id"]), payload={
                        "update_id": "frontierupdate_%s" % uuid4().hex, "transition_id": transition["transition_id"],
                        "ranking_event_id": transition["ranking_event_id"], "reward_event_id": reward_event.event_id,
                        "prior_value_event_id": transition["value_event_id"],
                        "strategy_arm_id": transition["strategy_arm_id"],
                        "strategy_arm_version": transition["strategy_arm_version"],
                        "learner_spec_digest": self._expedition_learner_spec_digest, "channel_order": channels,
                        "reward_vector": reward, "prior_value_vector": list(record.prior),
                        "next_value_vector": list(record.next_value), "raw_delta_vector": raw,
                        "clipped_delta_vector": clipped, "updated_value_vector": list(record.updated),
                        "alpha": self._expedition_learner_spec.alpha, "gamma": self._expedition_learner_spec.gamma,
                        "clip": self._expedition_learner_spec.clip, "scope": "frontier_ranking_only",
                        "formula_version": "frontier_td0_vector_v2"}), commit=False)
        except BaseException:
            self._expedition_learner.restore(checkpoint)
            raise
        self._expedition_rewarded_strategy_arms.add(transition["strategy_arm_id"])

    def stop_expedition(self, reason: str = "user_stop") -> Dict[str, Any]:
        if self._expedition is not None:
            self._expedition.stop(reason)
        self._expedition_active_transition = None
        self.stop_research_autonomy("user_stop")
        return self.expedition_status()

    @staticmethod
    def _expedition_refresh_error_class(error_category: Any) -> Optional[str]:
        """Classify only fixed, redacted bridge failures for terminal handling."""
        if not isinstance(error_category, str):
            return None
        authorization_errors = {
            "oauth_credentials_missing", "oauth_credentials_invalid",
            "oauth_access_token_missing", "oauth_unauthorized", "oauth_forbidden",
            "oauth_login_required", "bridge_auth_unavailable",
        }
        schema_errors = {
            "invalid_json", "invalid_payload", "invalid_transport_response",
            "unrecognized_usage_payload", "ambiguous_usage_windows",
            "bridge_invalid_json", "bridge_invalid_payload",
        }
        if error_category in authorization_errors:
            return "quota_refresh_permanent_authorization_error"
        if error_category in schema_errors:
            return "quota_refresh_permanent_schema_error"
        return None

    @staticmethod
    def _expedition_retry_at(refresh: Any, now: datetime) -> datetime:
        """Use a bounded local retry when the bridge gives no usable deadline."""
        candidates = (getattr(refresh, "next_refresh_at", None),
                      getattr(refresh, "next_retry_at", None))
        for candidate in candidates:
            if (isinstance(candidate, datetime) and candidate.tzinfo is not None
                    and candidate.utcoffset() is not None and candidate > now):
                return candidate
        return now + timedelta(seconds=5)

    def expedition_slice(self, usage_adapter: Any) -> Dict[str, Any]:
        """Run one bounded slice, then refresh the derived graph projection.

        The ``finally`` covers every quota, stop, experiment, and web-return
        branch without making graph maintenance part of any control decision.
        """
        try:
            return self._expedition_slice_impl(usage_adapter)
        finally:
            try:
                self._refresh_memory_graph()
            except Exception:
                pass

    def _expedition_slice_impl(self, usage_adapter: Any) -> Dict[str, Any]:
        if self._expedition is None or self._expedition_refresh_gate is None:
            raise RuntimeError("expedition is not configured")
        if (self._expedition.state.value == "waiting_quota_retry"
                and not self._expedition.quota_retry_due(self._sleep_now())):
            return self.expedition_status()
        if self._sleep_now() >= self._expedition_expires_at:
            self._expedition.complete_if_expired(self._sleep_now())
            return self.expedition_status()
        refresher = getattr(usage_adapter, "refresh_foreground_slice", None)
        if not callable(refresher) or self.quota_controller is None:
            now = self._sleep_now()
            self._expedition.defer_quota_retry(
                "quota_refresh_unknown_fail_closed", now + timedelta(seconds=5), now)
            return self.expedition_status()
        try:
            refresh = refresher(self.quota_controller, self._expedition_refresh_gate,
                                self._sleep_now(), force=False)
        except Exception:
            # The bridge boundary exposes no exception text to the scheduler.
            # A normal bridge failure has no authority to manufacture sleep,
            # K3 work, or grants; it only consumes its finite retry budget.
            now = self._sleep_now()
            self._expedition.defer_quota_retry(
                "quota_refresh_transient_retry_pending", now + timedelta(seconds=5), now)
            return self.expedition_status()
        terminal_error = self._expedition_refresh_error_class(
            getattr(refresh, "error_category", None))
        if terminal_error is not None:
            self._expedition.stop(terminal_error)
            return self.expedition_status()
        managed_sleep_entered = False
        if refresh.attempted and refresh.accepted and refresh.usage_result is not None:
            managed_sleep_entered = self.update_managed_usage(refresh.usage_result)
        # Store independently-auditable rolling-window evidence before the
        # scheduler makes another automatic choice.
        # The bridge's foreground gate is authoritative for automatic work.
        # A second fetch would be redundant and could race a reset; sleep is
        # entered only from explicit managed-usage observations elsewhere.
        coordinator_state = (self.sleep_coordinator.state
                             if self.sleep_coordinator is not None else None)
        if (refresh.archive_threshold_reached
                and managed_sleep_entered
                and coordinator_state == SleepState.SLEEPING
                and self._sleep_archive_event_id is not None):
            self._expedition.sleep("quota_sleep")
            return self.expedition_status()
        if refresh.archive_threshold_reached:
            # A bridge's threshold flag is never itself a lifecycle command.
            # Without a completed, authoritative rolling-window archive,
            # fail closed into bounded foreground retry, not a split sleep.
            now = self._sleep_now()
            self._expedition.defer_quota_retry(
                "quota_refresh_transient_retry_pending",
                self._expedition_retry_at(refresh, now), now)
            return self.expedition_status()
        if coordinator_state == SleepState.SLEEPING:
            self._expedition.sleep("quota_sleep")
            return self.expedition_status()
        if coordinator_state is not None and coordinator_state != SleepState.ACTIVE:
            self._expedition.stop("sleep_coordinator_inactive")
            return self.expedition_status()
        if not (refresh.attempted and refresh.accepted):
            interval_wait = refresh.reason == "quota_refresh_interval_waiting"
            reset_wait = refresh.reason == "quota_reset_requires_fresh_telemetry"
            transient_errors = {
                "bridge_start_failed", "bridge_start_timeout", "bridge_timeout",
                "bridge_network_error", "bridge_http_error", "provider_rate_limited",
            }
            # A gate's own interval/retry response and a specifically redacted
            # transient bridge error both mean "wait, then ask the provider
            # again".  They cannot authorize work, but must not turn a
            # temporary telemetry outage into a permanent expedition stop.
            transient_wait = (not refresh.attempted
                              or getattr(refresh, "error_category", None) in transient_errors)
            now = self._sleep_now()
            waiting_reason = ("quota_refresh_interval_waiting" if interval_wait else
                              "quota_reset_requires_fresh_telemetry" if reset_wait else
                              "quota_refresh_transient_retry_pending" if transient_wait else
                              "quota_refresh_unknown_fail_closed")
            self._expedition.defer_quota_retry(waiting_reason,
                                               self._expedition_retry_at(refresh, now), now)
            return self.expedition_status()
        if not refresh.automatic_allowed:
            interval_wait = refresh.reason == "quota_refresh_interval_waiting"
            reset_wait = refresh.reason == "quota_reset_requires_fresh_telemetry"
            now = self._sleep_now()
            self._expedition.defer_quota_retry(
                "quota_refresh_interval_waiting" if interval_wait else
                "quota_reset_requires_fresh_telemetry" if reset_wait else
                "quota_refresh_transient_retry_pending",
                self._expedition_retry_at(refresh, now), now)
            return self.expedition_status()
        if self._expedition.state.value == "waiting_quota_retry":
            if not self._expedition.resume_after_fresh_quota(self._sleep_now()):
                return self.expedition_status()
        # Reproject only at the slice boundary.  This cannot affect any
        # already active task and runs after all quota/sleep/stop gates above.
        guidance = self._expedition_seed_guidance()
        # Every selected slice gets its own durable context edge, including an
        # empty projection.  A correction, revocation, or expiry between
        # slices therefore becomes visible in the next ranking's lineage.
        # Failure is terminal before a task, tool grant, or model call exists.
        try:
            assert self._expedition_consumption_event_id is not None
            consumed = self.event_store.get(self._expedition_consumption_event_id)
            if consumed is None:
                raise RuntimeError("expedition consumption record is unavailable")
            self._expedition_seed_context_event_id = self._record_expedition_seed_context(
                str(consumed.payload["run_id"]), self._expedition_consumption_event_id, guidance).event_id
        except BaseException:
            self._frontier_fail_closed("seed_context_store_failure")
            raise
        if guidance:
            snapshot = SeedGuidanceSnapshot(
                seed_ids=tuple(item.seed_id for item in guidance),
                authority_event_ids=tuple(item.current_authority_event_id for item in guidance),
                digest=sha256(json.dumps([
                    {"seed_id": item.seed_id,
                     "authority_event_id": item.current_authority_event_id,
                     "snapshot_digest": item.snapshot_digest,
                     "priority_band": item.priority_band}
                    for item in guidance], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
            self._expedition.set_seed_guidance(snapshot)
        else:
            self._expedition.set_seed_guidance(None)
        decision = self._expedition.begin_slice(self._sleep_now())
        if decision is None:
            return self.expedition_status()
        # All hard gates above (expiry, quota, sleep and scheduler stop) run
        # before the optional ranker can become an auditable active choice.
        self._frontier_begin_transition(decision)
        if self._expedition.state.value != "active":
            return self.expedition_status()
        if getattr(decision.task, "experiment_kind", None):
            try:
                experiment = self._run_frontier_experiment(decision)
                if self._expedition.state.value == "active":
                    if experiment.get("reward_qualified"):
                        outcome = (ExpeditionOutcome.EXPERIMENT_SUPPORTED
                                   if experiment.get("last_status") == "supported"
                                   else ExpeditionOutcome.EXPERIMENT_REFUTED)
                        self._expedition.record_outcome(decision.task.task_id,
                            outcome, quality_score=.70,
                            quality_signals=("paired_baseline_treatment", "independent_reproduction"),
                            now=self._sleep_now())
                    elif (experiment.get("last_status") in {"supported", "refuted"}
                          and experiment.get("reproducible")
                          and experiment.get("is_novel_result") is False):
                        # A canonical duplicate is a valid, bounded fixture
                        # completion.  It is deliberately not new learning or
                        # expedition-quality progress, so close this exact
                        # task without a reward/TD chain or retry.
                        self._expedition.record_outcome(
                            decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                            now=self._sleep_now())
                    else:
                        # A failed, incomplete or irreproducible fixture audit
                        # is visible but cannot become expedition progress.
                        self._expedition.record_outcome(decision.task.task_id,
                            (ExpeditionOutcome.EXPERIMENT_INCONCLUSIVE
                             if experiment.get("last_status") == "inconclusive"
                             else ExpeditionOutcome.EXPERIMENT_INVALID), now=self._sleep_now())
            finally:
                self._expedition_active_transition = None
            return self.expedition_status()
        self._expedition_current_persona = decision.persona.value
        profile = ResearchAutonomyProfile(self.tool_session.workspace_id, public_web_only=True,
            budget=__import__("strangeloop.capabilities", fromlist=["ResearchBudget"]).ResearchBudget(
                max_tool_calls=self._expedition_max_calls, max_total_bytes=2 * 1024 * 1024,
                max_response_bytes=64 * 1024, max_wall_ms=decision.max_seconds * 1000,
                ttl_seconds=decision.max_seconds))
        task = decision.task
        slice_goal = "%s\nSlice target: persona=%s topic=%s source=%s stance=%s mode=%s. Use only public web reading." % (
            self._expedition_goal, decision.persona.value, task.topic_cluster, task.source_type,
            task.evidence_stance, task.action_mode)
        policy = UnattendedPolicy.user_issued(slice_goal, self._sleep_now() + timedelta(seconds=decision.max_seconds))
        try:
            self._start_unattended_research(profile, policy, self._expedition_approval_event_id,
                                            model_trigger_event_id=self._expedition_model_trigger_event_id)
            outcome = self.run_unattended_research()
            report = outcome["public_report"]
            actions = report.get("actions", ())
            if not actions:
                label = (ExpeditionOutcome.BRANCH_COMPLETE
                         if self._last_unattended_planner_outcome == "respond"
                         else ExpeditionOutcome.PLANNER_TRANSIENT_FAILURE)
            else:
                label, quality_score, quality_signals = self._expedition_quality(report)
            self._expedition.record_outcome(decision.task.task_id, label,
                quality_score=(quality_score if actions else 0.0),
                quality_signals=(quality_signals if actions else ()),
                now=self._sleep_now())
            # A direct controlled TOOL_RESULT is the only automatic evidence
            # source.  Planner text, web content, lifecycle state and quota
            # telemetry never become rewards.
            self._frontier_record_document(report)
        except (RuntimeError, ValueError, PermissionError):
            if self._expedition.state.value == "active":
                self._expedition.record_outcome(decision.task.task_id, ExpeditionOutcome.PLANNER_TRANSIENT_FAILURE,
                                                now=self._sleep_now())
        finally:
            # Each slice has a fresh profile and the previous grants are
            # immediately invalidated, regardless of planner outcome.
            self.stop_research_autonomy("slice_complete")
            self._expedition_active_transition = None
        return self.expedition_status()

    def _expedition_quality(self, report: Dict[str, Any]) -> Tuple[ExpeditionOutcome, float, Tuple[str, ...]]:
        """Classify only externally visible tool results, never model self-ratings."""
        actions = report.get("actions", ())
        findings = report.get("findings", ())
        if not any(isinstance(item, dict) and item.get("status") == "succeeded" for item in actions):
            return ExpeditionOutcome.LOW_VALUE, 0.0, ()
        summaries = [str(item.get("summary", "")) for item in findings if isinstance(item, dict)]
        if summaries and all(text == "No safe public HTTPS search results." for text in summaries):
            return ExpeditionOutcome.EMPTY_SEARCH, 0.0, ()
        # Search-result lists are leads, not evidence.  Quality progress needs
        # a successful direct read/fetch of a new source domain.
        direct = [item for item in findings if isinstance(item, dict)
                  and item.get("status") == "succeeded"
                  and item.get("tool") in ("web.fetch", "browser.read")]
        domains = set()
        for item in direct:
            for match in re.findall(r"\burl=(https://[^\s]+)", str(item.get("summary", ""))):
                host = urlsplit(match).hostname
                if host:
                    domains.add(host.lower())
        novel = sorted(domains.difference(self._expedition_domains))
        self._expedition_domains.update(domains)
        primary_suffixes = ("arxiv.org", "openreview.net", "proceedings.mlr.press",
                            "aclanthology.org", "dl.acm.org", "ieeexplore.ieee.org",
                            "nature.com", "science.org")
        primary = any(domain == suffix or domain.endswith("." + suffix)
                      for domain in domains for suffix in primary_suffixes)
        signals = []
        if novel:
            signals.append("novel_domain")
        if primary:
            signals.append("primary_source")
        if signals:
            self._expedition_useful_findings += 1
            return ExpeditionOutcome.QUALITY_PROGRESS, (.85 if len(signals) > 1 else .70), tuple(signals)
        return ExpeditionOutcome.BRANCH_COMPLETE, 0.0, ()

    def run_expedition_foreground(self, usage_adapter: Any, on_slice: Any = None) -> Dict[str, Any]:
        """Run every currently-ready bounded slice on the caller thread."""
        if (self._expedition is not None and self._expedition.state.value == "waiting_quota_retry"
                and not self._expedition.quota_retry_due(self._sleep_now())):
            return self.expedition_status()
        if self._expedition is not None and self._expedition.state.value == "waiting_quota_retry":
            status = self.expedition_slice(usage_adapter)
            if callable(on_slice):
                on_slice(status)
        while self._expedition is not None and self._expedition.state.value in ("ready", "active"):
            status = self.expedition_slice(usage_adapter)
            if callable(on_slice):
                on_slice(status)
            if status.get("state") not in ("ready", "active"):
                break
        return self.expedition_status()

    def stop_research_autonomy(self, reason: str = "user_stop") -> bool:
        """Stop current unattended research and invalidate its live grants."""
        controller = self._unattended
        changed = False
        if controller is not None:
            if reason == "sleep":
                changed = controller.on_sleep()
            elif reason == "quota_exhausted":
                changed = controller.on_quota_exhausted()
            else:
                changed = controller.stop(UnattendedStopReason.USER_STOP)
        if self.tool_session is not None:
            changed = self.tool_session.stop_research_autonomy(reason) or changed
        self._unattended_prepared, self._unattended_proposals = {}, {}
        if reason in ("user_stop", "terminal", "purge"):
            self._clear_wake_continuation(reason)
        return changed

    def unattended_step(self) -> Dict[str, Any]:
        if self._unattended is None:
            raise RuntimeError("unattended research is not active; start it with an explicit user policy")
        self._maybe_enter_sleep()
        receipt = self._unattended.step()
        return {"receipt": self._unattended_receipt(receipt), "status": self.unattended_status(),
                "public_report": self.unattended_public_report()}

    def run_unattended_research(self) -> Dict[str, Any]:
        if self._unattended is None:
            raise RuntimeError("unattended research is not active; start it with an explicit user policy")
        self._maybe_enter_sleep()
        receipts = self._unattended.run()
        return {"receipts": [self._unattended_receipt(item) for item in receipts],
                "status": self.unattended_status(), "public_report": self.unattended_public_report()}

    def unattended_public_report(self) -> Dict[str, Any]:
        """Deterministic report from existing bounded tool outcomes only.

        This intentionally makes no provider call and stores no raw web page
        or model reasoning.  Findings can be reproduced from the existing
        public TOOL_RESULT records while the live process remains available.
        """
        controller = self._unattended
        policy = getattr(controller, "_policy", None) if controller is not None else None
        receipts = controller.receipts() if controller is not None else ()
        actions = [{"tool": item.action.tool_name, "status": item.action.status,
                    "disposition": item.action.disposition}
                   for item in receipts if item.action is not None]
        report = {
            "goal_digest": sha256(policy.focus.encode("utf-8")).hexdigest() if policy is not None else None,
            "stop_reason": controller.stop_reason.value if controller is not None else "not_started",
            "actions": actions,
            "findings": [dict(item) for item in self._unattended_findings],
            "limitations": ["read_only_public_https_and_workspace_repository_only",
                            "no_credentials_cookies_uploads_writes_tests_or_commands",
                            "findings_are_untrusted_tool_summaries_not_verified_facts",
                            "no_additional_model_call_was_made_for_this_report"],
        }
        report["report_digest"] = sha256(
            __import__("json").dumps(report, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
        return report

    def unattended_status(self) -> Dict[str, Any]:
        controller = self._unattended
        session = self.tool_session.research_autonomy_status() if self.tool_session is not None else None
        result = {"configured": controller is not None, "active": False,
                  "restart_behavior": "stopped_requires_fresh_user_policy",
                  "authority": "explicit_user_read_only_profile"}
        if controller is not None:
            result.update(controller.snapshot())
            result["active"] = result["state"] == "running" and bool(session and session.get("active"))
            receipts = controller.receipts()
            policy = getattr(controller, "_policy", None)
            result["goal_digest"] = (sha256(policy.focus.encode("utf-8")).hexdigest()
                                     if policy is not None else None)
            result["last_tool"] = (receipts[-1].action.tool_name
                                   if receipts and receipts[-1].action is not None else None)
            result["last_action_status"] = (receipts[-1].action.status
                                            if receipts and receipts[-1].action is not None else None)
            result["budget"] = {"max_calls": controller.MAX_CALLS,
                                "max_wall_seconds": controller.MAX_WALL_SECONDS,
                                "max_output_bytes": controller.MAX_OUTPUT_BYTES}
            report = self.unattended_public_report()
            result["report_digest"] = report["report_digest"]
            result["finding_count"] = len(report["findings"])
        if session is not None:
            result["tool_profile"] = session
        result["wake_continuation"] = self.wake_continuation_status()
        return result

    def _unattended_plan(self, context: UnattendedContext,
                          token: CancellationToken) -> Sequence[ResearchProposal]:
        """Run K3 planning with compact public context; retain no raw output."""
        if token.is_set() or self.runtime is None or self._unattended_policy_event_id is None:
            return ()
        self._last_unattended_planner_outcome = "proposal"
        run_id = "researchplan_%s" % uuid4().hex
        invocation_id = "inv_%s" % uuid4().hex
        started_at = time.monotonic()
        self._append_model_invocation(invocation_id + ".start", run_id, self._unattended_policy_event_id,
                                      "started", 0, "K3 unattended read-only planning request started.", (),
                                      role="tool_planning", context_scope="tool_planning")
        try:
            intents = KimiToolPlanner(self.runtime).plan(context.focus, {
                "mode": "unattended_read_only_research",
                "tick": context.tick_number,
                "remaining_calls": context.remaining_calls,
                "allowed_tools": self._unattended_allowed_tools,
                # Results are bounded public tool summaries supplied as
                # untrusted data.  They have no authority over grants, quota,
                # policy, or execution and help K3 avoid repeated lookups.
                "previous_public_results": {"untrusted_data": [dict(item) for item in self._unattended_findings]},
            })
        except KimiCodeProviderError as error:
            self._last_unattended_planner_outcome = "provider_error"
            elapsed = int((time.monotonic() - started_at) * 1000)
            timed_out = error.category == "provider_timeout"
            outcome = "timed_out" if timed_out else ("budget_exhausted" if self._quota_denied() else "failed")
            self._append_model_invocation(invocation_id + ".finish", run_id, self._unattended_policy_event_id,
                                          outcome, elapsed,
                                          ("K3 unattended planning timed out; no tool was executed."
                                           if timed_out else "K3 unattended planning unavailable; no tool was executed."), (),
                                          role="tool_planning", context_scope="tool_planning")
            return ()
        except Exception:
            self._last_unattended_planner_outcome = "planner_error"
            elapsed = int((time.monotonic() - started_at) * 1000)
            self._append_model_invocation(invocation_id + ".finish", run_id, self._unattended_policy_event_id,
                                          "failed", elapsed,
                                          "K3 unattended planning unavailable; no tool was executed.", (),
                                          role="tool_planning", context_scope="tool_planning")
            return ()
        elapsed = int((time.monotonic() - started_at) * 1000)
        invocation = self._append_model_invocation(
            invocation_id + ".finish", run_id, self._unattended_policy_event_id, "completed", elapsed,
            "K3 proposed bounded actions; the host still enforces the read-only profile.", (),
            role="tool_planning", context_scope="tool_planning")
        # Provider calls may change quota state (including a provider-side
        # exhaustion response) while planning.  Re-check before turning an
        # intent into any host-side action; a same-tick proposal is never an
        # authority to continue past a sleep/quota boundary.
        if self._stop_unattended_if_gated():
            return ()
        proposals = []
        for intent in intents[:1]:
            # A direct response closes research planning.  It is not a tool
            # call and therefore cannot become a hidden action or extra K3
            # synthesis step; the deterministic report retains prior findings.
            if intent.tool_name == "respond":
                self._last_unattended_planner_outcome = "respond"
                return ()
            proposal_id = "researchproposal_%s" % uuid4().hex
            proposal = ResearchProposal(proposal_id, intent.tool_name, dict(intent.arguments),
                                        intent.rationale_summary)
            self._unattended_proposals[proposal_id] = (intent, invocation.event_id, run_id)
            proposals.append(proposal)
        return tuple(proposals)

    def _unattended_prepare(self, proposal: ResearchProposal, context: UnattendedContext,
                            token: CancellationToken) -> Optional[PreparedResearchAction]:
        del context
        if token.is_set() or self.tool_session is None or self._stop_unattended_if_gated():
            return None
        item = self._unattended_proposals.pop(proposal.proposal_id, None)
        if item is None:
            return None
        intent, invocation_id, run_id = item
        prepared = self.tool_session.prepare(intent, run_id, invocation_id)
        # Write/test/command proposals remain rejected even if a provider
        # violated its response schema.  Read-only calls are confirmation-free
        # only because the user activated this fixed profile.
        if not prepared.is_ready or prepared.requires_confirmation:
            return None
        action_id = "researchaction_%s" % uuid4().hex
        self._unattended_prepared[action_id] = prepared
        return PreparedResearchAction(action_id, proposal.host_tool_name, dict(proposal.arguments))

    def _unattended_execute(self, action: PreparedResearchAction,
                            token: CancellationToken) -> ResearchExecution:
        # This must precede ``pop`` and ToolSession.execute: stop clears all
        # pending actions, and more importantly prevents any TOOL_* execution
        # records or backend call after a just-observed quota/sleep boundary.
        if token.is_set() or self.tool_session is None or self._stop_unattended_if_gated():
            return ResearchExecution("cancelled", "Read-only research stopped before tool execution.", 0, False)
        prepared = self._unattended_prepared.pop(action.action_id, None)
        if prepared is None or self.tool_session is None:
            return ResearchExecution("refused", "Read-only research action was not prepared.", 0, False)
        result = self.tool_session.execute(prepared, cancel_token=token)
        # ToolSession's public success spelling is ``succeeded``.  Keep the
        # historical ``ok`` alias for injected compatibility, but do not turn
        # a completed TOOL_RESULT into a failed research receipt.
        status = {"ok": "succeeded", "succeeded": "succeeded", "responded": "succeeded", "timed_out": "timed_out",
                  "cancelled": "cancelled", "refused": "refused"}.get(result.status, "failed")
        summary = (result.public_summary or "").replace("\x00", " ")[:1024]
        result_event_ids = tuple(result.event_ids)
        result_event_id = next((event_id for event_id in reversed(result_event_ids)
                                if (self.event_store.get(event_id) is not None
                                    and self.event_store.get(event_id).kind == EventKind.TOOL_RESULT)), None)
        finding = {"tool": action.tool_name, "status": status, "summary": summary,
                   "result_event_id": result_event_id,
                   "summary_digest": sha256(summary.encode("utf-8")).hexdigest()}
        self._unattended_findings.append(finding)
        if len(self._unattended_findings) > 8:
            del self._unattended_findings[:-8]
        return ResearchExecution(status, summary, len(summary.encode("utf-8")), status == "succeeded")

    def _stop_unattended_if_gated(self) -> bool:
        """Close a research profile at every host boundary, not only tick entry."""
        sleeping = self.sleep_coordinator is not None and self.sleep_coordinator.state != SleepState.ACTIVE
        quota_denied = self._quota_denied()
        if sleeping:
            self.stop_research_autonomy("sleep")
        elif quota_denied:
            self.stop_research_autonomy("quota_exhausted")
        return sleeping or quota_denied

    @staticmethod
    def _unattended_receipt(receipt: Any) -> Dict[str, Any]:
        action = getattr(receipt, "action", None)
        return {"tick_number": receipt.tick_number, "state_before": receipt.state_before.value,
                "state_after": receipt.state_after.value, "stop_reason": receipt.stop_reason.value,
                "action": (None if action is None else {"tool_name": action.tool_name,
                           "disposition": action.disposition, "status": action.status,
                           "input_bytes": action.input_bytes, "output_bytes": action.output_bytes})}

    def record_user_feedback(self, target_event_id: str, feedback: str) -> Dict[str, Any]:
        """Apply explicit target-bound user feedback to the alignment channel only."""
        if self.drives is None:
            raise RuntimeError("drives are not configured")
        if feedback not in ("accept", "correct"):
            raise ValueError("feedback must be accept or correct")
        target = self.event_store.get(target_event_id)
        if target is None or target.session_id != self.session_id:
            raise ValueError("feedback target must belong to the current session")
        forbidden = {EventKind.MODEL_INVOCATION, EventKind.AUTONOMY_CONTROL,
                     EventKind.AUTONOMY_STOPPED, EventKind.PURGE,
                     EventKind.METACOGNITIVE_MIRROR,
                     EventKind.SEED_PROPOSED, EventKind.SEED_APPROVED,
                     EventKind.SEED_RETIRED, EventKind.SEED_STANDING_POLICY,
                     EventKind.SEED_STANDING_POLICY_REVOKED,
                     EventKind.SEED_UPDATE_PROPOSED,
                     EventKind.SEED_AUTO_ELIGIBILITY,
                     EventKind.SEED_AUTO_APPLIED}
        if target.kind in forbidden:
            raise ValueError("control, seed, quota, and mirror records cannot be reward targets")
        feedback_event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user_drive_feedback",
            payload={"content": "%s feedback for a bounded prior outcome" % feedback,
                     "channel": "feedback"}, parent_event_ids=(target.event_id,),
        ))
        kind = UserFeedbackKind.ACCEPTED if feedback == "accept" else UserFeedbackKind.CORRECTION
        observation = DriveObservation(
            observation_id="drive_%s" % uuid4().hex, source=ObservationSource.USER,
            source_ref="user", provenance_ids=(feedback_event.event_id, target.event_id),
            feedback_kind=kind, feedback_id="feedback_%s" % feedback_event.event_id,
            target_event_id=target.event_id,
        )
        reward, records = self.drives.observe(observation)
        return {"target_event_id": target.event_id, "feedback": feedback,
                "reward": reward.to_payload(), "channel_updates": [item.to_payload() for item in records]}

    def run_turn(self, user_text: str, authorized: bool = False) -> TurnResult:
        """Process one user observation and execute only a safe response action."""
        self._maybe_enter_sleep()
        observation = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            payload={"content": user_text, "channel": "text"},
        ))
        active_seeds = self.seed_store.retrieve(
            self.session_id, self._seed_cue_terms(user_text), scope="conversation"
        )
        # Build guidance from the pre-turn projection.  Any automatic seed
        # maintenance happens only after the result below, therefore it can
        # influence the next turn but cannot rewrite this model request.
        seed_guidance = self._seed_guidance_for(active_seeds)
        claims = self.self_model.current_claims(self.session_id)
        workspace = WorkspaceFrame(
            turn_id="turn_%s" % uuid4().hex,
            observation_event_ids=(observation.event_id,),
            retrieved_seed_ids=tuple(seed.seed_id for seed in active_seeds),
            self_claim_ids=tuple(claim.claim_id for claim in claims),
            hypotheses=(), uncertainties=(),
            percept_event_ids=self._event_ids(EventKind.PERCEPT),
            loop_tick_event_ids=self._event_ids(EventKind.LOOP_TICK),
            reward_event_ids=self._event_ids(EventKind.REWARD_OBSERVATION),
            value_estimate_event_ids=self._event_ids(EventKind.VALUE_ESTIMATE),
            seed_guidance=seed_guidance,
        )
        deliberation, provider_notice = self._deliberate_with_fallback(
            user_text, workspace, observation.event_id
        )
        proposed = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.ACTION_PROPOSED,
            source_kind=SourceKind.MODEL, source_ref="deliberator",
            payload=self._action_payload(deliberation.action),
            parent_event_ids=(observation.event_id,),
        ))
        policy = self.policy_gate.evaluate(deliberation.action, deliberation.response_text, authorized,
                                           requested_text=user_text)
        selected = deliberation.action
        response = deliberation.response_text
        notices = list(policy.reasons)
        if provider_notice:
            notices.append(provider_notice)
        if not policy.allowed:
            selected = ActionProposal(
                action_type="response", rationale_summary="Policy-safe fallback response.",
                arguments={"text": policy.safe_response}, is_mutating=False,
            )
            response = policy.safe_response
        decision = DecisionRecord(
            turn_id=workspace.turn_id, observation_event_ids=workspace.observation_event_ids,
            retrieved_seed_ids=workspace.retrieved_seed_ids, self_claim_ids=workspace.self_claim_ids,
            alternatives=deliberation.alternatives, selected_action=selected,
            uncertainties=deliberation.uncertainties, policy_reasons=policy.reasons,
        )
        decision_event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.DECISION,
            source_kind=SourceKind.POLICY if policy.reasons else SourceKind.MODEL,
            source_ref="PolicyGate" if policy.reasons else "deliberator",
            payload=self._decision_payload(decision),
            parent_event_ids=(observation.event_id, proposed.event_id),
        ))
        result_event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.ACTION_RESULT,
            source_kind=SourceKind.SYSTEM, source_ref="StrangeloopAgent",
            payload={"action_type": selected.action_type, "outcome": "response_returned",
                     "response_text": response},
            parent_event_ids=(decision_event.event_id,),
        ))
        mirror_event = self._append_mirror(
            episode_id=workspace.turn_id,
            target_event=observation,
            judgment_event=result_event,
            evidence_events=(observation, result_event),
        )
        seed_ids = ()
        seed_auto = self.seed_auto_update_status()
        if seed_auto["state"] == "active":
            seed_ids, seed_action = self._auto_process_text_seed(
                user_text, observation.event_id, decision_event.event_id)
            if seed_action in ("activate", "reinforce"):
                notices.append(
                    "Seed auto-update was applied by the host under the active user standing policy; "
                    "no per-seed approval was required."
                )
            else:
                notices.append(
                    "Seed auto-update was handled automatically under the active user standing policy; "
                    "no per-seed approval was requested."
                )
        elif deliberation.proposed_seed is not None:
            candidate = self._server_seed_candidate(
                user_text, observation.event_id, decision_event.event_id
            )
            stored = self.seed_store.propose(self.session_id, candidate, source_ref="deliberator")
            seed_ids = (stored.seed_id,)
            notices.append("Seed proposal is a candidate and requires external approval.")
        self._signal_external_salience(observation.event_id, "text")
        self._refresh_memory_graph()
        return TurnResult(
            session_id=self.session_id, response_text=response, decision=decision,
            # TurnResult continues to identify the four operational turn
            # records.  The mirror is an independent audit record, discoverable
            # through state/export, not an action result or reward target.
            event_ids=(observation.event_id, proposed.event_id, decision_event.event_id,
                       result_event.event_id),
            seed_proposal_ids=seed_ids, notices=tuple(notices),
        )

    def ingest_media(self, stream: BinaryIO, perceptor: Optional[MediaPerceptor] = None,
                     retention_scope: str = "ephemeral") -> Tuple[CognitiveEvent, ...]:
        """Inspect a transient user stream and persist only bounded public projections."""
        adapter = perceptor or BasicMetadataPerceptor()
        artifact, percepts = ingest_media(stream, adapter)
        adapter_id, adapter_version = self._adapter_identity(adapter.perceptor_id)
        percept_kind = "metadata" if isinstance(adapter, BasicMetadataPerceptor) else "annotation"
        return self.record_media(artifact, percepts, retention_scope, adapter_id,
                                 adapter_version, percept_kind)

    def record_media(self, artifact: MediaArtifact, percepts: Sequence[Percept] = (),
                     retention_scope: str = "ephemeral", adapter_id: str = "metadata",
                     adapter_version: str = "v1", percept_kind: str = "metadata") -> Tuple[CognitiveEvent, ...]:
        """Record a user-supplied image/audio artifact without storing bytes or paths.

        The adapter has already received a transient read-only stream via
        :meth:`ingest_media`; this method only receives public projections.
        """
        if not isinstance(artifact, MediaArtifact):
            raise ValueError("artifact must be a MediaArtifact")
        if retention_scope not in {"ephemeral", "session", "user_approved"}:
            raise ValueError("retention_scope is invalid")
        validate_percepts(artifact, percepts, adapter_id + "/" + adapter_version)
        media_candidate = CognitiveEvent(
            session_id=self.session_id, kind=EventKind.MEDIA_OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user_media",
            payload={"artifact_id": artifact.artifact_id, "modality": artifact.modality,
                     "sha256": artifact.sha256, "mime_type": artifact.mime_type,
                     "byte_length": artifact.byte_length,
                     "received_at": self._utc_now(), "duration_ms": artifact.duration_ms,
                     "retention_scope": retention_scope},
        )
        if not isinstance(percepts, Sequence) or len(percepts) > 32:
            raise ValueError("perceptor must return at most 32 percepts")
        percept_candidates = []
        for percept in percepts:
            if not isinstance(percept, Percept):
                raise ValueError("perceptor must return Percept values")
            if percept.artifact_id != artifact.artifact_id or percept.modality != artifact.modality:
                raise ValueError("percept must bind to the supplied artifact")
            percept_candidates.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.PERCEPT,
                source_kind=SourceKind.SYSTEM, source_ref="MediaPerceptor",
                payload=self._percept_payload(percept, artifact, adapter_id,
                                              adapter_version, percept_kind),
                confidence=float(percept.confidence), parent_event_ids=(media_candidate.event_id,),
            ))
        # Validate all perceptor projections before the transaction commits;
        # a bad adapter result leaves no partial media observation behind.
        with self.event_store.transaction():
            media = self.event_store.append(media_candidate, commit=False)
            records = [media]
            for candidate in percept_candidates:
                records.append(self.event_store.append(candidate, commit=False))
        self._signal_external_salience(media.event_id, "media")
        return tuple(records)

    def start_loop(self, config: Optional[LoopConfig] = None) -> Dict[str, Any]:
        """Create a new foreground-driven bounded functional loop, initially running."""
        if self._loop is not None and self._loop.state not in (LoopState.STOPPED, LoopState.EXHAUSTED):
            raise RuntimeError("stop the current loop run before starting a new one")
        loop_config = config or LoopConfig()
        self._validate_persisted_loop_config(loop_config)
        self._loop_run_id = "run_%s" % uuid4().hex
        self._loop = FunctionalLoopController(self._loop_cycle, config=loop_config)
        self._loop_remote_calls_remaining = self._loop_remote_call_budget
        self._loop_stop_recorded = False
        self._processed_focus_event_ids = set()
        control = self._append_loop_control("start")
        self._loop_parent_event_id = control.event_id
        self._loop.resume()
        return self.loop_status()

    def resume_loop(self) -> Dict[str, Any]:
        """Resume a paused run; exhausted/stopped runs must be explicitly restarted."""
        loop = self._require_loop()
        if not loop.resume():
            raise RuntimeError("this loop run cannot be resumed; start a new run")
        control = self._append_loop_control("resume")
        self._loop_parent_event_id = control.event_id
        return self.loop_status()

    def pause_loop(self) -> Dict[str, Any]:
        """Pause future ticks without starting any background worker."""
        loop = self._require_loop()
        if loop.pause():
            control = self._append_loop_control("pause")
            self._loop_parent_event_id = control.event_id
        return self.loop_status()

    def stop_loop(self) -> Dict[str, Any]:
        """Stop the current run and append an externally attributable halt record."""
        loop = self._require_loop()
        if loop.state in (LoopState.STOPPED, LoopState.EXHAUSTED):
            return self.loop_status()
        control = self._append_loop_control("stop")
        self._loop_parent_event_id = control.event_id
        loop.stop()
        self._append_loop_stopped("user_requested", control.event_id)
        return self.loop_status()

    def stop_sleep(self) -> None:
        """Terminally cancel pending wake authority on explicit host shutdown."""
        self.stop_research_autonomy("terminal")
        if self._expedition is not None:
            self._expedition.stop("terminal")
        self._expedition_active_transition = None
        self._clear_wake_continuation("terminal")
        if self.sleep_coordinator is not None:
            if (self.sleep_coordinator.state != SleepState.TERMINAL and self._sleep_epoch_id is not None):
                lifecycle = None
                for event in reversed(self.event_store.list(self.session_id)):
                    if (event.kind in (EventKind.SLEEP_ENTERED, EventKind.WAKE_CHECK,
                                       EventKind.WAKE_READY, EventKind.AWAKE)
                            and event.payload.get("epoch_id") == self._sleep_epoch_id):
                        lifecycle = event
                        break
                if lifecycle is not None:
                    self.event_store.append(CognitiveEvent(
                        session_id=self.session_id, kind=EventKind.WAKE_TERMINAL,
                        source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
                        payload={"wake_terminal_id": "terminal_%s" % uuid4().hex,
                                 "lifecycle_event_id": lifecycle.event_id,
                                 "epoch_id": self._sleep_epoch_id,
                                 "terminal_at": self._sleep_now().isoformat(),
                                 "outcome": "stopped", "protocol_version": "sleep_wake_v1"},
                        parent_event_ids=(lifecycle.event_id,)))
            self.sleep_coordinator.stop()
        if self.quota_controller is not None:
            self.quota_controller.invalidate_reservations(ReservationInvalidationReason.TERMINAL)
        self._pending_tool_plans = {}

    def loop_step(self) -> Dict[str, Any]:
        """Execute one bounded loop callback and append a public tick when it ran."""
        loop = self._require_loop()
        record = loop.step(TickTrigger.MANUAL)
        self._persist_tick(record)
        return self.loop_status()

    def run_loop(self) -> Dict[str, Any]:
        """Automatically drive the foreground loop until paused, idle, or exhausted."""
        loop = self._require_loop()
        for record in loop.run_until_stopped(TickTrigger.SCHEDULED):
            self._persist_tick(record)
        return self.loop_status()

    def loop_status(self) -> Dict[str, Any]:
        if self._loop is None:
            return {"state": "paused", "run_id": None, "restart_behavior": "paused"}
        status = self._loop.snapshot()
        status.update({"run_id": self._loop_run_id, "restart_behavior": "paused"})
        return status

    def record_value_estimate(self, target_event_id: str, action_key: str,
                              next_state_key: Optional[str] = None,
                              terminal: bool = True, confidence: float = 1.0) -> CognitiveEvent:
        """Create a value estimate for an observed action/tick, with no policy effect."""
        self._assert_td_current()
        target = self._rewardable_target(target_event_id)
        if action_key not in SAFE_ACTION_CLASSES:
            raise ValueError("action_key must be a fixed safe action class")
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be a boolean")
        state_key = ValueTable.preference_state_key(action_key)
        transition_id = "transition_%s" % uuid4().hex
        transition = Transition(transition_id=transition_id, context_id=self.session_id,
                                state_key=state_key, action_class=action_key,
                                next_state_key=next_state_key or state_key,
                                terminal=terminal)
        estimate = self.value_table.estimate(self.session_id, state_key)
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        with self.value_table.transaction(), self.event_store.transaction():
            self._assert_td_current()
            if any(event.kind == EventKind.VALUE_ESTIMATE
                   and event.payload["target_event_id"] == target.event_id
                   for event in self.event_store.list(self.session_id)):
                raise ValueError("only one value estimate may be recorded for a target")
            if any(event.kind == EventKind.REWARD_OBSERVATION
                   and event.payload["target_event_id"] == target.event_id
                   for event in self.event_store.list(self.session_id)):
                raise ValueError("a value estimate must be recorded before any reward for this target")
            self.value_table.register_transition(transition)
            event = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="ValueTable",
                payload={"estimate_id": "estimate_%s" % uuid4().hex,
                         "transition_id": transition_id, "target_event_id": target.event_id,
                         "state_key": state_key, "action_key": action_key,
                         "next_state_key": transition.next_state_key, "terminal": terminal,
                         "value": estimate.value,
                         "confidence": float(confidence), "estimator_id": "value_table",
                         "estimator_version": "v1", "alpha": self.value_table.config.alpha,
                         "gamma": self.value_table.config.gamma, "clip": self.value_table.config.clip,
                         "max_entries": self.value_table.max_entries,
                         "max_events": self.value_table.max_events,
                         "formula_version": "td0_v1"}, parent_event_ids=(target.event_id,),
            ), commit=False)
        self._td_transitions[event.event_id] = transition
        self._td_replay_head_sequence = event.sequence or self._td_replay_head_sequence
        return event

    def record_reward(self, target_event_id: str, normalized_value: float,
                      source: RewardSource, evaluator_ref: str,
                      signal_kind: str = "user_feedback") -> CognitiveEvent:
        """Record an externally attributable reward; model/system sources are rejected."""
        self._assert_td_current()
        target = self._rewardable_target(target_event_id)
        if not any(event.kind == EventKind.VALUE_ESTIMATE
                   and event.payload["target_event_id"] == target.event_id
                   for event in self.event_store.list(self.session_id)):
            raise ValueError("record a value estimate before recording reward")
        if not isinstance(source, RewardSource):
            raise ValueError("reward source must be user, tool, or external_verifier")
        if signal_kind not in {"user_feedback", "task_outcome", "external_score"}:
            raise ValueError("signal_kind is invalid")
        if (not isinstance(normalized_value, (int, float)) or isinstance(normalized_value, bool)
                or not math.isfinite(float(normalized_value))
                or not -1.0 <= float(normalized_value) <= 1.0):
            raise ValueError("normalized_value must be between -1 and 1")
        with self.event_store.transaction():
            self._assert_td_current()
            if any(event.kind == EventKind.REWARD_OBSERVATION
                   and event.payload["target_event_id"] == target.event_id
                   for event in self.event_store.list(self.session_id)):
                raise ValueError("only one reward may be recorded for a target")
            event = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.REWARD_OBSERVATION,
                source_kind=SourceKind(source.value), source_ref=evaluator_ref,
                payload={"reward_id": "reward_%s" % uuid4().hex,
                         "target_event_id": target.event_id, "signal_kind": signal_kind,
                         "normalized_value": float(normalized_value),
                         "evaluator_ref": evaluator_ref, "scale_version": "rpe_scale_v1",
                         "observed_at": self._utc_now()}, parent_event_ids=(target.event_id,),
            ), commit=False)
        self._td_replay_head_sequence = event.sequence or self._td_replay_head_sequence
        return event

    def apply_rpe_update(self, reward_event_id: str,
                         prior_value_event_id: str) -> CognitiveEvent:
        """Apply one bounded TD(0) update and persist its auditable arithmetic."""
        self._assert_td_current()
        reward_event = self._session_event(reward_event_id, EventKind.REWARD_OBSERVATION)
        value_event = self._session_event(prior_value_event_id, EventKind.VALUE_ESTIMATE)
        if reward_event.payload["target_event_id"] != value_event.payload["target_event_id"]:
            raise ValueError("reward and value estimate must bind the same target")
        transition = self._td_transitions.get(value_event.event_id)
        if transition is None:
            raise ValueError("value estimate is not available for this process-local TD table")
        reward = reward_event.payload["normalized_value"]
        prior = value_event.payload["value"]
        next_value = 0.0 if transition.terminal else self.value_table.estimate(
            self.session_id, transition.next_state_key).value
        with self.value_table.transaction(), self.event_store.transaction():
            self._assert_td_current()
            if any(event.kind == EventKind.RPE_UPDATE
                   and event.payload["reward_event_id"] == reward_event.event_id
                   for event in self.event_store.list(self.session_id)):
                raise ValueError("an RPE update already exists for this reward")
            update = self.value_table.apply_reward(RewardObservation(
                reward_id=reward_event.payload["reward_id"], transition_id=transition.transition_id,
                context_id=self.session_id, reward=reward,
                source=RewardSource(reward_event.source_kind.value),
                source_ref=reward_event.source_ref,
            ))
            persisted = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.RPE_UPDATE,
                source_kind=SourceKind.SYSTEM, source_ref="ValueTable",
                payload={"update_id": "rpe_%s" % uuid4().hex,
                         "transition_id": transition.transition_id,
                         "reward_event_id": reward_event.event_id,
                         "prior_value_event_id": value_event.event_id,
                         "state_key": value_event.payload["state_key"],
                         "action_key": value_event.payload["action_key"],
                         "reward": reward, "prior_value": prior, "next_value": next_value,
                         "alpha": self.value_table.config.alpha,
                         "gamma": self.value_table.config.gamma,
                         "raw_delta": update.raw_delta, "clipped_delta": update.clipped_delta,
                         "clip": self.value_table.config.clip,
                         "updated_value": update.updated_value, "formula_version": "td0_v1",
                         "scope": "research_ranking_only"},
                parent_event_ids=(reward_event.event_id, value_event.event_id),
            ), commit=False)
        self._td_replay_head_sequence = persisted.sequence or self._td_replay_head_sequence
        return persisted

    def rank_safe_actions(self, candidates: Iterable[str]) -> Tuple[SafeActionPreference, ...]:
        """Return research-only rankings; callers still require PolicyGate approval."""
        self._assert_td_current()
        return self.value_table.rank_action_classes(self.session_id, candidates)

    def approve_seed(self, seed_id: str) -> SeedDisposition:
        """Record explicit user approval before activating a candidate seed."""
        proposal_event_id = self.seed_store.proposal_event_id(seed_id, self.session_id)
        approval = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            payload={"approval": "seed", "seed_id": seed_id,
                     "proposal_event_id": proposal_event_id},
        ))
        return self.seed_store.approve(seed_id, approval.event_id)

    def enable_seed_auto_update(self, user_event_id: str,
                                policy: SeedStandingPolicy, policy_id: str,
                                nonce: str, issued_at: str) -> Dict[str, Any]:
        """Consume one explicit CLI/user standing authorization.

        The caller must have already appended the exact USER approval event.
        This method never manufactures user authority and gives no model,
        reward, tool, quota, sleep, or stop-control privilege.
        """
        self.seed_store.issue_standing_policy(
            self.session_id, user_event_id, policy, nonce=nonce,
            policy_id=policy_id, issued_at=issued_at,
            source_ref="cli_user_command")
        return self.seed_auto_update_status()

    def disable_seed_auto_update(self, user_event_id: str,
                                 reason: str = "user_requested") -> Dict[str, Any]:
        """Consume an exact USER revocation for the latest active policy."""
        current = self.seed_store.standing_policy_status(self.session_id)
        if current is None or current.get("status") != "active":
            raise RuntimeError("seed auto-update has no active standing policy")
        self.seed_store.revoke_standing_policy(
            self.session_id, current["policy_id"], user_event_id,
            reason=reason, source_ref="cli_user_command")
        return self.seed_auto_update_status()

    def seed_auto_update_status(self) -> Dict[str, Any]:
        """Return aggregate standing-policy state without seed or authority data."""
        current = self.seed_store.standing_policy_status(self.session_id)
        active_count = sum(
            seed.status.value == "active"
            for seed in self.seed_store.list(self.session_id)
        )
        if current is None:
            return {
                "configured": False, "state": "not_configured",
                "enabled": False, "active_seed_count": active_count,
                "auto_activations": 0, "auto_updates": 0,
                "per_seed_confirmation_required": True,
                "authority": "explicit_user_standing_policy_only",
            }
        state = current["status"]
        return {
            "configured": True, "state": state, "enabled": state == "active",
            "expires_at": current["expires_at"],
            "active_seed_count": active_count,
            "auto_activations": current["auto_activations_used"],
            "auto_updates": current["auto_updates_used"],
            "max_auto_activations": current["max_auto_activations"],
            "max_auto_updates": current["max_auto_updates"],
            "max_active_seeds": current["max_active_seeds"],
            "per_seed_confirmation_required": state != "active",
            "authority": "explicit_user_standing_policy_only",
            "isolation": "no_reward_td_capability_quota_sleep_or_stop_authority",
        }

    def _seed_guidance_for(self, seeds: Sequence[SeedDisposition]) -> Tuple[SeedGuidance, ...]:
        """Return at most two host-derived, content-free active seed cues.

        This is a projection of current ledger state, never a model memory
        proposal.  It deliberately does not expose cue terms, user text,
        confidence, strength, provenance, or policy nonces to K3.
        """
        active = [seed for seed in seeds if seed.status == SeedStatus.ACTIVE]
        if not active:
            return ()
        events = self.event_store.list(self.session_id)
        result = []
        for seed in active[:2]:
            digest = sha256(json.dumps(self._seed_export(seed), ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            authority_id = None
            for event in reversed(events):
                if event.kind == EventKind.SEED_AUTO_APPLIED:
                    if (event.payload.get("seed_id") == seed.seed_id
                            and event.payload.get("operation") != "retire"
                            and event.payload.get("new_seed_digest") == digest):
                        authority_id = event.event_id
                        break
                elif event.kind == EventKind.SEED_APPROVED:
                    if event.payload.get("seed_id") == seed.seed_id:
                        authority_id = event.event_id
                        break
            # A replayed active seed without a matching durable authority is
            # not guidance.  Fail closed rather than guessing its lineage.
            if authority_id is None:
                continue
            result.append(SeedGuidance(
                seed_id=seed.seed_id, current_authority_event_id=authority_id,
                snapshot_digest=digest,
                priority_band="primary" if not result else "secondary"))
        return tuple(result)

    def _expedition_seed_guidance(self) -> Tuple[SeedGuidance, ...]:
        """Project active guidance relevant to the explicit expedition goal."""
        if self._expedition_goal is None:
            return ()
        matches = self.seed_store.retrieve(
            self.session_id, self._seed_cue_terms(self._expedition_goal), scope="conversation")
        return self._seed_guidance_for(matches)

    def _record_expedition_seed_context(self, run_id: str,
                                        consumption_event_id: str,
                                        guidance: Sequence[SeedGuidance]) -> CognitiveEvent:
        """Append the redacted, non-authoritative expedition guidance edge."""
        authorities = [{"seed_id": item.seed_id,
                        "authority_event_id": item.current_authority_event_id,
                        "snapshot_digest": item.snapshot_digest,
                        "priority_band": item.priority_band}
                       for item in guidance]
        # Use the same host clock that established the expedition authorization
        # window.  Tests and embedders may inject that clock; mixing it with
        # wall time would fabricate an out-of-window context edge.
        created_at = self._sleep_now().isoformat()
        payload = {"context_id": "seedctx_%s" % uuid4().hex, "run_id": run_id,
                   "directive": "request_human_review", "seed_authorities": authorities,
                   "context_digest": "0" * 64, "created_at": created_at,
                   "version": "expedition_seed_context_v1"}
        payload["context_digest"] = SQLiteEventStore.expedition_seed_context_digest(payload)
        return self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.EXPEDITION_SEED_CONTEXT,
            source_kind=SourceKind.POLICY, source_ref="SeedGuidanceProjector",
            payload=payload, created_at=created_at,
            parent_event_ids=(consumption_event_id,) + tuple(
                item.current_authority_event_id for item in guidance)))

    def purge_session(self, confirmed: bool = False) -> Dict[str, Any]:
        """Delete logical records; direct file stores are never physical-purge claims."""
        if not confirmed:
            raise PermissionError("purge requires explicit confirmation")
        self.stop_research_autonomy("purge")
        if self._expedition is not None:
            self._expedition.stop("purge")
        self._expedition_active_transition = None
        self._expedition_learner = None
        self.stop_sleep()
        if self.quota_controller is not None:
            self.quota_controller.invalidate_reservations(ReservationInvalidationReason.PURGE)
        report = self.event_store.purge_session(self.session_id)
        # The logical ledger is gone, so its ephemeral learner projection must
        # be gone too. Keeping it would make a newly empty session inherit a
        # reward history that is no longer inspectable or revocable.
        self.value_table = ValueTable()
        self._td_transitions = {}
        self._td_replay_head_sequence = 0
        self._mirror_count = 0
        self._mirror_status = "ready"
        if self.event_store.path != ":memory:":
            report = dict(report)
            report["logical_only"] = True
            report["physical_purge_claimed"] = False
        return report

    def retire_seed(self, seed_id: str) -> SeedDisposition:
        """Retire a seed with an explicit user-attributed event."""
        return self.seed_store.retire(seed_id, source_kind=SourceKind.USER, source_ref="user")

    def revoke_claim(self, claim_id: str) -> None:
        """Revoke a self-model claim with an explicit user-attributed event."""
        self.self_model.revoke(self.session_id, claim_id, source_kind=SourceKind.USER, source_ref="user")

    def record_correction(self, target_event_id: str, counterevidence_event_id: str,
                          source_kind: SourceKind = SourceKind.USER,
                          source_ref: str = "user") -> CognitiveEvent:
        """Record a bounded conflict between two prior observable records.

        A correction always preserves history and requests review.  When it is
        USER-sourced and a standing seed policy is active, the host may also
        tighten or retire a causally related seed within that policy's limits.
        Other sources only create the correction record.  Self-model claims
        retain their separate explicit revocation flow.
        """
        permitted_sources = {
            SourceKind.USER, SourceKind.TOOL, SourceKind.EXTERNAL_VERIFIER,
        }
        if source_kind not in permitted_sources:
            raise ValueError("corrections require a user, tool, or external verifier source")
        if target_event_id == counterevidence_event_id:
            raise ValueError("correction target and counterevidence must differ")
        target = self.event_store.get(target_event_id)
        counterevidence = self.event_store.get(counterevidence_event_id)
        if target is None or counterevidence is None:
            raise ValueError("correction events must already exist")
        if target.session_id != self.session_id or counterevidence.session_id != self.session_id:
            raise ValueError("correction events must belong to the current session")
        if target.sequence is None or counterevidence.sequence is None:
            raise ValueError("correction events must be persisted")
        if target.sequence >= counterevidence.sequence:
            raise ValueError("counterevidence must be later than the corrected event")
        allowed_counterevidence = {
            EventKind.OBSERVATION, EventKind.TOOL_RESULT, EventKind.ACTION_RESULT,
        }
        if counterevidence.kind not in allowed_counterevidence:
            raise ValueError("counterevidence must be an observation or action result")
        correction = self.event_store.append(CognitiveEvent(
            session_id=self.session_id,
            kind=EventKind.CORRECTION,
            source_kind=source_kind,
            source_ref=source_ref,
            payload={
                "target_event_id": target_event_id,
                "counterevidence_event_id": counterevidence_event_id,
                "disposition": "review_required",
                "public_summary": "A later observable record conflicts with an earlier record.",
            },
            parent_event_ids=(target_event_id, counterevidence_event_id),
        ))
        self._auto_process_seed_correction(correction, target_event_id)
        return correction

    def export_session(self) -> Dict[str, Any]:
        """Export inspectable, current logical-session records."""
        events = self.event_store.list(self.session_id)
        graph = self._refresh_memory_graph()
        return {
            "session_id": self.session_id,
            "events": [event.to_dict() for event in events],
            "records_by_category": self._records_by_category(events),
            "seeds": [self._seed_export(seed) for seed in self.seed_store.list(self.session_id)],
            "current_claims": [self._claim_export(claim)
                               for claim in self.self_model.current_claims(self.session_id)],
            "memory_graph": graph.to_dict(),
            "sleep": self.sleep_status(),
        }

    def memory_graph_status(self) -> Dict[str, Any]:
        """Return a refreshed, metadata-only graph projection status."""
        return self._refresh_memory_graph().to_dict()

    def memory_graph_public_status(self) -> Dict[str, Any]:
        """Return the caller-thread refreshed graph summary without SQLite I/O.

        ``CognitiveMonitor`` calls this from an HTTP request thread.  Returning
        a copy prevents that monitoring path from becoming an authority or a
        writer for this rebuildable projection.
        """
        return dict(self._memory_graph_public_status)

    def _refresh_memory_graph(self):
        """Refresh the non-authoritative graph without affecting agent control.

        A graph failure is observable as a bounded status, but can never block
        quota, grants, seed policy, reward, sleep, or the current expedition.
        """
        try:
            graph = self.memory_graph.ensure_current(self.session_id)
            self._memory_graph_public_status = graph.to_dict()
            return graph
        except Exception:
            # Do not disclose database paths or exception text in public
            # monitoring data.  The ledger remains authoritative and intact.
            cached = dict(getattr(self, "_memory_graph_public_status", {}))
            cached["status"] = "unavailable"
            self._memory_graph_public_status = cached
            raise

    def _expedition_status_after_graph_refresh(self) -> Dict[str, Any]:
        """Publish graph progress at a completed expedition slice boundary."""
        try:
            self._refresh_memory_graph()
        except Exception:
            # Graph projection is deliberately fail-open relative to the
            # bounded research state machine; its cached status remains public.
            pass
        return self.expedition_status()

    def explain_memory_event(self, event_id: str) -> Dict[str, Any]:
        """Explain one event's bounded provenance neighborhood without payloads."""
        graph = self._refresh_memory_graph()
        return {"graph": graph.to_dict(), "event_id": event_id,
                "neighbors": self.memory_graph.neighbors(self.session_id, event_id)}

    def state(self) -> dict:
        """Return inspectable current state without implying an inner experience."""
        return {
            "session_id": self.session_id,
            "claims": [claim.statement for claim in self.self_model.current_claims(self.session_id)],
            "active_seed_ids": [seed.seed_id for seed in self.seed_store.list(self.session_id)
                                if seed.status.value == "active"],
            "seed_auto_update": self.seed_auto_update_status(),
            "memory_graph": self.memory_graph_status(),
            "event_count": len(self.event_store.list(self.session_id)),
            "loop": self.loop_status(),
            "provider": self.provider_status(),
            "quota": self.quota_status(),
            "tools": {"configured": self.tool_session is not None,
                      "grants": list(self.tool_session.grant_snapshots()) if self.tool_session else [],
                      "execution": "user_grants_and_per_call_confirmation_required"},
            "unattended_research": self.unattended_status(),
            "drives": self.drive_status(),
            "improvement": self.improvement_status(),
            "sleep": self.sleep_status(),
            "metacognition": {
                "mirror_status": self._mirror_status,
                "mirror_count": self._mirror_count,
                "method": "mirror_v1",
                "limit": "one deterministic public audit per episode",
            },
        }

    def provider_status(self) -> Dict[str, Any]:
        """Expose configured egress and fallback boundaries without resolving a key."""
        return {
            "provider": "kimi-code-k3",
            "model": self.runtime.settings.model if self.runtime is not None else None,
            "configured": self.runtime is not None,
            "egress": "https://api.kimi.com/coding/v1/chat/completions",
            "credential_source": "macOS Keychain service=moonshot account=strangeloop-kimi-api",
            "fallback": "transparent_heuristic_on_credential_network_or_parse_failure",
            "audio": "metadata_only_k3_has_no_native_audio_input",
            "tools": "controlled_host_tools" if self.tool_session is not None else "not_enabled",
            "unattended_research": "explicit_user_read_only_profile_only",
        }

    def quota_status(self) -> Dict[str, Any]:
        """Return a bounded public view of K3 capacity control.

        This is intentionally distinct from a provider response: it contains
        neither credentials nor raw usage data, and the local call/token
        ledger is labelled as local observation rather than a Code Plan
        balance.  ``QuotaController`` owns the lock around both telemetry and
        its decision, so this accessor is safe for the monitor request thread.
        """
        controller = self.quota_controller
        if controller is None:
            return {"configured": False, "authority": "unknown", "freshness": "unknown",
                    "allow_call": False, "reason": "not_configured",
                    "local_observed_calls": 0, "local_observed_tokens": 0,
                    "is_code_plan_balance": False}
        telemetry = controller.export_telemetry()
        decision = controller.decision()
        source = telemetry.get("source")
        snapshot = telemetry.get("snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else None
        provider_authoritative = source in ("provider_usage", "authoritative_header")
        freshness = "unknown"
        if snapshot is not None:
            if not provider_authoritative:
                freshness = "unverified"
            elif snapshot.get("is_estimate") is True:
                freshness = "estimated"
            elif decision.reason == "quota_telemetry_stale":
                freshness = "stale"
            elif decision.reason == "quota_reset_requires_fresh_telemetry":
                freshness = "reset_due"
            elif decision.reason in ("provider_quota_exhausted", "provider_rate_limit_cooldown"):
                freshness = "paused"
            else:
                freshness = "fresh"
        result = {"configured": True,
                  "authority": source if source in ("provider_usage", "authoritative_header",
                                                      "manual_snapshot", "error_signal", "local_ledger")
                  else "unknown",
                  "freshness": freshness,
                  "telemetry_known": snapshot is not None,
                  "local_observed_calls": telemetry.get("ledger_calls", 0),
                  "local_observed_tokens": telemetry.get("ledger_tokens", 0),
                  "allow_call": bool(decision.allow_call), "reason": decision.reason,
                  "is_code_plan_balance": bool(provider_authoritative and snapshot is not None
                                               and snapshot.get("primary_unit") == "provider_units"
                                               and snapshot.get("is_estimate") is False)}
        if snapshot is not None:
            # These fields come from QuotaSnapshot validation, never a raw
            # provider body.  Keep optional dimensions out of this primary
            # K3 Plan card rather than implying they are comparable units.
            for key in ("primary_unit", "remaining", "total", "reset_at", "observed_at"):
                if key in snapshot:
                    result[key] = snapshot[key]
            # The named primary window is useful operational context, but it
            # is meaningful only for an authoritative provider-unit balance.
            # Do not project it for manual/local/error telemetry or estimates.
            window_kind = snapshot.get("primary_window_kind")
            if (provider_authoritative and snapshot.get("primary_unit") == "provider_units"
                    and snapshot.get("is_estimate") is False
                    and window_kind in ("weekly", "rolling_5h")):
                result["primary_window_kind"] = window_kind
        return result

    def drive_status(self) -> Dict[str, Any]:
        if self.drives is None:
            return {"configured": False}
        return {"configured": True, "channels": self.drives.values(),
                "safe_ranks": [item.__dict__ for item in self.drives.rank_safe_research_actions()],
                "limit": "research-only; cannot grant capabilities or change policy"}

    def improvement_status(self) -> Dict[str, Any]:
        control = self.improvement_control
        if control is None:
            return {"configured": False}
        states = getattr(control, "_states", {})
        return {"configured": True, "protected": True,
                "restart_suspended": bool(control.restart_suspended),
                "candidate_count": len(states),
                "states": dict((key, value.value) for key, value in states.items()),
                "automatic_promotion": False}

    def _loop_cycle(self, context: TickContext) -> CycleResult:
        """Select a bounded external focus; this callback makes no tool call.

        It intentionally creates no event itself.  The caller persists exactly
        one fixed-schema tick after the controller has applied its budgets.
        """
        focus_id = None
        salience_reason = "scheduled_review"
        if context.salience:
            focus_id = context.salience.get("event_id")
            salience_reason = context.salience.get("reason", "external_salience")
        if focus_id is None or focus_id in self._processed_focus_event_ids:
            for event in reversed(self.event_store.list(self.session_id)):
                if event.kind in (EventKind.OBSERVATION, EventKind.MEDIA_OBSERVATION,
                                  EventKind.PERCEPT, EventKind.TOOL_RESULT,
                                  EventKind.ACTION_RESULT, EventKind.CORRECTION) and event.event_id not in self._processed_focus_event_ids:
                    focus_id = event.event_id
                    break
        new_focus = isinstance(focus_id, str) and focus_id not in self._processed_focus_event_ids
        if new_focus:
            self._processed_focus_event_ids.add(focus_id)
        focus = [focus_id] if new_focus else []
        summary = ("A bounded external record was selected for review."
                   if new_focus else "No unreviewed external record is available.")
        made_progress = new_focus
        if new_focus and self._loop_reflector is not None and self._loop_remote_calls_remaining > 0 and not self._quota_denied():
            self._loop_remote_calls_remaining -= 1
            try:
                reflection = self._loop_reflector.reflect(context)
                # The reflector can annotate this bounded review, but cannot
                # create actions, alter focus, or increase controller budgets.
                summary = reflection.summary
                made_progress = reflection.made_progress
            except KimiCodeProviderError:
                # Fixed public fallback: do not expose provider errors or any
                # request/response material in the autonomous event stream.
                summary = "A bounded external record was selected for review."
                made_progress = new_focus
        if new_focus and self._quota_denied():
            summary = "K3 reflection is paused by the host quota controller; no provider call was made."
        return CycleResult(event_count=1 if new_focus else 0, made_progress=made_progress, payload={
            "focus_event_ids": focus,
            "retrieved_event_ids": [],
            "progress": "made_progress" if made_progress else "no_progress",
            "salience_reason": salience_reason,
            "trigger": self._store_trigger(context.trigger, salience_reason),
            "phase": "quota_paused" if new_focus and self._quota_denied() else ("reflect" if new_focus else "idle"),
            "public_summary": summary,
        })

    def _deliberate_with_fallback(self, user_text: str, workspace: WorkspaceFrame,
                                  observation_event_id: str) -> Tuple[Deliberation, Optional[str]]:
        """Call K3 once when selected, with a public, deterministic fallback."""
        if not self._uses_kimi_deliberator:
            return self.deliberator.deliberate(user_text, workspace), None
        if self._quota_denied() or (self.sleep_coordinator is not None
                                    and self.sleep_coordinator.state == SleepState.SLEEPING):
            return self._heuristic_deliberator.deliberate(user_text, workspace), (
                "K3 deliberation is paused by the host quota controller or sleep control; used the local transparent fallback."
            )
        invocation_id = "inv_%s" % uuid4().hex
        started = time.monotonic()
        self._append_model_invocation(
            invocation_id + ".start", None, observation_event_id, "started", 0,
            "Kimi Code deliberation request started.", ()
        )
        try:
            result = self.deliberator.deliberate(user_text, workspace)
        except KimiCodeProviderError as error:
            elapsed = int((time.monotonic() - started) * 1000)
            self._append_model_invocation(
                invocation_id + ".finish", None, observation_event_id,
                "timed_out" if error.category == "provider_timeout" else "failed", elapsed,
                ("K3 deliberation timed out; no tool was executed."
                 if error.category == "provider_timeout"
                 else "Kimi Code deliberation unavailable; used the local transparent fallback."),
                ()
            )
            return self._heuristic_deliberator.deliberate(user_text, workspace), (
                ("K3 deliberation timed out; no tool was executed; used the local transparent fallback."
                 if error.category == "provider_timeout"
                 else "Kimi Code was unavailable; used the local transparent fallback.")
            )
        except Exception:
            elapsed = int((time.monotonic() - started) * 1000)
            self._append_model_invocation(
                invocation_id + ".finish", None, observation_event_id, "failed", elapsed,
                "Kimi Code deliberation unavailable; used the local transparent fallback.", ()
            )
            return self._heuristic_deliberator.deliberate(user_text, workspace), (
                "Kimi Code was unavailable; used the local transparent fallback."
            )
        elapsed = int((time.monotonic() - started) * 1000)
        self._append_model_invocation(
            invocation_id + ".finish", None, observation_event_id, "completed", elapsed,
            "Kimi Code deliberation returned a bounded public result.",
            ()
        )
        return result, None

    def _append_model_invocation(self, invocation_id: str, run_id: str,
                                 trigger_event_id: str, outcome: str, latency_ms: int,
                                 public_summary: str, parent_event_ids: Tuple[str, ...],
                                 role: str = "deliberation", context_scope: str = "text_turn") -> CognitiveEvent:
        """Persist metadata only: no prompts, outputs, keys, reasoning, or media bytes."""
        return self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.MODEL_INVOCATION,
            source_kind=SourceKind.SYSTEM, source_ref="KimiCodeRuntime",
            payload={"invocation_id": invocation_id, "run_id": run_id,
                     "trigger_event_id": trigger_event_id, "provider": "kimi-code-k3",
                     "model": self.runtime.settings.model, "role": role,
                     "outcome": outcome, "latency_ms": max(0, int(latency_ms)),
                     "context_scope": context_scope, "public_summary": public_summary},
            parent_event_ids=parent_event_ids or (trigger_event_id,),
        ))

    def _quota_denied(self) -> bool:
        return self.quota_controller is not None and not self.quota_controller.decision().allow_call

    def _sleep_now(self):
        # Ledger timestamps must never precede records appended immediately
        # before them, even when deterministic tests inject an older clock.
        wall = datetime.now(timezone.utc)
        if self.sleep_coordinator is not None:
            return max(wall, self.sleep_coordinator._now())
        return wall

    def _close_loop_for_sleep(self) -> None:
        if self._loop is not None and self._loop.state not in (LoopState.STOPPED, LoopState.EXHAUSTED):
            control = self._append_loop_control("stop")
            self._loop_parent_event_id = control.event_id
            self._loop.stop()
            self._append_loop_stopped("quota_sleep", control.event_id)

    def _start_wake_loop(self, awake_event: CognitiveEvent) -> None:
        """Start one small, system-attributed foreground-only post-wake run."""
        config = LoopConfig(max_ticks=4, max_wall_seconds=30, max_events_per_tick=1,
                            max_no_progress=2)
        self._loop_run_id = "run_%s" % uuid4().hex
        self._loop = FunctionalLoopController(self._loop_cycle, config=config)
        self._loop_remote_calls_remaining = 0  # post-wake loop never invokes K3/tools
        self._loop_stop_recorded = False
        self._processed_focus_event_ids = set()
        control = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
            payload={"run_id": self._loop_run_id, "action": "start", "config_version": "loop_config_v1",
                "max_ticks": 4, "max_wall_seconds": 30, "max_events_per_tick": 1,
                "max_no_progress": 2, "min_interval_seconds": 0.0, "next_tick_index": 0},
            parent_event_ids=(awake_event.event_id,)))
        self._loop_parent_event_id = control.event_id
        self._loop.resume()
        # Remains foreground synchronous; no tool path is called by _loop_cycle.
        self.run_loop()

    def _append_provider_usage_evidence(self, snapshot: Any, rolling_window: Any) -> CognitiveEvent:
        if self._sleep_event_id is None:
            raise RuntimeError("sleep lifecycle is unavailable")
        source = "provider_usage"
        payload = {"usage_evidence_id": "usage_%s" % uuid4().hex, "provider": "kimi_code",
                   "quota_source": source, "window_kind": "rolling_5h",
                   "observed_at": snapshot.observed_at.isoformat(),
                   "reset_at": rolling_window.reset_at.isoformat(), "total": rolling_window.total,
                   "remaining": rolling_window.remaining, "evidence_version": "provider_usage_v2"}
        payload["evidence_digest"] = SQLiteEventStore._provider_usage_digest_v2(payload)
        return self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.PROVIDER_USAGE_EVIDENCE,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier",
            payload=payload,
            parent_event_ids=(self._sleep_event_id,)))

    def _append_wake_check(self, usage_event: CognitiveEvent, checked_at: datetime) -> CognitiveEvent:
        if self._sleep_archive_event_id is None or self._sleep_event_id is None or self._sleep_epoch_id is None:
            raise RuntimeError("sleep lifecycle is unavailable")
        return self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.WAKE_CHECK, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", payload={"wake_check_id": "check_%s" % uuid4().hex,
                "archive_event_id": self._sleep_archive_event_id, "sleep_event_id": self._sleep_event_id,
                "epoch_id": self._sleep_epoch_id, "usage_evidence_event_id": usage_event.event_id,
                "checked_at": checked_at.isoformat(), "check_version": "wake_check_v1"},
            parent_event_ids=(self._sleep_event_id, usage_event.event_id)))

    def _persist_tick(self, record: TickRecord) -> Optional[CognitiveEvent]:
        if not record.payload or self._loop_run_id is None or self._loop_parent_event_id is None:
            return None
        payload = dict(record.payload)
        event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.LOOP_TICK,
            source_kind=SourceKind.SYSTEM, source_ref="FunctionalLoopController",
            payload={"run_id": self._loop_run_id, "tick_index": record.tick_number - 1,
                     "trigger": payload["trigger"], "phase": payload["phase"],
                     "focus_event_ids": list(payload["focus_event_ids"]),
                     "retrieved_event_ids": list(payload["retrieved_event_ids"]),
                     "progress": payload["progress"],
                     "salience_reason": payload["salience_reason"],
                     "budget_remaining": max(0, self._require_loop().config.max_events_per_tick - record.event_count),
                     "public_summary": payload["public_summary"]},
            parent_event_ids=(self._loop_parent_event_id,),
        ))
        self._loop_parent_event_id = event.event_id
        focus_event_ids = payload["focus_event_ids"]
        if focus_event_ids:
            target = self.event_store.get(focus_event_ids[0])
            if target is None:
                self._mirror_status = "needs_external_review"
            else:
                self._append_mirror(
                    episode_id="loop_%s_%d" % (self._loop_run_id, record.tick_number - 1),
                    target_event=target,
                    judgment_event=event,
                    evidence_events=(target, event),
                )
        if record.state_after in (LoopState.EXHAUSTED, LoopState.STOPPED):
            self._append_loop_stopped(self._stop_reason_for_record(record), event.event_id)
        return event

    def _append_mirror(self, episode_id: str, target_event: CognitiveEvent,
                       judgment_event: CognitiveEvent,
                       evidence_events: Tuple[CognitiveEvent, ...]) -> Optional[CognitiveEvent]:
        """Append one bounded two-sided public audit, or request outside review.

        This method is intentionally best-effort: a malformed public record
        must not re-run a controller callback, consume a loop budget, or cause
        a model/tool/drive/memory side effect.  The compact status is the only
        failure signal, keeping the event ledger free of private diagnostics.
        """
        try:
            rows = tuple(self._mirror_evidence(event) for event in evidence_events)
            record = self._mirror_auditor.assess(
                episode_id, target_event.event_id, judgment_event.event_id, rows
            )
            event = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.METACOGNITIVE_MIRROR,
                source_kind=SourceKind.SYSTEM, source_ref="MirrorAuditor",
                payload=record.to_payload(), confidence=record.meta_confidence_cap,
                parent_event_ids=(target_event.event_id, judgment_event.event_id),
            ))
        except Exception:
            self._mirror_status = "needs_external_review"
            return None
        self._mirror_status = "completed"
        self._mirror_count += 1
        return event

    @staticmethod
    def _mirror_evidence(event: CognitiveEvent) -> PublicEvidence:
        """Project an event into the closed mirror evidence vocabulary."""
        kinds = {
            EventKind.OBSERVATION: "observation",
            EventKind.MEDIA_OBSERVATION: "observation",
            EventKind.PERCEPT: "percept",
            EventKind.TOOL_RESULT: "tool_result",
            EventKind.ACTION_RESULT: "action_result",
            EventKind.CORRECTION: "correction",
            EventKind.LOOP_TICK: "loop_tick",
        }
        event_kind = kinds.get(event.kind)
        if event_kind is None:
            raise ValueError("event kind cannot be used as mirror evidence")
        return PublicEvidence(
            event_id=event.event_id, event_kind=event_kind,
            source_kind=event.source_kind, polarity=EvidencePolarity.SUPPORTS,
            confidence=event.confidence,
        )

    def _append_loop_control(self, action: str) -> CognitiveEvent:
        loop = self._require_loop()
        if self._loop_run_id is None:
            raise RuntimeError("loop run id is unavailable")
        config = loop.config
        return self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.USER, source_ref="user",
            payload={"run_id": self._loop_run_id, "action": action,
                     "config_version": "loop_config_v1", "max_ticks": config.max_ticks,
                     "max_wall_seconds": int(config.max_wall_seconds),
                     "max_events_per_tick": config.max_events_per_tick,
                     "max_no_progress": config.max_no_progress,
                     "min_interval_seconds": float(config.min_interval),
                     "next_tick_index": loop.ticks_completed},
        ))

    def _append_loop_stopped(self, reason: str, parent_event_id: str) -> Optional[CognitiveEvent]:
        if self._loop_stop_recorded or self._loop_run_id is None or self._loop is None:
            return None
        self._loop_stop_recorded = True
        event = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.AUTONOMY_STOPPED,
            source_kind=SourceKind.SYSTEM, source_ref="FunctionalLoopController",
            payload={"run_id": self._loop_run_id, "reason": reason,
                     "tick_count": self._loop.ticks_completed,
                     "stopped_at": self._utc_now()}, parent_event_ids=(parent_event_id,),
        ))
        self._loop_parent_event_id = event.event_id
        return event

    def _signal_external_salience(self, event_id: str, reason: str) -> None:
        if self._loop is not None and self._loop.state in (LoopState.RUNNING, LoopState.IDLE, LoopState.PAUSED):
            self._loop.signal_salience({"event_id": event_id, "reason": reason})

    def _require_loop(self) -> FunctionalLoopController:
        if self._loop is None:
            raise RuntimeError("no active loop run; call start_loop first")
        return self._loop

    @staticmethod
    def _validate_persisted_loop_config(config: LoopConfig) -> None:
        """Reject controller values that cannot be represented by the event protocol."""
        if (not isinstance(config.max_ticks, int) or isinstance(config.max_ticks, bool)
                or not 1 <= config.max_ticks <= 10000):
            raise ValueError("max_ticks must be an integer between 1 and 10000")
        if (not isinstance(config.max_wall_seconds, (int, float)) or isinstance(config.max_wall_seconds, bool)
                or not math.isfinite(float(config.max_wall_seconds))
                or not float(config.max_wall_seconds).is_integer()
                or not 1 <= config.max_wall_seconds <= 86400):
            raise ValueError("max_wall_seconds must be whole seconds between 1 and 86400")
        if (not isinstance(config.max_events_per_tick, int) or isinstance(config.max_events_per_tick, bool)
                or not 1 <= config.max_events_per_tick <= 256):
            raise ValueError("max_events_per_tick must be an integer between 1 and 256")
        if (not isinstance(config.max_no_progress, int) or isinstance(config.max_no_progress, bool)
                or not 0 <= config.max_no_progress <= 10000):
            raise ValueError("max_no_progress must be an integer between 0 and 10000")
        if (not isinstance(config.min_interval, (int, float)) or isinstance(config.min_interval, bool)
                or not math.isfinite(float(config.min_interval))
                or not 0.0 <= float(config.min_interval) <= 3600.0):
            raise ValueError("min_interval must be finite and between 0 and 3600 seconds")

    def _rewardable_target(self, event_id: str) -> CognitiveEvent:
        event = self._session_event(event_id)
        if event.kind not in (EventKind.ACTION_RESULT, EventKind.LOOP_TICK):
            raise ValueError("value and reward targets must be an action result or loop tick")
        return event

    def _restore_td_history(self, table_supplied: bool) -> None:
        """Rebuild the single live in-memory learner from auditable TD events."""
        if not self.event_store.verify_chain(self.session_id):
            raise ValueError("cannot replay TD history from an invalid event chain")
        events = self.event_store.list(self.session_id)
        values = [event for event in events if event.kind == EventKind.VALUE_ESTIMATE]
        if not values:
            if table_supplied and self.value_table.event_log():
                raise ValueError("a supplied ValueTable must be empty without persisted TD history")
            return
        first = values[0].payload
        pinned = TDConfig(alpha=first["alpha"], gamma=first["gamma"], clip=first["clip"])
        pinned_limits = (first["max_entries"], first["max_events"])
        if table_supplied:
            if (self.value_table.event_log() or self.value_table.config != pinned
                    or (self.value_table.max_entries, self.value_table.max_events) != pinned_limits):
                raise ValueError("a supplied ValueTable must be empty and match persisted TD config and limits")
        else:
            self.value_table = ValueTable(config=pinned, max_entries=pinned_limits[0],
                                          max_events=pinned_limits[1])
        rewards = {event.event_id: event for event in events
                   if event.kind == EventKind.REWARD_OBSERVATION}
        value_by_id = {}
        for event in events:
            if event.kind == EventKind.VALUE_ESTIMATE:
                payload = event.payload
                if (payload["alpha"], payload["gamma"], payload["clip"]) != (
                        pinned.alpha, pinned.gamma, pinned.clip):
                    raise ValueError("persisted TD config changes within one session")
                if (payload["max_entries"], payload["max_events"]) != pinned_limits:
                    raise ValueError("persisted TD limits change within one session")
                current = self.value_table.estimate(self.session_id, payload["state_key"]).value
                if not math.isclose(current, payload["value"], rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("persisted value estimate does not match deterministic replay")
                transition = Transition(
                    transition_id=payload["transition_id"], context_id=self.session_id,
                    state_key=payload["state_key"], action_class=payload["action_key"],
                    next_state_key=payload["next_state_key"], terminal=payload["terminal"])
                self.value_table.register_transition(transition)
                self._td_transitions[event.event_id] = transition
                value_by_id[event.event_id] = event
            elif event.kind == EventKind.RPE_UPDATE:
                payload = event.payload
                reward_event = rewards.get(payload["reward_event_id"])
                value_event = value_by_id.get(payload["prior_value_event_id"])
                if reward_event is None or value_event is None:
                    raise ValueError("RPE replay references unavailable prior TD records")
                transition = self._td_transitions[value_event.event_id]
                update = self.value_table.apply_reward(RewardObservation(
                    reward_id=reward_event.payload["reward_id"], transition_id=transition.transition_id,
                    context_id=self.session_id, reward=reward_event.payload["normalized_value"],
                    source=RewardSource(reward_event.source_kind.value), source_ref=reward_event.source_ref))
                for field, actual in (("raw_delta", update.raw_delta),
                                      ("next_value", update.next_value),
                                      ("clipped_delta", update.clipped_delta),
                                      ("updated_value", update.updated_value)):
                    if not math.isclose(payload[field], actual, rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError("persisted RPE update does not match deterministic replay")
        self._td_replay_head_sequence = max(event.sequence or 0 for event in events
                                            if event.kind in (EventKind.VALUE_ESTIMATE,
                                                              EventKind.REWARD_OBSERVATION,
                                                              EventKind.RPE_UPDATE))

    def _assert_td_current(self) -> None:
        """Fail closed if a second live agent changed this session's TD ledger."""
        head = max((event.sequence or 0 for event in self.event_store.list(self.session_id)
                    if event.kind in (EventKind.VALUE_ESTIMATE, EventKind.REWARD_OBSERVATION,
                                      EventKind.RPE_UPDATE)), default=0)
        if head != self._td_replay_head_sequence:
            raise RuntimeError("TD ledger changed; reopen the agent to replay and synchronize")

    def _session_event(self, event_id: str, expected_kind: Optional[EventKind] = None) -> CognitiveEvent:
        event = self.event_store.get(event_id)
        if event is None or event.session_id != self.session_id:
            raise ValueError("event must exist in the current session")
        if expected_kind is not None and event.kind != expected_kind:
            raise ValueError("event has an unexpected kind")
        return event

    def _event_ids(self, kind: EventKind) -> Tuple[str, ...]:
        return tuple(event.event_id for event in self.event_store.list(self.session_id)
                     if event.kind == kind)

    @staticmethod
    def _percept_payload(percept: Percept, artifact: MediaArtifact, adapter_id: str,
                         adapter_version: str, percept_kind: str) -> dict:
        # The cross-version store projection has one textual value and an
        # optional temporal span. Image regions are intentionally summarized
        # rather than retaining coordinates in this initial protocol slice.
        start_ms = 0
        end_ms = 0
        if percept.spans and percept.spans[0].kind == "time":
            start_ms = percept.spans[0].start_ms or 0
            end_ms = percept.spans[0].end_ms or start_ms
        return {"percept_id": percept.percept_id, "artifact_id": artifact.artifact_id,
                "artifact_sha256": artifact.sha256, "modality": artifact.modality,
                "span_start_ms": start_ms, "span_end_ms": end_ms,
                "percept_kind": percept_kind, "value": percept.summary,
                "confidence": float(percept.confidence),
                "adapter_id": adapter_id, "adapter_version": adapter_version}

    @staticmethod
    def _adapter_identity(perceptor_id: str) -> Tuple[str, str]:
        if not isinstance(perceptor_id, str) or perceptor_id.count("/") != 1:
            raise ValueError("perceptor_id must use adapter_id/version form")
        adapter_id, version = perceptor_id.split("/", 1)
        if not adapter_id or not version:
            raise ValueError("perceptor_id must use adapter_id/version form")
        return adapter_id, version

    @staticmethod
    def _store_trigger(trigger: TickTrigger, salience_reason: str) -> str:
        if trigger == TickTrigger.SALIENCE:
            return "media" if salience_reason == "media" else "text"
        return {TickTrigger.MANUAL: "control", TickTrigger.SCHEDULED: "timer",
                TickTrigger.RESUME: "control"}[trigger]

    @staticmethod
    def _stop_reason_for_record(record: TickRecord) -> str:
        if record.stop_reason in (StopReason.MAX_TICKS, StopReason.MAX_WALL_SECONDS,
                                  StopReason.MAX_EVENTS_PER_TICK):
            return "budget_exhausted"
        if record.stop_reason == StopReason.MAX_NO_PROGRESS:
            return "no_progress"
        if record.stop_reason == StopReason.CALLBACK_ERROR:
            return "error"
        return "completed"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _action_payload(action: ActionProposal) -> dict:
        """Persist a strict public projection, never deliberator free-form fields."""
        known_actions = {"response", "read", "write"}
        return {"action_type": action.action_type if action.action_type in known_actions else "other",
                "required_capability": None if action.required_capability is None else "declared",
                "is_mutating": bool(action.is_mutating),
                "public_summary": "Action proposal recorded for policy review."}

    def _decision_payload(self, decision: DecisionRecord) -> dict:
        return {"turn_id": decision.turn_id, "observation_event_ids": list(decision.observation_event_ids),
                "retrieved_seed_ids": list(decision.retrieved_seed_ids),
                "self_claim_ids": list(decision.self_claim_ids),
                "selected_action": self._action_payload(decision.selected_action),
                "public_summary": "Policy decision recorded for this turn.",
                "policy_reasons": list(decision.policy_reasons)}

    @staticmethod
    def _seed_export(seed: SeedDisposition) -> dict:
        return {"seed_id": seed.seed_id, "cue_terms": list(seed.cue_terms),
                "policy_bias": seed.policy_bias, "scope": seed.scope,
                "provenance_event_ids": list(seed.provenance_event_ids),
                "strength": seed.strength, "confidence": seed.confidence,
                "status": seed.status.value, "created_at": seed.created_at,
                "updated_at": seed.updated_at, "expires_at": seed.expires_at,
                "version": seed.version, "counterevidence": seed.counterevidence}

    @staticmethod
    def _server_seed_candidate(user_text: str, observation_id: str,
                               decision_id: str) -> SeedDisposition:
        """Create a bounded candidate without trusting any model-supplied seed fields."""
        tokens = StrangeloopAgent._seed_cue_terms(user_text)
        # SeedDisposition requires at least one cue.  This fixed fallback makes
        # punctuation-only input safe without adopting model text.
        cue_terms = tokens if tokens else ("current-input",)
        return SeedDisposition(
            cue_terms=cue_terms,
            policy_bias="request-human-review",
            scope="conversation",
            provenance_event_ids=(observation_id, decision_id),
            strength=0.25,
            confidence=0.25,
        )

    @staticmethod
    def _seed_cue_terms(user_text: str) -> Tuple[str, ...]:
        """Project public input into the deterministic bounded seed vocabulary."""
        tokens = []
        # Latin words and bounded CJK runs are both host-tokenized.  CJK runs
        # become overlapping two-character cues because whitespace cannot be
        # assumed to mark words.  Sorting the bounded lexical cues makes a
        # simple leading phrase less likely to create a duplicate identity.
        # This remains a deterministic text projection, not model semantics.
        pattern = (r"[A-Za-z0-9][A-Za-z0-9_-]{1,23}"
                   r"|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]{1,24}")
        for token in re.findall(pattern, user_text.casefold()):
            is_cjk = not token[0].isascii()
            projected = ([token[index:index + 2]
                          for index in range(len(token) - 1)]
                         if is_cjk and len(token) > 1 else [token])
            tokens.extend(projected)
        return tuple(sorted(set(tokens))[:2])

    def _auto_process_text_seed(self, user_text: str, observation_id: str,
                                decision_id: str) -> Tuple[Tuple[str, ...], str]:
        """Apply one bounded post-turn seed action under standing authority.

        The identity lookup, policy preflight, proposal, and application share
        one write transaction.  This prevents two processes handling the same
        session from both persisting a candidate for one lexical identity.
        """
        candidate = self._server_seed_candidate(user_text, observation_id, decision_id)
        identity = canonical_seed_identity(candidate)
        try:
            with self.event_store.transaction():
                # Re-read only after BEGIN IMMEDIATE has serialized writers;
                # a pre-transaction snapshot is insufficient across SQLite
                # connections sharing the same session database.
                matching = [seed for seed in self.seed_store.list(self.session_id)
                            if canonical_seed_identity(seed) == identity]
                # A retired lexical identity is a tombstone.  It is never
                # revived by changing a generated seed ID, and an existing
                # candidate is not retroactively activated by a later policy.
                if any(seed.status.value == "retired" for seed in matching):
                    return (), "retired_identity"
                active = next((seed for seed in matching
                               if seed.status.value == "active"), None)
                if active is not None:
                    policy = self.seed_store.standing_policy_status(self.session_id)
                    if (policy is None or policy.get("status") != "active"
                            or policy["auto_updates_used"] >= policy["max_auto_updates"]):
                        return (), "update_budget_exhausted"
                    proposal = self.seed_store.propose_seed_update(
                        active.seed_id, "reinforce", observation_id,
                        source_ref="server_seed_projection")
                    applied = self.seed_store.auto_apply_update(proposal.event_id)
                    return (), "reinforce" if applied is not None else "no_change"
                if matching:
                    return (), "existing_candidate"
                policy = self.seed_store.standing_policy_status(self.session_id)
                active_count = sum(
                    seed.status.value == "active"
                    for seed in self.seed_store.list(self.session_id)
                )
                if policy is None or policy.get("status") != "active":
                    return (), "policy_inactive"
                if policy["auto_activations_used"] >= policy["max_auto_activations"]:
                    return (), "activation_budget_exhausted"
                if active_count >= policy["max_active_seeds"]:
                    return (), "active_seed_cap_reached"
                stored = self.seed_store.propose(
                    self.session_id, candidate, source_ref="server_seed_projection")
                applied = self.seed_store.auto_apply_candidate(stored.seed_id)
        except (KeyError, RuntimeError, ValueError):
            return (), "rejected"
        return ((stored.seed_id,), "activate" if applied is not None else "rejected")

    def auto_seed_from_expedition_goal(self, goal: str, user_observation_event_id: str,
                                       decision_event_id: str) -> Tuple[Tuple[str, ...], str]:
        """Apply the standing policy to one explicit CLI expedition goal.

        The caller supplies the exact USER goal observation and a host policy
        decision bound to it.  This method accepts no model proposal, tool output, experiment result,
        reward, TD value, quota, sleep, or lifecycle record as seed evidence.
        """
        observation = self.event_store.get(user_observation_event_id)
        decision = self.event_store.get(decision_event_id)
        if (observation is None or observation.session_id != self.session_id
                or observation.kind != EventKind.OBSERVATION
                or observation.source_kind != SourceKind.USER
                or decision is None or decision.session_id != self.session_id
                or decision.kind != EventKind.DECISION
                or tuple(decision.parent_event_ids) != (observation.event_id,)):
            raise ValueError("expedition seed input requires exact USER observation and bound policy decision")
        if self.seed_auto_update_status().get("state") != "active":
            return (), "policy_inactive"
        return self._auto_process_text_seed(goal, user_observation_event_id,
                                            decision_event_id)

    def _auto_process_seed_correction(self, correction: CognitiveEvent,
                                      target_event_id: str) -> None:
        """Tighten a causally related active seed; retire at policy bounds."""
        if correction.source_kind != SourceKind.USER:
            return
        ancestry, pending = set(), [target_event_id]
        # Event parent traversal is bounded and uses only public provenance.
        while pending and len(ancestry) < 64:
            event_id = pending.pop()
            if event_id in ancestry:
                continue
            ancestry.add(event_id)
            event = self.event_store.get(event_id)
            if event is not None:
                pending.extend(event.parent_event_ids[:8])
        try:
            # Serialize correction budget/version checks with their proposal
            # and application for the same cross-process reason as text turns.
            with self.event_store.transaction():
                policy = self.seed_store.standing_policy_status(self.session_id)
                if policy is None or policy.get("status") != "active":
                    return
                for seed in self.seed_store.list(self.session_id):
                    if (seed.status.value != "active"
                            or not ancestry.intersection(seed.provenance_event_ids)):
                        continue
                    policy = self.seed_store.standing_policy_status(self.session_id)
                    if policy["auto_updates_used"] >= policy["max_auto_updates"]:
                        return
                    operation = "retire" if (
                        seed.counterevidence + 1 >= policy["max_counterevidence"]
                        or seed.strength <= policy["max_strength_step"]
                        or seed.confidence <= policy["max_confidence_step"]
                    ) else "tighten"
                    proposal = self.seed_store.propose_seed_update(
                        seed.seed_id, operation, correction.event_id,
                        source_ref="server_seed_projection")
                    self.seed_store.auto_apply_update(proposal.event_id)
        except (KeyError, RuntimeError, ValueError):
            # The correction remains inspectable if a concurrent lifecycle
            # change makes bounded automatic maintenance ineligible.
            return

    @staticmethod
    def _claim_export(claim: Any) -> dict:
        return {"claim_id": claim.claim_id, "kind": claim.kind.value,
                "statement": claim.statement,
                "evidence_event_ids": list(claim.evidence_event_ids),
                "confidence": claim.confidence, "created_at": claim.created_at,
                "expires_at": claim.expires_at}

    @staticmethod
    def _records_by_category(events: Any) -> Dict[str, list]:
        """Expose provenance categories without conflating observations and approvals."""
        categories = {
            "observations": [], "approvals": [], "tool_results": [], "inferences": [],
            "self_model_claims": [], "memory_proposals": [],
            "user_approved_memories": [], "memory_retirements": [],
            "corrections": [], "actions_and_decisions": [], "media_observations": [],
            "percepts": [], "autonomy": [], "reward_observations": [],
            "value_estimates": [], "rpe_updates": [], "metacognitive_mirrors": [],
        }
        for event in events:
            record = event.to_dict()
            if event.kind == EventKind.OBSERVATION:
                categories["approvals" if "approval" in event.payload else "observations"].append(record)
            elif event.kind == EventKind.MEDIA_OBSERVATION:
                categories["media_observations"].append(record)
            elif event.kind == EventKind.PERCEPT:
                categories["percepts"].append(record)
            elif event.kind == EventKind.TOOL_RESULT:
                categories["tool_results"].append(record)
            elif event.kind == EventKind.INFERENCE:
                categories["inferences"].append(record)
            elif event.kind in (EventKind.SELF_CLAIM_PROPOSED, EventKind.SELF_CLAIM_APPROVED,
                                EventKind.SELF_CLAIM_REVOKED):
                categories["self_model_claims"].append(record)
            elif event.kind == EventKind.SEED_PROPOSED:
                categories["memory_proposals"].append(record)
            elif event.kind == EventKind.SEED_APPROVED:
                categories["user_approved_memories"].append(record)
            elif event.kind == EventKind.SEED_RETIRED:
                categories["memory_retirements"].append(record)
            elif event.kind == EventKind.CORRECTION:
                categories["corrections"].append(record)
            elif event.kind in (EventKind.AUTONOMY_CONTROL, EventKind.LOOP_TICK,
                                EventKind.AUTONOMY_STOPPED):
                categories["autonomy"].append(record)
            elif event.kind == EventKind.REWARD_OBSERVATION:
                categories["reward_observations"].append(record)
            elif event.kind == EventKind.VALUE_ESTIMATE:
                categories["value_estimates"].append(record)
            elif event.kind == EventKind.RPE_UPDATE:
                categories["rpe_updates"].append(record)
            elif event.kind == EventKind.METACOGNITIVE_MIRROR:
                categories["metacognitive_mirrors"].append(record)
            elif event.kind in (EventKind.ACTION_PROPOSED, EventKind.ACTION_RESULT,
                                EventKind.DECISION):
                categories["actions_and_decisions"].append(record)
        return categories
