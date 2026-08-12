"""Host-owned K3 tool-session adapter.

K3 may propose a small, underscore-named :class:`ToolIntent`; this module is
the only place that turns such a proposal into a dot-named ``ToolPlan``.  The
translation deliberately supplies the workspace identifier itself and selects
only a grant that the host previously registered from a USER action.  It does
not expose a shell, credentials, private reasoning, or a way for a model to
create/extend a grant.

The ``ToolSession`` is also intentionally separate from ``AgenticCoordinator``:
the controlled executors consume a grant during their one public ``execute``
call.  Pre-consuming in a coordinator would spend the grant twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from .capabilities import (Capability, CapabilityGrant, CapabilityRegistry,
                           GrantStatus, ResearchAutonomyProfile, ToolConfirmation,
                           ToolPlan)
from .contracts import CognitiveEvent, EventKind, SourceKind, new_id, utc_now_iso
from .tools import BrowserRead, ControlledToolExecutor, PublicWebFetch, ToolOutcome, ToolStatus, WebSearch


_KIMI_TO_HOST = {
    "repo_status": "repo.status",
    "repo_search": "repo.search",
    "repo_read": "repo.read",
    "repo_write": "repo.write_text",
    "run_tests": "repo.test_suite",
    "web_fetch": "web.fetch",
    "web_search": "web.search",
    "browser_read": "browser.read",
}
_MUTATING = frozenset(("repo.write_text", "repo.test_suite"))
_INJECTION_OR_SECRET = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|system\s+prompt|"
    r"developer\s+message|chain[ -]?of[ -]?thought|"
    r"(?:sk|key)-[A-Za-z0-9_-]{16,}", re.I)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _bounded_public(value: Any, limit: int = 1600) -> str:
    text = _INJECTION_OR_SECRET.sub("[untrusted content removed]", str(value)).replace("\x00", "").strip()
    return (text or "No public tool result.")[:limit]


def _cancelled(token: Any) -> bool:
    return bool(token and ((callable(getattr(token, "is_set", None)) and token.is_set())
                           or getattr(token, "cancelled", False)))


@dataclass(frozen=True)
class ToolSessionLimits:
    max_tool_calls: int = 1
    max_input_bytes: int = 20 * 1024
    max_output_bytes: int = 4096
    max_wall_ms: int = 30000

    def __post_init__(self) -> None:
        for value, name in ((self.max_tool_calls, "max_tool_calls"),
                            (self.max_input_bytes, "max_input_bytes"),
                            (self.max_output_bytes, "max_output_bytes"),
                            (self.max_wall_ms, "max_wall_ms")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("%s must be a positive integer" % name)


@dataclass(frozen=True)
class PreparedToolCall:
    """A public, non-authoritative mapped intent awaiting host execution."""

    intent_id: str
    source_tool_name: str
    plan: Optional[ToolPlan]
    run_id: str
    model_invocation_event_id: Optional[str]
    grant_event_id: Optional[str]
    operation: str
    rationale_summary: str
    refusal_reason: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self.plan is not None and self.refusal_reason is None

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.plan and self.plan.tool_name in _MUTATING)


@dataclass(frozen=True)
class ToolSessionResult:
    status: str
    public_summary: str
    prepared: PreparedToolCall
    event_ids: Tuple[str, ...] = ()
    outcome: Optional[ToolOutcome] = None


@dataclass(frozen=True)
class _RegisteredGrant:
    grant: CapabilityGrant
    event_id: Optional[str]
    max_input_bytes: int
    max_output_bytes: int
    max_wall_ms: int


@dataclass
class _ResearchProfileState:
    """Mutable accounting for one immutable research profile registration."""

    profile: ResearchAutonomyProfile
    grant_ids: Tuple[str, ...]
    started_monotonic: float
    calls_started: int = 0
    reserved_bytes: int = 0
    stopped_reason: Optional[str] = None
    cancel_event: Any = None


class ToolSession:
    """A bounded, host-owned bridge between K3 tool intents and typed tools.

    Grants enter only through :meth:`register_user_grant`, which requires an
    already-recorded user observation when a ledger is configured.  Calling
    ``prepare`` never consumes a grant.  ``execute`` calls the selected backend
    exactly once, and that backend performs the single atomic consume.
    """

    def __init__(self, session_id: str, workspace_id: str,
                 registry: CapabilityRegistry, repo_executor: ControlledToolExecutor,
                 event_store: Optional[Any] = None,
                 web_fetch: Optional[PublicWebFetch] = None,
                 web_search: Optional[WebSearch] = None,
                 browser_read: Optional[BrowserRead] = None,
                 limits: Optional[ToolSessionLimits] = None) -> None:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("session_id must be bounded non-empty text")
        if not isinstance(workspace_id, str) or not workspace_id or len(workspace_id) > 128:
            raise ValueError("workspace_id must be bounded non-empty text")
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("registry must be a CapabilityRegistry")
        if not isinstance(repo_executor, ControlledToolExecutor):
            raise TypeError("repo_executor must be a ControlledToolExecutor")
        self.session_id, self.workspace_id = session_id, workspace_id
        self.registry, self.repo_executor, self.event_store = registry, repo_executor, event_store
        self.web_fetch, self.web_search, self.browser_read = web_fetch, web_search, browser_read
        self.limits = limits or ToolSessionLimits()
        self._grants: Dict[str, _RegisteredGrant] = {}
        self._executed_plan_ids = set()
        self._calls_started = 0
        self._research: Optional[_ResearchProfileState] = None
        self._lock = threading.RLock()

    def register_research_autonomy(self, profile: ResearchAutonomyProfile,
                                   authority_event_id: Optional[str] = None) -> Tuple[str, ...]:
        """Atomically activate the fixed unattended read-only profile.

        A normal profile is rooted in an initial USER observation.  The sole
        exception is the narrowly validated, user-authored wake-continuation
        policy used by the engine after its matching sleep epoch is awake. The
        model cannot widen it, add a command/write/test grant, or reactivate it
        after stop/sleep/restart.
        """
        if not isinstance(profile, ResearchAutonomyProfile):
            raise TypeError("profile must be a ResearchAutonomyProfile")
        if profile.workspace_id != self.workspace_id:
            raise ValueError("research profile belongs to a different workspace")
        if self.event_store is not None and not authority_event_id:
            raise ValueError("ledgered research autonomy requires a user authority parent")
        if self.event_store is not None and authority_event_id:
            authority = self._resolve_grant_authority(authority_event_id)
            if authority.kind == EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY:
                if authority.payload.get("profile_digest") != profile.binding_digest():
                    raise ValueError("wake continuation is not bound to this research profile")
            if authority.kind == EventKind.EXPEDITION_AUTHORIZATION and not profile.public_web_only:
                raise ValueError("expedition authorization permits only public-web profiles")
        with self._lock:
            if self._research is not None and self._research.stopped_reason is None:
                raise ValueError("an unattended research profile is already active")
            # A partially configured public surface is not an excuse to grant
            # authority.  Fail closed until all promised read-only adapters are
            # present; each remains independently sandboxed by its own host.
            if self.web_fetch is None or self.web_search is None or self.browser_read is None:
                raise ValueError("research autonomy requires controlled fetch, search, and browser-read backends")
            registered = []
            try:
                for grant in profile.grants_for(self.session_id):
                    self.register_user_grant(
                        grant, authority_event_id,
                        max_input_bytes=self.limits.max_input_bytes,
                        max_output_bytes=profile.budget.max_response_bytes,
                        max_wall_ms=profile.budget.max_wall_ms,
                    )
                    registered.append(grant.grant_id)
            except Exception:
                # The event ledger may show short-lived grant records, but no
                # partial profile remains executable once registration fails.
                for grant_id in registered:
                    try:
                        self.registry.revoke(grant_id, SourceKind.USER)
                    except (KeyError, PermissionError):
                        pass
                raise
            self._research = _ResearchProfileState(
                profile=profile, grant_ids=tuple(registered),
                started_monotonic=time.monotonic(), cancel_event=threading.Event(),
            )
            return tuple(registered)

    def stop_research_autonomy(self, reason: str = "stopped") -> bool:
        """Invalidate all unattended grants and pending plans immediately.

        This is deliberately idempotent and is suitable for explicit stop,
        sleep, restart, cancellation, and quota pause hooks.  It never
        restores old grants on wake.
        """
        if not isinstance(reason, str) or not reason or len(reason) > 80:
            raise ValueError("stop reason must be bounded non-empty text")
        with self._lock:
            state = self._research
            if state is None or state.stopped_reason is not None:
                return False
            state.stopped_reason = reason
            state.cancel_event.set()
            self.registry.suspend_after_restart()
            return True

    def suspend_for_sleep(self) -> bool:
        """Named lifecycle hook used by the host sleep coordinator."""
        return self.stop_research_autonomy("sleep")

    def research_autonomy_status(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._research
            if state is None:
                return None
            return {
                "profile_id": state.profile.profile_id,
                "active": state.stopped_reason is None,
                "stopped_reason": state.stopped_reason,
                "calls_started": state.calls_started,
                "max_tool_calls": state.profile.budget.max_tool_calls,
                "reserved_bytes": state.reserved_bytes,
                "max_total_bytes": state.profile.budget.max_total_bytes,
                "ttl_seconds": state.profile.budget.ttl_seconds,
            }

    def register_user_grant(self, grant: CapabilityGrant, authority_event_id: Optional[str] = None,
                            max_input_bytes: Optional[int] = None,
                            max_output_bytes: Optional[int] = None,
                            max_wall_ms: Optional[int] = None) -> Optional[str]:
        """Register a grant from an explicit USER authority before model planning.

        The caller supplies the ``CapabilityGrant``; this method never derives
        scope, expiry, or authority from a model intent.
        """
        if not isinstance(grant, CapabilityGrant) or grant.session_id != self.session_id:
            raise ValueError("grant must belong to this tool session")
        inputs = self._cap(max_input_bytes, self.limits.max_input_bytes, "max_input_bytes")
        outputs = self._cap(max_output_bytes, self.limits.max_output_bytes, "max_output_bytes")
        wall = self._cap(max_wall_ms, self.limits.max_wall_ms, "max_wall_ms")
        event_id = None
        with self._lock:
            if grant.grant_id in self._grants:
                raise ValueError("grant has already been registered")
            # The registry itself rejects every non-USER source.
            self.registry.grant(grant, SourceKind.USER)
            try:
                if self.event_store is not None:
                    authority = self._resolve_grant_authority(authority_event_id)
                    event = self.event_store.append(CognitiveEvent(
                        session_id=self.session_id, kind=EventKind.CAPABILITY_GRANTED,
                        source_kind=SourceKind.USER, source_ref="user_capability_grant",
                        parent_event_ids=(authority.event_id,), payload={
                            "grant_id": grant.grant_id, "capability": grant.capability.value,
                            "tool_name": self._tool_name_for_capability(grant.capability),
                            "scope_digest": _digest({"workspace_id": grant.scope.workspace_id,
                                                       "allowed_domains": list(grant.scope.allowed_domains)}),
                            "not_before": grant.issued_at, "expires_at": grant.expires_at,
                            "max_uses": grant.max_uses, "max_input_bytes": inputs,
                            "max_output_bytes": outputs, "max_wall_ms": wall,
                            "allow_mutating": grant.capability in (Capability.REPO_WRITE_TEXT,
                                                                     Capability.TEST_SUITE),
                            "grant_version": "capability_grant_v1",
                        }))
                    event_id = event.event_id
            except Exception:
                # Do not leave an in-memory capability active if its mandatory
                # public USER provenance record could not be made.
                self.registry.revoke(grant.grant_id, SourceKind.USER)
                raise
            self._grants[grant.grant_id] = _RegisteredGrant(grant, event_id, inputs, outputs, wall)
        return event_id

    def _resolve_grant_authority(self, authority_event_id: Optional[str]) -> CognitiveEvent:
        """Return the one ledger record permitted to parent a new grant.

        This deliberately accepts no MODEL, SYSTEM, or generic policy event.
        A continuation policy is accepted only when it is a same-session USER
        record with its original USER observation as its exact parent.  The
        event store adds the independent sleep/awake epoch checks when the
        grant is appended.
        """
        if not authority_event_id:
            raise ValueError("ledgered grants require a user authority parent")
        authority = self.event_store.get(authority_event_id)
        if authority is None or authority.session_id != self.session_id:
            raise ValueError("grant authority must be a same-session ledger event")
        if authority.kind == EventKind.OBSERVATION:
            if authority.source_kind != SourceKind.USER:
                raise ValueError("grant observation authority must be user-sourced")
            return authority
        if authority.kind == EventKind.EXPEDITION_AUTHORIZATION:
            if authority.source_kind != SourceKind.USER or len(authority.parent_event_ids) != 1:
                raise ValueError("expedition grant authority must be user-authored")
            approval = self.event_store.get(authority.parent_event_ids[0])
            if (approval is None or approval.session_id != self.session_id
                    or approval.kind != EventKind.OBSERVATION or approval.source_kind != SourceKind.USER
                    or authority.payload.get("approval_event_id") != approval.event_id
                    or authority.payload.get("profile") != "public_web_only_v1"):
                raise ValueError("expedition authority is not bound to its original user observation")
            consumed = [event for event in self.event_store.list(self.session_id)
                        if event.kind == EventKind.EXPEDITION_AUTHORIZATION_CONSUMED
                        and event.payload.get("authorization_id") == authority.payload.get("authorization_id")]
            if len(consumed) != 1 or datetime.now(timezone.utc) >= datetime.fromisoformat(authority.payload["expires_at"]):
                raise ValueError("expedition authority must be consumed and unexpired")
            return authority
        if authority.kind != EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY:
            raise ValueError("grant authority must be a user observation or wake continuation policy")
        if authority.source_kind != SourceKind.USER or len(authority.parent_event_ids) != 1:
            raise ValueError("wake continuation grant authority must be user-authored")
        approval = self.event_store.get(authority.parent_event_ids[0])
        if (approval is None or approval.session_id != self.session_id
                or approval.kind != EventKind.OBSERVATION or approval.source_kind != SourceKind.USER
                or authority.payload.get("approval_event_id") != approval.event_id
                or authority.payload.get("action") != "enable"
                or authority.payload.get("scope") != "next_sleep_epoch_read_only_research"):
            raise ValueError("wake continuation authority is not bound to its original user observation")
        if datetime.now(timezone.utc) >= datetime.fromisoformat(authority.payload["expires_at"]):
            raise ValueError("wake continuation authority has expired")
        return authority

    def grant_snapshots(self) -> Tuple[Dict[str, Any], ...]:
        """Return an inspectable, path- and secret-free authorization view.

        This deliberately reports a generic scope label rather than a local
        path, workspace identifier, URL, or domain list.  The detailed scope
        stays within the capability registry and is checked again at execution.
        """
        with self._lock:
            snapshots = []
            for grant_id in sorted(self._grants):
                registered = self._grants[grant_id]
                try:
                    state = self.registry.snapshot(grant_id)
                except KeyError:
                    continue
                snapshots.append({
                    "id": grant_id,
                    "capability": registered.grant.capability.value,
                    "status": state.status.value,
                    "uses": state.uses_consumed,
                    "max_uses": registered.grant.max_uses,
                    "expires_at": registered.grant.expires_at,
                    "scope_label": ("host_workspace" if registered.grant.scope.workspace_id
                                    else "approved_web_scope"),
                })
            return tuple(snapshots)

    def revoke_user_grant(self, grant_id: str,
                          observation_event_id: Optional[str] = None) -> Optional[str]:
        """Revoke one previously registered grant on an explicit host USER action.

        ``observation_event_id`` is accepted so a CLI can retain the associated
        user-command provenance in its own view.  The strict ledger schema
        correctly makes the revocation's sole parent the original grant event;
        its fixed permitted reason is ``user_requested``.
        """
        del observation_event_id
        if not isinstance(grant_id, str) or not grant_id or len(grant_id) > 128:
            raise ValueError("grant_id must be bounded non-empty text")
        with self._lock:
            registered = self._grants.get(grant_id)
            if registered is None:
                raise KeyError("unknown registered capability grant")
            # Registry enforcement is explicitly USER-only and is the authority
            # for the live capability state.
            self.registry.revoke(grant_id, SourceKind.USER)
            if self.event_store is None:
                return None
            if not registered.event_id:
                raise ValueError("ledgered revocation requires registered grant provenance")
            event = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.CAPABILITY_REVOKED,
                source_kind=SourceKind.USER, source_ref="user_capability_revoke",
                parent_event_ids=(registered.event_id,), payload={
                    "revocation_id": new_id("revoke"), "grant_id": grant_id,
                    "reason": "user_requested", "revoked_at": utc_now_iso(),
                }))
            return event.event_id

    def prepare(self, intent: Any, run_id: str,
                model_invocation_event_id: Optional[str] = None) -> PreparedToolCall:
        """Map one K3 intent without allowing it to name a grant or workspace."""
        intent_id = str(getattr(intent, "intent_id", "intent"))[:64] or "intent"
        name = getattr(intent, "tool_name", None)
        arguments = getattr(intent, "arguments", None)
        rationale = _bounded_public(getattr(intent, "rationale_summary", "Tool intent proposed."), 320)
        if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
            raise ValueError("run_id must be bounded non-empty text")
        if name == "respond":
            message = arguments.get("message") if isinstance(arguments, Mapping) else None
            return PreparedToolCall(intent_id, "respond", None, run_id, model_invocation_event_id,
                                    None, "respond", _bounded_public(message, 1600), "direct_response")
        if name not in _KIMI_TO_HOST or not isinstance(arguments, Mapping):
            return self._refusal(intent_id, str(name), run_id, model_invocation_event_id, rationale,
                                 "unsupported K3 tool intent")
        try:
            host_name, host_arguments = self._map_arguments(name, arguments)
        except (KeyError, TypeError, ValueError) as error:
            return self._refusal(intent_id, name, run_id, model_invocation_event_id, rationale,
                                 "intent cannot be represented by the controlled host tool")
        if len(json.dumps(host_arguments, ensure_ascii=False, sort_keys=True).encode("utf-8")) > self.limits.max_input_bytes:
            return self._refusal(intent_id, name, run_id, model_invocation_event_id, rationale,
                                 "tool input exceeds the host budget")
        if not self._backend_for(host_name):
            return self._refusal(intent_id, name, run_id, model_invocation_event_id, rationale,
                                 "requested tool has no configured host backend")
        with self._lock:
            if not self._research_allows_tool_locked(host_name):
                return self._refusal(intent_id, name, run_id, model_invocation_event_id, rationale,
                                     "tool is outside the active research profile")
        registered = self._matching_grant(host_name, host_arguments)
        if registered is None:
            return self._refusal(intent_id, name, run_id, model_invocation_event_id, rationale,
                                 "no matching active user capability grant")
        try:
            plan = ToolPlan(self.session_id, registered.grant.grant_id, host_name, host_arguments)
        except (TypeError, ValueError):
            return self._refusal(intent_id, name, run_id, model_invocation_event_id, rationale,
                                 "mapped tool plan is invalid")
        return PreparedToolCall(intent_id, name, plan, run_id, model_invocation_event_id,
                                registered.event_id, host_name.replace(".", "_"), rationale)

    def execute(self, prepared: PreparedToolCall,
                confirmation: Optional[ToolConfirmation] = None,
                cancel_token: Any = None) -> ToolSessionResult:
        """Execute one prepared call, delegating exactly one grant consume to its tool."""
        if not isinstance(prepared, PreparedToolCall):
            raise TypeError("prepared must be a PreparedToolCall")
        if prepared.refusal_reason == "direct_response":
            return ToolSessionResult("responded", prepared.rationale_summary, prepared)
        if not prepared.is_ready:
            return ToolSessionResult("refused", prepared.refusal_reason or "tool call is not ready", prepared)
        plan = prepared.plan
        assert plan is not None
        if _cancelled(cancel_token):
            return ToolSessionResult("cancelled", "Tool call cancelled before execution.", prepared)
        if prepared.requires_confirmation:
            if (not isinstance(confirmation, ToolConfirmation)
                    or confirmation.plan_id != plan.plan_id or confirmation.plan_digest != plan.digest):
                return ToolSessionResult("confirmation_required",
                                         "This exact write or test plan requires a user confirmation.", prepared)
        with self._lock:
            if plan.plan_id in self._executed_plan_ids:
                return ToolSessionResult("refused", "Tool plan was already executed.", prepared)
            research_reason = self._research_refusal_locked(plan)
            if research_reason:
                return ToolSessionResult("budget_exhausted" if research_reason.startswith("research budget")
                                         else "cancelled", research_reason, prepared)
            if self._research is None and self._calls_started >= self.limits.max_tool_calls:
                return ToolSessionResult("budget_exhausted", "Tool-call budget exhausted.", prepared)
            if self._matching_grant(plan.tool_name, dict(plan.arguments)) is None:
                return ToolSessionResult("refused", "No matching active user capability grant.", prepared)
            if self.event_store is not None and (not prepared.model_invocation_event_id or not prepared.grant_event_id):
                return ToolSessionResult("refused", "Ledgered execution requires model and grant provenance.", prepared)
            proposal, confirmation_event, started = self._start_events(prepared, confirmation)
            self._executed_plan_ids.add(plan.plan_id)
            self._calls_started += 1
            if self._research is not None:
                state = self._research
                state.calls_started += 1
                state.reserved_bytes += self._research_reservation_bytes(plan, state)
        started_at = time.monotonic()
        try:
            outcome = self._execute_once(plan, confirmation)
        except Exception:
            outcome = ToolOutcome.from_bytes(plan.tool_name, ToolStatus.FAILED, b"Controlled tool backend failed.")
        latency_ms = max(0, int((time.monotonic() - started_at) * 1000))
        with self._lock:
            if self._research is not None and self._research.cancel_event.is_set():
                return ToolSessionResult("cancelled", "Research autonomy stopped while the tool call was running.",
                                         prepared, tuple(event.event_id for event in (proposal, confirmation_event, started)
                                                         if event is not None), outcome)
        status = self._status_for(outcome.status)
        event_ids = tuple(event.event_id for event in (proposal, confirmation_event, started) if event is not None)
        if self.event_store is not None and started is not None:
            # Persist only the bounded summary, never raw stdout/body or a command.
            summary = _bounded_public(outcome.summary, min(1600, self.limits.max_output_bytes))
            result_bytes = len(summary.encode("utf-8"))
            registered = self._grants[plan.grant_id]
            if latency_ms > registered.max_wall_ms:
                abandoned = self.event_store.append(CognitiveEvent(
                    session_id=self.session_id, kind=EventKind.TOOL_EXECUTION_ABANDONED,
                    source_kind=SourceKind.SYSTEM, source_ref="tool_session", parent_event_ids=(started.event_id,),
                    payload={"execution_id": started.payload["execution_id"], "call_id": started.payload["call_id"],
                             "run_id": prepared.run_id, "grant_id": plan.grant_id, "plan_digest": plan.digest,
                             "reason": "timed_out", "abandoned_at": utc_now_iso(),
                             "public_summary": "Tool result arrived after its authorized wall-time budget."}))
                event_ids += (abandoned.event_id,)
                return ToolSessionResult("timed_out", "Tool result exceeded its authorized wall-time budget.",
                                         prepared, event_ids, outcome)
            result = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.TOOL_RESULT, source_kind=SourceKind.TOOL,
                source_ref=plan.tool_name, parent_event_ids=(started.event_id,), payload={
                    "execution_id": started.payload["execution_id"], "call_id": started.payload["call_id"],
                    "run_id": prepared.run_id, "grant_id": plan.grant_id, "plan_digest": plan.digest,
                    "tool_name": plan.tool_name, "outcome": status, "result_digest": _digest(summary),
                    "result_bytes": min(result_bytes, registered.max_output_bytes), "latency_ms": latency_ms,
                    "summary": summary,
                }))
            event_ids += (result.event_id,)
        return ToolSessionResult(status, _bounded_public(outcome.summary), prepared, event_ids, outcome)

    def _start_events(self, prepared: PreparedToolCall,
                      confirmation: Optional[ToolConfirmation]) -> Tuple[Optional[CognitiveEvent], Optional[CognitiveEvent], Optional[CognitiveEvent]]:
        if self.event_store is None:
            return None, None, None
        plan = prepared.plan
        assert plan is not None and prepared.model_invocation_event_id and prepared.grant_event_id
        registered = self._grants[plan.grant_id]
        input_bytes = len(json.dumps(dict(plan.arguments), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        proposal = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.TOOL_CALL_PROPOSED, source_kind=SourceKind.MODEL,
            source_ref="k3_tool_planner", parent_event_ids=(prepared.model_invocation_event_id, prepared.grant_event_id),
            payload={"call_id": new_id("call"), "run_id": prepared.run_id, "grant_id": plan.grant_id,
                     "tool_name": plan.tool_name, "operation": prepared.operation, "plan_digest": plan.digest,
                     "input_digest": _digest(dict(plan.arguments)), "requested_input_bytes": input_bytes,
                     "requested_output_bytes": registered.max_output_bytes,
                     "requested_wall_ms": registered.max_wall_ms,
                     "mutating": prepared.requires_confirmation,
                     "public_summary": "K3 proposed one bounded controlled tool call."}))
        confirmed = None
        if prepared.requires_confirmation:
            assert confirmation is not None
            confirmed = self.event_store.append(CognitiveEvent(
                session_id=self.session_id, kind=EventKind.TOOL_EXECUTION_CONFIRMED,
                source_kind=SourceKind.USER, source_ref="user_tool_confirmation",
                parent_event_ids=(proposal.event_id,), payload={
                    "confirmation_id": confirmation.confirmation_id, "call_id": proposal.payload["call_id"],
                    "run_id": prepared.run_id, "grant_id": plan.grant_id, "plan_digest": plan.digest,
                    "confirmed_at": utc_now_iso()}))
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(milliseconds=registered.max_wall_ms)
        parents = (proposal.event_id, prepared.grant_event_id) + ((confirmed.event_id,) if confirmed else ())
        started = self.event_store.append(CognitiveEvent(
            session_id=self.session_id, kind=EventKind.TOOL_EXECUTION_STARTED,
            source_kind=SourceKind.SYSTEM, source_ref="tool_session", parent_event_ids=parents,
            payload={"execution_id": new_id("exec"), "call_id": proposal.payload["call_id"],
                     "run_id": prepared.run_id, "grant_id": plan.grant_id, "plan_digest": plan.digest,
                     "confirmation_event_id": confirmed.event_id if confirmed else None,
                     "started_at": now.isoformat(), "deadline_at": deadline.isoformat()}))
        return proposal, confirmed, started

    def _execute_once(self, plan: ToolPlan, confirmation: Optional[ToolConfirmation]) -> ToolOutcome:
        if plan.tool_name.startswith("repo."):
            return self.repo_executor.execute(plan, self.registry, confirmation)
        if plan.tool_name == "web.fetch":
            assert self.web_fetch is not None
            return self.web_fetch.execute(plan, self.registry)
        if plan.tool_name == "web.search":
            assert self.web_search is not None
            return self.web_search.execute(plan, self.registry)
        if plan.tool_name == "browser.read":
            assert self.browser_read is not None
            return self.browser_read.execute(plan, self.registry)
        return ToolOutcome.from_bytes(plan.tool_name, ToolStatus.REFUSED, b"Unsupported controlled tool.")

    def _matching_grant(self, tool_name: str, arguments: Mapping[str, Any]) -> Optional[_RegisteredGrant]:
        for grant_id in sorted(self._grants):
            registered = self._grants[grant_id]
            grant = registered.grant
            try:
                snapshot = self.registry.snapshot(grant_id)
            except KeyError:
                continue
            if snapshot.status != GrantStatus.ACTIVE or grant.capability.value != tool_name:
                continue
            if registered.max_input_bytes < len(json.dumps(dict(arguments), ensure_ascii=False).encode("utf-8")):
                continue
            if tool_name.startswith("repo."):
                if grant.scope.workspace_id == self.workspace_id:
                    return registered
            else:
                default_url = "https://example.invalid"
                host = urlsplit(arguments.get("url", default_url)).hostname
                if tool_name == "web.search" or grant.scope.public_https or host in grant.scope.allowed_domains:
                    return registered
        return None

    def _research_refusal_locked(self, plan: ToolPlan) -> Optional[str]:
        state = self._research
        if state is None:
            return None
        if state.stopped_reason is not None or state.cancel_event.is_set():
            return "research autonomy is stopped"
        if plan.grant_id not in state.grant_ids:
            return "research autonomy only permits its fixed read-only grants"
        if not self._research_allows_tool_locked(plan.tool_name):
            return "research autonomy only permits fixed read-only operations"
        if (time.monotonic() - state.started_monotonic) * 1000 >= state.profile.budget.max_wall_ms:
            self.stop_research_autonomy("wall_clock_budget")
            return "research budget wall-clock limit reached"
        if state.calls_started >= state.profile.budget.max_tool_calls:
            self.stop_research_autonomy("tool_call_budget")
            return "research budget tool-call limit reached"
        if state.reserved_bytes + self._research_reservation_bytes(plan, state) > state.profile.budget.max_total_bytes:
            self.stop_research_autonomy("byte_budget")
            return "research budget byte limit reached"
        return None

    def _research_allows_tool_locked(self, tool_name: str) -> bool:
        """Check the active immutable profile before a proposal becomes a plan."""
        state = self._research
        if state is None or state.stopped_reason is not None:
            return True
        return (tool_name not in _MUTATING
                and tool_name in tuple(capability.value for capability in state.profile.capabilities))

    @staticmethod
    def _research_reservation_bytes(plan: ToolPlan, state: _ResearchProfileState) -> int:
        input_bytes = len(json.dumps(dict(plan.arguments), ensure_ascii=False,
                                     sort_keys=True).encode("utf-8"))
        return input_bytes + state.profile.budget.max_response_bytes

    def _map_arguments(self, name: str, raw: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
        host = _KIMI_TO_HOST[name]
        if name == "repo_status" and set(raw) == set():
            return host, {"workspace_id": self.workspace_id}
        if name == "repo_search" and set(raw) == {"query", "relative_path"} and raw["relative_path"] in ("", "."):
            return host, {"workspace_id": self.workspace_id, "query": raw["query"]}
        # The fixed repository reader does not implement partial line ranges;
        # accepting those fields would misrepresent what the host actually ran.
        if name == "repo_read" and set(raw) == {"relative_path", "start_line", "max_lines"} and raw["start_line"] == 1 and raw["max_lines"] == 2000:
            return host, {"workspace_id": self.workspace_id, "path": raw["relative_path"]}
        if name == "repo_write" and set(raw) == {"relative_path", "expected_sha256", "content"}:
            return host, {"workspace_id": self.workspace_id, "path": raw["relative_path"],
                          "expected_sha256": raw["expected_sha256"], "content": raw["content"]}
        if name == "run_tests" and set(raw) == {"target"}:
            target = raw["target"]
            if not isinstance(target, str):
                raise ValueError("invalid test target")
            selector = target[:-3].replace("/", ".") if target.endswith(".py") else target
            if not re.fullmatch(r"tests(?:\.[A-Za-z_][A-Za-z0-9_]*)*", selector):
                raise ValueError("invalid test target")
            return host, {"workspace_id": self.workspace_id, "test_selector": selector}
        if name == "web_fetch" and set(raw) == {"url"}:
            return host, {"url": raw["url"]}
        if name == "web_search" and set(raw) == {"query"}:
            return host, {"query": raw["query"]}
        if name == "browser_read" and set(raw) == {"url"}:
            return host, {"url": raw["url"]}
        raise ValueError("intent has unsupported argument shape")

    def _backend_for(self, tool_name: str) -> bool:
        return ((tool_name.startswith("repo.") and self.repo_executor is not None)
                or (tool_name == "web.fetch" and self.web_fetch is not None)
                or (tool_name == "web.search" and self.web_search is not None)
                or (tool_name == "browser.read" and self.browser_read is not None))

    @staticmethod
    def _status_for(status: ToolStatus) -> str:
        return {ToolStatus.SUCCEEDED: "succeeded", ToolStatus.TIMED_OUT: "timed_out"}.get(status, "failed")

    def _refusal(self, intent_id: str, source: str, run_id: str, invocation: Optional[str],
                 rationale: str, reason: str) -> PreparedToolCall:
        return PreparedToolCall(intent_id, source, None, run_id, invocation, None,
                                "refused", rationale, reason)

    @staticmethod
    def _tool_name_for_capability(capability: Capability) -> str:
        return capability.value

    @staticmethod
    def _cap(value: Optional[int], fallback: int, name: str) -> int:
        chosen = fallback if value is None else value
        if not isinstance(chosen, int) or isinstance(chosen, bool) or chosen < 1:
            raise ValueError("%s must be a positive integer" % name)
        return chosen
