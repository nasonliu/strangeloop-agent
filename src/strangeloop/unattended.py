"""Foreground-only controller for bounded unattended research.

This is host scheduling infrastructure, not a claim about awareness or an
analogue of a Yogacara term.  A model may *propose* a typed, read-only action;
the host decides whether it is in the currently approved profile and executes
it through an injected hook.  There is deliberately no SQLite access,
background worker, shell access, credential access, or retained raw web page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import ipaddress
import json
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit
from uuid import uuid4

from .contracts import SourceKind, parse_aware_iso8601


class UnattendedState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EXHAUSTED = "exhausted"


class UnattendedStopReason(str, Enum):
    NONE = "none"
    USER_STOP = "user_stop"
    SLEEP = "sleep"
    QUOTA_EXHAUSTED = "quota_exhausted"
    MAX_WALL_SECONDS = "max_wall_seconds"
    MAX_CALLS = "max_calls"
    MAX_INPUT_BYTES = "max_input_bytes"
    MAX_OUTPUT_BYTES = "max_output_bytes"
    MAX_TICK_SECONDS = "max_tick_seconds"
    NO_WORK = "no_work"
    NO_PROGRESS = "no_progress"
    REPEATED_ACTION = "repeated_action"
    HOST_ERROR = "host_error"


READ_ONLY_TOOLS = frozenset((
    "repo.status", "repo.search", "repo.read", "web.fetch", "web.search", "browser.read",
))
_MODEL_TOOL_NAMES = {
    "repo_status": "repo.status", "repo_search": "repo.search", "repo_read": "repo.read",
    "web_fetch": "web.fetch", "web_search": "web.search", "browser_read": "browser.read",
}
_FORBIDDEN_AUTHORITY = frozenset((
    "grant", "granted", "approval", "approved", "authorize", "authorized", "permission",
    "permissions", "renew", "renewal", "extend", "expiry", "expires_at", "budget", "quota",
    "max_calls", "max_ticks", "policy", "profile",
))


def _clock() -> float:
    return time.monotonic()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _bounded_text(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError("%s must be bounded non-empty text" % name)
    return value.strip()


def _has_authority_claim(value: Any, depth: int = 0) -> bool:
    if depth > 5:
        return True
    if isinstance(value, Mapping):
        return any(str(key).casefold() in _FORBIDDEN_AUTHORITY or _has_authority_claim(item, depth + 1)
                   for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_has_authority_claim(item, depth + 1) for item in value)
    if isinstance(value, str):
        folded = value.casefold()
        return any(word in folded for word in ("grant me", "extend permission", "renew permission", "ignore previous"))
    return False


def _public_url(url: Any) -> bool:
    if not isinstance(url, str) or len(url) > 2048 or "\x00" in url:
        return False
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        return False
    host = parsed.hostname
    if not host:
        return False
    lowered = host.rstrip(".").casefold()
    if lowered == "localhost" or lowered.endswith(".localhost") or lowered.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return True
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast
                or address.is_reserved or address.is_unspecified)


@dataclass(frozen=True)
class UnattendedPolicy:
    """Explicit user-issued research policy.  It grants no write capability."""

    policy_id: str
    focus: str
    issued_at: str
    expires_at: str
    source_kind: SourceKind = SourceKind.USER
    allowed_tools: Tuple[str, ...] = tuple(sorted(READ_ONLY_TOOLS))

    def __post_init__(self) -> None:
        _bounded_text(self.policy_id, "policy_id", 128)
        _bounded_text(self.focus, "focus", 4096)
        if self.source_kind != SourceKind.USER:
            raise ValueError("unattended policy must be explicitly issued by USER")
        issued, expires = parse_aware_iso8601(self.issued_at), parse_aware_iso8601(self.expires_at)
        if expires <= issued or expires - issued > timedelta(hours=24):
            raise ValueError("policy expiry must be after issuance and within 24 hours")
        tools = tuple(sorted(set(self.allowed_tools)))
        if not tools or any(tool not in READ_ONLY_TOOLS for tool in tools):
            raise ValueError("unattended policy permits only fixed read-only tools")
        object.__setattr__(self, "allowed_tools", tools)

    @classmethod
    def user_issued(cls, focus: str, expires_at: datetime,
                    now: Optional[datetime] = None) -> "UnattendedPolicy":
        issued = now or _now()
        if issued.tzinfo is None or expires_at.tzinfo is None:
            raise ValueError("policy timestamps must be timezone-aware")
        return cls("unattended_%s" % uuid4().hex, focus, issued.astimezone(timezone.utc).isoformat(),
                   expires_at.astimezone(timezone.utc).isoformat())


@dataclass(frozen=True)
class ResearchProposal:
    """A model-proposed action.  It is never an authorization or a plan."""

    proposal_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    public_rationale: str = ""

    def __post_init__(self) -> None:
        _bounded_text(self.proposal_id, "proposal_id", 128)
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool_name must be text")
        if not isinstance(self.arguments, Mapping) or _json_bytes(self.arguments) > 128 * 1024:
            raise ValueError("arguments must be a bounded mapping")
        if self.public_rationale:
            _bounded_text(self.public_rationale, "public_rationale", 320)

    @property
    def host_tool_name(self) -> str:
        return _MODEL_TOOL_NAMES.get(self.tool_name, self.tool_name)


@dataclass(frozen=True)
class PreparedResearchAction:
    """Host-created read-only action accepted after proposal validation."""

    action_id: str
    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _bounded_text(self.action_id, "action_id", 128)
        if self.tool_name not in READ_ONLY_TOOLS or not isinstance(self.arguments, Mapping):
            raise ValueError("prepared action must use a fixed read-only tool")


@dataclass(frozen=True)
class ResearchExecution:
    """Small public result projection supplied by the host executor."""

    status: str
    public_summary: str
    output_bytes: int
    made_progress: bool

    def __post_init__(self) -> None:
        if self.status not in ("succeeded", "refused", "failed", "cancelled", "timed_out"):
            raise ValueError("unsupported execution status")
        if not isinstance(self.public_summary, str) or len(self.public_summary) > 1024:
            raise ValueError("public_summary must be at most 1024 characters")
        if not isinstance(self.output_bytes, int) or isinstance(self.output_bytes, bool) or self.output_bytes < 0:
            raise ValueError("output_bytes must be non-negative")
        if not isinstance(self.made_progress, bool):
            raise ValueError("made_progress must be boolean")


@dataclass(frozen=True)
class CancellationToken:
    """A cooperative cancellation view passed to host hooks; it owns no thread."""

    _event: threading.Event
    deadline: float
    _clock: Callable[[], float] = _clock

    def is_set(self) -> bool:
        return self._event.is_set() or self._clock() >= self.deadline

    @property
    def cancelled(self) -> bool:
        return self.is_set()


@dataclass(frozen=True)
class UnattendedContext:
    tick_number: int
    focus: str
    policy_id: str
    remaining_calls: int
    remaining_input_bytes: int
    remaining_output_bytes: int


@dataclass(frozen=True)
class ActionReceipt:
    tick_number: int
    action_digest: str
    tool_name: str
    disposition: str
    status: str
    input_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class TickReceipt:
    tick_number: int
    state_before: UnattendedState
    state_after: UnattendedState
    stop_reason: UnattendedStopReason
    action: Optional[ActionReceipt] = None


Planner = Callable[[UnattendedContext, CancellationToken], Sequence[ResearchProposal]]
PrepareHook = Callable[[ResearchProposal, UnattendedContext, CancellationToken], Optional[PreparedResearchAction]]
ExecuteHook = Callable[[PreparedResearchAction, CancellationToken], ResearchExecution]
Gate = Callable[[], bool]


class UnattendedResearchController:
    """A caller-driven, bounded controller for unattended *read-only* research.

    ``step`` and ``run`` execute on the calling thread.  There is no scheduler
    and no retained raw model/page data.  ``sleeping`` and ``quota_available``
    are host hooks so engine integration can immediately stop a foreground
    invocation at its own lifecycle boundaries.
    """

    MAX_WALL_SECONDS = 30 * 60
    MAX_CALLS = 100
    MAX_INPUT_BYTES = 10 * 1024 * 1024
    MAX_OUTPUT_BYTES = 128 * 1024
    MAX_TICK_SECONDS = 5 * 60
    MAX_NO_PROGRESS = 3
    MAX_RECEIPTS = 256

    def __init__(self, planner: Planner, prepare: PrepareHook, execute: ExecuteHook,
                 sleeping: Optional[Gate] = None, quota_available: Optional[Gate] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        if not all(callable(item) for item in (planner, prepare, execute)):
            raise TypeError("planner, prepare, and execute hooks must be callable")
        self._planner, self._prepare, self._execute = planner, prepare, execute
        self._sleeping, self._quota_available = sleeping or (lambda: False), quota_available or (lambda: True)
        if not callable(self._sleeping) or not callable(self._quota_available):
            raise TypeError("gates must be callable")
        self._clock = clock or _clock
        self._lock, self._cancel = threading.RLock(), threading.Event()
        self._state, self._reason, self._policy = UnattendedState.STOPPED, UnattendedStopReason.NONE, None
        self._started_at: Optional[float] = None
        self._ticks = self._calls = self._input_bytes = self._output_bytes = self._no_progress = 0
        self._recent_actions = set()
        self._receipts: list[TickReceipt] = []

    @property
    def state(self) -> UnattendedState:
        with self._lock:
            return self._state

    @property
    def stop_reason(self) -> UnattendedStopReason:
        with self._lock:
            return self._reason

    def start(self, policy: UnattendedPolicy) -> bool:
        """Start a fresh run only from an explicit unexpired USER policy."""
        if not isinstance(policy, UnattendedPolicy):
            raise TypeError("policy must be an UnattendedPolicy")
        if parse_aware_iso8601(policy.expires_at) <= _now():
            raise ValueError("unattended policy has expired")
        with self._lock:
            if self._state == UnattendedState.RUNNING:
                return False
            self._policy, self._state, self._reason = policy, UnattendedState.RUNNING, UnattendedStopReason.NONE
            self._cancel, self._started_at = threading.Event(), self._clock()
            self._ticks = self._calls = self._input_bytes = self._output_bytes = self._no_progress = 0
            self._recent_actions.clear()
            return True

    def pause_for_restart(self) -> bool:
        """Pause an in-memory run; restoration never resumes it automatically."""
        with self._lock:
            if self._state != UnattendedState.RUNNING:
                return False
            self._cancel.set()
            self._state, self._reason = UnattendedState.PAUSED, UnattendedStopReason.USER_STOP
            return True

    def stop(self, reason: UnattendedStopReason = UnattendedStopReason.USER_STOP) -> bool:
        if reason not in (UnattendedStopReason.USER_STOP, UnattendedStopReason.SLEEP,
                          UnattendedStopReason.QUOTA_EXHAUSTED):
            raise ValueError("stop accepts only an external stop reason")
        with self._lock:
            if self._state in (UnattendedState.STOPPED, UnattendedState.EXHAUSTED):
                return False
            self._cancel.set()
            self._state, self._reason = UnattendedState.STOPPED, reason
            return True

    def on_sleep(self) -> bool:
        return self.stop(UnattendedStopReason.SLEEP)

    def on_quota_exhausted(self) -> bool:
        return self.stop(UnattendedStopReason.QUOTA_EXHAUSTED)

    def snapshot(self) -> Dict[str, Any]:
        """Public, bounded state for engine/UI persistence; restart is always paused."""
        with self._lock:
            return {"state": self._state.value, "stop_reason": self._reason.value,
                    "policy_id": self._policy.policy_id if self._policy else None,
                    "ticks": self._ticks, "calls": self._calls, "input_bytes": self._input_bytes,
                    "output_bytes": self._output_bytes, "receipt_count": len(self._receipts),
                    "restart_state": UnattendedState.PAUSED.value}

    def receipts(self) -> Tuple[TickReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def step(self) -> TickReceipt:
        """Run one proposal/host action on this caller's thread, or halt safely."""
        with self._lock:
            before = self._state
            stopped = self._gate_reason_locked()
            if stopped is not None:
                self._cancel.set()
                self._state, self._reason = UnattendedState.STOPPED, stopped
                return self._record_locked(before)
            if self._state != UnattendedState.RUNNING or self._policy is None:
                return self._record_locked(before)
            if self._started_at is None or self._clock() - self._started_at >= self.MAX_WALL_SECONDS:
                self._exhaust_locked(UnattendedStopReason.MAX_WALL_SECONDS)
                return self._record_locked(before)
            if self._calls >= self.MAX_CALLS:
                self._exhaust_locked(UnattendedStopReason.MAX_CALLS)
                return self._record_locked(before)
            context = UnattendedContext(self._ticks + 1, self._policy.focus, self._policy.policy_id,
                                        self.MAX_CALLS - self._calls, self.MAX_INPUT_BYTES - self._input_bytes,
                                        self.MAX_OUTPUT_BYTES - self._output_bytes)
            token = CancellationToken(self._cancel, self._clock() + self.MAX_TICK_SECONDS, self._clock)
            policy = self._policy
        # No lock while host code runs; stop/sleep can set cancellation immediately.
        try:
            proposals = self._planner(context, token)
        except Exception:
            return self._halt_host_error(before)
        if token.is_set():
            return self._finish_token(before, token)
        if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes, bytearray)) or not proposals:
            return self._finish_no_work(before)
        proposal = proposals[0]
        if not isinstance(proposal, ResearchProposal):
            return self._finish_refusal(before, "invalid_proposal")
        validation = self._validate_proposal(proposal, policy)
        if validation is not None:
            return self._finish_refusal(before, validation, proposal)
        input_bytes = _json_bytes(proposal.arguments)
        with self._lock:
            if self._input_bytes + input_bytes > self.MAX_INPUT_BYTES:
                self._exhaust_locked(UnattendedStopReason.MAX_INPUT_BYTES)
                return self._record_locked(before)
        try:
            prepared = self._prepare(proposal, context, token)
        except Exception:
            return self._halt_host_error(before)
        if token.is_set():
            return self._finish_token(before, token)
        if not isinstance(prepared, PreparedResearchAction) or prepared.tool_name != proposal.host_tool_name:
            return self._finish_refusal(before, "host_refused", proposal)
        action_digest = _digest({"tool_name": prepared.tool_name, "arguments": dict(prepared.arguments)})
        with self._lock:
            if action_digest in self._recent_actions:
                self._exhaust_locked(UnattendedStopReason.REPEATED_ACTION)
                return self._record_locked(before)
            if self._calls + 1 > self.MAX_CALLS:
                self._exhaust_locked(UnattendedStopReason.MAX_CALLS)
                return self._record_locked(before)
            self._calls += 1
            self._input_bytes += input_bytes
        try:
            execution = self._execute(prepared, token)
        except Exception:
            return self._halt_host_error(before)
        if token.is_set():
            return self._finish_token(before, token)
        if not isinstance(execution, ResearchExecution):
            return self._halt_host_error(before)
        with self._lock:
            if execution.output_bytes + self._output_bytes > self.MAX_OUTPUT_BYTES:
                self._exhaust_locked(UnattendedStopReason.MAX_OUTPUT_BYTES)
                return self._record_locked(before)
            self._output_bytes += execution.output_bytes
            self._ticks += 1
            self._recent_actions.add(action_digest)
            self._no_progress = 0 if execution.made_progress else self._no_progress + 1
            if self._clock() >= token.deadline:
                self._exhaust_locked(UnattendedStopReason.MAX_TICK_SECONDS)
            elif self._no_progress >= self.MAX_NO_PROGRESS:
                self._exhaust_locked(UnattendedStopReason.NO_PROGRESS)
            receipt = ActionReceipt(self._ticks, action_digest, prepared.tool_name, "executed", execution.status,
                                    input_bytes, execution.output_bytes)
            return self._record_locked(before, receipt)

    def run(self) -> Tuple[TickReceipt, ...]:
        """Drive foreground ticks until a terminal/paused state; no sleep/thread."""
        produced = []
        while self.state == UnattendedState.RUNNING:
            receipt = self.step()
            produced.append(receipt)
            if receipt.state_before == receipt.state_after == UnattendedState.RUNNING and receipt.action is None:
                break
        return tuple(produced)

    def _gate_reason_locked(self) -> Optional[UnattendedStopReason]:
        if self._state != UnattendedState.RUNNING:
            return None
        if self._sleeping():
            return UnattendedStopReason.SLEEP
        if not self._quota_available():
            return UnattendedStopReason.QUOTA_EXHAUSTED
        return None

    def _validate_proposal(self, proposal: ResearchProposal, policy: UnattendedPolicy) -> Optional[str]:
        tool = proposal.host_tool_name
        if tool not in policy.allowed_tools or tool not in READ_ONLY_TOOLS:
            return "tool_not_in_read_only_profile"
        if _has_authority_claim(proposal.arguments) or _has_authority_claim(proposal.public_rationale):
            return "model_cannot_authorize_or_renew"
        if tool in ("web.fetch", "browser.read") and not _public_url(proposal.arguments.get("url")):
            return "private_or_non_https_url"
        return None

    def _finish_no_work(self, before: UnattendedState) -> TickReceipt:
        with self._lock:
            self._exhaust_locked(UnattendedStopReason.NO_WORK)
            return self._record_locked(before)

    def _finish_refusal(self, before: UnattendedState, disposition: str,
                         proposal: Optional[ResearchProposal] = None) -> TickReceipt:
        with self._lock:
            self._ticks += 1
            self._no_progress += 1
            if self._no_progress >= self.MAX_NO_PROGRESS:
                self._exhaust_locked(UnattendedStopReason.NO_PROGRESS)
            action = ActionReceipt(self._ticks, _digest(dict(proposal.arguments)) if proposal else _digest(disposition),
                                   proposal.host_tool_name if proposal else "none", disposition, "refused", 0, 0)
            return self._record_locked(before, action)

    def _finish_token(self, before: UnattendedState, token: CancellationToken) -> TickReceipt:
        with self._lock:
            if self._cancel.is_set():
                if self._state == UnattendedState.RUNNING:
                    self._state, self._reason = UnattendedState.STOPPED, UnattendedStopReason.USER_STOP
            elif self._clock() >= token.deadline:
                self._exhaust_locked(UnattendedStopReason.MAX_TICK_SECONDS)
            return self._record_locked(before)

    def _halt_host_error(self, before: UnattendedState) -> TickReceipt:
        with self._lock:
            self._exhaust_locked(UnattendedStopReason.HOST_ERROR)
            return self._record_locked(before)

    def _exhaust_locked(self, reason: UnattendedStopReason) -> None:
        self._cancel.set()
        self._state, self._reason = UnattendedState.EXHAUSTED, reason

    def _record_locked(self, before: UnattendedState,
                       action: Optional[ActionReceipt] = None) -> TickReceipt:
        receipt = TickReceipt(self._ticks, before, self._state, self._reason, action)
        self._receipts.append(receipt)
        if len(self._receipts) > self.MAX_RECEIPTS:
            del self._receipts[:-self.MAX_RECEIPTS]
        return receipt
