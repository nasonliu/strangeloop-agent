"""Provider quota controls with explicit telemetry provenance.

This module regulates externally observable provider capacity.  It is not a
model of a mental state, nor an analogue for a Yogacara term.  In particular,
untrusted model, tool, and web content cannot alter its accounting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
import threading
from typing import Any, Dict, Mapping, Optional, Union
from uuid import uuid4


class QuotaSource(str, Enum):
    """The bounded classes of quota evidence understood by the controller."""

    AUTHORITATIVE_HEADER = "authoritative_header"
    PROVIDER_USAGE = "provider_usage"
    LOCAL_LEDGER = "local_ledger"
    MANUAL_SNAPSHOT = "manual_snapshot"
    ERROR_SIGNAL = "error_signal"


class ReservationInvalidationReason(str, Enum):
    """Inspectable causes for ending a reservation lifecycle epoch.

    These values describe controller lifecycle events, not provider outcomes or
    any motivational/reward signal.  Callers must choose one explicitly so an
    exported invalidation has bounded, reviewable provenance.
    """

    AUTHORITATIVE_SNAPSHOT = "authoritative_snapshot"
    SLEEP = "sleep"
    TERMINAL = "terminal"
    PURGE = "purge"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be a timezone-aware datetime" % name)
    return value.astimezone(timezone.utc)


def _non_negative(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("%s must be a non-negative integer or None" % name)
    return value


@dataclass(frozen=True)
class QuotaSnapshot:
    """A concise, inspectable provider-capacity observation.

    ``total``/``remaining`` are the primary quota dimension.  Token and call
    dimensions are optional because providers do not expose a uniform schema.
    No headers, credentials, or raw error bodies are retained here.
    """

    total: Optional[int]
    remaining: Optional[int]
    reset_at: datetime
    observed_at: datetime
    confidence: float
    is_estimate: bool
    token_total: Optional[int] = None
    token_remaining: Optional[int] = None
    call_total: Optional[int] = None
    call_remaining: Optional[int] = None
    # Generic provider balances are not necessarily token counts.  The
    # default keeps existing host-normalized token semantics compatible.
    primary_unit: str = "tokens"
    # Managed providers can expose multiple independent quota windows.  This
    # identifies the selected primary window without changing legacy payloads
    # that did not carry a window identity.
    primary_window_kind: Optional[str] = None

    def __post_init__(self) -> None:
        for value, name in ((self.total, "total"), (self.remaining, "remaining"),
                            (self.token_total, "token_total"), (self.token_remaining, "token_remaining"),
                            (self.call_total, "call_total"), (self.call_remaining, "call_remaining")):
            _non_negative(value, name)
        if self.total is not None and self.remaining is not None and self.remaining > self.total:
            raise ValueError("remaining cannot exceed total")
        for total, remaining, label in ((self.token_total, self.token_remaining, "token"),
                                        (self.call_total, self.call_remaining, "call")):
            if total is not None and remaining is not None and remaining > total:
                raise ValueError("%s_remaining cannot exceed %s_total" % (label, label))
        _aware(self.reset_at, "reset_at")
        _aware(self.observed_at, "observed_at")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ValueError("confidence must be a number")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.is_estimate, bool):
            raise ValueError("is_estimate must be a bool")
        if self.primary_unit not in ("tokens", "provider_units"):
            raise ValueError("primary_unit must be 'tokens' or 'provider_units'")
        if self.primary_window_kind not in (None, "weekly", "rolling_5h"):
            raise ValueError("primary_window_kind must be None, 'weekly', or 'rolling_5h'")


@dataclass(frozen=True)
class UsageRecord:
    """Token use known after a provider call, without prompt content."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        for value, name in ((self.prompt_tokens, "prompt_tokens"),
                            (self.completion_tokens, "completion_tokens"),
                            (self.reasoning_tokens, "reasoning_tokens"),
                            (self.cached_tokens, "cached_tokens")):
            _non_negative(value, name)

    @property
    def charged_tokens(self) -> int:
        return max(0, self.prompt_tokens + self.completion_tokens + self.reasoning_tokens - self.cached_tokens)


def usage_record_from_provider_response(payload: Mapping[str, Any]) -> UsageRecord:
    """Extract a local usage record from a successful provider response.

    This deliberately understands only token *usage*, not a plan balance.  A
    response ``usage`` object is evidence that one request consumed resources;
    it is not evidence of the remaining Kimi Code Plan quota.
    """
    if not isinstance(payload, Mapping):
        return UsageRecord()
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return UsageRecord()

    def number(*names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    details = usage.get("completion_tokens_details")
    reasoning = 0
    if isinstance(details, Mapping):
        value = details.get("reasoning_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            reasoning = value
    cached_details = usage.get("prompt_tokens_details")
    cached = 0
    if isinstance(cached_details, Mapping):
        value = cached_details.get("cached_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            cached = value
    return UsageRecord(
        prompt_tokens=number("prompt_tokens", "input_tokens"),
        completion_tokens=number("completion_tokens", "output_tokens"),
        reasoning_tokens=reasoning,
        cached_tokens=cached,
    )


def quota_snapshot_from_usage_payload(payload: Mapping[str, Any],
                                      observed_at: Optional[datetime] = None) -> QuotaSnapshot:
    """Parse a host-normalized ``/usages``-style quota observation.

    Kimi Code does not provide a documented, credential-compatible balance API
    for this project.  This parser is therefore intentionally transport-free:
    an integrator may use it only after independently verifying an endpoint and
    normalizing its response into the explicit fields below.  It must never be
    used to turn a local token ledger into a claimed Code Plan balance.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("usage payload must be a mapping")
    moment = _aware(observed_at or _utc_now(), "observed_at")
    reset_at = payload.get("reset_at")
    if isinstance(reset_at, str):
        try:
            reset_at = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("reset_at must be an ISO timestamp") from error
    if reset_at is None:
        raise ValueError("usage payload must include reset_at")
    if not isinstance(reset_at, datetime):
        raise ValueError("reset_at must be a datetime or ISO timestamp")
    return QuotaSnapshot(
        total=payload.get("total"), remaining=payload.get("remaining"),
        reset_at=_aware(reset_at, "reset_at"), observed_at=moment,
        confidence=payload.get("confidence", 1.0),
        is_estimate=payload.get("is_estimate", False),
        token_total=payload.get("token_total"), token_remaining=payload.get("token_remaining"),
        call_total=payload.get("call_total"), call_remaining=payload.get("call_remaining"),
        primary_unit=payload.get("primary_unit", "tokens"),
        primary_window_kind=payload.get("primary_window_kind"),
    )


@dataclass(frozen=True)
class QuotaPolicy:
    """Deterministic conservation limits, not engagement or survival goals."""

    soft_threshold: float = 0.20
    hard_threshold: float = 0.05
    max_staleness: timedelta = timedelta(minutes=15)
    max_future_observed_skew: timedelta = timedelta(minutes=2)
    # Kimi's human-facing reset hints may be rounded to whole minutes.  This
    # tolerance accepts only a tiny backward wobble within the same observed
    # provider window, while preserving the prior (later) reset boundary.
    max_provider_reset_hint_regression: timedelta = timedelta(seconds=120)
    normal_max_completion_tokens: int = 4096
    normal_max_tool_steps: int = 8

    def __post_init__(self) -> None:
        for value, name in ((self.soft_threshold, "soft_threshold"), (self.hard_threshold, "hard_threshold")):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
                raise ValueError("%s must be between 0 and 1" % name)
        if self.hard_threshold > self.soft_threshold:
            raise ValueError("hard_threshold cannot exceed soft_threshold")
        if not isinstance(self.max_staleness, timedelta) or self.max_staleness <= timedelta(0):
            raise ValueError("max_staleness must be positive")
        if (not isinstance(self.max_future_observed_skew, timedelta) or
                self.max_future_observed_skew < timedelta(0)):
            raise ValueError("max_future_observed_skew must be non-negative")
        if (not isinstance(self.max_provider_reset_hint_regression, timedelta)
                or self.max_provider_reset_hint_regression < timedelta(0)
                or self.max_provider_reset_hint_regression > timedelta(minutes=5)):
            raise ValueError("max_provider_reset_hint_regression must be between zero and five minutes")
        for value, name in ((self.normal_max_completion_tokens, "normal_max_completion_tokens"),
                            (self.normal_max_tool_steps, "normal_max_tool_steps")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("%s must be a positive integer" % name)


@dataclass(frozen=True)
class QuotaDecision:
    allow_call: bool
    reasoning_effort: str
    max_completion_tokens: int
    max_tool_steps: int
    reason: str
    must_pause_loop: bool


@dataclass(frozen=True)
class ForegroundRefreshPolicy:
    """Bounded cadence for a caller-driven provider-usage refresh.

    This is deliberately a policy object, not a scheduler: callers invoke the
    refresh at safe foreground slice boundaries.  ``archive_threshold`` is a
    checkpoint boundary for a provider allowance window, not a billing or
    reward signal.
    """

    minimum_interval: timedelta = timedelta(seconds=30)
    retry_delay: timedelta = timedelta(seconds=5)
    max_transient_retries: int = 1
    max_staleness: timedelta = timedelta(minutes=15)
    archive_threshold: float = 0.10

    def __post_init__(self) -> None:
        for value, name in ((self.minimum_interval, "minimum_interval"),
                            (self.retry_delay, "retry_delay"),
                            (self.max_staleness, "max_staleness")):
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise ValueError("%s must be positive" % name)
        if (not isinstance(self.max_transient_retries, int)
                or isinstance(self.max_transient_retries, bool)
                or not 0 <= self.max_transient_retries <= 3):
            raise ValueError("max_transient_retries must be an integer between 0 and 3")
        if (not isinstance(self.archive_threshold, (int, float))
                or isinstance(self.archive_threshold, bool)
                or not 0 <= float(self.archive_threshold) <= 1):
            raise ValueError("archive_threshold must be between 0 and 1")


@dataclass(frozen=True)
class ForegroundRefreshStatus:
    """Redacted outcome of one foreground quota-refresh decision.

    ``automatic_allowed`` is intentionally stricter than interactive
    :meth:`QuotaController.decision`: no fresh provider observation means an
    unattended caller must stop rather than infer capacity from local usage.
    ``cost_status`` makes explicit that Code Plan allowance units are not a
    price or currency measurement.
    """

    attempted: bool
    accepted: bool
    automatic_allowed: bool
    archive_threshold_reached: bool
    reason: str
    error_category: Optional[str]
    observed_at: Optional[datetime]
    next_refresh_at: Optional[datetime]
    cost_status: str
    # This is only a KimiCliUsageResult, whose schema excludes OAuth tokens,
    # bearer headers, and raw HTTP bodies.  It is optional because a skipped
    # slice may have no still-valid successful bridge observation to reuse.
    usage_result: Optional[Any] = None
    # Appended after the existing positional API for compatibility with
    # injected foreground adapters in tests and embedders.
    degraded: bool = False
    cache_used: bool = False
    next_retry_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not all(isinstance(value, bool) for value in (
                self.attempted, self.accepted, self.automatic_allowed,
                self.archive_threshold_reached, self.degraded, self.cache_used)):
            raise ValueError("foreground refresh flags must be bools")
        for value, name in ((self.observed_at, "observed_at"),
                            (self.next_refresh_at, "next_refresh_at")):
            if value is not None:
                _aware(value, name)
        if self.next_retry_at is not None:
            _aware(self.next_retry_at, "next_retry_at")
        if self.error_category is not None and (not isinstance(self.error_category, str)
                                                or not self.error_category):
            raise ValueError("error_category must be a non-empty string or None")
        if self.cost_status not in ("provider_plan_units_not_currency",
                                    "unknown_provider_plan_cost"):
            raise ValueError("cost_status is invalid")
        if self.usage_result is not None:
            # Keep the dependency lazy: kimi_cli imports QuotaController at
            # module import time, while this result is only constructed after
            # the bridge has completed its redacted normalization.
            from .kimi_cli import KimiCliUsageResult
            if not isinstance(self.usage_result, KimiCliUsageResult):
                raise TypeError("usage_result must be a KimiCliUsageResult or None")


class ForegroundRefreshGate:
    """State for synchronous refresh callers; it never starts work itself."""

    def __init__(self, policy: Optional[ForegroundRefreshPolicy] = None) -> None:
        self.policy = policy or ForegroundRefreshPolicy()
        if not isinstance(self.policy, ForegroundRefreshPolicy):
            raise TypeError("policy must be a ForegroundRefreshPolicy")
        self._lock = threading.RLock()
        self._last_attempt_at: Optional[datetime] = None
        self._last_success_at: Optional[datetime] = None
        self._last_archive_threshold_reached = False
        self._retry_at: Optional[datetime] = None
        self._transient_failures = 0

    def refresh_due(self, now: Optional[datetime] = None, *, force: bool = False) -> bool:
        """Return whether a foreground caller may perform one new refresh."""
        moment = _aware(now or _utc_now(), "now")
        if not isinstance(force, bool):
            raise TypeError("force must be a bool")
        with self._lock:
            if self._retry_at is not None:
                return force or moment >= self._retry_at
            return force or self._last_attempt_at is None or (
                moment - self._last_attempt_at >= self.policy.minimum_interval)

    def claim_refresh(self, now: Optional[datetime] = None, *, force: bool = False) -> bool:
        """Atomically claim one refresh slot for a foreground caller.

        This prevents two concurrent callers from both starting a loopback
        bridge during the same cadence interval.  It remains caller-driven;
        claiming a slot performs no I/O.
        """
        moment = _aware(now or _utc_now(), "now")
        if not isinstance(force, bool):
            raise TypeError("force must be a bool")
        with self._lock:
            due = (force or moment >= self._retry_at) if self._retry_at is not None else (
                force or self._last_attempt_at is None or
                moment - self._last_attempt_at >= self.policy.minimum_interval)
            if due:
                self._last_attempt_at = moment
                self._retry_at = None
            return due

    def record_transient_failure(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """Schedule at most the configured number of caller-driven retries."""
        moment = _aware(now or _utc_now(), "now")
        with self._lock:
            self._transient_failures += 1
            if self._transient_failures > self.policy.max_transient_retries:
                self._retry_at = None
                return None
            self._retry_at = moment + self.policy.retry_delay
            return self._retry_at

    def record_attempt(self, now: Optional[datetime] = None) -> None:
        moment = _aware(now or _utc_now(), "now")
        with self._lock:
            if self._last_attempt_at is not None and moment < self._last_attempt_at:
                raise ValueError("refresh attempt time cannot move backwards")
            self._last_attempt_at = moment

    def record_success(self, observed_at: datetime,
                       archive_threshold_reached: Optional[bool] = None) -> None:
        observed = _aware(observed_at, "observed_at")
        if archive_threshold_reached is not None and not isinstance(archive_threshold_reached, bool):
            raise TypeError("archive_threshold_reached must be a bool or None")
        with self._lock:
            if self._last_success_at is not None and observed < self._last_success_at:
                raise ValueError("refresh success time cannot move backwards")
            self._last_success_at = observed
            self._retry_at = None
            self._transient_failures = 0
            if archive_threshold_reached is not None:
                self._last_archive_threshold_reached = archive_threshold_reached

    def archive_threshold_reached(self) -> bool:
        with self._lock:
            return self._last_archive_threshold_reached

    def next_refresh_at(self) -> Optional[datetime]:
        with self._lock:
            return (None if self._last_attempt_at is None else
                    self._last_attempt_at + self.policy.minimum_interval)

    def next_retry_at(self) -> Optional[datetime]:
        with self._lock:
            return self._retry_at

@dataclass(frozen=True)
class AuthoritativeWakeEvidence:
    """Strict, inspectable evidence for an external scheduler to auto-wake.

    This is deliberately narrower than :meth:`decision`: interactive use can
    fall back when telemetry is unknown, while an automatic wake needs a
    current, provider-authoritative observation with explicit provenance.
    """

    allow: bool
    reason: str
    snapshot: Optional[QuotaSnapshot]


@dataclass(frozen=True)
class QuotaReservation:
    reservation_id: str
    created_at: datetime
    # ``None`` keeps direct construction source-compatible while making a
    # reservation without controller-issued lifecycle provenance unusable.
    generation: Optional[int] = None


class QuotaController:
    """Thread-safe controller for provider telemetry and locally known usage."""

    def __init__(self, policy: Optional[QuotaPolicy] = None) -> None:
        self.policy = policy or QuotaPolicy()
        self._lock = threading.RLock()
        self._snapshot: Optional[QuotaSnapshot] = None
        self._snapshot_source: Optional[QuotaSource] = None
        self._ledger_tokens = 0
        self._ledger_calls = 0
        self._reservations: Dict[str, QuotaReservation] = {}
        self._reservation_generation = 0
        self._reservation_invalidation_reason: Optional[ReservationInvalidationReason] = None
        self._pause_reason: Optional[str] = None
        self._pause_observed_at: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

    @staticmethod
    def _origin_name(origin: Any) -> str:
        return str(getattr(origin, "value", origin)).lower()

    def ingest_snapshot(self, snapshot: QuotaSnapshot, source: QuotaSource,
                        origin: Any, authenticated: bool = False,
                        now: Optional[datetime] = None) -> bool:
        """Accept only SYSTEM provider telemetry or authenticated USER input.

        Returns ``False`` for untrusted provenance instead of allowing model or
        tool text to become an exception-driven control path.
        """
        if not isinstance(snapshot, QuotaSnapshot):
            raise TypeError("snapshot must be a QuotaSnapshot")
        if not isinstance(source, QuotaSource):
            raise TypeError("source must be a QuotaSource")
        origin_name = self._origin_name(origin)
        allowed = ((origin_name == "system" and source in (
            QuotaSource.AUTHORITATIVE_HEADER, QuotaSource.PROVIDER_USAGE, QuotaSource.ERROR_SIGNAL)) or
            (origin_name == "user" and authenticated and source == QuotaSource.MANUAL_SNAPSHOT))
        if not allowed:
            return False
        moment = _aware(now or _utc_now(), "now")
        with self._lock:
            observed_at = _aware(snapshot.observed_at, "observed_at")
            if observed_at > moment + self.policy.max_future_observed_skew:
                return False
            previous = self._snapshot
            if previous is not None:
                previous_observed_at = _aware(previous.observed_at, "observed_at")
                if observed_at < previous_observed_at:
                    return False
                previous_reset = _aware(previous.reset_at, "reset_at")
                snapshot_reset = _aware(snapshot.reset_at, "reset_at")
                explicit_window_switch = (
                    previous.primary_window_kind is not None and
                    snapshot.primary_window_kind is not None and
                    previous.primary_window_kind != snapshot.primary_window_kind
                )
                if explicit_window_switch and not (
                        source == QuotaSource.PROVIDER_USAGE and
                        observed_at > previous_observed_at and
                        snapshot_reset > observed_at):
                    return False
                # Different explicit managed windows can legitimately have
                # reset boundaries in either order.  Legacy snapshots remain
                # deliberately strict because their window identity is not
                # auditable.
                same_or_legacy_window = (
                    previous.primary_window_kind is None or
                    snapshot.primary_window_kind is None or
                    previous.primary_window_kind == snapshot.primary_window_kind
                )
                if snapshot_reset < previous_reset and same_or_legacy_window:
                    # A provider can establish a new post-reset window, but a
                    # delayed response must never roll the known window back.
                    provider_new_window = (
                        source in (QuotaSource.AUTHORITATIVE_HEADER, QuotaSource.PROVIDER_USAGE) and
                        observed_at > previous_reset and snapshot_reset > observed_at
                    )
                    # A Kimi managed-usage reset hint is rounded by the
                    # provider.  Within a still-active window, accept a small
                    # rounded regression only from fresh provider telemetry,
                    # but retain the existing later reset.  No snapshot can
                    # shorten a known sleep/wake boundary.
                    rounded_provider_hint = (
                        source == QuotaSource.PROVIDER_USAGE
                        and observed_at > previous_observed_at
                        and observed_at < previous_reset and snapshot_reset > observed_at
                        and previous_reset - snapshot_reset <= self.policy.max_provider_reset_hint_regression
                    )
                    if provider_new_window:
                        pass
                    elif rounded_provider_hint:
                        snapshot = replace(snapshot, reset_at=previous_reset)
                    else:
                        return False
            self._snapshot = snapshot
            self._snapshot_source = source
            # A reconciled provider view begins a new lifecycle epoch.  A
            # pre-snapshot in-flight request must not commit local usage into
            # the reconciled ledger afterward.
            self._invalidate_reservations_locked(ReservationInvalidationReason.AUTHORITATIVE_SNAPSHOT)
            # A current provider/manual observation supersedes local estimates.
            self._ledger_tokens = 0
            self._ledger_calls = 0
            # A quota exhaustion epoch is only superseded by a newer provider
            # observation.  Manual or delayed snapshots cannot auto-unpause.
            if (self._pause_reason is None or
                    (source in (QuotaSource.AUTHORITATIVE_HEADER, QuotaSource.PROVIDER_USAGE) and
                     (self._pause_observed_at is None or observed_at > self._pause_observed_at))):
                self._pause_reason = None
                self._pause_observed_at = None
            self._cooldown_until = None
        return True

    def canonical_provider_snapshot(self) -> Optional[QuotaSnapshot]:
        """Return the controller's accepted provider snapshot, never raw input.

        This read-only copy point lets a bridge return exactly the reset
        boundary that the controller accepted after bounded reset-hint
        canonicalization.  Headers, manual input, and local ledgers are not
        exposed as provider-managed usage evidence.
        """
        with self._lock:
            if self._snapshot_source != QuotaSource.PROVIDER_USAGE:
                return None
            return self._snapshot

    def ingest_error(self, status_code: int, origin: Any, quota_specific: bool = False,
                     retry_after: Optional[timedelta] = None, now: Optional[datetime] = None) -> bool:
        """Ingest a redacted SYSTEM error classification, never a raw response."""
        if self._origin_name(origin) != "system" or not isinstance(status_code, int):
            return False
        moment = _aware(now or _utc_now(), "now")
        with self._lock:
            if status_code == 402 or (status_code == 429 and quota_specific):
                self._pause_reason = "provider_quota_exhausted"
                self._pause_observed_at = moment
                self._cooldown_until = None
            elif status_code == 429:
                wait = retry_after if isinstance(retry_after, timedelta) and retry_after > timedelta(0) else timedelta(minutes=1)
                self._cooldown_until = moment + wait
            return True

    def authoritative_wake_evidence(self, now: Optional[datetime] = None,
                                    required_source: Optional[QuotaSource] = None,
                                    minimum_observed_at: Optional[datetime] = None) -> AuthoritativeWakeEvidence:
        """Return whether an unattended wake has current authoritative evidence.

        It performs no network activity and does not reserve or consume quota.
        ``required_source`` makes the scheduler's provenance requirement
        explicit; model, tool, user, local-ledger, and estimate evidence can
        never satisfy this boundary.
        """
        moment = _aware(now or _utc_now(), "now")
        if required_source is not None and not isinstance(required_source, QuotaSource):
            raise TypeError("required_source must be a QuotaSource or None")
        minimum = None if minimum_observed_at is None else _aware(minimum_observed_at, "minimum_observed_at")
        provider_sources = (QuotaSource.AUTHORITATIVE_HEADER, QuotaSource.PROVIDER_USAGE)
        with self._lock:
            snap = self._snapshot
            source = self._snapshot_source
            if required_source is not None and required_source not in provider_sources:
                return AuthoritativeWakeEvidence(False, "required_source_not_provider_authoritative", snap)
            if self._pause_reason:
                return AuthoritativeWakeEvidence(False, self._pause_reason, snap)
            if self._cooldown_until is not None and moment < self._cooldown_until:
                return AuthoritativeWakeEvidence(False, "provider_rate_limit_cooldown", snap)
            if snap is None:
                return AuthoritativeWakeEvidence(False, "no_authoritative_quota_telemetry", None)
            if source not in provider_sources:
                return AuthoritativeWakeEvidence(False, "snapshot_source_not_provider_authoritative", snap)
            if required_source is not None and source != required_source:
                return AuthoritativeWakeEvidence(False, "snapshot_source_mismatch", snap)
            observed_at = _aware(snap.observed_at, "observed_at")
            if observed_at > moment + self.policy.max_future_observed_skew:
                return AuthoritativeWakeEvidence(False, "quota_telemetry_future", snap)
            if minimum is not None and observed_at < minimum:
                return AuthoritativeWakeEvidence(False, "quota_telemetry_before_minimum", snap)
            if moment - observed_at > self.policy.max_staleness:
                return AuthoritativeWakeEvidence(False, "quota_telemetry_stale", snap)
            if moment >= _aware(snap.reset_at, "reset_at"):
                return AuthoritativeWakeEvidence(False, "quota_reset_requires_fresh_telemetry", snap)
            if snap.is_estimate:
                return AuthoritativeWakeEvidence(False, "quota_snapshot_estimated", snap)
            values = self._effective()
            if self._is_exhausted(values):
                return AuthoritativeWakeEvidence(False, "provider_quota_exhausted", snap)
            return AuthoritativeWakeEvidence(True, "authoritative_quota_available", snap)

    def _effective(self) -> Dict[str, Optional[int]]:
        snap = self._snapshot
        if snap is None:
            return {"total": None, "remaining": None, "token_total": None,
                    "token_remaining": None, "call_total": None, "call_remaining": None}
        # Only values explicitly measured in tokens consume the local token
        # ledger.  A generic provider metric remains provider-authoritative.
        primary_remaining = snap.remaining
        if primary_remaining is not None and snap.primary_unit == "tokens":
            primary_remaining = max(0, primary_remaining - self._ledger_tokens)
        token_remaining = (None if snap.token_remaining is None else
                           max(0, snap.token_remaining - self._ledger_tokens))
        calls = None if snap.call_remaining is None else max(0, snap.call_remaining - self._ledger_calls - len(self._reservations))
        return {"total": snap.total, "remaining": primary_remaining,
                "token_total": snap.token_total, "token_remaining": token_remaining,
                "call_total": snap.call_total, "call_remaining": calls}

    @staticmethod
    def _is_exhausted(values: Mapping[str, Optional[int]]) -> bool:
        return (values["remaining"] == 0 or values["token_remaining"] == 0 or
                values["call_remaining"] == 0)

    def decision(self, now: Optional[datetime] = None) -> QuotaDecision:
        moment = _aware(now or _utc_now(), "now")
        with self._lock:
            if self._pause_reason:
                return self._deny(self._pause_reason)
            if self._cooldown_until is not None and moment < self._cooldown_until:
                return self._deny("provider_rate_limit_cooldown")
            snap = self._snapshot
            if snap is None:
                return self._allow("high", self.policy.normal_max_completion_tokens, self.policy.normal_max_tool_steps,
                                   "no_trusted_quota_telemetry")
            if moment - _aware(snap.observed_at, "observed_at") > self.policy.max_staleness:
                return self._deny("quota_telemetry_stale")
            if moment >= _aware(snap.reset_at, "reset_at"):
                return self._deny("quota_reset_requires_fresh_telemetry")
            values = self._effective()
            if self._is_exhausted(values):
                return self._deny("provider_quota_exhausted")
            fraction = self._fraction(values["remaining"], values["total"],
                                      values["call_remaining"], values["call_total"],
                                      values["token_remaining"], values["token_total"])
            if fraction is not None and fraction <= self.policy.hard_threshold:
                return self._allow("low", max(1, self.policy.normal_max_completion_tokens // 4),
                                   max(1, self.policy.normal_max_tool_steps // 4), "quota_hard_conservation_band")
            if fraction is not None and fraction <= self.policy.soft_threshold:
                return self._allow("low", max(1, self.policy.normal_max_completion_tokens // 2),
                                   max(1, self.policy.normal_max_tool_steps // 2), "quota_soft_conservation_band")
            return self._allow("high", self.policy.normal_max_completion_tokens, self.policy.normal_max_tool_steps,
                               "quota_available")

    @staticmethod
    def _fraction(remaining: Optional[int], total: Optional[int], call_remaining: Optional[int],
                  call_total: Optional[int], token_remaining: Optional[int] = None,
                  token_total: Optional[int] = None) -> Optional[float]:
        values = []
        if remaining is not None and total not in (None, 0):
            values.append(float(remaining) / total)
        if call_remaining is not None and call_total not in (None, 0):
            values.append(float(call_remaining) / call_total)
        if token_remaining is not None and token_total not in (None, 0):
            values.append(float(token_remaining) / token_total)
        return min(values) if values else None

    @staticmethod
    def _allow(effort: str, tokens: int, steps: int, reason: str) -> QuotaDecision:
        return QuotaDecision(True, effort, tokens, steps, reason, False)

    @staticmethod
    def _deny(reason: str) -> QuotaDecision:
        return QuotaDecision(False, "low", 0, 0, reason, True)

    def reserve_call(self, now: Optional[datetime] = None) -> Optional[QuotaReservation]:
        """Atomically reserve one provider call.  Call ``commit`` or ``release``."""
        with self._lock:
            if not self.decision(now).allow_call:
                return None
            reservation = QuotaReservation(uuid4().hex, _aware(now or _utc_now(), "now"),
                                           self._reservation_generation)
            self._reservations[reservation.reservation_id] = reservation
            # Defensive recheck is unnecessary under the same RLock, but makes
            # the boundary explicit if decision logic changes later.
            return reservation

    def commit(self, reservation: Union[QuotaReservation, str], usage: UsageRecord) -> bool:
        """Commit one completed reservation and its concise local usage."""
        if not isinstance(usage, UsageRecord):
            raise TypeError("usage must be a UsageRecord")
        with self._lock:
            if not self._is_current_reservation_locked(reservation):
                return False
            del self._reservations[reservation.reservation_id]
            self._ledger_calls += 1
            self._ledger_tokens += usage.charged_tokens
            return True

    def release(self, reservation: Union[QuotaReservation, str]) -> bool:
        """Release an unused slot; releasing twice is harmless and returns false."""
        with self._lock:
            if not self._is_current_reservation_locked(reservation):
                return False
            del self._reservations[reservation.reservation_id]
            return True

    def invalidate_reservations(self, reason: ReservationInvalidationReason,
                                generation: Optional[int] = None,
                                epoch: Optional[int] = None) -> int:
        """Atomically end the current reservation epoch and return the next one.

        ``generation`` and ``epoch`` are equivalent optional lower bounds for
        integration with lifecycle coordinators.  Supplying both is permitted
        only when they agree.  The returned generation always advances
        monotonically, even if there are no outstanding reservations.
        """
        if not isinstance(reason, ReservationInvalidationReason):
            raise TypeError("reason must be a ReservationInvalidationReason")
        requested = self._requested_generation(generation, epoch)
        with self._lock:
            return self._invalidate_reservations_locked(reason, requested)

    @staticmethod
    def _requested_generation(generation: Optional[int], epoch: Optional[int]) -> Optional[int]:
        for value, name in ((generation, "generation"), (epoch, "epoch")):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError("%s must be a non-negative integer or None" % name)
        if generation is not None and epoch is not None and generation != epoch:
            raise ValueError("generation and epoch must agree when both are supplied")
        return generation if generation is not None else epoch

    def _invalidate_reservations_locked(self, reason: ReservationInvalidationReason,
                                        requested_generation: Optional[int] = None) -> int:
        """Advance the lifecycle epoch.  Caller must hold ``_lock``."""
        next_generation = self._reservation_generation + 1
        if requested_generation is not None:
            next_generation = max(next_generation, requested_generation)
        self._reservation_generation = next_generation
        self._reservations.clear()
        self._reservation_invalidation_reason = reason
        return self._reservation_generation

    def _is_current_reservation_locked(self, reservation: Union[QuotaReservation, str]) -> bool:
        """Return whether an issued object belongs to the active epoch.

        ID-only completion is deliberately rejected: it lacks the lifecycle
        provenance needed to prevent a reservation from crossing an epoch.
        """
        if not isinstance(reservation, QuotaReservation):
            return False
        if reservation.generation is None or reservation.generation != self._reservation_generation:
            return False
        return self._reservations.get(reservation.reservation_id) == reservation

    def export_telemetry(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Return restart-reconstructable, safe telemetry without secrets/raw errors."""
        moment = _aware(now or _utc_now(), "now")
        with self._lock:
            snap = self._snapshot
            result: Dict[str, Any] = {
                "source": self._snapshot_source.value if self._snapshot_source else None,
                "ledger_tokens": self._ledger_tokens,
                "ledger_calls": self._ledger_calls,
                "reserved_calls": len(self._reservations),
                "reservation_generation": self._reservation_generation,
                "reservation_invalidation_reason": (
                    self._reservation_invalidation_reason.value
                    if self._reservation_invalidation_reason is not None else None),
                "pause_reason": self._pause_reason,
                "pause_observed_at": (self._pause_observed_at.isoformat()
                                      if self._pause_observed_at is not None else None),
                "cooldown_active": bool(self._cooldown_until and moment < self._cooldown_until),
            }
            if snap is not None:
                result["snapshot"] = {
                    "total": snap.total, "remaining": snap.remaining,
                    "primary_unit": snap.primary_unit,
                    "primary_window_kind": snap.primary_window_kind,
                    "token_total": snap.token_total, "token_remaining": snap.token_remaining,
                    "call_total": snap.call_total, "call_remaining": snap.call_remaining,
                    "reset_at": _aware(snap.reset_at, "reset_at").isoformat(),
                    "observed_at": _aware(snap.observed_at, "observed_at").isoformat(),
                    "confidence": snap.confidence, "is_estimate": snap.is_estimate,
                }
            return result
