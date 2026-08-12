"""Host-owned quota sleep/wake coordination.

This foreground-only state machine is a scheduling boundary, not a model of a
mental state.  Its names are inspired by Yogacara's emphasis on checking
conditions, but no one-to-one cognitive or subjective-experience claim is
intended.  Callers supply already-authenticated managed-usage observations;
this module never performs network I/O, sleeps, or persists memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import re
import threading
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class SleepState(str, Enum):
    ACTIVE = "active"
    PREPARING = "preparing"
    SLEEPING = "sleeping"
    CHECKING = "checking"
    READY = "ready"
    TERMINAL = "terminal"


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be a timezone-aware datetime" % name)
    return value.astimezone(timezone.utc)


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SleepWakePolicy:
    """Host policy for a single authoritative rolling five-hour window.

    ``threshold`` is the pre-archive coordination threshold.  It decides when
    the host should checkpoint and pause foreground work; it is not a provider
    quota hard stop and cannot be populated from a local usage ledger.
    """

    threshold: float = 0.10
    max_staleness: timedelta = timedelta(minutes=15)
    initial_backoff: timedelta = timedelta(seconds=30)
    max_backoff: timedelta = timedelta(minutes=15)

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, (int, float)) or isinstance(self.threshold, bool) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        for value, name in ((self.max_staleness, "max_staleness"), (self.initial_backoff, "initial_backoff"),
                            (self.max_backoff, "max_backoff")):
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise ValueError("%s must be positive" % name)
        if self.initial_backoff > self.max_backoff:
            raise ValueError("initial_backoff cannot exceed max_backoff")


@dataclass(frozen=True)
class SleepArchive:
    """A bounded public checkpoint; it contains IDs/digests, never content."""

    schema_version: int
    chain_head_sequence: int
    chain_head_hash: str
    active_seed_ids: Tuple[str, ...]
    active_claim_ids: Tuple[str, ...]
    pending_public_event_ids: Tuple[str, ...]
    quota_source: str
    quota_observed_at: str
    quota_reset_at: str
    quota_remaining: int
    quota_total: int
    quota_window_kind: str
    digest: str

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "chain_head_sequence": self.chain_head_sequence,
            "chain_head_hash": self.chain_head_hash,
            "active_seed_ids": list(self.active_seed_ids),
            "active_claim_ids": list(self.active_claim_ids),
            "pending_public_event_ids": list(self.pending_public_event_ids),
            "quota": {"source": self.quota_source, "observed_at": self.quota_observed_at,
                      "reset_at": self.quota_reset_at, "remaining": self.quota_remaining,
                      "total": self.quota_total},
            "digest": self.digest,
        }
        # v1/v2 manifests remain exactly readable for historical verification.
        # New archives make the provider's rolling window explicit.
        if self.schema_version >= 3:
            payload["quota"]["window_kind"] = self.quota_window_kind
        return payload


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


def _ids(values: Iterable[str], name: str, maximum: int = 256) -> Tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if len(result) > maximum or any(not isinstance(item, str) or not _ID.match(item) for item in result):
        raise ValueError("%s must contain at most %d bounded identifiers" % (name, maximum))
    return result


def build_public_archive(*, chain_head_sequence: int, chain_head_hash: str,
                         active_seed_ids: Iterable[str] = (), active_claim_ids: Iterable[str] = (),
                         pending_public_event_ids: Iterable[str] = (), quota_source: str,
                         quota_observed_at: datetime, quota_reset_at: datetime,
                         quota_remaining: int, quota_total: int,
                         quota_window_kind: str = "rolling_5h",
                         schema_version: int = 3) -> SleepArchive:
    """Build a canonical, model-free and side-effect-free public archive."""
    if not isinstance(chain_head_sequence, int) or isinstance(chain_head_sequence, bool) or chain_head_sequence < 0:
        raise ValueError("chain_head_sequence must be a non-negative integer")
    if not isinstance(chain_head_hash, str) or not _DIGEST.match(chain_head_hash):
        raise ValueError("chain_head_hash must be a sha256 digest")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ValueError("schema_version must be positive")
    if not isinstance(quota_window_kind, str) or quota_window_kind != "rolling_5h":
        raise ValueError("quota_window_kind must be rolling_5h")
    if not isinstance(quota_source, str) or not _ID.match(quota_source):
        raise ValueError("quota_source must be a bounded identifier")
    if (not isinstance(quota_remaining, int) or isinstance(quota_remaining, bool) or quota_remaining < 0 or
            not isinstance(quota_total, int) or isinstance(quota_total, bool) or quota_total < quota_remaining):
        raise ValueError("quota counts are invalid")
    observed, reset = _utc(quota_observed_at, "quota_observed_at"), _utc(quota_reset_at, "quota_reset_at")
    public = {
        "schema_version": schema_version, "chain_head_sequence": chain_head_sequence,
        "chain_head_hash": chain_head_hash, "active_seed_ids": list(_ids(active_seed_ids, "active_seed_ids")),
        "active_claim_ids": list(_ids(active_claim_ids, "active_claim_ids")),
        "pending_public_event_ids": list(_ids(pending_public_event_ids, "pending_public_event_ids")),
        "quota": {"source": quota_source, "observed_at": observed.isoformat(), "reset_at": reset.isoformat(),
                  "remaining": quota_remaining, "total": quota_total},
    }
    if schema_version >= 3:
        public["quota"]["window_kind"] = quota_window_kind
    digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    return SleepArchive(schema_version, chain_head_sequence, chain_head_hash, tuple(public["active_seed_ids"]),
                        tuple(public["active_claim_ids"]), tuple(public["pending_public_event_ids"]), quota_source,
                        public["quota"]["observed_at"], public["quota"]["reset_at"], quota_remaining, quota_total,
                        quota_window_kind, digest)


class SleepWakeCoordinator:
    """Caller-driven sleep gate guarded by epoch and compare-and-swap generation."""

    _PAYLOAD_KEYS = frozenset((
        "state", "epoch", "generation", "auto_wake_user_approved",
        "reset_at", "retry_at", "failure_count",
        "last_authoritative_observed_at",
    ))

    def __init__(self, policy: Optional[SleepWakePolicy] = None,
                 clock: Optional[Callable[[], datetime]] = None) -> None:
        self.policy = policy or SleepWakePolicy()
        if not isinstance(self.policy, SleepWakePolicy):
            raise TypeError("policy must be a SleepWakePolicy")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock or _default_clock
        self._lock = threading.RLock()
        self._state = SleepState.ACTIVE
        self._epoch = 0
        self._generation = 0
        self._auto_wake = False
        self._last_observed_at: Optional[datetime] = None
        self._reset_at: Optional[datetime] = None
        self._retry_at: Optional[datetime] = None
        self._failures = 0

    def _now(self) -> datetime:
        return _utc(self._clock(), "clock result")

    @property
    def state(self) -> SleepState:
        with self._lock:
            return self._state

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def authority_token(self) -> Optional[str]:
        with self._lock:
            return self._authority_token_locked()

    def _authority_token_locked(self) -> Optional[str]:
        if self._state in (SleepState.SLEEPING, SleepState.CHECKING):
            return "sleep:%d:%d" % (self._epoch, self._generation)
        if self._state == SleepState.READY:
            return "ready:%d:%d" % (self._epoch, self._generation)
        return None

    def _window(self, windows: Sequence[Any], observed_at: datetime,
                authoritative: bool, now: datetime) -> Optional[Tuple[int, int, datetime, datetime]]:
        if not isinstance(authoritative, bool):
            raise TypeError("authoritative must be a bool")
        observed = _utc(observed_at, "observed_at")
        if not authoritative or observed > now or now - observed > self.policy.max_staleness:
            return None
        if self._last_observed_at is not None and observed <= self._last_observed_at:
            return None
        if isinstance(windows, (str, bytes, bytearray, Mapping)):
            raise ValueError("windows must be a sequence of managed usage windows")
        try:
            matches = [item for item in windows if getattr(item, "kind", None) == "rolling_5h"]
        except TypeError as error:
            raise ValueError("windows must be a sequence of managed usage windows") from error
        if len(matches) != 1:
            return None
        item = matches[0]
        total, remaining = getattr(item, "total", None), getattr(item, "remaining", None)
        reset = getattr(item, "reset_at", None)
        if (not isinstance(total, int) or isinstance(total, bool) or total <= 0 or not isinstance(remaining, int) or
                isinstance(remaining, bool) or not 0 <= remaining <= total):
            return None
        reset = _utc(reset, "window.reset_at")
        if reset <= observed:
            return None
        return total, remaining, reset, observed

    def prepare_sleep(self, windows: Sequence[Any], observed_at: datetime, *, authoritative: bool = True,
                      user_auto_wake: bool = False) -> bool:
        """Atomically enter sleep after a fresh authoritative pre-archive check."""
        if not isinstance(user_auto_wake, bool):
            raise TypeError("user_auto_wake must be a bool")
        with self._lock:
            if self._state != SleepState.ACTIVE:
                return False
            # Validation happens before the first state write.  Consequently a
            # malformed/stale observation cannot expose PREPARING or disturb a
            # concurrent status read.
            values = self._window(windows, observed_at, authoritative, self._now())
            if values is None:
                return False
            total, remaining, reset, observed = values
            if float(remaining) / total > self.policy.threshold:
                return False
            self._state = SleepState.PREPARING
            self._epoch += 1
            self._generation += 1
            # This narrow argument represents already-authenticated USER policy;
            # model/tool/local-ledger values do not reach this host API.
            self._auto_wake = user_auto_wake
            self._last_observed_at = observed
            self._reset_at = reset
            self._retry_at = None
            self._failures = 0
            self._state = SleepState.SLEEPING
            return True

    enter_sleep = prepare_sleep
    request_sleep = prepare_sleep

    def set_user_auto_wake(self, approved: bool, *, origin: str = "user") -> bool:
        """Record a revocable, USER-derived policy for the current epoch.

        Model, tool, and local-ledger content must not grant wake authority.
        The caller is responsible for authenticating the user before passing
        this narrow host API.
        """
        origin_name = str(getattr(origin, "value", origin)).lower()
        with self._lock:
            if self._state == SleepState.TERMINAL or origin_name != "user" or not isinstance(approved, bool):
                return False
            self._auto_wake = approved
            self._generation += 1
            return True

    def revoke_auto_wake(self) -> bool:
        with self._lock:
            if self._state == SleepState.TERMINAL:
                return False
            self._auto_wake = False
            self._generation += 1
            return True

    def refresh_due(self) -> bool:
        """Reset time requests a host refresh; it never itself changes wake state."""
        with self._lock:
            moment = self._now()
            return bool(self._state == SleepState.SLEEPING and self._reset_at is not None and moment >= self._reset_at and
                        (self._retry_at is None or moment >= self._retry_at))

    def check_and_wake(self, windows: Sequence[Any], observed_at: datetime, *, authoritative: bool = True,
                       authority_token: Optional[str] = None, expected_generation: Optional[int] = None) -> bool:
        """Accept a new epoch only from the current, freshly fetched authority callback."""
        with self._lock:
            moment = self._now()
            if self._state != SleepState.SLEEPING or not self._auto_wake:
                return False
            if self._retry_at is not None and moment < self._retry_at:
                return False
            if authority_token != self._authority_token_locked() or expected_generation != self._generation:
                return False
            try:
                values = self._window(windows, observed_at, authoritative, moment)
            except (TypeError, ValueError):
                self._record_refresh_failure(moment)
                return False
            if values is None:
                self._record_refresh_failure(moment)
                return False
            total, remaining, reset, observed = values
            self._state = SleepState.CHECKING
            self._last_observed_at = observed
            self._reset_at = reset
            self._retry_at = None
            self._failures = 0
            # Every consumed CAS callback advances generation, even when the
            # refreshed balance remains below threshold.  A duplicate callback
            # therefore cannot replay or extend backoff.
            self._generation += 1
            if float(remaining) / total <= self.policy.threshold:
                self._state = SleepState.SLEEPING
                return False
            self._epoch += 1
            self._state = SleepState.READY
            return True

    def _record_refresh_failure(self, moment: datetime) -> None:
        self._failures += 1
        seconds = min(self.policy.max_backoff.total_seconds(), self.policy.initial_backoff.total_seconds() * (2 ** (self._failures - 1)))
        self._retry_at = moment + timedelta(seconds=seconds)
        self._generation += 1
        self._state = SleepState.SLEEPING

    def mark_awake(self, *, authority_token: Optional[str],
                   expected_generation: Optional[int]) -> bool:
        """Commit READY -> ACTIVE once after the host records public wake events."""
        with self._lock:
            if self._state != SleepState.READY or not self._auto_wake:
                return False
            if authority_token != self._authority_token_locked() or expected_generation != self._generation:
                return False
            self._state = SleepState.ACTIVE
            self._generation += 1
            # Approval belongs to the completed epoch and cannot silently carry
            # forward to a later sleep episode.
            self._auto_wake = False
            self._reset_at = None
            self._retry_at = None
            self._failures = 0
            return True

    def stop(self) -> None:
        with self._lock:
            if self._state == SleepState.TERMINAL:
                return
            self._state = SleepState.TERMINAL
            self._generation += 1
            self._auto_wake = False

    purge = stop

    def to_payload(self) -> Dict[str, Any]:
        """Exact bounded state needed by a future host/store adapter."""
        with self._lock:
            return {"state": self._state.value, "epoch": self._epoch, "generation": self._generation,
                    "auto_wake_user_approved": self._auto_wake, "reset_at": self._iso(self._reset_at),
                    "retry_at": self._iso(self._retry_at), "failure_count": self._failures,
                    "last_authoritative_observed_at": self._iso(self._last_observed_at)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], policy: Optional[SleepWakePolicy] = None,
                     clock: Optional[Callable[[], datetime]] = None) -> "SleepWakeCoordinator":
        """Restore a controlled public checkpoint without restoring execution.

        Only an unfinished SLEEPING episode or an irrevocable TERMINAL state is
        replayable.  USER auto-wake approval is process-local, so it is always
        cleared and generation advances to invalidate callbacks issued before
        restart.
        """
        if not isinstance(payload, Mapping) or set(payload) != cls._PAYLOAD_KEYS:
            raise ValueError("sleep payload must use the fixed public schema")
        try:
            state = SleepState(payload["state"])
        except (TypeError, ValueError) as error:
            raise ValueError("sleep payload state is invalid") from error
        if state not in (SleepState.SLEEPING, SleepState.TERMINAL):
            raise ValueError("only sleeping or terminal state can be restored")
        epoch = cls._counter(payload["epoch"], "epoch")
        generation = cls._counter(payload["generation"], "generation")
        failures = cls._counter(payload["failure_count"], "failure_count")
        if not isinstance(payload["auto_wake_user_approved"], bool):
            raise ValueError("auto_wake_user_approved must be a bool")
        reset = cls._payload_time(payload["reset_at"], "reset_at")
        retry = cls._payload_time(payload["retry_at"], "retry_at")
        observed = cls._payload_time(payload["last_authoritative_observed_at"],
                                     "last_authoritative_observed_at")
        if state == SleepState.SLEEPING:
            if reset is None or observed is None or reset <= observed:
                raise ValueError("sleeping restore requires valid authoritative quota times")
            if (failures == 0) != (retry is None):
                raise ValueError("retry_at must correspond to failure_count")
        coordinator = cls(policy=policy, clock=clock)
        with coordinator._lock:
            coordinator._state = state
            coordinator._epoch = epoch
            coordinator._generation = generation + 1
            coordinator._auto_wake = False
            coordinator._reset_at = reset
            coordinator._retry_at = retry
            coordinator._failures = failures
            coordinator._last_observed_at = observed
        return coordinator

    restore = from_payload
    replay = from_payload
    restore_from_payload = from_payload

    @staticmethod
    def _counter(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("%s must be a non-negative integer" % name)
        return value

    @staticmethod
    def _payload_time(value: Any, name: str) -> Optional[datetime]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("%s must be an ISO timestamp or None" % name)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("%s must be an ISO timestamp or None" % name) from error
        return _utc(parsed, name)

    @staticmethod
    def _iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value is not None else None
