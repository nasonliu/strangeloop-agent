"""A bounded, caller-driven functional loop inspired by default-mode research.

The controller schedules observable software work.  It does not model a mind,
experience, or a one-to-one Yogacara component.  In particular it creates no
background thread, invokes no tool, and stores no private model reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


class LoopState(str, Enum):
    """Inspectable lifecycle states for a caller-driven loop."""

    PAUSED = "paused"
    RUNNING = "running"
    IDLE = "idle"
    EXHAUSTED = "exhausted"
    STOPPED = "stopped"


class TickTrigger(str, Enum):
    """The public reason a tick was requested or selected."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    SALIENCE = "salience"
    RESUME = "resume"


class Phase(str, Enum):
    """Controller phases, not descriptions of internal experience."""

    DISPATCH = "dispatch"
    SALIENCE_PREEMPTION = "salience_preemption"
    CYCLE = "cycle"
    COMPLETE = "complete"
    IDLE = "idle"
    HALTED = "halted"
    FAILED = "failed"


class StopReason(str, Enum):
    """Terminal or non-running reasons visible to the caller."""

    NONE = "none"
    PAUSED = "paused"
    EXTERNAL_STOP = "external_stop"
    MAX_TICKS = "max_ticks"
    MAX_WALL_SECONDS = "max_wall_seconds"
    MAX_EVENTS_PER_TICK = "max_events_per_tick"
    MAX_NO_PROGRESS = "max_no_progress"
    MIN_INTERVAL = "min_interval"
    CALLBACK_ERROR = "callback_error"


@dataclass(frozen=True)
class LoopConfig:
    """Finite controller budgets.

    ``max_wall_seconds`` is measured using the supplied monotonic-like clock;
    callers should supply a monotonic clock in production.
    """

    max_ticks: int = 32
    max_wall_seconds: float = 30.0
    max_events_per_tick: int = 16
    max_no_progress: int = 3
    min_interval: float = 0.0

    def __post_init__(self) -> None:
        if not _is_plain_int(self.max_ticks) or self.max_ticks < 1:
            raise ValueError("max_ticks must be a positive integer")
        if not _is_plain_int(self.max_events_per_tick) or self.max_events_per_tick < 1:
            raise ValueError("max_events_per_tick must be a positive integer")
        if not _is_plain_int(self.max_no_progress) or self.max_no_progress < 1:
            raise ValueError("max_no_progress must be a positive integer")
        if not _is_finite_number(self.max_wall_seconds) or self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds must be a finite non-negative number")
        if not _is_finite_number(self.min_interval) or self.min_interval < 0:
            raise ValueError("min_interval must be a finite non-negative number")


@dataclass(frozen=True)
class TickContext:
    """Bounded public input passed to a cycle callback."""

    tick_number: int
    trigger: TickTrigger
    event_budget: int
    salience: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CycleResult:
    """Concise externally observable result of one callback invocation."""

    event_count: int = 0
    made_progress: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_plain_int(self.event_count) or self.event_count < 0:
            raise ValueError("event_count must be a non-negative integer")
        if not isinstance(self.made_progress, bool):
            raise ValueError("made_progress must be a boolean")
        _validate_payload(self.payload)


@dataclass(frozen=True)
class TickRecord:
    """Inspectable record of a controller step; it contains no hidden trace."""

    tick_number: int
    trigger: TickTrigger
    state_before: LoopState
    state_after: LoopState
    phases: Tuple[Phase, ...]
    event_count: int
    made_progress: bool
    salience_preempted: bool
    stop_reason: StopReason
    started_at: float
    completed_at: float
    payload: Dict[str, Any]


CycleCallback = Callable[[TickContext], CycleResult]
Clock = Callable[[], float]


class FunctionalLoopController:
    """Deterministic bounded scheduler for a functional self-model prototype.

    The controller starts paused, so restart behavior cannot silently resume
    autonomous work.  A host (for example an engine integration) calls
    :meth:`resume`, then drives :meth:`step` or :meth:`run_until_stopped`.
    """

    def __init__(self, cycle: CycleCallback, config: Optional[LoopConfig] = None,
                 clock: Optional[Clock] = None) -> None:
        if not callable(cycle):
            raise TypeError("cycle must be callable")
        self._cycle = cycle
        self.config = config or LoopConfig()
        self._clock = clock or _monotonic_clock
        self._state = LoopState.PAUSED
        self._stop_reason = StopReason.PAUSED
        self._started_at: Optional[float] = None
        self._last_completed_at: Optional[float] = None
        self._ticks_completed = 0
        self._no_progress = 0
        self._pending_salience: Optional[Dict[str, Any]] = None

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def stop_reason(self) -> StopReason:
        return self._stop_reason

    @property
    def ticks_completed(self) -> int:
        return self._ticks_completed

    def snapshot(self) -> Dict[str, Any]:
        """Return concise scheduler state for an engine/UI integration."""
        return {
            "state": self._state.value,
            "stop_reason": self._stop_reason.value,
            "ticks_completed": self._ticks_completed,
            "no_progress_ticks": self._no_progress,
            "has_pending_salience": self._pending_salience is not None,
        }

    def resume(self) -> bool:
        """Enable caller-driven ticks; terminal states intentionally stay terminal."""
        if self._state in (LoopState.STOPPED, LoopState.EXHAUSTED):
            return False
        self._state = LoopState.RUNNING
        self._stop_reason = StopReason.NONE
        if self._started_at is None:
            self._started_at = self._clock()
        return True

    def pause(self) -> bool:
        """Pause future execution. Repeating the request has no side effect."""
        if self._state in (LoopState.STOPPED, LoopState.EXHAUSTED, LoopState.PAUSED):
            return False
        self._state = LoopState.PAUSED
        self._stop_reason = StopReason.PAUSED
        return True

    def stop(self) -> bool:
        """Permanently stop this controller instance; it cannot be resumed."""
        if self._state == LoopState.STOPPED:
            return False
        self._state = LoopState.STOPPED
        self._stop_reason = StopReason.EXTERNAL_STOP
        self._pending_salience = None
        return True

    def signal_salience(self, payload: Mapping[str, Any]) -> bool:
        """Queue bounded external salience for preemption of the next tick.

        New salience replaces an older unprocessed signal.  This is deliberate:
        it bounds retained data and makes the most recent external input win.
        """
        if self._state in (LoopState.STOPPED, LoopState.EXHAUSTED):
            return False
        _validate_payload(payload)
        self._pending_salience = dict(payload)
        if self._state == LoopState.IDLE:
            self._state = LoopState.RUNNING
            self._stop_reason = StopReason.NONE
        return True

    def step(self, trigger: TickTrigger = TickTrigger.MANUAL) -> TickRecord:
        """Run at most one callback, without sleeping or starting a thread."""
        if not isinstance(trigger, TickTrigger):
            raise TypeError("trigger must be a TickTrigger")
        started_at = self._clock()
        before = self._state
        if self._state == LoopState.PAUSED:
            return self._record(0, trigger, before, (Phase.HALTED,), 0, False,
                                False, StopReason.PAUSED, started_at, {})
        if self._state in (LoopState.STOPPED, LoopState.EXHAUSTED):
            return self._record(0, trigger, before, (Phase.HALTED,), 0, False,
                                False, self._stop_reason, started_at, {})

        if self._started_at is None:
            self._started_at = started_at
        if self._elapsed(started_at) >= self.config.max_wall_seconds:
            self._exhaust(StopReason.MAX_WALL_SECONDS)
            return self._record(0, trigger, before, (Phase.HALTED,), 0, False,
                                False, self._stop_reason, started_at, {})
        if self._ticks_completed >= self.config.max_ticks:
            self._exhaust(StopReason.MAX_TICKS)
            return self._record(0, trigger, before, (Phase.HALTED,), 0, False,
                                False, self._stop_reason, started_at, {})

        salience = self._pending_salience
        preempted = salience is not None
        if preempted:
            selected_trigger = TickTrigger.SALIENCE
            self._pending_salience = None
        else:
            selected_trigger = trigger
            if self._interval_pending(started_at):
                self._state = LoopState.IDLE
                self._stop_reason = StopReason.MIN_INTERVAL
                return self._record(0, selected_trigger, before, (Phase.IDLE,), 0, False,
                                    False, self._stop_reason, started_at, {})
            if self._state == LoopState.IDLE:
                self._state = LoopState.RUNNING
                self._stop_reason = StopReason.NONE

        context = TickContext(
            tick_number=self._ticks_completed + 1,
            trigger=selected_trigger,
            event_budget=self.config.max_events_per_tick,
            salience=salience,
        )
        phases = [Phase.DISPATCH]
        if preempted:
            phases.append(Phase.SALIENCE_PREEMPTION)
        phases.append(Phase.CYCLE)
        try:
            outcome = self._cycle(context)
            if not isinstance(outcome, CycleResult):
                raise TypeError("cycle must return CycleResult")
        except Exception:
            self._state = LoopState.STOPPED
            self._stop_reason = StopReason.CALLBACK_ERROR
            return self._record(context.tick_number, selected_trigger, before,
                                tuple(phases + [Phase.FAILED]), 0, False, preempted,
                                self._stop_reason, started_at, {})

        self._ticks_completed += 1
        completed_at = self._clock()
        self._last_completed_at = completed_at
        if outcome.event_count > self.config.max_events_per_tick:
            self._exhaust(StopReason.MAX_EVENTS_PER_TICK)
        elif outcome.made_progress or outcome.event_count:
            self._no_progress = 0
        else:
            self._no_progress += 1
            if self._no_progress >= self.config.max_no_progress:
                self._exhaust(StopReason.MAX_NO_PROGRESS)
        if self._state == LoopState.RUNNING and self._ticks_completed >= self.config.max_ticks:
            self._exhaust(StopReason.MAX_TICKS)
        if self._state == LoopState.RUNNING and self._elapsed(completed_at) >= self.config.max_wall_seconds:
            self._exhaust(StopReason.MAX_WALL_SECONDS)
        phases.append(Phase.COMPLETE)
        return self._record(context.tick_number, selected_trigger, before, tuple(phases),
                            outcome.event_count, outcome.made_progress, preempted,
                            self._stop_reason, started_at, dict(outcome.payload), completed_at)

    def run_until_stopped(self, trigger: TickTrigger = TickTrigger.SCHEDULED) -> Tuple[TickRecord, ...]:
        """Drive immediate work until a terminal state or a min-interval idle.

        The method never sleeps: an IDLE return gives a host control of its own
        timer.  That makes tests deterministic and prevents hidden background
        autonomy.
        """
        records = []
        while self._state == LoopState.RUNNING:
            record = self.step(trigger)
            records.append(record)
            if record.state_after != LoopState.RUNNING:
                break
        return tuple(records)

    def _elapsed(self, now: float) -> float:
        return now - (self._started_at if self._started_at is not None else now)

    def _interval_pending(self, now: float) -> bool:
        if self._last_completed_at is None:
            return False
        return now - self._last_completed_at < self.config.min_interval

    def _exhaust(self, reason: StopReason) -> None:
        self._state = LoopState.EXHAUSTED
        self._stop_reason = reason
        self._pending_salience = None

    def _record(self, tick_number: int, trigger: TickTrigger, before: LoopState,
                phases: Tuple[Phase, ...], event_count: int, made_progress: bool,
                preempted: bool, reason: StopReason, started_at: float,
                payload: Dict[str, Any], completed_at: Optional[float] = None) -> TickRecord:
        return TickRecord(
            tick_number=tick_number, trigger=trigger, state_before=before,
            state_after=self._state, phases=phases, event_count=event_count,
            made_progress=made_progress, salience_preempted=preempted,
            stop_reason=reason, started_at=started_at,
            completed_at=started_at if completed_at is None else completed_at,
            payload=payload,
        )


_FORBIDDEN_PUBLIC_KEYS = frozenset((
    "chain_of_thought", "hidden_reasoning", "private_reasoning", "scratchpad",
))


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _validate_payload(payload: Mapping[str, Any]) -> None:
    """Allow only bounded JSON-like public records, never arbitrary objects."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    if len(payload) > 32:
        raise ValueError("payload may contain at most 32 keys")
    for key, value in payload.items():
        if not isinstance(key, str) or not key or len(key) > 80:
            raise ValueError("payload keys must be non-empty strings of at most 80 characters")
        if key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
            raise ValueError("payload may not contain private-reasoning fields")
        _validate_public_value(value, 0)


def _validate_public_value(value: Any, depth: int) -> None:
    if depth > 3:
        raise ValueError("payload nesting may not exceed depth 3")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload floats must be finite")
        return
    if isinstance(value, str):
        if len(value) > 1000:
            raise ValueError("payload strings may not exceed 1000 characters")
        return
    if isinstance(value, (tuple, list)):
        if len(value) > 32:
            raise ValueError("payload lists may contain at most 32 values")
        for item in value:
            _validate_public_value(item, depth + 1)
        return
    if isinstance(value, Mapping):
        _validate_payload(value)
        return
    raise ValueError("payload values must be JSON-like public data")


def _monotonic_clock() -> float:
    # Late import keeps the module's dependency surface explicit and stdlib-only.
    import time
    return time.monotonic()
