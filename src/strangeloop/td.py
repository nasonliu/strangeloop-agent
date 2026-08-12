"""Bounded temporal-difference updates for safe action-class preferences.

This module is a small, inspectable reinforcement-learning primitive.  It is
not a model of dopamine or subjective reward: a ``RewardObservation`` is an
externally attributable scalar record used to update a value estimate.  It
does not grant capabilities, authorise actions, or execute tools.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


MAX_IDENTIFIER_LENGTH = 128
MAX_ABS_VALUE = 1000.0
MAX_ABS_RAW_DELTA = 2.0 * MAX_ABS_VALUE + 1.0
DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_EVENTS = 1024


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be a finite number" % name)
    return result


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTIFIER_LENGTH:
        raise ValueError("%s must be a non-empty bounded string" % name)
    if value.strip() != value or any(ord(char) < 32 for char in value):
        raise ValueError("%s contains unsupported characters" % name)
    return value


class RewardSource(str, Enum):
    """Permitted origins of a reward observation.

    Model, system, or self-reported reward is deliberately absent.  Callers
    must give a source reference so every update remains attributable.
    """

    USER = "user"
    TOOL = "tool"
    EXTERNAL_VERIFIER = "external_verifier"


SAFE_ACTION_CLASSES = frozenset((
    "ask_clarifying_question",
    "record_observation",
    "respond",
    "retrieve_user_approved_memory",
    "summarize",
    "wait",
))


@dataclass(frozen=True)
class TDConfig:
    """Bounded TD(0) parameters.

    ``clip`` bounds the prediction error before the learning-rate update, not
    a permission score.  Values themselves are capped to keep a malformed or
    long-running stream from growing without limit.
    """

    alpha: float = 0.2
    gamma: float = 0.95
    clip: float = 1.0

    def __post_init__(self) -> None:
        alpha = _finite_number(self.alpha, "alpha")
        gamma = _finite_number(self.gamma, "gamma")
        clip = _finite_number(self.clip, "clip")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 < clip <= MAX_ABS_VALUE:
            raise ValueError("clip must be positive and bounded")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "clip", clip)


@dataclass(frozen=True)
class ValueEstimate:
    """A finite, context-scoped scalar estimate; never an authorization."""

    context_id: str
    state_key: str
    value: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.context_id, "context_id")
        _identifier(self.state_key, "state_key")
        value = _finite_number(self.value, "value")
        if abs(value) > MAX_ABS_VALUE:
            raise ValueError("value exceeds the bounded estimate range")
        object.__setattr__(self, "value", value)

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Transition:
    """An externally supplied state transition, before any reward is applied."""

    transition_id: str
    context_id: str
    state_key: str
    action_class: str
    next_state_key: str
    terminal: bool = False

    def __post_init__(self) -> None:
        _identifier(self.transition_id, "transition_id")
        _identifier(self.context_id, "context_id")
        _identifier(self.state_key, "state_key")
        _identifier(self.next_state_key, "next_state_key")
        _identifier(self.action_class, "action_class")
        if self.action_class not in SAFE_ACTION_CLASSES:
            raise ValueError("action_class is not an allowed safe action class")
        if not isinstance(self.terminal, bool):
            raise ValueError("terminal must be a boolean")

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardObservation:
    """An attributable reward record bound to one previously registered transition."""

    reward_id: str
    transition_id: str
    context_id: str
    reward: float
    source: RewardSource
    source_ref: str

    def __post_init__(self) -> None:
        _identifier(self.reward_id, "reward_id")
        _identifier(self.transition_id, "transition_id")
        _identifier(self.context_id, "context_id")
        _identifier(self.source_ref, "source_ref")
        reward = _finite_number(self.reward, "reward")
        if abs(reward) > 1.0:
            raise ValueError("reward must be between -1 and 1")
        if not isinstance(self.source, RewardSource):
            # This explicitly rejects SourceKind.MODEL/SYSTEM and plain
            # strings, rather than silently coercing a spoofed provenance.
            raise ValueError("reward source must be user, tool, or external_verifier")
        object.__setattr__(self, "reward", reward)

    def to_payload(self) -> dict:
        return {
            "reward_id": self.reward_id,
            "transition_id": self.transition_id,
            "context_id": self.context_id,
            "reward": self.reward,
            "source": self.source.value,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class RPEUpdate:
    """Public TD/RPE calculation, with no private model reasoning fields."""

    transition_id: str
    reward_id: str
    context_id: str
    current_value: float
    next_value: float
    raw_delta: float
    clipped_delta: float
    updated_value: float

    def __post_init__(self) -> None:
        _identifier(self.transition_id, "transition_id")
        _identifier(self.reward_id, "reward_id")
        _identifier(self.context_id, "context_id")
        for field_name in ("current_value", "next_value", "raw_delta", "clipped_delta", "updated_value"):
            value = _finite_number(getattr(self, field_name), field_name)
            limit = MAX_ABS_RAW_DELTA if field_name == "raw_delta" else MAX_ABS_VALUE
            if abs(value) > limit:
                raise ValueError("%s exceeds the bounded update range" % field_name)
            object.__setattr__(self, field_name, value)

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SafeActionPreference:
    """A ranking result only; it deliberately has no authorization field."""

    action_class: str
    value: float

    def __post_init__(self) -> None:
        if self.action_class not in SAFE_ACTION_CLASSES:
            raise ValueError("action_class is not an allowed safe action class")
        object.__setattr__(self, "value", _finite_number(self.value, "value"))

    def to_payload(self) -> dict:
        return asdict(self)


class ValueTable:
    """An in-memory bounded, event-sourced TD value table.

    The caller owns persistence of the returned public payloads.  That keeps
    this module free of hidden reasoning and prevents a reward score from
    being mistaken for a tool permission.
    """

    def __init__(self, config: TDConfig = TDConfig(), max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_events: int = DEFAULT_MAX_EVENTS) -> None:
        if not isinstance(config, TDConfig):
            raise ValueError("config must be TDConfig")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.config = config
        self.max_entries = max_entries
        self.max_events = max_events
        self._values: Dict[Tuple[str, str], float] = {}
        self._transitions: Dict[str, Transition] = {}
        self._reward_ids = set()
        self._events: List[dict] = []

    @contextmanager
    def transaction(self) -> Iterator["ValueTable"]:
        """Roll back every table resource if the enclosed operation fails.

        Each nesting level owns an independent snapshot.  This makes a caught
        inner failure restore the inner entry state while allowing the outer
        transaction to continue, and makes an uncaught failure restore all the
        way to the outer entry state.  ``BaseException`` is included so an
        interrupt cannot leave a partially updated preference projection.

        When coordinating with another transactional resource, keep this
        context outermost so a failure from the inner resource's commit is
        observed here and restores this in-memory table as well.
        """
        snapshot = (
            dict(self._values),
            dict(self._transitions),
            set(self._reward_ids),
            deepcopy(self._events),
        )
        try:
            yield self
        except BaseException:
            self._values, self._transitions, self._reward_ids, self._events = snapshot
            raise

    @staticmethod
    def preference_state_key(action_class: str) -> str:
        if action_class not in SAFE_ACTION_CLASSES:
            raise ValueError("action_class is not an allowed safe action class")
        return "preference:" + action_class

    def _ensure_capacity(self, keys: Sequence[Tuple[str, str]]) -> None:
        new_keys = {key for key in keys if key not in self._values}
        if len(self._values) + len(new_keys) > self.max_entries:
            raise ValueError("value table entry limit reached")

    def _append_event(self, event: dict) -> None:
        if len(self._events) >= self.max_events:
            raise ValueError("value table event limit reached")
        self._events.append(event)

    def estimate(self, context_id: str, state_key: str) -> ValueEstimate:
        _identifier(context_id, "context_id")
        _identifier(state_key, "state_key")
        return ValueEstimate(context_id, state_key, self._values.get((context_id, state_key), 0.0))

    def register_transition(self, transition: Transition) -> Transition:
        if not isinstance(transition, Transition):
            raise ValueError("transition must be Transition")
        if transition.transition_id in self._transitions:
            raise ValueError("transition_id has already been registered")
        self._ensure_capacity(((transition.context_id, transition.state_key),
                               (transition.context_id, transition.next_state_key)))
        self._append_event({"event_type": "transition", "transition": transition.to_payload()})
        self._transitions[transition.transition_id] = transition
        self._values.setdefault((transition.context_id, transition.state_key), 0.0)
        self._values.setdefault((transition.context_id, transition.next_state_key), 0.0)
        return transition

    def apply_reward(self, observation: RewardObservation) -> RPEUpdate:
        if not isinstance(observation, RewardObservation):
            raise ValueError("observation must be RewardObservation")
        if observation.reward_id in self._reward_ids:
            raise ValueError("reward_id has already been applied")
        transition = self._transitions.get(observation.transition_id)
        if transition is None:
            raise ValueError("reward refers to an unknown transition")
        if observation.context_id != transition.context_id:
            raise ValueError("reward context does not match its transition")
        current = self._values[(transition.context_id, transition.state_key)]
        next_value = 0.0 if transition.terminal else self._values[(transition.context_id, transition.next_state_key)]
        raw_delta = observation.reward + self.config.gamma * next_value - current
        clipped_delta = max(-self.config.clip, min(self.config.clip, raw_delta))
        updated = max(-MAX_ABS_VALUE, min(MAX_ABS_VALUE,
                                          current + self.config.alpha * clipped_delta))
        update = RPEUpdate(transition_id=transition.transition_id,
                           reward_id=observation.reward_id,
                           context_id=transition.context_id,
                           current_value=current, next_value=next_value,
                           raw_delta=raw_delta, clipped_delta=clipped_delta,
                           updated_value=updated)
        self._append_event({"event_type": "rpe_update", "reward": observation.to_payload(),
                            "update": update.to_payload()})
        self._values[(transition.context_id, transition.state_key)] = updated
        self._reward_ids.add(observation.reward_id)
        return update

    def rank_action_classes(self, context_id: str,
                            candidates: Iterable[str]) -> Tuple[SafeActionPreference, ...]:
        _identifier(context_id, "context_id")
        candidate_values = list(candidates)
        if not candidate_values:
            raise ValueError("at least one candidate action class is required")
        if len(candidate_values) > len(SAFE_ACTION_CLASSES):
            raise ValueError("candidate action class limit exceeded")
        if len(set(candidate_values)) != len(candidate_values):
            raise ValueError("candidate action classes must be unique")
        ranked = [SafeActionPreference(action_class, self.estimate(
            context_id, self.preference_state_key(action_class)).value)
            for action_class in candidate_values]
        return tuple(sorted(ranked, key=lambda item: (-item.value, item.action_class)))

    def event_log(self) -> Tuple[dict, ...]:
        """Return a shallow public projection suitable for audit/replay."""
        return tuple(dict(event) for event in self._events)

    def export(self) -> dict:
        """Export bounded public state without a chain-of-thought field."""
        estimates = [ValueEstimate(context_id, state_key, value).to_payload()
                     for (context_id, state_key), value in sorted(self._values.items())]
        return {"config": asdict(self.config), "estimates": estimates,
                "events": list(self.event_log()), "limits": {"max_entries": self.max_entries,
                "max_events": self.max_events}}

    @classmethod
    def replay(cls, transitions: Iterable[Transition], rewards: Iterable[RewardObservation],
               config: TDConfig = TDConfig(), max_entries: int = DEFAULT_MAX_ENTRIES,
               max_events: int = DEFAULT_MAX_EVENTS) -> "ValueTable":
        table = cls(config=config, max_entries=max_entries, max_events=max_events)
        for transition in transitions:
            table.register_transition(transition)
        for reward in rewards:
            table.apply_reward(reward)
        return table


# The alias names the intended use without suggesting subjective experience.
SafePreferenceTable = ValueTable
