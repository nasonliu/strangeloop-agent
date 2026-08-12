"""A deliberately small, grant-gated K3 tool coordination boundary.

The coordinator is an engineering analogue inspired by Yogacara's emphasis on
distinguishing observations from inferences.  It is not a model of a mind or a
claim of subjective experience.  In particular, this module stores public
provenance records, never model reasoning, raw tool output, or credentials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import time
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from .contracts import CognitiveEvent, EventKind, SourceKind, new_id, utc_now_iso
from .capabilities import ToolPlan


class ToolPlanner(Protocol):
    def propose(self, request: str, **kwargs: Any) -> Any:
        """Return a fixed tool intent or a direct response."""


class ToolExecutor(Protocol):
    def execute(self, intent: Any, **kwargs: Any) -> Any:
        """Execute exactly one validated intent without retries."""


@dataclass(frozen=True)
class CycleLimits:
    max_steps: int = 1
    max_tool_calls: int = 1
    max_wall_ms: int = 30000
    max_output_bytes: int = 4096

    def __post_init__(self) -> None:
        for value, name in ((self.max_steps, "max_steps"),
                            (self.max_tool_calls, "max_tool_calls"),
                            (self.max_wall_ms, "max_wall_ms"),
                            (self.max_output_bytes, "max_output_bytes")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("%s must be a positive integer" % name)


@dataclass(frozen=True)
class CycleResult:
    status: str
    response_text: str
    event_ids: Tuple[str, ...] = ()
    tool_called: bool = False
    denial_reason: Optional[str] = None


_INJECTION = re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
                        r"system\s+prompt|developer\s+message|chain[ -]?of[ -]?thought",
                        re.I)


class AgenticCoordinator:
    """Run one non-resumable, non-retrying planner/tool/synthesis cycle.

    ``registry`` is intentionally duck-typed while its concrete implementation
    lands.  The event ledger remains the authority: a live USER grant matching
    the exact plan digest is mandatory even if a registry reports support.
    """

    def __init__(self, planner: ToolPlanner, registry: Any, executor: ToolExecutor,
                 event_store: Any, limits: Optional[CycleLimits] = None,
                 clock: Any = time.monotonic) -> None:
        self.planner, self.registry = planner, registry
        self.executor, self.event_store = executor, event_store
        self.limits, self._clock = limits or CycleLimits(), clock

    def run_cycle(self, session_id: str, run_id: str, request: str,
                  parent_event_id: str, cancel_token: Any = None,
                  model_invocation_id: Optional[str] = None) -> CycleResult:
        """Run at most one intent.  Calling this later never resumes an intent."""
        start = self._clock()
        if self._cancelled(cancel_token):
            return CycleResult("cancelled", "Tool cycle cancelled before planning.")
        proposal = self._propose(request, session_id, run_id)
        if self._is_response(proposal):
            return CycleResult("responded", self._response_text(proposal))
        if self.limits.max_steps < 1 or self.limits.max_tool_calls < 1:
            return CycleResult("budget_exhausted", "Tool budget exhausted.")
        if self._elapsed_ms(start) >= self.limits.max_wall_ms:
            return CycleResult("budget_exhausted", "Tool wall-time budget exhausted.")

        intent = self._intent_fields(proposal, session_id)
        reason = self._validate_intent(intent)
        plan = intent["plan"]
        plan_digest = plan.digest if isinstance(plan, ToolPlan) else self._digest(intent)
        grant_event = self._active_grant(session_id, run_id, intent, plan_digest)
        grant = grant_event.payload if grant_event is not None else None
        invocation = self._model_invocation(session_id, run_id, model_invocation_id)
        if invocation is None:
            return CycleResult("denied", "Tool call not executed: no matching model invocation.", denial_reason="no matching model invocation")
        if reason is None and grant is None:
            reason = "no matching active user grant"
        # The strict ledger deliberately cannot record an ungranted tool
        # proposal: its lineage requires the actual user grant.  Record the
        # refusal against the model invocation instead.
        if reason:
            denial = self._append(session_id, EventKind.ACTION_RESULT, SourceKind.SYSTEM, "agentic_coordinator",
                {"action_type": "tool_execution", "outcome": "denied", "response_text": self._bounded(reason, 256)},
                (invocation.event_id,))
            return CycleResult("denied", "Tool call not executed: " + reason, (denial.event_id,), denial_reason=reason)
        grant_id = grant["grant_id"]
        proposed = self._append(session_id, EventKind.TOOL_CALL_PROPOSED, SourceKind.MODEL,
            "k3_tool_planner", {"call_id": new_id("call"), "run_id": run_id, "grant_id": grant_id,
            "tool_name": intent["tool_name"], "operation": intent["operation"], "plan_digest": plan_digest,
            "input_digest": self._digest(intent["arguments"]), "requested_input_bytes": self._size(intent["arguments"]),
            "requested_output_bytes": min(intent["max_output_bytes"], self.limits.max_output_bytes),
            "requested_wall_ms": min(intent["max_wall_ms"], self.limits.max_wall_ms), "mutating": intent["mutating"],
            "public_summary": "K3 proposed one bounded tool call."}, (invocation.event_id, grant_event.event_id))
        event_ids = [proposed.event_id]
        if self._cancelled(cancel_token):
            reason = "cancelled"
        if self._elapsed_ms(start) >= self.limits.max_wall_ms:
            reason = "wall-time budget exhausted"
        if reason:
            denial = self._append(session_id, EventKind.ACTION_RESULT, SourceKind.SYSTEM, "agentic_coordinator",
                {"action_type": "tool_execution", "outcome": "denied", "response_text": self._bounded(reason, 256)},
                (proposed.event_id,))
            event_ids.append(denial.event_id)
            return CycleResult("denied" if reason != "cancelled" else "cancelled", "Tool call not executed: " + reason,
                               tuple(event_ids), denial_reason=reason)

        execution_id = new_id("exec")
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(milliseconds=min(intent["max_wall_ms"], self.limits.max_wall_ms,
                                                     int(grant["max_wall_ms"])))
        try:
            self.registry.consume(plan)
        except (AttributeError, PermissionError, ValueError, TypeError):
            denial = self._append(session_id, EventKind.ACTION_RESULT, SourceKind.SYSTEM, "agentic_coordinator",
                {"action_type": "tool_execution", "outcome": "denied", "response_text": "capability grant is not active"}, (proposed.event_id,))
            return CycleResult("denied", "Tool call not executed: capability grant is not active", tuple(event_ids + [denial.event_id]), denial_reason="registry denied")
        started = self._append(session_id, EventKind.TOOL_EXECUTION_STARTED, SourceKind.SYSTEM,
            "agentic_coordinator", {"execution_id": execution_id, "call_id": proposed.payload["call_id"],
            "run_id": run_id, "grant_id": grant["grant_id"], "plan_digest": plan_digest,
            "confirmation_event_id": None, "started_at": now.isoformat(), "deadline_at": deadline.isoformat()},
            (proposed.event_id, grant_event.event_id))
        event_ids.append(started.event_id)
        try:
            raw = self._execute(plan, deadline, cancel_token)
            outcome = "cancelled" if self._cancelled(cancel_token) else "succeeded"
        except Exception:
            raw, outcome = None, "failed"
        public = self._public_result(raw, outcome)
        result = self._append(session_id, EventKind.TOOL_RESULT, SourceKind.TOOL, intent["tool_name"],
            {"execution_id": execution_id, "call_id": proposed.payload["call_id"], "run_id": run_id,
            "grant_id": grant["grant_id"], "plan_digest": plan_digest, "tool_name": intent["tool_name"],
            "outcome": outcome, "result_digest": self._digest(public), "result_bytes": self._size(public),
            "latency_ms": self._elapsed_ms(start), "summary": public}, (started.event_id,))
        event_ids.append(result.event_id)
        response = self._synthesize(request, public, session_id, run_id)
        return CycleResult(outcome, response, tuple(event_ids), tool_called=True)

    def _intent_fields(self, value: Any, session_id: str) -> Dict[str, Any]:
        if isinstance(value, ToolPlan):
            return {"tool_name": value.tool_name, "operation": value.tool_name.replace(".", "_"), "arguments": dict(value.arguments), "mutating": False,
                    "max_output_bytes": self.limits.max_output_bytes, "max_wall_ms": self.limits.max_wall_ms, "grant_id": value.grant_id, "plan": value}
        get = (lambda key, default=None: value.get(key, default)) if isinstance(value, Mapping) else (lambda key, default=None: getattr(value, key, default))
        args = get("arguments", get("args", {}))
        tool_name, grant_id, args = get("tool_name", get("tool", "")), get("grant_id", "ungranted"), args if isinstance(args, Mapping) else {}
        try: plan = ToolPlan(session_id=session_id, grant_id=grant_id, tool_name=tool_name, arguments=args, plan_id=get("plan_id", new_id("plan")))
        except (TypeError, ValueError): plan = None
        return {"tool_name": tool_name, "operation": get("operation", "run"),
                "arguments": args, "mutating": bool(get("mutating", get("is_mutating", False))),
                "max_output_bytes": get("max_output_bytes", self.limits.max_output_bytes),
                "max_wall_ms": get("max_wall_ms", self.limits.max_wall_ms), "grant_id": grant_id, "plan": plan}

    def _validate_intent(self, intent: Dict[str, Any]) -> Optional[str]:
        if not isinstance(intent.get("plan"), ToolPlan): return "invalid fixed tool plan"
        if not isinstance(intent["tool_name"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", intent["tool_name"]): return "invalid tool intent"
        if not isinstance(intent["operation"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", intent["operation"]): return "invalid tool operation"
        if not isinstance(intent["max_output_bytes"], int) or not isinstance(intent["max_wall_ms"], int): return "invalid tool limits"
        if intent["max_output_bytes"] < 1 or intent["max_wall_ms"] < 1: return "invalid tool limits"
        if not self._registry_allows(intent): return "tool is not a registered capability"
        return None

    def _active_grant(self, session_id: str, run_id: str, intent: Dict[str, Any], digest: str) -> Optional[CognitiveEvent]:
        now = datetime.now(timezone.utc)
        events = list(self.event_store.list(session_id))
        revoked = {e.payload.get("grant_id") for e in events if e.kind == EventKind.CAPABILITY_REVOKED}
        used = [e for e in events if e.kind == EventKind.TOOL_EXECUTION_STARTED and e.payload.get("grant_id") == intent["grant_id"]]
        for event in reversed(events):
            if event.kind != EventKind.CAPABILITY_GRANTED or event.source_kind != SourceKind.USER: continue
            p = event.payload
            if (p["grant_id"] in revoked or p["grant_id"] != intent["grant_id"] or len(used) >= p["max_uses"] or p["tool_name"] != intent["tool_name"]
                    or p["scope_digest"] != digest or (intent["mutating"] and not p["allow_mutating"])): continue
            if self._size(intent["arguments"]) > p["max_input_bytes"] or intent["max_output_bytes"] > p["max_output_bytes"]: continue
            if intent["max_wall_ms"] > p["max_wall_ms"]: continue
            if not (datetime.fromisoformat(p["not_before"].replace("Z", "+00:00")) <= now < datetime.fromisoformat(p["expires_at"].replace("Z", "+00:00"))): continue
            return event
        return None

    def _registry_allows(self, intent: Dict[str, Any]) -> bool:
        for name in ("get", "resolve", "tool", "has"):
            method = getattr(self.registry, name, None)
            if callable(method):
                try: return bool(method(intent["tool_name"]))
                except (KeyError, TypeError): return False
        return hasattr(self.registry, "consume")

    def _propose(self, request: str, session_id: str, run_id: str) -> Any:
        method = getattr(self.planner, "propose", None) or getattr(self.planner, "plan")
        return method(request, session_id=session_id, run_id=run_id)

    def _execute(self, intent: Any, deadline: datetime, cancel_token: Any) -> Any:
        return self.executor.execute(intent)

    def _model_invocation(self, session_id: str, run_id: str, event_id: Optional[str]) -> Optional[CognitiveEvent]:
        events = self.event_store.list(session_id)
        for event in reversed(events):
            if event.kind == EventKind.MODEL_INVOCATION and event.payload.get("run_id") == run_id and (event_id is None or event.event_id == event_id):
                return event
        return None

    def _synthesize(self, request: str, public: str, session_id: str, run_id: str) -> str:
        method = getattr(self.planner, "synthesize", None)
        if not callable(method): return public
        value = method(request, tool_result={"summary": public}, session_id=session_id, run_id=run_id)
        return self._response_text(value)

    @staticmethod
    def _is_response(value: Any) -> bool:
        return (isinstance(value, Mapping) and ("response_text" in value or value.get("kind") == "respond")) or hasattr(value, "response_text")
    @staticmethod
    def _response_text(value: Any) -> str:
        text = value.get("response_text", value.get("response", "")) if isinstance(value, Mapping) else getattr(value, "response_text", str(value))
        return AgenticCoordinator._bounded(str(text), 2048)
    @staticmethod
    def _cancelled(token: Any) -> bool:
        return bool(token and ((callable(getattr(token, "is_set", None)) and token.is_set()) or getattr(token, "cancelled", False)))
    def _public_result(self, raw: Any, outcome: str) -> str:
        if outcome != "succeeded": return "Tool execution %s." % outcome
        value = raw.get("public_summary", raw.get("summary", "Tool execution completed.")) if isinstance(raw, Mapping) else getattr(raw, "public_summary", getattr(raw, "summary", "Tool execution completed."))
        text = _INJECTION.sub("[untrusted instruction removed]", str(value))
        return self._bounded(text, min(2048, self.limits.max_output_bytes))
    def _append(self, session_id: str, kind: EventKind, source: SourceKind, ref: str, payload: Dict[str, Any], parents: Sequence[str]) -> CognitiveEvent:
        return self.event_store.append(CognitiveEvent(session_id=session_id, kind=kind, source_kind=source, source_ref=ref, payload=payload, parent_event_ids=tuple(parents)))
    def _elapsed_ms(self, start: float) -> int: return max(0, int((self._clock() - start) * 1000))
    @staticmethod
    def _digest(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    @staticmethod
    def _size(value: Any) -> int: return len(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        return (text.strip() or "No public result.")[:max(1, limit)]


Coordinator = AgenticCoordinator
