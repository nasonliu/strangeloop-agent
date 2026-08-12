"""A localhost-only monitor for public, auditable Strangeloop records.

This is an inspection server, not an agent control plane.  It deliberately
projects a small allowlist of event fields and never returns raw prompt text,
tool input/output, credentials, base64 material, or hidden reasoning.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

from .monitor_ui import dashboard_html


_FORBIDDEN = frozenset(("chain_of_thought", "hidden_reasoning", "private_reasoning",
                        "scratchpad", "secret", "token", "password", "api_key",
                        "authorization", "stdout", "stderr", "web_body", "base64"))
_PUBLIC_TEXT = frozenset(("public_summary", "summary", "outcome", "reason", "state",
                          "phase", "progress", "role", "provider", "model", "operation",
                          "action", "action_type", "signal_kind", "scope", "context_scope"))
_PUBLIC_SCALARS = frozenset(("run_id", "turn_id", "seed_id", "claim_id", "grant_id", "call_id",
    "execution_id", "invocation_id", "tool_name", "capability", "mutating", "allow_mutating",
    "latency_ms", "requested_input_bytes", "requested_output_bytes", "requested_wall_ms",
    "result_bytes", "max_uses", "max_input_bytes", "max_output_bytes", "max_wall_ms",
    "budget_remaining", "tick_index", "tick_count", "normalized_value", "reward", "raw_delta",
    "clipped_delta", "updated_value", "value", "confidence", "alpha", "gamma", "clip"))
_MIRROR_KIND = "metacognitive_mirror"
_MIRROR_TEXT_FIELDS = frozenset((
    "mirror_id", "episode_id", "target_event_id", "judgment_event_id",
    "self_status", "self_uncertainty", "meta_status", "disposition",
    "method_version", "public_summary",
))
_MIRROR_NUMERIC_FIELDS = frozenset(("self_confidence", "meta_confidence_cap"))
_MIRROR_LIST_FIELDS = frozenset(("evidence_event_ids", "check_codes"))
_MIRROR_CHECK_CODES = frozenset((
    "provenance_complete", "scope_bounded", "confidence_capped",
    "counterevidence_clear", "evidence_missing", "source_conflict",
))
_MIRROR_ENUMS = {
    "self_status": frozenset(("supported", "insufficient", "conflicted")),
    "self_uncertainty": frozenset(("low", "medium", "high")),
    "meta_status": frozenset(("confirmed", "limited", "conflicted")),
    "disposition": frozenset(("provisional", "review_required", "abstain")),
    "method_version": frozenset(("mirror_v1",)),
}
_STAGE_KINDS = frozenset(("tool_call_proposed", "tool_execution_confirmed",
    "tool_execution_started", "tool_result", "tool_execution_abandoned",
    "capability_granted", "capability_revoked"))
_TEXT_LIMIT = 512
_BASE64 = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")
_SLEEP_STATES = frozenset(("active", "preparing", "sleeping", "checking", "ready", "terminal"))
_SLEEP_ARCHIVE_FIELDS = frozenset((
    "archive_id", "epoch_id", "chain_head_hash", "quota_source",
    "quota_observed_at", "quota_reset_at", "archive_digest", "event_count",
    "archive_version", "schema_version", "chain_head_sequence",
    "quota_remaining", "quota_total", "quota_window_kind",
))
_SEED_POLICY_KINDS = frozenset((
    "seed_standing_policy", "seed_standing_policy_revoked",
    "seed_auto_eligibility", "seed_auto_applied",
))
_SEED_AUTO_OPERATIONS = frozenset(("activate", "reinforce", "tighten", "retire"))
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ISO8601 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_URL = re.compile(r"(?i)\bhttps?://[^\s<>\"']+")
_LOCAL_PATH = re.compile(r"(?:(?<=\s)|^)/(?:Users|home|private|tmp|var|Volumes|etc|opt)/[^\s<>\"']+")
_SECRETISH = re.compile(r"(?i)(?:\bcookie\b|set-cookie|\bsk-[a-z0-9_-]{8,}|\bkey\s*[=:]|\btoken\s*[=:])")
_READ_ONLY_TOOL_NAMES = frozenset((
    "repo.status", "repo.search", "repo.read", "web.fetch", "web.search", "browser.read",
))
_UNATTENDED_STATES = frozenset(("stopped", "running", "paused", "exhausted"))
_UNATTENDED_REASONS = frozenset((
    "none", "user_stop", "sleep", "quota_exhausted", "max_wall_seconds", "max_calls",
    "max_input_bytes", "max_output_bytes", "max_tick_seconds", "no_work", "no_progress",
    "repeated_action", "host_error", "stopped", "wall_clock_budget", "tool_call_budget",
    "byte_budget",
))
_EXPEDITION_STATES = frozenset((
    "idle", "ready", "active", "waiting_quota_retry", "running", "sleeping", "paused", "stopped", "completed", "failed", "exhausted",
))
_EXPEDITION_STOP_REASONS = frozenset((
    "none", "user_stop", "sleep", "quota_exhausted", "max_wall_seconds", "max_calls",
    "max_slices", "max_domains", "max_empty_searches", "no_work", "no_progress",
    "planner_failure", "host_error", "terminal", "purge", "frontier_exhausted",
    "authorization_window_elapsed", "quota_unknown_fail_closed", "quota_sleep",
    "archive_store_failure",
))
_FRONTIER_LEARNING_MODES = frozenset(("off", "shadow", "active"))
_FRONTIER_RANKING_REASONS = frozenset((
    "legacy_persona_rotation", "ranker_shadow_observed_baseline",
    "ranker_shadow_invalid_baseline", "ranker_invalid_fallback",
    "fairness_persona_due", "ranker_active_recommendation",
    "experiment_cadence_due", "experiment_cadence_due_shadow",
    "experiment_cadence_due_ranker",
))
_STRATEGY_ARM_VERSION = "frontier_strategy_arm_v1"
_POST_REWARD_SELECTION_REASONS = frozenset((
    "strategy_reward_preferred", "strategy_ranked", "none",
))
_EXPERIMENT_MODES = frozenset(("off", "shadow", "active"))
_EXPERIMENT_KINDS = frozenset((
    "frontier_learner_ab", "frontier_replay", "duplicate_suppression", "td_invariants",
))
_EXPERIMENT_STATUSES = frozenset(("supported", "refuted", "inconclusive", "invalid"))
_QUOTA_AUTHORITIES = frozenset((
    "unknown", "provider_usage", "authoritative_header", "manual_snapshot",
    "error_signal", "local_ledger",
))
_QUOTA_FRESHNESS = frozenset((
    "unknown", "unverified", "estimated", "fresh", "stale", "reset_due", "paused",
))
_QUOTA_REASONS = frozenset((
    "not_configured", "no_trusted_quota_telemetry", "quota_telemetry_stale",
    "quota_reset_requires_fresh_telemetry", "provider_quota_exhausted",
    "provider_rate_limit_cooldown", "quota_hard_conservation_band",
    "quota_soft_conservation_band", "quota_available",
))
_QUOTA_UNITS = frozenset(("tokens", "provider_units"))
_QUOTA_WINDOW_KINDS = frozenset(("weekly", "rolling_5h"))


def _kind(value: Any) -> str:
    return getattr(value, "value", value) if isinstance(getattr(value, "value", value), str) else "unknown"


def _safe_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if (_BASE64.search(value) or _SECRETISH.search(value)
            or re.search(r"(?i)(bearer\s+|api[_-]?key|password|secret|token)", value)
            or _URL.search(value) or _LOCAL_PATH.search(value)):
        return "[redacted]"
    return html.escape(value[:_TEXT_LIMIT], quote=True)


def _safe_mirror_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Project only the fixed public mirror-record surface.

    This deliberately does not recursively traverse arbitrary structures.  A
    mirror is an auditable bounded check, not a place to expose its model
    context, rationale, or unreviewed evidence contents.
    """
    result: Dict[str, Any] = {}
    for key in _MIRROR_TEXT_FIELDS:
        raw = payload.get(key)
        if key in _MIRROR_ENUMS and raw not in _MIRROR_ENUMS[key]:
            continue
        value = _safe_text(raw)
        if value is not None:
            result[key] = value
    for key in _MIRROR_NUMERIC_FIELDS:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0:
            result[key] = value
    evidence = payload.get("evidence_event_ids")
    if isinstance(evidence, list) and len(evidence) <= 32 and all(isinstance(item, str) for item in evidence):
        result["evidence_event_ids"] = [_safe_text(item) or "" for item in evidence]
    check_codes = payload.get("check_codes")
    if isinstance(check_codes, list) and len(check_codes) <= 16 and all(
            isinstance(item, str) and item in _MIRROR_CHECK_CODES for item in check_codes):
        result["check_codes"] = list(check_codes)
    return result


def _safe_sleep_archive_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Expose archive integrity metadata only; never archived record contents."""
    result: Dict[str, Any] = {}
    for key in ("archive_id", "epoch_id"):
        value = payload.get(key)
        if isinstance(value, str) and _PUBLIC_ID.match(value):
            result[key] = value
    for key in ("chain_head_hash", "archive_digest"):
        value = payload.get(key)
        if isinstance(value, str) and _SHA256.match(value):
            result[key] = value
    for key in ("quota_observed_at", "quota_reset_at"):
        value = payload.get(key)
        if isinstance(value, str) and _ISO8601.match(value):
            result[key] = value
    if payload.get("quota_source") in ("provider_usage", "authoritative_header"):
        result["quota_source"] = payload["quota_source"]
    schema_version = payload.get("schema_version")
    archive_version = payload.get("archive_version")
    # v3 is deliberately a closed triplet: it records that formal sleep was
    # governed by the rolling five-hour window.  Older archives keep their
    # historical projection; malformed/unknown v3 labels are omitted.
    if archive_version in ("sleep_archive_v1", "sleep_archive_v2"):
        result["archive_version"] = archive_version
        if isinstance(schema_version, int) and not isinstance(schema_version, bool) and 1 <= schema_version <= 16:
            result["schema_version"] = schema_version
    elif (archive_version == "sleep_archive_v3" and schema_version == 3
          and payload.get("quota_window_kind") == "rolling_5h"):
        result["archive_version"] = archive_version
        result["schema_version"] = 3
        result["quota_window_kind"] = "rolling_5h"
    sequence = payload.get("chain_head_sequence")
    if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0:
        result["chain_head_sequence"] = sequence
    value = payload.get("event_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        result["event_count"] = value
    for key in ("quota_remaining", "quota_total"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    if ("quota_remaining" in result and "quota_total" in result
            and result["quota_remaining"] > result["quota_total"]):
        result.pop("quota_remaining"); result.pop("quota_total")
    return result


def _safe_seed_policy_payload(payload: Dict[str, Any], event_kind: str) -> Dict[str, Any]:
    """Keep only metadata needed for an aggregate, redacted policy count.

    The timeline never exposes seed cues, provenance, policy limits, nonce,
    policy digest, or policy identifier.  ``policy_event_id`` is retained
    internally in the existing public event relation so the state projection
    can count revocations/applications, but it is not returned by the
    aggregate projection.
    """
    result: Dict[str, Any] = {}
    if event_kind == "seed_standing_policy":
        value = payload.get("expires_at")
        if isinstance(value, str) and _ISO8601.match(value):
            result["expires_at"] = value
    elif event_kind == "seed_standing_policy_revoked":
        value = payload.get("policy_event_id")
        if isinstance(value, str) and _PUBLIC_ID.match(value):
            result["policy_event_id"] = value
    elif event_kind == "seed_auto_applied":
        policy_event_id = payload.get("policy_event_id")
        operation = payload.get("operation")
        if isinstance(policy_event_id, str) and _PUBLIC_ID.match(policy_event_id):
            result["policy_event_id"] = policy_event_id
        if operation in _SEED_AUTO_OPERATIONS:
            result["operation"] = operation
    return result


def _safe_payload(payload: Any, event_kind: str = "") -> Dict[str, Any]:
    """Return a strict public projection; unknown/nested data is omitted."""
    if not isinstance(payload, dict):
        return {}
    if event_kind == _MIRROR_KIND:
        return _safe_mirror_payload(payload)
    if event_kind == "sleep_archive":
        return _safe_sleep_archive_payload(payload)
    if event_kind in _SEED_POLICY_KINDS:
        return _safe_seed_policy_payload(payload, event_kind)
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        name = str(key).lower()
        if name in _FORBIDDEN or any(part in name for part in _FORBIDDEN):
            continue
        if key in _PUBLIC_TEXT:
            text = _safe_text(value)
            if text is not None:
                result[key] = text
        elif key in _PUBLIC_SCALARS and (isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes)):
            if isinstance(value, str):
                text = _safe_text(value)
                if text is not None:
                    result[key] = text
            else:
                result[key] = value
    return result


def project_event(event: Any) -> Dict[str, Any]:
    """Project one record without raw payloads or hash material."""
    return {
        "sequence": getattr(event, "sequence", None),
        "event_id": _safe_text(getattr(event, "event_id", "")) or "",
        "kind": _kind(getattr(event, "kind", "unknown")),
        "source_kind": _kind(getattr(event, "source_kind", "unknown")),
        "created_at": _safe_text(getattr(event, "created_at", "")) or "",
        "confidence": getattr(event, "confidence", None),
        "parent_event_ids": [str(item)[:256] for item in getattr(event, "parent_event_ids", ())],
        "payload": _safe_payload(getattr(event, "payload", {}), _kind(getattr(event, "kind", "unknown"))),
    }


def _sleep_projection(raw: Any, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the fixed public sleep/wake inspection surface.

    This is deliberately a projection of coordinator metadata and archive
    records, never of the managed-usage response, archive contents, prompts,
    local paths, or scheduler internals.  ``sleep`` and ``sleep_wake`` are
    accepted only as host snapshot keys to keep the monitor adapter neutral.
    """
    source = raw if isinstance(raw, dict) else {}
    candidate = source.get("sleep_wake", source.get("sleep", {}))
    candidate = candidate if isinstance(candidate, dict) else {}
    state: Dict[str, Any] = {}
    if candidate.get("state") in _SLEEP_STATES:
        state["state"] = candidate["state"]
    for key in ("epoch", "generation", "failure_count"):
        value = candidate.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            state[key] = value
    if isinstance(candidate.get("auto_wake_user_approved"), bool):
        state["auto_wake_user_approved"] = candidate["auto_wake_user_approved"]
    for source_key, public_key in (("last_authoritative_observed_at", "authoritative_observed_at"),
                                   ("reset_at", "authoritative_reset_at")):
        value = candidate.get(source_key)
        if isinstance(value, str) and _ISO8601.match(value):
            state[public_key] = value
    retry_value = candidate.get("retry_at")
    retry_at = retry_value if isinstance(retry_value, str) and _ISO8601.match(retry_value) else None
    reset_at = state.get("authoritative_reset_at")
    if retry_at is not None:
        state["next_check_at"] = retry_at
    elif state.get("state") == "sleeping" and reset_at is not None:
        state["next_check_at"] = reset_at
    # This is a monitor-derived result label, not a claim about a quota reset.
    failures = state.get("failure_count", 0)
    if failures:
        state["last_refresh_outcome"] = "retry_scheduled"
    elif "authoritative_observed_at" in state:
        state["last_refresh_outcome"] = "fresh_authoritative_observation"
    else:
        state["last_refresh_outcome"] = "not_checked"

    archives: List[Dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "sleep_archive":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        archive = {key: payload[key] for key in _SLEEP_ARCHIVE_FIELDS if key in payload}
        if (isinstance(archive.get("archive_digest"), str) and
                isinstance(archive.get("chain_head_hash"), str) and
                isinstance(archive.get("event_count"), int) and
                not isinstance(archive["event_count"], bool)):
            archives.append(archive)
    return {"coordinator": state, "archives": archives[-20:],
            "notice": "Sleep/wake are resource-management metaphors, not biological sleep or consciousness."}


def _unattended_projection(raw: Any, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Project the fixed, non-content-bearing unattended-research surface.

    This adapter accepts either the engine's ``unattended`` snapshot or the
    tool-session's ``research_autonomy`` snapshot.  It intentionally does not
    recursively copy either object: focus text, proposals, URLs, arguments,
    page material, paths, cookies and model rationale do not belong in a local
    monitor.  The card is accounting/termination telemetry, not a control API.
    """
    source = raw if isinstance(raw, dict) else {}
    candidate = source.get("unattended", source.get(
        "unattended_research", source.get("research_autonomy", {})))
    candidate = candidate if isinstance(candidate, dict) else {}
    tool_profile = candidate.get("tool_profile")
    tool_profile = tool_profile if isinstance(tool_profile, dict) else {}
    budget = candidate.get("budget")
    budget = budget if isinstance(budget, dict) else {}

    def _value(*keys: str) -> Any:
        for key in keys:
            if key in candidate:
                return candidate[key]
            if key in tool_profile:
                return tool_profile[key]
            if key in budget:
                return budget[key]
        return None

    state: Dict[str, Any] = {}
    value = _value("state")
    if value in _UNATTENDED_STATES:
        state["state"] = value
    profile_id = _value("profile_id")
    if isinstance(profile_id, str) and _PUBLIC_ID.match(profile_id):
        state["profile_id"] = profile_id
    # A focus/goal is deliberately represented only by a digest; it may
    # contain user text or untrusted instructions and must never be displayed.
    for key in ("goal_digest", "focus_digest"):
        digest = _value(key)
        if isinstance(digest, str) and _SHA256.match(digest):
            state["goal_digest"] = digest
            break
    for target, aliases in (
            ("calls", ("calls", "calls_started")),
            ("max_calls", ("max_calls", "max_tool_calls")),
            ("max_bytes", ("max_bytes", "max_total_bytes", "max_output_bytes")),
            ("ticks", ("ticks",)),
            ("max_ticks", ("max_ticks",)),
    ):
        for key in aliases:
            value = _value(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                state[target] = value
                break
    input_bytes, output_bytes = _value("input_bytes"), _value("output_bytes")
    if (isinstance(input_bytes, int) and not isinstance(input_bytes, bool) and input_bytes >= 0
            and isinstance(output_bytes, int) and not isinstance(output_bytes, bool) and output_bytes >= 0):
        state["bytes_used"] = input_bytes + output_bytes
    else:
        value = _value("bytes_used", "reserved_bytes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            state["bytes_used"] = value
    if "calls" in state and "max_calls" in state:
        state["calls_remaining"] = max(0, state["max_calls"] - state["calls"])
    if "bytes_used" in state and "max_bytes" in state:
        state["bytes_remaining"] = max(0, state["max_bytes"] - state["bytes_used"])
    status = _value("last_status", "last_action_status", "status")
    if isinstance(status, str) and status in ("succeeded", "refused", "failed", "cancelled", "timed_out"):
        state["last_status"] = status
    reason = _value("stop_reason", "stopped_reason")
    if isinstance(reason, str) and reason in _UNATTENDED_REASONS:
        state["stop_reason"] = reason
    tool = _value("last_tool")
    if isinstance(tool, str) and tool in _READ_ONLY_TOOL_NAMES:
        state["last_tool"] = tool
    # A public report is itself an externally inspectable artifact.  The
    # monitor shows only its content-independent identity and cardinality;
    # report text/findings may contain untrusted page material and must not be
    # made available through this local status surface.
    report = candidate.get("public_report")
    report = report if isinstance(report, dict) else {}
    # ``unattended_status`` exposes the compact runtime field ``report_digest``;
    # older/test adapters may nest the same identity under ``public_report``.
    # Prefer the live compact field, but admit only a fixed SHA-256 surface.
    report_digest = candidate.get("report_digest", report.get(
        "report_digest", report.get("digest", candidate.get("public_report_digest"))))
    if isinstance(report_digest, str) and _SHA256.match(report_digest):
        state["public_report_digest"] = report_digest
    finding_count = report.get("finding_count", candidate.get("finding_count"))
    if (isinstance(finding_count, int) and not isinstance(finding_count, bool)
            and 0 <= finding_count <= 100000):
        state["finding_count"] = finding_count
    # A runtime may not retain the last tool in its compact snapshot.  It is
    # safe to derive its *name* and fixed status only from already-projected
    # tool records; do not copy their public summaries (which can contain an
    # untrusted webpage or URL).
    for event in reversed(events):
        if event.get("kind") not in _STAGE_KINDS:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool = payload.get("tool_name")
        if "last_tool" not in state and tool in _READ_ONLY_TOOL_NAMES:
            state["last_tool"] = tool
        status = payload.get("status")
        if "last_status" not in state and status in ("succeeded", "refused", "failed", "cancelled", "timed_out"):
            state["last_status"] = status
        if "last_tool" in state and "last_status" in state:
            break
    return state


def _expedition_projection(raw: Any, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Project fixed expedition accounting without its research content.

    An expedition may contain goals, search queries, URLs, page summaries and
    planner output.  They are all untrusted content and intentionally absent
    here.  This projection admits only compact progress counters and stable
    identifiers; unknown shapes fail closed.
    """
    source = raw if isinstance(raw, dict) else {}
    candidate = source.get("expedition", {})
    candidate = candidate if isinstance(candidate, dict) else {}
    state: Dict[str, Any] = {}

    value = candidate.get("state")
    if value in _EXPEDITION_STATES:
        state["state"] = value
    # Persona and slice are labels for a fixed runtime mode/partition, never
    # user text or planner output.  Accept only identifier-shaped values.
    persona = candidate.get("persona")
    if isinstance(persona, str) and _PUBLIC_ID.match(persona):
        state["persona"] = persona
    for key in ("slice", "slice_index"):
        value = candidate.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000000:
            state["slice"] = value
            break
    for key in ("frontier_coverage", "coverage"):
        value = candidate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0:
            state["frontier_coverage"] = float(value)
            break
    # This is terminal scheduler bookkeeping only.  It is not a quality,
    # evidence, reward, or research-progress signal.
    value = candidate.get("task_terminal_coverage")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0:
        state["task_terminal_coverage"] = float(value)
    for target, aliases in (
            ("domain_count", ("domain_count", "domains_seen")),
            ("useful_findings", ("useful_findings", "useful_finding_count")),
            ("planner_failures", ("planner_failures", "planner_failure_count")),
            ("empty_searches", ("empty_searches", "empty_search_count")),
    ):
        for key in aliases:
            value = candidate.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000000000:
                state[target] = value
                break
    for key in ("seed_digest", "frontier_seed_digest"):
        value = candidate.get(key)
        if isinstance(value, str) and _SHA256.match(value):
            state["seed_digest"] = value
            break
    # The expedition's lifecycle may summarize quota/sleep, but must not
    # become a second source of provider telemetry or expose nested payloads.
    quota = candidate.get("quota", candidate.get("quota_state"))
    if isinstance(quota, str) and quota in _QUOTA_REASONS.union(_QUOTA_FRESHNESS):
        state["quota"] = quota
    sleep = candidate.get("sleep", candidate.get("sleep_state"))
    if isinstance(sleep, str) and sleep in _SLEEP_STATES:
        state["sleep"] = sleep
    reason = candidate.get("stop_reason")
    if isinstance(reason, str) and reason in _EXPEDITION_STOP_REASONS:
        state["stop_reason"] = reason

    # Frontier learning is an auditable *ranking* experiment.  Only its mode,
    # fixed spec fingerprint, fixed scheduler reason, bounded opaque strategy
    # arm, and aggregate counters are public here.  A strategy arm is never a
    # task instance: it can credit only the authorized experiment kind that
    # defines it.  Candidates, task IDs, URLs, query text, seeds and
    # value/reward vectors are deliberately not projected.
    selection = candidate.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    learning = candidate.get("frontier_learning")
    learning = learning if isinstance(learning, dict) else {}
    experiment = candidate.get("experiment")
    experiment = experiment if isinstance(experiment, dict) else {}
    mode = candidate.get("learning_mode", learning.get("mode", selection.get("ranker_mode")))
    if mode in _FRONTIER_LEARNING_MODES:
        state["learning_mode"] = mode
    for key in ("learner_spec_digest", "spec_digest"):
        digest = candidate.get(key, learning.get(key))
        if isinstance(digest, str) and _SHA256.match(digest):
            state["learner_spec_digest"] = digest
            break
    ranking_reason = candidate.get("last_ranking_reason", selection.get("last_reason"))
    if ranking_reason in _FRONTIER_RANKING_REASONS:
        state["last_ranking_reason"] = ranking_reason
    strategy_arm_id = candidate.get("strategy_arm_id")
    strategy_arm_version = candidate.get("strategy_arm_version")
    if not (isinstance(strategy_arm_id, str) and _SHA256.match(strategy_arm_id)
            and strategy_arm_version == _STRATEGY_ARM_VERSION):
        strategy_arm_id = learning.get("strategy_arm_id")
        strategy_arm_version = learning.get("strategy_arm_version")
    if isinstance(strategy_arm_id, str) and _SHA256.match(strategy_arm_id) and strategy_arm_version == _STRATEGY_ARM_VERSION:
        state["strategy_arm_id"] = strategy_arm_id
        state["strategy_arm_version"] = strategy_arm_version
        # This fixed marker makes the causal boundary explicit without
        # exposing a concrete task identifier or experimental input.
        state["strategy_credit_scope"] = "authorized_experiment_kind_only"
    for target, aliases in (
            ("post_reward_selection_count", ("post_reward_selection_count",)),
            ("duplicate_experiment_result_count", ("duplicate_experiment_result_count",)),
    ):
        # Runtime duplicate-result bookkeeping lives in its bounded experiment
        # summary.  We read its scalar only, never any experiment artifact.
        sources = (candidate, learning, experiment) if target == "duplicate_experiment_result_count" else (candidate, learning)
        for source in sources:
            for key in aliases:
                value = source.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000000000:
                    state[target] = value
                    break
            if target in state:
                break
    post_reward_reason = candidate.get("last_post_reward_selection_reason")
    if post_reward_reason not in _POST_REWARD_SELECTION_REASONS:
        post_reward_reason = learning.get("last_post_reward_selection_reason")
    if post_reward_reason in _POST_REWARD_SELECTION_REASONS:
        state["last_post_reward_selection_reason"] = post_reward_reason
    elif mode in _FRONTIER_LEARNING_MODES:
        state["last_post_reward_selection_reason"] = "none"
    if mode in _FRONTIER_LEARNING_MODES and "post_reward_selection_count" not in state:
        state["post_reward_selection_count"] = 0
    if mode in _FRONTIER_LEARNING_MODES and "duplicate_experiment_result_count" not in state:
        state["duplicate_experiment_result_count"] = 0
    for target, aliases in (
            ("duplicate_suppressed", ("duplicate_suppressed", "duplicate_suppressed_count")),
            ("reward_count", ("frontier_reward_count", "reward_count")),
            ("td_update_count", ("frontier_td_update_count", "td_update_count")),
    ):
        for key in aliases:
            value = candidate.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000000000:
                state[target] = value
                break
    # Ledger counts are the source of truth when available.  They also let a
    # legacy snapshot remain safe and useful without opening its internals.
    frontier_reward_count = sum(1 for event in events if event.get("kind") == "frontier_vector_reward")
    frontier_td_count = sum(1 for event in events if event.get("kind") == "frontier_td_update")
    if frontier_reward_count:
        state["reward_count"] = frontier_reward_count
    if frontier_td_count:
        state["td_update_count"] = frontier_td_count

    # Offline fixture experiments are intentionally a second, even smaller
    # projection.  The monitor must not reveal a hypothesis, metric, seed,
    # baseline/treatment artifact, path, output or model self-report.  A
    # passed process or execution count is not a reward qualification.
    experiment_state: Dict[str, Any] = {}
    mode = experiment.get("mode")
    if mode in _EXPERIMENT_MODES:
        experiment_state["mode"] = mode
    approved_kinds = experiment.get("approved_kinds")
    if (isinstance(approved_kinds, list) and 1 <= len(approved_kinds) <= len(_EXPERIMENT_KINDS)
            and len(set(approved_kinds)) == len(approved_kinds)
            and all(item in _EXPERIMENT_KINDS for item in approved_kinds)):
        experiment_state["approved_kinds"] = list(approved_kinds)
    value = experiment.get("completed_count")
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000000000:
        experiment_state["completed_count"] = value
    value = experiment.get("last_kind")
    if value in _EXPERIMENT_KINDS:
        experiment_state["last_kind"] = value
    value = experiment.get("last_status")
    if value in _EXPERIMENT_STATUSES:
        experiment_state["last_status"] = value
    for key in ("reproducible", "control_valid", "reward_qualified"):
        if isinstance(experiment.get(key), bool):
            experiment_state[key] = experiment[key]
    digest = experiment.get("result_digest")
    if isinstance(digest, str) and _SHA256.match(digest):
        experiment_state["result_digest"] = digest
    if experiment_state:
        state["experiment"] = experiment_state
    return state


def _quota_projection(raw: Any) -> Dict[str, Any]:
    """Project only auditable capacity state for the K3 conservation card.

    The monitor must not become a second provider usage API.  In particular,
    it never traverses a provider body and treats unknown/malformed values as
    absent.  Local counters are exposed solely as host-observed accounting and
    are explicitly not represented as a Code Plan balance.
    """
    source = raw if isinstance(raw, dict) else {}
    candidate = source.get("quota", {})
    candidate = candidate if isinstance(candidate, dict) else {}
    state: Dict[str, Any] = {}
    if isinstance(candidate.get("configured"), bool):
        state["configured"] = candidate["configured"]
    authority = candidate.get("authority")
    if authority in _QUOTA_AUTHORITIES:
        state["authority"] = authority
    freshness = candidate.get("freshness")
    if freshness in _QUOTA_FRESHNESS:
        state["freshness"] = freshness
    for key in ("telemetry_known", "allow_call", "is_code_plan_balance"):
        if isinstance(candidate.get(key), bool):
            state[key] = candidate[key]
    reason = candidate.get("reason")
    if reason in _QUOTA_REASONS:
        state["reason"] = reason
    for key in ("local_observed_calls", "local_observed_tokens", "remaining", "total"):
        value = candidate.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000000000:
            state[key] = value
    if ("remaining" in state and "total" in state and
            state["remaining"] > state["total"]):
        state.pop("remaining"); state.pop("total")
    unit = candidate.get("primary_unit")
    if unit in _QUOTA_UNITS:
        state["primary_unit"] = unit
    for key in ("reset_at", "observed_at"):
        value = candidate.get(key)
        if isinstance(value, str) and _ISO8601.match(value):
            state[key] = value
    # A balance is only a provider plan balance when the bounded engine
    # accessor says so *and* a complete authoritative primary measurement is
    # present.  Otherwise fail closed instead of implying remaining capacity.
    if state.get("is_code_plan_balance") is not True:
        state.pop("remaining", None); state.pop("total", None); state.pop("primary_unit", None)
    # This is a label for the normalized provider primary measurement, not a
    # generic lifecycle or local-ledger field.  Keep it coupled to the same
    # authoritative provider-units guard as the displayed plan balance.
    window_kind = candidate.get("primary_window_kind")
    if (state.get("is_code_plan_balance") is True
            and state.get("authority") in ("provider_usage", "authoritative_header")
            and state.get("primary_unit") == "provider_units"
            and window_kind in _QUOTA_WINDOW_KINDS):
        state["primary_window_kind"] = window_kind
    state["local_counters_note"] = "Host-observed calls/tokens; not a K3 Code Plan balance."
    state["resource_note"] = "Resource conservation only; not a reward, drive, or consciousness signal."
    return state


class CognitiveMonitor:
    """Thread-safe source adapter and lifecycle owner for the local server.

    ``snapshot_source`` and ``event_source`` should return public snapshots or
    event objects and be safe to call from a request thread.  For a file-backed
    ``SQLiteEventStore`` the monitor opens a short-lived read-only connection;
    callers using in-memory stores should supply callbacks instead.
    """
    def __init__(self, snapshot_source: Optional[Callable[[], Dict[str, Any]]] = None,
                 event_source: Optional[Callable[[], Iterable[Any]]] = None,
                 event_store: Optional[Any] = None, agent: Optional[Any] = None,
                 session_id: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        # Calling agent.state() implicitly is unsafe for common thread-affine
        # SQLite connections.  A supplied callback explicitly promises it can
        # run in a request thread; file-backed event stores need no callback.
        self._snapshot_source = snapshot_source
        self._loop_source = getattr(agent, "loop_status", None) if agent else None
        # These are bounded in-memory lifecycle accessors.  They are kept
        # separate from agent.state(), which may enumerate a thread-affine
        # SQLite connection and is therefore unsafe in a request thread.
        self._unattended_source = getattr(agent, "unattended_status", None) if agent else None
        self._expedition_source = getattr(agent, "expedition_status", None) if agent else None
        self._sleep_source = getattr(agent, "sleep_status", None) if agent else None
        # Like the lifecycle accessors above, this must be a bounded,
        # thread-safe public accessor; never call agent.state() here because
        # common stores are thread-affine.
        self._quota_source = getattr(agent, "quota_status", None) if agent else None
        self._event_source = event_source
        self._store = event_store or (getattr(agent, "event_store", None) if agent else None)
        self._session_id = session_id or getattr(agent, "session_id", None) or getattr(self._store, "session_id", None)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _events(self) -> List[Any]:
        if self._event_source is not None:
            return list(self._event_source())
        if self._store is None or not self._session_id:
            return []
        path = getattr(self._store, "path", None)
        if path and path != ":memory:":
            connection = sqlite3.connect(path, timeout=1.0)
            try:
                rows = connection.execute("SELECT event_id, sequence, kind, source_kind, payload_json, confidence, parent_event_ids_json, created_at FROM cognitive_events WHERE session_id=? ORDER BY sequence", (self._session_id,)).fetchall()
                return [_SQLitePublicEvent(row) for row in rows]
            finally:
                connection.close()
        # A caller-provided event callback is required for an in-memory,
        # thread-affine SQLite connection.  Failing closed is safer than
        # weakening SQLite's threading guarantees.
        return []

    def state(self) -> Dict[str, Any]:
        with self._lock:
            raw = {}
            if callable(self._snapshot_source):
                try:
                    raw = self._snapshot_source()
                except Exception:
                    # Exception text may contain provider content or paths.
                    raw = {"monitor_snapshot_unavailable": True}
            elif callable(self._loop_source):
                try:
                    raw = {"loop": self._loop_source()}
                except Exception:
                    raw = {"monitor_snapshot_unavailable": True}
            raw = raw if isinstance(raw, dict) else {}
            if callable(self._unattended_source):
                try:
                    raw["unattended_research"] = self._unattended_source()
                except Exception:
                    # Exception content could include a provider message or a path.
                    raw["unattended_research"] = {"state": "stopped", "stop_reason": "host_error"}
            if callable(self._expedition_source):
                try:
                    raw["expedition"] = self._expedition_source()
                except Exception:
                    # Exception text can contain untrusted planner/page data.
                    raw["expedition"] = {"state": "failed", "stop_reason": "host_error"}
            if callable(self._sleep_source):
                try:
                    raw["sleep_wake"] = self._sleep_source()
                except Exception:
                    raw["sleep_wake"] = {"state": "terminal"}
            if callable(self._quota_source):
                try:
                    raw["quota"] = self._quota_source()
                except Exception:
                    raw["quota"] = {"configured": False, "authority": "unknown",
                                    "freshness": "unknown", "allow_call": False,
                                    "reason": "not_configured"}
            try:
                events = [project_event(event) for event in self._events()]
            except Exception:
                events = []
        return _state_projection(raw, events)

    def events(self, after_sequence: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                projected = [project_event(event) for event in self._events()]
            except Exception:
                projected = []
        return [event for event in projected if isinstance(event["sequence"], int) and event["sequence"] > after_sequence]

    def start_background(self, host: str = "127.0.0.1", port: int = 0) -> str:
        if host != "127.0.0.1":
            raise ValueError("monitor may bind only to 127.0.0.1")
        with self._lock:
            if self._server is not None:
                return self.url
            monitor = self
            class Handler(_Handler):
                source = monitor
            self._server = ThreadingHTTPServer((host, port), Handler)
            self._thread = threading.Thread(target=self._server.serve_forever, name="strangeloop-monitor", daemon=True)
            self._thread.start()
            return self.url

    start = start_background

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("monitor is not running")
        return "http://127.0.0.1:%s" % self._server.server_address[1]

    def stop(self) -> None:
        with self._lock:
            server, thread = self._server, self._thread
            self._server = self._thread = None
        if server is not None:
            server.shutdown(); server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)

    close = stop


class _SQLitePublicEvent:
    def __init__(self, row: Any) -> None:
        self.event_id, self.sequence, self.kind, self.source_kind = row[:4]
        self.payload = json.loads(row[4]); self.confidence = row[5]
        self.parent_event_ids = tuple(json.loads(row[6])); self.created_at = row[7]


def _standing_seed_policy_projection(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return counts only for standing seed policy lifecycle records.

    The result intentionally has no policy ID, expiry timestamp, nonce,
    digest, cue, provenance, seed ID, or limit.  It is an operational audit
    summary, not a memory retrieval surface or a new authorization path.
    """
    policies = {}
    revoked = set()
    applications = []
    for record in events:
        kind, payload = record.get("kind"), record.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if kind == "seed_standing_policy":
            policies[record.get("event_id")] = payload.get("expires_at")
        elif kind == "seed_standing_policy_revoked":
            policy_event_id = payload.get("policy_event_id")
            if isinstance(policy_event_id, str):
                revoked.add(policy_event_id)
        elif kind == "seed_auto_applied":
            applications.append(payload)
    now = datetime.now(timezone.utc)
    active = expired = revoked_count = 0
    for event_id, expires_at in policies.items():
        if event_id in revoked:
            revoked_count += 1
            continue
        try:
            is_expired = (not isinstance(expires_at, str)
                          or datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= now)
        except ValueError:
            is_expired = True
        if is_expired:
            expired += 1
        else:
            active += 1
    activation_count = sum(1 for payload in applications if payload.get("operation") == "activate")
    update_count = sum(1 for payload in applications if payload.get("operation") in _SEED_AUTO_OPERATIONS
                       and payload.get("operation") != "activate")
    return {
        "active_count": active,
        "revoked_count": revoked_count,
        "expired_count": expired,
        "auto_activation_count": activation_count,
        "auto_update_count": update_count,
        "notice": "Aggregate policy telemetry only; not memory content, authority, reward, or lifecycle control.",
    }


def _state_projection(raw: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest = lambda kinds: [event for event in events if event["kind"] in kinds][-20:]
    current_goal = _safe_text(raw.get("current_goal") or raw.get("goal") or "")
    decisions = latest(frozenset(("decision",)))
    if not current_goal and decisions:
        current_goal = decisions[-1]["payload"].get("public_summary", "")
    current_turn = raw.get("current_turn") or (decisions[-1]["payload"].get("turn_id") if decisions else None)
    loop = raw.get("loop") if isinstance(raw.get("loop"), dict) else {}
    safe_loop = {key: value for key, value in loop.items() if key in ("state", "run_id", "ticks_completed", "max_ticks", "elapsed_seconds", "restart_behavior") and isinstance(value, (str, int, float, bool))}
    return {"current_goal": current_goal or "", "current_turn": _safe_text(current_turn) if current_turn else None,
            "dmn_loop": safe_loop, "model_invocations": latest(frozenset(("model_invocation",))),
            "tool_stages": latest(_STAGE_KINDS), "seeds_and_claims": latest(frozenset(("seed_proposed", "seed_approved", "seed_retired", "self_claim_proposed", "self_claim_approved", "self_claim_revoked"))),
            "metacognitive_mirrors": latest(frozenset((_MIRROR_KIND,))),
            "standing_seed_policy": _standing_seed_policy_projection(events),
            "learning_and_stops": latest(frozenset(("value_estimate", "rpe_update", "autonomy_stopped", "correction"))),
            "sleep_wake": _sleep_projection(raw, events),
            "quota": _quota_projection(raw),
            "unattended": _unattended_projection(raw, events),
            "expedition": _expedition_projection(raw, events),
            "event_count": len(events), "notice": "Hidden chain-of-thought is neither stored nor shown."}


class _Handler(BaseHTTPRequestHandler):
    source: CognitiveMonitor
    def log_message(self, format: str, *args: Any) -> None:  # no request content in logs
        return
    def _send(self, code: int, body: Any, content_type: str = "application/json; charset=utf-8") -> None:
        encoded = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(encoded))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(encoded)
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/": self._send(200, dashboard_html(), "text/html; charset=utf-8"); return
        if parsed.path == "/api/health": self._send(200, {"status": "ok", "bind": "127.0.0.1"}); return
        if parsed.path == "/api/state": self._send(200, self.source.state()); return
        if parsed.path == "/api/events":
            raw = parse_qs(parsed.query).get("after_sequence", ["0"])[0]
            try: after = max(0, int(raw))
            except (TypeError, ValueError): self._send(400, {"error": "after_sequence must be a non-negative integer"}); return
            self._send(200, {"events": self.source.events(after), "after_sequence": after}); return
        self._send(404, {"error": "not found"})
