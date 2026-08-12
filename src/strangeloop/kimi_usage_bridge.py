"""A bounded loopback bridge to Kimi CLI's OAuth-managed usage endpoint.

The Kimi CLI owns OAuth refresh.  This module starts its local ``kimi web``
server only long enough to read the fixed managed-usage route, then stops it.
It never reads OAuth credentials, persists the server bearer token, or makes a
semantic model request.  The bridge is deliberately separate from the normal
Kimi API-key runtime: its only output is the same redacted usage result used by
the quota controller.
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .kimi_cli import (KimiCliUsageError, KimiCliUsageResult, ManagedUsageWindow,
                       normalize_managed_usage, normalize_managed_usage_windows)
from .quota import (ForegroundRefreshGate, ForegroundRefreshStatus,
                    QuotaController, QuotaSource)


_MAX_RESPONSE_BYTES = 64 * 1024
_STARTUP_OUTPUT_BYTES = 16 * 1024
_USAGE_PATH = "/api/v1/oauth/usage"
_USAGE_QUERY = "provider=managed%3Akimi-code"
_LOCAL_URL_RE = re.compile(r"^\s*local\s*:\s*(\S+)\s*$", re.I)
_TOKEN_LINE_RE = re.compile(
    r"^\s*(?:bearer[ _-]*token|token)\s*[:=]\s*([A-Za-z0-9._~+\-/=]+)\s*$", re.I)
_RESET_HINT_PREFIX = "resets in "
_RESET_HINT_TOKEN_RE = re.compile(r"[0-9]+[dhm]")
_RESET_HINT_ORDER = {"d": 0, "h": 1, "m": 2}
_WEEKLY_RESET_MAX_SECONDS = 8 * 24 * 60 * 60
_ROLLING_5H_RESET_MAX_SECONDS = 6 * 60 * 60
_TRANSIENT_REFRESH_ERRORS = frozenset((
    "bridge_start_failed", "bridge_start_timeout", "bridge_timeout",
    "bridge_network_error", "bridge_http_error", "provider_rate_limited",
))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise KimiCliUsageError("bridge_unexpected_redirect")


@dataclass(frozen=True)
class BridgeHttpResponse:
    """A decoded, body-free loopback response."""

    status_code: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class BridgeReadiness:
    """CLI-selected loopback binding and ephemeral bearer; repr never exposes it."""

    port: int
    bearer_token: str = field(repr=False)


class BridgeTransport(Protocol):
    def __call__(self, url: str, bearer_token: str,
                 timeout_seconds: float) -> BridgeHttpResponse:
        """Perform the one fixed loopback GET."""


class ProcessFactory(Protocol):
    def __call__(self, args: Sequence[str]) -> Any:
        """Start Kimi CLI with stdout/stderr captured."""


class ReadinessReader(Protocol):
    def __call__(self, process: Any, timeout_seconds: float) -> BridgeReadiness:
        """Return the actual loopback binding and bearer advertised at startup."""


def _validate_port(port: Any) -> int:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("bridge_port_invalid")
    return port


def _usage_url(port: int) -> str:
    return "http://127.0.0.1:%d%s?%s" % (_validate_port(port), _USAGE_PATH, _USAGE_QUERY)


def _validate_usage_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.path != _USAGE_PATH or parsed.query != _USAGE_QUERY
            or parsed.fragment):
        raise ValueError("bridge_non_loopback")
    _validate_port(parsed.port)
    return url


def _startup_local_url(value: str) -> tuple[int, str]:
    """Accept only the exact loopback URL printed by ``kimi web``.

    The URL's fragment token is deliberately compared with the separate Token
    line before a readiness value is returned.  Neither raw line is retained.
    """
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query):
        raise KimiCliUsageError("bridge_start_failed")
    try:
        port = _validate_port(parsed.port)
    except (TypeError, ValueError):
        raise KimiCliUsageError("bridge_start_failed")
    try:
        fragment = parse_qs(parsed.fragment, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise KimiCliUsageError("bridge_start_failed")
    if set(fragment) != {"token"} or len(fragment["token"]) != 1:
        raise KimiCliUsageError("bridge_start_failed")
    token = fragment["token"][0]
    if not _TOKEN_LINE_RE.match("Token: " + token):
        raise KimiCliUsageError("bridge_start_failed")
    return port, token


def _validated_readiness(value: Any) -> BridgeReadiness:
    """Validate injected readers too, so they cannot redirect the fixed GET."""
    if not isinstance(value, BridgeReadiness):
        raise KimiCliUsageError("bridge_start_failed")
    try:
        port = _validate_port(value.port)
    except (TypeError, ValueError):
        raise KimiCliUsageError("bridge_start_failed")
    if (not isinstance(value.bearer_token, str)
            or not _TOKEN_LINE_RE.match("Token: " + value.bearer_token)):
        raise KimiCliUsageError("bridge_start_failed")
    return BridgeReadiness(port=port, bearer_token=value.bearer_token)


def _startup_line_queue(stream: Any) -> "queue.Queue[Optional[str]]":
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def collect() -> None:
        total = 0
        try:
            while total <= _STARTUP_OUTPUT_BYTES:
                line = stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line)
                total += len(text.encode("utf-8", "replace"))
                lines.put(text)
        finally:
            lines.put(None)

    threading.Thread(target=collect, daemon=True).start()
    return lines


def default_readiness_reader(process: Any, timeout_seconds: float) -> BridgeReadiness:
    """Read bounded startup output and require matching URL and bearer lines."""
    stream = getattr(process, "stdout", None)
    if stream is None:
        raise KimiCliUsageError("bridge_start_failed")
    deadline = time.monotonic() + timeout_seconds
    lines = _startup_line_queue(stream)
    actual_port = None
    url_token = None
    bearer_token = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise KimiCliUsageError("bridge_start_timeout")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty as error:
            raise KimiCliUsageError("bridge_start_timeout") from error
        if line is None:
            raise KimiCliUsageError("bridge_start_failed")
        local = _LOCAL_URL_RE.match(line)
        if local:
            port, token = _startup_local_url(local.group(1))
            if ((actual_port is not None and actual_port != port)
                    or (url_token is not None and url_token != token)):
                raise KimiCliUsageError("bridge_start_failed")
            actual_port, url_token = port, token
        token_line = _TOKEN_LINE_RE.match(line)
        if token_line:
            token = token_line.group(1)
            if bearer_token is not None and bearer_token != token:
                raise KimiCliUsageError("bridge_start_failed")
            bearer_token = token
        if actual_port is not None and url_token is not None and bearer_token is not None:
            if url_token != bearer_token:
                raise KimiCliUsageError("bridge_start_failed")
            return BridgeReadiness(port=actual_port, bearer_token=bearer_token)


def default_process_factory(args: Sequence[str]) -> Any:
    return subprocess.Popen(list(args), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            close_fds=True)


def default_bridge_transport(url: str, bearer_token: str,
                             timeout_seconds: float) -> BridgeHttpResponse:
    """GET the one route with no redirects, cookie jar, or raw-body retention."""
    _validate_usage_url(url)
    if not isinstance(bearer_token, str) or not bearer_token:
        raise KimiCliUsageError("bridge_auth_unavailable")
    request = Request(url, headers={"Authorization": "Bearer " + bearer_token,
                                    "Accept": "application/json"}, method="GET")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise KimiCliUsageError("bridge_response_too_large")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise KimiCliUsageError("bridge_invalid_json") from error
            if not isinstance(payload, Mapping):
                raise KimiCliUsageError("bridge_invalid_payload")
            return BridgeHttpResponse(int(response.getcode()), payload)
    except KimiCliUsageError:
        raise
    except HTTPError as error:
        return BridgeHttpResponse(int(error.code), {})
    except (socket.timeout, TimeoutError) as error:
        raise KimiCliUsageError("bridge_timeout") from error
    except URLError as error:
        if isinstance(getattr(error, "reason", None), socket.timeout):
            raise KimiCliUsageError("bridge_timeout") from error
        raise KimiCliUsageError("bridge_network_error") from error
    except OSError as error:
        raise KimiCliUsageError("bridge_network_error") from error


def _http_error_category(status_code: int) -> str:
    return {
        401: "oauth_login_required",
        403: "oauth_forbidden",
        404: "usage_endpoint_unavailable",
        408: "bridge_timeout",
        429: "provider_rate_limited",
    }.get(status_code, "bridge_http_error")


def _loopback_usage_error() -> KimiCliUsageError:
    """Use one public category for every rejected loopback response shape."""
    return KimiCliUsageError("unrecognized_usage_payload")


def _strict_usage_value(value: Any) -> int:
    """Accept only native non-negative integers from the CLI loopback API."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _loopback_usage_error()
    return value


def _reset_hint_seconds(value: Any, maximum_seconds: int) -> int:
    """Parse Kimi CLI's bounded, human-readable reset hint without retaining it."""
    if not isinstance(value, str) or not value.startswith(_RESET_HINT_PREFIX):
        raise _loopback_usage_error()
    tail = value[len(_RESET_HINT_PREFIX):]
    tokens = _RESET_HINT_TOKEN_RE.findall(tail)
    # Rebuilding the complete tail rejects unknown words, duplicate spaces,
    # repeated units, and non-canonical ordering rather than guessing intent.
    if not tokens or " ".join(tokens) != tail:
        raise _loopback_usage_error()
    seconds = 0
    previous = -1
    for token in tokens:
        unit = token[-1]
        order = _RESET_HINT_ORDER[unit]
        if order <= previous:
            raise _loopback_usage_error()
        previous = order
        amount = int(token[:-1])
        if unit == "d":
            seconds += amount * 24 * 60 * 60
        elif unit == "h":
            seconds += amount * 60 * 60
        else:
            seconds += amount * 60
    if seconds <= 0 or seconds > maximum_seconds:
        raise _loopback_usage_error()
    return seconds


def _loopback_usage_record(raw: Any, label: str, maximum_reset_seconds: int) -> Mapping[str, int]:
    """Validate one exact provider record and emit only legacy-normalizer fields."""
    if not isinstance(raw, Mapping) or set(raw) != {"label", "used", "limit", "reset_hint"}:
        raise _loopback_usage_error()
    if raw.get("label") != label:
        raise _loopback_usage_error()
    used = _strict_usage_value(raw.get("used"))
    limit = _strict_usage_value(raw.get("limit"))
    if used > limit:
        raise _loopback_usage_error()
    return {"used": used, "limit": limit,
            "reset_in": _reset_hint_seconds(raw.get("reset_hint"), maximum_reset_seconds)}


def _unwrap_loopback_managed_usage(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert the CLI web endpoint's fixed envelope to the legacy normalizer.

    The loopback endpoint is a distinct provider contract from the older direct
    managed-usage endpoint.  This strict adapter deliberately discards its
    message, request id, and every unknown field before downstream parsing.
    Legacy payloads are still returned unchanged for compatibility with the
    older adapter and its tests.
    """
    if "code" not in payload and "data" not in payload:
        return payload
    if (set(payload) != {"code", "msg", "data", "request_id"}
            or payload.get("code") != 0
            or not isinstance(payload.get("msg"), str)
            or not isinstance(payload.get("request_id"), str)):
        raise _loopback_usage_error()
    data = payload.get("data")
    if not isinstance(data, Mapping) or set(data) != {"kind", "summary", "limits", "extra_usage"}:
        raise _loopback_usage_error()
    if data.get("kind") != "ok" or data.get("extra_usage") is not None:
        raise _loopback_usage_error()
    weekly = _loopback_usage_record(data.get("summary"), "Weekly limit",
                                    _WEEKLY_RESET_MAX_SECONDS)
    limits = data.get("limits")
    if not isinstance(limits, list) or len(limits) != 1:
        raise _loopback_usage_error()
    five_hour = _loopback_usage_record(limits[0], "5h limit", _ROLLING_5H_RESET_MAX_SECONDS)
    # The old normalizer already owns the public QuotaSnapshot conversion.
    return {"usage": weekly,
            "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                        "detail": five_hour}]}


def _stop_process(process: Any) -> None:
    """Best-effort process cleanup.  Cleanup errors cannot reveal CLI output."""
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except Exception:
                process.kill()
                process.wait(timeout=2.0)
    except Exception:
        pass
    for name in ("stdout", "stderr"):
        stream = getattr(process, name, None)
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


class KimiCliOAuthUsageBridge:
    """One-shot Kimi CLI OAuth refresh-and-usage bridge over loopback only."""

    def __init__(self, kimi_binary: str = "kimi", timeout_seconds: float = 8.0,
                 process_factory: ProcessFactory = default_process_factory,
                 readiness_reader: ReadinessReader = default_readiness_reader,
                 transport: BridgeTransport = default_bridge_transport) -> None:
        if not isinstance(kimi_binary, str) or not kimi_binary or os.path.sep in kimi_binary:
            raise ValueError("kimi_binary_invalid")
        if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
                or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive")
        if not all(callable(item) for item in (process_factory, readiness_reader, transport)):
            raise TypeError("bridge collaborators must be callable")
        self._kimi_binary = kimi_binary
        self._timeout_seconds = float(timeout_seconds)
        self._process_factory = process_factory
        self._readiness_reader = readiness_reader
        self._transport = transport
        # Process-local only.  The cached result is the already-normalized,
        # redacted KimiCliUsageResult; it contains neither the loopback bearer
        # nor an OAuth credential/raw response.  A new bridge instance starts
        # empty after process restart.
        self._foreground_lock = threading.RLock()
        self._foreground_cached_result: Optional[KimiCliUsageResult] = None

    def _clear_foreground_cache(self) -> None:
        with self._foreground_lock:
            self._foreground_cached_result = None

    def _cached_foreground_result(self, evidence: Any,
                                  now: datetime) -> Optional[KimiCliUsageResult]:
        """Return only a cache still identical to controller evidence."""
        snapshot = getattr(evidence, "snapshot", None)
        if not getattr(evidence, "allow", False) or snapshot is None:
            return None
        with self._foreground_lock:
            cached = self._foreground_cached_result
            if (cached is None or cached.snapshot is None
                    or cached.snapshot.observed_at != snapshot.observed_at
                    or cached.snapshot.reset_at != snapshot.reset_at
                    or cached.snapshot.primary_window_kind != snapshot.primary_window_kind
                    or now >= cached.snapshot.reset_at):
                return None
            return cached

    def _cache_foreground_result(self, result: KimiCliUsageResult) -> None:
        with self._foreground_lock:
            self._foreground_cached_result = result

    @staticmethod
    def _canonicalize_controller_result(controller: QuotaController,
                                        result: KimiCliUsageResult) -> KimiCliUsageResult:
        """Mirror a controller-clamped provider reset into safe result fields.

        Kimi's rounded five-hour reset hint can be accepted by the controller
        with the earlier raw reset clamped upward.  Returning that raw hint to
        the engine would split the same provider observation into two reset
        epochs.  The bridge therefore returns the controller's canonical
        snapshot and only the matching primary managed window, without
        retaining the raw hint or any credential/HTTP data.  A weekly primary
        snapshot must never rewrite a rolling-five-hour reset boundary: the
        two windows are independent provider epochs.
        """
        if result.snapshot is None:
            return result
        canonical = controller.canonical_provider_snapshot()
        if canonical is None or canonical.observed_at != result.snapshot.observed_at:
            return result
        windows = tuple(
            replace(window, reset_at=canonical.reset_at)
            if (isinstance(window, ManagedUsageWindow)
                and window.kind == canonical.primary_window_kind
                and window.reset_at == result.snapshot.reset_at)
            else window
            for window in result.windows)
        return replace(result, snapshot=canonical, windows=windows)

    @staticmethod
    def _is_transient_refresh_error(category: Optional[str]) -> bool:
        return category in _TRANSIENT_REFRESH_ERRORS

    @staticmethod
    def _rolling_archive_threshold(windows: Sequence[Any], threshold: float) -> Optional[bool]:
        """Return a bounded rolling-five-hour checkpoint result.

        The bridge never derives it from model output, a token ledger, or a
        generic plan balance.  Missing or ambiguous rolling-window telemetry is
        represented by ``None`` so unattended callers can fail closed.
        """
        rolling = [item for item in windows if getattr(item, "kind", None) == "rolling_5h"]
        if len(rolling) != 1:
            return None
        window = rolling[0]
        total, remaining = getattr(window, "total", None), getattr(window, "remaining", None)
        if (not isinstance(total, int) or isinstance(total, bool) or total <= 0
                or not isinstance(remaining, int) or isinstance(remaining, bool)
                or not 0 <= remaining <= total):
            return None
        return float(remaining) / total <= threshold

    def fetch(self, observed_at: Optional[datetime] = None) -> KimiCliUsageResult:
        """Start once, GET once, normalize once, and always stop the local server."""
        process = None
        try:
            # Let the CLI bind atomically to an OS-selected port.  It emits
            # that actual binding and a matching one-time bearer on startup;
            # the reader is the sole stdout consumer and nothing is forwarded.
            args = (self._kimi_binary, "web", "--no-open", "--host", "127.0.0.1",
                    "--port", "0")
            process = self._process_factory(args)
            if process is None:
                return KimiCliUsageResult(None, "bridge_start_failed")
            readiness = _validated_readiness(
                self._readiness_reader(process, self._timeout_seconds))
            response = self._transport(_usage_url(readiness.port), readiness.bearer_token,
                                       self._timeout_seconds)
            if not isinstance(response, BridgeHttpResponse):
                return KimiCliUsageResult(None, "invalid_transport_response")
            if response.status_code < 200 or response.status_code >= 300:
                return KimiCliUsageResult(None, _http_error_category(response.status_code))
            try:
                normalized_payload = _unwrap_loopback_managed_usage(response.payload)
                windows = normalize_managed_usage_windows(normalized_payload, observed_at)
                snapshot = normalize_managed_usage(normalized_payload, observed_at)
            except KimiCliUsageError as error:
                return KimiCliUsageResult(None, error.category)
            return KimiCliUsageResult(snapshot, None,
                                      source="kimi_cli_loopback_oauth_managed_usage",
                                      windows=windows)
        except KimiCliUsageError as error:
            return KimiCliUsageResult(None, error.category,
                                      source="kimi_cli_loopback_oauth_managed_usage")
        # Collaborators are injectable for tests, but their diagnostics may
        # include command output or a bearer token.  Collapse every ordinary
        # failure to the same stable public category.
        except Exception:
            return KimiCliUsageResult(None, "bridge_start_failed",
                                      source="kimi_cli_loopback_oauth_managed_usage")
        finally:
            _stop_process(process)

    def refresh_controller(self, controller: QuotaController,
                           observed_at: Optional[datetime] = None) -> KimiCliUsageResult:
        """Fetch one redacted OAuth-managed observation and ingest it once.

        This mirrors the legacy adapter surface so the foreground sleep poll
        can use the bridge without ever receiving an OAuth token or raw body.
        """
        if not isinstance(controller, QuotaController):
            raise TypeError("controller must be a QuotaController")
        result = self.fetch(observed_at)
        if result.snapshot is not None:
            accepted = controller.ingest_snapshot(result.snapshot, QuotaSource.PROVIDER_USAGE,
                                                  "system", now=result.snapshot.observed_at)
            if accepted:
                result = self._canonicalize_controller_result(controller, result)
        return result

    def refresh_foreground_slice(self, controller: QuotaController,
                                 gate: ForegroundRefreshGate,
                                 now: Optional[datetime] = None,
                                 *, force: bool = False) -> ForegroundRefreshStatus:
        """Perform at most one safe foreground refresh and classify auto-use.

        This helper starts no thread, timer, or background process.  A runtime
        calls it at a foreground slice boundary and must honour
        ``automatic_allowed`` before scheduling further unattended work.  An
        unavailable bridge, rejected snapshot, or stale controller evidence is
        explicitly fail-closed for automatic operation while leaving the
        controller's interactive semantics unchanged.
        """
        if not isinstance(controller, QuotaController):
            raise TypeError("controller must be a QuotaController")
        if not isinstance(gate, ForegroundRefreshGate):
            raise TypeError("gate must be a ForegroundRefreshGate")
        if not isinstance(force, bool):
            raise TypeError("force must be a bool")
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        moment = moment.astimezone(timezone.utc)
        if not gate.claim_refresh(moment, force=force):
            evidence = controller.authoritative_wake_evidence(
                moment, required_source=QuotaSource.PROVIDER_USAGE)
            retry_at = gate.next_retry_at()
            retry_pending = retry_at is not None
            return ForegroundRefreshStatus(
                attempted=False, accepted=False,
                # A skipped refresh never grants authority for a new automatic
                # slice.  The cache is retained only to classify cadence and
                # bounded retry state; callers must wait for a newly attempted,
                # accepted provider observation before doing work.
                automatic_allowed=False, archive_threshold_reached=False,
                reason=("quota_refresh_retry_pending" if retry_pending else
                        "quota_refresh_interval_waiting"),
                error_category=None, observed_at=(None if evidence.snapshot is None
                                                   else evidence.snapshot.observed_at),
                next_refresh_at=gate.next_refresh_at(),
                cost_status="unknown_provider_plan_cost",
                usage_result=None, degraded=retry_pending,
                cache_used=False, next_retry_at=retry_at)
        result = self.fetch(moment)
        if result.snapshot is None:
            evidence = controller.authoritative_wake_evidence(
                moment, required_source=QuotaSource.PROVIDER_USAGE)
            cached = self._cached_foreground_result(evidence, moment)
            retry_at = (gate.record_transient_failure(moment)
                        if self._is_transient_refresh_error(result.error_category) and cached is not None
                        else None)
            if retry_at is None:
                self._clear_foreground_cache()
            return ForegroundRefreshStatus(
                attempted=True, accepted=False, automatic_allowed=False,
                archive_threshold_reached=False,
                reason=("quota_refresh_transient_retry_pending" if retry_at is not None
                        else "quota_refresh_unknown_fail_closed"),
                error_category=result.error_category, observed_at=None,
                next_refresh_at=gate.next_refresh_at(),
                # The cache remains internal evidence for the bounded retry;
                # never hand it to a caller as a substitute for this failed
                # refresh or as permission to execute another slice.
                cost_status="unknown_provider_plan_cost", usage_result=(None if retry_at is not None else result),
                degraded=retry_at is not None, cache_used=retry_at is not None,
                next_retry_at=retry_at)
        accepted = controller.ingest_snapshot(result.snapshot, QuotaSource.PROVIDER_USAGE,
                                              "system", now=moment)
        if not accepted:
            self._clear_foreground_cache()
            return ForegroundRefreshStatus(
                attempted=True, accepted=False, automatic_allowed=False,
                archive_threshold_reached=False,
                reason="quota_refresh_rejected_fail_closed", error_category=None,
                observed_at=result.snapshot.observed_at,
                next_refresh_at=gate.next_refresh_at(),
                cost_status="unknown_provider_plan_cost", usage_result=result)
        result = self._canonicalize_controller_result(controller, result)
        evidence = controller.authoritative_wake_evidence(
            moment, required_source=QuotaSource.PROVIDER_USAGE,
            minimum_observed_at=result.snapshot.observed_at)
        threshold = self._rolling_archive_threshold(result.windows, gate.policy.archive_threshold)
        if threshold is None:
            self._clear_foreground_cache()
            return ForegroundRefreshStatus(
                attempted=True, accepted=True, automatic_allowed=False,
                archive_threshold_reached=False,
                reason="quota_refresh_missing_rolling_5h_fail_closed", error_category=None,
                observed_at=result.snapshot.observed_at,
                next_refresh_at=gate.next_refresh_at(),
                cost_status="unknown_provider_plan_cost", usage_result=result)
        gate.record_success(result.snapshot.observed_at, bool(threshold))
        if evidence.allow:
            self._cache_foreground_result(result)
        else:
            self._clear_foreground_cache()
        return ForegroundRefreshStatus(
            attempted=True, accepted=True, automatic_allowed=evidence.allow,
            archive_threshold_reached=bool(threshold),
            reason=("quota_archive_threshold_reached" if threshold else
                    ("authoritative_quota_available" if evidence.allow else
                     # A just-accepted pre-reset observation is valid
                     # provenance but cannot authorize post-reset work.  Keep
                     # the controller's exact recoverable reason so the
                     # foreground scheduler can retry at its next cadence.
                     "quota_reset_requires_fresh_telemetry"
                     if evidence.reason == "quota_reset_requires_fresh_telemetry" else
                     "quota_refresh_old_or_unknown_fail_closed")),
            error_category=None, observed_at=result.snapshot.observed_at,
            next_refresh_at=gate.next_refresh_at(),
            cost_status=("provider_plan_units_not_currency" if evidence.allow
                         else "unknown_provider_plan_cost"),
            usage_result=result)

    refresh_foreground = refresh_foreground_slice
