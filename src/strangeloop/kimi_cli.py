"""Read-only managed-usage adapter for the locally installed Kimi Code CLI.

The adapter deliberately uses the CLI's OAuth access token only in memory and
only for Kimi Code's fixed managed-usage endpoint.  It does not accept regular
API keys, alter CLI configuration, refresh credentials, or retain raw HTTP
bodies.  The returned observation is suitable for the host quota controller;
it is not a reward, drive, or model-visible credential.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .quota import QuotaController, QuotaSnapshot, QuotaSource


KIMI_CODE_MANAGED_USAGE_URL = "https://api.kimi.com/coding/v1/usages"
_MAX_RESPONSE_BYTES = 64 * 1024


class KimiCliUsageError(RuntimeError):
    """A redacted managed-usage failure with a stable public category."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class UsageHttpResponse:
    """Decoded transport result.  It intentionally has no headers or body text."""

    status_code: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class KimiCliUsageResult:
    """A safe result: one snapshot or a concise unknown/error category."""

    snapshot: Optional[QuotaSnapshot]
    error_category: Optional[str]
    source: str = "kimi_cli_oauth_managed_usage"
    windows: Tuple["ManagedUsageWindow", ...] = ()

    def __post_init__(self) -> None:
        if (self.snapshot is None) == (self.error_category is None):
            raise ValueError("result must contain exactly one of snapshot or error_category")
        if self.snapshot is None and self.windows:
            raise ValueError("unknown usage cannot contain quota windows")
        if any(not isinstance(window, ManagedUsageWindow) for window in self.windows):
            raise ValueError("windows must contain ManagedUsageWindow values")

    @property
    def known(self) -> bool:
        return self.snapshot is not None


@dataclass(frozen=True)
class ManagedUsageWindow:
    """One public Kimi Code allowance window with no account identifiers."""

    kind: str
    total: int
    remaining: int
    reset_at: datetime

    def __post_init__(self) -> None:
        if self.kind not in ("weekly", "rolling_5h"):
            raise ValueError("managed usage window kind is invalid")
        if (not isinstance(self.total, int) or isinstance(self.total, bool)
                or not isinstance(self.remaining, int) or isinstance(self.remaining, bool)
                or self.total < 0 or not 0 <= self.remaining <= self.total):
            raise ValueError("managed usage window values are invalid")
        if self.reset_at.tzinfo is None or self.reset_at.utcoffset() is None:
            raise ValueError("managed usage reset_at must be timezone-aware")

    def to_public_dict(self) -> Mapping[str, Any]:
        return {"kind": self.kind, "used": self.total - self.remaining,
                "limit": self.total, "remaining": self.remaining,
                "remaining_ratio": (float(self.remaining) / self.total
                                    if self.total else 0.0),
                "reset_at": self.reset_at.astimezone(timezone.utc).isoformat()}


class CredentialReader(Protocol):
    def __call__(self) -> Mapping[str, Any]:
        """Return one parsed OAuth credential mapping without logging it."""


class UsageTransport(Protocol):
    def __call__(self, url: str, headers: Mapping[str, str], timeout_seconds: float) -> UsageHttpResponse:
        """Perform exactly one managed-usage GET request."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str):
        try:
            converted = int(value.strip())
        except (TypeError, ValueError):
            return None
        return converted if converted >= 0 else None
    return None


def _parse_timestamp(value: Any, observed_at: datetime) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Both epoch seconds and epoch milliseconds occur in OAuth-adjacent APIs.
        seconds = float(value) / 1000.0 if value > 100000000000 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    return None


def _reset_at(raw: Mapping[str, Any], observed_at: datetime) -> Optional[datetime]:
    for key in ("reset_at", "resetAt", "reset_time", "resetTime"):
        parsed = _parse_timestamp(raw.get(key), observed_at)
        if parsed is not None:
            return parsed
    for key in ("reset_in", "resetIn", "ttl"):
        seconds = _as_non_negative_int(raw.get(key))
        if seconds is not None:
            return observed_at + timedelta(seconds=seconds)
    return None


def _usage_values(raw: Any, observed_at: datetime) -> Optional[Tuple[int, int, datetime]]:
    if not isinstance(raw, Mapping):
        return None
    total = _as_non_negative_int(raw.get("limit"))
    used = _as_non_negative_int(raw.get("used"))
    remaining = _as_non_negative_int(raw.get("remaining"))
    if total is None:
        return None
    if remaining is None and used is not None:
        remaining = total - used
    if remaining is None or remaining > total:
        return None
    reset_at = _reset_at(raw, observed_at)
    if reset_at is None:
        return None
    return total, remaining, reset_at


def _window_seconds(item: Mapping[str, Any], detail: Mapping[str, Any]) -> Optional[int]:
    window = item.get("window")
    sources = (window,) if isinstance(window, Mapping) else ()
    sources += (item, detail)
    for source in sources:
        duration = _as_non_negative_int(source.get("duration"))
        unit = source.get("timeUnit")
        if duration is None or not isinstance(unit, str):
            continue
        normalized = unit.upper()
        if "MINUTE" in normalized:
            return duration * 60
        if "HOUR" in normalized:
            return duration * 60 * 60
        if "DAY" in normalized:
            return duration * 24 * 60 * 60
        if "SECOND" in normalized:
            return duration
    return None


def normalize_managed_usage(payload: Mapping[str, Any], observed_at: Optional[datetime] = None) -> QuotaSnapshot:
    """Normalize the CLI's weekly plus five-hour windows to the tightest one.

    Kimi CLI 0.29.2 reads ``usage`` for the weekly summary and ``limits`` for
    individual rolling windows.  The controller accepts one primary dimension,
    so this deliberately chooses the least available valid window rather than
    adding incompatible window units together.
    """
    if not isinstance(payload, Mapping):
        raise KimiCliUsageError("invalid_payload")
    moment = observed_at or _utc_now()
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    moment = moment.astimezone(timezone.utc)
    windows = normalize_managed_usage_windows(payload, moment)
    candidates = [(window.total, window.remaining, window.reset_at, window)
                  for window in windows]
    total, remaining, reset_at, primary_window = min(
        candidates,
        key=lambda row: (float(row[1]) / row[0] if row[0] else 0.0,
                         row[2].timestamp(), row[0]))
    return QuotaSnapshot(total=total, remaining=remaining, reset_at=reset_at,
                         observed_at=moment, confidence=1.0, is_estimate=False,
                         primary_unit="provider_units",
                         primary_window_kind=primary_window.kind)


def normalize_managed_usage_windows(
        payload: Mapping[str, Any], observed_at: Optional[datetime] = None
) -> Tuple[ManagedUsageWindow, ...]:
    """Return the independently auditable weekly and rolling-five-hour windows."""
    if not isinstance(payload, Mapping):
        raise KimiCliUsageError("invalid_payload")
    moment = observed_at or _utc_now()
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    moment = moment.astimezone(timezone.utc)
    windows = []
    weekly = _usage_values(payload.get("usage"), moment)
    if weekly is not None:
        windows.append(ManagedUsageWindow("weekly", weekly[0], weekly[1], weekly[2]))
    limits = payload.get("limits")
    if isinstance(limits, list):
        for item in limits:
            if not isinstance(item, Mapping):
                continue
            detail = item.get("detail")
            detail = detail if isinstance(detail, Mapping) else item
            if _window_seconds(item, detail) != 5 * 60 * 60:
                continue
            parsed = _usage_values(detail, moment)
            if parsed is not None:
                windows.append(ManagedUsageWindow("rolling_5h", parsed[0], parsed[1], parsed[2]))
    if not windows:
        raise KimiCliUsageError("unrecognized_usage_payload")
    # At most one of each known window is expected.  Ambiguity is rejected
    # rather than silently selecting a provider response the host cannot audit.
    if len({window.kind for window in windows}) != len(windows):
        raise KimiCliUsageError("ambiguous_usage_windows")
    return tuple(sorted(windows, key=lambda window: window.kind))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise KimiCliUsageError("unexpected_redirect")


def _validate_usage_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme, parsed.hostname, parsed.port, parsed.path, parsed.query, parsed.fragment) != (
            "https", "api.kimi.com", None, "/coding/v1/usages", "", ""):
        raise ValueError("managed usage URL must be the fixed Kimi Code endpoint")
    return url


def default_oauth_credential_reader() -> Mapping[str, Any]:
    """Read the documented local CLI OAuth JSON once; do not mutate it."""
    path = Path.home() / ".kimi-code" / "credentials" / "kimi-code.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise KimiCliUsageError("oauth_credentials_missing") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KimiCliUsageError("oauth_credentials_invalid") from error
    if not isinstance(data, Mapping):
        raise KimiCliUsageError("oauth_credentials_invalid")
    return data


def default_usage_transport(url: str, headers: Mapping[str, str], timeout_seconds: float) -> UsageHttpResponse:
    """One bounded GET with redirect refusal.  No retry and no raw-body retention."""
    _validate_usage_url(url)
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise KimiCliUsageError("response_too_large")
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise KimiCliUsageError("invalid_json") from error
            if not isinstance(decoded, Mapping):
                raise KimiCliUsageError("invalid_payload")
            return UsageHttpResponse(int(response.getcode()), decoded)
    except KimiCliUsageError:
        raise
    except HTTPError as error:
        return UsageHttpResponse(error.code, {})
    except socket.timeout as error:
        raise KimiCliUsageError("timeout") from error
    except URLError as error:
        if isinstance(getattr(error, "reason", None), socket.timeout):
            raise KimiCliUsageError("timeout") from error
        raise KimiCliUsageError("network_error") from error
    except OSError as error:
        raise KimiCliUsageError("network_error") from error


def _http_error_category(status_code: int) -> str:
    return {
        401: "oauth_unauthorized",
        403: "oauth_forbidden",
        404: "usage_endpoint_unavailable",
        408: "timeout",
        429: "provider_rate_limited",
        402: "provider_quota_exhausted",
    }.get(status_code, "usage_http_error")


class KimiCliManagedUsageAdapter:
    """Safely bridge Kimi CLI managed OAuth usage into host quota telemetry."""

    def __init__(self, credential_reader: CredentialReader = default_oauth_credential_reader,
                 transport: UsageTransport = default_usage_transport, timeout_seconds: float = 8.0,
                 usage_url: str = KIMI_CODE_MANAGED_USAGE_URL) -> None:
        _validate_usage_url(usage_url)
        if not callable(credential_reader) or not callable(transport):
            raise TypeError("credential_reader and transport must be callable")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._credential_reader = credential_reader
        self._transport = transport
        self._timeout_seconds = float(timeout_seconds)
        self._usage_url = usage_url

    def fetch(self, observed_at: Optional[datetime] = None) -> KimiCliUsageResult:
        """Fetch exactly once.  Token, headers, and raw response do not escape."""
        try:
            credential = self._credential_reader()
            token = credential.get("access_token") if isinstance(credential, Mapping) else None
            if not isinstance(token, str) or not token.strip():
                return KimiCliUsageResult(None, "oauth_access_token_missing")
            # Keep this mapping ephemeral: never assign it to object state/result.
            response = self._transport(self._usage_url, {
                "Authorization": "Bearer " + token,
                "Accept": "application/json",
            }, self._timeout_seconds)
            if not isinstance(response, UsageHttpResponse):
                return KimiCliUsageResult(None, "invalid_transport_response")
            if response.status_code < 200 or response.status_code >= 300:
                return KimiCliUsageResult(None, _http_error_category(response.status_code))
            try:
                windows = normalize_managed_usage_windows(response.payload, observed_at)
                snapshot = normalize_managed_usage(response.payload, observed_at)
            except KimiCliUsageError as error:
                return KimiCliUsageResult(None, error.category)
            return KimiCliUsageResult(snapshot, None, windows=windows)
        except KimiCliUsageError as error:
            return KimiCliUsageResult(None, error.category)
        except (OSError, ValueError, TypeError):
            return KimiCliUsageResult(None, "usage_adapter_error")

    def refresh_controller(self, controller: QuotaController,
                           observed_at: Optional[datetime] = None) -> KimiCliUsageResult:
        """Ingest only a verified result as SYSTEM provider usage telemetry."""
        if not isinstance(controller, QuotaController):
            raise TypeError("controller must be a QuotaController")
        result = self.fetch(observed_at)
        if result.snapshot is not None:
            # Keep injected clocks deterministic while retaining the live
            # controller's future-timestamp guard.  ``fetch`` has already
            # validated this aware timestamp and stamped the snapshot with it.
            controller.ingest_snapshot(
                result.snapshot, QuotaSource.PROVIDER_USAGE, "system",
                now=observed_at,
            )
        return result
