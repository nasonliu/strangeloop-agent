"""Session-local capability grants and tamper-evident tool plans.

This module deliberately does *not* execute anything or persist grants.  A
host must create a fresh registry for every process/session and explicitly
pass a user-authorized :class:`ToolPlan` to a controlled executor.  This keeps
an old grant from silently becoming active after a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
import threading
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from uuid import uuid4
from urllib.parse import urlsplit

from .contracts import SourceKind, parse_aware_iso8601, utc_now_iso


class Capability(str, Enum):
    """Narrow, read-only capabilities supported by the first tool slice."""

    REPO_STATUS = "repo.status"
    REPO_SEARCH = "repo.search"
    REPO_READ = "repo.read"
    REPO_WRITE_TEXT = "repo.write_text"
    TEST_SUITE = "repo.test_suite"
    WEB_FETCH = "web.fetch"
    WEB_SEARCH = "web.search"
    BROWSER_READ = "browser.read"


class GrantStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXHAUSTED = "exhausted"
    RESTART_SUSPENDED = "restart_suspended"


_RESEARCH_CAPABILITIES = (
    Capability.REPO_STATUS, Capability.REPO_SEARCH, Capability.REPO_READ,
    Capability.WEB_FETCH, Capability.WEB_SEARCH, Capability.BROWSER_READ,
)
_PUBLIC_WEB_RESEARCH_CAPABILITIES = (
    Capability.WEB_FETCH, Capability.WEB_SEARCH, Capability.BROWSER_READ,
)
_PUBLIC_WEB_BLOCKED_HOSTS = frozenset((
    "localhost", "metadata.google.internal", "metadata", "instance-data",
))


TOOL_CAPABILITIES = {
    "repo.status": Capability.REPO_STATUS,
    "repo.search": Capability.REPO_SEARCH,
    "repo.read": Capability.REPO_READ,
    "repo.write_text": Capability.REPO_WRITE_TEXT,
    "repo.test_suite": Capability.TEST_SUITE,
    "web.fetch": Capability.WEB_FETCH,
    "web.search": Capability.WEB_SEARCH,
    "browser.read": Capability.BROWSER_READ,
}

_TOOL_ARGUMENT_KEYS = {
    "repo.status": frozenset(("workspace_id",)),
    "repo.search": frozenset(("workspace_id", "query")),
    "repo.read": frozenset(("workspace_id", "path")),
    "repo.write_text": frozenset(("workspace_id", "path", "expected_sha256", "content")),
    "repo.test_suite": frozenset(("workspace_id", "test_selector")),
    "web.fetch": frozenset(("url",)),
    "web.search": frozenset(("query",)),
    "browser.read": frozenset(("url",)),
}


def _new_id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid4().hex)


def _now(value: Optional[str] = None) -> datetime:
    return parse_aware_iso8601(value) if value is not None else datetime.now(timezone.utc)


def _normal_domain(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 253:
        raise ValueError("allowed domain must be a non-empty hostname")
    if value != value.strip() or ":" in value or "/" in value or "@" in value:
        raise ValueError("allowed domain must be a hostname, not a URL")
    try:
        normalized = value.encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise ValueError("allowed domain is not valid IDNA") from error
    labels = normalized.split(".")
    if any(not label or len(label) > 63 or not label.replace("-", "").isalnum()
           or label.startswith("-") or label.endswith("-") for label in labels):
        raise ValueError("allowed domain is malformed")
    return normalized


def _is_public_web_hostname(hostname: str) -> bool:
    """Reject names that are never meaningful public-web destinations.

    This is intentionally only a syntactic guard.  The network adapter remains
    responsible for resolving a hostname and rejecting private/reserved IP
    addresses on every connection and redirect.
    """
    host = _normal_domain(hostname)
    if host in _PUBLIC_WEB_BLOCKED_HOSTS or host.endswith(".localhost"):
        return False
    # A literal address is not an ordinary public web name and bypasses the
    # adapter's DNS policy.  Avoid importing an IP parser here to keep this
    # scope module independent of networking behaviour.
    if all(part.isdigit() for part in host.split(".")):
        return False
    return True


def _freeze(value: Any, depth: int = 0) -> Any:
    """Copy only small JSON-like data so plan digests cannot be mutated later."""
    if depth > 4:
        raise ValueError("tool plan arguments are nested too deeply")
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (value != value or value in (float("inf"), -float("inf"))):
            raise ValueError("tool plan arguments cannot contain non-finite floats")
        if isinstance(value, str) and len(value) > 2048:
            raise ValueError("tool plan strings are too long")
        return value
    if isinstance(value, (tuple, list)):
        if len(value) > 32:
            raise ValueError("tool plan lists are too long")
        return tuple(_freeze(item, depth + 1) for item in value)
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise ValueError("tool plan mappings are too large")
        copied = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key or len(key) > 80:
                raise ValueError("tool plan keys must be bounded non-empty strings")
            if key.casefold() in {"chain_of_thought", "hidden_reasoning", "private_reasoning", "scratchpad"}:
                raise ValueError("tool plans cannot contain private-reasoning fields")
            copied[key] = _freeze(nested, depth + 1)
        return MappingProxyType(copied)
    raise ValueError("tool plan arguments must be JSON-like values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


@dataclass(frozen=True)
class GrantScope:
    """A non-secret scope reference; filesystem roots stay in the host runtime."""

    workspace_id: Optional[str] = None
    allowed_domains: Tuple[str, ...] = ()
    public_https: bool = False

    def __post_init__(self) -> None:
        if self.workspace_id is not None:
            if (not isinstance(self.workspace_id, str) or not self.workspace_id
                    or len(self.workspace_id) > 128):
                raise ValueError("workspace_id must be a bounded non-empty string")
        domains = tuple(_normal_domain(domain) for domain in self.allowed_domains)
        if len(domains) > 32 or len(set(domains)) != len(domains):
            raise ValueError("allowed domains must be a bounded unique list")
        if not isinstance(self.public_https, bool):
            raise ValueError("public_https must be a boolean")
        if self.public_https and (self.workspace_id is not None or domains):
            raise ValueError("public HTTPS scope cannot be combined with another scope")
        object.__setattr__(self, "allowed_domains", domains)


@dataclass(frozen=True)
class ResearchBudget:
    """Fixed host-side limits for one unattended, read-only research session."""

    max_tool_calls: int = 100
    max_total_bytes: int = 10 * 1024 * 1024
    max_response_bytes: int = 128 * 1024
    max_wall_ms: int = 5 * 60 * 1000
    ttl_seconds: int = 30 * 60

    def __post_init__(self) -> None:
        limits = ((self.max_tool_calls, "max_tool_calls", 1, 1000),
                  (self.max_total_bytes, "max_total_bytes", 1024, 100 * 1024 * 1024),
                  (self.max_response_bytes, "max_response_bytes", 1024, 1024 * 1024),
                  (self.max_wall_ms, "max_wall_ms", 1000, 60 * 60 * 1000),
                  (self.ttl_seconds, "ttl_seconds", 1, 24 * 60 * 60))
        for value, name, minimum, maximum in limits:
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise ValueError("%s is outside its safe research-profile range" % name)
        if self.max_response_bytes > self.max_total_bytes:
            raise ValueError("max_response_bytes cannot exceed max_total_bytes")


@dataclass(frozen=True)
class ResearchAutonomyProfile:
    """Immutable, read-only unattended research policy.

    It is a host configuration, not a model authority.  It never includes
    shell/command execution, credential access, writes, tests, uploads,
    POST-like requests, purchases, or authenticated browsing.
    """

    workspace_id: str
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    profile_id: str = field(default_factory=lambda: _new_id("research"))
    issued_at: str = field(default_factory=utc_now_iso)
    public_web_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, str) or not self.workspace_id or len(self.workspace_id) > 128:
            raise ValueError("workspace_id must be bounded non-empty text")
        if not isinstance(self.budget, ResearchBudget):
            raise ValueError("budget must be a ResearchBudget")
        if not isinstance(self.profile_id, str) or not self.profile_id or len(self.profile_id) > 128:
            raise ValueError("profile_id must be bounded non-empty text")
        if not isinstance(self.public_web_only, bool):
            raise ValueError("public_web_only must be a boolean")
        parse_aware_iso8601(self.issued_at)

    @property
    def capabilities(self) -> Tuple[Capability, ...]:
        return (_PUBLIC_WEB_RESEARCH_CAPABILITIES if self.public_web_only
                else _RESEARCH_CAPABILITIES)

    def binding_digest(self) -> str:
        """Return the stable public profile binding used for wake continuations.

        The mode is represented by the exact capability tuple, so a public-web
        profile cannot be substituted for a repository-capable profile (or the
        reverse) after user approval.
        """
        budget = self.budget
        public = {"workspace_id": self.workspace_id, "budget": {
            "max_tool_calls": budget.max_tool_calls, "max_total_bytes": budget.max_total_bytes,
            "max_response_bytes": budget.max_response_bytes, "max_wall_ms": budget.max_wall_ms,
            "ttl_seconds": budget.ttl_seconds},
            "capabilities": tuple(item.value for item in self.capabilities)}
        return sha256(_canonical(public).encode("utf-8")).hexdigest()

    def grants_for(self, session_id: str) -> Tuple[CapabilityGrant, ...]:
        """Mint session-local grants; callers must still register them atomically."""
        issued = parse_aware_iso8601(self.issued_at)
        expires = (issued + timedelta(seconds=self.budget.ttl_seconds)).isoformat()
        grants = []
        for capability in self.capabilities:
            scope = (GrantScope(workspace_id=self.workspace_id)
                     if capability.value.startswith("repo.") else GrantScope(public_https=True))
            grants.append(CapabilityGrant(session_id=session_id, capability=capability, scope=scope,
                                          expires_at=expires, max_uses=self.budget.max_tool_calls,
                                          requires_per_call_confirmation=False))
        return tuple(grants)


@dataclass(frozen=True)
class CapabilityGrant:
    """A user-issued, bounded grant held only in a live registry."""

    session_id: str
    capability: Capability
    scope: GrantScope
    expires_at: str
    max_uses: int
    requires_per_call_confirmation: bool = False
    grant_id: str = field(default_factory=lambda: _new_id("grant"))
    issued_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id or len(self.session_id) > 128:
            raise ValueError("session_id must be a bounded non-empty string")
        if not isinstance(self.capability, Capability):
            raise ValueError("capability must be a Capability")
        if not isinstance(self.scope, GrantScope):
            raise ValueError("scope must be a GrantScope")
        if not isinstance(self.max_uses, int) or isinstance(self.max_uses, bool) or not 1 <= self.max_uses <= 1000:
            raise ValueError("max_uses must be an integer between 1 and 1000")
        if not isinstance(self.requires_per_call_confirmation, bool):
            raise ValueError("requires_per_call_confirmation must be a boolean")
        if not isinstance(self.grant_id, str) or not self.grant_id or len(self.grant_id) > 128:
            raise ValueError("grant_id must be a bounded non-empty string")
        issued = parse_aware_iso8601(self.issued_at)
        if parse_aware_iso8601(self.expires_at) <= issued:
            raise ValueError("grant expiry must be after issuance")
        if self.capability in (Capability.REPO_STATUS, Capability.REPO_SEARCH, Capability.REPO_READ,
                               Capability.REPO_WRITE_TEXT, Capability.TEST_SUITE):
            if self.scope.workspace_id is None or self.scope.allowed_domains or self.scope.public_https:
                raise ValueError("repository grants require only a workspace_id scope")
        else:
            if self.scope.workspace_id is not None or (not self.scope.allowed_domains and not self.scope.public_https):
                raise ValueError("web/browser grants require an explicit domain list or public HTTPS scope")
        if self.capability in (Capability.REPO_WRITE_TEXT, Capability.TEST_SUITE):
            if not self.requires_per_call_confirmation:
                raise ValueError("high-risk repository grants require per-call confirmation")


@dataclass(frozen=True)
class ToolPlan:
    """A typed, digest-bound proposal.  It is not an execution permission itself."""

    session_id: str
    grant_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    plan_id: str = field(default_factory=lambda: _new_id("toolplan"))
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id or len(self.session_id) > 128:
            raise ValueError("session_id must be a bounded non-empty string")
        if not isinstance(self.grant_id, str) or not self.grant_id or len(self.grant_id) > 128:
            raise ValueError("grant_id must be a bounded non-empty string")
        if self.tool_name not in TOOL_CAPABILITIES:
            raise ValueError("tool_name is not a supported controlled tool")
        if not isinstance(self.plan_id, str) or not self.plan_id or len(self.plan_id) > 128:
            raise ValueError("plan_id must be a bounded non-empty string")
        frozen = _freeze(self.arguments)
        if not isinstance(frozen, Mapping):
            raise ValueError("tool plan arguments must be a mapping")
        object.__setattr__(self, "arguments", frozen)
        expected = self.compute_digest(self.session_id, self.grant_id, self.tool_name,
                                       self.plan_id, frozen)
        if self.digest and self.digest != expected:
            raise ValueError("tool plan digest does not match its public fields")
        object.__setattr__(self, "digest", expected)

    @staticmethod
    def compute_digest(session_id: str, grant_id: str, tool_name: str,
                       plan_id: str, arguments: Mapping[str, Any]) -> str:
        public = {"session_id": session_id, "grant_id": grant_id, "tool_name": tool_name,
                  "plan_id": plan_id, "arguments": _thaw(arguments)}
        return sha256(_canonical(public).encode("utf-8")).hexdigest()

    def to_public_dict(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "session_id": self.session_id,
                "grant_id": self.grant_id, "tool_name": self.tool_name,
                "arguments": _thaw(self.arguments), "digest": self.digest}


@dataclass(frozen=True)
class GrantSnapshot:
    grant: CapabilityGrant
    status: GrantStatus
    uses_consumed: int


@dataclass(frozen=True)
class ToolConfirmation:
    """A one-use user confirmation bound to a particular immutable plan digest."""

    plan_id: str
    plan_digest: str
    source_kind: SourceKind
    confirmation_id: str = field(default_factory=lambda: _new_id("confirm"))

    def __post_init__(self) -> None:
        if self.source_kind != SourceKind.USER:
            raise PermissionError("only a user may confirm a high-risk tool plan")
        for value, name, length in ((self.plan_id, "plan_id", 128),
                                    (self.confirmation_id, "confirmation_id", 128)):
            if not isinstance(value, str) or not value or len(value) > length:
                raise ValueError("%s must be bounded non-empty text" % name)
        if (not isinstance(self.plan_digest, str) or len(self.plan_digest) != 64
                or any(char not in "0123456789abcdef" for char in self.plan_digest)):
            raise ValueError("plan_digest must be a lowercase SHA-256 digest")


class CapabilityRegistry:
    """Thread-safe, process-local registry with fail-closed atomic consumption."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._grants: Dict[str, CapabilityGrant] = {}
        self._status: Dict[str, GrantStatus] = {}
        self._uses: Dict[str, int] = {}
        self._used_confirmations = set()

    def grant(self, grant: CapabilityGrant, source_kind: SourceKind) -> CapabilityGrant:
        if source_kind != SourceKind.USER:
            raise PermissionError("only a user may grant capabilities")
        with self._lock:
            if grant.grant_id in self._grants:
                raise ValueError("grant_id already exists")
            self._grants[grant.grant_id] = grant
            self._status[grant.grant_id] = GrantStatus.ACTIVE
            self._uses[grant.grant_id] = 0
        return grant

    def revoke(self, grant_id: str, source_kind: SourceKind) -> GrantSnapshot:
        if source_kind != SourceKind.USER:
            raise PermissionError("only a user may revoke capabilities")
        with self._lock:
            grant = self._require(grant_id)
            self._status[grant_id] = GrantStatus.REVOKED
            return self._snapshot(grant_id)

    def suspend_after_restart(self) -> Tuple[GrantSnapshot, ...]:
        """Explicitly make every in-memory grant inactive after host recovery."""
        with self._lock:
            for grant_id, status in list(self._status.items()):
                if status == GrantStatus.ACTIVE:
                    self._status[grant_id] = GrantStatus.RESTART_SUSPENDED
            return tuple(self._snapshot(grant_id) for grant_id in sorted(self._grants))

    def snapshot(self, grant_id: str) -> GrantSnapshot:
        with self._lock:
            self._require(grant_id)
            return self._snapshot(grant_id)

    def consume(self, plan: ToolPlan, now: Optional[str] = None,
                confirmation: Optional[ToolConfirmation] = None) -> CapabilityGrant:
        """Atomically spend one use only after all scope and digest checks pass."""
        if not isinstance(plan, ToolPlan):
            raise TypeError("plan must be a ToolPlan")
        expected = ToolPlan.compute_digest(plan.session_id, plan.grant_id, plan.tool_name,
                                           plan.plan_id, plan.arguments)
        if plan.digest != expected:
            raise ValueError("tool plan digest verification failed")
        with self._lock:
            grant = self._require(plan.grant_id)
            if grant.session_id != plan.session_id:
                raise PermissionError("tool plan does not belong to the grant session")
            if grant.capability != TOOL_CAPABILITIES[plan.tool_name]:
                raise PermissionError("tool plan is outside the granted capability")
            if self._status[grant.grant_id] != GrantStatus.ACTIVE:
                raise PermissionError("capability grant is not active")
            current = _now(now)
            if current >= parse_aware_iso8601(grant.expires_at):
                self._status[grant.grant_id] = GrantStatus.REVOKED
                raise PermissionError("capability grant has expired")
            if self._uses[grant.grant_id] >= grant.max_uses:
                self._status[grant.grant_id] = GrantStatus.EXHAUSTED
                raise PermissionError("capability grant has no remaining uses")
            self._validate_scope(grant, plan)
            if grant.requires_per_call_confirmation:
                if not isinstance(confirmation, ToolConfirmation):
                    raise PermissionError("this high-risk capability requires user confirmation")
                if (confirmation.plan_id != plan.plan_id or confirmation.plan_digest != plan.digest
                        or confirmation.confirmation_id in self._used_confirmations):
                    raise PermissionError("tool confirmation does not bind one unused plan")
                self._used_confirmations.add(confirmation.confirmation_id)
            self._uses[grant.grant_id] += 1
            if self._uses[grant.grant_id] >= grant.max_uses:
                self._status[grant.grant_id] = GrantStatus.EXHAUSTED
            return grant

    def _require(self, grant_id: str) -> CapabilityGrant:
        try:
            return self._grants[grant_id]
        except KeyError as error:
            raise KeyError("unknown capability grant") from error

    def _snapshot(self, grant_id: str) -> GrantSnapshot:
        return GrantSnapshot(self._grants[grant_id], self._status[grant_id],
                             self._uses[grant_id])

    @staticmethod
    def _validate_scope(grant: CapabilityGrant, plan: ToolPlan) -> None:
        args = plan.arguments
        if set(args) != _TOOL_ARGUMENT_KEYS[plan.tool_name]:
            raise ValueError("tool plan arguments do not match the typed tool schema")
        if grant.capability in (Capability.REPO_STATUS, Capability.REPO_SEARCH, Capability.REPO_READ,
                                Capability.REPO_WRITE_TEXT, Capability.TEST_SUITE):
            if args.get("workspace_id") != grant.scope.workspace_id:
                raise PermissionError("repository plan workspace is outside the grant")
            return
        if grant.capability == Capability.WEB_SEARCH:
            query = args["query"]
            if not isinstance(query, str) or not query.strip() or len(query) > 512:
                raise ValueError("web search query must be bounded non-empty text")
            return
        url = args["url"]
        if not isinstance(url, str) or len(url) > 2048:
            raise ValueError("web/browser URL must be bounded text")
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
            raise PermissionError("web/browser plan URL is not an allowed HTTPS URL")
        normalized = _normal_domain(hostname)
        if grant.scope.public_https:
            if not _is_public_web_hostname(normalized):
                raise PermissionError("web/browser plan host is not a public HTTPS hostname")
            # The controlled backend must enforce public-IP resolution and
            # redirects.  Scope validation deliberately cannot trust DNS.
            return
        if normalized not in grant.scope.allowed_domains:
            raise PermissionError("web/browser plan host is outside the grant")
