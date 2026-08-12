"""Controlled, read-only tools with no default browser/search implementation.

These tools return small public outcomes and intentionally do not know about
the event store, model provider, or credentials.  Callers should consume a
``ToolPlan`` from ``CapabilityRegistry`` before calling the executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import http.client
import ipaddress
import json
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import ssl
import stat
import subprocess
import tempfile
import time
import threading
from typing import Callable, Iterable, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urljoin, urlsplit

from .capabilities import CapabilityRegistry, ToolConfirmation, ToolPlan


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(?:api[ _-]?key|access[ _-]?token|refresh[ _-]?token|auth(?:orization)?|"
               r"password|passwd|secret|client[ _-]?secret)\b\s*[:=]\s*[\"']?[^\s\"'<>]{4,}"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|"
               r"gh[pousr]_[A-Za-z0-9]{20,}|xox[a-z]-[A-Za-z0-9-]{10,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]{8,})?\b"),
)


def _redact_public_text(text: str, limit: int, forbidden: Iterable[str] = ()) -> str:
    """Remove common credential shapes and caller-supplied private markers.

    This is defense in depth, not a secret detector.  The network adapters
    never persist raw bodies; this projection is the only body text permitted
    into a :class:`ToolOutcome`.
    """
    if not isinstance(text, str):
        return ""
    value = text.replace("\x00", "")
    for _ in range(3):
        before = value
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub("[REDACTED_SECRET]", value)
        for marker in forbidden:
            if isinstance(marker, str) and marker:
                for representation in (marker, quote(marker, safe=""), quote_plus(marker, safe="")):
                    value = re.sub(re.escape(representation), "[REDACTED_QUERY]", value,
                                   flags=re.IGNORECASE)
        if value == before:
            break
    return " ".join(value.split())[:limit]


def _public_url_projection(url: str) -> str:
    """Project an HTTPS URL without query, fragment, userinfo, or port."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("cannot project a non-HTTPS URL")
    path = parsed.path or "/"
    path = "".join(char for char in path if char >= " " and char not in "\x7f\r\n")
    return "https://%s%s" % (parsed.hostname.encode("idna").decode("ascii").lower(), path)


class ToolStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class ToolOutcome:
    """A bounded public projection; raw tool output is discarded after return."""

    tool_name: str
    status: ToolStatus
    summary: str
    output_sha256: str
    output_bytes: int
    truncated: bool = False
    exit_code: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name or len(self.tool_name) > 80:
            raise ValueError("tool_name must be bounded text")
        if not isinstance(self.status, ToolStatus):
            raise ValueError("status must be a ToolStatus")
        if not isinstance(self.summary, str) or len(self.summary) > 2048:
            raise ValueError("summary must be bounded public text")
        if (not isinstance(self.output_sha256, str) or len(self.output_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.output_sha256)):
            raise ValueError("output_sha256 must be lowercase SHA-256")
        if not isinstance(self.output_bytes, int) or self.output_bytes < 0:
            raise ValueError("output_bytes must be non-negative")

    @classmethod
    def from_bytes(cls, tool_name: str, status: ToolStatus, output: bytes,
                   truncated: bool = False, exit_code: Optional[int] = None,
                   workspace_root: Optional[Path] = None) -> "ToolOutcome":
        text = output.decode("utf-8", "replace").replace("\x00", "")
        if workspace_root is not None:
            text = text.replace(str(workspace_root), "<workspace>")
        text = text.strip()
        if len(text) > 2048:
            text = text[:2048] + " [summary truncated]"
            truncated = True
        if not text:
            text = "No public output."
        return cls(tool_name=tool_name, status=status, summary=text,
                   output_sha256=sha256(output).hexdigest(), output_bytes=len(output),
                   truncated=truncated, exit_code=exit_code)


class WebSearch(Protocol):
    """Host-injected search boundary; this package ships no network backend."""

    def execute(self, plan: ToolPlan, registry: CapabilityRegistry) -> ToolOutcome:
        """Return an already bounded, untrusted public search summary."""


class BrowserRead(Protocol):
    """Host-injected read-only browser boundary; no default browser is supplied."""

    def execute(self, plan: ToolPlan, registry: CapabilityRegistry) -> ToolOutcome:
        """Return bounded visible text only; never submit, authenticate, or upload."""


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_relative(root: Path, relative: str, must_exist: bool = True) -> Path:
    if (not isinstance(relative, str) or not relative or len(relative) > 512
            or "\x00" in relative):
        raise ValueError("path must be bounded non-empty relative text")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PermissionError("path must not escape the workspace")
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PermissionError("symlink paths are not allowed")
    resolved = (root / path).resolve(strict=must_exist)
    if not _inside(root, resolved):
        raise PermissionError("path resolves outside the workspace")
    return resolved


class ControlledToolExecutor:
    """Read-only repository commands with a scrubbed environment and hard caps."""

    def __init__(self, workspace_root: str, timeout_seconds: float = 5.0,
                 max_output_bytes: int = 32 * 1024, max_read_bytes: int = 32 * 1024) -> None:
        root = Path(workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < float(timeout_seconds) <= 60:
            raise ValueError("timeout_seconds must be between 0 and 60")
        for value, name in ((max_output_bytes, "max_output_bytes"), (max_read_bytes, "max_read_bytes")):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1024 * 1024:
                raise ValueError("%s must be between 1 and 1048576" % name)
        self.root = root
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.max_read_bytes = max_read_bytes
        self._write_lock = threading.RLock()

    def execute(self, plan: ToolPlan, registry: Optional[CapabilityRegistry] = None,
                confirmation: Optional[ToolConfirmation] = None) -> ToolOutcome:
        """Consume an optional grant, then execute exactly one supported read-only plan."""
        try:
            if registry is None:
                return self._refused(plan.tool_name, "an active capability registry is required")
            self._validate_plan(plan)
            registry.consume(plan, confirmation=confirmation)
            if plan.tool_name == "repo.status":
                return self._repo_status()
            if plan.tool_name == "repo.search":
                return self._repo_search(plan.arguments["query"])
            if plan.tool_name == "repo.read":
                return self._repo_read(plan.arguments["path"])
            if plan.tool_name == "repo.write_text":
                return self._repo_write_text(plan.arguments["path"], plan.arguments["expected_sha256"],
                                             plan.arguments["content"])
            if plan.tool_name == "repo.test_suite":
                return self._run_test_suite(plan.arguments["test_selector"])
            return self._refused(plan.tool_name, "Unsupported executor tool.")
        except (OSError, PermissionError, ValueError, KeyError) as error:
            return self._refused(plan.tool_name, str(error))

    def _validate_plan(self, plan: ToolPlan) -> None:
        expected = {
            "repo.status": {"workspace_id"}, "repo.search": {"workspace_id", "query"},
            "repo.read": {"workspace_id", "path"},
            "repo.write_text": {"workspace_id", "path", "expected_sha256", "content"},
            "repo.test_suite": {"workspace_id", "test_selector"},
        }
        if plan.tool_name not in expected or set(plan.arguments) != expected[plan.tool_name]:
            raise ValueError("tool plan arguments do not match the fixed template")
        if plan.tool_name in ("repo.read", "repo.write_text"):
            relative = plan.arguments["path"]
            if plan.tool_name == "repo.read":
                _safe_relative(self.root, relative)
            else:
                self._safe_write_target(relative)
                digest = plan.arguments["expected_sha256"]
                content = plan.arguments["content"]
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
                if not isinstance(content, str) or len(content.encode("utf-8")) > self.max_read_bytes:
                    raise ValueError("content must fit the single-file UTF-8 byte limit")
        elif plan.tool_name == "repo.search":
            query = plan.arguments["query"]
            if not isinstance(query, str) or not query or len(query) > 256 or "\x00" in query:
                raise ValueError("query must be bounded non-empty text")
        elif plan.tool_name == "repo.test_suite":
            import re
            if (not isinstance(plan.arguments["test_selector"], str) or not re.fullmatch(
                    r"tests(?:\.[A-Za-z_][A-Za-z0-9_]*)*", plan.arguments["test_selector"])):
                raise ValueError("test_selector must be a dotted tests module")

    def _repo_status(self) -> ToolOutcome:
        return self._run("repo.status", ("/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=no"))

    def _repo_read(self, relative_path: str) -> ToolOutcome:
        try:
            path = _safe_relative(self.root, relative_path)
            if not path.is_file():
                raise ValueError("path is not a regular file")
            with path.open("rb") as handle:
                data = handle.read(self.max_read_bytes + 1)
            truncated = len(data) > self.max_read_bytes
            public = data[:self.max_read_bytes]
            return ToolOutcome.from_bytes("repo.read", ToolStatus.SUCCEEDED, public,
                                          truncated=truncated, workspace_root=self.root)
        except (OSError, PermissionError, ValueError) as error:
            return self._refused("repo.read", str(error))

    def _repo_write_text(self, relative_path: str, expected_sha256: str, content: str) -> ToolOutcome:
        """Atomically replace one regular UTF-8 file after a local hash recheck.

        The two hash checks are optimistic concurrency control.  They protect
        against ordinary competing writers but are not a substitute for a
        filesystem sandbox against a hostile process racing directory entries.
        """
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha256)):
            return self._refused("repo.write_text", "expected_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(content, str):
            return self._refused("repo.write_text", "content must be UTF-8 text")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_read_bytes:
            return self._refused("repo.write_text", "content exceeds the single-file byte limit")
        try:
            with self._write_lock:
                return self._repo_write_text_locked(relative_path, expected_sha256, encoded)
        except (OSError, PermissionError, ValueError) as error:
            return self._refused("repo.write_text", str(error))

    def _repo_write_text_locked(self, relative_path: str, expected_sha256: str,
                                encoded: bytes) -> ToolOutcome:
        """Run a write under the executor-local lock after argument validation."""
        try:
            target, parent = self._safe_write_target(relative_path)
            name = target.name
            directory_fd = os.open(str(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                   | getattr(os, "O_NOFOLLOW", 0))
            try:
                before = self._hash_regular_at(directory_fd, name)
                if before != expected_sha256:
                    return self._refused("repo.write_text", "expected file digest does not match current content")
                temporary = ".strangeloop-write-%d-%d" % (os.getpid(), time.time_ns())
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600, dir_fd=directory_fd)
                try:
                    offset = 0
                    while offset < len(encoded):
                        offset += os.write(fd, encoded[offset:])
                    os.fsync(fd)
                finally:
                    os.close(fd)
                try:
                    if self._hash_regular_at(directory_fd, name) != expected_sha256:
                        return self._refused("repo.write_text", "file changed before atomic replacement")
                    os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                    os.fsync(directory_fd)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(directory_fd)
            return ToolOutcome.from_bytes("repo.write_text", ToolStatus.SUCCEEDED,
                                          ("Wrote %d UTF-8 bytes after digest verification." % len(encoded)).encode("utf-8"))
        except FileNotFoundError as error:
            raise ValueError("write target parent no longer exists") from error

    def _safe_write_target(self, relative_path: str) -> Tuple[Path, Path]:
        if (not isinstance(relative_path, str) or not relative_path or len(relative_path) > 512
                or "\x00" in relative_path):
            raise ValueError("path must be bounded non-empty relative text")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts or path.name in ("", "."):
            raise PermissionError("path must name one file inside the workspace")
        parent_relative = path.parent.as_posix()
        parent = self.root if parent_relative == "." else _safe_relative(self.root, parent_relative)
        target = parent / path.name
        if target.exists() and target.is_symlink():
            raise PermissionError("symlink targets are not allowed")
        return target, parent

    def _hash_regular_at(self, directory_fd: int, name: str) -> str:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return sha256(b"").hexdigest()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise PermissionError("write target must be a regular file")
        if info.st_size > self.max_read_bytes:
            raise PermissionError("write target exceeds the single-file byte limit")
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        try:
            chunks = []
            while True:
                block = os.read(descriptor, 8192)
                if not block:
                    break
                chunks.append(block)
            return sha256(b"".join(chunks)).hexdigest()
        finally:
            os.close(descriptor)

    def _run_test_suite(self, test_selector: str) -> ToolOutcome:
        """Execute only a dotted unittest module selector, never arbitrary Python."""
        import re
        if (not isinstance(test_selector, str) or not re.fullmatch(
                r"tests(?:\.[A-Za-z_][A-Za-z0-9_]*)*", test_selector)):
            return self._refused("repo.test_suite", "test_selector must be a dotted tests module")
        # /usr/bin/python3 and the fixed unittest module are deliberate; neither
        # the model nor a plan can substitute an interpreter, flag, or shell.
        return self._run("repo.test_suite", ("/usr/bin/python3", "-m", "unittest", test_selector),
                         extra_env={"PYTHONPATH": str(self.root / "src")})

    def _repo_search(self, query: str) -> ToolOutcome:
        if not isinstance(query, str) or not query or len(query) > 256 or "\x00" in query:
            return self._refused("repo.search", "query must be bounded non-empty text")
        matches = []
        inspected = 0
        for directory, names, filenames in os.walk(str(self.root), followlinks=False):
            names[:] = [name for name in names if name != ".git" and not (Path(directory) / name).is_symlink()]
            for filename in filenames:
                if len(matches) >= 100 or inspected >= 1000:
                    break
                candidate = Path(directory) / filename
                try:
                    if candidate.is_symlink() or not _inside(self.root, candidate.resolve(strict=True)):
                        continue
                    data = candidate.read_bytes()[:self.max_read_bytes]
                except OSError:
                    continue
                inspected += 1
                if b"\x00" in data:
                    continue
                for line_number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                    if query.casefold() in line.casefold():
                        relative = candidate.relative_to(self.root).as_posix()
                        matches.append("%s:%d:%s" % (relative, line_number, line[:240]))
                        if len(matches) >= 100:
                            break
            if len(matches) >= 100 or inspected >= 1000:
                break
        output = "\n".join(matches).encode("utf-8")
        if not output:
            output = b"No matching readable text files."
        return ToolOutcome.from_bytes("repo.search", ToolStatus.SUCCEEDED, output,
                                      truncated=(len(matches) >= 100 or inspected >= 1000),
                                      workspace_root=self.root)

    def _clean_environment(self, home: str) -> Mapping[str, str]:
        return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                "HOME": home, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}

    def _run(self, tool_name: str, argv: Sequence[str],
             extra_env: Optional[Mapping[str, str]] = None) -> ToolOutcome:
        """Run fixed argv without a shell, bounding output while the process runs."""
        with tempfile.TemporaryDirectory(prefix="strangeloop-tool-") as tool_home:
            try:
                environment = dict(self._clean_environment(tool_home))
                if extra_env:
                    environment.update(extra_env)
                process = subprocess.Popen(
                    list(argv), cwd=str(self.root), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False,
                    close_fds=True, start_new_session=True,
                    env=environment,
                )
            except OSError as error:
                return self._refused(tool_name, "controlled executable could not start: %s" % error.__class__.__name__)
            output, truncated, timed_out = self._read_bounded(process)
            if timed_out:
                return ToolOutcome.from_bytes(tool_name, ToolStatus.TIMED_OUT, output,
                                              truncated=truncated, workspace_root=self.root)
            status = ToolStatus.SUCCEEDED if process.returncode == 0 else ToolStatus.FAILED
            return ToolOutcome.from_bytes(tool_name, status, output, truncated=truncated,
                                          exit_code=process.returncode, workspace_root=self.root)

    def _read_bounded(self, process: subprocess.Popen) -> Tuple[bytes, bool, bool]:
        if process.stdout is None:  # pragma: no cover - defensive
            return b"", False, False
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        chunks = []
        size = 0
        truncated = False
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._terminate_group(process)
                    return b"".join(chunks), truncated, True
                events = selector.select(min(remaining, 0.1))
                for key, _ in events:
                    data = os.read(key.fd, min(8192, self.max_output_bytes + 1 - size))
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    chunks.append(data)
                    size += len(data)
                    if size >= self.max_output_bytes:
                        truncated = True
                        self._terminate_group(process)
                        return b"".join(chunks)[:self.max_output_bytes], truncated, False
                if process.poll() is not None and not events:
                    # EOF will normally unregister; this protects unusual pipe states.
                    for key in list(selector.get_map().values()):
                        try:
                            data = os.read(key.fd, self.max_output_bytes + 1 - size)
                        except OSError:
                            data = b""
                        if data:
                            chunks.append(data)
                            size += len(data)
                        selector.unregister(key.fileobj)
            process.wait(timeout=0.2)
            try:
                process.stdout.close()
            except OSError:
                pass
            return b"".join(chunks)[:self.max_output_bytes], truncated, False
        finally:
            selector.close()

    @staticmethod
    def _terminate_group(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _refused(tool_name: str, message: str) -> ToolOutcome:
        return ToolOutcome.from_bytes(tool_name, ToolStatus.REFUSED, message.encode("utf-8"))


Resolver = Callable[[str, int], Iterable[str]]
ConnectionFactory = Callable[[str, int, float], http.client.HTTPSConnection]
CancelCheck = Callable[[], bool]
BudgetCheck = Callable[[int], bool]
DoHTransport = Callable[[str, Mapping[str, str], float], Tuple[int, Mapping[str, str], bytes]]


def _resolve_public(hostname: str, port: int) -> Tuple[str, ...]:
    values = []
    for result in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM):
        address = result[4][0]
        try:
            if not ipaddress.ip_address(address).is_global:
                raise PermissionError("host resolves to a non-public address")
        except ValueError as error:
            raise PermissionError("host resolution is invalid") from error
        if address not in values:
            values.append(address)
    if not values:
        raise PermissionError("host did not resolve to a public address")
    return tuple(values)


class FixedDoHResolver:
    """Resolve public hosts through one fixed, TLS-verified DoH endpoint.

    This resolver is intended for the unattended public-web profile on hosts
    where the normal resolver may synthesize Fake-IP answers.  It bootstraps
    only through hard-coded Cloudflare numeric addresses, with TLS SNI and
    certificate verification for the fixed DoH hostname.  Every resulting
    address is still pinned and rechecked by :class:`PublicWebFetch`.  The
    transport seam exists solely for deterministic tests and host adapters.
    """

    endpoint = "https://cloudflare-dns.com/dns-query"
    _host = "cloudflare-dns.com"
    _bootstrap_addresses = ("1.1.1.1", "1.0.0.1")
    _path = "/dns-query"
    _max_response_bytes = 32 * 1024

    def __init__(self, timeout_seconds: float = 5.0,
                 transport: Optional[DoHTransport] = None) -> None:
        if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
                or not 0 < float(timeout_seconds) <= 30):
            raise ValueError("timeout_seconds must be between 0 and 30")
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport or self._https_transport

    def __call__(self, hostname: str, port: int) -> Tuple[str, ...]:
        # ``port`` is deliberately ignored: PublicWebFetch permits HTTPS:443
        # only, while DoH resolution itself has no caller-controlled endpoint.
        del port
        name = self._normal_name(hostname)
        started = time.monotonic()
        values = []  # type: list[str]
        # v1 intentionally asks only A.  This avoids accepting NAT64 or other
        # IPv6 transition answers until that trust boundary has its own review;
        # an IPv6-only public site therefore fails closed.
        record_type = 1
        remaining = self.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("DoH resolution exceeded total timeout")
        status, headers, body = self._transport(
            self._path + "?" + urlencode({"name": name, "type": str(record_type)}),
            {"Accept": "application/dns-json", "User-Agent": "strangeloop-agent/0.2"}, remaining)
        addresses = self._validate_reply(name, record_type, status, headers, body)
        for address in addresses:
            if address not in values:
                values.append(address)
            if len(values) >= 16:
                return tuple(values)
        if not values:
            raise PermissionError("DoH returned no public address")
        return tuple(values)

    @classmethod
    def _https_transport(cls, path: str, headers: Mapping[str, str], timeout: float
                         ) -> Tuple[int, Mapping[str, str], bytes]:
        """Direct numeric bootstrap; deliberately never calls getaddrinfo.

        Both addresses are Cloudflare-operated fixed bootstrap targets.  A
        connection error may move to the other fixed address inside the same
        caller deadline, but there is never a system-DNS, proxy, redirect, or
        caller-supplied endpoint fallback.
        """
        deadline = time.monotonic() + timeout
        last_error = None
        def remaining_timeout() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DoH resolution exceeded total timeout")
            return remaining
        for address in cls._bootstrap_addresses:
            raw = None
            tls = None
            try:
                raw = socket.create_connection((address, 443), remaining_timeout())
                raw.settimeout(remaining_timeout())
                tls = ssl.create_default_context().wrap_socket(raw, server_hostname=cls._host)
                tls.settimeout(remaining_timeout())
                request = "GET %s HTTP/1.1\r\nHost: %s\r\nAccept: %s\r\nUser-Agent: %s\r\nConnection: close\r\n\r\n" % (
                    path, cls._host, headers["Accept"], headers["User-Agent"])
                tls.sendall(request.encode("ascii"))
                tls.settimeout(remaining_timeout())
                response = http.client.HTTPResponse(tls)
                response.begin()
                length = response.getheader("Content-Length")
                if length is not None and (not length.isdigit() or int(length) > cls._max_response_bytes):
                    raise ValueError("DoH response exceeds the configured size limit")
                tls.settimeout(remaining_timeout())
                body = response.read(cls._max_response_bytes + 1)
                if len(body) > cls._max_response_bytes:
                    raise ValueError("DoH response exceeds the configured size limit")
                return response.status, {"Content-Type": response.getheader("Content-Type") or "",
                                         "Content-Encoding": response.getheader("Content-Encoding") or ""}, body
            except (OSError, ssl.SSLError, http.client.HTTPException, TimeoutError) as error:
                last_error = error
            finally:
                if tls is not None:
                    tls.close()
                elif raw is not None:
                    raw.close()
        if last_error is not None:
            raise last_error
        raise OSError("DoH numeric bootstrap failed")

    @staticmethod
    def _normal_name(value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 253 or "\x00" in value:
            raise PermissionError("DoH hostname is invalid")
        try:
            normalized = value.encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise PermissionError("DoH hostname is invalid") from error
        if (normalized.endswith(".") or any(not label or len(label) > 63
                                              or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                              for label in normalized.split("."))):
            raise PermissionError("DoH hostname is invalid")
        return normalized

    @classmethod
    def _response_name(cls, value: str) -> str:
        """Canonicalize exactly one DNS presentation trailing dot.

        This is intentionally not used for the caller request or Question:
        only response record owners and CNAME targets use DNS's common fully
        qualified presentation form.
        """
        if isinstance(value, str) and value.endswith("."):
            value = value[:-1]
        return cls._normal_name(value)

    @classmethod
    def _validate_reply(cls, name: str, record_type: int, status: int,
                        headers: Mapping[str, str], body: bytes) -> Tuple[str, ...]:
        if type(status) is not int or status != 200:
            raise PermissionError("DoH endpoint returned a non-success status")
        content_type = ""
        content_encoding = ""
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == "content-type":
                content_type = str(value).split(";", 1)[0].strip().lower()
            elif isinstance(key, str) and key.lower() == "content-encoding":
                content_encoding = str(value).strip().lower()
        if content_type not in ("application/dns-json", "application/json"):
            raise PermissionError("DoH response content type is invalid")
        if content_encoding not in ("", "identity"):
            raise PermissionError("compressed DoH responses are not accepted")
        if not isinstance(body, bytes) or len(body) > cls._max_response_bytes:
            raise PermissionError("DoH response is invalid")
        try:
            def no_duplicate_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate JSON key")
                    result[key] = value
                return result
            payload = json.loads(body.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
        except (UnicodeDecodeError, ValueError) as error:
            raise PermissionError("DoH response is not valid JSON") from error
        if (not isinstance(payload, dict) or type(payload.get("Status")) is not int
                or payload.get("Status") != 0):
            raise PermissionError("DoH response status is invalid")
        questions = payload.get("Question")
        question_name = questions[0].get("name") if (isinstance(questions, list)
                                                        and len(questions) == 1
                                                        and isinstance(questions[0], dict)) else None
        if (not isinstance(questions, list) or len(questions) != 1
                or not isinstance(questions[0], dict)
                # DNS JSON providers commonly render the echoed Question as a
                # fully-qualified name.  Apply the same one-trailing-dot
                # normalization already used for signed response owners, then
                # still require an exact requested host and A record type.
                or not isinstance(question_name, str) or cls._response_name(question_name) != name
                or type(questions[0].get("type")) is not int
                or questions[0].get("type") != record_type):
            raise PermissionError("DoH response question does not match request")
        answers = payload.get("Answer", [])
        if not isinstance(answers, list) or len(answers) > 32:
            raise PermissionError("DoH answer set is invalid")
        records = []
        for answer in answers:
            if not isinstance(answer, dict) or set(answer) != {"name", "type", "TTL", "data"}:
                raise PermissionError("DoH answer record is invalid")
            answer_type, owner, data, ttl = (answer.get("type"), answer.get("name"),
                                             answer.get("data"), answer.get("TTL"))
            if (type(answer_type) is not int or answer_type not in (1, 5, 28)
                    # TTL is validated for response shape only; this resolver
                    # does not cache it.  Keep the full conventional 32-bit
                    # signed range rather than imposing an arbitrary week cap.
                    or type(ttl) is not int or not 0 <= ttl <= 2147483647
                    or not isinstance(owner, str) or not isinstance(data, str)):
                raise PermissionError("DoH answer record is invalid")
            if answer_type in (1, 28) and answer_type != record_type:
                raise PermissionError("DoH answer type does not match request")
            records.append((cls._response_name(owner), answer_type, data))
        owner_types = {}  # type: dict[str, set[int]]
        for owner, kind, _ in records:
            owner_types.setdefault(owner, set()).add(kind)
        if any(5 in kinds and (1 in kinds or 28 in kinds) for kinds in owner_types.values()):
            raise PermissionError("DoH CNAME owner also has an address record")
        # Follow a CNAME chain only inside the signed JSON response.  Every
        # answer owner must be the request name or a target already reached.
        reachable = [name]
        cursor = name
        for _ in range(4):
            targets = [cls._response_name(data) for owner, kind, data in records
                       if owner == cursor and kind == 5]
            if len(targets) > 1 or (targets and targets[0] in reachable):
                raise PermissionError("DoH CNAME chain is invalid")
            if not targets:
                break
            cursor = targets[0]
            reachable.append(cursor)
        else:
            if any(owner == cursor and kind == 5 for owner, kind, _ in records):
                raise PermissionError("DoH CNAME chain is too long")
        if any(owner not in reachable for owner, _, _ in records):
            raise PermissionError("DoH answer owner is not in CNAME chain")
        values = []
        for owner, kind, data in records:
            if kind == 5:
                continue
            try:
                address = ipaddress.ip_address(data)
            except ValueError as error:
                raise PermissionError("DoH answer address is invalid") from error
            if ((kind == 1 and address.version != 4) or (kind == 28 and address.version != 6)
                    or not address.is_global):
                raise PermissionError("DoH answer is not a public address")
            if str(address) not in values:
                values.append(str(address))
        return tuple(values)


class CloudflareDoHResolver(FixedDoHResolver):
    """Cloudflare's fixed numeric DoH resolver."""

    endpoint = "https://cloudflare-dns.com/dns-query"
    _host = "cloudflare-dns.com"
    _bootstrap_addresses = ("1.1.1.1", "1.0.0.1")
    _path = "/dns-query"


class GoogleDoHResolver(FixedDoHResolver):
    """Google's fixed numeric DoH resolver.

    This is an independently routed, still fixed trust anchor for hosts where
    the Cloudflare numeric bootstrap is unreachable.  It shares the strict
    response parser and direct pinned-TLS transport above: no system DNS,
    proxy discovery, redirect following, or caller-configured endpoint is
    introduced by this alternate provider.
    """

    endpoint = "https://dns.google/resolve"
    _host = "dns.google"
    _bootstrap_addresses = ("8.8.8.8", "8.8.4.4")
    _path = "/resolve"


class PublicWebFetch:
    """Stateless, bounded public-HTTPS reader.

    A fetch is deliberately not a browser: it sends no cookies, has no login
    state, accepts only GET/HEAD, and never writes a response to disk.  Every
    hop is resolved afresh and the default connector dials the validated IP
    directly while retaining the hostname solely for TLS SNI/certificate
    verification.  HTML is *untrusted data*, never an authority to create a
    capability, confirmation, or tool plan.

    ``allowed_domains=()`` means the caller has selected the public-HTTPS
    profile.  The capability registry remains the authorization boundary; old
    registries that require exact domains will still reject plans before this
    class makes a network request.
    """

    def __init__(self, allowed_domains: Iterable[str] = (), timeout_seconds: float = 5.0,
                 max_bytes: int = 128 * 1024, max_redirects: int = 3,
                 resolver: Resolver = _resolve_public,
                 connection_factory: Optional[ConnectionFactory] = None,
                 cancel_check: Optional[CancelCheck] = None,
                 budget_check: Optional[BudgetCheck] = None) -> None:
        from .capabilities import _normal_domain  # keep public API small
        domains = tuple(_normal_domain(domain) for domain in allowed_domains)
        if len(domains) > 32 or len(set(domains)) != len(domains):
            raise ValueError("allowed_domains must be a bounded unique sequence")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < float(timeout_seconds) <= 60:
            raise ValueError("timeout_seconds must be between 0 and 60")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 1 <= max_bytes <= 1024 * 1024:
            raise ValueError("max_bytes must be between 1 and 1048576")
        if not isinstance(max_redirects, int) or isinstance(max_redirects, bool) or not 0 <= max_redirects <= 5:
            raise ValueError("max_redirects must be between 0 and 5")
        self.allowed_domains = domains
        self.timeout_seconds = float(timeout_seconds)
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.resolver = resolver
        # A supplied factory is a test/host integration seam.  It receives the
        # already validated numeric address, never a hostname it could rebind.
        self.connection_factory = connection_factory
        self.cancel_check = cancel_check
        self.budget_check = budget_check

    def execute(self, plan: ToolPlan, registry: CapabilityRegistry) -> ToolOutcome:
        """Consume a web.fetch grant before any URL validation or connection."""
        try:
            if plan.tool_name != "web.fetch":
                return self._failed("web.fetch", "tool plan does not select web.fetch")
            registry.consume(plan)
            return self._fetch(plan.arguments["url"], "GET")
        except (KeyError, PermissionError, ValueError) as error:
            return self._failed("web.fetch", "web fetch refused: %s" % error.__class__.__name__)

    def _check_cancelled(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise TimeoutError("web request was cancelled")

    def _check_budget(self, used_bytes: int) -> None:
        if self.budget_check is not None and not self.budget_check(used_bytes):
            raise TimeoutError("web request exceeded the host network budget")

    def _connection(self, hostname: str, address: str, port: int) -> http.client.HTTPSConnection:
        if self.connection_factory is not None:
            return self.connection_factory(address, port, self.timeout_seconds)
        return _PinnedHTTPSConnection(hostname, address, port, self.timeout_seconds)

    def _fetch(self, url: str, method: str = "GET") -> ToolOutcome:
        """Fetch one constrained request and expose metadata, never raw HTML."""
        try:
            final_url, content_type, data, truncated = self._retrieve(url, method)
            digest = sha256(data).hexdigest()
            private_query_values = tuple(value for candidate in (url, final_url)
                                         for _, value in parse_qsl(urlsplit(candidate).query,
                                                                   keep_blank_values=False) if value)
            public_url = _redact_public_text(_public_url_projection(final_url), 1024,
                                             private_query_values)
            summary = ("Untrusted public HTTPS %s completed: url=%s type=%s bytes=%d sha256=%s"
                       % (method, public_url, content_type or "unknown", len(data), digest))
            return ToolOutcome("web.fetch", ToolStatus.SUCCEEDED, summary, digest, len(data), truncated)
        except TimeoutError as error:
            return ToolOutcome.from_bytes("web.fetch", ToolStatus.TIMED_OUT, str(error).encode("utf-8"))
        except (OSError, ValueError, PermissionError, http.client.HTTPException) as error:
            return self._failed("web.fetch", "web fetch refused: %s" % error.__class__.__name__)

    def _retrieve(self, url: str, method: str = "GET") -> Tuple[str, str, bytes, bool]:
        """Return bounded raw bytes to another safe adapter in this module only."""
        if method not in ("GET", "HEAD"):
            raise PermissionError("only GET and HEAD requests are allowed")
        current = url
        for _ in range(self.max_redirects + 1):
            self._check_cancelled()
            parsed, host, port = self._validate_url(current)
            addresses = tuple(self.resolver(host, port))
            if not addresses:
                raise PermissionError("host did not resolve to a public address")
            address = addresses[0]
            # A custom resolver is not trusted: repeat the public-address check.
            if not ipaddress.ip_address(address).is_global:
                raise PermissionError("host resolves to a non-public address")
            connection = self._connection(host, address, port)
            try:
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                connection.request(method, target, headers={
                    "Accept": "text/plain,text/html,application/json;q=0.9",
                    "Accept-Encoding": "identity", "User-Agent": "strangeloop-agent/0.2",
                    "Host": host,
                    "Cookie": "",
                })
                self._check_cancelled()
                response = connection.getresponse()
                encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
                if encoding not in ("", "identity"):
                    raise PermissionError("compressed responses are not accepted")
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    if not location:
                        raise ValueError("redirect response omitted Location")
                    current = urljoin(current, location)
                    continue
                if not 200 <= response.status < 300:
                    raise ValueError("HTTPS %s returned status %d" % (method, response.status))
                length = response.getheader("Content-Length")
                if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
                    raise ValueError("response exceeds the configured size limit")
                content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
                if content_type and content_type not in {"text/plain", "text/html", "application/json"}:
                    raise ValueError("response content type is not readable text")
                self._check_budget(0)
                data = b"" if method == "HEAD" else response.read(self.max_bytes + 1)
                self._check_cancelled()
                self._check_budget(len(data))
                if len(data) > self.max_bytes:
                    raise ValueError("response exceeds the configured size limit")
                return current, content_type, data, False
            finally:
                connection.close()
        raise ValueError("redirect limit exceeded")

    def _validate_url(self, url: str):
        from .capabilities import _normal_domain
        if not isinstance(url, str) or not url or len(url) > 2048 or "\x00" in url:
            raise ValueError("URL must be bounded non-empty text")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.fragment):
            raise PermissionError("only credential-free HTTPS URLs are allowed")
        if parsed.port not in (None, 443):
            raise PermissionError("only the HTTPS default port is allowed")
        host = _normal_domain(parsed.hostname)
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise PermissionError("literal IP addresses are not allowed")
        if self.allowed_domains and host not in self.allowed_domains:
            raise PermissionError("URL host is outside the fetch allowlist")
        return parsed, host, 443

    @staticmethod
    def _failed(tool_name: str, message: str) -> ToolOutcome:
        return ToolOutcome.from_bytes(tool_name, ToolStatus.FAILED, message.encode("utf-8"))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection that never asks the OS to resolve ``hostname``."""

    def __init__(self, hostname: str, address: str, port: int, timeout: float) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._pinned_address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._pinned_address, self.port), self.timeout,
                                       self.source_address)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


class _VisibleText(HTMLParser):
    """Bounded, inert HTML-to-text projection; it does not execute anything."""

    def __init__(self, limit: int = 1200) -> None:
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.parts = []  # type: list[str]
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        # Form controls and their labels/values are intentionally absent from
        # the persistent projection.  Attribute values are never collected.
        if tag in ("script", "style", "noscript", "template", "form"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "template", "form") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and sum(len(item) for item in self.parts) < self.limit:
            clean = " ".join(data.replace("\x00", "").split())
            if clean:
                self.parts.append(clean)

    def text(self) -> str:
        return _redact_public_text(" ".join(self.parts), self.limit)


class StatelessBrowserRead:
    """Cookie-free HTTP rendering equivalent, not a real browser or sandbox."""

    def __init__(self, fetcher: Optional[PublicWebFetch] = None) -> None:
        self.fetcher = fetcher or PublicWebFetch()

    def execute(self, plan: ToolPlan, registry: CapabilityRegistry) -> ToolOutcome:
        try:
            if plan.tool_name != "browser.read":
                return PublicWebFetch._failed("browser.read", "tool plan does not select browser.read")
            registry.consume(plan)
            requested_url = plan.arguments["url"]
            final_url, content_type, data, truncated = self.fetcher._retrieve(requested_url, "GET")
            private_query_values = tuple(value for candidate in (requested_url, final_url)
                                         for _, value in parse_qsl(urlsplit(candidate).query,
                                                                   keep_blank_values=False) if value)
            if content_type == "text/html":
                parser = _VisibleText()
                parser.feed(data.decode("utf-8", "replace"))
                excerpt = _redact_public_text(parser.text(), 1200, private_query_values)
            else:
                excerpt = _redact_public_text(data.decode("utf-8", "replace"), 1200,
                                              private_query_values)
            digest = sha256(data).hexdigest()
            public_url = _redact_public_text(_public_url_projection(final_url), 1024,
                                             private_query_values)
            summary = ("UNTRUSTED_DATA stateless browser-read: url=%s type=%s bytes=%d sha256=%s\n"
                       "UNTRUSTED_DATA excerpt=%s"
                       % (public_url, content_type or "unknown", len(data), digest,
                          excerpt or "[no visible text]"))
            return ToolOutcome("browser.read", ToolStatus.SUCCEEDED, summary, digest, len(data), truncated)
        except TimeoutError as error:
            return ToolOutcome.from_bytes("browser.read", ToolStatus.TIMED_OUT, str(error).encode("utf-8"))
        except (KeyError, PermissionError, ValueError, OSError, http.client.HTTPException) as error:
            return PublicWebFetch._failed("browser.read", "browser read refused: %s" % error.__class__.__name__)


class _SearchResults(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results = []  # type: list[Tuple[str, str, str]]
        self._href = None
        self._title = []
        self._snippet = []
        self._mode = None

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        classes = values.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._href, self._title, self._mode = values.get("href"), [], "title"
        elif "result__snippet" in classes:
            self._snippet, self._mode = [], "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._mode == "title":
            self._mode = None
            if self._href and len(self.results) < 5:
                self.results.append((self._href, " ".join(self._title), ""))
        elif self._mode == "snippet" and self.results:
            url, title, _ = self.results[-1]
            self.results[-1] = (url, title, " ".join(self._snippet))
            self._mode = None

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if clean and self._mode == "title":
            self._title.append(clean)
        elif clean and self._mode == "snippet":
            self._snippet.append(clean)


class SafePublicWebSearch:
    """Fixed, cookie-free DuckDuckGo HTML search adapter with bounded output."""

    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, fetcher: Optional[PublicWebFetch] = None) -> None:
        self.fetcher = fetcher or PublicWebFetch(("html.duckduckgo.com",))

    def execute(self, plan: ToolPlan, registry: CapabilityRegistry) -> ToolOutcome:
        try:
            if plan.tool_name != "web.search":
                return PublicWebFetch._failed("web.search", "tool plan does not select web.search")
            registry.consume(plan)
            query = plan.arguments["query"]
            url = self.endpoint + "?" + urlencode({"q": query})
            _, _, data, truncated = self.fetcher._retrieve(url, "GET")
            parser = _SearchResults()
            parser.feed(data.decode("utf-8", "replace"))
            rows = []
            for target, title, snippet in parser.results:
                parsed = urlsplit(target)
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                    continue
                try:
                    ipaddress.ip_address(parsed.hostname)
                    continue
                except ValueError:
                    pass
                safe_url = _redact_public_text(_public_url_projection(target), 512, (query,))
                safe_title = _redact_public_text(title, 300, (query,))
                safe_snippet = _redact_public_text(snippet, 500, (query,))
                rows.append("UNTRUSTED_DATA result_url=%s title=%s snippet=%s" %
                            (safe_url, safe_title, safe_snippet))
            raw = data
            summary = "\n".join(rows) or "No safe public HTTPS search results."
            return ToolOutcome("web.search", ToolStatus.SUCCEEDED, summary, sha256(raw).hexdigest(),
                               len(raw), truncated)
        except TimeoutError as error:
            return ToolOutcome.from_bytes("web.search", ToolStatus.TIMED_OUT, str(error).encode("utf-8"))
        except (KeyError, PermissionError, ValueError, OSError, http.client.HTTPException) as error:
            return PublicWebFetch._failed("web.search", "web search refused: %s" % error.__class__.__name__)
