"""Bounded metadata extraction for audio and image observations.

This module deliberately retains neither media bytes nor filesystem paths.
It produces small, inspectable metadata records which a future event-store
schema may attach to a user observation or a tool result.  A ``Percept`` is a
bounded external annotation, not a record of a model's private reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import io
import struct
from collections.abc import Sequence as SequenceABC
from typing import Any, BinaryIO, Dict, Iterable, Optional, Protocol, Sequence, Tuple
from uuid import uuid4


MAX_MEDIA_BYTES = 25 * 1024 * 1024
MAX_IMAGE_DIMENSION = 16384
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_PERCEPTS = 32
MAX_LABELS = 16
MAX_TEXT_LENGTH = 512
MEDIA_PAYLOAD_SCHEMA = "strangeloop.media.v1"
_CHUNK_SIZE = 64 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _new_media_id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid4().hex)


def _bounded_text(value: str, field_name: str, limit: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % field_name)
    if len(value) > limit:
        raise ValueError("%s exceeds its length limit" % field_name)
    return value


@dataclass(frozen=True)
class MediaSpan:
    """An optional temporal audio span or rectangular image region."""

    kind: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "time":
            if (not isinstance(self.start_ms, int) or isinstance(self.start_ms, bool)
                    or not isinstance(self.end_ms, int) or isinstance(self.end_ms, bool)
                    or self.start_ms < 0 or self.end_ms <= self.start_ms):
                raise ValueError("time spans require non-negative start_ms before end_ms")
            if any(value is not None for value in (self.x, self.y, self.width, self.height)):
                raise ValueError("time spans cannot include image-region fields")
        elif self.kind == "region":
            values = (self.x, self.y, self.width, self.height)
            if (any(not isinstance(value, int) or isinstance(value, bool) for value in values)
                    or self.x is None or self.y is None or self.width is None or self.height is None
                    or self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0):
                raise ValueError("region spans require non-negative x/y and positive width/height")
            if self.start_ms is not None or self.end_ms is not None:
                raise ValueError("region spans cannot include time fields")
        else:
            raise ValueError("span kind must be 'time' or 'region'")

    def to_payload(self) -> Dict[str, Any]:
        if self.kind == "time":
            return {"kind": self.kind, "start_ms": self.start_ms, "end_ms": self.end_ms}
        return {"kind": self.kind, "x": self.x, "y": self.y,
                "width": self.width, "height": self.height}

    @classmethod
    def from_payload(cls, value: Dict[str, Any]) -> "MediaSpan":
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            raise ValueError("media span must be an object with a kind")
        if value["kind"] == "time":
            if set(value) != {"kind", "start_ms", "end_ms"}:
                raise ValueError("time span uses an exact schema")
            return cls(kind="time", start_ms=value["start_ms"], end_ms=value["end_ms"])
        if value["kind"] == "region":
            if set(value) != {"kind", "x", "y", "width", "height"}:
                raise ValueError("region span uses an exact schema")
            return cls(kind="region", x=value["x"], y=value["y"],
                       width=value["width"], height=value["height"])
        raise ValueError("span kind must be 'time' or 'region'")


@dataclass(frozen=True)
class MediaArtifact:
    """Metadata for an inspected media byte stream, with no bytes or path."""

    modality: str
    mime_type: str
    byte_length: int
    sha256: str
    artifact_id: str = field(default_factory=lambda: _new_media_id("media"))
    width: Optional[int] = None
    height: Optional[int] = None
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    frame_count: Optional[int] = None
    duration_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if self.modality not in ("image", "audio"):
            raise ValueError("media modality must be image or audio")
        expected = {"image": {"image/png", "image/jpeg"}, "audio": {"audio/wav"}}
        if self.mime_type not in expected[self.modality]:
            raise ValueError("mime type is not supported for this modality")
        if not isinstance(self.byte_length, int) or isinstance(self.byte_length, bool) or not 0 < self.byte_length <= MAX_MEDIA_BYTES:
            raise ValueError("byte_length must be within the media byte budget")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        _bounded_text(self.artifact_id, "artifact_id", 128)
        if self.modality == "image":
            if (not isinstance(self.width, int) or isinstance(self.width, bool)
                    or not isinstance(self.height, int) or isinstance(self.height, bool)
                    or self.width <= 0 or self.height <= 0):
                raise ValueError("image artifact requires positive width and height")
            if (self.width > MAX_IMAGE_DIMENSION or self.height > MAX_IMAGE_DIMENSION
                    or self.width * self.height > MAX_IMAGE_PIXELS):
                raise ValueError("image dimensions exceed the decoded-image budget")
            if any(value is not None for value in (self.sample_rate_hz, self.channels,
                                                    self.frame_count, self.duration_ms)):
                raise ValueError("image artifact cannot include audio metadata")
        else:
            audio_values = (self.sample_rate_hz, self.channels, self.frame_count, self.duration_ms)
            if (any(not isinstance(value, int) or isinstance(value, bool) for value in audio_values)
                    or any(value is None or value < 0 for value in audio_values)
                    or self.sample_rate_hz is None or self.sample_rate_hz <= 0
                    or self.channels is None or self.channels <= 0):
                raise ValueError("audio artifact requires valid non-negative audio metadata")
            if self.width is not None or self.height is not None:
                raise ValueError("audio artifact cannot include image metadata")

    def to_payload(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.modality == "image":
            return {key: value[key] for key in ("artifact_id", "modality", "mime_type",
                                                 "byte_length", "sha256", "width", "height")}
        return {key: value[key] for key in ("artifact_id", "modality", "mime_type",
                                             "byte_length", "sha256", "sample_rate_hz",
                                             "channels", "frame_count", "duration_ms")}

    @classmethod
    def from_payload(cls, value: Dict[str, Any]) -> "MediaArtifact":
        if not isinstance(value, dict) or value.get("modality") not in ("image", "audio"):
            raise ValueError("media artifact has an invalid modality")
        common = {"artifact_id", "modality", "mime_type", "byte_length", "sha256"}
        image_keys = common | {"width", "height"}
        audio_keys = common | {"sample_rate_hz", "channels", "frame_count", "duration_ms"}
        expected = image_keys if value["modality"] == "image" else audio_keys
        if set(value) != expected:
            raise ValueError("media artifact uses an exact schema")
        return cls(**value)


@dataclass(frozen=True)
class Percept:
    """A concise, provenance-bearing annotation returned by a perceptor."""

    artifact_id: str
    modality: str
    summary: str
    labels: Tuple[str, ...]
    confidence: float
    perceptor_id: str
    spans: Tuple[MediaSpan, ...] = ()
    percept_id: str = field(default_factory=lambda: _new_media_id("percept"))

    def __post_init__(self) -> None:
        _bounded_text(self.artifact_id, "artifact_id", 128)
        if self.modality not in ("image", "audio"):
            raise ValueError("percept modality must be image or audio")
        _bounded_text(self.summary, "summary")
        _bounded_text(self.perceptor_id, "perceptor_id", 128)
        _bounded_text(self.percept_id, "percept_id", 128)
        if not isinstance(self.labels, tuple) or len(self.labels) > MAX_LABELS:
            raise ValueError("labels must be a bounded tuple")
        for label in self.labels:
            _bounded_text(label, "label", 128)
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.spans, tuple):
            raise ValueError("spans must be a tuple")
        for span in self.spans:
            if not isinstance(span, MediaSpan):
                raise ValueError("spans must contain MediaSpan values")
            if self.modality == "image" and span.kind != "region":
                raise ValueError("image percept spans must be regions")
            if self.modality == "audio" and span.kind != "time":
                raise ValueError("audio percept spans must be temporal")

    def to_payload(self) -> Dict[str, Any]:
        return {"percept_id": self.percept_id, "artifact_id": self.artifact_id,
                "modality": self.modality, "summary": self.summary,
                "labels": list(self.labels), "confidence": float(self.confidence),
                "perceptor_id": self.perceptor_id,
                "spans": [span.to_payload() for span in self.spans]}

    @classmethod
    def from_payload(cls, value: Dict[str, Any]) -> "Percept":
        expected = {"percept_id", "artifact_id", "modality", "summary", "labels",
                    "confidence", "perceptor_id", "spans"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("percept uses an exact schema")
        if not isinstance(value["labels"], list) or not isinstance(value["spans"], list):
            raise ValueError("percept labels and spans must be lists in payloads")
        return cls(percept_id=value["percept_id"], artifact_id=value["artifact_id"],
                   modality=value["modality"], summary=value["summary"],
                   labels=tuple(value["labels"]), confidence=value["confidence"],
                   perceptor_id=value["perceptor_id"],
                   spans=tuple(MediaSpan.from_payload(span) for span in value["spans"]))


class MediaPerceptor(Protocol):
    """Adapter boundary for transient, read-only media analysis.

    ``stream`` is valid only during the call.  Implementations must return
    concise public observations rather than private deliberation or bytes.
    """

    perceptor_id: str

    def perceive(self, artifact: MediaArtifact, stream: BinaryIO) -> Sequence[Percept]:
        """Read a transient stream and return bounded public annotations."""


class NullMediaPerceptor:
    perceptor_id = "null-media-perceptor/v1"

    def perceive(self, artifact: MediaArtifact, stream: BinaryIO) -> Sequence[Percept]:
        del artifact, stream
        return ()


class BasicMetadataPerceptor:
    """Produces only metadata descriptions; it does not inspect semantic content."""

    perceptor_id = "basic-metadata-perceptor/v1"

    def perceive(self, artifact: MediaArtifact, stream: BinaryIO) -> Sequence[Percept]:
        del stream
        if artifact.modality == "image":
            summary = "%s image metadata: %dx%d pixels" % (
                artifact.mime_type, artifact.width, artifact.height)
            labels = ("image", artifact.mime_type.rsplit("/", 1)[1])
        else:
            summary = "%s audio metadata: %d Hz, %d channels, %d ms" % (
                artifact.mime_type, artifact.sample_rate_hz, artifact.channels, artifact.duration_ms)
            labels = ("audio", "wav")
        return (Percept(artifact_id=artifact.artifact_id, modality=artifact.modality,
                        summary=summary, labels=labels, confidence=1.0,
                        perceptor_id=self.perceptor_id),)


def sha256_stream(stream: BinaryIO, byte_limit: int = MAX_MEDIA_BYTES) -> Tuple[str, int, bytes]:
    """Read a binary stream in bounded chunks and return digest, length, and bytes.

    The returned bytes are an ephemeral parsing buffer; callers must not
    persist them.  A bounded buffer is required because image/WAV headers may
    be non-contiguous.
    """
    if not hasattr(stream, "read"):
        raise ValueError("stream must provide read()")
    if not isinstance(byte_limit, int) or isinstance(byte_limit, bool) or not 0 < byte_limit <= MAX_MEDIA_BYTES:
        raise ValueError("byte_limit must be positive and no greater than MAX_MEDIA_BYTES")
    digest = sha256()
    chunks = []
    count = 0
    while True:
        chunk = stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise ValueError("media stream must return bytes")
        count += len(chunk)
        if count > byte_limit:
            raise ValueError("media exceeds byte budget")
        digest.update(chunk)
        chunks.append(chunk)
    if count == 0:
        raise ValueError("media stream is empty")
    return digest.hexdigest(), count, b"".join(chunks)


def inspect_media(stream: BinaryIO, declared_mime_type: Optional[str] = None,
                  byte_limit: int = MAX_MEDIA_BYTES) -> MediaArtifact:
    """Hash and inspect PNG, JPEG, or WAV streams without retaining their source."""
    digest, length, data = sha256_stream(stream, byte_limit)
    return _artifact_from_data(digest, length, data, declared_mime_type)


class ReadOnlyMediaStream(io.BytesIO):
    """A fresh in-memory stream that permits seeking and reading only.

    The underlying bytes are deliberately scoped to one adapter invocation;
    the object is never included in artifacts, percepts, events, or payloads.
    """

    def write(self, value: bytes) -> int:
        del value
        raise io.UnsupportedOperation("media stream is read-only")

    def writelines(self, lines: Iterable[bytes]) -> None:
        del lines
        raise io.UnsupportedOperation("media stream is read-only")

    def truncate(self, size: Optional[int] = None) -> int:
        del size
        raise io.UnsupportedOperation("media stream is read-only")

    def getbuffer(self) -> memoryview:
        raise io.UnsupportedOperation("media stream buffer is not exposed")


def ingest_media(stream: BinaryIO, perceptor: MediaPerceptor,
                 declared_mime_type: Optional[str] = None,
                 byte_limit: int = MAX_MEDIA_BYTES) -> Tuple[MediaArtifact, Tuple[Percept, ...]]:
    """Inspect one input stream and deliver a transient read-only copy to an adapter.

    Input bytes are consumed exactly once by :func:`sha256_stream`.  They are
    subsequently held only in this stack frame long enough for format parsing
    and the adapter call, then discarded.  Adapter errors deliberately
    propagate; this layer does not create events or silently retain failures.
    """
    if not hasattr(perceptor, "perceive") or not hasattr(perceptor, "perceptor_id"):
        raise ValueError("perceptor must expose perceptor_id and perceive()")
    perceptor_id = _bounded_text(perceptor.perceptor_id, "perceptor_id", 128)
    digest, length, data = sha256_stream(stream, byte_limit)
    artifact = _artifact_from_data(digest, length, data, declared_mime_type)
    transient_stream = ReadOnlyMediaStream(data)
    try:
        result = perceptor.perceive(artifact, transient_stream)
    finally:
        transient_stream.close()
    return artifact, validate_percepts(artifact, result, perceptor_id)


def validate_percepts(artifact: MediaArtifact, percepts: Sequence[Percept],
                      perceptor_id: str) -> Tuple[Percept, ...]:
    """Fail closed unless adapter annotations are bounded and artifact-local."""
    _bounded_text(perceptor_id, "perceptor_id", 128)
    if not isinstance(percepts, SequenceABC) or isinstance(percepts, (str, bytes, bytearray)):
        raise ValueError("perceptor must return a finite sequence of percepts")
    values = tuple(percepts)
    if len(values) > MAX_PERCEPTS:
        raise ValueError("too many percepts for one media payload")
    for percept in values:
        if not isinstance(percept, Percept):
            raise ValueError("perceptor returned a non-Percept value")
        if percept.artifact_id != artifact.artifact_id:
            raise ValueError("percept artifact_id does not match the inspected artifact")
        if percept.modality != artifact.modality:
            raise ValueError("percept modality does not match the inspected artifact")
        if percept.perceptor_id != perceptor_id:
            raise ValueError("percept perceptor_id does not match its adapter")
        for span in percept.spans:
            if span.kind == "region":
                if (span.x is None or span.y is None or span.width is None or span.height is None
                        or artifact.width is None or artifact.height is None
                        or span.x + span.width > artifact.width
                        or span.y + span.height > artifact.height):
                    raise ValueError("image region span exceeds artifact dimensions")
            elif (span.start_ms is None or span.end_ms is None or artifact.duration_ms is None
                  or span.end_ms > artifact.duration_ms):
                raise ValueError("audio time span exceeds artifact duration")
    return values


def _artifact_from_data(digest: str, length: int, data: bytes,
                        declared_mime_type: Optional[str]) -> MediaArtifact:
    mime_type = _detect_mime_type(data)
    if declared_mime_type is not None and declared_mime_type != mime_type:
        raise ValueError("declared MIME type does not match media magic bytes")
    if mime_type == "image/png":
        width, height = _parse_png(data)
        return MediaArtifact(modality="image", mime_type=mime_type, byte_length=length,
                             sha256=digest, width=width, height=height)
    if mime_type == "image/jpeg":
        width, height = _parse_jpeg(data)
        return MediaArtifact(modality="image", mime_type=mime_type, byte_length=length,
                             sha256=digest, width=width, height=height)
    sample_rate_hz, channels, frame_count, duration_ms = _parse_wav(data)
    return MediaArtifact(modality="audio", mime_type=mime_type, byte_length=length,
                         sha256=digest, sample_rate_hz=sample_rate_hz, channels=channels,
                         frame_count=frame_count, duration_ms=duration_ms)


def media_payload(artifact: MediaArtifact, percepts: Iterable[Percept] = ()) -> Dict[str, Any]:
    """Create an exact, path-free payload for a future event-store media schema."""
    values = tuple(percepts)
    if len(values) > MAX_PERCEPTS:
        raise ValueError("too many percepts for one media payload")
    for percept in values:
        if not isinstance(percept, Percept) or percept.artifact_id != artifact.artifact_id:
            raise ValueError("each percept must belong to the artifact")
        if percept.modality != artifact.modality:
            raise ValueError("percept modality must match artifact modality")
    return {"schema": MEDIA_PAYLOAD_SCHEMA, "artifact": artifact.to_payload(),
            "percepts": [percept.to_payload() for percept in values]}


def media_from_payload(value: Dict[str, Any]) -> Tuple[MediaArtifact, Tuple[Percept, ...]]:
    """Validate and reconstruct a payload without any original bytes or path."""
    if not isinstance(value, dict) or set(value) != {"schema", "artifact", "percepts"}:
        raise ValueError("media payload uses an exact schema")
    if value["schema"] != MEDIA_PAYLOAD_SCHEMA or not isinstance(value["percepts"], list):
        raise ValueError("media payload has an unsupported schema")
    if len(value["percepts"]) > MAX_PERCEPTS:
        raise ValueError("too many percepts for one media payload")
    artifact = MediaArtifact.from_payload(value["artifact"])
    percepts = tuple(Percept.from_payload(item) for item in value["percepts"])
    media_payload(artifact, percepts)
    return artifact, percepts


def _detect_mime_type(data: bytes) -> str:
    if data.startswith(_PNG_SIGNATURE):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    raise ValueError("unsupported media magic bytes")


def _parse_png(data: bytes) -> Tuple[int, int]:
    if len(data) < 24 or data[8:12] != b"\x00\x00\x00\r" or data[12:16] != b"IHDR":
        raise ValueError("malformed PNG IHDR")
    width, height = struct.unpack(">II", data[16:24])
    if width == 0 or height == 0:
        raise ValueError("PNG dimensions must be positive")
    return width, height


def _parse_jpeg(data: bytes) -> Tuple[int, int]:
    offset = 2
    while offset < len(data):
        while offset < len(data) and data[offset] == 0xff:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xd8, 0xd9):
            continue
        if marker == 0xda:
            break
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[offset:offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
                      0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            if segment_length < 8:
                break
            height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
            if width and height:
                return width, height
        offset += segment_length
    raise ValueError("malformed JPEG without a frame header")


def _parse_wav(data: bytes) -> Tuple[int, int, int, int]:
    if len(data) < 12:
        raise ValueError("malformed WAV header")
    riff_size = struct.unpack("<I", data[4:8])[0]
    if riff_size + 8 != len(data):
        raise ValueError("WAV RIFF size does not match stream length")
    offset, fmt, data_length = 12, None, None
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_length = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(data):
            raise ValueError("truncated WAV chunk")
        if chunk_id == b"fmt ":
            if chunk_length < 16:
                raise ValueError("malformed WAV fmt chunk")
            fmt = struct.unpack("<HHIIHH", data[chunk_start:chunk_start + 16])
        elif chunk_id == b"data":
            data_length = chunk_length
        offset = chunk_end + (chunk_length % 2)
    if fmt is None or data_length is None:
        raise ValueError("WAV requires fmt and data chunks")
    _format, channels, sample_rate_hz, _byte_rate, block_align, _bits = fmt
    if channels <= 0 or sample_rate_hz <= 0 or block_align <= 0 or data_length % block_align:
        raise ValueError("WAV has invalid audio parameters")
    frame_count = data_length // block_align
    duration_ms = (frame_count * 1000) // sample_rate_hz
    return sample_rate_hz, channels, frame_count, duration_ms
