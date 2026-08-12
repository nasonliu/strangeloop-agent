"""Bounded dual-aspect audit records inspired by Yogacara terminology.

``自证分`` / ``证自证分`` are used here only as a design inspiration: a
first public assessment plus one deterministic independent check.  They are
not assertions of subjective experience, an intrinsic self, or an inner
observer.  The implementation neither persists private reasoning nor has any
authority to grant capabilities, approve memory, issue reward, or act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Iterable, Tuple
from uuid import uuid4

from .contracts import SourceKind


METHOD_VERSION = "mirror_v1"
_META_CAP = 0.85
_MAX_EVIDENCE = 32
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SENSITIVE = re.compile(r"(?:api[_-]?key|secret|password|token|authorization|bearer|sk-[A-Za-z0-9])", re.IGNORECASE)
_PUBLIC_SOURCES = frozenset((SourceKind.USER, SourceKind.TOOL, SourceKind.SYSTEM,
                             SourceKind.EXTERNAL_VERIFIER))


class SelfStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"


class SelfUncertainty(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MetaStatus(str, Enum):
    CONFIRMED = "confirmed"
    LIMITED = "limited"
    CONFLICTED = "conflicted"


class MirrorDisposition(str, Enum):
    PROVISIONAL = "provisional"
    REVIEW_REQUIRED = "review_required"
    ABSTAIN = "abstain"


class CheckCode(str, Enum):
    PROVENANCE_COMPLETE = "provenance_complete"
    SCOPE_BOUNDED = "scope_bounded"
    CONFIDENCE_CAPPED = "confidence_capped"
    COUNTEREVIDENCE_CLEAR = "counterevidence_clear"
    EVIDENCE_MISSING = "evidence_missing"
    SOURCE_CONFLICT = "source_conflict"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class EvidencePolarity(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("%s must be a safe public identifier" % name)
    return value


def _unit(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % name)
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("%s must be between 0 and 1" % name)
    return value


@dataclass(frozen=True)
class PublicEvidence:
    """Whitelist-only event metadata used by the mirror computation.

    No event payload, prompt, model output, command, URL, secret, or chain of
    thought is represented here.  ``event_kind`` is deliberately a closed
    externally observable evidence vocabulary rather than arbitrary text.
    """

    event_id: str
    event_kind: str
    source_kind: SourceKind
    polarity: EvidencePolarity
    confidence: float
    # Optional digest/identifier only; kept outside the persisted MirrorRecord.
    public_ref: str = ""

    def __post_init__(self) -> None:
        _identifier("event_id", self.event_id)
        if self.event_kind not in ("observation", "tool_result", "action_result",
                                   "correction", "percept", "loop_tick",
                                   "external_verification"):
            raise ValueError("event_kind is not public mirror evidence")
        if self.source_kind not in _PUBLIC_SOURCES:
            raise ValueError("model-originated evidence is not allowed")
        if not isinstance(self.polarity, EvidencePolarity):
            raise ValueError("polarity must be EvidencePolarity")
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        if self.public_ref:
            _identifier("public_ref", self.public_ref)
            if _SENSITIVE.search(self.public_ref):
                raise ValueError("public_ref must not contain a credential or secret marker")


@dataclass(frozen=True)
class MirrorRecord:
    """Flat, serializable public result of one atomic two-sided check."""

    episode_id: str
    target_event_id: str
    judgment_event_id: str
    evidence_event_ids: Tuple[str, ...]
    self_status: SelfStatus
    self_confidence: float
    self_uncertainty: SelfUncertainty
    meta_status: MetaStatus
    meta_confidence_cap: float
    check_codes: Tuple[CheckCode, ...]
    disposition: MirrorDisposition
    method_version: str = METHOD_VERSION
    public_summary: str = "Public evidence mirror completed."
    mirror_id: str = field(default_factory=lambda: "mirror_" + uuid4().hex)

    def __post_init__(self) -> None:
        for name in ("mirror_id", "episode_id", "target_event_id", "judgment_event_id"):
            _identifier(name, getattr(self, name))
        if not self.evidence_event_ids or len(self.evidence_event_ids) > _MAX_EVIDENCE:
            raise ValueError("evidence_event_ids must be non-empty and bounded")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("evidence_event_ids must be unique")
        for event_id in self.evidence_event_ids:
            _identifier("evidence event ID", event_id)
        if not isinstance(self.self_status, SelfStatus) or not isinstance(self.self_uncertainty, SelfUncertainty):
            raise ValueError("self fields must use their closed enums")
        if not isinstance(self.meta_status, MetaStatus) or not isinstance(self.disposition, MirrorDisposition):
            raise ValueError("meta fields must use their closed enums")
        object.__setattr__(self, "self_confidence", _unit("self_confidence", self.self_confidence))
        object.__setattr__(self, "meta_confidence_cap", _unit("meta_confidence_cap", self.meta_confidence_cap))
        if self.meta_confidence_cap > self.self_confidence:
            raise ValueError("meta confidence cap cannot exceed self confidence")
        if self.method_version != METHOD_VERSION:
            raise ValueError("unsupported mirror method version")
        if not isinstance(self.public_summary, str) or not self.public_summary.strip() or len(self.public_summary) > 512:
            raise ValueError("public_summary must be bounded public text")
        if not self.check_codes or len(self.check_codes) > 8 or len(set(self.check_codes)) != len(self.check_codes):
            raise ValueError("check_codes must be unique and bounded")
        if any(not isinstance(code, CheckCode) for code in self.check_codes):
            raise ValueError("check_codes must use the closed CheckCode enum")
        if self.self_status in (SelfStatus.INSUFFICIENT, SelfStatus.CONFLICTED):
            if self.meta_status is MetaStatus.CONFIRMED:
                raise ValueError("insufficient or conflicted self status cannot be confirmed")

    def to_payload(self) -> dict:
        """Return only the exact public event schema; no hidden deliberation."""
        return {
            "mirror_id": self.mirror_id, "episode_id": self.episode_id,
            "target_event_id": self.target_event_id, "judgment_event_id": self.judgment_event_id,
            "evidence_event_ids": list(self.evidence_event_ids),
            "self_status": self.self_status.value, "self_confidence": self.self_confidence,
            "self_uncertainty": self.self_uncertainty.value, "meta_status": self.meta_status.value,
            "meta_confidence_cap": self.meta_confidence_cap,
            "check_codes": [code.value for code in self.check_codes],
            "disposition": self.disposition.value, "method_version": self.method_version,
            "public_summary": self.public_summary,
        }


class MirrorAuditor:
    """Pure deterministic auditor; its maximum reflection depth is exactly one."""

    maximum_depth = 1

    @staticmethod
    def _rows(evidence_events: Iterable[PublicEvidence]) -> Tuple[PublicEvidence, ...]:
        rows = tuple(evidence_events)
        if not rows or len(rows) > _MAX_EVIDENCE:
            raise ValueError("evidence_events must be non-empty and bounded")
        if any(not isinstance(row, PublicEvidence) for row in rows):
            raise ValueError("evidence_events must be PublicEvidence")
        if len({row.event_id for row in rows}) != len(rows):
            raise ValueError("evidence event IDs must be unique")
        return tuple(sorted(rows, key=lambda row: row.event_id))

    @staticmethod
    def _self_assessment(rows: Tuple[PublicEvidence, ...]) -> Tuple[SelfStatus, float, SelfUncertainty]:
        support = sum(row.confidence for row in rows if row.polarity is EvidencePolarity.SUPPORTS)
        contradict = sum(row.confidence for row in rows if row.polarity is EvidencePolarity.CONTRADICTS)
        decisive = support + contradict
        if decisive == 0.0:
            return SelfStatus.INSUFFICIENT, 0.0, SelfUncertainty.HIGH
        margin = abs(support - contradict) / decisive
        confidence = round(margin * decisive / len(rows), 6)
        if contradict and support:
            return SelfStatus.CONFLICTED, confidence, SelfUncertainty.HIGH
        if confidence < 0.5:
            return SelfStatus.INSUFFICIENT, confidence, SelfUncertainty.MEDIUM
        return SelfStatus.SUPPORTED, confidence, (SelfUncertainty.LOW if confidence >= .75 else SelfUncertainty.MEDIUM)

    @staticmethod
    def _independent_recheck(rows: Tuple[PublicEvidence, ...]) -> Tuple[SelfStatus, float]:
        """Separate calculation path; it deliberately does not read self output."""
        weights = {EvidencePolarity.SUPPORTS: 0.0, EvidencePolarity.CONTRADICTS: 0.0}
        for row in rows:
            if row.polarity in weights:
                weights[row.polarity] += row.confidence
        total = weights[EvidencePolarity.SUPPORTS] + weights[EvidencePolarity.CONTRADICTS]
        if total == 0.0:
            return SelfStatus.INSUFFICIENT, 0.0
        strength = round(abs(weights[EvidencePolarity.SUPPORTS] - weights[EvidencePolarity.CONTRADICTS]) / len(rows), 6)
        if weights[EvidencePolarity.SUPPORTS] and weights[EvidencePolarity.CONTRADICTS]:
            return SelfStatus.CONFLICTED, strength
        if strength < .5:
            return SelfStatus.INSUFFICIENT, strength
        return SelfStatus.SUPPORTED, strength

    def assess(self, episode_id: str, target_event_id: str, judgment_event_id: str,
               evidence_events: Iterable[PublicEvidence]) -> MirrorRecord:
        """Atomically construct first assessment and its sole deterministic mirror."""
        _identifier("episode_id", episode_id)
        _identifier("target_event_id", target_event_id)
        _identifier("judgment_event_id", judgment_event_id)
        rows = self._rows(evidence_events)
        self_status, self_confidence, uncertainty = self._self_assessment(rows)
        audited_status, audited_confidence = self._independent_recheck(rows)
        codes = [CheckCode.PROVENANCE_COMPLETE, CheckCode.SCOPE_BOUNDED,
                 CheckCode.CONFIDENCE_CAPPED]
        if self_status is SelfStatus.SUPPORTED and audited_status is SelfStatus.SUPPORTED:
            meta_status, disposition = MetaStatus.CONFIRMED, MirrorDisposition.PROVISIONAL
            codes.append(CheckCode.COUNTEREVIDENCE_CLEAR)
        elif self_status is SelfStatus.CONFLICTED or audited_status is SelfStatus.CONFLICTED:
            meta_status, disposition = MetaStatus.CONFLICTED, MirrorDisposition.REVIEW_REQUIRED
            codes.append(CheckCode.SOURCE_CONFLICT)
        else:
            meta_status, disposition = MetaStatus.LIMITED, MirrorDisposition.ABSTAIN
            codes.append(CheckCode.EVIDENCE_MISSING)
        cap = min(_META_CAP, self_confidence, audited_confidence)
        return MirrorRecord(
            episode_id=episode_id, target_event_id=target_event_id,
            judgment_event_id=judgment_event_id,
            evidence_event_ids=tuple(row.event_id for row in rows), self_status=self_status,
            self_confidence=self_confidence, self_uncertainty=uncertainty,
            meta_status=meta_status, meta_confidence_cap=cap, check_codes=tuple(codes),
            disposition=disposition,
            public_summary="Public evidence was assessed and independently bounded once.",
        )
