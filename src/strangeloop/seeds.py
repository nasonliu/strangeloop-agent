"""Inspectable, approval-gated persistent dispositions inspired by Yogacara."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from .contracts import (CognitiveEvent, EventKind, SeedDisposition, SeedStatus,
                        SourceKind, parse_aware_iso8601, utc_now_iso)
from .store import SQLiteEventStore, canonical_json


SEED_STANDING_POLICY_VERSION = "seed_standing_policy_v1"
SEED_UPDATE_PROPOSAL_VERSION = "seed_update_proposal_v1"
SEED_AUTO_ELIGIBILITY_VERSION = "seed_auto_eligibility_v1"
SEED_AUTO_APPLICATION_VERSION = "seed_auto_application_v1"
_POLICY_SCOPE = "conversation"
_POLICY_BIAS = "request-human-review"
_FLOAT_EPSILON = 1e-12


def _normalized_text(value: str) -> str:
    """Return the deterministic public-text normalization used for identity.

    This is deliberately a small Unicode/case/whitespace normalization, not a
    semantic embedding or a model judgment.
    """
    if not isinstance(value, str):
        raise ValueError("seed identity text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def canonical_seed_identity(seed: SeedDisposition) -> str:
    """Identify one disposition meaning independently of its generated ID."""
    cues = sorted(set(_normalized_text(term) for term in seed.cue_terms))
    manifest = {"cue_terms": cues, "policy_bias": _normalized_text(seed.policy_bias),
                "scope": _normalized_text(seed.scope)}
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


def seed_digest(seed: SeedDisposition) -> str:
    return hashlib.sha256(canonical_json(SQLiteSeedStore._to_dict(seed)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SeedStandingPolicy:
    """User-issued envelope for deterministic low-impact seed maintenance.

    It is operational authorization, not model memory or approval authority.
    The v1 scope and bias are fixed so a caller cannot silently widen it.
    """

    expires_at: str
    scope: str = _POLICY_SCOPE
    max_auto_activations: int = 16
    max_auto_updates: int = 64
    max_active_seeds: int = 16
    max_cue_terms: int = 2
    max_cue_length: int = 24
    max_strength: float = 0.5
    max_confidence: float = 0.5
    max_seed_ttl_seconds: int = 7 * 24 * 60 * 60
    allowed_policy_bias: Tuple[str, ...] = (_POLICY_BIAS,)
    max_strength_step: float = 0.1
    max_confidence_step: float = 0.1
    max_counterevidence: int = 3
    max_seed_versions: int = 16

    def __post_init__(self) -> None:
        parse_aware_iso8601(self.expires_at)
        if (self.scope != _POLICY_SCOPE
                or tuple(self.allowed_policy_bias) != (_POLICY_BIAS,)):
            raise ValueError("standing seed policy v1 has a fixed scope and policy bias")
        nonnegative = (self.max_auto_activations, self.max_auto_updates,
                       self.max_counterevidence)
        positive = (self.max_active_seeds, self.max_cue_terms, self.max_cue_length,
                    self.max_seed_ttl_seconds)
        if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in nonnegative)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                       for value in positive)
                or isinstance(self.max_seed_versions, bool)
                or not isinstance(self.max_seed_versions, int)
                or self.max_seed_versions < 2):
            raise ValueError("standing seed policy limits are outside their supported ranges")
        numbers = (self.max_strength, self.max_confidence,
                   self.max_strength_step, self.max_confidence_step)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not 0.0 <= float(value) <= 1.0 for value in numbers):
            raise ValueError("standing seed policy strengths must be between zero and one")

    def to_payload(self, policy_id: str, user_event_id: str, nonce: str,
                   issued_at: str) -> dict:
        return {
            "policy_id": policy_id,
            "user_observation_event_id": user_event_id,
            "nonce": nonce,
            "issued_at": issued_at,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "max_auto_activations": self.max_auto_activations,
            "max_auto_updates": self.max_auto_updates,
            "max_active_seeds": self.max_active_seeds,
            "max_cue_terms": self.max_cue_terms,
            "max_cue_length": self.max_cue_length,
            "max_strength": float(self.max_strength),
            "max_confidence": float(self.max_confidence),
            "max_seed_ttl_seconds": self.max_seed_ttl_seconds,
            "allowed_policy_bias": list(self.allowed_policy_bias),
            "max_strength_step": float(self.max_strength_step),
            "max_confidence_step": float(self.max_confidence_step),
            "max_counterevidence": self.max_counterevidence,
            "max_seed_versions": self.max_seed_versions,
            "policy_version": SEED_STANDING_POLICY_VERSION,
        }


def standing_policy_manifest(policy: SeedStandingPolicy, policy_id: str,
                             user_event_id: str, nonce: str,
                             issued_at: str) -> Tuple[dict, str]:
    """Build the exact manifest and digest an explicit approval must bind."""
    payload = policy.to_payload(policy_id, user_event_id, nonce, issued_at)
    return payload, SQLiteEventStore.standing_seed_policy_digest(payload)


class SQLiteSeedStore:
    """Seed projection backed by a causal, user-approved event history."""

    def __init__(self, event_store: SQLiteEventStore) -> None:
        self.event_store = event_store
        self.connection = event_store.connection
        self.connection.execute("""CREATE TABLE IF NOT EXISTS seeds (
            seed_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seed_json TEXT NOT NULL,
            status TEXT NOT NULL, proposal_event_id TEXT, approval_event_id TEXT UNIQUE,
            approved_event_id TEXT, last_event_id TEXT, policy_event_id TEXT,
            semantic_identity TEXT
        )""")
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(seeds)")}
        for name in ("proposal_event_id", "approved_event_id", "last_event_id",
                     "policy_event_id", "semantic_identity"):
            if name not in columns:
                self.connection.execute("ALTER TABLE seeds ADD COLUMN %s TEXT" % name)
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_seeds_session ON seeds(session_id)")

    @staticmethod
    def _to_dict(seed: SeedDisposition) -> dict:
        return {"cue_terms": list(seed.cue_terms), "policy_bias": seed.policy_bias,
                "scope": seed.scope, "provenance_event_ids": list(seed.provenance_event_ids),
                "strength": seed.strength, "confidence": seed.confidence,
                "status": seed.status.value, "seed_id": seed.seed_id,
                "created_at": seed.created_at, "updated_at": seed.updated_at,
                "expires_at": seed.expires_at, "version": seed.version,
                "counterevidence": seed.counterevidence}

    @staticmethod
    def _from_dict(value: dict) -> SeedDisposition:
        value = dict(value)
        value["cue_terms"] = tuple(value["cue_terms"])
        value["provenance_event_ids"] = tuple(value["provenance_event_ids"])
        value["status"] = SeedStatus(value["status"])
        return SeedDisposition(**value)

    def propose(self, session_id: str, seed: SeedDisposition, source_ref: str = "model") -> SeedDisposition:
        candidate = replace(seed, status=SeedStatus.CANDIDATE, updated_at=utc_now_iso())
        proposal = CognitiveEvent(session_id=session_id, kind=EventKind.SEED_PROPOSED,
            source_kind=SourceKind.MODEL, source_ref=source_ref,
            payload={"seed": self._to_dict(candidate)}, parent_event_ids=candidate.provenance_event_ids)
        with self.event_store.transaction():
            proposal = self.event_store.append(proposal, commit=False)
            self.connection.execute("""INSERT INTO seeds
                (seed_id, session_id, seed_json, status, proposal_event_id,
                 last_event_id, semantic_identity)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", (candidate.seed_id, session_id,
                canonical_json(self._to_dict(candidate)), candidate.status.value,
                proposal.event_id, proposal.event_id, canonical_seed_identity(candidate)))
        return candidate

    def _row(self, seed_id: str):
        row = self.connection.execute("SELECT * FROM seeds WHERE seed_id = ?", (seed_id,)).fetchone()
        if row is None:
            raise KeyError("unknown seed: %s" % seed_id)
        return row

    def proposal_event_id(self, seed_id: str, session_id: Optional[str] = None) -> str:
        """Return a candidate's immutable proposal event ID for explicit approval.

        Callers such as the engine must put this ID in the user approval event
        payload; it is not an authority to activate the seed by itself.
        """
        row = self._row(seed_id)
        if session_id is not None and row["session_id"] != session_id:
            raise ValueError("seed does not belong to this session")
        if not row["proposal_event_id"]:
            raise ValueError("seed has no proposal event")
        return str(row["proposal_event_id"])

    def _valid_proposal(self, row, seed: SeedDisposition) -> CognitiveEvent:
        proposal_id = row["proposal_event_id"]
        proposal = self.event_store.get(proposal_id) if proposal_id else None
        if proposal is None or proposal.session_id != row["session_id"] or proposal.kind != EventKind.SEED_PROPOSED:
            raise ValueError("seed projection is not bound to a valid proposal event")
        if proposal.payload.get("seed") != self._to_dict(seed):
            raise ValueError("proposal event does not describe this candidate seed")
        return proposal

    def approve(self, seed_id: str, approval_event_id: str) -> SeedDisposition:
        with self.event_store.transaction():
            row = self._row(seed_id)
            seed = self._from_dict(json.loads(row["seed_json"]))
            if seed.status != SeedStatus.CANDIDATE:
                raise ValueError("only candidate seeds can be approved")
            proposal = self._valid_proposal(row, seed)
            approval = self.event_store.get(approval_event_id)
            if approval is None or approval.session_id != row["session_id"]:
                raise ValueError("approval event must exist in the seed session")
            if approval.source_kind != SourceKind.USER:
                raise ValueError("only a user may approve a persistent seed")
            if (approval.payload.get("approval") != "seed" or approval.payload.get("seed_id") != seed_id
                    or approval.payload.get("proposal_event_id") != proposal.event_id):
                raise ValueError("approval event must explicitly bind this seed proposal")
            if approval.sequence <= proposal.sequence:
                raise ValueError("seed approval must occur after its proposal")
            already_used = self.connection.execute("SELECT seed_id FROM seeds WHERE approval_event_id = ?", (approval_event_id,)).fetchone()
            if already_used is not None:
                raise ValueError("approval event has already been used")
            active = replace(seed, status=SeedStatus.ACTIVE, updated_at=utc_now_iso(), version=seed.version + 1)
            approved = self.event_store.append(CognitiveEvent(
                session_id=row["session_id"], kind=EventKind.SEED_APPROVED,
                source_kind=SourceKind.USER, source_ref=approval.source_ref,
                payload={"seed_id": seed_id, "proposal_event_id": proposal.event_id,
                         "approval_event_id": approval_event_id},
                parent_event_ids=(proposal.event_id, approval_event_id)), commit=False)
            self.connection.execute("""UPDATE seeds SET seed_json=?, status=?, approval_event_id=?,
                approved_event_id=?, last_event_id=? WHERE seed_id=?""",
                (canonical_json(self._to_dict(active)), active.status.value,
                 approval_event_id, approved.event_id, approved.event_id, seed_id))
        return active

    @staticmethod
    def _event_kind(name: str):
        kind = getattr(EventKind, name, None)
        if kind is None:
            raise RuntimeError("standing seed policy event schema is not installed: %s" % name)
        return kind

    @staticmethod
    def _event_is(event: CognitiveEvent, name: str) -> bool:
        kind = getattr(EventKind, name, None)
        return kind is not None and event.kind == kind

    @staticmethod
    def _plain_user_observation(event: Optional[CognitiveEvent]) -> bool:
        return (event is not None and event.kind == EventKind.OBSERVATION
                and event.source_kind == SourceKind.USER
                and "approval" not in event.payload
                and set(event.payload).issubset({"content", "channel"})
                and isinstance(event.payload.get("content"), str))

    def issue_standing_policy(self, session_id: str, user_event_id: str,
                              policy: Optional[SeedStandingPolicy] = None,
                              nonce: Optional[str] = None, *, policy_id: Optional[str] = None,
                              issued_at: Optional[str] = None,
                              source_ref: str = "user",
                              **policy_limits) -> CognitiveEvent:
        """Append one explicit user standing-policy authorization.

        ``policy_id`` makes retries idempotent.  A different payload under the
        same ID is rejected rather than silently replacing authorization.
        """
        if policy is not None and policy_limits:
            raise ValueError("pass either a policy object or policy fields")
        policy = policy or SeedStandingPolicy(**policy_limits)
        issued = issued_at or utc_now_iso()
        if parse_aware_iso8601(policy.expires_at) <= parse_aware_iso8601(issued):
            raise ValueError("standing seed policy must expire after issuance")
        identifier = policy_id or "seedpolicy_%s" % uuid4().hex
        approval = self.event_store.get(user_event_id)
        if (approval is None or approval.session_id != session_id
                or approval.kind != EventKind.OBSERVATION
                or approval.source_kind != SourceKind.USER
                or approval.payload.get("approval") != "seed_standing_policy"):
            raise ValueError("standing seed policy requires an explicit same-session user approval")
        bound_nonce = nonce or approval.payload.get("nonce")
        if not isinstance(bound_nonce, str) or not bound_nonce:
            raise ValueError("standing seed policy requires a nonce")
        payload, expected_digest = standing_policy_manifest(
            policy, identifier, user_event_id, bound_nonce, issued)
        if (approval.payload.get("nonce") != bound_nonce
                or approval.payload.get("policy_digest") != expected_digest):
            raise ValueError("standing seed policy approval does not bind its exact manifest")
        existing = next((event for event in self.event_store.list(session_id)
                         if self._event_is(event, "SEED_STANDING_POLICY")
                         and event.payload.get("policy_id") == identifier), None)
        if existing is not None:
            if existing.payload != payload or existing.parent_event_ids != (user_event_id,):
                raise ValueError("standing seed policy idempotency key has conflicting content")
            return existing
        return self.event_store.append(CognitiveEvent(
            session_id=session_id, kind=self._event_kind("SEED_STANDING_POLICY"),
            source_kind=SourceKind.USER, source_ref=source_ref, payload=payload,
            parent_event_ids=(user_event_id,), created_at=issued,
        ))

    def revoke_standing_policy(self, session_id: str, policy_id: str,
                               user_event_id: str, reason: str = "user_requested",
                               *, revocation_id: Optional[str] = None,
                               revoked_at: Optional[str] = None,
                               source_ref: str = "user") -> CognitiveEvent:
        events = self.event_store.list(session_id)
        policy_event = next((event for event in reversed(events)
                             if self._event_is(event, "SEED_STANDING_POLICY")
                             and event.payload.get("policy_id") == policy_id), None)
        if policy_event is None:
            raise KeyError("unknown standing seed policy: %s" % policy_id)
        user_event = self.event_store.get(user_event_id)
        if (user_event is None or user_event.session_id != session_id
                or user_event.source_kind != SourceKind.USER
                or user_event.kind != EventKind.OBSERVATION
                or user_event.payload.get("approval") != "seed_standing_policy_revoke"
                or user_event.payload.get("policy_id") != policy_id
                or user_event.payload.get("policy_event_id") != policy_event.event_id):
            raise ValueError("policy revocation requires an explicit bound user approval")
        existing = next((event for event in events
                         if self._event_is(event, "SEED_STANDING_POLICY_REVOKED")
                         and event.payload.get("policy_id") == policy_id), None)
        if existing is not None:
            return existing
        payload = {
            "revocation_id": revocation_id or "seedpolicyrevoke_%s" % uuid4().hex,
            "policy_id": policy_id,
            "policy_event_id": policy_event.event_id,
            "reason": reason,
            "revoked_at": revoked_at or utc_now_iso(),
            "version": "seed_standing_policy_revocation_v1",
        }
        return self.event_store.append(CognitiveEvent(
            session_id=session_id,
            kind=self._event_kind("SEED_STANDING_POLICY_REVOKED"),
            source_kind=SourceKind.USER, source_ref=source_ref, payload=payload,
            parent_event_ids=(policy_event.event_id, user_event_id),
            created_at=payload["revoked_at"],
        ))

    def _policy_statuses(self, session_id: str, now: datetime) -> List[dict]:
        events = self.event_store.list(session_id)
        revocations = {}
        applications = []
        for event in events:
            if self._event_is(event, "SEED_STANDING_POLICY_REVOKED"):
                revocations[event.payload.get("policy_event_id")] = event
            elif self._event_is(event, "SEED_AUTO_APPLIED"):
                applications.append(event)
        statuses = []
        for event in events:
            if not self._event_is(event, "SEED_STANDING_POLICY"):
                continue
            revoked = revocations.get(event.event_id)
            expired = parse_aware_iso8601(event.payload["expires_at"]) <= now
            status = "revoked" if revoked is not None else "expired" if expired else "active"
            related = [item for item in applications
                       if item.payload.get("policy_event_id") == event.event_id]
            activations = sum(item.payload.get("operation") == "activate" for item in related)
            updates = len(related) - activations
            value = dict(event.payload)
            value.update({
                "event_id": event.event_id,
                "status": status,
                "revocation_event_id": None if revoked is None else revoked.event_id,
                "auto_activations_used": activations,
                "auto_updates_used": updates,
            })
            statuses.append(value)
        return statuses

    def standing_policy_status(self, session_id: str, now: Optional[str] = None,
                               policy_id: Optional[str] = None) -> Optional[dict]:
        current = parse_aware_iso8601(now) if now is not None else datetime.now(timezone.utc)
        statuses = self._policy_statuses(session_id, current)
        if policy_id is not None:
            statuses = [item for item in statuses if item["policy_id"] == policy_id]
        return None if not statuses else statuses[-1]

    def _active_policy(self, session_id: str, now: datetime) -> Optional[dict]:
        statuses = self._policy_statuses(session_id, now)
        # Never fall back to an older authorization after a newer policy has
        # expired or been revoked.
        return statuses[-1] if statuses and statuses[-1]["status"] == "active" else None

    def propose_seed_update(self, seed_id: str, operation: str,
                            evidence_event_id: str, *,
                            proposal_id: Optional[str] = None,
                            source_ref: str = "model") -> CognitiveEvent:
        """Propose one evidence-bound successor using optimistic concurrency."""
        if operation not in {"reinforce", "tighten", "retire"}:
            raise ValueError("seed update operation must be reinforce, tighten, or retire")
        row = self._row(seed_id)
        session_id = str(row["session_id"])
        with self.event_store.transaction():
            evidence = self.event_store.get(evidence_event_id)
            if evidence is None or evidence.session_id != session_id:
                raise ValueError("seed update requires same-session evidence")
            evidence_time = parse_aware_iso8601(evidence.created_at)
            states, metadata = self._rebuild(
                session_id, evidence_time, filter_inactive_policy=False)
            current = states.get(seed_id)
            if current is None or current.status != SeedStatus.ACTIVE:
                raise ValueError("only an active ledger seed can be updated")
            expected_version = current.version
            base_event_id = metadata[seed_id]["base_event_id"]
            base_event = self.event_store.get(base_event_id)
            if (base_event is None or evidence is None
                    or evidence.session_id != session_id
                    or evidence.sequence is None or base_event.sequence is None
                    or evidence.sequence <= base_event.sequence):
                raise ValueError("seed update requires later same-session evidence")
            policy = self._active_policy(session_id, evidence_time)
            if policy is None:
                raise ValueError("seed update requires an active standing policy")
            records = {event.event_id: event
                       for event in self.event_store.list(session_id)}
            evidence_reason = self._update_evidence_reason(
                operation, evidence, current, base_event, records)
            if evidence_reason is not None:
                raise ValueError(evidence_reason)
            expected_provenance = current.provenance_event_ids + (evidence_event_id,)
            expiry_cap = min(parse_aware_iso8601(policy["expires_at"]),
                             evidence_time + timedelta(seconds=policy["max_seed_ttl_seconds"]))
            current_expiry = (None if current.expires_at is None
                              else parse_aware_iso8601(current.expires_at))
            if current_expiry is not None and current_expiry <= evidence_time:
                raise ValueError("an expired seed cannot be automatically updated")
            # Standing updates never extend a seed's lifetime.  A legacy
            # unbounded seed may only become bounded under the policy window.
            successor_expiry = (expiry_cap if current_expiry is None
                                else min(current_expiry, expiry_cap))
            if successor_expiry <= evidence_time:
                raise ValueError("seed update requires a future bounded expiry")
            if operation == "reinforce":
                strength = min(policy["max_strength"], round(
                    current.strength + policy["max_strength_step"], 12))
                confidence = min(policy["max_confidence"], round(
                    current.confidence + policy["max_confidence_step"], 12))
                if strength == current.strength and confidence == current.confidence:
                    raise ValueError("reinforcement requires a bounded strength or confidence increase")
                normalized = replace(
                    current,
                    strength=strength, confidence=confidence,
                    expires_at=successor_expiry.isoformat(), updated_at=evidence.created_at,
                    version=expected_version + 1, provenance_event_ids=expected_provenance)
            else:
                counterevidence = current.counterevidence + 1
                strength = max(0.0, round(
                    current.strength - policy["max_strength_step"], 12))
                confidence = max(0.0, round(
                    current.confidence - policy["max_confidence_step"], 12))
                must_retire = (operation == "retire"
                               or counterevidence >= policy["max_counterevidence"]
                               or strength <= 0.0 or confidence <= 0.0)
                normalized = replace(
                    current, strength=strength, confidence=confidence,
                    counterevidence=counterevidence,
                    status=SeedStatus.RETIRED if must_retire else SeedStatus.ACTIVE,
                    expires_at=successor_expiry.isoformat(),
                    updated_at=evidence.created_at, version=expected_version + 1,
                    provenance_event_ids=expected_provenance)
            if canonical_seed_identity(normalized) != canonical_seed_identity(current):
                raise ValueError("seed update cannot change semantic identity")
            identifier = proposal_id or "seedupdate_%s" % uuid4().hex
            delta_manifest = {
                "base_event_id": base_event_id, "base_version": expected_version,
                "evidence_event_id": evidence_event_id, "operation": operation,
                "proposed_seed": self._to_dict(normalized), "seed_id": seed_id,
            }
            payload = {
                "proposal_id": identifier, "seed_id": seed_id,
                "base_event_id": base_event_id, "base_version": expected_version,
                "operation": operation, "proposed_seed": self._to_dict(normalized),
                "delta_digest": hashlib.sha256(
                    canonical_json(delta_manifest).encode("utf-8")).hexdigest(),
                "version": SEED_UPDATE_PROPOSAL_VERSION,
            }
            existing = next((event for event in self.event_store.list(session_id)
                             if self._event_is(event, "SEED_UPDATE_PROPOSED")
                             and event.payload.get("proposal_id") == identifier), None)
            if existing is not None:
                if existing.payload != payload:
                    raise ValueError("seed update idempotency key has conflicting content")
                return existing
            return self.event_store.append(CognitiveEvent(
                session_id=session_id, kind=self._event_kind("SEED_UPDATE_PROPOSED"),
                source_kind=SourceKind.MODEL, source_ref=source_ref, payload=payload,
                parent_event_ids=(base_event_id, evidence_event_id),
                created_at=evidence.created_at,
            ), commit=False)

    # Compatibility with the shorter name used in early design notes.
    propose_update = propose_seed_update

    @staticmethod
    def _activation_provenance_is_server_candidate(
            seed: SeedDisposition, records: Dict[str, CognitiveEvent]) -> bool:
        if len(seed.provenance_event_ids) != 2:
            return False
        observation = records.get(seed.provenance_event_ids[0])
        decision = records.get(seed.provenance_event_ids[1])
        if not SQLiteSeedStore._plain_user_observation(observation):
            return False
        if (decision is None or decision.kind != EventKind.DECISION
                or observation.event_id not in decision.payload.get("observation_event_ids", ())
                or observation.event_id not in decision.parent_event_ids):
            return False
        return True

    @classmethod
    def _underlying_user_evidence_id(
            cls, evidence: Optional[CognitiveEvent],
            records: Dict[str, CognitiveEvent]) -> Optional[str]:
        """Return the plain USER observation consumed by one seed update."""
        if cls._plain_user_observation(evidence):
            return evidence.event_id
        if (evidence is None or evidence.kind != EventKind.CORRECTION
                or evidence.source_kind != SourceKind.USER):
            return None
        counterevidence = records.get(
            evidence.payload.get("counterevidence_event_id"))
        return (counterevidence.event_id
                if cls._plain_user_observation(counterevidence) else None)

    def _user_evidence_was_consumed(
            self, observation_event_id: str,
            records: Dict[str, CognitiveEvent], *,
            excluding_proposal_id: Optional[str] = None,
            before_sequence: Optional[int] = None) -> bool:
        """Enforce one update use for the underlying USER observation."""
        for proposal in records.values():
            if (not self._event_is(proposal, "SEED_UPDATE_PROPOSED")
                    or proposal.event_id == excluding_proposal_id
                    or len(proposal.parent_event_ids) != 2):
                continue
            if (before_sequence is not None and proposal.sequence is not None
                    and proposal.sequence >= before_sequence):
                continue
            evidence = records.get(proposal.parent_event_ids[1])
            if self._underlying_user_evidence_id(
                    evidence, records) == observation_event_id:
                return True
        return False

    def _update_evidence_reason(
            self, operation: str, evidence: CognitiveEvent,
            current: SeedDisposition, base_event: CognitiveEvent,
            records: Dict[str, CognitiveEvent], *,
            excluding_proposal_id: Optional[str] = None,
            before_sequence: Optional[int] = None) -> Optional[str]:
        """Validate deterministic, target-bound evidence for a seed delta."""
        if (evidence.sequence is None or base_event.sequence is None
                or evidence.sequence <= base_event.sequence):
            return "seed update requires evidence later than its current base"

        if operation == "reinforce":
            if not self._plain_user_observation(evidence):
                return "seed reinforcement requires a plain USER observation"
            if not SQLiteEventStore.seed_evidence_literal_cue_match(
                    evidence.payload["content"], list(current.cue_terms)):
                return "seed reinforcement evidence must literally match a current cue"
            user_evidence_id = evidence.event_id
        else:
            if (evidence.kind != EventKind.CORRECTION
                    or evidence.source_kind != SourceKind.USER):
                return "seed tightening or retirement requires a USER correction"
            target_id = evidence.payload.get("target_event_id")
            counterevidence_id = evidence.payload.get("counterevidence_event_id")
            target = records.get(target_id)
            counterevidence = records.get(counterevidence_id)
            allowed_targets = {base_event.event_id} | set(
                current.provenance_event_ids)
            if (target is None or target.event_id not in allowed_targets):
                return "seed correction target must bind the current base or prior provenance"
            if (not self._plain_user_observation(counterevidence)
                    or tuple(evidence.parent_event_ids)
                    != (target.event_id, counterevidence.event_id)):
                return "seed correction must bind exact target and plain USER counterevidence parents"
            if (counterevidence.sequence is None
                    or counterevidence.sequence <= base_event.sequence):
                return "seed correction counterevidence must be later than the current base"
            user_evidence_id = counterevidence.event_id

        if self._user_evidence_was_consumed(
                user_evidence_id, records,
                excluding_proposal_id=excluding_proposal_id,
                before_sequence=before_sequence):
            return "seed update USER evidence has already been consumed"
        return None

    def _eligibility_reason(self, session_id: str, proposal: CognitiveEvent,
                            policy: dict, now: datetime,
                            states: Dict[str, SeedDisposition],
                            metadata: Dict[str, dict]) -> str:
        records = {event.event_id: event for event in self.event_store.list(session_id)}
        policy_event = records[policy["event_id"]]
        payload = proposal.payload
        operation = "activate" if proposal.kind == EventKind.SEED_PROPOSED else payload["operation"]
        proposed = self._from_dict(payload["seed"] if operation == "activate"
                                   else payload["proposed_seed"])
        if not (parse_aware_iso8601(policy["issued_at"]) <= now
                < parse_aware_iso8601(policy["expires_at"])):
            return "policy_inactive"
        if (proposal.sequence is None or policy_event.sequence is None
                or proposal.sequence <= policy_event.sequence):
            return "proposal_predates_policy"
        if policy["auto_activations_used"] >= policy["max_auto_activations"] and operation == "activate":
            return "activation_budget_exhausted"
        if policy["auto_updates_used"] >= policy["max_auto_updates"] and operation != "activate":
            return "update_budget_exhausted"
        active_count = sum(seed.status == SeedStatus.ACTIVE for seed in states.values())
        if operation == "activate" and active_count >= policy["max_active_seeds"]:
            return "active_seed_cap_reached"
        if (proposed.scope != policy["scope"]
                or proposed.policy_bias not in policy["allowed_policy_bias"]
                or len(proposed.cue_terms) > policy["max_cue_terms"]
                or any(len(term) > policy["max_cue_length"] for term in proposed.cue_terms)
                or proposed.strength > policy["max_strength"]
                or proposed.confidence > policy["max_confidence"]
                or proposed.version >= policy["max_seed_versions"]):
            return "seed_outside_policy_envelope"
        proposed_expiry = (None if proposed.expires_at is None
                           else parse_aware_iso8601(proposed.expires_at))
        if proposed_expiry is None and operation != "activate":
            return "seed_expiry_outside_policy"
        if (operation != "activate" and proposed_expiry is not None
                and (proposed_expiry <= now
                     or proposed_expiry > parse_aware_iso8601(policy["expires_at"])
                     or proposed_expiry > now + timedelta(
                         seconds=policy["max_seed_ttl_seconds"]))):
            return "seed_expiry_outside_policy"
        if operation == "activate":
            if not self._activation_provenance_is_server_candidate(proposed, records):
                return "untrusted_activation_provenance"
            identity = canonical_seed_identity(proposed)
            for event in self.event_store.list(session_id):
                if event.kind != EventKind.SEED_PROPOSED or event.event_id == proposal.event_id:
                    continue
                try:
                    other = self._from_dict(event.payload["seed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if canonical_seed_identity(other) == identity and event.sequence < proposal.sequence:
                    return "semantic_identity_already_seen"
            return "eligible"
        seed_id = proposed.seed_id
        current = states.get(seed_id)
        base_event_id = payload["base_event_id"]
        if (current is None or current.status != SeedStatus.ACTIVE
                or current.version != payload["base_version"]
                or metadata.get(seed_id, {}).get("base_event_id") != base_event_id):
            return "version_conflict"
        if proposed.provenance_event_ids[:-1] != current.provenance_event_ids:
            return "provenance_rewrite"
        evidence = records.get(proposed.provenance_event_ids[-1])
        base_event = records.get(base_event_id)
        if (evidence is None or base_event is None
                or tuple(proposal.parent_event_ids)
                != (base_event_id, evidence.event_id)
                or self._update_evidence_reason(
                    operation, evidence, current, base_event, records,
                    excluding_proposal_id=proposal.event_id,
                    before_sequence=proposal.sequence) is not None):
            return "ineligible_update_evidence"
        current_expiry = (None if current.expires_at is None
                          else parse_aware_iso8601(current.expires_at))
        if (proposed_expiry is None
                or (current_expiry is not None and proposed_expiry > current_expiry)):
            return "seed_expiry_extension_forbidden"
        if operation == "reinforce":
            if (proposed.status != SeedStatus.ACTIVE
                    or proposed.strength < current.strength
                    or proposed.confidence < current.confidence
                    or proposed.strength - current.strength
                    > policy["max_strength_step"] + _FLOAT_EPSILON
                    or proposed.confidence - current.confidence
                    > policy["max_confidence_step"] + _FLOAT_EPSILON
                    or proposed.counterevidence < current.counterevidence):
                return "reinforcement_not_monotonic"
            if (proposed.strength == current.strength
                    and proposed.confidence == current.confidence):
                return "reinforcement_no_change"
        elif operation == "tighten":
            if (proposed.status != SeedStatus.ACTIVE
                    or proposed.strength > current.strength
                    or proposed.confidence > current.confidence
                    or proposed.counterevidence <= current.counterevidence):
                return "tightening_not_monotonic"
            if (proposed.counterevidence >= policy["max_counterevidence"]
                    or proposed.strength <= 0.0 or proposed.confidence <= 0.0):
                return "retirement_required"
        else:
            if (proposed.status != SeedStatus.RETIRED
                    or proposed.strength > current.strength
                    or proposed.confidence > current.confidence
                    or proposed.counterevidence <= current.counterevidence):
                return "retirement_not_monotonic"
        return "eligible"

    def _apply_proposal(self, proposal_event_id: str, now: Optional[str] = None,
                        *, request_id: Optional[str] = None) -> Optional[SeedDisposition]:
        """Evaluate and atomically apply one candidate or update proposal."""
        evaluated_at = now or utc_now_iso()
        current_time = parse_aware_iso8601(evaluated_at)
        proposal = self.event_store.get(proposal_event_id)
        if (proposal is None or proposal.kind not in
                {EventKind.SEED_PROPOSED, self._event_kind("SEED_UPDATE_PROPOSED")}):
            raise ValueError("standing policy requires a seed proposal")
        session_id = proposal.session_id
        with self.event_store.transaction():
            policy = self._active_policy(session_id, current_time)
            if policy is None:
                return None
            events = self.event_store.list(session_id)
            existing_eligibility = next((event for event in events
                if self._event_is(event, "SEED_AUTO_ELIGIBILITY")
                and event.payload.get("proposal_event_id") == proposal_event_id
                and event.payload.get("policy_event_id") == policy["event_id"]), None)
            if existing_eligibility is not None:
                applied = next((event for event in events
                    if self._event_is(event, "SEED_AUTO_APPLIED")
                    and event.payload.get("eligibility_event_id") == existing_eligibility.event_id), None)
                if applied is None:
                    return None
                states, _ = self._rebuild(session_id, current_time, filter_inactive_policy=False)
                return states.get(applied.payload["seed_id"])
            states, metadata = self._rebuild(
                session_id, current_time, filter_inactive_policy=False)
            operation = "activate" if proposal.kind == EventKind.SEED_PROPOSED else proposal.payload["operation"]
            proposed = self._from_dict(proposal.payload["seed"] if operation == "activate"
                                       else proposal.payload["proposed_seed"])
            reason = self._eligibility_reason(
                session_id, proposal, policy, current_time, states, metadata)
            decision = "eligible" if reason == "eligible" else "rejected"
            base_event_id = None if operation == "activate" else proposal.payload["base_event_id"]
            base_version = proposed.version if operation == "activate" else proposal.payload["base_version"]
            active_count = sum(seed.status == SeedStatus.ACTIVE for seed in states.values())
            suffix = request_id or uuid4().hex
            eligibility = self.event_store.append(CognitiveEvent(
                session_id=session_id, kind=self._event_kind("SEED_AUTO_ELIGIBILITY"),
                source_kind=SourceKind.POLICY, source_ref="SeedStandingPolicy",
                payload={
                    "eligibility_id": "seedeligibility_%s" % suffix,
                    "policy_id": policy["policy_id"],
                    "policy_event_id": policy["event_id"],
                    "proposal_event_id": proposal.event_id,
                    "seed_id": proposed.seed_id, "base_event_id": base_event_id,
                    "base_version": base_version, "operation": operation,
                    "decision": decision, "reason": reason,
                    "auto_activations_used": policy["auto_activations_used"],
                    "auto_updates_used": policy["auto_updates_used"],
                    "active_seeds": active_count, "evaluated_at": evaluated_at,
                    "version": SEED_AUTO_ELIGIBILITY_VERSION,
                }, parent_event_ids=(proposal.event_id, policy["event_id"]),
                created_at=evaluated_at,
            ), commit=False)
            if decision != "eligible":
                return None
            if operation == "activate":
                original_candidate = self._from_dict(proposal.payload["seed"])
                next_seed = self._from_dict(
                    SQLiteEventStore.standing_seed_activation_snapshot(
                        self._to_dict(original_candidate), policy, evaluated_at,
                        original_candidate.version + 1))
                prior_digest = seed_digest(original_candidate)
            else:
                next_seed = proposed
                current = states[proposed.seed_id]
                prior_digest = seed_digest(current)
            application = self.event_store.append(CognitiveEvent(
                session_id=session_id, kind=self._event_kind("SEED_AUTO_APPLIED"),
                source_kind=SourceKind.POLICY, source_ref="SeedStandingPolicy",
                payload={
                    "application_id": "seedapplication_%s" % suffix,
                    "eligibility_event_id": eligibility.event_id,
                    "policy_id": policy["policy_id"],
                    "policy_event_id": policy["event_id"],
                    "proposal_event_id": proposal.event_id,
                    "seed_id": next_seed.seed_id, "base_event_id": base_event_id,
                    "base_version": base_version, "new_version": next_seed.version,
                    "operation": operation, "prior_seed_digest": prior_digest,
                    "new_seed_digest": seed_digest(next_seed),
                    "applied_at": evaluated_at, "version": SEED_AUTO_APPLICATION_VERSION,
                }, parent_event_ids=(eligibility.event_id, proposal.event_id,
                                     policy["event_id"]),
                created_at=evaluated_at,
            ), commit=False)
            cursor = self.connection.execute("""UPDATE seeds SET seed_json=?, status=?,
                last_event_id=?, policy_event_id=?, semantic_identity=? WHERE seed_id=?""",
                (canonical_json(self._to_dict(next_seed)), next_seed.status.value,
                 application.event_id, policy["event_id"],
                 canonical_seed_identity(next_seed), next_seed.seed_id))
            if cursor.rowcount != 1:
                raise ValueError("seed projection row is unavailable")
            return next_seed

    def apply_standing_policy(self, session_id: str, now: Optional[str] = None) -> List[SeedDisposition]:
        """Apply all currently pending eligible proposals in ledger order."""
        current = parse_aware_iso8601(now) if now is not None else datetime.now(timezone.utc)
        policy = self._active_policy(session_id, current)
        if policy is None:
            return []
        events = self.event_store.list(session_id)
        decided = {(event.payload.get("proposal_event_id"), event.payload.get("policy_event_id"))
                   for event in events if self._event_is(event, "SEED_AUTO_ELIGIBILITY")}
        proposals = [event for event in events
                     if event.kind in (EventKind.SEED_PROPOSED,
                                       self._event_kind("SEED_UPDATE_PROPOSED"))
                     and (event.event_id, policy["event_id"]) not in decided
                     and event.sequence is not None]
        applied = []
        for proposal in proposals:
            value = self._apply_proposal(proposal.event_id, now=now)
            if value is not None:
                applied.append(value)
        return applied

    def auto_apply_candidate(self, seed_id: str, now: Optional[str] = None,
                             *, request_id: Optional[str] = None) -> Optional[SeedDisposition]:
        row = self._row(seed_id)
        return self._apply_proposal(
            str(row["proposal_event_id"]), now=now, request_id=request_id)

    def auto_apply_update(self, proposal_event_id: str, now: Optional[str] = None,
                          *, request_id: Optional[str] = None) -> Optional[SeedDisposition]:
        proposal = self.event_store.get(proposal_event_id)
        if proposal is None or not self._event_is(proposal, "SEED_UPDATE_PROPOSED"):
            raise KeyError("unknown seed update proposal")
        return self._apply_proposal(
            proposal_event_id, now=now, request_id=request_id)

    def retire(self, seed_id: str, source_kind: SourceKind = SourceKind.USER,
               source_ref: str = "user", reason: str = "retired") -> SeedDisposition:
        if source_kind != SourceKind.USER:
            raise ValueError("only a user may retire a persistent seed")
        with self.event_store.transaction():
            row = self._row(seed_id)
            rebuilt, metadata = self._rebuild(row["session_id"], datetime.now(timezone.utc),
                                              filter_inactive_policy=False)
            seed = rebuilt.get(seed_id)
            if seed is None:
                raise ValueError("seed has no valid ledger state")
            if seed.status == SeedStatus.RETIRED:
                return seed
            if seed.status != SeedStatus.ACTIVE:
                raise ValueError("only an active seed can be retired")
            parent_id = metadata[seed_id]["base_event_id"]
            if not parent_id:
                raise ValueError("seed projection has no causal event")
            retired = replace(seed, status=SeedStatus.RETIRED, updated_at=utc_now_iso(), version=seed.version + 1)
            self.event_store.append(CognitiveEvent(session_id=row["session_id"], kind=EventKind.SEED_RETIRED,
                source_kind=SourceKind.USER, source_ref=source_ref,
                payload={"seed_id": seed_id, "reason": reason}, parent_event_ids=(parent_id,)), commit=False)
            self.connection.execute("UPDATE seeds SET seed_json=?, status=? WHERE seed_id=?",
                                    (canonical_json(self._to_dict(retired)), retired.status.value, seed_id))
        return retired

    def _rebuild(self, session_id: str, now: datetime,
                 filter_inactive_policy: bool = True
                 ) -> Tuple[Dict[str, SeedDisposition], Dict[str, dict]]:
        """Replay legacy and standing-policy seed state from the ledger."""
        events = self.event_store.list(session_id)
        by_id = {event.event_id: event for event in events}
        proposals: Dict[str, Tuple[CognitiveEvent, SeedDisposition]] = {}
        for event in events:
            if event.kind != EventKind.SEED_PROPOSED or event.source_kind != SourceKind.MODEL:
                continue
            try:
                seed = self._from_dict(event.payload["seed"])
            except (KeyError, TypeError, ValueError):
                continue
            if seed.status != SeedStatus.CANDIDATE:
                continue
            if tuple(event.parent_event_ids) != seed.provenance_event_ids:
                continue
            if not all(parent in by_id and by_id[parent].sequence < event.sequence
                       for parent in event.parent_event_ids):
                continue
            if seed.seed_id not in proposals:
                proposals[seed.seed_id] = (event, seed)
        states = {seed_id: pair[1] for seed_id, pair in proposals.items()}
        metadata = {seed_id: {"base_event_id": pair[0].event_id,
                              "policy_event_id": None,
                              "proposal_event_id": pair[0].event_id}
                    for seed_id, pair in proposals.items()}
        used_proposals, used_approvals = set(), set()
        for event in events:
            if event.kind == EventKind.SEED_APPROVED and event.source_kind == SourceKind.USER:
                seed_id = event.payload.get("seed_id")
                proposal_id = event.payload.get("proposal_event_id")
                approval_id = event.payload.get("approval_event_id")
                pair = proposals.get(seed_id)
                approval = by_id.get(approval_id)
                if pair is None or approval is None or states.get(seed_id).status != SeedStatus.CANDIDATE:
                    continue
                proposal, seed = pair
                if (proposal.event_id != proposal_id or approval.source_kind != SourceKind.USER
                        or approval.payload.get("approval") != "seed"
                        or approval.payload.get("seed_id") != seed_id
                        or approval.payload.get("proposal_event_id") != proposal_id
                        or approval.sequence <= proposal.sequence
                        or tuple(event.parent_event_ids) != (proposal_id, approval_id)
                        or proposal_id in used_proposals or approval_id in used_approvals):
                    continue
                states[seed_id] = replace(seed, status=SeedStatus.ACTIVE,
                                          updated_at=event.created_at,
                                          version=seed.version + 1)
                metadata[seed_id]["base_event_id"] = event.event_id
                used_proposals.add(proposal_id)
                used_approvals.add(approval_id)
                continue
            if self._event_is(event, "SEED_AUTO_APPLIED"):
                seed_id = event.payload.get("seed_id")
                proposal = by_id.get(event.payload.get("proposal_event_id"))
                eligibility = by_id.get(event.payload.get("eligibility_event_id"))
                policy = by_id.get(event.payload.get("policy_event_id"))
                if (seed_id not in states or proposal is None or eligibility is None or policy is None
                        or tuple(event.parent_event_ids) != (
                            eligibility.event_id, proposal.event_id, policy.event_id)
                        or eligibility.payload.get("decision") != "eligible"):
                    continue
                operation = event.payload.get("operation")
                current = states[seed_id]
                if operation == "activate":
                    if (proposal.kind != EventKind.SEED_PROPOSED
                            or current.status != SeedStatus.CANDIDATE
                            or metadata[seed_id]["base_event_id"] != proposal.event_id):
                        continue
                    candidate = self._from_dict(proposal.payload["seed"])
                    next_seed = self._from_dict(
                        SQLiteEventStore.standing_seed_activation_snapshot(
                            self._to_dict(candidate), policy.payload,
                            event.payload["applied_at"],
                            event.payload["new_version"]))
                    if event.payload.get("prior_seed_digest") != seed_digest(candidate):
                        continue
                else:
                    if (proposal.kind != self._event_kind("SEED_UPDATE_PROPOSED")
                            or current.status != SeedStatus.ACTIVE
                            or event.payload.get("base_event_id") != metadata[seed_id]["base_event_id"]
                            or event.payload.get("base_version") != current.version
                            or len(proposal.parent_event_ids) != 2
                            or tuple(proposal.parent_event_ids)[0]
                            != metadata[seed_id]["base_event_id"]):
                        continue
                    next_seed = self._from_dict(proposal.payload["proposed_seed"])
                    base_event = by_id.get(metadata[seed_id]["base_event_id"])
                    evidence = by_id.get(proposal.parent_event_ids[1])
                    if (base_event is None or evidence is None
                            or next_seed.provenance_event_ids
                            != current.provenance_event_ids + (evidence.event_id,)
                            or self._update_evidence_reason(
                                operation, evidence, current, base_event, by_id,
                                excluding_proposal_id=proposal.event_id,
                                before_sequence=proposal.sequence) is not None
                            or event.payload.get("prior_seed_digest")
                            != seed_digest(current)):
                        continue
                if (next_seed.version != event.payload.get("new_version")
                        or event.payload.get("new_seed_digest") != seed_digest(next_seed)
                        or canonical_seed_identity(next_seed) != canonical_seed_identity(current)):
                    continue
                states[seed_id] = next_seed
                metadata[seed_id]["base_event_id"] = event.event_id
                metadata[seed_id]["policy_event_id"] = policy.event_id
                continue
            if event.kind == EventKind.SEED_RETIRED and event.source_kind == SourceKind.USER:
                seed_id = event.payload.get("seed_id")
                current = states.get(seed_id)
                if (current is None or current.status == SeedStatus.RETIRED
                        or tuple(event.parent_event_ids) != (metadata[seed_id]["base_event_id"],)):
                    continue
                states[seed_id] = replace(current, status=SeedStatus.RETIRED,
                                          updated_at=event.created_at,
                                          version=current.version + 1)
                metadata[seed_id]["base_event_id"] = event.event_id
                metadata[seed_id]["policy_event_id"] = None
        # Revocation or policy expiry stops future automatic applications.  It
        # does not rewrite history or retroactively delete an activated seed;
        # the seed's own bounded expiry remains authoritative for retrieval.
        del filter_inactive_policy
        return states, metadata

    def list(self, session_id: str, status: Optional[SeedStatus] = None,
             now: Optional[str] = None) -> List[SeedDisposition]:
        """Rebuild the public seed view from causally valid ledger events."""
        current = parse_aware_iso8601(now) if now is not None else datetime.now(timezone.utc)
        states, _ = self._rebuild(session_id, current)
        rebuilt = list(states.values())
        if status is not None:
            rebuilt = [seed for seed in rebuilt if seed.status == status]
        return sorted(rebuilt, key=lambda seed: seed.seed_id)

    @staticmethod
    def _matches(seed: SeedDisposition, cues: Sequence[str], scope: str, now: datetime) -> int:
        if seed.status != SeedStatus.ACTIVE or (seed.expires_at is not None and parse_aware_iso8601(seed.expires_at) <= now):
            return 0
        if seed.scope not in ("*", scope):
            return 0
        haystack = " ".join(cue.lower() for cue in cues)
        return sum(1 for term in seed.cue_terms if term.lower() in haystack)

    def retrieve(self, session_id: str, cues: Sequence[str], scope: str,
                 now: Optional[str] = None) -> List[SeedDisposition]:
        current = parse_aware_iso8601(now) if now is not None else datetime.now(timezone.utc)
        matches = [(self._matches(seed, cues, scope, current), seed)
                   for seed in self.list(session_id, SeedStatus.ACTIVE, now=now)]
        matches = [(count, seed) for count, seed in matches if count]
        matches.sort(key=lambda item: (item[0], item[1].strength, item[1].confidence), reverse=True)
        return [seed for _, seed in matches]
