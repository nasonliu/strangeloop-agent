"""Inspectable, approval-gated persistent dispositions inspired by Yogacara."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from typing import List, Optional, Sequence

from .contracts import (CognitiveEvent, EventKind, SeedDisposition, SeedStatus,
                        SourceKind, parse_aware_iso8601, utc_now_iso)
from .store import SQLiteEventStore, canonical_json


class SQLiteSeedStore:
    """Seed projection backed by a causal, user-approved event history."""

    def __init__(self, event_store: SQLiteEventStore) -> None:
        self.event_store = event_store
        self.connection = event_store.connection
        self.connection.execute("""CREATE TABLE IF NOT EXISTS seeds (
            seed_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seed_json TEXT NOT NULL,
            status TEXT NOT NULL, proposal_event_id TEXT, approval_event_id TEXT UNIQUE,
            approved_event_id TEXT
        )""")
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(seeds)")}
        for name in ("proposal_event_id", "approved_event_id"):
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
                (seed_id, session_id, seed_json, status, proposal_event_id)
                VALUES (?, ?, ?, ?, ?)""", (candidate.seed_id, session_id,
                canonical_json(self._to_dict(candidate)), candidate.status.value, proposal.event_id))
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
                approved_event_id=? WHERE seed_id=?""", (canonical_json(self._to_dict(active)),
                active.status.value, approval_event_id, approved.event_id, seed_id))
        return active

    def retire(self, seed_id: str, source_kind: SourceKind = SourceKind.USER,
               source_ref: str = "user", reason: str = "retired") -> SeedDisposition:
        if source_kind != SourceKind.USER:
            raise ValueError("only a user may retire a persistent seed")
        with self.event_store.transaction():
            row = self._row(seed_id)
            seed = self._from_dict(json.loads(row["seed_json"]))
            if seed.status == SeedStatus.RETIRED:
                return seed
            parent_id = row["approved_event_id"] or row["proposal_event_id"]
            if not parent_id:
                raise ValueError("seed projection has no causal event")
            retired = replace(seed, status=SeedStatus.RETIRED, updated_at=utc_now_iso(), version=seed.version + 1)
            self.event_store.append(CognitiveEvent(session_id=row["session_id"], kind=EventKind.SEED_RETIRED,
                source_kind=SourceKind.USER, source_ref=source_ref,
                payload={"seed_id": seed_id, "reason": reason}, parent_event_ids=(parent_id,)), commit=False)
            self.connection.execute("UPDATE seeds SET seed_json=?, status=? WHERE seed_id=?",
                                    (canonical_json(self._to_dict(retired)), retired.status.value, seed_id))
        return retired

    def list(self, session_id: str, status: Optional[SeedStatus] = None) -> List[SeedDisposition]:
        """Rebuild the public seed view from causally valid events.

        The ``seeds`` table is a write projection, never an authority for
        retrieval. This prevents a modified projection row from activating a
        disposition while the immutable event history says otherwise.
        """
        events = self.event_store.list(session_id)
        by_id = {event.event_id: event for event in events}
        proposals = {}
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

        active, used_proposals, used_approvals, approved_event_ids = {}, set(), set(), {}
        for event in events:
            if event.kind != EventKind.SEED_APPROVED or event.source_kind != SourceKind.USER:
                continue
            seed_id = event.payload.get("seed_id")
            proposal_id = event.payload.get("proposal_event_id")
            approval_id = event.payload.get("approval_event_id")
            pair = proposals.get(seed_id)
            approval = by_id.get(approval_id)
            if pair is None or approval is None:
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
            active[seed_id] = replace(seed, status=SeedStatus.ACTIVE, version=seed.version + 1)
            approved_event_ids[seed_id] = event.event_id
            used_proposals.add(proposal_id)
            used_approvals.add(approval_id)

        retired = set()
        for event in events:
            if event.kind != EventKind.SEED_RETIRED or event.source_kind != SourceKind.USER:
                continue
            seed_id = event.payload.get("seed_id")
            if seed_id in active and tuple(event.parent_event_ids) == (approved_event_ids.get(seed_id),):
                retired.add(seed_id)

        rebuilt = []
        for seed_id, (proposal, candidate) in proposals.items():
            if seed_id in retired:
                rebuilt.append(replace(active[seed_id], status=SeedStatus.RETIRED,
                                       version=active[seed_id].version + 1))
            elif seed_id in active:
                rebuilt.append(active[seed_id])
            else:
                rebuilt.append(candidate)
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
                   for seed in self.list(session_id, SeedStatus.ACTIVE)]
        matches = [(count, seed) for count, seed in matches if count]
        matches.sort(key=lambda item: (item[0], item[1].strength, item[1].confidence), reverse=True)
        return [seed for _, seed in matches]
