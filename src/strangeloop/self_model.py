"""Event-sourced, revocable self-model claims.

This is an auditable software self-description, not a claim of an intrinsic
self or subjective experience.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .contracts import (CognitiveEvent, EventKind, SelfClaimKind, SelfModelClaim,
                        SourceKind, parse_aware_iso8601)
from .store import SQLiteEventStore


class EventSourcedSelfModel:
    """A revocable, event-sourced software self-description.

    Claims remain candidates until the user explicitly approves them.
    """

    def __init__(self, event_store: SQLiteEventStore) -> None:
        self.event_store = event_store

    @staticmethod
    def _claim_payload(claim: SelfModelClaim) -> dict:
        return {"claim_id": claim.claim_id, "kind": claim.kind.value,
                "statement": claim.statement, "evidence_event_ids": list(claim.evidence_event_ids),
                "confidence": claim.confidence, "created_at": claim.created_at,
                "expires_at": claim.expires_at}

    @staticmethod
    def _claim_from_payload(value: dict) -> SelfModelClaim:
        return SelfModelClaim(claim_id=value["claim_id"], kind=SelfClaimKind(value["kind"]),
                              statement=value["statement"], evidence_event_ids=tuple(value["evidence_event_ids"]),
                              confidence=float(value["confidence"]), created_at=value["created_at"],
                              expires_at=value.get("expires_at"))

    def _validate_evidence(self, session_id: str, claim: SelfModelClaim) -> None:
        if not claim.evidence_event_ids:
            raise ValueError("a self-model claim requires evidence events")
        for evidence_id in claim.evidence_event_ids:
            event = self.event_store.get(evidence_id)
            if event is None or event.session_id != session_id:
                raise ValueError("claim evidence must exist in the claim session")

    def propose_claim(self, session_id: str, claim: SelfModelClaim,
                      source_kind: SourceKind = SourceKind.MODEL,
                      source_ref: str = "model") -> SelfModelClaim:
        """Record a candidate claim without making it part of the current model."""
        self._validate_evidence(session_id, claim)
        for event in self.event_store.list(session_id):
            if event.kind != EventKind.SELF_CLAIM_PROPOSED:
                continue
            try:
                existing = self._claim_from_payload(event.payload["claim"])
            except (KeyError, TypeError, ValueError):
                continue
            if existing.claim_id == claim.claim_id:
                raise ValueError("claim ID has already been proposed in this session")
        self.event_store.append(CognitiveEvent(
            session_id=session_id, kind=EventKind.SELF_CLAIM_PROPOSED,
            source_kind=source_kind, source_ref=source_ref,
            payload={"claim": self._claim_payload(claim), "state": "candidate"},
            parent_event_ids=claim.evidence_event_ids,
        ))
        return claim

    def approve_claim(self, session_id: str, claim_id: str,
                      approval_event_id: str) -> SelfModelClaim:
        """Activate a proposed claim after an explicit user approval event."""
        proposed = None
        proposal_event = None
        for event in self.event_store.list(session_id):
            if event.kind == EventKind.SELF_CLAIM_PROPOSED:
                claim = self._claim_from_payload(event.payload["claim"])
                if claim.claim_id == claim_id:
                    proposed = claim
                    proposal_event = event
        if proposed is None:
            raise KeyError("unknown proposed claim: %s" % claim_id)
        approval = self.event_store.get(approval_event_id)
        if approval is None or approval.session_id != session_id:
            raise ValueError("approval event must exist in the claim session")
        if approval.source_kind != SourceKind.USER:
            raise ValueError("only a user may approve a persistent claim")
        if (approval.payload.get("approval") != "self_claim" or approval.payload.get("claim_id") != claim_id
                or approval.payload.get("proposal_event_id") != proposal_event.event_id):
            raise ValueError("approval event must explicitly bind this self-model proposal")
        if approval.sequence <= proposal_event.sequence:
            raise ValueError("claim approval must occur after its proposal")
        for event in self.event_store.list(session_id):
            if event.kind != EventKind.SELF_CLAIM_APPROVED:
                continue
            if (event.payload.get("proposal_event_id") == proposal_event.event_id
                    or event.payload.get("approval_event_id") == approval_event_id):
                raise ValueError("claim proposal or approval event has already been used")
        self.event_store.append(CognitiveEvent(
            session_id=session_id, kind=EventKind.SELF_CLAIM_APPROVED,
            source_kind=SourceKind.USER, source_ref=approval.source_ref,
            payload={"claim": self._claim_payload(proposed),
                     "proposal_event_id": proposal_event.event_id,
                     "approval_event_id": approval_event_id},
            parent_event_ids=(proposal_event.event_id, approval_event_id),
        ))
        return proposed

    def revoke(self, session_id: str, claim_id: str, source_kind: SourceKind = SourceKind.USER,
               source_ref: str = "user", reason: str = "revoked") -> None:
        if source_kind != SourceKind.USER:
            raise ValueError("only a user may revoke a persistent claim")
        parent_event_id = None
        for established in self._valid_established_events(session_id).items():
            established_claim_id, event_id = established
            if established_claim_id == claim_id:
                parent_event_id = event_id
        if parent_event_id is None:
            raise KeyError("unknown established claim: %s" % claim_id)
        self.event_store.append(CognitiveEvent(
            session_id=session_id, kind=EventKind.SELF_CLAIM_REVOKED,
            source_kind=source_kind, source_ref=source_ref,
            payload={"claim_id": claim_id, "reason": reason}, parent_event_ids=(parent_event_id,),
        ))

    def current_claims(self, session_id: str, now: Optional[str] = None) -> List[SelfModelClaim]:
        current = parse_aware_iso8601(now) if now is not None else datetime.now(timezone.utc)
        claims: Dict[str, SelfModelClaim] = {}
        revoked = set()
        events = self.event_store.list(session_id)
        by_id = {event.event_id: event for event in events}
        proposals = {}
        proposed_claim_ids = set()
        used_proposals = set()
        used_approvals = set()
        established_events = {}
        for event in events:
            if event.kind != EventKind.SELF_CLAIM_PROPOSED:
                continue
            try:
                claim = self._claim_from_payload(event.payload["claim"])
            except (KeyError, TypeError, ValueError):
                continue
            if tuple(event.parent_event_ids) != claim.evidence_event_ids:
                continue
            if all(parent in by_id and by_id[parent].sequence < event.sequence for parent in event.parent_event_ids):
                if claim.claim_id not in proposed_claim_ids:
                    proposals[event.event_id] = claim
                    proposed_claim_ids.add(claim.claim_id)
        for event in events:
            if event.kind == EventKind.SELF_CLAIM_APPROVED:
                proposal_id = event.payload.get("proposal_event_id")
                approval_id = event.payload.get("approval_event_id")
                proposal = proposals.get(proposal_id)
                approval = by_id.get(approval_id)
                if (proposal is None or approval is None or approval.source_kind != SourceKind.USER
                        or approval.payload.get("approval") != "self_claim"
                        or approval.payload.get("claim_id") != proposal.claim_id
                        or approval.payload.get("proposal_event_id") != proposal_id
                        or tuple(event.parent_event_ids) != (proposal_id, approval_id)
                        or event.source_kind != SourceKind.USER
                        or approval.sequence <= by_id[proposal_id].sequence
                        or proposal_id in used_proposals or approval_id in used_approvals):
                    continue
                if event.payload.get("claim") != self._claim_payload(proposal):
                    continue
                claims[proposal.claim_id] = proposal
                established_events[proposal.claim_id] = event.event_id
                used_proposals.add(proposal_id)
                used_approvals.add(approval_id)
        for event in events:
            if event.kind == EventKind.SELF_CLAIM_REVOKED and event.source_kind == SourceKind.USER:
                claim_id = event.payload.get("claim_id")
                if (claim_id and tuple(event.parent_event_ids) ==
                        (established_events.get(claim_id),)):
                    revoked.add(claim_id)
        return [claim for claim_id, claim in claims.items()
                if claim_id not in revoked and (claim.expires_at is None or parse_aware_iso8601(claim.expires_at) > current)]

    def _valid_established_events(self, session_id: str) -> Dict[str, str]:
        """Return claim IDs and their valid approval event IDs for causal writes."""
        events = self.event_store.list(session_id)
        by_id = {event.event_id: event for event in events}
        proposals = {}
        proposed_claim_ids = set()
        for event in events:
            if event.kind == EventKind.SELF_CLAIM_PROPOSED:
                try:
                    claim = self._claim_from_payload(event.payload["claim"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (tuple(event.parent_event_ids) == claim.evidence_event_ids
                        and all(parent in by_id and by_id[parent].sequence < event.sequence
                                for parent in event.parent_event_ids)):
                    if claim.claim_id not in proposed_claim_ids:
                        proposals[event.event_id] = claim
                        proposed_claim_ids.add(claim.claim_id)
        established, used_proposals, used_approvals = {}, set(), set()
        for event in events:
            if event.kind != EventKind.SELF_CLAIM_APPROVED:
                continue
            proposal_id, approval_id = event.payload.get("proposal_event_id"), event.payload.get("approval_event_id")
            proposal, approval = proposals.get(proposal_id), by_id.get(approval_id)
            if (proposal is None or approval is None or approval.source_kind != SourceKind.USER
                    or event.source_kind != SourceKind.USER
                    or approval.payload.get("approval") != "self_claim"
                    or approval.payload.get("claim_id") != proposal.claim_id
                    or approval.payload.get("proposal_event_id") != proposal_id
                    or approval.sequence <= by_id[proposal_id].sequence
                    or tuple(event.parent_event_ids) != (proposal_id, approval_id)
                    or event.payload.get("claim") != self._claim_payload(proposal)
                    or proposal_id in used_proposals or approval_id in used_approvals):
                continue
            established[proposal.claim_id] = event.event_id
            used_proposals.add(proposal_id)
            used_approvals.add(approval_id)
        return established
