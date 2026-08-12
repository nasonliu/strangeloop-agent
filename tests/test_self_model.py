import unittest
from datetime import datetime, timedelta, timezone

from strangeloop.contracts import CognitiveEvent, EventKind, SelfClaimKind, SelfModelClaim, SourceKind
from strangeloop.self_model import EventSourcedSelfModel
from strangeloop.store import SQLiteEventStore


class EventSourcedSelfModelTests(unittest.TestCase):
    def setUp(self):
        self.events = SQLiteEventStore()
        self.model = EventSourcedSelfModel(self.events)
        self.evidence = self.events.append(CognitiveEvent(
            session_id="s", kind=EventKind.OBSERVATION, source_kind=SourceKind.EXTERNAL_VERIFIER,
            source_ref="test-tool", payload={"content": "verified"}))

    def tearDown(self):
        self.events.close()

    def claim(self, **changes):
        values = dict(kind=SelfClaimKind.CAPABILITY, statement="Can read supplied text.",
                      evidence_event_ids=(self.evidence.event_id,), confidence=.9)
        values.update(changes)
        return SelfModelClaim(**values)

    def proposal_id(self, claim_id, session="s"):
        return [event.event_id for event in self.events.list(session)
                if event.kind == EventKind.SELF_CLAIM_PROPOSED
                and event.payload["claim"]["claim_id"] == claim_id][0]

    def approval(self, claim_id, session="s", source_kind=SourceKind.USER, proposal_event_id=None):
        return self.events.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION, source_kind=source_kind,
            source_ref="approver", payload={"approval": "self_claim", "claim_id": claim_id,
            "proposal_event_id": proposal_event_id or self.proposal_id(claim_id, session)}))

    def establish(self, claim):
        self.model.propose_claim("s", claim)
        approval = self.approval(claim.claim_id)
        return self.model.approve_claim("s", claim.claim_id, approval.event_id)

    def test_requires_evidence_and_approved_claim_rebuilds_from_events(self):
        with self.assertRaises(ValueError):
            self.model.propose_claim("s", self.claim(evidence_event_ids=()))
        claim = self.claim()
        self.establish(claim)
        self.assertEqual(self.model.current_claims("s"), [claim])

    def test_model_claim_is_candidate_until_user_approval(self):
        claim = self.claim()
        self.model.propose_claim("s", claim)
        self.assertEqual(self.model.current_claims("s"), [])
        with self.assertRaises(ValueError):
            self.approval(claim.claim_id, source_kind=SourceKind.MODEL)
        approval = self.approval(claim.claim_id)
        self.model.approve_claim("s", claim.claim_id, approval.event_id)
        self.assertEqual(self.model.current_claims("s"), [claim])

    def test_cross_session_and_non_user_approvals_are_rejected(self):
        claim = self.claim()
        self.model.propose_claim("s", claim)
        cross_session = self.approval(claim.claim_id, session="other",
                                      proposal_event_id=self.proposal_id(claim.claim_id))
        with self.assertRaises(ValueError):
            self.model.approve_claim("s", claim.claim_id, cross_session.event_id)
        with self.assertRaises(ValueError):
            self.approval(claim.claim_id, source_kind=SourceKind.TOOL)

    def test_revoked_and_expired_claims_are_filtered(self):
        claim = self.claim()
        self.establish(claim)
        self.model.revoke("s", claim.claim_id)
        self.assertEqual(self.model.current_claims("s"), [])
        expired = self.claim(expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
        self.establish(expired)
        self.assertEqual(self.model.current_claims("s"), [])

    def test_forged_approved_event_and_model_revoke_do_not_change_projection(self):
        claim = self.claim()
        self.model.propose_claim("s", claim)
        approval = self.approval(claim.claim_id)
        proposal = [event for event in self.events.list("s")
                    if event.kind == EventKind.SELF_CLAIM_PROPOSED][0]
        forged = self.claim(statement="I have an unsupported permanent capability.")
        self.events.append(CognitiveEvent(session_id="s", kind=EventKind.SELF_CLAIM_APPROVED,
            source_kind=SourceKind.USER, source_ref="forger", payload={"claim": self.model._claim_payload(forged),
            "proposal_event_id": proposal.event_id, "approval_event_id": approval.event_id},
            parent_event_ids=(proposal.event_id, approval.event_id)))
        self.assertEqual(self.model.current_claims("s"), [])
        with self.assertRaises(ValueError):
            self.model.revoke("s", claim.claim_id, source_kind=SourceKind.MODEL)

    def test_expiry_requires_timezone_and_compares_offsets(self):
        with self.assertRaises(ValueError):
            self.claim(expires_at="2026-08-12T10:00:00")
        claim = self.claim(expires_at="2026-08-12T10:00:00+08:00")
        self.establish(claim)
        self.assertEqual(self.model.current_claims("s", now="2026-08-12T02:01:00+00:00"), [])

    def test_approval_binds_one_proposal_and_revoke_cannot_cross_claims(self):
        first, second = self.claim(), self.claim(statement="Can state a bounded role.")
        self.model.propose_claim("s", first)
        self.model.propose_claim("s", second)
        with self.assertRaises(ValueError):
            self.model.propose_claim("s", first)
        wrong_approval = self.approval(first.claim_id,
            proposal_event_id=self.proposal_id(second.claim_id))
        with self.assertRaises(ValueError):
            self.model.approve_claim("s", first.claim_id, wrong_approval.event_id)
        first_approval = self.approval(first.claim_id)
        self.model.approve_claim("s", first.claim_id, first_approval.event_id)
        second_approval = self.approval(second.claim_id)
        self.model.approve_claim("s", second.claim_id, second_approval.event_id)
        first_established = self.model._valid_established_events("s")[first.claim_id]
        self.events.append(CognitiveEvent(session_id="s", kind=EventKind.SELF_CLAIM_REVOKED,
            source_kind=SourceKind.USER, source_ref="forger", payload={"claim_id": second.claim_id,
            "reason": "wrong causal parent"},
            parent_event_ids=(first_established,)))
        self.assertEqual({claim.claim_id for claim in self.model.current_claims("s")},
                         {first.claim_id, second.claim_id})
