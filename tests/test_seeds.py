import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from strangeloop.contracts import CognitiveEvent, EventKind, SeedDisposition, SeedStatus, SourceKind
from strangeloop.seeds import SQLiteSeedStore
from strangeloop.store import SQLiteEventStore


class SQLiteSeedStoreTests(unittest.TestCase):
    def setUp(self):
        self.events = SQLiteEventStore()
        self.seeds = SQLiteSeedStore(self.events)
        self.session = "s"
        self.evidence = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={"content": "prefer citations"}))

    def tearDown(self):
        self.events.close()

    def seed(self, **changes):
        seed = SeedDisposition(cue_terms=("citation",), policy_bias="cite sources",
                               scope="research", provenance_event_ids=(self.evidence.event_id,),
                               strength=.8, confidence=.9)
        return replace(seed, **changes)

    def approve(self, seed):
        approval = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
            source_ref="user", payload={"approval": "seed", "seed_id": seed.seed_id,
            "proposal_event_id": self.seeds.proposal_event_id(seed.seed_id, self.session)}))
        return self.seeds.approve(seed.seed_id, approval.event_id)

    def test_candidate_is_not_retrieved_until_explicitly_approved(self):
        candidate = self.seeds.propose(self.session, self.seed())
        self.assertEqual(candidate.status, SeedStatus.CANDIDATE)
        self.assertEqual(self.seeds.retrieve(self.session, ["need a citation"], "research"), [])
        self.approve(candidate)
        self.assertEqual([seed.seed_id for seed in self.seeds.retrieve(self.session, ["citation"], "research")],
                         [candidate.seed_id])

    def test_model_cannot_approve_and_expired_or_retired_seeds_do_not_retrieve(self):
        candidate = self.seeds.propose(self.session, self.seed())
        with self.assertRaises(ValueError):
            self.events.append(CognitiveEvent(
                session_id=self.session, kind=EventKind.OBSERVATION, source_kind=SourceKind.MODEL,
                source_ref="model", payload={"approval": "seed", "seed_id": candidate.seed_id,
                "proposal_event_id": self.seeds.proposal_event_id(candidate.seed_id, self.session)}))
        expired = self.seeds.propose(self.session, self.seed(
            expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()))
        self.approve(expired)
        self.assertEqual(self.seeds.retrieve(self.session, ["citation"], "research"), [])
        active = self.seeds.propose(self.session, self.seed())
        self.approve(active)
        self.seeds.retire(active.seed_id)
        self.assertEqual(self.seeds.retrieve(self.session, ["citation"], "research"), [])

    def test_purge_removes_seed_rows(self):
        self.seeds.propose(self.session, self.seed())
        self.events.purge_session(self.session)
        self.assertEqual(self.seeds.list(self.session), [])
        count = self.events.connection.execute(
            "SELECT COUNT(*) FROM seeds WHERE session_id = ?", (self.session,)
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_approval_is_bound_to_user_payload_and_projection_events(self):
        first = self.seeds.propose(self.session, self.seed())
        second = self.seeds.propose(self.session, self.seed())
        approval = self.events.append(CognitiveEvent(session_id=self.session,
            kind=EventKind.OBSERVATION, source_kind=SourceKind.USER, source_ref="user",
            payload={"approval": "seed", "seed_id": first.seed_id,
            "proposal_event_id": self.seeds.proposal_event_id(first.seed_id, self.session)}))
        with self.assertRaises(ValueError):
            self.seeds.approve(second.seed_id, approval.event_id)
        active = self.seeds.approve(first.seed_id, approval.event_id)
        row = self.events.connection.execute("SELECT * FROM seeds WHERE seed_id = ?", (first.seed_id,)).fetchone()
        approved = self.events.get(row["approved_event_id"])
        self.assertEqual(active.status, SeedStatus.ACTIVE)
        self.assertEqual(approved.parent_event_ids, (row["proposal_event_id"], approval.event_id))
        self.assertEqual(row["approval_event_id"], approval.event_id)

    def test_trigger_failure_rolls_back_projection_and_event(self):
        self.events.connection.execute("""CREATE TRIGGER fail_seed_proposal
            BEFORE INSERT ON cognitive_events WHEN NEW.kind = 'seed_proposed'
            BEGIN SELECT RAISE(ABORT, 'forced failure'); END""")
        with self.assertRaises(Exception):
            self.seeds.propose(self.session, self.seed())
        self.assertEqual(self.seeds.list(self.session), [])
        self.assertEqual([event for event in self.events.list(self.session)
                          if event.kind == EventKind.SEED_PROPOSED], [])

    def test_approval_event_failure_rolls_back_active_projection(self):
        candidate = self.seeds.propose(self.session, self.seed())
        approval = self.events.append(CognitiveEvent(session_id=self.session,
            kind=EventKind.OBSERVATION, source_kind=SourceKind.USER, source_ref="user",
            payload={"approval": "seed", "seed_id": candidate.seed_id,
            "proposal_event_id": self.seeds.proposal_event_id(candidate.seed_id, self.session)}))
        self.events.connection.execute("""CREATE TRIGGER fail_seed_approval
            BEFORE INSERT ON cognitive_events WHEN NEW.kind = 'seed_approved'
            BEGIN SELECT RAISE(ABORT, 'forced failure'); END""")
        with self.assertRaises(Exception):
            self.seeds.approve(candidate.seed_id, approval.event_id)
        self.assertEqual(self.seeds.list(self.session)[0].status, SeedStatus.CANDIDATE)
        self.assertEqual([event for event in self.events.list(self.session)
                          if event.kind == EventKind.SEED_APPROVED], [])

    def test_aware_expiry_compares_instants_and_rejects_naive(self):
        with self.assertRaises(ValueError):
            self.seed(expires_at="2026-08-12T10:00:00")
        expired = self.seeds.propose(self.session, self.seed(expires_at="2026-08-12T10:00:00+08:00"))
        self.approve(expired)
        self.assertEqual(self.seeds.retrieve(self.session, ["citation"], "research",
                                             now="2026-08-12T02:01:00+00:00"), [])

    def test_sql_projection_tampering_does_not_activate_candidate(self):
        candidate = self.seeds.propose(self.session, self.seed())
        self.events.connection.execute("UPDATE seeds SET status = ? WHERE seed_id = ?",
                                       (SeedStatus.ACTIVE.value, candidate.seed_id))
        self.assertTrue(self.events.verify_chain(self.session))
        self.assertEqual(self.seeds.list(self.session)[0].status, SeedStatus.CANDIDATE)
        self.assertEqual(self.seeds.retrieve(self.session, ["citation"], "research"), [])

    def test_approval_payload_must_bind_proposal_and_cannot_precede_it(self):
        candidate = self.seeds.propose(self.session, self.seed())
        wrong = self.events.append(CognitiveEvent(session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={"approval": "seed",
            "seed_id": candidate.seed_id, "proposal_event_id": "evt_wrong"}))
        with self.assertRaises(ValueError):
            self.seeds.approve(candidate.seed_id, wrong.event_id)
        self.assertEqual(self.seeds.proposal_event_id(candidate.seed_id, self.session),
                         self.events.list(self.session)[1].event_id)
