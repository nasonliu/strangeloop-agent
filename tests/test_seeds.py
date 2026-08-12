import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from strangeloop.contracts import CognitiveEvent, EventKind, SeedDisposition, SeedStatus, SourceKind
from strangeloop.seeds import (SQLiteSeedStore, SeedStandingPolicy,
                               canonical_seed_identity, standing_policy_manifest)
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

    def test_canonical_semantic_identity_normalizes_case_spacing_and_cue_order(self):
        first = self.seed(cue_terms=(" Citation ", "CHECK"))
        second = replace(first, seed_id="seed_alias", cue_terms=("check", "citation"))
        self.assertEqual(canonical_seed_identity(first), canonical_seed_identity(second))

    def test_standing_policy_requires_exact_digest_and_nonce(self):
        issued = "2026-08-12T10:00:01+00:00"
        policy = SeedStandingPolicy(expires_at="2026-08-13T10:00:00+00:00")
        approval_id = "evt_seed_policy_approval"
        _, digest = standing_policy_manifest(
            policy, "seedpolicy_test", approval_id, "nonce_test", issued)
        approval = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", event_id=approval_id,
            created_at="2026-08-12T10:00:00+00:00",
            payload={"approval": "seed_standing_policy", "policy_digest": digest,
                     "nonce": "nonce_test"}))
        event = self.seeds.issue_standing_policy(
            self.session, approval.event_id, policy, "nonce_test",
            policy_id="seedpolicy_test", issued_at=issued)
        self.assertEqual(EventKind.SEED_STANDING_POLICY, event.kind)
        self.assertEqual("active", self.seeds.standing_policy_status(
            self.session, now="2026-08-12T10:00:02+00:00")["status"])


class StandingSeedEvidenceBindingTests(unittest.TestCase):
    """Standing updates consume only deterministic, seed-bound USER evidence."""

    def setUp(self):
        self.origin = datetime.now(timezone.utc)
        self.clock_now = self.origin
        self.events = SQLiteEventStore(trusted_clock=lambda: self.clock_now)
        self.seeds = SQLiteSeedStore(self.events)
        self.session = "seed-evidence-binding"
        self.offset = 0
        self._issue_policy()

    def tearDown(self):
        self.events.close()

    def _time(self):
        self.offset += 1
        self.clock_now = self.origin + timedelta(seconds=self.offset)
        return self.clock_now.isoformat()

    def _observation(self, content, source=SourceKind.USER):
        return self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=source, source_ref=source.value,
            created_at=self._time(), payload={"content": content, "channel": "test"},
        ))

    def _issue_policy(self):
        policy = SeedStandingPolicy(
            expires_at=(self.origin + timedelta(hours=1)).isoformat(),
            max_auto_activations=4, max_auto_updates=8, max_active_seeds=4,
            max_seed_ttl_seconds=3600,
        )
        approval_id = "evt_evidence_policy_approval"
        issued_at = (self.origin + timedelta(seconds=2)).isoformat()
        _, digest = standing_policy_manifest(
            policy, "seedpolicy_evidence", approval_id, "nonce_evidence", issued_at)
        approval = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", event_id=approval_id,
            created_at=(self.origin + timedelta(seconds=1)).isoformat(),
            payload={"approval": "seed_standing_policy",
                     "policy_digest": digest, "nonce": "nonce_evidence"},
        ))
        self.offset = 2
        self.clock_now = self.origin + timedelta(seconds=self.offset)
        self.seeds.issue_standing_policy(
            self.session, approval.event_id, policy, nonce="nonce_evidence",
            policy_id="seedpolicy_evidence", issued_at=issued_at,
        )

    def _activate(self, cue="citation", seed_id=None):
        evidence = self._observation("%s requested" % cue)
        decision = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.DECISION,
            source_kind=SourceKind.POLICY, source_ref="policy",
            created_at=self._time(), parent_event_ids=(evidence.event_id,),
            payload={
                "turn_id": "turn_%s" % (seed_id or cue),
                "observation_event_ids": [evidence.event_id],
                "retrieved_seed_ids": [], "self_claim_ids": [],
                "policy_reasons": ["bounded"],
                "selected_action": {
                    "action_type": "response", "required_capability": None,
                    "is_mutating": False,
                    "public_summary": "Action proposal recorded for policy review.",
                },
                "public_summary": "Policy decision recorded for this turn.",
            },
        ))
        seed = SeedDisposition(
            cue_terms=(cue,), policy_bias="request-human-review",
            scope="conversation",
            provenance_event_ids=(evidence.event_id, decision.event_id),
            strength=.2, confidence=.2,
            seed_id=seed_id or "seed_%s" % cue,
            created_at=self._time(), updated_at=self._time(),
            expires_at=(self.origin + timedelta(minutes=30)).isoformat(),
        )
        candidate = self.seeds.propose(self.session, seed)
        active = self.seeds.auto_apply_candidate(candidate.seed_id, now=self._time())
        self.assertIsNotNone(active)
        return active

    def _authority(self, seed_id):
        for event in reversed(self.events.list(self.session)):
            if (event.kind in (EventKind.SEED_APPROVED, EventKind.SEED_AUTO_APPLIED)
                    and event.payload.get("seed_id") == seed_id):
                return event
        self.fail("active seed authority is missing")

    def _correction(self, target, counterevidence, source=SourceKind.USER):
        return self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.CORRECTION,
            source_kind=source, source_ref=source.value,
            created_at=self._time(),
            payload={
                "target_event_id": target.event_id,
                "counterevidence_event_id": counterevidence.event_id,
                "disposition": "review_required",
                "public_summary":
                    "A later observable record conflicts with an earlier record.",
            },
            parent_event_ids=(target.event_id, counterevidence.event_id),
        ))

    def test_reinforcement_requires_literal_current_cue_in_new_user_observation(self):
        active = self._activate("citation")
        unrelated = self._observation("weather forecast remains sunny")
        with self.assertRaisesRegex(ValueError, "literally match a current cue"):
            self.seeds.propose_seed_update(
                active.seed_id, "reinforce", unrelated.event_id)

        matching = self._observation("  CITATION   is still required ")
        proposal = self.seeds.propose_seed_update(
            active.seed_id, "reinforce", matching.event_id)
        updated = self.seeds.auto_apply_update(proposal.event_id, now=self._time())
        self.assertEqual(active.version + 1, updated.version)
        self.assertGreater(updated.strength, active.strength)

    def test_tighten_requires_user_correction_bound_to_seed_and_plain_counterevidence(self):
        active = self._activate("citation")
        direct = self._observation("citation is contradicted")
        with self.assertRaisesRegex(ValueError, "requires a USER correction"):
            self.seeds.propose_seed_update(active.seed_id, "tighten", direct.event_id)

        authority = self._authority(active.seed_id)
        external_counter = self._observation("citation conflict")
        external = self._correction(
            authority, external_counter, source=SourceKind.EXTERNAL_VERIFIER)
        with self.assertRaisesRegex(ValueError, "requires a USER correction"):
            self.seeds.propose_seed_update(active.seed_id, "tighten", external.event_id)

        unrelated_target = self._observation("unrelated target")
        wrong_counter = self._observation("citation counterevidence")
        wrong = self._correction(unrelated_target, wrong_counter)
        with self.assertRaisesRegex(ValueError, "target must bind"):
            self.seeds.propose_seed_update(active.seed_id, "tighten", wrong.event_id)

        counter = self._observation("citation counterevidence from user")
        correction = self._correction(authority, counter)
        proposal = self.seeds.propose_seed_update(
            active.seed_id, "tighten", correction.event_id)
        tightened = self.seeds.auto_apply_update(proposal.event_id, now=self._time())
        self.assertEqual(active.counterevidence + 1, tightened.counterevidence)
        self.assertLessEqual(tightened.strength, active.strength)

    def test_bottom_user_counterevidence_is_consumed_only_once_across_seeds(self):
        first = self._activate("citation", "seed_citation")
        second = self._activate("verification", "seed_verification")
        counter = self._observation("one conflicting user observation")
        first_correction = self._correction(self._authority(first.seed_id), counter)
        self.seeds.propose_seed_update(
            first.seed_id, "tighten", first_correction.event_id)

        second_correction = self._correction(self._authority(second.seed_id), counter)
        with self.assertRaisesRegex(ValueError, "already been consumed"):
            self.seeds.propose_seed_update(
                second.seed_id, "retire", second_correction.event_id)
