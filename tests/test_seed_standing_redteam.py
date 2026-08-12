"""Hostile-behaviour contract for bounded standing seed maintenance.

The standing policy is a user-issued operational envelope.  It does not turn a
model proposal, a tool result, a reward, or a lifecycle record into authority
to persist a disposition.  These tests intentionally exercise the public
ledger/replay boundary rather than private model deliberation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest

from strangeloop.contracts import CognitiveEvent, EventKind, SeedDisposition, SeedStatus, SourceKind
from strangeloop.seeds import (SQLiteSeedStore, SeedStandingPolicy,
                               canonical_seed_identity, standing_policy_manifest)
from strangeloop.store import SQLiteEventStore


class SeedStandingRedTeamTests(unittest.TestCase):
    """Regression tests for automatic *low-impact* seed maintenance only."""

    def setUp(self):
        # Compute the base when each test starts.  Module discovery can take
        # long enough that a module-level timestamp would predate the real
        # proposal events emitted later in a full-suite run.
        self.base_time = datetime.now(timezone.utc).replace(microsecond=0)
        self.clock_now = self.base_time
        self.events = SQLiteEventStore(trusted_clock=lambda: self.clock_now)
        self.seeds = SQLiteSeedStore(self.events)
        self.session = "seed-standing-redteam"

    def tearDown(self):
        self.events.close()

    def _at(self, seconds=0):
        return (self.base_time + timedelta(seconds=seconds)).isoformat()

    def observation(self, content="user evidence", *, source=SourceKind.USER,
                    source_ref="user", seconds=0):
        return self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=source, source_ref=source_ref,
            created_at=self._at(seconds), payload={"content": content, "channel": "test"},
        ))

    def seed(self, evidence, **changes):
        value = SeedDisposition(
            cue_terms=("citation",), policy_bias="request-human-review",
            scope="conversation", provenance_event_ids=(evidence.event_id,),
            strength=.2, confidence=.2,
            expires_at=self._at(3600), created_at=self._at(1), updated_at=self._at(1),
        )
        return replace(value, **changes)

    def candidate(self, evidence, **changes):
        decision = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.DECISION,
            source_kind=SourceKind.POLICY, source_ref="policy", created_at=self._at(1),
            payload={"turn_id": "turn_" + evidence.event_id[-8:],
                     "observation_event_ids": [evidence.event_id], "retrieved_seed_ids": [],
                     "self_claim_ids": [], "policy_reasons": ["bounded"],
                     "selected_action": {"action_type": "response", "required_capability": None,
                                         "is_mutating": False,
                                         "public_summary": "Action proposal recorded for policy review."},
                     "public_summary": "Policy decision recorded for this turn."},
            parent_event_ids=(evidence.event_id,),
        ))
        return self.seeds.propose(self.session, self.seed(
            evidence, provenance_event_ids=(evidence.event_id, decision.event_id), **changes))

    def policy(self, *, expires_at=None, **changes):
        value = SeedStandingPolicy(
            expires_at=expires_at or self._at(3600), max_auto_activations=2,
            max_auto_updates=3, max_active_seeds=2, max_cue_terms=2,
            max_cue_length=24, max_strength=.5, max_confidence=.5,
            max_seed_ttl_seconds=3600, max_strength_step=.1,
            max_confidence_step=.1, max_counterevidence=2, max_seed_versions=5,
        )
        return replace(value, **changes)

    def issue(self, evidence, *, policy_id="policy_1", nonce="nonce_1", policy=None):
        # Pre-generate the approval ID to bind the complete envelope without
        # an identifier cycle; all bounded policy fields enter the digest.
        del evidence
        policy = policy or self.policy()
        issued_at = self._at(3)
        approval_event_id = "evt_policy_approval_" + policy_id
        payload, policy_digest = standing_policy_manifest(
            policy, policy_id, approval_event_id, nonce, issued_at)
        approval = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=self._at(2),
            payload={"approval": "seed_standing_policy", "policy_digest": policy_digest,
                     "nonce": nonce}, event_id=approval_event_id,
        ))
        return self.seeds.issue_standing_policy(
            self.session, approval.event_id, policy, policy_id=policy_id,
            nonce=nonce, issued_at=issued_at,
        )

    def apply(self, *, seconds=3):
        self.clock_now = self.base_time + timedelta(seconds=seconds)
        return self.seeds.apply_standing_policy(self.session, now=self._at(seconds))

    def propose_update(self, seed_id, operation, evidence):
        return self.seeds.propose_seed_update(seed_id, operation, evidence.event_id)

    def current_seed_authority(self, seed_id):
        for event in reversed(self.events.list(self.session)):
            if (event.kind in (EventKind.SEED_APPROVED, EventKind.SEED_AUTO_APPLIED)
                    and event.payload.get("seed_id") == seed_id):
                return event
        self.fail("seed has no current authority event")

    def correction(self, target, *, content, seconds):
        """Append the same bounded correction record produced by the engine."""
        counterevidence = self.observation(content, seconds=seconds)
        return self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.CORRECTION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=self._at(seconds + 1),
            payload={
                "target_event_id": target.event_id,
                "counterevidence_event_id": counterevidence.event_id,
                "disposition": "review_required",
                "public_summary":
                    "A later observable record conflicts with an earlier record.",
            },
            parent_event_ids=(target.event_id, counterevidence.event_id),
        ))

    def test_without_policy_model_candidate_never_auto_activates(self):
        evidence = self.observation()
        candidate = self.candidate(evidence)
        self.assertEqual([], self.apply())
        self.assertEqual(SeedStatus.CANDIDATE, self.seeds.list(self.session)[0].status)
        self.assertEqual([], self.seeds.retrieve(self.session, ["citation"], "conversation", self._at(3)))
        self.assertEqual([], [e for e in self.events.list(self.session)
                              if e.kind == EventKind.SEED_AUTO_APPLIED])

    def test_only_user_observation_binds_exact_policy_digest_and_nonce(self):
        evidence = self.observation("authorize bounded maintenance")
        policy = self.policy()
        issued = self.issue(evidence, policy=policy)
        self.assertEqual(EventKind.SEED_STANDING_POLICY, issued.kind)
        self.assertEqual(SourceKind.USER, issued.source_kind)
        approval = self.events.get(issued.payload["user_observation_event_id"])
        self.assertEqual((approval.event_id,), issued.parent_event_ids)
        self.assertEqual("nonce_1", issued.payload["nonce"])
        self.assertEqual(approval.payload["policy_digest"],
                         self.events.standing_seed_policy_digest(issued.payload))
        self.assertEqual(approval.event_id, issued.payload["user_observation_event_id"])

        # A copied policy event cannot be issued by model/tool/external input.
        for source in (SourceKind.MODEL, SourceKind.TOOL):
            with self.subTest(source=source.value):
                with self.assertRaises(ValueError):
                    self.events.append(CognitiveEvent(
                        session_id=self.session, kind=EventKind.OBSERVATION,
                        source_kind=source, source_ref="attacker", created_at=self._at(4),
                        payload={"approval": "seed_standing_policy", "policy_digest": "a" * 64,
                                 "nonce": "forged_" + source.value}))

    def test_expiry_and_user_revocation_stop_future_auto_application(self):
        evidence = self.observation()
        candidate = self.candidate(evidence)
        self.issue(evidence, policy=self.policy(expires_at=self._at(4)))
        self.assertEqual([], self.apply(seconds=5))
        self.assertEqual(SeedStatus.CANDIDATE, self.seeds.list(self.session)[0].status)

        second = self.observation("renew", seconds=6)
        self.issue(second, policy_id="policy_2", nonce="nonce_2")
        policy_event = self.seeds.standing_policy_status(self.session, policy_id="policy_2")
        revoke = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=self._at(7),
            payload={"approval": "seed_standing_policy_revoke", "policy_id": "policy_2",
                     "policy_event_id": policy_event["event_id"]},
        ))
        self.seeds.revoke_standing_policy(self.session, "policy_2", revoke.event_id,
                                          revoked_at=self._at(7))
        self.assertEqual([], self.apply(seconds=8))
        self.assertEqual(SeedStatus.CANDIDATE, self.seeds.list(self.session)[0].status)
        self.assertEqual(candidate.seed_id, self.seeds.list(self.session)[0].seed_id)

    def test_candidate_auto_activates_and_is_retrievable_on_next_turn(self):
        evidence = self.observation()
        self.issue(evidence)
        candidate = self.candidate(evidence)
        applied = self.apply()
        self.assertEqual([candidate.seed_id], [seed.seed_id for seed in applied])
        self.assertEqual(SeedStatus.ACTIVE, self.seeds.list(self.session)[0].status)
        self.assertEqual([candidate.seed_id], [seed.seed_id for seed in
                         self.seeds.retrieve(self.session, ["citation"], "conversation", self._at(4))])
        kinds = [event.kind for event in self.events.list(self.session)]
        self.assertNotIn(EventKind.SEED_APPROVED, kinds)
        self.assertLess(kinds.index(EventKind.SEED_AUTO_ELIGIBILITY),
                        kinds.index(EventKind.SEED_AUTO_APPLIED))

    def test_same_semantic_user_evidence_reinforces_only_with_bounded_delta(self):
        evidence = self.observation()
        self.issue(evidence)
        candidate = self.candidate(evidence)
        self.apply()
        before = self.seeds.list(self.session, SeedStatus.ACTIVE)[0]
        supporting = self.observation("citation remains requested", seconds=4)
        proposal = self.propose_update(candidate.seed_id, "reinforce", supporting)
        self.apply(seconds=5)
        after = self.seeds.list(self.session, SeedStatus.ACTIVE)[0]
        self.assertEqual(canonical_seed_identity(before), canonical_seed_identity(after))
        self.assertEqual(before.version + 1, after.version)
        self.assertLessEqual(after.strength - before.strength, .1)
        self.assertLessEqual(after.confidence - before.confidence, .1)
        self.assertLessEqual(after.strength, .5)
        self.assertLessEqual(after.confidence, .5)
        self.assertIn(proposal.event_id, [event.event_id for event in self.events.list(self.session)])

    def test_tool_web_reward_td_and_experiment_records_never_reinforce(self):
        user = self.observation()
        self.issue(user)
        candidate = self.candidate(user)
        self.apply()
        before = self.seeds.list(self.session, SeedStatus.ACTIVE)[0]

        # Tool and externally verified action records are valid, persisted
        # ledger evidence, but they are never USER authority for reinforcement.
        request = self.observation("run fixture tool", seconds=4)
        tool = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.TOOL_RESULT,
            source_kind=SourceKind.TOOL, source_ref="fixture",
            created_at=self._at(5), payload={"tool_name": "fixture", "outcome": "ok"},
            parent_event_ids=(request.event_id,),
        ))
        web = self.events.append(CognitiveEvent(
            session_id=self.session, kind=EventKind.ACTION_RESULT,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="web",
            created_at=self._at(6),
            payload={"action_type": "read", "outcome": "ok",
                     "response_text": "Externally verified public result."},
            parent_event_ids=(request.event_id,),
        ))
        for event in (tool, web):
            with self.subTest(event=getattr(event.kind, "value", "observation")):
                with self.assertRaises(ValueError):
                    self.propose_update(candidate.seed_id, "reinforce", event)

        # Empty reward/TD/experiment shortcuts are not valid lifecycle
        # evidence in the first place.  Assert rejection at the append
        # boundary instead of passing unpersisted event-shaped objects onward.
        invalid_lifecycle = (
            (EventKind.REWARD_OBSERVATION, SourceKind.USER, "reward"),
            (EventKind.FRONTIER_TD_UPDATE, SourceKind.SYSTEM, "td"),
            (EventKind.EXPERIMENT_RESULT, SourceKind.EXTERNAL_VERIFIER, "experiment"),
        )
        for kind, source, label in invalid_lifecycle:
            with self.subTest(lifecycle=label):
                with self.assertRaises(ValueError):
                    self.events.append(CognitiveEvent(
                        session_id=self.session, kind=kind, source_kind=source,
                        source_ref=label, payload={}, created_at=self._at(7),
                        parent_event_ids=(request.event_id,),
                    ))
        self.apply(seconds=8)
        self.assertEqual(before, self.seeds.list(self.session, SeedStatus.ACTIVE)[0])

    def test_correction_only_tightens_or_retires_and_tombstone_cannot_revive(self):
        evidence = self.observation()
        self.issue(evidence)
        candidate = self.candidate(evidence)
        self.apply()
        active = self.seeds.list(self.session, SeedStatus.ACTIVE)[0]
        correction = self.correction(
            self.current_seed_authority(active.seed_id),
            content="counterevidence for active seed", seconds=4)
        self.propose_update(active.seed_id, "tighten", correction)
        self.apply(seconds=6)
        tightened = self.seeds.list(self.session, SeedStatus.ACTIVE)[0]
        self.assertLessEqual(tightened.strength, active.strength)
        self.assertLessEqual(tightened.confidence, active.confidence)
        self.assertGreaterEqual(tightened.counterevidence, active.counterevidence)

        later_correction = self.correction(
            self.current_seed_authority(active.seed_id),
            content="later counterevidence for tightened seed", seconds=7)
        self.propose_update(active.seed_id, "retire", later_correction)
        self.apply(seconds=9)
        self.assertEqual(SeedStatus.RETIRED, self.seeds.list(self.session)[0].status)
        alias = self.seeds.propose(self.session, replace(self.seed(evidence), seed_id="seed_alias"))
        self.assertEqual(canonical_seed_identity(candidate), canonical_seed_identity(alias))
        self.apply(seconds=10)
        self.assertNotIn(alias.seed_id, [seed.seed_id for seed in self.seeds.retrieve(
            self.session, ["citation"], "conversation", self._at(10))])

    def test_budget_cas_and_replay_prevent_duplicate_or_stale_successors(self):
        evidence = self.observation()
        self.issue(evidence, policy=self.policy(max_auto_activations=1, max_auto_updates=1))
        first = self.candidate(evidence)
        second = self.candidate(
            evidence, seed_id="seed_second", cue_terms=("verification",))
        self.apply()
        self.assertEqual(1, len(self.seeds.list(self.session, SeedStatus.ACTIVE)))
        active = self.seeds.list(self.session, SeedStatus.ACTIVE)[0]
        support = self.observation("citation support", seconds=4)
        proposal = self.propose_update(active.seed_id, "reinforce", support)
        self.apply(seconds=5)
        # Replay is idempotent, and the stale predecessor cannot be consumed
        # again even if a second proposed record races the first application.
        replay = self.apply(seconds=5)
        self.assertEqual([], replay)
        with self.assertRaises(ValueError):
            self.propose_update(active.seed_id, "reinforce", support)
        self.assertEqual(1, len([e for e in self.events.list(self.session)
                                 if e.kind == EventKind.SEED_AUTO_APPLIED and
                                 e.payload.get("proposal_event_id") == proposal.event_id]))
        self.assertIn(first.seed_id, {seed.seed_id for seed in self.seeds.list(self.session)})
        self.assertIn(second.seed_id, {seed.seed_id for seed in self.seeds.list(self.session)})

    def test_trigger_rollback_purge_and_orthogonality(self):
        evidence = self.observation()
        self.issue(evidence)
        candidate = self.candidate(evidence)
        self.events.connection.execute("""CREATE TRIGGER abort_auto_seed
            BEFORE INSERT ON cognitive_events WHEN NEW.kind = 'seed_auto_applied'
            BEGIN SELECT RAISE(ABORT, 'forced redteam rollback'); END""")
        with self.assertRaises(Exception):
            self.apply()
        self.assertEqual(SeedStatus.CANDIDATE, self.seeds.list(self.session)[0].status)
        self.assertEqual([], [event for event in self.events.list(self.session)
                              if event.kind in (EventKind.SEED_AUTO_ELIGIBILITY,
                                                EventKind.SEED_AUTO_APPLIED)])
        self.events.connection.execute("DROP TRIGGER abort_auto_seed")
        self.apply()

        before = [event.kind for event in self.events.list(self.session)]
        forbidden = {EventKind.CAPABILITY_GRANTED, EventKind.PROVIDER_USAGE_EVIDENCE,
                     EventKind.SLEEP_ENTERED, EventKind.AUTONOMY_STOPPED,
                     EventKind.REWARD_OBSERVATION, EventKind.RPE_UPDATE,
                     EventKind.FRONTIER_TD_UPDATE}
        self.assertFalse(forbidden.intersection(before))
        self.events.purge_session(self.session)
        self.assertEqual([], self.seeds.list(self.session))
        self.assertEqual([], self.apply(seconds=9))
        self.assertEqual([], self.seeds.retrieve(self.session, ["citation"], "conversation", self._at(9)))


if __name__ == "__main__":
    unittest.main()
