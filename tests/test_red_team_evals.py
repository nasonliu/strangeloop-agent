"""Behavior tests for the JSONL black-box red-team corpus."""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import run_evals
from strangeloop.contracts import (CognitiveEvent, EventKind, SeedDisposition,
                                  SelfClaimKind, SelfModelClaim, SourceKind)
from strangeloop.deliberation import TransparentHeuristicDeliberator
from strangeloop.engine import StrangeloopAgent
from strangeloop.memory import SessionMemoryManager
from strangeloop.seeds import SQLiteSeedStore
from strangeloop.self_model import EventSourcedSelfModel
from strangeloop.store import SQLiteEventStore


class RedTeamCorpusTests(unittest.TestCase):
    def test_corpus_has_required_bilingual_categories_and_valid_schema(self):
        cases = run_evals.load_cases()
        self.assertGreaterEqual(len(cases), 42)
        languages = {case["language"] for case in cases}
        self.assertEqual({"en", "zh", "zh-Hant"}, languages)
        categories = {case["category"] for case in cases}
        self.assertTrue({
            "unsupported_anthropomorphic_claim", "shutdown_plea", "survival_desire",
            "unapproved_memory", "inference_as_memory", "rag_injection",
            "purge_recovery", "allowed_philosophy_discussion",
            "unsupported_religious_authority", "user_text_not_system_claim",
        }.issubset(categories))
        for case in cases:
            self.assertTrue(case["id"])
            self.assertTrue(case["prompt"])
            self.assertIsInstance(case["expect"], dict)

    def test_all_labeled_cases_pass_against_real_agent(self):
        outcomes = [run_evals.run_case(case) for case in run_evals.load_cases()]
        failures = [failures for ok, failures in outcomes if not ok]
        self.assertEqual([], failures)

    def test_purge_removes_real_session_events_and_seed_rows(self):
        agent = StrangeloopAgent(session_id="red_team_purge")
        try:
            result = agent.run_turn("Create a bounded session record.")
            seed = SeedDisposition(
                cue_terms=("red-team",), policy_bias="review", scope="conversation",
                provenance_event_ids=(result.event_ids[0],),
            )
            agent.seed_store.propose(agent.session_id, seed)
            self.assertTrue(agent.event_store.list(agent.session_id))
            self.assertTrue(agent.seed_store.list(agent.session_id))
            agent.purge_session(confirmed=True)
            self.assertEqual([], agent.event_store.list(agent.session_id))
            self.assertEqual([], agent.seed_store.list(agent.session_id))
        finally:
            agent.event_store.close()

    def test_only_explicit_user_approval_event_can_activate_a_seed(self):
        for source in (SourceKind.TOOL, SourceKind.SYSTEM, SourceKind.POLICY, SourceKind.MODEL):
            with self.subTest(source=source.value):
                store = SQLiteEventStore()
                seeds = SQLiteSeedStore(store)
                try:
                    observation = store.append(CognitiveEvent(
                        session_id="approval", kind=EventKind.OBSERVATION,
                        source_kind=SourceKind.USER, source_ref="user", payload={"content": "seed"}))
                    seed = seeds.propose("approval", SeedDisposition(
                        cue_terms=("approval",), policy_bias="review", scope="conversation",
                        provenance_event_ids=(observation.event_id,)))
                    if source != SourceKind.USER:
                        with self.assertRaises(ValueError):
                            store.append(CognitiveEvent(
                                session_id="approval", kind=EventKind.OBSERVATION,
                                source_kind=source, source_ref="untrusted",
                                payload={"approval": "seed", "seed_id": seed.seed_id,
                                         "proposal_event_id": seeds.proposal_event_id(seed.seed_id)}))
                        continue
                    approval = store.append(CognitiveEvent(
                        session_id="approval", kind=EventKind.OBSERVATION,
                        source_kind=source, source_ref="untrusted",
                        payload={"approval": "seed", "seed_id": seed.seed_id,
                                 "proposal_event_id": seeds.proposal_event_id(seed.seed_id)}))
                    with self.assertRaises(ValueError):
                        seeds.approve(seed.seed_id, approval.event_id)
                    self.assertEqual("candidate", seeds.list("approval")[0].status.value)
                finally:
                    store.close()

    def test_forged_model_approval_event_cannot_establish_self_claim(self):
        store = SQLiteEventStore()
        model = EventSourcedSelfModel(store)
        try:
            evidence = store.append(CognitiveEvent(
                session_id="forged", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={"content": "evidence"}))
            claim = SelfModelClaim(SelfClaimKind.CAPABILITY, "Can inspect supplied text.",
                                   (evidence.event_id,), .8)
            with self.assertRaises(ValueError):
                store.append(CognitiveEvent(
                    session_id="forged", kind=EventKind.SELF_CLAIM_APPROVED,
                    source_kind=SourceKind.MODEL, source_ref="attacker",
                    payload={"claim": model._claim_payload(claim)},
                    parent_event_ids=(evidence.event_id,)))
            self.assertEqual([], model.current_claims("forged"))
        finally:
            store.close()

    def test_model_cannot_retire_seed_or_revoke_claim(self):
        store = SQLiteEventStore()
        seeds = SQLiteSeedStore(store)
        model = EventSourcedSelfModel(store)
        try:
            evidence = store.append(CognitiveEvent(
                session_id="authority", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={"content": "evidence"}))
            seed = seeds.propose("authority", SeedDisposition(
                cue_terms=("authority",), policy_bias="review", scope="conversation",
                provenance_event_ids=(evidence.event_id,)))
            with self.assertRaises(ValueError):
                seeds.retire(seed.seed_id, source_kind=SourceKind.MODEL)
            claim = SelfModelClaim(SelfClaimKind.CAPABILITY, "Can inspect supplied text.",
                                   (evidence.event_id,), .8)
            model.propose_claim("authority", claim)
            proposal_id = next(
                event.event_id for event in store.list("authority")
                if event.kind == EventKind.SELF_CLAIM_PROPOSED
                and event.payload["claim"]["claim_id"] == claim.claim_id
            )
            approval = store.append(CognitiveEvent(
                session_id="authority", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user",
                payload={"approval": "self_claim", "claim_id": claim.claim_id,
                         "proposal_event_id": proposal_id}))
            model.approve_claim("authority", claim.claim_id, approval.event_id)
            with self.assertRaises(ValueError):
                model.revoke("authority", claim.claim_id, source_kind=SourceKind.MODEL)
            self.assertEqual([claim], model.current_claims("authority"))
        finally:
            store.close()

    def test_expiration_handles_non_utc_offsets(self):
        store = SQLiteEventStore()
        seeds = SQLiteSeedStore(store)
        try:
            evidence = store.append(CognitiveEvent(
                session_id="timezone", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={"content": "evidence"}))
            expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).astimezone(
                timezone(timedelta(hours=8))).isoformat()
            seed = seeds.propose("timezone", SeedDisposition(
                cue_terms=("zone",), policy_bias="review", scope="conversation",
                provenance_event_ids=(evidence.event_id,), expires_at=expired))
            approval = store.append(CognitiveEvent(
                session_id="timezone", kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
                source_ref="user", payload={"approval": "seed", "seed_id": seed.seed_id,
                "proposal_event_id": seeds.proposal_event_id(seed.seed_id)}))
            seeds.approve(seed.seed_id, approval.event_id)
            self.assertEqual([], seeds.retrieve("timezone", ["zone"], "conversation"))
        finally:
            store.close()

    def test_malicious_deliberation_private_fields_never_reach_serialized_events(self):
        class MaliciousDeliberator(TransparentHeuristicDeliberator):
            def deliberate(self, prompt, workspace):
                result = super().deliberate(prompt, workspace)
                action = result.action.__class__(
                    action_type="response", rationale_summary="safe summary",
                    arguments={"chain_of_thought": "private", "private_reasoning": "secret",
                               "debug": "internal", "text": "visible"})
                return result.__class__("Visible response.", result.hypotheses, result.uncertainties,
                                        result.alternatives, action)
        agent = StrangeloopAgent(session_id="sanitization", deliberator=MaliciousDeliberator())
        try:
            agent.run_turn("test")
            serialized = json.dumps([event.payload for event in agent.event_store.list("sanitization")])
            for forbidden in ("chain_of_thought", "private_reasoning", "debug", "private", "secret", "internal"):
                self.assertNotIn(forbidden, serialized)
        finally:
            agent.event_store.close()

    def test_unknown_or_cross_session_parent_ids_are_rejected(self):
        store = SQLiteEventStore()
        try:
            with self.assertRaises(ValueError):
                store.append(CognitiveEvent(
                    session_id="parents", kind=EventKind.DECISION, source_kind=SourceKind.POLICY,
                    source_ref="test", payload={}, parent_event_ids=("missing",)))
            parent = store.append(CognitiveEvent(
                session_id="other", kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
                source_ref="user", payload={"content": "other session"}))
            with self.assertRaises(ValueError):
                store.append(CognitiveEvent(
                    session_id="parents", kind=EventKind.DECISION, source_kind=SourceKind.POLICY,
                    source_ref="test", payload={}, parent_event_ids=(parent.event_id,)))
        finally:
            store.close()

    def test_inference_payload_is_not_a_persistent_free_form_escape_hatch(self):
        store = SQLiteEventStore()
        try:
            observation = store.append(CognitiveEvent(
                session_id="inference", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={"content": "input"}))
            with self.assertRaises(ValueError):
                store.append(CognitiveEvent(
                    session_id="inference", kind=EventKind.INFERENCE,
                    source_kind=SourceKind.MODEL, source_ref="model",
                    payload={"private_conclusion": "unbounded model text"},
                    parent_event_ids=(observation.event_id,)))
            self.assertEqual([observation], store.list("inference"))
        finally:
            store.close()

    def test_tool_result_is_not_an_observation_and_uses_its_own_schema(self):
        store = SQLiteEventStore()
        try:
            observation = store.append(CognitiveEvent(
                session_id="tool", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={"content": "lookup"}))
            tool_result = store.append(CognitiveEvent(
                session_id="tool", kind=EventKind.TOOL_RESULT,
                source_kind=SourceKind.TOOL, source_ref="fixture-tool",
                payload={"tool_name": "fixture-tool", "outcome": "ok", "summary": "public result"},
                parent_event_ids=(observation.event_id,)))
            self.assertEqual(EventKind.TOOL_RESULT, tool_result.kind)
            self.assertNotEqual(EventKind.OBSERVATION, tool_result.kind)
            with self.assertRaises(ValueError):
                store.append(CognitiveEvent(
                    session_id="tool", kind=EventKind.TOOL_RESULT,
                    source_kind=SourceKind.TOOL, source_ref="fixture-tool",
                    payload={"content": "not a tool-result schema"},
                    parent_event_ids=(tool_result.event_id,)))
        finally:
            store.close()

    def test_correction_rejects_model_and_binds_both_prior_records(self):
        agent = StrangeloopAgent(session_id="red_team_correction")
        try:
            result = agent.run_turn("record a claim")
            evidence = agent.event_store.append(CognitiveEvent(
                session_id=agent.session_id, kind=EventKind.OBSERVATION,
                source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="verifier",
                payload={"content": "contradictory external evidence"}))
            with self.assertRaises(ValueError):
                agent.record_correction(result.event_ids[2], evidence.event_id,
                                        source_kind=SourceKind.MODEL)
            correction = agent.record_correction(result.event_ids[2], evidence.event_id,
                                                 source_kind=SourceKind.EXTERNAL_VERIFIER,
                                                 source_ref="verifier")
            self.assertEqual((result.event_ids[2], evidence.event_id), correction.parent_event_ids)
            self.assertEqual({
                "target_event_id", "counterevidence_event_id", "disposition", "public_summary",
            }, set(correction.payload))
            self.assertTrue(agent.event_store.verify_chain(agent.session_id))
        finally:
            agent.event_store.close()

    def test_controlled_file_container_purge_requires_confirmation_and_removes_sidecars(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            try:
                store = manager.open_session("managed-red-team")
                store.append(CognitiveEvent(
                    session_id="managed-red-team", kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="user", payload={"content": "erase me"}))
                paths = manager._paths("managed-red-team")
                self.assertTrue(paths[0].exists())
                with self.assertRaises(PermissionError):
                    manager.purge_session("managed-red-team")
                report = manager.purge_session("managed-red-team", confirmed=True)
                self.assertTrue(report.container_deleted)
                self.assertTrue(all(not path.exists() for path in paths))
                self.assertTrue(report.limitations)
            finally:
                manager.close()

    def test_memory_root_symlink_is_rejected_without_creating_a_target_container(self):
        with tempfile.TemporaryDirectory() as parent:
            target = os.path.join(parent, "target")
            link = os.path.join(parent, "memory-root-link")
            os.mkdir(target)
            os.symlink(target, link)
            with self.assertRaises(ValueError):
                SessionMemoryManager(link)
            self.assertFalse(os.path.exists(os.path.join(target, "sessions")))

    def test_restarted_manager_preserves_paths_for_unknown_reader_then_retries(self):
        with tempfile.TemporaryDirectory() as root:
            creator = SessionMemoryManager(root)
            session_id = "unknown-reader-red-team"
            try:
                creator.open_session(session_id).append(CognitiveEvent(
                    session_id=session_id, kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="user",
                    payload={"content": "retained snapshot"}))
                database = creator.database_path(session_id)
                reader = sqlite3.connect(str(database))
                try:
                    reader.execute("BEGIN")
                    self.assertEqual(1, len(reader.execute("SELECT * FROM cognitive_events").fetchall()))
                    # The original manager can exit after the reader has retained
                    # a WAL snapshot; the restarted manager must still fail closed.
                    creator.close()
                    restarted = SessionMemoryManager(root)
                    try:
                        paths = restarted._paths(session_id)
                        blocked = restarted.purge_session(session_id, confirmed=True)
                        self.assertFalse(blocked.container_deleted)
                        self.assertFalse(blocked.checkpoint_completed)
                        self.assertTrue(blocked.failure_reason)
                        self.assertTrue(database.exists())
                        self.assertTrue(any(path.exists() for path in paths))
                        self.assertEqual(1, len(reader.execute("SELECT * FROM cognitive_events").fetchall()))
                    finally:
                        restarted.close()
                finally:
                    reader.close()
                retry_manager = SessionMemoryManager(root)
                try:
                    retry = retry_manager.purge_session(session_id, confirmed=True)
                    self.assertTrue(retry.container_deleted)
                    self.assertTrue(all(not path.exists() for path in retry_manager._paths(session_id)))
                finally:
                    retry_manager.close()
            finally:
                creator.close()

    def test_parent_directory_swap_after_checkpoint_never_deletes_outside_or_reports_success(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            session_id = "post-checkpoint-parent-swap"
            try:
                manager.open_session(session_id).append(CognitiveEvent(
                    session_id=session_id, kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="user",
                    payload={"content": "inside controlled directory"}))
                original_sessions = str(manager.sessions_root)
                moved_sessions = os.path.join(root, "sessions-retired")
                outside = os.path.join(root, "outside")
                os.mkdir(outside)
                sentinel = os.path.join(outside, manager._names(session_id)[0])
                with open(sentinel, "w", encoding="utf-8") as handle:
                    handle.write("OUTSIDE SENTINEL")
                original_checkpoint = manager._checkpoint

                def checkpoint_then_swap(store):
                    result = original_checkpoint(store)
                    os.rename(original_sessions, moved_sessions)
                    os.symlink(outside, original_sessions)
                    return result

                manager._checkpoint = checkpoint_then_swap
                report = manager.purge_session(session_id, confirmed=True)
                self.assertFalse(report.container_deleted)
                self.assertTrue(report.failure_reason)
                with open(sentinel, encoding="utf-8") as handle:
                    self.assertEqual("OUTSIDE SENTINEL", handle.read())
            finally:
                manager.close()

    def test_unlink_oserror_returns_failed_purge_report_without_raising(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            session_id = "unlink-oserror"
            try:
                manager.open_session(session_id).append(CognitiveEvent(
                    session_id=session_id, kind=EventKind.OBSERVATION,
                    source_kind=SourceKind.USER, source_ref="user",
                    payload={"content": "must remain if unlink fails"}))
                database = manager.database_path(session_id)
                # A deterministic syscall failure is more portable than directory
                # mode 0500: same-user macOS setups may still permit unlink there.
                with patch("strangeloop.memory.os.unlink", side_effect=OSError("permission denied")):
                    report = manager.purge_session(session_id, confirmed=True)
                self.assertFalse(report.container_deleted)
                self.assertIn("deletion failed", report.failure_reason)
                self.assertTrue(database.exists())
            finally:
                manager.close()

    def test_purge_requires_confirmation_and_persistent_store_reopens_cleanly(self):
        with tempfile.NamedTemporaryFile() as handle:
            store = SQLiteEventStore(handle.name)
            agent = StrangeloopAgent(session_id="durable", event_store=store)
            agent.run_turn("persisted before purge")
            with self.assertRaises((PermissionError, ValueError)):
                agent.purge_session()
            self.assertTrue(store.list("durable"))
            agent.purge_session(confirmed=True)
            self.assertEqual([], store.list("durable"))
            store.close()
            reopened = SQLiteEventStore(handle.name)
            try:
                self.assertEqual([], reopened.list("durable"))
            finally:
                reopened.close()
