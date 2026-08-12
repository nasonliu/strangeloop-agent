import os
import hashlib
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone

from strangeloop.contracts import (CognitiveEvent, EventKind, SourceKind,
                                   FRONTIER_STRATEGY_ARM_VERSION,
                                   frontier_strategy_arm_id)
from strangeloop.store import (FRONTIER_CHANNEL_ORDER, SQLiteEventStore, canonical_json,
                               frontier_experiment_reward_vector)


class SQLiteEventStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteEventStore()

    def tearDown(self):
        self.store.close()

    def event(self, session, payload):
        return CognitiveEvent(session_id=session, kind=EventKind.OBSERVATION,
                              source_kind=SourceKind.USER, source_ref="test", payload=payload)

    def expedition_authorization(self, session="expedition", expires_delta=300):
        approval = self.store.append(self.event(session, {"content": "approve expedition", "channel": "test"}))
        now = datetime.fromisoformat(approval.created_at) + timedelta(seconds=1)
        event = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.EXPEDITION_AUTHORIZATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(), payload={
                "authorization_id": "expedition_auth_1", "approval_event_id": approval.event_id,
                "nonce": "nonce_1", "issued_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=expires_delta)).isoformat(),
                "goal_digest": "a" * 64, "host_seed_digest": "b" * 64,
                "max_calls_per_slice": 4, "slice_seconds": 60,
                "authorization_seconds": expires_delta, "profile": "public_web_only_v1",
                "version": "expedition_authorization_v1"}, parent_event_ids=(approval.event_id,)))
        return event, now

    def test_expedition_authorization_is_user_bound_and_consumed_once(self):
        authorization, now = self.expedition_authorization()
        payload = {"consumption_id": "expedition_consume_1", "authorization_id": "expedition_auth_1",
                   "consumed_at": (now + timedelta(seconds=1)).isoformat(), "run_id": "expedition_run_1",
                   "version": "expedition_authorization_consumed_v1"}
        self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
                          source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload=payload,
                          parent_event_ids=(authorization.event_id,)))
        duplicate = dict(payload, consumption_id="expedition_consume_2", run_id="expedition_run_2")
        with self.assertRaisesRegex(ValueError, "already been consumed"):
            self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
                              source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload=duplicate,
                              parent_event_ids=(authorization.event_id,)))

    def test_expedition_consumption_rejects_expired_or_wrong_parent(self):
        authorization, now = self.expedition_authorization(expires_delta=1)
        expired = {"consumption_id": "expedition_consume_1", "authorization_id": "expedition_auth_1",
                   "consumed_at": (now + timedelta(seconds=1)).isoformat(), "run_id": "expedition_run_1",
                   "version": "expedition_authorization_consumed_v1"}
        with self.assertRaisesRegex(ValueError, "authorization window"):
            self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
                              source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload=expired,
                              parent_event_ids=(authorization.event_id,)))

    def test_expedition_consumed_authorization_can_parent_only_read_grant(self):
        authorization, now = self.expedition_authorization()
        consumed = self.store.append(CognitiveEvent(
            session_id="expedition", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
            source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload={
                "consumption_id": "expedition_consume_1", "authorization_id": "expedition_auth_1",
                "consumed_at": (now + timedelta(seconds=1)).isoformat(), "run_id": "expedition_run_1",
                "version": "expedition_authorization_consumed_v1"}, parent_event_ids=(authorization.event_id,)))
        grant_payload = {"grant_id": "grant_expedition_1", "capability": "web_fetch", "tool_name": "web.fetch",
                         "scope_digest": "c" * 64, "not_before": (now + timedelta(seconds=1)).isoformat(),
                         "expires_at": (now + timedelta(seconds=20)).isoformat(), "max_uses": 1,
                         "max_input_bytes": 1, "max_output_bytes": 1, "max_wall_ms": 1,
                         "allow_mutating": False, "grant_version": "k3_v1"}
        self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.CAPABILITY_GRANTED,
                          source_kind=SourceKind.USER, source_ref="user", payload=grant_payload,
                          parent_event_ids=(authorization.event_id,)))
        self.assertIsNotNone(consumed)
        bad = dict(grant_payload, grant_id="grant_expedition_2", allow_mutating=True)
        with self.assertRaisesRegex(ValueError, "read-only public-web"):
            self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.CAPABILITY_GRANTED,
                              source_kind=SourceKind.USER, source_ref="user", payload=bad,
                              parent_event_ids=(authorization.event_id,)))

    def test_expedition_authorization_rejects_reused_nonce_and_wrong_window(self):
        authorization, now = self.expedition_authorization()
        duplicate = dict(authorization.payload, authorization_id="expedition_auth_2")
        with self.assertRaisesRegex(ValueError, "nonce already exists"):
            self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.EXPEDITION_AUTHORIZATION,
                              source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(),
                              payload=duplicate, parent_event_ids=(authorization.parent_event_ids[0],)))
        bad = dict(authorization.payload, authorization_id="expedition_auth_3", nonce="nonce_3",
                   expires_at=(now + timedelta(seconds=299)).isoformat())
        with self.assertRaisesRegex(ValueError, "equal authorization_seconds"):
            self.store.append(CognitiveEvent(session_id="expedition", kind=EventKind.EXPEDITION_AUTHORIZATION,
                              source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(),
                              payload=bad, parent_event_ids=(authorization.parent_event_ids[0],)))

    def test_frontier_vector_protocol_is_active_v2_only_and_has_fixed_channel_order(self):
        approval = self.store.append(self.event("frontier", {"content": "approve", "channel": "test"}))
        now = datetime.fromisoformat(approval.created_at) + timedelta(seconds=1)
        digest = "a" * 64
        authorization = self.store.append(CognitiveEvent(
            session_id="frontier", kind=EventKind.EXPEDITION_AUTHORIZATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(), payload={
                "authorization_id": "frontier_auth", "approval_event_id": approval.event_id, "nonce": "frontier_nonce",
                "issued_at": now.isoformat(), "expires_at": (now + timedelta(seconds=60)).isoformat(),
                "goal_digest": digest, "host_seed_digest": "b" * 64, "max_calls_per_slice": 1,
                "slice_seconds": 30, "authorization_seconds": 60, "profile": "public_web_only_v1",
                "version": "expedition_authorization_v2", "learning_mode": "active",
                "learner_spec_digest": "c" * 64}, parent_event_ids=(approval.event_id,)))
        consumed = self.store.append(CognitiveEvent(
            session_id="frontier", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
            source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload={
                "consumption_id": "frontier_consumed", "authorization_id": "frontier_auth",
                "consumed_at": (now + timedelta(seconds=1)).isoformat(), "run_id": "frontier_run",
                "version": "expedition_authorization_consumed_v1"}, parent_event_ids=(authorization.event_id,)))
        common = {"transition_id": "transition_1", "learner_spec_digest": "c" * 64,
                  "scope": "frontier_ranking_only"}
        ranking = self.store.append(CognitiveEvent(
            session_id="frontier", kind=EventKind.FRONTIER_RANKING_DECISION,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(common,
                ranking_id="ranking_1", frontier_task_id="task_1", candidate_digest="d" * 64,
                channel_order=list(FRONTIER_CHANNEL_ORDER), ranking_version="frontier_ranking_v1"),
            parent_event_ids=(consumed.event_id,)))
        vector = [.0, .0, .0, .0, .0]
        value = self.store.append(CognitiveEvent(
            session_id="frontier", kind=EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(common,
                estimate_id="estimate_1", ranking_event_id=ranking.event_id, channel_order=list(FRONTIER_CHANNEL_ORDER),
                value_vector=vector, alpha=.5, gamma=0.0, clip=1.0, formula_version="frontier_td0_vector_v1"),
            parent_event_ids=(ranking.event_id,)))
        external = self.store.append(self.event("frontier", {"content": "external finding", "channel": "test"}))
        evidence = self.store.append(CognitiveEvent(
            session_id="frontier", kind=EventKind.FRONTIER_EVIDENCE_OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload=dict(common,
                evidence_id="evidence_1", ranking_event_id=ranking.event_id, source_event_id=external.event_id,
                evidence_digest="e" * 64, evidence_kind="validated_external_evidence", evidence_version="frontier_evidence_v1"),
            parent_event_ids=(ranking.event_id, external.event_id)))
        with self.assertRaisesRegex(ValueError, "qualifying experiment result"):
            self.store.append(CognitiveEvent(
                session_id="frontier", kind=EventKind.FRONTIER_VECTOR_REWARD,
                source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(common,
                    reward_id="reward_1", ranking_event_id=ranking.event_id, evidence_event_id=evidence.event_id,
                    value_event_id=value.event_id, channel_order=list(FRONTIER_CHANNEL_ORDER),
                    reward_vector=[.2, .3, .0, .4, .1], reward_version="frontier_reward_v1"),
                parent_event_ids=(evidence.event_id, value.event_id)))
        self.assertEqual(list(FRONTIER_CHANNEL_ORDER), ranking.payload["channel_order"])
        with self.assertRaisesRegex(ValueError, "fixed v1 sequence"):
            self.store.append(CognitiveEvent(session_id="frontier", kind=EventKind.FRONTIER_RANKING_DECISION,
                source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(common,
                    ranking_id="ranking_bad", frontier_task_id="task_bad", candidate_digest="f" * 64,
                    channel_order=list(reversed(FRONTIER_CHANNEL_ORDER)), ranking_version="frontier_ranking_v1"),
                parent_event_ids=(consumed.event_id,)))

    def test_frontier_experiment_projection_is_fixed_and_status_sensitive(self):
        self.assertEqual([.4, .7, 1.0, .8, .0],
                         frontier_experiment_reward_vector("frontier_replay", "supported", True))
        self.assertEqual([.1, .0, 1.0, .8, .0],
                         frontier_experiment_reward_vector("duplicate_suppression", "refuted", False))
        self.assertEqual([.4, .7, 1.0, .3, .0],
                         frontier_experiment_reward_vector("td_invariants", "inconclusive", True))
        with self.assertRaises(ValueError):
            frontier_experiment_reward_vector("frontier_replay", "invalid", True)

    def test_offline_experiment_requires_v3_lock_execution_and_independent_valid_result(self):
        approval = self.store.append(self.event("experiment", {"content": "approve", "channel": "test"}))
        now = datetime.fromisoformat(approval.created_at) + timedelta(seconds=1)
        authorization = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.EXPEDITION_AUTHORIZATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(), payload={
                "authorization_id": "auth_exp", "approval_event_id": approval.event_id, "nonce": "nonce_exp",
                "issued_at": now.isoformat(), "expires_at": (now + timedelta(seconds=60)).isoformat(),
                "goal_digest": "a" * 64, "host_seed_digest": "b" * 64, "max_calls_per_slice": 1,
                "slice_seconds": 30, "authorization_seconds": 60, "profile": "public_web_only_v1",
                "version": "expedition_authorization_v3", "learning_mode": "active",
                "learner_spec_digest": "c" * 64, "experiment_kinds": ["frontier_replay"],
                "experiment_registry_digest": "d" * 64, "max_experiments": 1,
                "experiment_max_trials": 4, "experiment_max_steps": 100, "experiment_max_wall_ms": 500}, parent_event_ids=(approval.event_id,)))
        consumed = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
            source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload={"consumption_id": "consume_exp",
                "authorization_id": "auth_exp", "consumed_at": (now + timedelta(seconds=1)).isoformat(),
                "run_id": "run_exp", "version": "expedition_authorization_consumed_v1"}, parent_event_ids=(authorization.event_id,)))
        plan_payload = {"experiment_id": "exp_1", "authorization_id": "auth_exp", "learner_spec_digest": "c" * 64,
            "experiment_registry_digest": "d" * 64, "experiment_instance_digest": "8" * 64, "experiment_kind": "frontier_replay", "hypothesis_digest": "e" * 64,
            "baseline_digest": "f" * 64, "treatment_digest": "0" * 64, "input_digest": "1" * 64,
            "evaluator_digest": "2" * 64, "host_seed_digest": "3" * 64, "budget_digest": "4" * 64,
            "max_trials": 2, "max_steps": 20, "max_wall_ms": 100, "no_external_side_effects": True,
            "claim_scope": "offline_simulation", "template_version": "experiment_template_v1"}
        plan = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.EXPERIMENT_PLAN_LOCKED,
            source_kind=SourceKind.SYSTEM, source_ref="ExperimentCoordinator", payload=plan_payload,
            parent_event_ids=(consumed.event_id,)))
        started = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.EXPERIMENT_EXECUTION_STARTED,
            source_kind=SourceKind.SYSTEM, source_ref="ExperimentHarness", payload={"execution_id": "exec_1",
                "experiment_id": "exp_1", "plan_event_id": plan.event_id, "experiment_registry_digest": "d" * 64, "experiment_instance_digest": "8" * 64,
                "execution_nonce": "execution_nonce_1", "started_at": (now + timedelta(seconds=2)).isoformat(),
                "max_trials": 2, "max_steps": 20, "max_wall_ms": 100, "execution_version": "experiment_execution_v1"},
            parent_event_ids=(plan.event_id,)))
        result_payload = {"result_id": "result_1", "execution_id": "exec_1", "experiment_id": "exp_1",
            "plan_event_id": plan.event_id, "authorization_id": "auth_exp", "learner_spec_digest": "c" * 64,
            "experiment_registry_digest": "d" * 64, "experiment_instance_digest": "8" * 64, "status": "supported", "complete_run_set": True,
            "baseline_control_valid": True, "reproducible": True, "metric_digest": "5" * 64,
            "metric_values": [1.0], "effect_values": [.5], "result_digest": "6" * 64,
            "reproduction_digest": "7" * 64, "invariant_codes": ["deterministic_replay"],
            "claim_scope": "offline_simulation", "result_version": "experiment_result_v1"}
        result = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.EXPERIMENT_RESULT,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="IndependentExperimentVerifier", payload=result_payload,
            parent_event_ids=(started.event_id,)))
        self.assertEqual(EventKind.EXPERIMENT_RESULT, result.kind)
        arm = frontier_strategy_arm_id("d" * 64, "c" * 64, "frontier_replay")
        arm_common = {"transition_id": "exp_transition", "learner_spec_digest": "c" * 64,
                      "scope": "frontier_ranking_only", "strategy_arm_id": arm,
                      "strategy_arm_version": FRONTIER_STRATEGY_ARM_VERSION}
        ranking = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.FRONTIER_RANKING_DECISION,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(arm_common, ranking_id="exp_ranking",
                frontier_task_id="different_task_instance", candidate_digest="9" * 64,
                channel_order=list(FRONTIER_CHANNEL_ORDER), ranking_version="frontier_ranking_v2"), parent_event_ids=(consumed.event_id,)))
        value = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(arm_common, estimate_id="exp_value",
                ranking_event_id=ranking.event_id, channel_order=list(FRONTIER_CHANNEL_ORDER), value_vector=[0.0] * 5,
                alpha=.5, gamma=0.0, clip=1.0, formula_version="frontier_td0_vector_v2"), parent_event_ids=(ranking.event_id,)))
        evidence = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.FRONTIER_EVIDENCE_OBSERVATION,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="IndependentExperimentVerifier", payload=dict(arm_common,
                evidence_id="exp_evidence", ranking_event_id=ranking.event_id, source_event_id=result.event_id,
                evidence_digest="9" * 64, evidence_kind="validated_external_evidence", evidence_version="frontier_evidence_v2"),
            parent_event_ids=(ranking.event_id, result.event_id)))
        reward_payload = dict(arm_common, reward_id="exp_reward", ranking_event_id=ranking.event_id,
            evidence_event_id=evidence.event_id, value_event_id=value.event_id, channel_order=list(FRONTIER_CHANNEL_ORDER),
            reward_vector=[.4, .7, 1.0, .8, .0], reward_version="frontier_reward_v2")
        with self.assertRaisesRegex(ValueError, "strategy arm must match its input parents"):
            self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.FRONTIER_VECTOR_REWARD,
                source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(reward_payload, strategy_arm_id="f" * 64),
                parent_event_ids=(evidence.event_id, value.event_id)))
        reward = self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.FRONTIER_VECTOR_REWARD,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=reward_payload,
            parent_event_ids=(evidence.event_id, value.event_id)))
        self.assertEqual(arm, reward.payload["strategy_arm_id"])
        bad = dict(result_payload, result_id="result_bad", reproducible=False)
        with self.assertRaisesRegex(ValueError, "verifier result"):
            self.store.append(CognitiveEvent(session_id="experiment", kind=EventKind.EXPERIMENT_RESULT,
                source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="IndependentExperimentVerifier", payload=bad,
                parent_event_ids=(started.event_id,)))

    def test_frontier_v2_strategy_arm_is_stable_and_parent_bound(self):
        approval = self.store.append(self.event("arm", {"content": "approve", "channel": "test"}))
        now = datetime.fromisoformat(approval.created_at) + timedelta(seconds=1)
        registry_digest, learner_digest = "a" * 64, "b" * 64
        authorization = self.store.append(CognitiveEvent(session_id="arm", kind=EventKind.EXPEDITION_AUTHORIZATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(), payload={
                "authorization_id": "arm_auth", "approval_event_id": approval.event_id, "nonce": "arm_nonce",
                "issued_at": now.isoformat(), "expires_at": (now + timedelta(seconds=60)).isoformat(),
                "goal_digest": "c" * 64, "host_seed_digest": "d" * 64, "max_calls_per_slice": 1,
                "slice_seconds": 30, "authorization_seconds": 60, "profile": "public_web_only_v1",
                "version": "expedition_authorization_v3", "learning_mode": "active", "learner_spec_digest": learner_digest,
                "experiment_kinds": ["frontier_replay"], "experiment_registry_digest": registry_digest,
                "max_experiments": 1, "experiment_max_trials": 4, "experiment_max_steps": 100,
                "experiment_max_wall_ms": 500}, parent_event_ids=(approval.event_id,)))
        consumed = self.store.append(CognitiveEvent(session_id="arm", kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
            source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler", payload={"consumption_id": "arm_consumed",
                "authorization_id": "arm_auth", "consumed_at": (now + timedelta(seconds=1)).isoformat(), "run_id": "arm_run",
                "version": "expedition_authorization_consumed_v1"}, parent_event_ids=(authorization.event_id,)))
        arm = frontier_strategy_arm_id(registry_digest, learner_digest, "frontier_replay")
        self.assertEqual(arm, frontier_strategy_arm_id(registry_digest, learner_digest, "frontier_replay"))
        common = {"transition_id": "arm_transition", "learner_spec_digest": learner_digest, "scope": "frontier_ranking_only",
                  "strategy_arm_id": arm, "strategy_arm_version": FRONTIER_STRATEGY_ARM_VERSION}
        ranking = self.store.append(CognitiveEvent(session_id="arm", kind=EventKind.FRONTIER_RANKING_DECISION,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(common, ranking_id="arm_ranking",
                frontier_task_id="task_instance_one", candidate_digest="e" * 64, channel_order=list(FRONTIER_CHANNEL_ORDER),
                ranking_version="frontier_ranking_v2"), parent_event_ids=(consumed.event_id,)))
        value_payload = dict(common, estimate_id="arm_value", ranking_event_id=ranking.event_id,
            channel_order=list(FRONTIER_CHANNEL_ORDER), value_vector=[0.0] * 5, alpha=.5, gamma=0.0, clip=1.0,
            formula_version="frontier_td0_vector_v2")
        with self.assertRaisesRegex(ValueError, "strategy arm must match its ranking parent"):
            self.store.append(CognitiveEvent(session_id="arm", kind=EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=dict(value_payload, strategy_arm_id="f" * 64),
                parent_event_ids=(ranking.event_id,)))
        value = self.store.append(CognitiveEvent(session_id="arm", kind=EventKind.FRONTIER_VECTOR_VALUE_ESTIMATE,
            source_kind=SourceKind.SYSTEM, source_ref="FrontierLearner", payload=value_payload, parent_event_ids=(ranking.event_id,)))
        self.assertEqual(arm, value.payload["strategy_arm_id"])

    def sleep_wake_lifecycle(self, session="sleep_session", terminal=True, auto_policy=False):
        """Append the complete metadata-only v1 lifecycle for one fresh epoch."""
        base = datetime.now(timezone.utc).replace(microsecond=0)
        digest = "a" * 64
        policy = None
        if auto_policy:
            policy = self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.AUTO_WAKE_POLICY,
                source_kind=SourceKind.USER, source_ref="user", created_at=(base - timedelta(seconds=2)).isoformat(), payload={
                    "policy_id": "policy_1", "action": "enable", "scope": "next_sleep_epoch",
                    "max_auto_runs": 1, "max_ticks": 4, "expires_at": (base + timedelta(hours=1)).isoformat(),
                    "issued_at": (base - timedelta(seconds=2)).isoformat(), "policy_version": "auto_wake_policy_v1"}))
        archive_payload = {
            "archive_id": "archive_1", "epoch_id": "epoch_1", "schema_version": 2,
            "chain_head_sequence": 0 if policy is None else 1,
            "chain_head_hash": "0" * 64 if policy is None else policy.content_hash,
            "approved_seed_ids": [], "approved_claim_ids": [], "pending_task_ids": ["task_1"],
            "quota_source": "provider_usage", "quota_observed_at": (base - timedelta(seconds=30)).isoformat(),
            "quota_reset_at": (base + timedelta(hours=1)).isoformat(), "quota_remaining": 10,
            "quota_total": 100, "event_count": 0 if policy is None else 1,
            "archive_version": "sleep_archive_v2",
        }
        archive_payload["archive_digest"] = SQLiteEventStore._archive_digest_v2(archive_payload)
        archive = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SLEEP_ARCHIVE,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
            created_at=base.isoformat(), payload=archive_payload))
        sleeping = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SLEEP_ENTERED,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
            created_at=(base + timedelta(seconds=1)).isoformat(), payload={
                "sleep_id": "sleep_1", "archive_event_id": archive.event_id,
                "epoch_id": "epoch_1", "auto_wake_policy_event_id": None if policy is None else policy.event_id,
                "slept_at": (base + timedelta(seconds=1)).isoformat(),
                "protocol_version": "sleep_wake_v1",
            }, parent_event_ids=(archive.event_id,) if policy is None else (archive.event_id, policy.event_id)))
        evidence = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.PROVIDER_USAGE_EVIDENCE,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier",
            created_at=(base + timedelta(seconds=2)).isoformat(), payload={
                "usage_evidence_id": "usage_1", "provider": "fixture_provider",
                "quota_source": "provider_usage", "observed_at": (base + timedelta(seconds=2)).isoformat(),
                "reset_at": (base + timedelta(hours=1)).isoformat(), "total": 100,
                "remaining": 90, "evidence_digest": digest,
                "evidence_version": "provider_usage_v1",
            }, parent_event_ids=(sleeping.event_id,)))
        check = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.WAKE_CHECK,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
            created_at=(base + timedelta(seconds=3)).isoformat(), payload={
                "wake_check_id": "check_1", "archive_event_id": archive.event_id,
                "sleep_event_id": sleeping.event_id, "epoch_id": "epoch_1",
                "usage_evidence_event_id": evidence.event_id,
                "checked_at": (base + timedelta(seconds=3)).isoformat(),
                "check_version": "wake_check_v1",
            }, parent_event_ids=(sleeping.event_id, evidence.event_id)))
        ready = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.WAKE_READY,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
            created_at=(base + timedelta(seconds=4)).isoformat(), payload={
                "wake_ready_id": "ready_1", "wake_check_event_id": check.event_id,
                "epoch_id": "epoch_1", "ready_at": (base + timedelta(seconds=4)).isoformat(),
                "protocol_version": "sleep_wake_v1",
            }, parent_event_ids=(check.event_id,)))
        awake = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.AWAKE,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
            created_at=(base + timedelta(seconds=5)).isoformat(), payload={
                "awake_id": "awake_1", "wake_ready_event_id": ready.event_id,
                "epoch_id": "epoch_1", "awakened_at": (base + timedelta(seconds=5)).isoformat(),
                "protocol_version": "sleep_wake_v1",
            }, parent_event_ids=(ready.event_id,)))
        terminal_event = None
        if terminal:
            terminal_event = self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.WAKE_TERMINAL,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator",
                created_at=(base + timedelta(seconds=6)).isoformat(), payload={
                    "wake_terminal_id": "terminal_1", "lifecycle_event_id": awake.event_id,
                    "epoch_id": "epoch_1", "terminal_at": (base + timedelta(seconds=6)).isoformat(),
                    "outcome": "completed", "protocol_version": "sleep_wake_v1",
                }, parent_event_ids=(awake.event_id,)))
        return base, archive, sleeping, evidence, check, ready, awake, terminal_event

    def test_system_autorun_requires_one_current_user_policy_bound_awake(self):
        _, _, _, _, _, _, awake, _ = self.sleep_wake_lifecycle("autorun", terminal=False, auto_policy=True)
        payload = {"run_id": "wake_run", "action": "start", "config_version": "auto_wake_v1",
                   "max_ticks": 4, "max_wall_seconds": 30, "max_events_per_tick": 1,
                   "max_no_progress": 2, "min_interval_seconds": 0.0, "next_tick_index": 0}
        started = self.store.append(CognitiveEvent(session_id="autorun", kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=payload,
            parent_event_ids=(awake.event_id,)))
        self.assertEqual("wake_run", started.payload["run_id"])
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="autorun", kind=EventKind.AUTONOMY_CONTROL,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=dict(payload, run_id="wake_run_2"),
                parent_event_ids=(awake.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="autorun", kind=EventKind.AUTONOMY_CONTROL,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=dict(payload, run_id="oversize", max_ticks=5),
                parent_event_ids=(awake.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="autorun", kind=EventKind.AUTONOMY_CONTROL,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=dict(payload, run_id="no_parent")))

    def test_sleep_archive_v2_recomputes_and_survives_reopen_while_corruption_rejects(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        try:
            store = SQLiteEventStore(handle.name)
            base = datetime.now(timezone.utc).replace(microsecond=0)
            payload = {"archive_id": "archive_v2", "epoch_id": "epoch_v2", "schema_version": 2,
                "chain_head_sequence": 0, "chain_head_hash": "0" * 64, "approved_seed_ids": ["seed_a"],
                "approved_claim_ids": ["claim_a"], "pending_task_ids": ["task_a"], "quota_source": "provider_usage",
                "quota_observed_at": base.isoformat(), "quota_reset_at": (base + timedelta(hours=1)).isoformat(),
                "quota_remaining": 1, "quota_total": 10, "event_count": 0, "archive_version": "sleep_archive_v2"}
            payload["archive_digest"] = SQLiteEventStore._archive_digest_v2(payload)
            stored = store.append(CognitiveEvent(session_id="archive_v2", kind=EventKind.SLEEP_ARCHIVE,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=payload))
            store.close()
            reopened = SQLiteEventStore(handle.name)
            recovered = reopened.get(stored.event_id)
            self.assertEqual(payload["archive_digest"], SQLiteEventStore._archive_digest_v2(recovered.payload))
            corrupt = dict(payload, archive_id="archive_corrupt", epoch_id="epoch_corrupt", quota_total=11)
            with self.assertRaises(ValueError):
                reopened.append(CognitiveEvent(session_id="other_archive", kind=EventKind.SLEEP_ARCHIVE,
                    source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=corrupt))
            reopened.close()
        finally:
            os.unlink(handle.name)

    def test_sleep_wake_lifecycle_requires_fresh_provider_evidence_and_new_epoch(self):
        _, archive, sleeping, evidence, check, ready, awake, terminal = self.sleep_wake_lifecycle()
        self.assertTrue(self.store.verify_chain("sleep_session"))
        self.assertEqual((archive.kind, sleeping.kind, evidence.kind, check.kind,
                          ready.kind, awake.kind, terminal.kind),
                         (EventKind.SLEEP_ARCHIVE, EventKind.SLEEP_ENTERED,
                          EventKind.PROVIDER_USAGE_EVIDENCE, EventKind.WAKE_CHECK,
                          EventKind.WAKE_READY, EventKind.AWAKE, EventKind.WAKE_TERMINAL))
        duplicate = dict(check.payload)
        duplicate["wake_check_id"] = "check_2"
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="sleep_session", kind=EventKind.WAKE_CHECK,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload=duplicate,
                parent_event_ids=(sleeping.event_id, evidence.event_id)))

    def test_rolling_v3_archive_and_usage_digest_are_tamper_evident(self):
        base = datetime.now(timezone.utc).replace(microsecond=0)
        archive_payload = {
            "archive_id": "rolling_archive", "epoch_id": "rolling_epoch", "schema_version": 3,
            "chain_head_sequence": 0, "chain_head_hash": "0" * 64, "approved_seed_ids": [],
            "approved_claim_ids": [], "pending_task_ids": [], "quota_source": "provider_usage",
            "quota_observed_at": base.isoformat(), "quota_reset_at": (base + timedelta(hours=5)).isoformat(),
            "quota_remaining": 1, "quota_total": 100, "quota_window_kind": "rolling_5h",
            "event_count": 0, "archive_version": "sleep_archive_v3",
        }
        archive_payload["archive_digest"] = SQLiteEventStore._archive_digest(archive_payload)
        archive = self.store.append(CognitiveEvent(
            session_id="rolling_v3", kind=EventKind.SLEEP_ARCHIVE, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=base.isoformat(), payload=archive_payload))
        sleeping = self.store.append(CognitiveEvent(
            session_id="rolling_v3", kind=EventKind.SLEEP_ENTERED, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=1)).isoformat(), payload={
                "sleep_id": "rolling_sleep", "archive_event_id": archive.event_id, "epoch_id": "rolling_epoch",
                "auto_wake_policy_event_id": None, "slept_at": (base + timedelta(seconds=1)).isoformat(),
                "protocol_version": "sleep_wake_v1"}, parent_event_ids=(archive.event_id,)))
        evidence_payload = {
            "usage_evidence_id": "rolling_usage", "provider": "fixture", "quota_source": "provider_usage",
            "window_kind": "rolling_5h", "observed_at": (base + timedelta(seconds=2)).isoformat(),
            "reset_at": (base + timedelta(hours=5)).isoformat(), "total": 100, "remaining": 100,
            "evidence_version": "provider_usage_v2",
        }
        evidence_payload["evidence_digest"] = SQLiteEventStore._provider_usage_digest_v2(evidence_payload)
        evidence = self.store.append(CognitiveEvent(
            session_id="rolling_v3", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier",
            created_at=(base + timedelta(seconds=2)).isoformat(), payload=evidence_payload,
            parent_event_ids=(sleeping.event_id,)))
        check = self.store.append(CognitiveEvent(
            session_id="rolling_v3", kind=EventKind.WAKE_CHECK, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=3)).isoformat(), payload={
                "wake_check_id": "rolling_check", "archive_event_id": archive.event_id,
                "sleep_event_id": sleeping.event_id, "epoch_id": "rolling_epoch",
                "usage_evidence_event_id": evidence.event_id, "checked_at": (base + timedelta(seconds=3)).isoformat(),
                "check_version": "wake_check_v1"}, parent_event_ids=(sleeping.event_id, evidence.event_id)))
        ready = self.store.append(CognitiveEvent(
            session_id="rolling_v3", kind=EventKind.WAKE_READY, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=4)).isoformat(), payload={
                "wake_ready_id": "rolling_ready", "wake_check_event_id": check.event_id, "epoch_id": "rolling_epoch",
                "ready_at": (base + timedelta(seconds=4)).isoformat(), "protocol_version": "sleep_wake_v2"},
            parent_event_ids=(check.event_id,)))
        self.assertEqual(EventKind.WAKE_READY, ready.kind)
        tampered = dict(evidence_payload, usage_evidence_id="rolling_usage_bad", remaining=99)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="rolling_v3", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
                source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier", payload=tampered,
                parent_event_ids=(sleeping.event_id,)))
        legacy_downgrade = dict(evidence_payload, usage_evidence_id="rolling_usage_v1")
        legacy_downgrade.pop("window_kind")
        legacy_downgrade["evidence_version"] = "provider_usage_v1"
        legacy_downgrade["evidence_digest"] = "a" * 64
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="rolling_v3", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
                source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier", payload=legacy_downgrade,
                parent_event_ids=(sleeping.event_id,)))

    def test_sleep_wake_rejects_malformed_untrusted_stale_and_cross_session_records(self):
        base, archive, sleeping, _, _, _, _, _ = self.sleep_wake_lifecycle(terminal=False)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="sleep_session", kind=EventKind.SLEEP_ARCHIVE,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={"prompt": "forbidden"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="sleep_session", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
                source_kind=SourceKind.TOOL, source_ref="ProviderUsageVerifier", payload={}))
        stale = self.store.append(CognitiveEvent(
            session_id="sleep_session", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier",
            created_at=(base + timedelta(seconds=7)).isoformat(), payload={
                "usage_evidence_id": "usage_old", "provider": "fixture_provider",
                "quota_source": "provider_usage", "observed_at": archive.payload["quota_observed_at"],
                "reset_at": (base + timedelta(hours=1)).isoformat(), "total": 100,
                "remaining": 90, "evidence_digest": "b" * 64,
                "evidence_version": "provider_usage_v1",
            }, parent_event_ids=(sleeping.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="sleep_session", kind=EventKind.WAKE_CHECK,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={
                    "wake_check_id": "old_check", "archive_event_id": archive.event_id,
                    "sleep_event_id": sleeping.event_id, "epoch_id": "epoch_1",
                    "usage_evidence_event_id": stale.event_id,
                    "checked_at": (base + timedelta(seconds=8)).isoformat(), "check_version": "wake_check_v1",
                }, parent_event_ids=(sleeping.event_id, stale.event_id)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="other", kind=EventKind.SLEEP_ENTERED,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={
                    "sleep_id": "cross_sleep", "archive_event_id": archive.event_id,
                    "epoch_id": "epoch_1", "slept_at": (base + timedelta(seconds=8)).isoformat(),
                    "protocol_version": "sleep_wake_v1",
                }, parent_event_ids=(archive.event_id,)))

    def test_sleep_wake_records_cannot_be_td_or_durable_claim_evidence(self):
        _, archive, _, _, _, _, awake, _ = self.sleep_wake_lifecycle()
        value = {
            "estimate_id": "estimate_sleep", "transition_id": "transition_sleep",
            "target_event_id": awake.event_id, "state_key": "state_sleep", "action_key": "respond",
            "next_state_key": "next_sleep", "terminal": True, "value": 0.0,
            "confidence": 1.0, "estimator_id": "fixture", "estimator_version": "v1",
            "alpha": 0.1, "gamma": 0.0, "clip": 1.0, "max_entries": 1,
            "max_events": 1, "formula_version": "td0_v1",
        }
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="sleep_session", kind=EventKind.VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="td", payload=value,
                parent_event_ids=(awake.event_id,)))
        root = self.store.append(self.event("sleep_session", {"content": "normal evidence"}))
        seed = {"cue_terms": ["cue"], "policy_bias": "bounded", "scope": "test",
                "provenance_event_ids": [archive.event_id], "strength": 0.5, "confidence": 0.5,
                "status": "candidate", "seed_id": "seed_sleep", "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00", "expires_at": None,
                "version": 1, "counterevidence": 0}
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="sleep_session", kind=EventKind.SEED_PROPOSED,
                source_kind=SourceKind.MODEL, source_ref="model", payload={"seed": seed},
                parent_event_ids=(root.event_id,)))

    def test_wake_ready_rejects_zero_quota_and_terminal_can_stop_sleeping_epoch(self):
        base = datetime.now(timezone.utc).replace(microsecond=0)
        archive_payload = {"archive_id": "archive_zero", "epoch_id": "epoch_zero", "schema_version": 2,
            "chain_head_sequence": 0, "chain_head_hash": "0" * 64, "approved_seed_ids": [],
            "approved_claim_ids": [], "pending_task_ids": [], "quota_source": "provider_usage",
            "quota_observed_at": (base - timedelta(seconds=2)).isoformat(),
            "quota_reset_at": (base + timedelta(hours=1)).isoformat(), "quota_remaining": 0,
            "quota_total": 10, "event_count": 0, "archive_version": "sleep_archive_v2"}
        archive_payload["archive_digest"] = SQLiteEventStore._archive_digest_v2(archive_payload)
        archive = self.store.append(CognitiveEvent(
            session_id="zero_quota", kind=EventKind.SLEEP_ARCHIVE,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", created_at=base.isoformat(), payload=archive_payload))
        sleeping = self.store.append(CognitiveEvent(
            session_id="zero_quota", kind=EventKind.SLEEP_ENTERED,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={
                "sleep_id": "sleep_zero", "archive_event_id": archive.event_id, "epoch_id": "epoch_zero", "auto_wake_policy_event_id": None,
                "slept_at": (base + timedelta(seconds=1)).isoformat(), "protocol_version": "sleep_wake_v1"},
            parent_event_ids=(archive.event_id,)))
        evidence = self.store.append(CognitiveEvent(
            session_id="zero_quota", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier", payload={
                "usage_evidence_id": "usage_zero", "provider": "fixture", "quota_source": "provider_usage",
                "observed_at": (base + timedelta(seconds=2)).isoformat(), "reset_at": (base + timedelta(hours=1)).isoformat(),
                "total": 10, "remaining": 0, "evidence_digest": "b" * 64, "evidence_version": "provider_usage_v1"},
            parent_event_ids=(sleeping.event_id,)))
        check = self.store.append(CognitiveEvent(
            session_id="zero_quota", kind=EventKind.WAKE_CHECK,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={
                "wake_check_id": "check_zero", "archive_event_id": archive.event_id,
                "sleep_event_id": sleeping.event_id, "epoch_id": "epoch_zero",
                "usage_evidence_event_id": evidence.event_id, "checked_at": (base + timedelta(seconds=3)).isoformat(),
                "check_version": "wake_check_v1"}, parent_event_ids=(sleeping.event_id, evidence.event_id)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="zero_quota", kind=EventKind.WAKE_READY,
                source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={
                    "wake_ready_id": "ready_zero", "wake_check_event_id": check.event_id, "epoch_id": "epoch_zero",
                    "ready_at": (base + timedelta(seconds=4)).isoformat(), "protocol_version": "sleep_wake_v1"},
                parent_event_ids=(check.event_id,)))
        terminal = self.store.append(CognitiveEvent(
            session_id="zero_quota", kind=EventKind.WAKE_TERMINAL,
            source_kind=SourceKind.SYSTEM, source_ref="SleepWakeCoordinator", payload={
                "wake_terminal_id": "terminal_zero", "lifecycle_event_id": sleeping.event_id,
                "epoch_id": "epoch_zero", "terminal_at": (base + timedelta(seconds=4)).isoformat(),
                "outcome": "stopped", "protocol_version": "sleep_wake_v1"}, parent_event_ids=(sleeping.event_id,)))
        self.assertEqual(EventKind.WAKE_TERMINAL, terminal.kind)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="zero_quota", kind=EventKind.PROVIDER_USAGE_EVIDENCE,
                source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier", payload={
                    "usage_evidence_id": "usage_late", "provider": "fixture", "quota_source": "provider_usage",
                    "observed_at": (base + timedelta(seconds=5)).isoformat(), "reset_at": (base + timedelta(hours=1)).isoformat(),
                    "total": 10, "remaining": 10, "evidence_digest": "c" * 64, "evidence_version": "provider_usage_v1"},
                parent_event_ids=(sleeping.event_id,)))

    def test_one_shot_unattended_wake_continuation_is_user_bound_and_terminal(self):
        session = "continuation"
        base = datetime.now(timezone.utc).replace(microsecond=0)
        approval = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
            source_ref="user", created_at=(base - timedelta(seconds=3)).isoformat(),
            payload={"content": "Approve one bounded public research continuation."}))
        auto = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.AUTO_WAKE_POLICY, source_kind=SourceKind.USER,
            source_ref="user", created_at=(base - timedelta(seconds=2)).isoformat(), payload={
                "policy_id": "auto_policy", "action": "enable", "scope": "next_sleep_epoch",
                "max_auto_runs": 1, "max_ticks": 4,
                "expires_at": (base + timedelta(hours=1)).isoformat(),
                "issued_at": (base - timedelta(seconds=2)).isoformat(),
                "policy_version": "auto_wake_policy_v1"}))
        continuation_payload = {
            "continuation_id": "continuation_1", "action": "enable",
            "scope": "next_sleep_epoch_read_only_research", "approval_event_id": approval.event_id,
            "unattended_policy_id": "research_policy_1", "focus_digest": "c" * 64,
            "profile_digest": "d" * 64, "max_auto_runs": 1,
            "expires_at": (base + timedelta(hours=1)).isoformat(),
            "issued_at": (base - timedelta(seconds=1)).isoformat(),
            "policy_version": "unattended_wake_continuation_policy_v1",
        }
        continuation = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(base - timedelta(seconds=1)).isoformat(), payload=continuation_payload,
            parent_event_ids=(approval.event_id,)))
        archive_payload = {
            "archive_id": "archive_continue", "epoch_id": "epoch_continue", "schema_version": 2,
            "chain_head_sequence": 3, "chain_head_hash": continuation.content_hash,
            "approved_seed_ids": [], "approved_claim_ids": [], "pending_task_ids": [],
            "quota_source": "provider_usage", "quota_observed_at": (base - timedelta(seconds=20)).isoformat(),
            "quota_reset_at": (base + timedelta(hours=1)).isoformat(), "quota_remaining": 1,
            "quota_total": 10, "event_count": 3, "archive_version": "sleep_archive_v2",
        }
        archive_payload["archive_digest"] = SQLiteEventStore._archive_digest_v2(archive_payload)
        archive = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SLEEP_ARCHIVE, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=base.isoformat(), payload=archive_payload))
        sleep = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SLEEP_ENTERED, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=1)).isoformat(), payload={
                "sleep_id": "sleep_continue", "archive_event_id": archive.event_id,
                "epoch_id": "epoch_continue", "auto_wake_policy_event_id": auto.event_id,
                "continuation_policy_event_id": continuation.event_id,
                "slept_at": (base + timedelta(seconds=1)).isoformat(), "protocol_version": "sleep_wake_v1",
            }, parent_event_ids=(archive.event_id, auto.event_id, continuation.event_id)))
        evidence = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.PROVIDER_USAGE_EVIDENCE,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="ProviderUsageVerifier",
            created_at=(base + timedelta(seconds=2)).isoformat(), payload={
                "usage_evidence_id": "usage_continue", "provider": "fixture",
                "quota_source": "provider_usage", "observed_at": (base + timedelta(seconds=2)).isoformat(),
                "reset_at": (base + timedelta(hours=1)).isoformat(), "total": 10, "remaining": 9,
                "evidence_digest": "e" * 64, "evidence_version": "provider_usage_v1",
            }, parent_event_ids=(sleep.event_id,)))
        check = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.WAKE_CHECK, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=3)).isoformat(), payload={
                "wake_check_id": "check_continue", "archive_event_id": archive.event_id,
                "sleep_event_id": sleep.event_id, "epoch_id": "epoch_continue",
                "usage_evidence_event_id": evidence.event_id,
                "checked_at": (base + timedelta(seconds=3)).isoformat(), "check_version": "wake_check_v1",
            }, parent_event_ids=(sleep.event_id, evidence.event_id)))
        ready = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.WAKE_READY, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=4)).isoformat(), payload={
                "wake_ready_id": "ready_continue", "wake_check_event_id": check.event_id,
                "epoch_id": "epoch_continue", "ready_at": (base + timedelta(seconds=4)).isoformat(),
                "protocol_version": "sleep_wake_v1",
            }, parent_event_ids=(check.event_id,)))
        awake = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.AWAKE, source_kind=SourceKind.SYSTEM,
            source_ref="SleepWakeCoordinator", created_at=(base + timedelta(seconds=5)).isoformat(), payload={
                "awake_id": "awake_continue", "wake_ready_event_id": ready.event_id,
                "epoch_id": "epoch_continue", "awakened_at": (base + timedelta(seconds=5)).isoformat(),
                "protocol_version": "sleep_wake_v1",
            }, parent_event_ids=(ready.event_id,)))
        invocation = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.MODEL_INVOCATION, source_kind=SourceKind.SYSTEM,
            source_ref="kimi", created_at=(base + timedelta(seconds=5)).isoformat(), payload={
                "invocation_id": "continuation_planning", "run_id": None,
                "trigger_event_id": awake.event_id, "provider": "fixture", "model": "fixture",
                "role": "tool_planning", "outcome": "completed", "latency_ms": 1,
                "context_scope": "tool_planning", "public_summary": "Bounded continuation planning.",
            }, parent_event_ids=(awake.event_id,)))
        self.assertEqual(awake.event_id, invocation.payload["trigger_event_id"])
        run_payload = {"continuation_id": "continuation_1", "awake_event_id": awake.event_id,
                       "profile_id": "profile_1", "authorization_policy_id": "research_policy_1",
                       "runtime_policy_id": "runtime_policy_1", "run_index": 1,
                       "status": "completed", "protocol_version": "unattended_wake_run_v1"}
        run = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.UNATTENDED_WAKE_RUN, source_kind=SourceKind.SYSTEM,
            source_ref="UnattendedResearchController", created_at=(base + timedelta(seconds=6)).isoformat(), payload=run_payload,
            parent_event_ids=(awake.event_id, continuation.event_id)))
        self.assertEqual(EventKind.UNATTENDED_WAKE_RUN, run.kind)
        self.assertTrue(self.store.verify_chain(session))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.UNATTENDED_WAKE_RUN, source_kind=SourceKind.SYSTEM,
                source_ref="UnattendedResearchController", created_at=(base + timedelta(seconds=7)).isoformat(), payload=dict(run_payload, status="started"),
                parent_event_ids=(awake.event_id, continuation.event_id)))

    def test_continuation_policy_rejects_non_user_or_expired_or_wrong_sleep_lineage(self):
        session = "continuation_invalid"
        base = datetime.now(timezone.utc).replace(microsecond=0)
        external = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION, source_kind=SourceKind.EXTERNAL_VERIFIER,
            source_ref="fixture", payload={"content": "not user approval"}))
        payload = {"continuation_id": "bad_continue", "action": "enable",
                   "scope": "next_sleep_epoch_read_only_research", "approval_event_id": external.event_id,
                   "unattended_policy_id": "policy", "focus_digest": "a" * 64,
                   "profile_digest": "b" * 64, "max_auto_runs": 1,
                   "issued_at": base.isoformat(), "expires_at": (base + timedelta(hours=1)).isoformat(),
                   "policy_version": "unattended_wake_continuation_policy_v1"}
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.UNATTENDED_WAKE_CONTINUATION_POLICY,
                source_kind=SourceKind.USER, source_ref="user", payload=payload,
                parent_event_ids=(external.event_id,)))

    def test_plain_auto_wake_cannot_be_a_model_planning_trigger(self):
        base, _, _, _, _, _, awake, _ = self.sleep_wake_lifecycle(
            "plain_auto_wake", terminal=False, auto_policy=True)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="plain_auto_wake", kind=EventKind.MODEL_INVOCATION,
                source_kind=SourceKind.SYSTEM, source_ref="kimi",
                created_at=(base + timedelta(seconds=6)).isoformat(), payload={
                    "invocation_id": "plain_auto_planning", "run_id": None,
                    "trigger_event_id": awake.event_id, "provider": "fixture", "model": "fixture",
                    "role": "tool_planning", "outcome": "completed", "latency_ms": 1,
                    "context_scope": "tool_planning", "public_summary": "Must be rejected.",
                }, parent_event_ids=(awake.event_id,)))

    def test_hash_chain_and_sessions_are_isolated(self):
        first = self.store.append(self.event("one", {"content": "1"}))
        second = self.store.append(self.event("one", {"content": "2"}))
        other = self.store.append(self.event("two", {"content": "3"}))
        self.assertEqual((first.sequence, second.sequence, other.sequence), (1, 2, 1))
        self.assertEqual(second.previous_hash, first.content_hash)
        self.assertTrue(self.store.verify_chain("one"))
        self.assertEqual([event.event_id for event in self.store.list("two")], [other.event_id])

    def test_tampering_is_detected(self):
        event = self.store.append(self.event("one", {"content": "1"}))
        with self.store.connection:
            self.store.connection.execute("UPDATE cognitive_events SET payload_json = ? WHERE event_id = ?",
                                        ('{"content":"9"}', event.event_id))
        self.assertFalse(self.store.verify_chain("one"))

    def test_purge_logically_removes_session_rows(self):
        self.store.append(self.event("one", {"content": "1"}))
        self.store.append(self.event("two", {"content": "2"}))
        self.store.purge_session("one")
        self.assertEqual(self.store.list("one"), [])
        count = self.store.connection.execute(
            "SELECT COUNT(*) FROM cognitive_events WHERE session_id = ?", ("one",)
        ).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertEqual(len(self.store.list("two")), 1)

    def test_parents_and_hidden_reasoning_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.DECISION,
                source_kind=SourceKind.MODEL, source_ref="test", payload={}))
        root = self.store.append(self.event("one", {"content": "1"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="two", kind=EventKind.DECISION,
                source_kind=SourceKind.MODEL, source_ref="test", payload={}, parent_event_ids=(root.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(self.event("one", {"nested": {"scratchpad": "secret"}}))
        with self.assertRaises(ValueError):
            self.store.append(self.event("one", {"content": "x" * 70000}))

    def test_event_kind_source_pairs_are_enforced(self):
        parent = self.store.append(self.event("one", {"content": "1"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="one", kind=EventKind.ACTION_RESULT,
                source_kind=SourceKind.MODEL, source_ref="model", payload={},
                parent_event_ids=(parent.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="one", kind=EventKind.SEED_APPROVED,
                source_kind=SourceKind.SYSTEM, source_ref="system", payload={},
                parent_event_ids=(parent.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="model-root", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.MODEL, source_ref="model", payload={}))

    def test_payload_schemas_and_disabled_inference_are_enforced(self):
        with self.assertRaises(ValueError):
            self.store.append(self.event("one", {"unknown": "field"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.INFERENCE,
                source_kind=SourceKind.MODEL, source_ref="model", payload={"summary": "not implemented"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user",
                payload={"approval": "seed", "seed_id": "seed_x"}))
        tool = self.store.append(CognitiveEvent(session_id="one", kind=EventKind.TOOL_RESULT,
            source_kind=SourceKind.TOOL, source_ref="tool", payload={"tool_name": "fixture", "outcome": "ok"},
            parent_event_ids=(self.store.append(self.event("one", {"content": "go"})).event_id,)))
        self.assertEqual(tool.kind, EventKind.TOOL_RESULT)

    def test_strict_sources_nested_models_and_correction_values_are_enforced(self):
        root = self.store.append(self.event("one", {"content": "root"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.TOOL, source_ref="tool", payload={"content": "forged"}))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.TOOL_RESULT,
                source_kind=SourceKind.USER, source_ref="user", payload={"tool_name": "x", "outcome": "ok"},
                parent_event_ids=(root.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.SEED_PROPOSED,
                source_kind=SourceKind.MODEL, source_ref="model", payload={"seed": {"analysis": "secret"}},
                parent_event_ids=(root.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="one", kind=EventKind.CORRECTION,
                source_kind=SourceKind.POLICY, source_ref="policy", payload={"target_event_id": root.event_id,
                "counterevidence_event_id": root.event_id, "disposition": "override", "public_summary": "free"},
                parent_event_ids=(root.event_id,)))

    def test_file_store_enables_secure_delete_and_concurrent_appends_are_serial(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        try:
            first = SQLiteEventStore(handle.name)
            barrier, failures = threading.Barrier(2), []
            def append(label):
                try:
                    barrier.wait()
                    store = SQLiteEventStore(handle.name)
                    store.append(CognitiveEvent(session_id="shared", kind=EventKind.OBSERVATION,
                        source_kind=SourceKind.USER, source_ref=label, payload={"content": label}))
                    store.close()
                except Exception as error:
                    failures.append(error)
            threads = [threading.Thread(target=append, args=("a",)),
                       threading.Thread(target=append, args=("b",))]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(failures, [])
            self.assertEqual([event.sequence for event in first.list("shared")], [1, 2])
            self.assertTrue(first.verify_chain("shared"))
            self.assertEqual(first.connection.execute("PRAGMA secure_delete").fetchone()[0], 1)
            report = first.purge_session("shared")
            self.assertTrue(report["logical_deletion"])
            self.assertTrue(report["secure_delete"])
            self.assertEqual(first.list("shared"), [])
            first.close()
        finally:
            os.unlink(handle.name)

    def test_transaction_rolls_back_process_level_interrupts_and_releases_lock(self):
        for exception_type in (KeyboardInterrupt, SystemExit):
            session = "interrupt_" + exception_type.__name__
            with self.subTest(exception_type=exception_type):
                with self.assertRaises(exception_type):
                    with self.store.transaction():
                        self.store.append(self.event(session, {"content": "partial"}),
                                          commit=False)
                        raise exception_type()
                self.assertFalse(self.store.connection.in_transaction)
                self.assertEqual(self.store.list(session), [])
                stored = self.store.append(self.event(session, {"content": "after"}))
                self.assertEqual(stored.sequence, 1)

    def test_rollback_failure_does_not_mask_original_base_exception(self):
        delegate = self.store.connection

        class RollbackFailingConnection:
            def __getattr__(self, name):
                return getattr(delegate, name)

            @property
            def in_transaction(self):
                return delegate.in_transaction

            def rollback(self):
                raise RuntimeError("rollback failed")

        marker = KeyboardInterrupt("original interrupt")
        caught = None
        self.store.connection = RollbackFailingConnection()
        try:
            try:
                with self.store.transaction():
                    self.store.append(self.event("rollback_error", {"content": "partial"}),
                                      commit=False)
                    raise marker
            except BaseException as error:
                caught = error
        finally:
            delegate.rollback()
            self.store.connection = delegate
        self.assertIs(caught, marker)
        self.assertEqual(self.store.list("rollback_error"), [])

    def test_media_percept_and_autonomy_lineage_are_strict(self):
        now = datetime.now(timezone.utc).isoformat()
        digest = "a" * 64
        media = self.store.append(CognitiveEvent(
            session_id="sensory", kind=EventKind.MEDIA_OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "artifact_id": "artifact_1", "modality": "audio", "sha256": digest,
                "mime_type": "audio/wav", "byte_length": 42, "received_at": now,
                "duration_ms": 50, "retention_scope": "session",
            }))
        percept = self.store.append(CognitiveEvent(
            session_id="sensory", kind=EventKind.PERCEPT,
            source_kind=SourceKind.SYSTEM, source_ref="adapter", payload={
                "percept_id": "percept_1", "artifact_id": "artifact_1",
                "artifact_sha256": digest, "modality": "audio", "span_start_ms": 0,
                "span_end_ms": 50, "percept_kind": "speech_segment", "value": "hello",
                "confidence": 0.9, "adapter_id": "fixture", "adapter_version": "v1",
            }, parent_event_ids=(media.event_id,)))
        self.assertEqual(percept.kind, EventKind.PERCEPT)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="sensory", kind=EventKind.PERCEPT,
                source_kind=SourceKind.SYSTEM, source_ref="adapter", payload={
                    "percept_id": "percept_2", "artifact_id": "artifact_other",
                    "artifact_sha256": digest, "modality": "audio", "span_start_ms": 0,
                    "span_end_ms": 1, "percept_kind": "speech_segment", "value": "x",
                    "confidence": 0.9, "adapter_id": "fixture", "adapter_version": "v1",
                }, parent_event_ids=(media.event_id,)))
        control = self.store.append(CognitiveEvent(
            session_id="sensory", kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "run_id": "run_1", "action": "start", "config_version": "v1",
                "max_ticks": 3, "max_wall_seconds": 30, "max_events_per_tick": 4,
                "max_no_progress": 2, "min_interval_seconds": 0.0,
                "next_tick_index": 0,
            }))
        tick = self.store.append(CognitiveEvent(
            session_id="sensory", kind=EventKind.LOOP_TICK,
            source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                "run_id": "run_1", "tick_index": 0, "trigger": "timer", "phase": "observe",
                "focus_event_ids": [media.event_id], "retrieved_event_ids": [],
                "progress": "made_progress", "salience_reason": "new sensory record",
                "budget_remaining": 2, "public_summary": "Sensory record considered.",
            }, parent_event_ids=(control.event_id,)))
        self.assertEqual(tick.kind, EventKind.LOOP_TICK)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="sensory", kind=EventKind.LOOP_TICK,
                source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                    "run_id": "run_1", "tick_index": 2, "trigger": "timer", "phase": "observe",
                    "focus_event_ids": [], "retrieved_event_ids": [], "progress": "no_progress",
                    "salience_reason": "idle", "budget_remaining": 1, "public_summary": "No progress.",
                }, parent_event_ids=(tick.event_id,)))

    def test_quota_paused_loop_tick_is_a_public_no_external_action_phase(self):
        control = self.store.append(CognitiveEvent(
            session_id="quota_phase", kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "run_id": "quota_run", "action": "start", "config_version": "v1",
                "max_ticks": 1, "max_wall_seconds": 1, "max_events_per_tick": 1,
                "max_no_progress": 0, "min_interval_seconds": 0.0, "next_tick_index": 0}))
        tick = self.store.append(CognitiveEvent(
            session_id="quota_phase", kind=EventKind.LOOP_TICK,
            source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                "run_id": "quota_run", "tick_index": 0, "trigger": "control", "phase": "quota_paused",
                "focus_event_ids": [], "retrieved_event_ids": [], "progress": "blocked",
                "salience_reason": "quota gate", "budget_remaining": 0,
                "public_summary": "K3 reflection is paused by the host quota controller; no provider call was made."},
            parent_event_ids=(control.event_id,)))
        self.assertEqual("quota_paused", tick.payload["phase"])

    def test_autonomy_pause_resume_preserves_tick_continuity(self):
        def control(action, next_tick_index):
            return self.store.append(CognitiveEvent(
                session_id="resume", kind=EventKind.AUTONOMY_CONTROL,
                source_kind=SourceKind.USER, source_ref="user", payload={
                    "run_id": "run_resume", "action": action, "config_version": "v1",
                    "max_ticks": 5, "max_wall_seconds": 30, "max_events_per_tick": 4,
                    "max_no_progress": 2, "min_interval_seconds": 0.0,
                    "next_tick_index": next_tick_index,
                }))

        def tick(parent_id, tick_index):
            return self.store.append(CognitiveEvent(
                session_id="resume", kind=EventKind.LOOP_TICK,
                source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                    "run_id": "run_resume", "tick_index": tick_index,
                    "trigger": "timer", "phase": "reflect", "focus_event_ids": [],
                    "retrieved_event_ids": [], "progress": "made_progress",
                    "salience_reason": "scheduled bounded reflection",
                    "budget_remaining": max(0, 4 - tick_index),
                    "public_summary": "Bounded tick completed.",
                }, parent_event_ids=(parent_id,)))

        start = control("start", 0)
        first = tick(start.event_id, 0)
        pause = control("pause", 1)
        resume = control("resume", 1)
        second = tick(resume.event_id, 1)
        self.assertEqual((first.payload["tick_index"], second.payload["tick_index"]), (0, 1))

        with self.assertRaises(ValueError):
            tick(pause.event_id, 1)
        with self.assertRaises(ValueError):
            tick(resume.event_id, 2)

        stop = control("stop", 2)
        with self.assertRaises(ValueError):
            tick(stop.event_id, 2)
        stopped = self.store.append(CognitiveEvent(
            session_id="resume", kind=EventKind.AUTONOMY_STOPPED,
            source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                "run_id": "run_resume", "reason": "user_requested",
                "tick_count": 2, "stopped_at": datetime.now(timezone.utc).isoformat(),
            }, parent_event_ids=(stop.event_id,)))
        self.assertEqual(stopped.payload["tick_count"], 2)

    def test_autonomy_start_index_and_stop_count_are_exact(self):
        base = {"run_id": "run_exact", "config_version": "v1", "max_ticks": 5,
                "max_wall_seconds": 30, "max_events_per_tick": 4,
                "max_no_progress": 2, "min_interval_seconds": 0.0}
        bad_start = dict(base, action="start", next_tick_index=1)
        start = self.store.append(CognitiveEvent(
            session_id="exact", kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.USER, source_ref="user", payload=bad_start))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="exact", kind=EventKind.LOOP_TICK,
                source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                    "run_id": "run_exact", "tick_index": 0, "trigger": "timer",
                    "phase": "idle", "focus_event_ids": [], "retrieved_event_ids": [],
                    "progress": "no_progress", "salience_reason": "idle",
                    "budget_remaining": 4, "public_summary": "No work selected.",
                }, parent_event_ids=(start.event_id,)))
        stop = self.store.append(CognitiveEvent(
            session_id="exact", kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.USER, source_ref="user",
            payload=dict(base, action="stop", next_tick_index=3)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="exact", kind=EventKind.AUTONOMY_STOPPED,
                source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                    "run_id": "run_exact", "reason": "user_requested", "tick_count": 2,
                    "stopped_at": datetime.now(timezone.utc).isoformat(),
                }, parent_event_ids=(stop.event_id,)))

    def test_reward_value_and_rpe_require_bound_inputs_and_formula(self):
        root = self.store.append(self.event("td", {"content": "go"}))
        action = self.store.append(CognitiveEvent(
            session_id="td", kind=EventKind.ACTION_RESULT, source_kind=SourceKind.SYSTEM,
            source_ref="system", payload={"action_type": "response", "outcome": "ok", "response_text": "done"},
            parent_event_ids=(root.event_id,)))
        now = datetime.now(timezone.utc).isoformat()
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="td", kind=EventKind.VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="estimator",
                payload={"estimate_id": "old_schema", "target_event_id": action.event_id,
                    "state_key": "state_1", "action_key": "respond", "value": 0.2,
                    "confidence": 0.8, "estimator_id": "td", "estimator_version": "v1"},
                parent_event_ids=(action.event_id,)))
        estimate = self.store.append(CognitiveEvent(
            session_id="td", kind=EventKind.VALUE_ESTIMATE, source_kind=SourceKind.SYSTEM,
            source_ref="estimator", payload={"estimate_id": "estimate_1", "transition_id": "transition_1",
                "target_event_id": action.event_id, "state_key": "state_1", "action_key": "respond",
                "next_state_key": "state_2", "terminal": True, "value": 0.2, "confidence": 0.8,
                "estimator_id": "td", "estimator_version": "v1", "alpha": 0.5, "gamma": 0.9,
                "clip": 1.0, "max_entries": 256, "max_events": 1024,
                "formula_version": "td0_v1"}, parent_event_ids=(action.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="td", kind=EventKind.VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="estimator",
                payload=dict(estimate.payload, estimate_id="estimate_unsafe",
                             transition_id="transition_unsafe", action_key="write"),
                parent_event_ids=(action.event_id,)))
        reward = self.store.append(CognitiveEvent(
            session_id="td", kind=EventKind.REWARD_OBSERVATION, source_kind=SourceKind.USER,
            source_ref="user", payload={"reward_id": "reward_1", "target_event_id": action.event_id,
                "signal_kind": "user_feedback", "normalized_value": 1.0, "evaluator_ref": "user_1",
                "scale_version": "v1", "observed_at": now}, parent_event_ids=(action.event_id,)))
        for changed in (
                {"transition_id": "transition_other"},
                {"alpha": 0.4},
                {"gamma": 0.8},
                {"clip": 0.5}):
            payload = {"update_id": "invalid_" + next(iter(changed)),
                "transition_id": "transition_1", "reward_event_id": reward.event_id,
                "prior_value_event_id": estimate.event_id, "state_key": "state_1",
                "action_key": "respond", "reward": 1.0, "prior_value": 0.2,
                "next_value": 0.0, "alpha": 0.5, "gamma": 0.9, "raw_delta": 0.8,
                "clipped_delta": 0.8, "clip": 1.0, "updated_value": 0.6,
                "formula_version": "td0_v1", "scope": "research_ranking_only"}
            payload.update(changed)
            with self.assertRaises(ValueError):
                self.store.append(CognitiveEvent(
                    session_id="td", kind=EventKind.RPE_UPDATE,
                    source_kind=SourceKind.SYSTEM, source_ref="td", payload=payload,
                    parent_event_ids=(reward.event_id, estimate.event_id)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="td", kind=EventKind.RPE_UPDATE, source_kind=SourceKind.SYSTEM,
                source_ref="td", payload={"update_id": "update_2", "transition_id": "transition_1",
                    "reward_event_id": reward.event_id, "prior_value_event_id": estimate.event_id,
                    "state_key": "state_1", "action_key": "respond",
                    "reward": 1.0, "prior_value": 0.2, "next_value": 0.0, "alpha": 0.5, "gamma": 0.9,
                    "raw_delta": 0.7, "clipped_delta": 0.7, "clip": 1.0, "updated_value": 0.55, "formula_version": "td0_v1",
                    "scope": "research_ranking_only"}, parent_event_ids=(reward.event_id, estimate.event_id)))
        update = self.store.append(CognitiveEvent(
            session_id="td", kind=EventKind.RPE_UPDATE, source_kind=SourceKind.SYSTEM,
            source_ref="td", payload={"update_id": "update_1", "transition_id": "transition_1",
                "reward_event_id": reward.event_id, "prior_value_event_id": estimate.event_id,
                "state_key": "state_1", "action_key": "respond",
                "reward": 1.0, "prior_value": 0.2, "next_value": 0.0, "alpha": 0.5, "gamma": 0.9,
                "raw_delta": 0.8, "clipped_delta": 0.8, "clip": 1.0, "updated_value": 0.6, "formula_version": "td0_v1",
                "scope": "research_ranking_only"}, parent_event_ids=(reward.event_id, estimate.event_id)))
        self.assertEqual(update.kind, EventKind.RPE_UPDATE)

    def test_value_capacity_is_pinned_and_terminal_next_value_is_zero(self):
        def target(session, label):
            root = self.store.append(self.event(session, {"content": label}))
            return self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.ACTION_RESULT,
                source_kind=SourceKind.SYSTEM, source_ref="system",
                payload={"action_type": "response", "outcome": "ok", "response_text": label},
                parent_event_ids=(root.event_id,)))

        def estimate(session, target_event, suffix, max_entries, max_events, terminal=True):
            return self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="td",
                payload={"estimate_id": "estimate_" + suffix,
                    "transition_id": "transition_" + suffix,
                    "target_event_id": target_event.event_id, "state_key": "state_" + suffix,
                    "action_key": "respond", "next_state_key": "next_" + suffix,
                    "terminal": terminal, "value": 0.0, "confidence": 1.0,
                    "estimator_id": "td", "estimator_version": "v1",
                    "alpha": 0.5, "gamma": 0.9, "clip": 1.0,
                    "max_entries": max_entries, "max_events": max_events,
                    "formula_version": "td0_v1"},
                parent_event_ids=(target_event.event_id,)))

        first_target = target("limits", "one")
        first = estimate("limits", first_target, "one", 600, 2000)
        self.assertEqual((first.payload["max_entries"], first.payload["max_events"]),
                         (600, 2000))
        second_target = target("limits", "two")
        with self.assertRaises(ValueError):
            estimate("limits", second_target, "bad", 599, 2000)
        with self.assertRaises(ValueError):
            estimate("limits", second_target, "bad_events", 600, 1999)
        second = estimate("limits", second_target, "two", 600, 2000)
        reward = self.store.append(CognitiveEvent(
            session_id="limits", kind=EventKind.REWARD_OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            payload={"reward_id": "reward_limits", "target_event_id": second_target.event_id,
                "signal_kind": "user_feedback", "normalized_value": 1.0,
                "evaluator_ref": "user_1", "scale_version": "v1",
                "observed_at": datetime.now(timezone.utc).isoformat()},
            parent_event_ids=(second_target.event_id,)))
        forged = {"update_id": "update_forged", "transition_id": second.payload["transition_id"],
            "reward_event_id": reward.event_id, "prior_value_event_id": second.event_id,
            "state_key": second.payload["state_key"], "action_key": "respond", "reward": 1.0,
            "prior_value": 0.0, "next_value": 0.5, "alpha": 0.5, "gamma": 0.9,
            "raw_delta": 1.45, "clipped_delta": 1.0, "clip": 1.0,
            "updated_value": 0.5, "formula_version": "td0_v1",
            "scope": "research_ranking_only"}
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id="limits", kind=EventKind.RPE_UPDATE,
                source_kind=SourceKind.SYSTEM, source_ref="td", payload=forged,
                parent_event_ids=(reward.event_id, second.event_id)))

        low_target = target("low_limits", "low")
        low = estimate("low_limits", low_target, "low", 1, 1)
        self.assertEqual((low.payload["max_entries"], low.payload["max_events"]), (1, 1))

    def test_rpe_records_clipped_delta_and_rejects_unclipped_update(self):
        root = self.store.append(self.event("clip", {"content": "go"}))
        action = self.store.append(CognitiveEvent(
            session_id="clip", kind=EventKind.ACTION_RESULT, source_kind=SourceKind.SYSTEM,
            source_ref="system", payload={"action_type": "response", "outcome": "ok", "response_text": "done"},
            parent_event_ids=(root.event_id,)))
        now = datetime.now(timezone.utc).isoformat()
        estimate = self.store.append(CognitiveEvent(
            session_id="clip", kind=EventKind.VALUE_ESTIMATE, source_kind=SourceKind.SYSTEM,
            source_ref="estimator", payload={"estimate_id": "estimate_clip", "transition_id": "transition_clip",
                "target_event_id": action.event_id, "state_key": "state_1", "action_key": "respond",
                "next_state_key": "state_2", "terminal": True, "value": 0.0, "confidence": 1.0,
                "estimator_id": "td", "estimator_version": "v1", "alpha": 0.5, "gamma": 0.0,
                "clip": 0.2, "max_entries": 256, "max_events": 1024,
                "formula_version": "td0_v1"}, parent_event_ids=(action.event_id,)))
        reward = self.store.append(CognitiveEvent(
            session_id="clip", kind=EventKind.REWARD_OBSERVATION, source_kind=SourceKind.USER,
            source_ref="user", payload={"reward_id": "reward_clip", "target_event_id": action.event_id,
                "signal_kind": "user_feedback", "normalized_value": 1.0, "evaluator_ref": "user_1",
                "scale_version": "v1", "observed_at": now}, parent_event_ids=(action.event_id,)))
        base = {"update_id": "update_clip", "transition_id": "transition_clip",
            "reward_event_id": reward.event_id, "prior_value_event_id": estimate.event_id,
            "state_key": "state_1", "action_key": "respond",
            "reward": 1.0, "prior_value": 0.0, "next_value": 0.0, "alpha": 0.5, "gamma": 0.0,
            "raw_delta": 1.0, "clipped_delta": 0.2, "clip": 0.2, "updated_value": 0.1,
            "formula_version": "td0_v1", "scope": "research_ranking_only"}
        bad = dict(base, update_id="update_bad", updated_value=0.5)
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(session_id="clip", kind=EventKind.RPE_UPDATE,
                source_kind=SourceKind.SYSTEM, source_ref="td", payload=bad,
                parent_event_ids=(reward.event_id, estimate.event_id)))
        stored = self.store.append(CognitiveEvent(session_id="clip", kind=EventKind.RPE_UPDATE,
            source_kind=SourceKind.SYSTEM, source_ref="td", payload=base,
            parent_event_ids=(reward.event_id, estimate.event_id)))
        self.assertEqual(stored.payload["clipped_delta"], 0.2)

    def test_td_protocol_rejects_late_prediction_and_replayed_reward(self):
        now = datetime.now(timezone.utc).isoformat()

        def action(session):
            root = self.store.append(self.event(session, {"content": "go"}))
            return self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.ACTION_RESULT,
                source_kind=SourceKind.SYSTEM, source_ref="system",
                payload={"action_type": "response", "outcome": "ok", "response_text": "done"},
                parent_event_ids=(root.event_id,)))

        def estimate(session, target, estimate_id):
            return self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.VALUE_ESTIMATE,
                source_kind=SourceKind.SYSTEM, source_ref="estimator",
                payload={"estimate_id": estimate_id, "transition_id": "transition_" + session,
                    "target_event_id": target.event_id, "state_key": "state_1",
                    "action_key": "respond", "next_state_key": "state_2", "terminal": True,
                    "value": 0.0, "confidence": 1.0, "estimator_id": "td",
                    "estimator_version": "v1", "alpha": 0.5, "gamma": 0.0,
                    "clip": 1.0, "max_entries": 256, "max_events": 1024,
                    "formula_version": "td0_v1"},
                parent_event_ids=(target.event_id,)))

        def reward(session, target, reward_id):
            return self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.REWARD_OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user",
                payload={"reward_id": reward_id, "target_event_id": target.event_id,
                    "signal_kind": "user_feedback", "normalized_value": 1.0,
                    "evaluator_ref": "user_1", "scale_version": "v1", "observed_at": now},
                parent_event_ids=(target.event_id,)))

        def rpe(session, reward_event, estimate_event, update_id):
            return self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.RPE_UPDATE,
                source_kind=SourceKind.SYSTEM, source_ref="td",
                payload={"update_id": update_id, "transition_id": "transition_" + session,
                    "reward_event_id": reward_event.event_id,
                    "prior_value_event_id": estimate_event.event_id, "state_key": "state_1",
                    "action_key": "respond", "reward": 1.0, "prior_value": 0.0,
                    "next_value": 0.0, "alpha": 0.5, "gamma": 0.0,
                    "raw_delta": 1.0, "clipped_delta": 1.0, "clip": 1.0,
                    "updated_value": 0.5, "formula_version": "td0_v1",
                    "scope": "research_ranking_only"},
                parent_event_ids=(reward_event.event_id, estimate_event.event_id)))

        late_target = action("late")
        with self.assertRaises(ValueError):
            reward("late", late_target, "reward_late")
        late_estimate = estimate("late", late_target, "estimate_late")
        with self.assertRaises(ValueError):
            estimate("late", late_target, "estimate_late_second")

        duplicate_target = action("duplicate")
        estimate_event = estimate("duplicate", duplicate_target, "estimate_duplicate")
        reward_event = reward("duplicate", duplicate_target, "reward_duplicate")
        with self.assertRaises(ValueError):
            reward("duplicate", duplicate_target, "reward_second_for_target")

        other_target = action("duplicate")
        estimate("duplicate", other_target, "estimate_other")
        with self.assertRaises(ValueError):
            reward("duplicate", other_target, "reward_duplicate")

        first = rpe("duplicate", reward_event, estimate_event, "update_first")
        self.assertEqual(first.payload["reward_event_id"], reward_event.event_id)
        with self.assertRaises(ValueError):
            rpe("duplicate", reward_event, estimate_event, "update_second")

    def test_k3_tool_execution_protocol_binds_grant_run_and_terminal_result(self):
        """K3 records public provenance, never prompts, commands, or raw output."""
        session = "k3"
        now = datetime.now(timezone.utc)
        created = now.isoformat()
        digest = "a" * 64
        input_digest = "b" * 64
        result_digest = "c" * 64
        request = self.store.append(self.event(session, {"content": "use the bounded test tool"}))
        grant = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.CAPABILITY_GRANTED,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "grant_id": "grant_1", "capability": "test_read", "tool_name": "fixture_tool",
                "scope_digest": digest, "not_before": (now - timedelta(seconds=1)).isoformat(),
                "expires_at": (now + timedelta(minutes=1)).isoformat(), "max_uses": 1,
                "max_input_bytes": 64, "max_output_bytes": 128, "max_wall_ms": 500,
                "allow_mutating": True, "grant_version": "k3_v1",
            }, parent_event_ids=(request.event_id,)))
        control = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.AUTONOMY_CONTROL,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "run_id": "run_1", "action": "start", "config_version": "loop_config_v1",
                "max_ticks": 3, "max_wall_seconds": 30, "max_events_per_tick": 4,
                "max_no_progress": 2, "min_interval_seconds": 0.0, "next_tick_index": 0,
            }))
        tick = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.LOOP_TICK,
            source_kind=SourceKind.SYSTEM, source_ref="loop", payload={
                "run_id": "run_1", "tick_index": 0, "trigger": "control", "phase": "reflect",
                "focus_event_ids": [request.event_id], "retrieved_event_ids": [],
                "progress": "made_progress", "salience_reason": "bounded test",
                "budget_remaining": 3, "public_summary": "A bounded record was selected.",
            }, parent_event_ids=(control.event_id,)))
        invocation = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.MODEL_INVOCATION,
            source_kind=SourceKind.SYSTEM, source_ref="model_gateway", payload={
                "invocation_id": "invoke_1", "run_id": "run_1", "trigger_event_id": tick.event_id,
                "provider": "fixture_provider", "model": "fixture_model", "role": "tool_planning",
                "outcome": "completed", "latency_ms": 4, "context_scope": "loop_tick",
                "public_summary": "A bounded tool plan was requested.",
            }, parent_event_ids=(tick.event_id,)))
        proposal = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.TOOL_CALL_PROPOSED,
            source_kind=SourceKind.MODEL, source_ref="model", payload={
                "call_id": "call_1", "run_id": "run_1", "grant_id": "grant_1",
                "tool_name": "fixture_tool", "operation": "read_status", "plan_digest": digest,
                "input_digest": input_digest, "requested_input_bytes": 32,
                "requested_output_bytes": 64, "requested_wall_ms": 100, "mutating": True,
                "public_summary": "A bounded tool call was proposed for review.",
            }, parent_event_ids=(invocation.event_id, grant.event_id)))
        confirmation = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.TOOL_EXECUTION_CONFIRMED,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "confirmation_id": "confirm_1", "call_id": "call_1", "run_id": "run_1",
                "grant_id": "grant_1", "plan_digest": digest, "confirmed_at": created,
            }, parent_event_ids=(proposal.event_id,)))
        started_at = datetime.now(timezone.utc)
        started = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.TOOL_EXECUTION_STARTED,
            source_kind=SourceKind.SYSTEM, source_ref="executor", payload={
                "execution_id": "exec_1", "call_id": "call_1", "run_id": "run_1",
                "grant_id": "grant_1", "plan_digest": digest,
                "confirmation_event_id": confirmation.event_id, "started_at": started_at.isoformat(),
                "deadline_at": (started_at + timedelta(milliseconds=100)).isoformat(),
            }, parent_event_ids=(proposal.event_id, grant.event_id, confirmation.event_id)))
        result_payload = {
            "execution_id": "exec_1", "call_id": "call_1", "run_id": "run_1",
            "grant_id": "grant_1", "plan_digest": digest, "tool_name": "fixture_tool",
            "outcome": "succeeded", "result_digest": result_digest, "result_bytes": 64,
            "latency_ms": 10, "summary": "Fixture tool completed within its bounded budget.",
        }
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.TOOL_RESULT, source_kind=SourceKind.TOOL,
                source_ref="fixture_tool", payload=dict(result_payload, result_bytes=129),
                parent_event_ids=(started.event_id,)))
        result = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.TOOL_RESULT, source_kind=SourceKind.TOOL,
            source_ref="fixture_tool", payload=result_payload, parent_event_ids=(started.event_id,)))
        self.assertEqual(result.payload["execution_id"], "exec_1")
        self.assertTrue(self.store.verify_chain(session))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.TOOL_EXECUTION_ABANDONED,
                source_kind=SourceKind.SYSTEM, source_ref="executor", payload={
                    "execution_id": "exec_1", "call_id": "call_1", "run_id": "run_1",
                    "grant_id": "grant_1", "plan_digest": digest, "reason": "cancelled",
                    "abandoned_at": datetime.now(timezone.utc).isoformat(),
                    "public_summary": "The tool execution was cancelled.",
                }, parent_event_ids=(started.event_id,)))

    def test_k3_refuses_prompt_fields_revocation_and_run_renewal(self):
        session = "k3_refusal"
        now = datetime.now(timezone.utc)
        request = self.store.append(self.event(session, {"content": "bounded work"}))
        grant = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.CAPABILITY_GRANTED,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "grant_id": "grant_refusal", "capability": "test_read", "tool_name": "fixture_tool",
                "scope_digest": "d" * 64, "not_before": (now - timedelta(seconds=1)).isoformat(),
                "expires_at": (now + timedelta(minutes=1)).isoformat(), "max_uses": 1,
                "max_input_bytes": 10, "max_output_bytes": 10, "max_wall_ms": 50,
                "allow_mutating": False, "grant_version": "k3_v1",
            }, parent_event_ids=(request.event_id,)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.MODEL_INVOCATION,
                source_kind=SourceKind.SYSTEM, source_ref="gateway", payload={
                    "invocation_id": "bad_prompt", "run_id": "run_missing", "trigger_event_id": request.event_id,
                    "provider": "fixture", "model": "fixture", "role": "tool_planning",
                    "outcome": "completed", "latency_ms": 1, "context_scope": "loop_tick",
                    "public_summary": "summary", "prompt": "must not persist",
                }, parent_event_ids=(request.event_id,)))
        revoked = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.CAPABILITY_REVOKED,
            source_kind=SourceKind.USER, source_ref="user", payload={
                "revocation_id": "revoke_1", "grant_id": "grant_refusal", "reason": "user_requested",
                "revoked_at": datetime.now(timezone.utc).isoformat(),
            }, parent_event_ids=(grant.event_id,)))
        self.assertEqual(revoked.payload["grant_id"], grant.payload["grant_id"])

    def test_metacognitive_mirror_is_bounded_audit_not_authority(self):
        session = "mirror"
        target = self.store.append(self.event(session, {"content": "observable input"}))
        judgment = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.ACTION_RESULT,
            source_kind=SourceKind.SYSTEM, source_ref="runtime",
            payload={"action_type": "response", "outcome": "recorded",
                     "response_text": "A public response was recorded."},
            parent_event_ids=(target.event_id,)))

        def payload(mirror_id="mirror_1", episode_id="episode_1"):
            return {
                "mirror_id": mirror_id, "episode_id": episode_id,
                "target_event_id": target.event_id, "judgment_event_id": judgment.event_id,
                "evidence_event_ids": [target.event_id, judgment.event_id],
                "self_status": "supported", "self_confidence": 0.8,
                "self_uncertainty": "low", "meta_status": "limited",
                "meta_confidence_cap": 0.6,
                "check_codes": ["provenance_complete", "scope_bounded", "confidence_capped"],
                "disposition": "review_required", "method_version": "mirror_v1",
                "public_summary": "The bounded public record is suitable for review.",
            }

        mirror = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.METACOGNITIVE_MIRROR,
            source_kind=SourceKind.SYSTEM, source_ref="MirrorAuditor", payload=payload(),
            confidence=0.6, parent_event_ids=(target.event_id, judgment.event_id)))
        self.assertEqual(mirror.kind, EventKind.METACOGNITIVE_MIRROR)
        self.assertTrue(self.store.verify_chain(session))

        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.METACOGNITIVE_MIRROR,
                source_kind=SourceKind.SYSTEM, source_ref="MirrorAuditor",
                payload=payload("mirror_2"), confidence=0.6,
                parent_event_ids=(target.event_id, judgment.event_id)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.METACOGNITIVE_MIRROR,
                source_kind=SourceKind.SYSTEM, source_ref="untrusted",
                payload=payload("mirror_3", "episode_3"), confidence=0.7,
                parent_event_ids=(target.event_id, judgment.event_id)))
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.REWARD_OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={
                    "reward_id": "reward_mirror", "target_event_id": mirror.event_id,
                    "signal_kind": "user_feedback", "normalized_value": 1.0,
                    "evaluator_ref": "user_1", "scale_version": "v1",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }, parent_event_ids=(mirror.event_id,)))
        claim = {
            "claim_id": "claim_mirror", "kind": "epistemic",
            "statement": "This must not be approved from mirror evidence.",
            "evidence_event_ids": [mirror.event_id], "confidence": 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(), "expires_at": None,
        }
        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.SELF_CLAIM_PROPOSED,
                source_kind=SourceKind.MODEL, source_ref="model",
                payload={"claim": claim, "state": "candidate"},
                parent_event_ids=(target.event_id,)))

    def test_standing_seed_policy_requires_digest_bound_user_approval(self):
        session = "standing_seed_policy"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        payload = {
            "policy_id": "seed_policy_1", "user_observation_event_id": "placeholder",
            "nonce": "seed_policy_nonce_1", "issued_at": (now + timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(), "scope": "research",
            "max_auto_activations": 2, "max_auto_updates": 4, "max_active_seeds": 2,
            "max_cue_terms": 4, "max_cue_length": 64, "max_strength": .8,
            "max_confidence": .9, "max_seed_ttl_seconds": 3600,
            "max_strength_step": .1, "max_confidence_step": .1,
            "max_counterevidence": 10, "max_seed_versions": 8,
            "allowed_policy_bias": ["cite_sources"],
            "policy_version": "seed_standing_policy_v1",
        }
        # The approval event ID is part of the signed policy manifest.  Freeze
        # it before computing the digest so the policy can bind it exactly.
        approval_id = "evt_seed_policy_approval"
        payload["user_observation_event_id"] = approval_id
        approval = self.store.append(CognitiveEvent(
            event_id=approval_id, session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(),
            payload={"approval": "seed_standing_policy",
                     "policy_digest": SQLiteEventStore.standing_seed_policy_digest(payload),
                     "nonce": payload["nonce"]}))
        policy = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_STANDING_POLICY,
            source_kind=SourceKind.USER, source_ref="user", payload=payload,
            created_at=payload["issued_at"], parent_event_ids=(approval.event_id,)))
        self.assertEqual("seed_policy_1", policy.payload["policy_id"])
        self.assertTrue(self.store.verify_chain(session))

        wrong = dict(payload, policy_id="seed_policy_2", nonce="seed_policy_nonce_2")
        root = self.store.append(self.event("standing_bad", {"content": "plain request"}))
        with self.assertRaisesRegex(ValueError, "explicit user approval"):
            self.store.append(CognitiveEvent(
                session_id="standing_bad", kind=EventKind.SEED_STANDING_POLICY,
                source_kind=SourceKind.USER, source_ref="user", payload=wrong,
                parent_event_ids=(root.event_id,)))

    def test_nested_store_transaction_uses_savepoint_without_partial_commit(self):
        with self.store.transaction():
            outer = self.store.append(self.event("nested", {"content": "outer"}), commit=False)
            with self.assertRaisesRegex(RuntimeError, "abort inner"):
                with self.store.transaction():
                    inner = self.store.append(CognitiveEvent(
                        session_id="nested", kind=EventKind.OBSERVATION,
                        source_kind=SourceKind.USER, source_ref="test",
                        payload={"content": "inner"}), commit=False)
                    raise RuntimeError("abort inner")
            self.assertIsNone(self.store.get(inner.event_id))
            self.assertIsNotNone(self.store.get(outer.event_id))
        self.assertTrue(self.store.verify_chain("nested"))

    def test_standing_seed_activation_keeps_shorter_candidate_expiry(self):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        candidate = {
            "cue_terms": ["citation"], "policy_bias": "cite_sources",
            "scope": "research", "provenance_event_ids": ["evidence_1"],
            "strength": .25, "confidence": .25, "status": "candidate",
            "seed_id": "seed_short_ttl", "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=5)).isoformat(),
            "version": 1, "counterevidence": 0,
        }
        policy = {"expires_at": (now + timedelta(hours=2)).isoformat(),
                  "max_seed_ttl_seconds": 3600}
        active = SQLiteEventStore.standing_seed_activation_snapshot(
            candidate, policy, now.isoformat(), 2)
        self.assertEqual(candidate["expires_at"], active["expires_at"])
        without_expiry = dict(candidate, expires_at=None, seed_id="seed_no_ttl")
        derived = SQLiteEventStore.standing_seed_activation_snapshot(
            without_expiry, policy, now.isoformat(), 2)
        self.assertEqual((now + timedelta(hours=1)).isoformat(), derived["expires_at"])

    def test_seed_evidence_literal_cue_match_is_deterministic(self):
        self.assertTrue(SQLiteEventStore.seed_evidence_literal_cue_match(
            "Please add CITATION checks", ["citation"]))
        self.assertTrue(SQLiteEventStore.seed_evidence_literal_cue_match(
            "请加入引用检查", ["引用"]))
        self.assertFalse(SQLiteEventStore.seed_evidence_literal_cue_match(
            "unrelated preference", ["citation"]))

    def test_seed_update_rejects_unrelated_reinforcement_and_unbound_correction(self):
        session = "standing_seed_evidence_binding"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        evidence = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=now.isoformat(), payload={"content": "citation requested"}))
        seed = {"cue_terms": ["citation"], "policy_bias": "cite_sources",
                "scope": "research", "provenance_event_ids": [evidence.event_id],
                "strength": .25, "confidence": .25, "status": "candidate",
                "seed_id": "seed_evidence_bound", "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=1)).isoformat(),
                "version": 1, "counterevidence": 0}
        proposal = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_PROPOSED,
            source_kind=SourceKind.MODEL, source_ref="model",
            created_at=(now + timedelta(seconds=1)).isoformat(),
            payload={"seed": seed}, parent_event_ids=(evidence.event_id,)))
        approval = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=2)).isoformat(),
            payload={"approval": "seed", "seed_id": seed["seed_id"],
                     "proposal_event_id": proposal.event_id}))
        active_event = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_APPROVED,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=3)).isoformat(),
            payload={"seed_id": seed["seed_id"], "proposal_event_id": proposal.event_id,
                     "approval_event_id": approval.event_id},
            parent_event_ids=(proposal.event_id, approval.event_id)))
        prior = dict(seed, status="active", version=2,
                     updated_at=active_event.created_at)

        unrelated = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=4)).isoformat(),
            payload={"content": "weather preference"}))
        reinforced = dict(prior, strength=.3, version=3,
                          updated_at=(now + timedelta(seconds=5)).isoformat(),
                          provenance_event_ids=prior["provenance_event_ids"] + [unrelated.event_id])
        manifest = {"base_event_id": active_event.event_id, "base_version": 2,
                    "evidence_event_id": unrelated.event_id, "operation": "reinforce",
                    "proposed_seed": reinforced, "seed_id": seed["seed_id"]}
        with self.assertRaisesRegex(ValueError, "user evidence.*monotonic delta"):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.SEED_UPDATE_PROPOSED,
                source_kind=SourceKind.MODEL, source_ref="model",
                created_at=reinforced["updated_at"],
                payload={"proposal_id": "update_unrelated", "seed_id": seed["seed_id"],
                         "base_event_id": active_event.event_id, "base_version": 2,
                         "operation": "reinforce", "proposed_seed": reinforced,
                         "delta_digest": hashlib.sha256(
                             canonical_json(manifest).encode("utf-8")).hexdigest(),
                         "version": "seed_update_proposal_v1"},
                parent_event_ids=(active_event.event_id, unrelated.event_id)))

        counterevidence = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=5)).isoformat(),
            payload={"content": "citation conflict"}))
        correction = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.CORRECTION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=6)).isoformat(),
            payload={"target_event_id": unrelated.event_id,
                     "counterevidence_event_id": counterevidence.event_id,
                     "disposition": "review_required",
                     "public_summary": "A later observable record conflicts with an earlier record."},
            parent_event_ids=(unrelated.event_id, counterevidence.event_id)))
        tightened = dict(prior, strength=.2, confidence=.2, counterevidence=1,
                         version=3, updated_at=(now + timedelta(seconds=7)).isoformat(),
                         provenance_event_ids=prior["provenance_event_ids"] + [correction.event_id])
        tighten_manifest = {"base_event_id": active_event.event_id, "base_version": 2,
            "evidence_event_id": correction.event_id, "operation": "tighten",
            "proposed_seed": tightened, "seed_id": seed["seed_id"]}
        with self.assertRaisesRegex(ValueError, "bound correction evidence"):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.SEED_UPDATE_PROPOSED,
                source_kind=SourceKind.MODEL, source_ref="model",
                created_at=tightened["updated_at"],
                payload={"proposal_id": "update_unbound_correction",
                         "seed_id": seed["seed_id"],
                         "base_event_id": active_event.event_id, "base_version": 2,
                         "operation": "tighten", "proposed_seed": tightened,
                         "delta_digest": hashlib.sha256(
                             canonical_json(tighten_manifest).encode("utf-8")).hexdigest(),
                         "version": "seed_update_proposal_v1"},
                parent_event_ids=(active_event.event_id, correction.event_id)))

    def test_standing_seed_auto_activation_is_policy_only_and_parent_bound(self):
        session = "standing_seed_apply"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        self.store._trusted_clock = lambda: now + timedelta(seconds=3)
        policy_payload = {"policy_id": "seed_policy_apply", "user_observation_event_id": "evt_policy_apply_approval",
            "nonce": "seed_policy_apply_nonce", "issued_at": (now + timedelta(seconds=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(), "scope": "research",
            "max_auto_activations": 1, "max_auto_updates": 2, "max_active_seeds": 2,
            "max_cue_terms": 4, "max_cue_length": 64, "max_strength": .8,
            "max_confidence": .9, "max_seed_ttl_seconds": 3600, "max_strength_step": .1,
            "max_confidence_step": .1, "max_counterevidence": 10, "max_seed_versions": 8,
            "allowed_policy_bias": ["cite_sources"], "policy_version": "seed_standing_policy_v1"}
        approval = self.store.append(CognitiveEvent(
            event_id="evt_policy_apply_approval", session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", created_at=now.isoformat(),
            payload={"approval": "seed_standing_policy",
                     "policy_digest": SQLiteEventStore.standing_seed_policy_digest(policy_payload),
                     "nonce": policy_payload["nonce"]}))
        policy = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_STANDING_POLICY,
            source_kind=SourceKind.USER, source_ref="user", created_at=policy_payload["issued_at"],
            payload=policy_payload, parent_event_ids=(approval.event_id,)))
        evidence = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
            source_ref="user", created_at=(now + timedelta(seconds=1)).isoformat(),
            payload={"content": "prefer citations"}))
        action = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.ACTION_PROPOSED, source_kind=SourceKind.MODEL,
            source_ref="model", created_at=(now + timedelta(seconds=1)).isoformat(),
            payload={"action_type": "response", "required_capability": None,
                     "is_mutating": False,
                     "public_summary": "Action proposal recorded for policy review."},
            parent_event_ids=(evidence.event_id,)))
        decision_payload = {"turn_id": "standing_turn", "observation_event_ids": [evidence.event_id],
            "retrieved_seed_ids": [], "self_claim_ids": [], "policy_reasons": ["bounded"],
            "selected_action": {"action_type": "response", "required_capability": None,
                "is_mutating": False, "public_summary": "Action proposal recorded for policy review."},
            "public_summary": "Policy decision recorded for this turn."}
        decision = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.DECISION, source_kind=SourceKind.POLICY,
            source_ref="policy", created_at=(now + timedelta(seconds=1)).isoformat(),
            payload=decision_payload, parent_event_ids=(evidence.event_id, action.event_id)))
        seed = {"cue_terms": ["citation"], "policy_bias": "cite_sources", "scope": "research",
                "provenance_event_ids": [evidence.event_id, decision.event_id],
                "strength": .5, "confidence": .6, "status": "candidate",
                "seed_id": "seed_auto_1", "created_at": (now + timedelta(seconds=1)).isoformat(),
                "updated_at": (now + timedelta(seconds=1)).isoformat(), "expires_at": None,
                "version": 1, "counterevidence": 0}
        proposal = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_PROPOSED, source_kind=SourceKind.MODEL,
            source_ref="model", payload={"seed": seed},
            created_at=(now + timedelta(seconds=1)).isoformat(),
            parent_event_ids=(evidence.event_id, decision.event_id)))
        eligibility_payload = {"eligibility_id": "eligibility_1", "policy_id": policy.payload["policy_id"],
            "policy_event_id": policy.event_id, "proposal_event_id": proposal.event_id,
            "seed_id": seed["seed_id"], "base_event_id": None, "base_version": 1,
            "operation": "activate", "decision": "eligible", "reason": "within_policy",
            "auto_activations_used": 0, "auto_updates_used": 0, "active_seeds": 0,
            "evaluated_at": (now + timedelta(seconds=2)).isoformat(),
            "version": "seed_auto_eligibility_v1"}
        eligibility = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_AUTO_ELIGIBILITY,
            source_kind=SourceKind.POLICY, source_ref="StandingSeedPolicy",
            payload=eligibility_payload, parent_event_ids=(proposal.event_id, policy.event_id),
            created_at=eligibility_payload["evaluated_at"]))
        activated_seed = dict(seed, status="active", version=2,
                              updated_at=(now + timedelta(seconds=3)).isoformat(),
                              expires_at=(now + timedelta(hours=1)).isoformat())
        application_payload = {"application_id": "application_1", "eligibility_event_id": eligibility.event_id,
            "policy_id": policy.payload["policy_id"], "policy_event_id": policy.event_id,
            "proposal_event_id": proposal.event_id, "seed_id": seed["seed_id"],
            "base_event_id": None, "base_version": 1, "new_version": 2,
            "operation": "activate", "prior_seed_digest": SQLiteEventStore.seed_snapshot_digest(seed),
            "new_seed_digest": SQLiteEventStore.seed_snapshot_digest(activated_seed),
            "applied_at": activated_seed["updated_at"],
            "version": "seed_auto_application_v1"}
        applied = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_AUTO_APPLIED,
            source_kind=SourceKind.POLICY, source_ref="StandingSeedPolicy",
            payload=application_payload,
            parent_event_ids=(eligibility.event_id, proposal.event_id, policy.event_id),
            created_at=application_payload["applied_at"]))
        self.assertEqual("activate", applied.payload["operation"])
        self.assertTrue(self.store.verify_chain(session))

        expedition_approval = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=4)).isoformat(),
            payload={"content": "run bounded expedition"}))
        authorization_at = now + timedelta(seconds=5)
        authorization = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.EXPEDITION_AUTHORIZATION,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=authorization_at.isoformat(),
            payload={"authorization_id": "guidance_authorization",
                     "approval_event_id": expedition_approval.event_id,
                     "nonce": "guidance_nonce", "issued_at": authorization_at.isoformat(),
                     "expires_at": (authorization_at + timedelta(minutes=5)).isoformat(),
                     "goal_digest": "a" * 64, "host_seed_digest": "b" * 64,
                     "max_calls_per_slice": 1, "slice_seconds": 10,
                     "authorization_seconds": 300, "profile": "public_web_only_v1",
                     "version": "expedition_authorization_v1"},
            parent_event_ids=(expedition_approval.event_id,)))
        consumption_at = now + timedelta(seconds=6)
        consumed = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.EXPEDITION_AUTHORIZATION_CONSUMED,
            source_kind=SourceKind.SYSTEM, source_ref="ExpeditionScheduler",
            created_at=consumption_at.isoformat(),
            payload={"consumption_id": "guidance_consumption",
                     "authorization_id": "guidance_authorization",
                     "consumed_at": consumption_at.isoformat(), "run_id": "guidance_run",
                     "version": "expedition_authorization_consumed_v1"},
            parent_event_ids=(authorization.event_id,)))
        guidance_at = now + timedelta(seconds=7)
        context_payload = {
            "context_id": "guidance_context", "run_id": "guidance_run",
            "directive": "request_human_review",
            "seed_authorities": [{"seed_id": seed["seed_id"],
                                   "authority_event_id": applied.event_id,
                                   "snapshot_digest": SQLiteEventStore.seed_snapshot_digest(activated_seed),
                                   "priority_band": "primary"}],
            "context_digest": "0" * 64, "created_at": guidance_at.isoformat(),
            "version": "expedition_seed_context_v1",
        }
        context_payload["context_digest"] = SQLiteEventStore.expedition_seed_context_digest(context_payload)
        context = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.EXPEDITION_SEED_CONTEXT,
            source_kind=SourceKind.POLICY, source_ref="SeedGuidanceProjector",
            created_at=guidance_at.isoformat(), payload=context_payload,
            parent_event_ids=(consumed.event_id, applied.event_id)))
        self.assertEqual(EventKind.EXPEDITION_SEED_CONTEXT, context.kind)
        self.assertTrue(self.store.verify_chain(session))

        with self.assertRaisesRegex(ValueError, "event source is not allowed"):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.EXPEDITION_SEED_CONTEXT,
                source_kind=SourceKind.MODEL, source_ref="model",
                created_at=(now + timedelta(seconds=8)).isoformat(), payload=dict(
                    context_payload, context_id="model_context",
                    created_at=(now + timedelta(seconds=8)).isoformat()),
                parent_event_ids=(consumed.event_id, applied.event_id)))

        retired = self.store.append(CognitiveEvent(
            session_id=session, kind=EventKind.SEED_RETIRED,
            source_kind=SourceKind.USER, source_ref="user",
            created_at=(now + timedelta(seconds=8)).isoformat(),
            payload={"seed_id": seed["seed_id"], "reason": "user_revoke"},
            parent_event_ids=(applied.event_id,)))
        retired_payload = dict(context_payload, context_id="retired_context",
                               created_at=(now + timedelta(seconds=9)).isoformat(),
                               context_digest="0" * 64)
        retired_payload["context_digest"] = SQLiteEventStore.expedition_seed_context_digest(retired_payload)
        with self.assertRaisesRegex(ValueError, "current active seed authorities"):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.EXPEDITION_SEED_CONTEXT,
                source_kind=SourceKind.POLICY, source_ref="SeedGuidanceProjector",
                created_at=retired_payload["created_at"], payload=retired_payload,
                parent_event_ids=(consumed.event_id, applied.event_id)))
        self.assertEqual(EventKind.SEED_RETIRED, retired.kind)

        # A caller cannot revive an expired policy by supplying an old event
        # timestamp: automatic authority is checked against the trusted host
        # clock before uniqueness or lineage processing.
        self.store._trusted_clock = lambda: now + timedelta(hours=2)
        backdated = dict(eligibility_payload, eligibility_id="eligibility_backdated")
        with self.assertRaisesRegex(ValueError, "trusted host clock"):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.SEED_AUTO_ELIGIBILITY,
                source_kind=SourceKind.POLICY, source_ref="StandingSeedPolicy",
                payload=backdated,
                parent_event_ids=(proposal.event_id, policy.event_id),
                created_at=backdated["evaluated_at"]))

        with self.assertRaises(ValueError):
            self.store.append(CognitiveEvent(
                session_id=session, kind=EventKind.SEED_AUTO_APPLIED,
                source_kind=SourceKind.USER, source_ref="user", payload=application_payload,
                parent_event_ids=(eligibility.event_id, proposal.event_id, policy.event_id)))
