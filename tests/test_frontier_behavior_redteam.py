"""Behavior-level red test for whether experiment reward reaches a later choice.

The frontier learner is an auditable ranking mechanism inspired by Yogacara;
this test concerns only scheduler eligibility and recorded TD provenance, not
any claim about subjective experience or an intrinsic self.
"""

from __future__ import annotations

import hashlib
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from strangeloop.capabilities import CapabilityRegistry
from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind
from strangeloop.engine import StrangeloopAgent
from strangeloop.experiment_harness import ExperimentKind, ExperimentRegistryPolicy
from strangeloop.expedition import ExpeditionOutcome
from strangeloop.frontier_learning import FrontierLearningSpec
from strangeloop.providers.kimi_code import (CallableSecretResolver, KimiCodeRuntime,
                                             KimiCodeSettings, ToolIntent)
from strangeloop.quota import (ForegroundRefreshStatus, QuotaController, QuotaSnapshot,
                               QuotaSource)
from strangeloop.store import SQLiteEventStore
from strangeloop.tool_session import ToolSession
from strangeloop.tools import (ControlledToolExecutor, PublicWebFetch, SafePublicWebSearch,
                               StatelessBrowserRead)


class _NoNetworkTransport:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("fixture experiments must not use web/K3 calls")


def _agent(session_id):
    store = SQLiteEventStore(session_id=session_id)
    tools = ToolSession(
        session_id, "workspace", CapabilityRegistry(), ControlledToolExecutor(os.getcwd()), event_store=store,
        web_fetch=PublicWebFetch(()), web_search=SafePublicWebSearch(PublicWebFetch(())),
        browser_read=StatelessBrowserRead(PublicWebFetch(())))
    runtime = KimiCodeRuntime(KimiCodeSettings(max_calls=1000),
                              CallableSecretResolver(lambda: "fixture-key"), _NoNetworkTransport())
    quota = QuotaController()
    now = datetime.now(timezone.utc)
    quota.ingest_snapshot(QuotaSnapshot(10, 10, now + timedelta(hours=1), now, 1.0, False),
                          QuotaSource.PROVIDER_USAGE, "fixture", now=now)
    return StrangeloopAgent(session_id=session_id, event_store=store, runtime=runtime,
                             quota_controller=quota, tool_session=tools)


def _start(agent, mode, seed):
    goal = "frontier TD scheduling fixture"
    # Multiple kinds and future instances make a later scheduler choice a
    # genuine active-vs-baseline comparison, not a replay of one task ID.
    policy = ExperimentRegistryPolicy((ExperimentKind.FRONTIER_REPLAY,
                                       ExperimentKind.TD_INVARIANTS), max_experiments=8)
    spec = FrontierLearningSpec()
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
        source_ref="user", payload={"content": agent.expedition_authorization_content(
            goal, seed, 60, 30, 1, mode, spec, policy), "channel": "expedition_authorization"}))
    issued = datetime.now(timezone.utc)
    authorization = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.EXPEDITION_AUTHORIZATION,
        source_kind=SourceKind.USER, source_ref="user", parent_event_ids=(approval.event_id,), payload={
            "authorization_id": "%s-auth" % mode, "approval_event_id": approval.event_id,
            "nonce": "%s-nonce" % mode, "issued_at": issued.isoformat(),
            "expires_at": (issued + timedelta(seconds=60)).isoformat(),
            "goal_digest": hashlib.sha256(goal.encode()).hexdigest(),
            "host_seed_digest": hashlib.sha256(seed.encode()).hexdigest(),
            "max_calls_per_slice": 1, "slice_seconds": 30, "authorization_seconds": 60,
            "profile": "public_web_only_v1", "version": "expedition_authorization_v3",
            "learning_mode": mode, "learner_spec_digest": spec.spec_digest,
            "experiment_kinds": [item.value for item in policy.approved_kinds],
            "experiment_registry_digest": policy.registry_digest, "max_experiments": policy.max_experiments,
            "experiment_max_trials": policy.max_trials, "experiment_max_steps": policy.max_steps,
            "experiment_max_wall_ms": policy.max_wall_ms}))
    agent.start_expedition(goal, seed, 60, 30, 1, authorization.event_id, mode, spec, policy)


def _scheduler_selected_experiment(agent):
    for _ in range(6):
        decision = agent._expedition.begin_slice()
        if decision.task.experiment_kind is not None:
            return decision
        agent._expedition.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE)
    raise AssertionError("scheduler did not select its authorized experiment")


class _FreshQuotaAdapter:
    def refresh_foreground_slice(self, quota, gate, observed, force=False):
        del quota, gate, force
        return ForegroundRefreshStatus(True, True, True, False,
                                       "authoritative_quota_available", None, observed,
                                       observed + timedelta(seconds=30),
                                       "provider_plan_units_not_currency", None)


class FrontierBehaviorRedTeamTests(unittest.TestCase):
    def test_qualified_experiment_td_must_target_a_subsequent_eligible_action(self):
        """RED: a qualified reward must not be stranded on the completed task ID.

        Two active schedulers begin from the same seed and take the same first
        scheduler action.  Only the treatment executes the verified reward
        path, isolating the later selection difference to the TD update.
        """
        active, control = _agent("causal-treatment"), _agent("causal-control")
        try:
            seed = "causal-seed-5"
            _start(active, "active", seed)
            _start(control, "active", seed)
            active_decision = _scheduler_selected_experiment(active)
            control_decision = _scheduler_selected_experiment(control)
            self.assertEqual(active_decision.task.task_id, control_decision.task.task_id)
            active._frontier_begin_transition(active_decision)
            result = active._run_frontier_experiment(active_decision)
            self.assertTrue(result["reward_qualified"])
            active._expedition.record_outcome(
                active_decision.task.task_id, ExpeditionOutcome.EXPERIMENT_SUPPORTED, quality_score=.70,
                quality_signals=("paired_baseline_treatment", "independent_reproduction"))
            active._expedition_active_transition = None

            td_update = [event for event in active.event_store.list(active.session_id)
                         if event.kind == EventKind.FRONTIER_TD_UPDATE][-1]
            completed_task_id = active_decision.task.task_id
            ranking = [event for event in active.event_store.list(active.session_id)
                       if event.kind == EventKind.FRONTIER_RANKING_DECISION
                       and event.payload["transition_id"] == td_update.payload["transition_id"]][-1]
            self.assertEqual(completed_task_id, ranking.payload["frontier_task_id"])
            completed = [item for item in active._expedition.public_snapshot()["tasks"]
                         if item["task_id"] == completed_task_id][0]
            self.assertEqual("complete", completed["status"])
            # Compare the immediate next scheduler decision.  Both agents
            # received the same scheduler outcome; only treatment has TD.
            control._expedition.record_outcome(control_decision.task.task_id,
                                               ExpeditionOutcome.EXPERIMENT_SUPPORTED, quality_score=.70,
                                               quality_signals=("paired_baseline_treatment", "independent_reproduction"))
            active_next = active._expedition.begin_slice()
            control_next = control._expedition.begin_slice()
            eligible_ids = {item["task_id"] for item in active._expedition.candidate_snapshot()["candidates"]}
            self.assertNotIn(completed_task_id, eligible_ids)
            successor = next(item for item in active._expedition.public_snapshot()["tasks"]
                             if item["strategy_arm_id"] == ranking.payload["strategy_arm_id"]
                             and item["task_id"] != completed_task_id)
            self.assertEqual(ranking.payload["strategy_arm_id"], successor["strategy_arm_id"])
            self.assertNotEqual(completed_task_id, successor["task_id"])
            self.assertEqual(ranking.payload["strategy_arm_id"], td_update.payload["strategy_arm_id"])
            self.assertNotEqual(active_next.task.task_id, control_next.task.task_id)
            self.assertNotEqual(completed_task_id, active_next.task.task_id)
            self.assertEqual(ranking.payload["strategy_arm_id"], active_next.task.strategy_arm_id)
            self.assertNotEqual(ranking.payload["strategy_arm_id"], control_next.task.strategy_arm_id)
            self.assertIn(active_next.selection_reason,
                          ("experiment_cadence_due_ranker", "ranker_active_recommendation"))
            active._frontier_begin_transition(active_next)
            self.assertGreater(active.expedition_status()["frontier_learning"]["post_reward_selection_count"], 0)
        finally:
            active.event_store.close()
            control.event_store.close()

    def test_duplicate_canonical_result_has_no_second_reward_or_td_update(self):
        agent = _agent("frontier-duplicate-result")
        try:
            _start(agent, "active", "causal-seed-5")
            agent.runtime.plan_tools = lambda prompt, context: (
                ToolIntent("done", "respond", {"message": "branch complete"}, "done"),)
            adapter = _FreshQuotaAdapter()
            first_rewards = first_updates = None
            first_task_id = first_arm_id = second_task_id = None
            for _ in range(16):
                prior_count = sum(event.kind == EventKind.EXPERIMENT_RESULT
                                  for event in agent.event_store.list(agent.session_id))
                agent.expedition_slice(adapter)
                events = agent.event_store.list(agent.session_id)
                if sum(event.kind == EventKind.EXPERIMENT_RESULT for event in events) == prior_count:
                    continue
                ranking = [event for event in events
                           if event.kind == EventKind.FRONTIER_RANKING_DECISION][-1]
                if first_rewards is None:
                    first_task_id, first_arm_id = (ranking.payload["frontier_task_id"],
                                                    ranking.payload["strategy_arm_id"])
                    first_rewards = sum(event.kind == EventKind.FRONTIER_VECTOR_REWARD for event in events)
                    first_updates = sum(event.kind == EventKind.FRONTIER_TD_UPDATE for event in events)
                else:
                    second_task_id = ranking.payload["frontier_task_id"]
                    self.assertEqual(first_arm_id, ranking.payload["strategy_arm_id"])
                    break
            self.assertIsNotNone(second_task_id)
            self.assertNotEqual(first_task_id, second_task_id)
            events = agent.event_store.list(agent.session_id)
            self.assertEqual(first_rewards, sum(event.kind == EventKind.FRONTIER_VECTOR_REWARD for event in events))
            self.assertEqual(first_updates, sum(event.kind == EventKind.FRONTIER_TD_UPDATE for event in events))
            duplicate_task = next(item for item in agent._expedition.public_snapshot()["tasks"]
                                  if item["task_id"] == second_task_id)
            self.assertEqual("complete", duplicate_task["status"])
            self.assertEqual(0, duplicate_task["low_value_results"])
            self.assertEqual(2, agent.expedition_status()["experiment"]["completed_count"])
            self.assertEqual(1, agent.expedition_status()["experiment"]["duplicate_experiment_result_count"])
            self.assertEqual("branch_complete", [item for item in agent._expedition._history
                                                   if item["task_id"] == second_task_id][-1]["outcome"])
            next_decision = agent._expedition.begin_slice()
            self.assertNotEqual(second_task_id, next_decision.task.task_id)
        finally:
            agent.event_store.close()

    def test_reward_append_failure_restores_learner_and_stops_scheduler(self):
        agent = _agent("frontier-reward-rollback")
        try:
            _start(agent, "active", "reward-rollback-seed")
            decision = _scheduler_selected_experiment(agent)
            agent._frontier_begin_transition(decision)
            original = agent.event_store.append

            def reject_reward(event, *args, **kwargs):
                if event.kind == EventKind.FRONTIER_VECTOR_REWARD:
                    raise RuntimeError("fixture reward write failure")
                return original(event, *args, **kwargs)

            with patch.object(agent.event_store, "append", side_effect=reject_reward):
                with self.assertRaises(RuntimeError):
                    agent._run_frontier_experiment(decision)
            self.assertEqual("stopped", agent.expedition_status()["state"])
            events = agent.event_store.list(agent.session_id)
            self.assertEqual(0, sum(event.kind == EventKind.FRONTIER_VECTOR_REWARD for event in events))
            self.assertEqual(0, sum(event.kind == EventKind.FRONTIER_TD_UPDATE for event in events))
            self.assertEqual((0.0,) * 5, agent._expedition_learner.values_for(
                agent._expedition_frontier_state_id, decision.task.strategy_arm_id))
        finally:
            agent.event_store.close()


if __name__ == "__main__":
    unittest.main()
