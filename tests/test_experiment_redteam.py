"""End-to-end red-team checks for the bounded offline experiment lane."""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

from strangeloop.capabilities import CapabilityRegistry
from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind
from strangeloop.engine import StrangeloopAgent
from strangeloop.experiment_harness import ExperimentKind, ExperimentRegistryPolicy
from strangeloop.expedition import ExpeditionOutcome
from strangeloop.frontier_learning import FrontierLearningSpec
from strangeloop.providers.kimi_code import (CallableSecretResolver, KimiCodeRuntime,
                                             KimiCodeSettings)
from strangeloop.quota import QuotaController, QuotaSnapshot, QuotaSource
from strangeloop.store import SQLiteEventStore
from strangeloop.tool_session import ToolSession
from strangeloop.tools import ControlledToolExecutor, PublicWebFetch, SafePublicWebSearch, StatelessBrowserRead


class _NoNetworkTransport:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("fixture experiments must precede and avoid web/K3 calls")


def _agent():
    session_id = "experiment-redteam"
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
                             quota_controller=quota, tool_session=tools), tools


class ExperimentRedTeamTests(unittest.TestCase):
    def test_active_v3_verified_fixture_records_reward_td_without_any_web_grant(self):
        """A qualified fixture may update ranking only after its three audit records.

        This calls the host-only branch directly because cadence selection is
        separately scheduler-owned; it must still preserve all runtime gates.
        """
        agent, tools = _agent()
        try:
            goal, seed = "bounded offline fixture", "private-host-seed"
            policy = ExperimentRegistryPolicy((ExperimentKind.FRONTIER_REPLAY,), max_experiments=1)
            spec = FrontierLearningSpec()
            approval = agent.event_store.append(CognitiveEvent(
                session_id=agent.session_id, kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user", payload={
                    "content": agent.expedition_authorization_content(goal, seed, 60, 30, 1,
                        "active", spec, policy), "channel": "expedition_authorization"}))
            issued = datetime.now(timezone.utc)
            authorization = agent.event_store.append(CognitiveEvent(
                session_id=agent.session_id, kind=EventKind.EXPEDITION_AUTHORIZATION,
                source_kind=SourceKind.USER, source_ref="user", parent_event_ids=(approval.event_id,), payload={
                    "authorization_id": "experiment_auth", "approval_event_id": approval.event_id,
                    "nonce": "experiment_nonce", "issued_at": issued.isoformat(),
                    "expires_at": (issued + timedelta(seconds=60)).isoformat(),
                    "goal_digest": __import__("hashlib").sha256(goal.encode()).hexdigest(),
                    "host_seed_digest": __import__("hashlib").sha256(seed.encode()).hexdigest(),
                    "max_calls_per_slice": 1, "slice_seconds": 30, "authorization_seconds": 60,
                    "profile": "public_web_only_v1", "version": "expedition_authorization_v3",
                    "learning_mode": "active", "learner_spec_digest": spec.spec_digest,
                    "experiment_kinds": [ExperimentKind.FRONTIER_REPLAY.value],
                    "experiment_registry_digest": policy.registry_digest, "max_experiments": 1,
                    "experiment_max_trials": policy.max_trials, "experiment_max_steps": policy.max_steps,
                    "experiment_max_wall_ms": policy.max_wall_ms}))
            agent.start_expedition(goal, seed, 60, 30, 1, authorization.event_id, "active", spec, policy)
            # The scheduler must run the user-authorized fixture at least
            # once per six-slice persona cycle.  It may choose it earlier if
            # the active ranker gives it priority.  We close ordinary branches
            # directly so this test makes no model or network call.
            decision = None
            for _ in range(6):
                candidate = agent._expedition.begin_slice()
                if candidate.task.experiment_kind is not None:
                    decision = candidate
                    break
                agent._expedition.record_outcome(candidate.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE)
            self.assertIsNotNone(decision)
            self.assertEqual(ExperimentKind.FRONTIER_REPLAY.value, decision.task.experiment_kind)
            agent._frontier_begin_transition(decision)
            result = agent._run_frontier_experiment(decision)
            events = agent.event_store.list(agent.session_id)
            kinds = [event.kind for event in events]
            plan_i = kinds.index(EventKind.EXPERIMENT_PLAN_LOCKED)
            start_i = kinds.index(EventKind.EXPERIMENT_EXECUTION_STARTED)
            result_i = kinds.index(EventKind.EXPERIMENT_RESULT)
            evidence_i = kinds.index(EventKind.FRONTIER_EVIDENCE_OBSERVATION)
            reward_i = kinds.index(EventKind.FRONTIER_VECTOR_REWARD)
            td_i = kinds.index(EventKind.FRONTIER_TD_UPDATE)
            self.assertTrue(result["reward_qualified"])
            self.assertLess(plan_i, start_i)
            self.assertLess(start_i, result_i)
            self.assertLess(result_i, evidence_i)
            self.assertLess(evidence_i, reward_i)
            self.assertLess(reward_i, td_i)
            self.assertIsNone(tools.research_autonomy_status())
            self.assertEqual([], [event for event in events if event.kind == EventKind.CAPABILITY_GRANTED])
            self.assertEqual([], [event for event in events if event.kind == EventKind.TOOL_EXECUTION_STARTED])
        finally:
            agent.event_store.close()


if __name__ == "__main__":
    unittest.main()
