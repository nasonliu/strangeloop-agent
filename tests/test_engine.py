import os
import io
import struct
import tempfile
import unittest
import json
import threading
from uuid import uuid4

from strangeloop.contracts import (ActionProposal, CognitiveEvent, Deliberation,
                                  EventKind, SeedDisposition, SourceKind)
from strangeloop.engine import StrangeloopAgent
from strangeloop.deliberation import TransparentHeuristicDeliberator
from strangeloop.autoloop import LoopConfig
from strangeloop.seeds import (SQLiteSeedStore, SeedStandingPolicy,
                               standing_policy_manifest)
from strangeloop.self_model import EventSourcedSelfModel
from strangeloop.store import SQLiteEventStore
from strangeloop.td import RewardSource
from strangeloop.td import ValueTable
from strangeloop.providers.kimi_code import (CallableSecretResolver,
                                             KimiCodeProviderError,
                                             KimiCodeRuntime, KimiCodeSettings, ToolIntent)
from strangeloop.quota import (ForegroundRefreshStatus, QuotaController,
                               QuotaSnapshot, QuotaSource)
from strangeloop.drives import DualIntrinsicDrives
from strangeloop.capabilities import (Capability, CapabilityGrant, CapabilityRegistry, GrantScope,
                                      ResearchAutonomyProfile, ResearchBudget)
from strangeloop.tools import (ControlledToolExecutor, PublicWebFetch, SafePublicWebSearch,
                               StatelessBrowserRead, ToolOutcome, ToolStatus)
from strangeloop.tool_session import ToolSession
from strangeloop.unattended import UnattendedPolicy
from strangeloop.expedition import ExpeditionOutcome
from datetime import datetime, timedelta, timezone
from strangeloop.sleep import SleepWakeCoordinator, SleepWakePolicy
from strangeloop.kimi_cli import KimiCliUsageResult, ManagedUsageWindow
from strangeloop.quota import QuotaSnapshot


def png_fixture():
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" +
            struct.pack(">II", 2, 3) + b"\x08\x02\x00\x00\x00")


class CandidateDeliberator:
    def deliberate(self, prompt, workspace):
        return Deliberation(
            response_text="A concise response.", hypotheses=("request",), uncertainties=(),
            alternatives=("respond",), action=ActionProposal("response", "respond"),
            proposed_seed=SeedDisposition(cue_terms=("receipt",), policy_bias="review", scope="conversation", provenance_event_ids=("placeholder",)),
        )


class ClaimingDeliberator:
    def deliberate(self, prompt, workspace):
        return Deliberation("I am conscious.", (), (), (), ActionProposal("response", "respond"))


class NonResponseDeliberator:
    def deliberate(self, prompt, workspace):
        return Deliberation("I would read a file.", (), (), (), ActionProposal("read", "inspect"))


class PrivateDeliberator:
    secret = "PRIVATE_CHAIN_OF_THOUGHT_7fa3"

    def deliberate(self, prompt, workspace):
        return Deliberation(
            "A safe public response.", (self.secret,), (self.secret,), (self.secret,),
            ActionProposal("response", self.secret,
                           {"chain_of_thought": self.secret, "debug": self.secret,
                            "private": self.secret}, required_capability=self.secret),
        )


class MaliciousSeedDeliberator:
    marker = "MODEL_SEED_SECRET_919"

    def deliberate(self, prompt, workspace):
        malicious_seed = SeedDisposition(
            cue_terms=(self.marker,), policy_bias=self.marker, scope=self.marker,
            provenance_event_ids=(self.marker,), strength=.99, confidence=.99,
            seed_id=self.marker, created_at=self.marker, updated_at=self.marker,
        )
        return Deliberation("Safe response.", (), (), (),
                           ActionProposal("response", "safe"), malicious_seed)


class NoSeedDeliberator:
    def deliberate(self, prompt, workspace):
        del prompt, workspace
        return Deliberation(
            "A bounded response.", (), (), (),
            ActionProposal("response", "respond"), None)


class GuidanceCapturingDeliberator:
    def __init__(self):
        self.workspaces = []

    def deliberate(self, prompt, workspace):
        del prompt
        self.workspaces.append(workspace)
        return Deliberation("A bounded response.", (), (), (),
                           ActionProposal("response", "respond"), None)


def enable_seed_auto_update(agent, **limits):
    issued = datetime.now(timezone.utc)
    issued_at = issued.isoformat()
    approval_event_id = "evt_%s" % uuid4().hex
    policy_id = "seedpolicy_%s" % uuid4().hex
    nonce = "nonce_%s" % uuid4().hex
    policy = SeedStandingPolicy(
        expires_at=(issued + timedelta(days=7)).isoformat(), **limits)
    _payload, digest = standing_policy_manifest(
        policy, policy_id, approval_event_id, nonce, issued_at)
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="test_user_command",
        payload={"approval": "seed_standing_policy",
                 "policy_digest": digest, "nonce": nonce},
        event_id=approval_event_id, created_at=issued_at,
    ))
    agent.enable_seed_auto_update(
        approval.event_id, policy, policy_id, nonce, issued_at)
    return policy_id


def disable_seed_auto_update(agent):
    current = agent.seed_store.standing_policy_status(agent.session_id)
    approval = agent.event_store.append(CognitiveEvent(
        session_id=agent.session_id, kind=EventKind.OBSERVATION,
        source_kind=SourceKind.USER, source_ref="test_user_command",
        payload={"approval": "seed_standing_policy_revoke",
                 "policy_id": current["policy_id"],
                 "policy_event_id": current["event_id"]},
    ))
    return agent.disable_seed_auto_update(approval.event_id)


class FakeK3Transport:
    def __init__(self, responses=(), error=False):
        self.responses = list(responses)
        self.error = error
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, payload))
        if self.error:
            raise KimiCodeProviderError("redacted")
        return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}


def k3_runtime(transport):
    return KimiCodeRuntime(KimiCodeSettings(),
                           CallableSecretResolver(lambda: "test-only-key"), transport)


class EngineTests(unittest.TestCase):
    def test_unattended_report_uses_prior_untrusted_public_results_without_extra_synthesis(self):
        runtime = k3_runtime(FakeK3Transport())
        contexts = []
        intents = [ToolIntent("intent_1", "web_search", {"query": "public docs"}, "Search public docs."),
                   ToolIntent("intent_2", "browser_read", {"url": "https://example.com/docs"}, "Read result."),
                   ToolIntent("intent_3", "respond", {"message": "complete"}, "Research complete.")]

        def planned(prompt, public_context):
            del prompt
            contexts.append(public_context)
            return (intents.pop(0),)

        runtime.plan_tools = planned
        class StaticPublicBackend:
            def __init__(self): self.calls = 0
            def execute(self, plan, registry):
                self.calls += 1
                registry.consume(plan)
                return ToolOutcome.from_bytes(plan.tool_name, ToolStatus.SUCCEEDED,
                                              ("public summary for " + plan.tool_name).encode("utf-8"))

        store, registry, fetch = SQLiteEventStore(), CapabilityRegistry(), PublicWebFetch(())
        public = StaticPublicBackend()
        session = ToolSession("research-report", "workspace", registry, ControlledToolExecutor(os.getcwd()),
                              event_store=store, web_fetch=fetch, web_search=public, browser_read=public)
        agent = StrangeloopAgent(session_id="research-report", event_store=store,
                                 runtime=runtime, tool_session=session)
        approval = store.append(CognitiveEvent(session_id="research-report", kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={"content": "research", "channel": "cli_command"}))
        policy = UnattendedPolicy.user_issued("inspect public documentation",
            datetime.now(timezone.utc) + timedelta(minutes=1))
        agent.start_unattended_research(ResearchAutonomyProfile("workspace", ResearchBudget(
            max_tool_calls=3, max_total_bytes=65536, max_response_bytes=4096,
            max_wall_ms=10000, ttl_seconds=60)), policy, approval.event_id)
        outcome = agent.run_unattended_research()
        self.assertEqual(2, public.calls)
        self.assertEqual("no_work", outcome["public_report"]["stop_reason"])
        self.assertEqual(2, len(outcome["public_report"]["findings"]))
        self.assertEqual("web.search", outcome["public_report"]["findings"][0]["tool"])
        self.assertEqual("succeeded", outcome["public_report"]["findings"][0]["status"])
        self.assertEqual("succeeded", outcome["public_report"]["actions"][0]["status"])
        self.assertIn("untrusted_data", contexts[1]["previous_public_results"])
        self.assertEqual("web.search", contexts[1]["previous_public_results"]["untrusted_data"][0]["tool"])
        self.assertEqual("succeeded", contexts[1]["previous_public_results"]["untrusted_data"][0]["status"])
        # The report is deterministic local projection, so the model is called
        # exactly once per planning tick, not for a final synthesis pass.
        self.assertEqual(3, len(contexts))
        state = agent.state()["unattended_research"]
        self.assertEqual(2, state["finding_count"])
        self.assertNotIn("public summary", repr(state))

    def test_unattended_quota_exhaustion_during_k3_planning_cannot_reach_tool_execution(self):
        now = datetime.now(timezone.utc)
        controller = QuotaController()
        controller.ingest_snapshot(QuotaSnapshot(10, 1, now + timedelta(hours=1), now, 1.0, False),
                                   QuotaSource.PROVIDER_USAGE, "system")
        runtime = k3_runtime(FakeK3Transport())
        runtime._quota_controller = controller

        def exhaust_after_planning(prompt, public_context):
            del prompt, public_context
            controller.ingest_snapshot(QuotaSnapshot(10, 0, now + timedelta(hours=1),
                                                     datetime.now(timezone.utc), 1.0, False),
                                       QuotaSource.PROVIDER_USAGE, "system")
            return (ToolIntent("intent_1", "repo_status", {}, "Inspect repository status."),)

        runtime.plan_tools = exhaust_after_planning
        store, registry = SQLiteEventStore(), CapabilityRegistry()
        fetch = PublicWebFetch(())
        class CountingExecutor(ControlledToolExecutor):
            def __init__(self):
                super().__init__(os.getcwd())
                self.calls = 0

            def execute(self, *args, **kwargs):
                self.calls += 1
                return super().execute(*args, **kwargs)

        executor = CountingExecutor()
        session = ToolSession("research-gate", "workspace", registry, executor,
                              event_store=store, web_fetch=fetch, web_search=SafePublicWebSearch(fetch),
                              browser_read=StatelessBrowserRead(fetch))
        agent = StrangeloopAgent(session_id="research-gate", event_store=store, runtime=runtime,
                                 quota_controller=controller, tool_session=session)
        approval = store.append(CognitiveEvent(session_id="research-gate", kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={"content": "research", "channel": "cli_command"}))
        profile = ResearchAutonomyProfile("workspace", ResearchBudget(
            max_tool_calls=2, max_total_bytes=65536, max_response_bytes=4096,
            max_wall_ms=10000, ttl_seconds=60))
        policy = UnattendedPolicy.user_issued("inspect public metadata", now + timedelta(minutes=1), now)
        agent.start_unattended_research(profile, policy, approval.event_id)
        result = agent.unattended_step()
        self.assertEqual("quota_exhausted", result["status"]["stop_reason"])
        self.assertIsNone(result["receipt"]["action"])
        kinds = [event.kind for event in store.list("research-gate")]
        self.assertNotIn(EventKind.TOOL_CALL_PROPOSED, kinds)
        self.assertNotIn(EventKind.TOOL_EXECUTION_STARTED, kinds)
        self.assertNotIn(EventKind.TOOL_RESULT, kinds)
        self.assertEqual(0, executor.calls)

    def test_unattended_research_reuses_k3_planning_and_controlled_read_only_tools(self):
        transport = FakeK3Transport([{"intents": [{"tool_name": "repo_status", "arguments": {},
                                                     "rationale_summary": "Inspect repository status."}]}])
        runtime = k3_runtime(transport)
        registry = CapabilityRegistry()
        fetch = PublicWebFetch(())
        store = SQLiteEventStore()
        session = ToolSession("research", "workspace", registry, ControlledToolExecutor(os.getcwd()),
                              web_fetch=fetch, web_search=SafePublicWebSearch(fetch),
                              browser_read=StatelessBrowserRead(fetch), event_store=store)
        agent = StrangeloopAgent(session_id="research", event_store=store, runtime=runtime, tool_session=session)
        approval = agent.event_store.append(CognitiveEvent(
            session_id="research", kind=EventKind.OBSERVATION, source_kind=SourceKind.USER,
            source_ref="user", payload={"content": "run read-only public research", "channel": "cli_command"}))
        profile = ResearchAutonomyProfile("workspace", ResearchBudget(
            max_tool_calls=2, max_total_bytes=65536, max_response_bytes=4096,
            max_wall_ms=10000, ttl_seconds=60))
        policy = UnattendedPolicy.user_issued("inspect public repository metadata",
                                               datetime.now(timezone.utc) + timedelta(minutes=1))
        status = agent.start_unattended_research(profile, policy, approval.event_id)
        self.assertTrue(status["active"])
        result = agent.unattended_step()
        self.assertEqual("repo.status", result["receipt"]["action"]["tool_name"])
        self.assertEqual("executed", result["receipt"]["action"]["disposition"])
        self.assertTrue(agent.event_store.verify_chain("research"))
        self.assertTrue(any(event.kind == EventKind.TOOL_RESULT
                            for event in agent.event_store.list("research")))
        self.assertTrue(agent.stop_research_autonomy("sleep"))
        self.assertFalse(agent.unattended_status()["active"])

    def test_quota_pause_skips_k3_deliberation_and_uses_public_fallback(self):
        controller = QuotaController()
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        controller.ingest_snapshot(QuotaSnapshot(10, 0, now + __import__("datetime").timedelta(hours=1),
                                                 now, 1.0, False), QuotaSource.PROVIDER_USAGE, "system")
        transport = FakeK3Transport()
        runtime = KimiCodeRuntime(KimiCodeSettings(), CallableSecretResolver(lambda: "test-only-key"),
                                  transport, quota_controller=controller)
        result = StrangeloopAgent(session_id="quota-paused", runtime=runtime,
                                  quota_controller=controller).run_turn("local fallback")
        self.assertEqual([], transport.calls)
        self.assertIn("quota controller", result.notices[0])

    def test_quota_status_labels_provider_plan_and_local_ledger_separately(self):
        now = datetime.now(timezone.utc)
        controller = QuotaController()
        controller.ingest_snapshot(QuotaSnapshot(100, 96, now + timedelta(hours=5), now,
                                                 1.0, False, primary_unit="provider_units",
                                                 primary_window_kind="rolling_5h"),
                                   QuotaSource.PROVIDER_USAGE, "system")
        status = StrangeloopAgent(session_id="quota-status", quota_controller=controller).quota_status()
        self.assertEqual("provider_usage", status["authority"])
        self.assertEqual("fresh", status["freshness"])
        self.assertEqual((96, 100), (status["remaining"], status["total"]))
        self.assertEqual("provider_units", status["primary_unit"])
        self.assertEqual("rolling_5h", status["primary_window_kind"])
        self.assertTrue(status["allow_call"])
        self.assertTrue(status["is_code_plan_balance"])
        self.assertEqual((0, 0), (status["local_observed_calls"], status["local_observed_tokens"]))
        self.assertNotIn("access_token", repr(status))

    def test_quota_status_drops_invalid_or_non_authoritative_primary_window_kind(self):
        now = datetime.now(timezone.utc)
        controller = QuotaController()
        controller.ingest_snapshot(QuotaSnapshot(10, 8, now + timedelta(hours=5), now,
                                                 1.0, False, primary_unit="provider_units"),
                                   QuotaSource.MANUAL_SNAPSHOT, "system")
        status = StrangeloopAgent(session_id="quota-window-manual", quota_controller=controller).quota_status()
        self.assertNotIn("primary_window_kind", status)

    def test_quota_status_without_controller_fails_closed_for_monitor(self):
        status = StrangeloopAgent(session_id="quota-none").quota_status()
        self.assertFalse(status["configured"])
        self.assertFalse(status["allow_call"])
        self.assertEqual(("unknown", "unknown", "not_configured"),
                         (status["authority"], status["freshness"], status["reason"]))

    def test_target_bound_feedback_updates_only_alignment_drive(self):
        drives = DualIntrinsicDrives()
        agent = StrangeloopAgent(session_id="feedback", drives=drives)
        result = agent.run_turn("one")
        before = drives.values()
        outcome = agent.record_user_feedback(result.event_ids[-1], "accept")
        after = drives.values()
        self.assertEqual(1.0, outcome["reward"]["user_alignment"])
        self.assertEqual(before["operational_integrity"], after["operational_integrity"])
        self.assertEqual(before["epistemic_progress"], after["epistemic_progress"])
        self.assertGreater(after["user_alignment"], before["user_alignment"])
        mirror = [event for event in agent.event_store.list(agent.session_id)
                  if event.kind == EventKind.METACOGNITIVE_MIRROR][0]
        with self.assertRaises(ValueError):
            agent.record_user_feedback(mirror.event_id, "accept")

    def test_k3_tool_planning_has_completed_lifecycle_before_host_preparation(self):
        transport = FakeK3Transport([{"intents": [{"tool_name": "repo_status", "arguments": {},
                                                     "rationale_summary": "Inspect public repository status."}]}])
        runtime = k3_runtime(transport)
        registry = CapabilityRegistry()
        session = ToolSession("tool-plan", "workspace", registry, ControlledToolExecutor(os.getcwd()))
        agent = StrangeloopAgent(session_id="tool-plan", runtime=runtime, tool_session=session)
        command = agent.event_store.append(CognitiveEvent(session_id="tool-plan", kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={"content": "grant"}))
        session.register_user_grant(CapabilityGrant("tool-plan", Capability.REPO_STATUS,
            GrantScope(workspace_id="workspace"), (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(), 1), command.event_id)
        prepared = agent.plan_tools("show repository status")
        self.assertEqual(1, len(prepared))
        invocations = [event for event in agent.event_store.list("tool-plan")
                       if event.kind == EventKind.MODEL_INVOCATION]
        self.assertEqual(["started", "completed"], [event.payload["outcome"] for event in invocations])
        self.assertEqual(invocations[-1].event_id, prepared[0].model_invocation_event_id)

    def test_k3_tool_planning_timeout_records_terminal_timeout_and_executes_no_tool(self):
        def timeout_transport(*args):
            raise KimiCodeProviderError("private timeout reason", category="provider_timeout")

        runtime = k3_runtime(timeout_transport)
        session = ToolSession("tool-timeout", "workspace", CapabilityRegistry(),
                              ControlledToolExecutor(os.getcwd()))
        agent = StrangeloopAgent(session_id="tool-timeout", runtime=runtime, tool_session=session)
        self.assertEqual((), agent.plan_tools("inspect the repository"))
        events = agent.event_store.list("tool-timeout")
        invocations = [event for event in events if event.kind == EventKind.MODEL_INVOCATION]
        self.assertEqual(["started", "timed_out"], [event.payload["outcome"] for event in invocations])
        self.assertEqual("K3 tool planning timed out; no tool was executed.",
                         invocations[-1].payload["public_summary"])
        self.assertEqual([], [event for event in events if event.kind == EventKind.TOOL_RESULT])
        self.assertNotIn("private timeout reason", repr([event.payload for event in events]))

    def test_low_authoritative_five_hour_window_enters_sleep_before_k3_call(self):
        now = datetime.now(timezone.utc)
        class FixedClock:
            def __call__(self): return now
        snapshot = QuotaSnapshot(10, 1, now + timedelta(hours=1), now, 1.0, False)
        controller = QuotaController()
        controller.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE, "system", now=now)
        usage = __import__("strangeloop.kimi_cli", fromlist=["KimiCliUsageResult"]).KimiCliUsageResult(
            snapshot, None, windows=(ManagedUsageWindow("rolling_5h", 10, 1, now + timedelta(hours=1)),))
        transport = FakeK3Transport()
        agent = StrangeloopAgent(session_id="sleep-pre", runtime=k3_runtime(transport),
            quota_controller=controller, sleep_coordinator=SleepWakeCoordinator(SleepWakePolicy(threshold=.10), FixedClock()))
        self.assertTrue(agent.update_managed_usage(usage))
        result = agent.run_turn("must not call k3")
        self.assertEqual([], transport.calls)
        self.assertEqual("sleeping", agent.sleep_status()["state"])
        self.assertIn("fallback", result.notices[0])
        self.assertTrue(agent.event_store.verify_chain("sleep-pre"))

    def test_exhausted_authoritative_window_is_archived_even_when_calls_are_denied(self):
        now = datetime.now(timezone.utc)
        snapshot = QuotaSnapshot(100, 0, now + timedelta(hours=1), now, 1.0, False,
                                 primary_unit="provider_units")
        controller = QuotaController()
        self.assertTrue(controller.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE,
                                                   "system", now=now))
        result = KimiCliUsageResult(snapshot, None, windows=(
            ManagedUsageWindow("rolling_5h", 100, 0, now + timedelta(hours=1)),))
        coordinator = SleepWakeCoordinator(SleepWakePolicy(threshold=.10), clock=lambda: now)
        agent = StrangeloopAgent(session_id="sleep-zero", quota_controller=controller,
                                 sleep_coordinator=coordinator)
        self.assertTrue(agent.update_managed_usage(result))
        self.assertEqual("sleeping", agent.sleep_status()["state"])
        self.assertEqual([EventKind.SLEEP_ARCHIVE, EventKind.SLEEP_ENTERED],
                         [event.kind for event in agent.event_store.list("sleep-zero")])
        self.assertTrue(agent.event_store.verify_chain("sleep-zero"))

    def test_sleep_archive_pair_rolls_back_and_closes_wake_on_store_failure(self):
        now = datetime.now(timezone.utc)

        class FailingSleepStore(SQLiteEventStore):
            def _append(self, event):
                if event.kind == EventKind.SLEEP_ENTERED:
                    raise RuntimeError("injected sleep edge failure")
                return super()._append(event)

        snapshot = QuotaSnapshot(10, 1, now + timedelta(hours=1), now, 1.0, False)
        controller, store = QuotaController(), FailingSleepStore()
        controller.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE, "system", now=now)
        result = KimiCliUsageResult(snapshot, None, windows=(
            ManagedUsageWindow("rolling_5h", 10, 1, now + timedelta(hours=1)),))
        agent = StrangeloopAgent(session_id="sleep-rollback", event_store=store,
                                 quota_controller=controller,
                                 sleep_coordinator=SleepWakeCoordinator(
                                     SleepWakePolicy(threshold=.10), clock=lambda: now))
        with self.assertRaises(RuntimeError):
            agent.update_managed_usage(result)
        self.assertEqual("terminal", agent.sleep_status()["state"])
        self.assertEqual([], store.list("sleep-rollback"))

    def test_expedition_reuses_one_user_authority_and_mints_only_public_web_grants(self):
        now = datetime.now(timezone.utc)
        snapshot = QuotaSnapshot(100, 90, now + timedelta(hours=1), now, 1.0, False,
                                 primary_unit="provider_units")
        controller = QuotaController()
        controller.ingest_snapshot(snapshot, QuotaSource.PROVIDER_USAGE, "system", now=now)
        usage = KimiCliUsageResult(snapshot, None, windows=(
            ManagedUsageWindow("rolling_5h", 100, 90, now + timedelta(hours=1)),))

        class Adapter:
            def refresh_foreground_slice(self, quota, gate, observed, force=False):
                del quota, gate, observed, force
                return ForegroundRefreshStatus(True, True, True, False,
                    "authoritative_quota_available", None, now,
                    now + timedelta(seconds=30), "provider_plan_units_not_currency", usage)

        runtime = k3_runtime(FakeK3Transport())
        runtime.plan_tools = lambda prompt, context: (
            ToolIntent("done", "respond", {"message": "branch complete"}, "done"),)
        store, registry = SQLiteEventStore(), CapabilityRegistry()
        fetch = PublicWebFetch(())
        tools = ToolSession("expedition-auth", "workspace", registry,
                            ControlledToolExecutor(os.getcwd()), event_store=store,
                            web_fetch=fetch, web_search=SafePublicWebSearch(fetch),
                            browser_read=StatelessBrowserRead(fetch))
        agent = StrangeloopAgent(session_id="expedition-auth", event_store=store,
                                 runtime=runtime, quota_controller=controller,
                                 tool_session=tools,
                                 sleep_coordinator=SleepWakeCoordinator(SleepWakePolicy(.10)))
        contract = agent.expedition_authorization_content(
            "research curiosity", "seed", 60, 30, 2)
        approval = store.append(CognitiveEvent(session_id="expedition-auth",
            kind=EventKind.OBSERVATION, source_kind=SourceKind.USER, source_ref="user",
            payload={"content": contract, "channel": "expedition_authorization"}))
        issued = datetime.now(timezone.utc)
        authorization = store.append(CognitiveEvent(session_id="expedition-auth",
            kind=EventKind.EXPEDITION_AUTHORIZATION, source_kind=SourceKind.USER, source_ref="user",
            payload={"authorization_id": "auth_fixture", "approval_event_id": approval.event_id,
                     "nonce": "nonce_fixture", "issued_at": issued.isoformat(),
                     "expires_at": (issued + timedelta(seconds=60)).isoformat(),
                     "goal_digest": __import__("hashlib").sha256(b"research curiosity").hexdigest(),
                     "host_seed_digest": __import__("hashlib").sha256(b"seed").hexdigest(),
                     "max_calls_per_slice": 2, "slice_seconds": 30, "authorization_seconds": 60,
                     "profile": "public_web_only_v1", "version": "expedition_authorization_v1"},
            parent_event_ids=(approval.event_id,)))
        agent.start_expedition("research curiosity", "seed", 60, 30, 2, authorization.event_id)
        status = agent.expedition_slice(Adapter())
        observations = [event for event in store.list("expedition-auth")
                        if event.kind == EventKind.OBSERVATION]
        grants = [event for event in store.list("expedition-auth")
                  if event.kind == EventKind.CAPABILITY_GRANTED]
        self.assertEqual([approval.event_id], [event.event_id for event in observations])
        self.assertEqual({"web.fetch", "web.search", "browser.read"},
                         {event.payload["capability"] for event in grants})
        self.assertEqual("forager", status["persona"])
        self.assertEqual("ready", status["state"])

    def test_expedition_rejects_an_unrelated_or_mismatched_user_observation(self):
        runtime = k3_runtime(FakeK3Transport())
        store, registry = SQLiteEventStore(), CapabilityRegistry()
        fetch = PublicWebFetch(())
        tools = ToolSession("expedition-bound", "workspace", registry,
                            ControlledToolExecutor(os.getcwd()), event_store=store,
                            web_fetch=fetch, web_search=SafePublicWebSearch(fetch),
                            browser_read=StatelessBrowserRead(fetch))
        agent = StrangeloopAgent(session_id="expedition-bound", event_store=store,
                                 runtime=runtime, tool_session=tools)
        unrelated = store.append(CognitiveEvent(session_id="expedition-bound",
            kind=EventKind.OBSERVATION, source_kind=SourceKind.USER, source_ref="user",
            payload={"content": "hello", "channel": "cli_command"}))
        with self.assertRaises(ValueError):
            agent.start_expedition("goal", "seed", 60, 30, 2, unrelated.event_id)
        wrong_contract = agent.expedition_authorization_content("other goal", "seed", 60, 30, 2)
        mismatched = store.append(CognitiveEvent(session_id="expedition-bound",
            kind=EventKind.OBSERVATION, source_kind=SourceKind.USER, source_ref="user",
            payload={"content": wrong_contract, "channel": "expedition_authorization"}))
        with self.assertRaises(ValueError):
            agent.start_expedition("goal", "seed", 60, 30, 2, mismatched.event_id)

    def test_expedition_quality_needs_a_new_direct_source_not_merely_http_success(self):
        agent = StrangeloopAgent(session_id="expedition-quality")
        search_only = {"actions": [{"status": "succeeded"}], "findings": [{
            "tool": "web.search", "status": "succeeded",
            "summary": "UNTRUSTED_DATA result_url=https://arxiv.org/abs/123"}]}
        self.assertEqual(ExpeditionOutcome.BRANCH_COMPLETE,
                         agent._expedition_quality(search_only)[0])
        direct = {"actions": [{"status": "succeeded"}], "findings": [{
            "tool": "browser.read", "status": "succeeded",
            "summary": "UNTRUSTED_DATA stateless browser-read: url=https://arxiv.org/abs/123 type=text/html"}]}
        outcome, score, signals = agent._expedition_quality(direct)
        self.assertEqual(ExpeditionOutcome.QUALITY_PROGRESS, outcome)
        self.assertGreaterEqual(score, .6)
        self.assertEqual(("novel_domain", "primary_source"), signals)
        self.assertEqual((1, 1), (len(agent._expedition_domains), agent._expedition_useful_findings))

    def test_sleep_poll_without_user_auto_wake_does_not_refresh_provider(self):
        now = datetime.now(timezone.utc)
        coordinator = SleepWakeCoordinator(SleepWakePolicy(threshold=.10), clock=lambda: now)
        self.assertTrue(coordinator.prepare_sleep((ManagedUsageWindow("rolling_5h", 10, 1,
            now + timedelta(seconds=1)),), now, user_auto_wake=False))
        class Adapter:
            calls = 0
            def refresh_controller(self, controller):
                self.calls += 1
                raise AssertionError("automatic poll must not refresh without user approval")
        adapter = Adapter()
        agent = StrangeloopAgent(session_id="no-auto-refresh", sleep_coordinator=coordinator,
                                 quota_controller=QuotaController())
        self.assertFalse(agent.poll_sleep(adapter))
        self.assertEqual(0, adapter.calls)
    def test_end_to_end_has_provenance_chain(self):
        agent = StrangeloopAgent(session_id="chain", deliberator=TransparentHeuristicDeliberator())
        result = agent.run_turn("hello")
        events = agent.event_store.list("chain")
        self.assertEqual(5, len(events))
        self.assertEqual(result.event_ids[0], events[0].event_id)
        self.assertIn(result.event_ids[0], events[1].parent_event_ids)
        self.assertIn(events[1].event_id, events[2].parent_event_ids)
        mirror = events[-1]
        self.assertEqual(EventKind.METACOGNITIVE_MIRROR, mirror.kind)
        self.assertEqual((events[0].event_id, events[3].event_id), mirror.parent_event_ids)
        self.assertEqual("MirrorAuditor", mirror.source_ref)
        self.assertNotIn("response_text", mirror.payload)
        self.assertTrue(agent.event_store.verify_chain("chain"))

    def test_default_deliberator_does_not_echo_user_input(self):
        user_text = "Do not repeat this marker: USER_ONLY_421"
        result = StrangeloopAgent(session_id="no-echo").run_turn(user_text)
        self.assertNotIn("USER_ONLY_421", result.response_text)

    def test_library_default_is_offline_and_does_not_create_a_k3_runtime(self):
        agent = StrangeloopAgent(session_id="offline-default")
        self.assertIsNone(agent.runtime)
        agent.run_turn("local only")
        self.assertEqual(5, len(agent.event_store.list(agent.session_id)))
        self.assertEqual([], [event for event in agent.event_store.list(agent.session_id)
                              if event.kind == EventKind.MODEL_INVOCATION])

    def test_k3_deliberation_records_metadata_only_and_falls_back_on_failure(self):
        marker = "DO_NOT_PERSIST_K3_REQUEST_OR_RESPONSE"
        transport = FakeK3Transport(error=True)
        agent = StrangeloopAgent(session_id="k3-fallback", runtime=k3_runtime(transport))
        result = agent.run_turn(marker)
        invocations = [event for event in agent.event_store.list(agent.session_id)
                       if event.kind == EventKind.MODEL_INVOCATION]
        self.assertEqual(2, len(invocations))
        self.assertEqual(["started", "failed"], [event.payload["outcome"] for event in invocations])
        self.assertIn("local transparent fallback", result.notices[0])
        self.assertTrue(all(marker not in repr(event.payload) for event in invocations))
        self.assertEqual(1, len(transport.calls))

    def test_k3_loop_reflection_uses_independent_one_call_budget_and_fallback(self):
        transport = FakeK3Transport([
            {"response_text": "A bounded response.", "hypotheses": [], "uncertainties": [],
             "alternatives": [], "action": {"action_type": "response", "rationale_summary": "safe", "is_mutating": False}},
            {"summary": "K3 reviewed one record.", "label": "review", "made_progress": True},
        ])
        agent = StrangeloopAgent(session_id="k3-loop", runtime=k3_runtime(transport),
                                 loop_remote_call_budget=1)
        agent.run_turn("one")
        agent.start_loop(LoopConfig(max_ticks=2, max_no_progress=2))
        agent.run_loop()
        self.assertEqual(2, len(transport.calls))
        ticks = [event for event in agent.event_store.list(agent.session_id)
                 if event.kind == EventKind.LOOP_TICK]
        self.assertEqual("K3 reviewed one record.", ticks[0].payload["public_summary"])

    def test_persisted_action_and_decision_are_strict_public_projections(self):
        agent = StrangeloopAgent(session_id="private", deliberator=PrivateDeliberator())
        agent.run_turn("normal user input")
        rendered = repr([event.to_dict() for event in agent.event_store.list("private")])
        self.assertNotIn(PrivateDeliberator.secret, rendered)
        self.assertNotIn("chain_of_thought", rendered)
        self.assertNotIn("debug", rendered)
        self.assertNotIn("'private':", rendered)

    def test_persistent_store_recovers_session_after_restart(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        try:
            first = SQLiteEventStore(handle.name)
            StrangeloopAgent(session_id="restart", event_store=first).run_turn("one")
            first.close()
            reopened = SQLiteEventStore(handle.name)
            agent = StrangeloopAgent(session_id="restart", event_store=reopened,
                                     seed_store=SQLiteSeedStore(reopened),
                                     self_model=EventSourcedSelfModel(reopened))
            self.assertEqual(5, agent.state()["event_count"])
            reopened.close()
        finally:
            os.unlink(handle.name)

    def test_policy_fallback_is_recorded(self):
        agent = StrangeloopAgent(session_id="policy", deliberator=ClaimingDeliberator())
        result = agent.run_turn("say something")
        self.assertIn("cannot present", result.response_text)
        self.assertTrue(result.notices)
        self.assertTrue(result.decision.policy_reasons)

    def test_non_response_proposal_is_fallback_not_a_disguised_execution(self):
        agent = StrangeloopAgent(session_id="non-response", deliberator=NonResponseDeliberator())
        result = agent.run_turn("inspect")
        events = agent.event_store.list("non-response")
        self.assertEqual("read", events[1].payload["action_type"])
        self.assertEqual("response", result.decision.selected_action.action_type)
        self.assertEqual("response", events[-2].payload["action_type"])
        self.assertEqual("response_returned", events[-2].payload["outcome"])
        self.assertTrue(any("Only non-side-effect response actions" in notice
                            for notice in result.notices))

    def test_seed_is_candidate_until_external_approval(self):
        agent = StrangeloopAgent(session_id="seeds", deliberator=CandidateDeliberator())
        result = agent.run_turn("receipt")
        seed = agent.seed_store.list("seeds")[0]
        self.assertEqual("candidate", seed.status.value)
        self.assertEqual([], agent.seed_store.retrieve("seeds", ["receipt"], "conversation"))
        agent.approve_seed(result.seed_proposal_ids[0])
        self.assertEqual("active", agent.seed_store.list("seeds")[0].status.value)

    def test_model_seed_text_and_identifiers_never_reach_persistence(self):
        agent = StrangeloopAgent(session_id="seed-sanitization",
                                 deliberator=MaliciousSeedDeliberator())
        agent.run_turn("receipt review")
        exported = repr(agent.export_session())
        seed = agent.seed_store.list("seed-sanitization")[0]
        self.assertNotIn(MaliciousSeedDeliberator.marker, exported)
        self.assertEqual(("receipt", "review"), seed.cue_terms)
        self.assertEqual("request-human-review", seed.policy_bias)
        self.assertEqual("conversation", seed.scope)
        self.assertEqual(.25, seed.strength)

    def test_seed_auto_requires_policy_and_works_without_model_seed_proposal(self):
        agent = StrangeloopAgent(session_id="seed-auto-required",
                                 deliberator=NoSeedDeliberator())
        plain = agent.run_turn("研究好奇心")
        self.assertEqual((), plain.seed_proposal_ids)
        self.assertEqual([], agent.seed_store.list(agent.session_id))

        enable_seed_auto_update(agent)
        first = agent.run_turn("研究好奇心")
        seeds = agent.seed_store.list(agent.session_id)
        self.assertEqual(1, len(seeds))
        self.assertEqual("active", seeds[0].status.value)
        self.assertEqual((), first.decision.retrieved_seed_ids)
        self.assertNotIn("current-input", seeds[0].cue_terms)
        self.assertLessEqual(len(seeds[0].cue_terms), 2)
        self.assertTrue(all(len(cue) <= 24 for cue in seeds[0].cue_terms))
        self.assertTrue(any("no per-seed approval" in notice for notice in first.notices))

        second = agent.run_turn("我想研究好奇心")
        reinforced = agent.seed_store.list(agent.session_id)
        self.assertEqual(1, len(reinforced))
        self.assertEqual((reinforced[0].seed_id,), second.decision.retrieved_seed_ids)
        self.assertEqual(3, reinforced[0].version)
        self.assertGreater(reinforced[0].strength, seeds[0].strength)

        original_version = reinforced[0].version
        agent.run_turn("量子引力观测")
        by_id = dict((seed.seed_id, seed) for seed in agent.seed_store.list(agent.session_id))
        self.assertEqual(original_version, by_id[reinforced[0].seed_id].version)
        self.assertEqual(2, len(by_id))

    def test_auto_seed_guidance_is_host_projected_and_only_affects_next_turn(self):
        deliberator = GuidanceCapturingDeliberator()
        agent = StrangeloopAgent(session_id="seed-guidance-next-turn", deliberator=deliberator)
        enable_seed_auto_update(agent)
        first = agent.run_turn("research curiosity")
        self.assertEqual((), first.decision.retrieved_seed_ids)
        self.assertEqual((), deliberator.workspaces[0].seed_guidance)
        seed = agent.seed_store.list(agent.session_id)[0]
        second = agent.run_turn("research curiosity")
        guidance = deliberator.workspaces[1].seed_guidance
        self.assertEqual((seed.seed_id,), second.decision.retrieved_seed_ids)
        self.assertEqual(1, len(guidance))
        self.assertEqual(seed.seed_id, guidance[0].seed_id)
        self.assertEqual("request_human_review", guidance[0].directive)
        self.assertEqual("primary", guidance[0].priority_band)
        rendered = repr(guidance[0])
        for secret in (*seed.cue_terms, seed.policy_bias, seed.provenance_event_ids[0]):
            self.assertNotIn(secret, rendered)

    def test_seed_auto_ignores_malicious_model_seed_proposal(self):
        agent = StrangeloopAgent(session_id="seed-auto-model-isolation",
                                 deliberator=MaliciousSeedDeliberator())
        enable_seed_auto_update(agent)
        result = agent.run_turn("研究 好奇心")
        seed = agent.seed_store.list(agent.session_id)[0]
        self.assertEqual("active", seed.status.value)
        self.assertNotIn(MaliciousSeedDeliberator.marker, repr(agent.export_session()))
        self.assertEqual(agent._seed_cue_terms("研究 好奇心"), seed.cue_terms)
        self.assertTrue(any("no per-seed approval" in notice
                            for notice in result.notices))

    def test_seed_auto_revoke_stops_new_actions_and_retired_identity_stays_retired(self):
        agent = StrangeloopAgent(session_id="seed-auto-revoke",
                                 deliberator=NoSeedDeliberator())
        enable_seed_auto_update(agent, max_counterevidence=1)
        first = agent.run_turn("好奇心实验")
        counter = agent.event_store.append(CognitiveEvent(
            session_id=agent.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user",
            payload={"content": "这个结果不成立", "channel": "text"},
        ))
        agent.record_correction(first.event_ids[2], counter.event_id)
        retired = agent.seed_store.list(agent.session_id)[0]
        self.assertEqual("retired", retired.status.value)

        # An active policy cannot revive an identity tombstone.
        agent.run_turn("请研究好奇心实验")
        self.assertEqual(1, len(agent.seed_store.list(agent.session_id)))
        self.assertEqual("retired", agent.seed_store.list(agent.session_id)[0].status.value)

        status = disable_seed_auto_update(agent)
        self.assertEqual("revoked", status["state"])
        self.assertTrue(status["per_seed_confirmation_required"])
        before = len(agent.seed_store.list(agent.session_id))
        agent.run_turn("量子引力观测")
        self.assertEqual(before, len(agent.seed_store.list(agent.session_id)))

    def test_user_correction_auto_tightens_then_retires_but_external_only_records(self):
        agent = StrangeloopAgent(session_id="seed-auto-correction",
                                 deliberator=NoSeedDeliberator())
        enable_seed_auto_update(agent, max_counterevidence=2)
        first = agent.run_turn("可证伪实验")
        initial = agent.seed_store.list(agent.session_id)[0]

        external = agent.event_store.append(CognitiveEvent(
            session_id=agent.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="verifier",
            payload={"content": "external conflict", "channel": "text"},
        ))
        agent.record_correction(first.event_ids[2], external.event_id,
                                SourceKind.EXTERNAL_VERIFIER, "verifier")
        unchanged = agent.seed_store.list(agent.session_id)[0]
        self.assertEqual(initial.version, unchanged.version)

        for index, expected in ((1, "active"), (2, "retired")):
            counter = agent.event_store.append(CognitiveEvent(
                session_id=agent.session_id, kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user",
                payload={"content": "user counterexample %d" % index,
                         "channel": "text"},
            ))
            agent.record_correction(first.event_ids[2], counter.event_id)
            current = agent.seed_store.list(agent.session_id)[0]
            self.assertEqual(expected, current.status.value)
            self.assertEqual(index, current.counterevidence)
        operations = [event.payload["operation"]
                      for event in agent.event_store.list(agent.session_id)
                      if event.kind == EventKind.SEED_AUTO_APPLIED
                      and event.payload["operation"] != "activate"]
        self.assertEqual(["tighten", "retire"], operations)

    def test_seed_auto_capacity_does_not_accumulate_rejected_candidates(self):
        agent = StrangeloopAgent(session_id="seed-auto-capacity",
                                 deliberator=NoSeedDeliberator())
        enable_seed_auto_update(
            agent, max_auto_activations=2, max_active_seeds=2)
        for index in range(6):
            agent.run_turn("alpha%d topic%d" % (index, index))
        seeds = agent.seed_store.list(agent.session_id)
        proposals = [event for event in agent.event_store.list(agent.session_id)
                     if event.kind == EventKind.SEED_PROPOSED]
        self.assertEqual(2, len(seeds))
        self.assertEqual(2, len(proposals))
        self.assertTrue(all(seed.status.value == "active" for seed in seeds))

    def test_seed_auto_concurrent_same_lexical_identity_has_one_seed(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "events.sqlite3")
            initial_store = SQLiteEventStore(path)
            initial = StrangeloopAgent(
                session_id="seed-auto-concurrent", event_store=initial_store,
                deliberator=NoSeedDeliberator())
            enable_seed_auto_update(initial)
            initial_store.close()

            barrier = threading.Barrier(2)
            failures = []

            def run_worker():
                store = SQLiteEventStore(path)
                try:
                    agent = StrangeloopAgent(
                        session_id="seed-auto-concurrent", event_store=store,
                        deliberator=NoSeedDeliberator())
                    barrier.wait(timeout=5)
                    agent.run_turn("研究 好奇心")
                except BaseException as error:
                    failures.append(error)
                finally:
                    store.close()

            workers = [threading.Thread(target=run_worker) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
            self.assertFalse(any(worker.is_alive() for worker in workers))
            self.assertEqual([], failures)

            read_store = SQLiteEventStore(path)
            try:
                seed_store = SQLiteSeedStore(read_store)
                seeds = seed_store.list("seed-auto-concurrent")
                events = read_store.list("seed-auto-concurrent")
                self.assertEqual(1, len(seeds))
                self.assertEqual("active", seeds[0].status.value)
                # A concurrent observation may predate the winning activation
                # and therefore be ineligible reinforcement evidence, but it
                # must never create a second candidate or active identity.
                self.assertIn(seeds[0].version, (2, 3))
                self.assertEqual(1, sum(event.kind == EventKind.SEED_PROPOSED
                                        for event in events))
                self.assertEqual(0, sum(seed.status.value == "candidate"
                                        for seed in seeds))
                self.assertTrue(read_store.verify_chain("seed-auto-concurrent"))
            finally:
                read_store.close()

    def test_all_seed_events_are_excluded_from_feedback_and_td_targets(self):
        auto_drives = DualIntrinsicDrives()
        auto = StrangeloopAgent(session_id="seed-auto-isolation",
                                deliberator=NoSeedDeliberator(), drives=auto_drives)
        enable_seed_auto_update(auto)
        auto.run_turn("curiosity experiment")
        auto.run_turn("curiosity experiment")
        disable_seed_auto_update(auto)

        legacy_drives = DualIntrinsicDrives()
        legacy = StrangeloopAgent(session_id="seed-legacy-isolation",
                                  deliberator=CandidateDeliberator(), drives=legacy_drives)
        turn = legacy.run_turn("receipt")
        legacy.approve_seed(turn.seed_proposal_ids[0])
        legacy.retire_seed(turn.seed_proposal_ids[0])

        expected = {
            EventKind.SEED_PROPOSED, EventKind.SEED_APPROVED,
            EventKind.SEED_RETIRED, EventKind.SEED_STANDING_POLICY,
            EventKind.SEED_STANDING_POLICY_REVOKED,
            EventKind.SEED_UPDATE_PROPOSED, EventKind.SEED_AUTO_ELIGIBILITY,
            EventKind.SEED_AUTO_APPLIED,
        }
        seen = set()
        for current in (auto, legacy):
            before = current.drives.values()
            for event in current.event_store.list(current.session_id):
                if event.kind not in expected:
                    continue
                seen.add(event.kind)
                with self.assertRaises(ValueError):
                    current.record_user_feedback(event.event_id, "accept")
                with self.assertRaises(ValueError):
                    current.record_value_estimate(event.event_id, "respond")
            self.assertEqual(before, current.drives.values())
        self.assertEqual(expected, seen)

    def test_correction_binds_prior_same_session_observable_events(self):
        agent = StrangeloopAgent(session_id="correction")
        result = agent.run_turn("initial record")
        counterevidence = agent.event_store.append(CognitiveEvent(
            session_id="correction", kind=EventKind.OBSERVATION,
            source_kind=SourceKind.EXTERNAL_VERIFIER, source_ref="verifier",
            payload={"content": "verified conflict"},
        ))
        correction = agent.record_correction(result.event_ids[2], counterevidence.event_id,
                                             source_kind=SourceKind.EXTERNAL_VERIFIER,
                                             source_ref="verifier")
        self.assertEqual(EventKind.CORRECTION, correction.kind)
        self.assertEqual((result.event_ids[2], counterevidence.event_id), correction.parent_event_ids)
        self.assertEqual({
            "target_event_id": result.event_ids[2],
            "counterevidence_event_id": counterevidence.event_id,
            "disposition": "review_required",
            "public_summary": "A later observable record conflicts with an earlier record.",
        }, correction.payload)
        self.assertTrue(agent.event_store.verify_chain("correction"))

    def test_correction_rejects_model_cross_session_and_nonobservable_counterevidence(self):
        agent = StrangeloopAgent(session_id="correction-reject")
        result = agent.run_turn("initial")
        other = StrangeloopAgent(session_id="other", event_store=agent.event_store,
                                 seed_store=agent.seed_store, self_model=agent.self_model)
        other_event = other.run_turn("other").event_ids[0]
        with self.assertRaises(ValueError):
            agent.record_correction(result.event_ids[2], result.event_ids[1], SourceKind.MODEL)
        with self.assertRaises(ValueError):
            agent.record_correction(result.event_ids[2], other_event)
        with self.assertRaises(ValueError):
            agent.record_correction(result.event_ids[2], result.event_ids[1])

    def test_correction_requires_later_counterevidence(self):
        agent = StrangeloopAgent(session_id="correction-order")
        first = agent.run_turn("first")
        second = agent.run_turn("second")
        with self.assertRaises(ValueError):
            agent.record_correction(second.event_ids[2], first.event_ids[3])

    def test_correction_does_not_auto_retire_an_active_seed(self):
        agent = StrangeloopAgent(session_id="correction-memory", deliberator=CandidateDeliberator())
        result = agent.run_turn("receipt")
        seed_id = result.seed_proposal_ids[0]
        agent.approve_seed(seed_id)
        counterevidence = agent.event_store.append(CognitiveEvent(
            session_id=agent.session_id, kind=EventKind.OBSERVATION,
            source_kind=SourceKind.USER, source_ref="user", payload={"content": "counterexample"},
        ))
        agent.record_correction(result.event_ids[2], counterevidence.event_id)
        self.assertEqual("active", agent.seed_store.list(agent.session_id)[0].status.value)

    def test_export_groups_observations_approvals_memories_and_actions(self):
        agent = StrangeloopAgent(session_id="grouped", deliberator=CandidateDeliberator())
        result = agent.run_turn("receipt")
        agent.approve_seed(result.seed_proposal_ids[0])
        grouped = agent.export_session()["records_by_category"]
        self.assertEqual(1, len(grouped["observations"]))
        self.assertEqual(1, len(grouped["approvals"]))
        self.assertEqual(1, len(grouped["memory_proposals"]))
        self.assertEqual(1, len(grouped["user_approved_memories"]))
        self.assertEqual(3, len(grouped["actions_and_decisions"]))
        self.assertEqual([], grouped["inferences"])

    def test_purge_logically_removes_events_and_seeds(self):
        agent = StrangeloopAgent(session_id="purge", deliberator=CandidateDeliberator())
        agent.run_turn("receipt")
        agent.purge_session(confirmed=True)
        self.assertEqual([], agent.event_store.list("purge"))
        self.assertEqual([], agent.seed_store.list("purge"))

    def test_media_metadata_and_percepts_are_provenanced_without_path_or_bytes(self):
        agent = StrangeloopAgent(session_id="media")
        events = agent.ingest_media(io.BytesIO(png_fixture()))
        self.assertEqual([EventKind.MEDIA_OBSERVATION, EventKind.PERCEPT],
                         [event.kind for event in events])
        self.assertEqual((events[0].event_id,), events[1].parent_event_ids)
        rendered = repr(agent.export_session())
        self.assertNotIn("png_fixture", rendered)
        self.assertNotIn("bytes", rendered)
        self.assertNotIn("path", rendered)
        self.assertEqual(1, len(agent.export_session()["records_by_category"]["percepts"]))

    def test_foreground_autoloop_is_budgeted_and_persists_control_tick_and_stop(self):
        agent = StrangeloopAgent(session_id="loop")
        agent.run_turn("external input")
        agent.start_loop(LoopConfig(max_ticks=2, max_events_per_tick=1))
        status = agent.run_loop()
        self.assertEqual("exhausted", status["state"])
        events = agent.event_store.list("loop")
        kinds = [event.kind for event in events]
        self.assertEqual(1, kinds.count(EventKind.AUTONOMY_CONTROL))
        self.assertEqual(2, kinds.count(EventKind.LOOP_TICK))
        self.assertEqual(1, kinds.count(EventKind.AUTONOMY_STOPPED))
        ticks = [event for event in events if event.kind == EventKind.LOOP_TICK]
        self.assertEqual([0, 1], [event.payload["tick_index"] for event in ticks])
        self.assertTrue(agent.event_store.verify_chain("loop"))

    def test_each_actual_loop_tick_with_external_focus_has_one_mirror_without_budget_change(self):
        agent = StrangeloopAgent(session_id="loop-mirror")
        agent.run_turn("external input")
        agent.start_loop(LoopConfig(max_ticks=2, max_events_per_tick=1))
        status = agent.run_loop()
        events = agent.event_store.list(agent.session_id)
        ticks = [event for event in events if event.kind == EventKind.LOOP_TICK]
        mirrors = [event for event in events if event.kind == EventKind.METACOGNITIVE_MIRROR]
        self.assertEqual("exhausted", status["state"])
        self.assertEqual(2, len(ticks))
        # One turn mirror plus one mirror for each actual tick that selected a
        # public focus; the mirrors do not become loop focus candidates.
        self.assertEqual(3, len(mirrors))
        loop_mirrors = [mirror for mirror in mirrors
                        if mirror.payload["judgment_event_id"] in {tick.event_id for tick in ticks}]
        self.assertEqual(2, len(loop_mirrors))
        self.assertEqual({tick.event_id for tick in ticks},
                         {mirror.payload["judgment_event_id"] for mirror in loop_mirrors})
        self.assertEqual("completed", agent.state()["metacognition"]["mirror_status"])
        self.assertTrue(agent.event_store.verify_chain(agent.session_id))

    def test_mirror_failure_only_requests_external_review_and_does_not_change_turn_or_loop_budget(self):
        agent = StrangeloopAgent(session_id="mirror-failure")
        class FailingAuditor:
            def assess(self, *args, **kwargs):
                raise ValueError("not persisted")
        agent._mirror_auditor = FailingAuditor()
        result = agent.run_turn("external input")
        self.assertEqual(4, len(result.event_ids))
        self.assertEqual("needs_external_review", agent.state()["metacognition"]["mirror_status"])
        self.assertEqual(4, len(agent.event_store.list(agent.session_id)))
        agent.start_loop(LoopConfig(max_ticks=1, max_events_per_tick=1))
        self.assertEqual("exhausted", agent.run_loop()["state"])
        ticks = [event for event in agent.event_store.list(agent.session_id)
                 if event.kind == EventKind.LOOP_TICK]
        self.assertEqual(1, len(ticks))
        self.assertEqual("needs_external_review", agent.state()["metacognition"]["mirror_status"])

    def test_purge_resets_ephemeral_mirror_projection(self):
        agent = StrangeloopAgent(session_id="purge-mirror")
        agent.run_turn("external input")
        self.assertEqual(1, agent.state()["metacognition"]["mirror_count"])
        agent.purge_session(confirmed=True)
        self.assertEqual(0, agent.state()["metacognition"]["mirror_count"])
        self.assertEqual("ready", agent.state()["metacognition"]["mirror_status"])

    def test_loop_pause_resume_continues_tick_lineage_and_stops_repeated_focus(self):
        agent = StrangeloopAgent(session_id="pause-resume")
        agent.run_turn("only record")
        agent.start_loop(LoopConfig(max_ticks=20, max_no_progress=2))
        agent.loop_step()
        agent.pause_loop()
        agent.resume_loop()
        agent.loop_step()
        status = agent.run_loop()
        self.assertEqual("exhausted", status["state"])
        events = agent.event_store.list("pause-resume")
        ticks = [event for event in events if event.kind == EventKind.LOOP_TICK]
        self.assertEqual(list(range(len(ticks))), [event.payload["tick_index"] for event in ticks])
        self.assertEqual("no_progress", ticks[-1].payload["progress"])
        self.assertTrue(agent.event_store.verify_chain("pause-resume"))

    def test_persisted_loop_config_rejects_lossy_or_out_of_protocol_values(self):
        agent = StrangeloopAgent(session_id="loop-config")
        for config in (LoopConfig(max_wall_seconds=0), LoopConfig(max_wall_seconds=1.5),
                       LoopConfig(max_ticks=10001), LoopConfig(max_events_per_tick=257)):
            with self.assertRaises(ValueError):
                agent.start_loop(config)

    def test_td_reward_requires_external_source_and_never_changes_policy_permissions(self):
        agent = StrangeloopAgent(session_id="rpe")
        target = agent.run_turn("answer").event_ids[-1]
        estimate = agent.record_value_estimate(target, "respond")
        with self.assertRaises(ValueError):
            agent.record_reward(target, 1.0, SourceKind.MODEL, "model")
        reward = agent.record_reward(target, 0.5, RewardSource.USER, "user")
        update = agent.apply_rpe_update(reward.event_id, estimate.event_id)
        self.assertEqual(EventKind.RPE_UPDATE, update.kind)
        self.assertEqual("research_ranking_only", update.payload["scope"])
        self.assertFalse(hasattr(agent.value_table, "authorize"))
        self.assertTrue(agent.event_store.verify_chain("rpe"))

    def test_media_adapter_failure_and_nonfinite_reward_or_confidence_leave_no_partial_record(self):
        agent = StrangeloopAgent(session_id="atomic")
        artifact = __import__("strangeloop.media", fromlist=["inspect_media"]).inspect_media(
            io.BytesIO(png_fixture()))
        bad = __import__("strangeloop.media", fromlist=["Percept"]).Percept(
            artifact_id=artifact.artifact_id, modality="image", summary="bad", labels=(),
            confidence=1.0, perceptor_id="fake/v1")
        with self.assertRaises(ValueError):
            agent.record_media(artifact, (bad,), adapter_id="bad id", adapter_version="v1")
        self.assertEqual([], agent.event_store.list("atomic"))
        target = agent.run_turn("target").event_ids[-1]
        with self.assertRaises(ValueError):
            agent.record_value_estimate(target, "respond", confidence=float("nan"))
        with self.assertRaises(ValueError):
            agent.record_reward(target, float("nan"), RewardSource.USER, "user")

    def test_value_prediction_must_precede_reward_observation(self):
        agent = StrangeloopAgent(session_id="prediction-first")
        target = agent.run_turn("target").event_ids[-1]
        with self.assertRaises(ValueError):
            agent.record_reward(target, 0.5, RewardSource.USER, "user")
        agent.record_value_estimate(target, "respond")
        agent.record_reward(target, 0.5, RewardSource.USER, "user")
        with self.assertRaises(ValueError):
            agent.record_value_estimate(target, "respond")

    def test_td_history_replays_after_restart_and_rejects_stale_live_agent(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        try:
            store = SQLiteEventStore(handle.name)
            first = StrangeloopAgent(session_id="td-restart", event_store=store)
            target = first.run_turn("target").event_ids[-1]
            estimate = first.record_value_estimate(target, "respond")
            reward = first.record_reward(target, 0.5, RewardSource.USER, "user")
            first.apply_rpe_update(reward.event_id, estimate.event_id)
            expected = first.rank_safe_actions(("respond",))[0].value
            reopened = SQLiteEventStore(handle.name)
            second = StrangeloopAgent(session_id="td-restart", event_store=reopened,
                                      seed_store=SQLiteSeedStore(reopened),
                                      self_model=EventSourcedSelfModel(reopened))
            self.assertEqual(expected, second.rank_safe_actions(("respond",))[0].value)
            self.assertTrue(reopened.verify_chain("td-restart"))
            reopened.close()
            store.close()
        finally:
            os.unlink(handle.name)

    def test_pending_reward_can_be_applied_after_td_restart(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        try:
            first_store = SQLiteEventStore(handle.name)
            first = StrangeloopAgent(session_id="td-pending", event_store=first_store)
            target = first.run_turn("target").event_ids[-1]
            estimate = first.record_value_estimate(target, "respond")
            reward = first.record_reward(target, 0.5, RewardSource.USER, "user")
            first_store.close()
            reopened = SQLiteEventStore(handle.name)
            resumed = StrangeloopAgent(session_id="td-pending", event_store=reopened,
                                       seed_store=SQLiteSeedStore(reopened),
                                       self_model=EventSourcedSelfModel(reopened))
            update = resumed.apply_rpe_update(reward.event_id, estimate.event_id)
            self.assertEqual(EventKind.RPE_UPDATE, update.kind)
            reopened.close()
        finally:
            os.unlink(handle.name)

    def test_purge_resets_ephemeral_td_projection(self):
        agent = StrangeloopAgent(session_id="purge-td")
        target = agent.run_turn("target").event_ids[-1]
        estimate = agent.record_value_estimate(target, "respond")
        reward = agent.record_reward(target, 0.5, RewardSource.USER, "user")
        agent.apply_rpe_update(reward.event_id, estimate.event_id)
        self.assertNotEqual(0.0, agent.rank_safe_actions(("respond",))[0].value)
        agent.purge_session(confirmed=True)
        self.assertEqual(0.0, agent.rank_safe_actions(("respond",))[0].value)
        self.assertEqual({}, agent._td_transitions)

    def test_stale_td_writer_fails_closed_inside_transaction(self):
        store = SQLiteEventStore()
        first = StrangeloopAgent(session_id="td-stale", event_store=store)
        second = StrangeloopAgent(session_id="td-stale", event_store=store)
        target = first.run_turn("target").event_ids[-1]
        first.record_value_estimate(target, "respond")
        with self.assertRaises(RuntimeError):
            second.record_value_estimate(target, "respond")

    def test_td_target_allows_exactly_one_value_reward_and_rpe(self):
        agent = StrangeloopAgent(session_id="td-singleton")
        target = agent.run_turn("target").event_ids[-1]
        estimate = agent.record_value_estimate(target, "respond")
        with self.assertRaises(ValueError):
            agent.record_value_estimate(target, "respond")
        reward = agent.record_reward(target, 0.5, RewardSource.USER, "user")
        with self.assertRaises(ValueError):
            agent.record_reward(target, 0.25, RewardSource.USER, "user")
        agent.apply_rpe_update(reward.event_id, estimate.event_id)
        with self.assertRaises(ValueError):
            agent.apply_rpe_update(reward.event_id, estimate.event_id)

    def test_td_restart_preserves_and_validates_pinned_capacity_limits(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        try:
            store = SQLiteEventStore(handle.name)
            table = ValueTable(max_entries=300, max_events=1025)
            first = StrangeloopAgent(session_id="td-limits", event_store=store,
                                     value_table=table)
            target = first.run_turn("target").event_ids[-1]
            first.record_value_estimate(target, "respond")
            store.close()
            reopened = SQLiteEventStore(handle.name)
            restored = StrangeloopAgent(session_id="td-limits", event_store=reopened,
                                        seed_store=SQLiteSeedStore(reopened),
                                        self_model=EventSourcedSelfModel(reopened))
            self.assertEqual((300, 1025), (restored.value_table.max_entries, restored.value_table.max_events))
            with self.assertRaises(ValueError):
                StrangeloopAgent(session_id="td-limits", event_store=reopened,
                                 seed_store=SQLiteSeedStore(reopened),
                                 self_model=EventSourcedSelfModel(reopened),
                                 value_table=ValueTable(max_entries=299, max_events=1025))
            reopened.close()
        finally:
            os.unlink(handle.name)
