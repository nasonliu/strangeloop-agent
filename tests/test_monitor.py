import json
import shutil
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

from strangeloop.monitor import CognitiveMonitor, _state_projection, project_event
from strangeloop.monitor_ui import dashboard_html
from strangeloop.engine import StrangeloopAgent
from strangeloop.quota import QuotaController, QuotaSnapshot, QuotaSource


def event(sequence, kind, payload):
    return SimpleNamespace(sequence=sequence, event_id="evt_%s" % sequence,
                           kind=kind, source_kind="system", created_at="2026-08-12T00:00:00+00:00",
                           confidence=1.0, parent_event_ids=(), payload=payload)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            event(1, "model_invocation", {"provider": "K3", "model": "kimi", "outcome": "completed", "public_summary": "Model call complete"}),
            event(2, "tool_call_proposed", {"tool_name": "search", "operation": "read", "public_summary": "Plan a read", "input": "must not leak"}),
            event(3, "rpe_update", {"reward": 0.5, "raw_delta": 0.3, "secret": "nope", "chain_of_thought": "never"}),
            event(4, "metacognitive_mirror", {
                "mirror_id": "mirror_1", "episode_id": "episode_1", "target_event_id": "evt_1",
                "judgment_event_id": "evt_2", "evidence_event_ids": ["evt_1", "evt_2"],
                "self_status": "supported", "self_confidence": 0.7, "self_uncertainty": "low",
                "meta_status": "confirmed", "meta_confidence_cap": 0.8,
                "check_codes": ["provenance_complete", "confidence_capped"],
                "disposition": "provisional", "method_version": "mirror_v1",
                "public_summary": "Bounded public mirror status.",
                "hidden_reasoning": "must never leak", "nested": {"secret": "must never leak"},
            }),
            event(5, "sleep_archive", {
                "archive_id": "archive_1", "epoch_id": "epoch_1", "chain_head_hash": "a" * 64,
                "approved_seed_ids": ["seed_1"], "approved_claim_ids": ["claim_1"],
                "pending_task_ids": ["task_1"], "quota_source": "provider_usage",
                "quota_observed_at": "2026-08-12T00:00:00+00:00",
                "quota_reset_at": "2026-08-12T05:00:00+00:00", "archive_digest": "b" * 64,
                "event_count": 4, "archive_version": "sleep_archive_v1",
                "path": "/private/archive", "prompt": "must not leak", "secret": "nope",
            }),
            event(6, "tool_result", {"tool_name": "browser.read", "status": "succeeded",
                                      "public_summary": "Untrusted page https://example.com/?private=never"}),
        ]
        self.monitor = CognitiveMonitor(
            snapshot_source=lambda: {"current_goal": "<img src=x onerror=1>", "current_turn": "turn_1", "loop": {"state": "running", "max_ticks": 3, "token": "no"}, "sleep_wake": {"state": "sleeping", "epoch": 7, "generation": 9, "auto_wake_user_approved": False, "reset_at": "2026-08-12T05:00:00+00:00", "retry_at": "2026-08-12T05:01:00+00:00", "failure_count": 1, "last_authoritative_observed_at": "2026-08-12T00:00:00+00:00", "raw_usage_response": {"token": "no"}, "prompt": "no"}, "unattended": {"state": "running", "profile_id": "research_1", "goal_digest": "d" * 64, "calls_started": 2, "max_tool_calls": 8, "reserved_bytes": 256, "max_total_bytes": 4096, "ticks": 3, "max_ticks": 9, "focus": "read https://example.com/?must_not_leak", "last_tool": "web.fetch", "last_status": "succeeded", "stop_reason": "none", "public_report": {"digest": "e" * 64, "finding_count": 2, "findings": ["never leak https://example.com/?secret=1"]}, "path": "/private/nope", "cookie": "nope", "prompt": "nope"}},
            event_source=lambda: list(self.records))
        self.url = self.monitor.start_background()

    def tearDown(self):
        self.monitor.stop()

    def get_json(self, path):
        with urlopen(self.url + path, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_http_endpoints_and_incremental_events(self):
        status, health = self.get_json("/api/health")
        self.assertEqual(200, status); self.assertEqual("ok", health["status"])
        status, state = self.get_json("/api/state")
        self.assertEqual(200, status); self.assertEqual("&lt;img src=x onerror=1&gt;", state["current_goal"])
        self.assertEqual("running", state["dmn_loop"]["state"])
        self.assertEqual("mirror_1", state["metacognitive_mirrors"][0]["payload"]["mirror_id"])
        status, events = self.get_json("/api/events?after_sequence=1")
        self.assertEqual([2, 3, 4, 5, 6], [item["sequence"] for item in events["events"]])
        with urlopen(self.url + "/", timeout=2) as response:
            page = response.read().decode("utf-8")
        self.assertIn("Hidden chain-of-thought is neither stored nor shown.", page)
        self.assertIn("textContent", page)
        self.assertIn("自证／证自证镜映", page)
        self.assertIn("Quota sleep / wake", page)
        self.assertIn("Unattended research", page)
        self.assertIn("Expedition", page)
        self.assertIn("textContent=JSON.stringify(state.expedition||{},null,2)", page)
        self.assertNotIn("innerHTML", page)

    def test_dashboard_script_is_syntactically_valid(self):
        with urlopen(self.url + "/", timeout=2) as response:
            page = response.read().decode("utf-8")
        script = page.split("<script>", 1)[1].split("</script>", 1)[0]
        self.assertIn("\\nreset:", script)
        self.assertNotIn("balance+'\nreset:", script)
        node = shutil.which("node")
        if node is not None:
            checked = subprocess.run(
                [node, "--check", "-"], input=script, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(0, checked.returncode, checked.stderr)

    def test_sleep_wake_projection_is_metadata_only_and_never_claims_reset_wakes(self):
        _, state = self.get_json("/api/state")
        sleep = state["sleep_wake"]
        self.assertEqual("sleeping", sleep["coordinator"]["state"])
        self.assertEqual((7, 9), (sleep["coordinator"]["epoch"], sleep["coordinator"]["generation"]))
        self.assertFalse(sleep["coordinator"]["auto_wake_user_approved"])
        self.assertEqual("retry_scheduled", sleep["coordinator"]["last_refresh_outcome"])
        self.assertEqual("2026-08-12T05:01:00+00:00", sleep["coordinator"]["next_check_at"])
        archive = sleep["archives"][0]
        self.assertEqual("b" * 64, archive["archive_digest"])
        self.assertEqual(4, archive["event_count"])
        self.assertEqual("a" * 64, archive["chain_head_hash"])
        rendered = json.dumps(sleep)
        for forbidden in ("approved_seed_ids", "approved_claim_ids", "pending_task_ids", "path",
                          "prompt", "secret", "raw_usage_response", "token"):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("not biological sleep or consciousness", sleep["notice"])

    def test_unattended_card_is_strictly_accounting_only(self):
        _, state = self.get_json("/api/state")
        unattended = state["unattended"]
        self.assertEqual("running", unattended["state"])
        self.assertEqual("research_1", unattended["profile_id"])
        self.assertEqual("d" * 64, unattended["goal_digest"])
        self.assertEqual(6, unattended["calls_remaining"])
        self.assertEqual(3840, unattended["bytes_remaining"])
        self.assertEqual(3, unattended["ticks"])
        self.assertEqual("web.fetch", unattended["last_tool"])
        self.assertEqual("succeeded", unattended["last_status"])
        self.assertEqual("none", unattended["stop_reason"])
        self.assertEqual(("e" * 64, 2),
                         (unattended["public_report_digest"], unattended["finding_count"]))
        encoded = json.dumps(unattended)
        for forbidden in ("focus", "example.com", "private", "cookie", "prompt", "path", "findings"):
            self.assertNotIn(forbidden, encoded)

    def test_unattended_projection_accepts_engine_status_with_nested_tool_profile(self):
        state = _state_projection({"unattended_research": {
            "configured": True, "active": True, "state": "running", "stop_reason": "none",
            "calls": 2, "ticks": 3, "input_bytes": 40, "output_bytes": 100,
            "last_action_status": "succeeded", "last_tool": "web.search",
            "budget": {"max_calls": 7, "max_output_bytes": 2048},
            "tool_profile": {"profile_id": "research_engine", "max_tool_calls": 7,
                             "max_total_bytes": 2048, "focus": "do not expose me"},
        }}, [project_event(item) for item in self.records])
        unattended = state["unattended"]
        self.assertEqual("research_engine", unattended["profile_id"])
        self.assertEqual((2, 5), (unattended["calls"], unattended["calls_remaining"]))
        self.assertEqual(1908, unattended["bytes_remaining"])
        self.assertEqual(("web.search", "succeeded"), (unattended["last_tool"], unattended["last_status"]))
        self.assertNotIn("focus", json.dumps(unattended))

    def test_agent_monitor_uses_bounded_unattended_accessor_not_agent_state(self):
        agent = SimpleNamespace(
            loop_status=lambda: {"state": "running", "max_ticks": 2},
            unattended_status=lambda: {"state": "running", "profile_id": "research_agent",
                                        "goal_digest": "a" * 64, "calls": 1,
                                        "report_digest": "b" * 64, "finding_count": 2,
                                        "budget": {"max_calls": 2, "max_output_bytes": 1024}},
            sleep_status=lambda: {"state": "active", "epoch": 1, "generation": 1},
        )
        monitor = CognitiveMonitor(agent=agent, event_source=lambda: [])
        try:
            monitor.start_background()
            with urlopen(monitor.url + "/api/state", timeout=2) as response:
                state = json.loads(response.read().decode("utf-8"))
            self.assertEqual("research_agent", state["unattended"]["profile_id"])
            self.assertEqual(("b" * 64, 2),
                             (state["unattended"]["public_report_digest"],
                              state["unattended"]["finding_count"]))
            self.assertEqual("active", state["sleep_wake"]["coordinator"]["state"])
        finally:
            monitor.stop()

    def test_agent_monitor_projects_public_provider_quota_not_raw_usage(self):
        now = datetime.now(timezone.utc)
        controller = QuotaController()
        controller.ingest_snapshot(QuotaSnapshot(100, 96, now + timedelta(hours=5), now,
                                                 1.0, False, primary_unit="provider_units",
                                                 primary_window_kind="rolling_5h"),
                                   QuotaSource.PROVIDER_USAGE, "system")
        agent = StrangeloopAgent(session_id="monitor-quota", quota_controller=controller)
        monitor = CognitiveMonitor(agent=agent, event_source=lambda: [])
        try:
            monitor.start_background()
            with urlopen(monitor.url + "/api/state", timeout=2) as response:
                state = json.loads(response.read().decode("utf-8"))
            quota = state["quota"]
            self.assertEqual((96, 100), (quota["remaining"], quota["total"]))
            self.assertEqual("provider_units", quota["primary_unit"])
            self.assertEqual("rolling_5h", quota["primary_window_kind"])
            self.assertEqual("provider_usage", quota["authority"])
            self.assertEqual("fresh", quota["freshness"])
            self.assertTrue(quota["allow_call"])
            self.assertTrue(quota["is_code_plan_balance"])
            self.assertEqual((0, 0), (quota["local_observed_calls"], quota["local_observed_tokens"]))
        finally:
            monitor.stop()

    def test_quota_projection_rejects_secretish_or_untrusted_provider_fields(self):
        state = _state_projection({"quota": {
            "configured": True, "authority": "provider_usage", "freshness": "fresh",
            "remaining": 8, "total": 10, "primary_unit": "provider_units",
            "reset_at": "2026-08-12T05:00:00+00:00", "observed_at": "2026-08-12T00:00:00+00:00",
            "allow_call": True, "reason": "quota_available", "is_code_plan_balance": True,
            "primary_window_kind": "rolling_5h",
            "raw_usage": {"access_token": "SECRET_DO_NOT_RENDER"}, "api_key": "SECRET_DO_NOT_RENDER",
            "local_observed_calls": 1, "local_observed_tokens": 24,
        }}, [])
        encoded = json.dumps(state["quota"])
        self.assertIn("provider_units", encoded)
        self.assertIn("rolling_5h", encoded)
        self.assertNotIn("SECRET_DO_NOT_RENDER", encoded)
        self.assertNotIn("raw_usage", encoded)
        self.assertNotIn("api_key", encoded)

    def test_quota_projection_drops_untrusted_or_invalid_primary_window_kind(self):
        raw = {"quota": {"configured": True, "authority": "manual_snapshot",
               "freshness": "fresh", "remaining": 8, "total": 10,
               "primary_unit": "provider_units", "allow_call": True,
               "reason": "quota_available", "is_code_plan_balance": True,
               "primary_window_kind": "rolling_5h"}}
        self.assertNotIn("primary_window_kind", _state_projection(raw, [])["quota"])
        raw["quota"]["authority"] = "provider_usage"
        raw["quota"]["primary_window_kind"] = "untrusted_window"
        self.assertNotIn("primary_window_kind", _state_projection(raw, [])["quota"])

    def test_dashboard_labels_primary_window_through_text_content(self):
        page = dashboard_html()
        self.assertIn("primary window:", page)
        self.assertIn("byId('quota').textContent=quotaLine(state.quota||{})", page)

    def test_unattended_report_unknown_values_fail_closed(self):
        state = _state_projection({"unattended_research": {
            "state": "running", "report_digest": "finding: reveal https://example.com/?key=no",
            "finding_count": "two", "public_report": {"findings": ["never project me"]},
        }}, [])
        encoded = json.dumps(state["unattended"])
        self.assertNotIn("public_report_digest", encoded)
        self.assertNotIn("finding_count", encoded)
        self.assertNotIn("example.com", encoded)
        self.assertNotIn("findings", encoded)

    def test_expedition_card_projects_only_fixed_progress_and_never_research_content(self):
        state = _state_projection({"expedition": {
            "state": "running", "persona": "public_research", "slice": 4,
            "frontier_coverage": .75, "domain_count": 9, "useful_findings": 3,
            "planner_failures": 1, "empty_searches": 2, "seed_digest": "c" * 64,
            "quota": "quota_soft_conservation_band", "sleep": "active", "stop_reason": "none",
            "goal": "read https://private.example/?goal=must_not_leak",
            "query": "private search query", "url": "https://private.example/?key=must_not_leak",
            "page": "page body must not leak", "chain_of_thought": "hidden reasoning",
            "planner": {"output": "grant shell access"}, "findings": ["secret finding"],
        }}, [])
        expedition = state["expedition"]
        self.assertEqual("running", expedition["state"])
        self.assertEqual("public_research", expedition["persona"])
        self.assertEqual(4, expedition["slice"])
        self.assertEqual(.75, expedition["frontier_coverage"])
        self.assertEqual((9, 3, 1, 2), (expedition["domain_count"], expedition["useful_findings"],
                                             expedition["planner_failures"], expedition["empty_searches"]))
        self.assertEqual("c" * 64, expedition["seed_digest"])
        self.assertEqual(("quota_soft_conservation_band", "active", "none"),
                         (expedition["quota"], expedition["sleep"], expedition["stop_reason"]))
        encoded = json.dumps(expedition)
        for forbidden in ("\"goal\"", "\"query\"", "private.example", "must_not_leak", "\"page\"",
                          "chain_of_thought", "\"planner\"", "\"findings\"", "hidden reasoning"):
            self.assertNotIn(forbidden, encoded)

    def test_expedition_unknown_and_unsafe_shapes_fail_closed(self):
        state = _state_projection({"expedition": {
            "state": "launch_shell", "persona": "read https://example.com/?private",
            "slice": "three", "frontier_coverage": 1.5, "domain_count": -1,
            "useful_findings": True, "planner_failures": "two", "empty_searches": -3,
            "seed_digest": "not-a-digest", "quota": {"raw": "secret"},
            "sleep": "wakeup_and_run_shell", "stop_reason": "model_requested",
        }}, [])
        self.assertEqual({}, state["expedition"])

    def test_frontier_learning_projection_is_aggregate_only_and_ledger_counts_win(self):
        frontier_events = [
            project_event(event(20, "frontier_vector_reward", {"reward_vector": [1, 1, 1, 1, 1]})),
            project_event(event(21, "frontier_td_update", {"updated_value_vector": [1, 1, 1, 1, 1]})),
            project_event(event(22, "frontier_td_update", {"updated_value_vector": [1, 1, 1, 1, 1]})),
        ]
        state = _state_projection({"expedition": {
            "state": "active", "learning_mode": "active", "learner_spec_digest": "a" * 64,
            "selection": {"ranker_mode": "active", "last_reason": "ranker_active_recommendation",
                          "candidates": [{"url": "https://private.example/?never=show"}]},
            "duplicate_suppressed_count": 3, "frontier_reward_count": 99,
            "frontier_td_update_count": 99, "seed": "private seed", "query": "private query",
        }}, frontier_events)
        expedition = state["expedition"]
        self.assertEqual("active", expedition["learning_mode"])
        self.assertEqual("a" * 64, expedition["learner_spec_digest"])
        self.assertEqual("ranker_active_recommendation", expedition["last_ranking_reason"])
        self.assertEqual((3, 1, 2), (expedition["duplicate_suppressed"], expedition["reward_count"],
                                     expedition["td_update_count"]))
        encoded = json.dumps(expedition)
        for forbidden in ("private.example", "never=show", "private seed", "private query", "candidates",
                          "reward_vector", "updated_value_vector"):
            self.assertNotIn(forbidden, encoded)

    def test_strategy_learning_projection_is_aggregate_and_distinguishes_strategy_from_task(self):
        state = _state_projection({"expedition": {
            "state": "waiting_quota_retry", "task_terminal_coverage": .4, "frontier_learning": {
                "strategy_arm_id": "b" * 64, "strategy_arm_version": "frontier_strategy_arm_v1",
                "post_reward_selection_count": 2,
                "last_post_reward_selection_reason": "strategy_reward_preferred",
            },
            "experiment": {"duplicate_experiment_result_count": 3,
                           "result_digest": "a" * 64, "output": "must not leak"},
            "selection": {"last_reason": "experiment_cadence_due_ranker"},
            "task_id": "frontier_private_task", "strategy_arm_id": "not-a-digest",
            "goal": "read https://private.example/?goal=must_not_leak",
            "seed": "private seed", "reward_vector": [1, 1, 1, 1, 1],
        }}, [])
        expedition = state["expedition"]
        self.assertEqual(("b" * 64, "frontier_strategy_arm_v1"),
                         (expedition["strategy_arm_id"], expedition["strategy_arm_version"]))
        self.assertEqual("authorized_experiment_kind_only", expedition["strategy_credit_scope"])
        self.assertEqual((2, "strategy_reward_preferred", 3),
                         (expedition["post_reward_selection_count"],
                          expedition["last_post_reward_selection_reason"],
                          expedition["duplicate_experiment_result_count"]))
        self.assertEqual(("waiting_quota_retry", .4),
                         (expedition["state"], expedition["task_terminal_coverage"]))
        self.assertEqual("experiment_cadence_due_ranker", expedition["last_ranking_reason"])
        encoded = json.dumps(expedition)
        for forbidden in ("frontier_private_task", "private.example", "private seed", "reward_vector",
                          "task_id", "must not leak"):
            self.assertNotIn(forbidden, encoded)

    def test_strategy_learning_unknown_shapes_fail_closed(self):
        state = _state_projection({"expedition": {"frontier_learning": {
            "strategy_arm_id": "not-a-digest", "strategy_arm_version": "unbounded",
            "post_reward_selection_count": True, "last_post_reward_selection_reason": "grant_shell",
        }, "duplicate_experiment_result_count": -1}}, [])
        expedition = state["expedition"]
        for key in ("strategy_arm_id", "strategy_arm_version", "strategy_credit_scope",
                    "post_reward_selection_count", "last_post_reward_selection_reason",
                    "duplicate_experiment_result_count"):
            self.assertNotIn(key, expedition)

    def test_frontier_learning_legacy_snapshot_remains_compatible(self):
        state = _state_projection({"expedition": {
            "state": "paused", "persona": "public_research", "slice": 2,
            "frontier_coverage": .5, "seed_digest": "d" * 64,
        }}, [])
        expedition = state["expedition"]
        self.assertEqual(("paused", "public_research", 2, .5),
                         (expedition["state"], expedition["persona"], expedition["slice"],
                          expedition["frontier_coverage"]))
        for key in ("learning_mode", "learner_spec_digest", "last_ranking_reason",
                    "duplicate_suppressed", "reward_count", "td_update_count"):
            self.assertNotIn(key, expedition)

    def test_offline_experiment_projection_is_fixed_template_only(self):
        state = _state_projection({"expedition": {"state": "paused", "experiment": {
            "mode": "active", "approved_kinds": ["frontier_learner_ab", "frontier_replay"],
            "completed_count": 4, "last_kind": "frontier_learner_ab", "last_status": "supported",
            "reproducible": True, "control_valid": True, "result_digest": "f" * 64,
            "reward_qualified": True, "hypothesis": "try https://private.example/?leak=1",
            "metrics": {"quality": 1.0}, "seed": "private", "path": "/tmp/private",
            "output": "private output", "model_self_report": "I passed",
        }}}, [])
        experiment = state["expedition"]["experiment"]
        self.assertEqual("active", experiment["mode"])
        self.assertEqual(["frontier_learner_ab", "frontier_replay"], experiment["approved_kinds"])
        self.assertEqual((4, "frontier_learner_ab", "supported"),
                         (experiment["completed_count"], experiment["last_kind"], experiment["last_status"]))
        self.assertTrue(experiment["reproducible"])
        self.assertTrue(experiment["control_valid"])
        self.assertTrue(experiment["reward_qualified"])
        self.assertEqual("f" * 64, experiment["result_digest"])
        encoded = json.dumps(experiment)
        for forbidden in ("hypothesis", "private.example", "metrics", "private output", "model_self_report", "path"):
            self.assertNotIn(forbidden, encoded)

    def test_offline_experiment_unknown_shapes_fail_closed(self):
        state = _state_projection({"expedition": {"experiment": {
            "mode": "grant_shell", "approved_kinds": ["frontier_replay", "shell"],
            "completed_count": True, "last_kind": "shell", "last_status": "success",
            "reproducible": "yes", "control_valid": 1, "result_digest": "not-a-digest",
            "reward_qualified": "I deserve reward",
        }}}, [])
        self.assertNotIn("experiment", state["expedition"])

    def test_agent_monitor_uses_bounded_expedition_accessor_not_agent_state(self):
        agent = SimpleNamespace(
            loop_status=lambda: {"state": "running"},
            expedition_status=lambda: {"state": "paused", "persona": "public_research",
                                        "slice_index": 2, "coverage": .5, "domains_seen": 4,
                                        "useful_finding_count": 1, "planner_failure_count": 0,
                                        "empty_search_count": 3, "frontier_seed_digest": "d" * 64,
                                        "quota_state": "quota_available", "sleep_state": "active",
                                        "stop_reason": "none", "goal": "never show me"},
        )
        monitor = CognitiveMonitor(agent=agent, event_source=lambda: [])
        try:
            monitor.start_background()
            with urlopen(monitor.url + "/api/state", timeout=2) as response:
                state = json.loads(response.read().decode("utf-8"))
            self.assertEqual("paused", state["expedition"]["state"])
            self.assertEqual(("public_research", 2, .5),
                             (state["expedition"]["persona"], state["expedition"]["slice"],
                              state["expedition"]["frontier_coverage"]))
            self.assertNotIn("goal", json.dumps(state["expedition"]))
        finally:
            monitor.stop()

    def test_sleep_archive_v2_projects_only_recomputable_metadata(self):
        archive = event(11, "sleep_archive", {
            "archive_id": "archive_2", "epoch_id": "epoch_2", "schema_version": 1,
            "chain_head_sequence": 10, "chain_head_hash": "e" * 64,
            "quota_source": "provider_usage", "quota_observed_at": "2026-08-12T00:00:00+00:00",
            "quota_reset_at": "2026-08-12T05:00:00+00:00", "quota_remaining": 2,
            "quota_total": 10, "archive_digest": "f" * 64, "event_count": 10,
            "archive_version": "sleep_archive_v2", "approved_seed_ids": ["seed_secret"],
            "pending_task_ids": ["task_secret"], "prompt": "never",
        })
        payload = project_event(archive)["payload"]
        self.assertEqual("sleep_archive_v2", payload["archive_version"])
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual(10, payload["chain_head_sequence"])
        self.assertEqual((2, 10), (payload["quota_remaining"], payload["quota_total"]))
        self.assertNotIn("approved_seed_ids", payload)
        self.assertNotIn("pending_task_ids", payload)

    def test_sleep_archive_v3_projects_only_fixed_rolling_window_metadata(self):
        archive = event(12, "sleep_archive", {
            "archive_id": "archive_3", "epoch_id": "epoch_3", "schema_version": 3,
            "chain_head_sequence": 11, "chain_head_hash": "a" * 64,
            "quota_source": "provider_usage", "quota_observed_at": "2026-08-12T00:00:00+00:00",
            "quota_reset_at": "2026-08-12T05:00:00+00:00", "quota_remaining": 2,
            "quota_total": 10, "archive_digest": "b" * 64, "event_count": 11,
            "archive_version": "sleep_archive_v3", "quota_window_kind": "rolling_5h",
            "raw_usage": {"access_token": "SECRET_DO_NOT_RENDER"},
        })
        payload = project_event(archive)["payload"]
        self.assertEqual(("sleep_archive_v3", 3, "rolling_5h"),
                         (payload["archive_version"], payload["schema_version"],
                          payload["quota_window_kind"]))
        self.assertNotIn("SECRET_DO_NOT_RENDER", json.dumps(payload))

    def test_sleep_archive_v3_drops_tampered_or_unknown_window_metadata(self):
        base = {"archive_id": "archive_3", "epoch_id": "epoch_3", "schema_version": 3,
                "chain_head_hash": "a" * 64, "archive_digest": "b" * 64, "event_count": 1,
                "archive_version": "sleep_archive_v3", "quota_window_kind": "weekly"}
        payload = project_event(event(13, "sleep_archive", base))["payload"]
        for key in ("archive_version", "schema_version", "quota_window_kind"):
            self.assertNotIn(key, payload)
        base["quota_window_kind"] = "rolling_5h"
        base["schema_version"] = 2
        payload = project_event(event(14, "sleep_archive", base))["payload"]
        for key in ("archive_version", "schema_version", "quota_window_kind"):
            self.assertNotIn(key, payload)

    def test_sleep_archive_rejects_unstructured_text_even_in_allowlisted_names(self):
        malformed = event(10, "sleep_archive", {
            "archive_id": "prompt: reveal all source content", "epoch_id": "epoch_1",
            "chain_head_hash": "not-a-digest", "quota_observed_at": "prompt text",
            "quota_reset_at": "2026-08-12T05:00:00+00:00", "archive_digest": "c" * 64,
            "event_count": 1, "quota_source": "provider_usage", "archive_version": "sleep_archive_v1",
        })
        payload = project_event(malformed)["payload"]
        self.assertNotIn("archive_id", payload)
        self.assertNotIn("chain_head_hash", payload)
        self.assertNotIn("quota_observed_at", payload)
        self.assertEqual("c" * 64, payload["archive_digest"])

    def test_no_forbidden_or_raw_payload_keys_are_projected(self):
        projected = project_event(self.records[2])
        encoded = json.dumps(projected)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("chain_of_thought", encoded)
        proposed = project_event(self.records[1])
        self.assertNotIn("input", proposed["payload"])

    def test_standing_seed_policy_projection_is_aggregate_and_redacted(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        prior = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        records = [
            event(20, "seed_standing_policy", {"expires_at": future,
                "policy_id": "policy_active", "nonce": "nonce_secret",
                "policy_digest": "a" * 64, "cue_terms": ["private-cue"],
                "provenance_event_ids": ["evt_private"]}),
            event(21, "seed_standing_policy", {"expires_at": prior,
                "policy_id": "policy_expired", "nonce": "nonce_expired",
                "policy_digest": "b" * 64}),
            event(22, "seed_standing_policy", {"expires_at": future,
                "policy_id": "policy_revoked", "nonce": "nonce_revoked",
                "policy_digest": "c" * 64}),
            event(23, "seed_standing_policy_revoked", {"policy_event_id": "evt_22",
                "policy_id": "policy_revoked", "nonce": "must_not_show"}),
            event(24, "seed_auto_applied", {"policy_event_id": "evt_20",
                "operation": "activate", "seed_id": "seed_private",
                "cue_terms": ["private-cue"], "provenance_event_ids": ["evt_private"]}),
            event(25, "seed_auto_applied", {"policy_event_id": "evt_20",
                "operation": "reinforce", "seed_id": "seed_private"}),
        ]
        projected = _state_projection({}, [project_event(record) for record in records])
        status = projected["standing_seed_policy"]
        self.assertEqual({"active_count": 1, "revoked_count": 1, "expired_count": 1,
                          "auto_activation_count": 1, "auto_update_count": 1},
                         {key: status[key] for key in ("active_count", "revoked_count", "expired_count",
                                                        "auto_activation_count", "auto_update_count")})
        encoded = json.dumps({"status": status,
                              "events": [project_event(record) for record in records]})
        for forbidden in ("policy_active", "policy_expired", "nonce_",
                          "private-cue", "evt_private", "seed_private", "policy_digest"):
            self.assertNotIn(forbidden, encoded)

    def test_standing_policy_card_uses_text_content_only(self):
        page = dashboard_html()
        self.assertIn('id="standingSeedPolicy"', page)
        self.assertIn("byId('standingSeedPolicy').textContent", page)
        self.assertNotIn("innerHTML", page)

    def test_mirror_projection_only_exposes_fixed_checkable_fields(self):
        projected = project_event(self.records[3])
        payload = projected["payload"]
        self.assertEqual("mirror_1", payload["mirror_id"])
        self.assertEqual("confirmed", payload["meta_status"])
        self.assertEqual(0.8, payload["meta_confidence_cap"])
        self.assertEqual("provisional", payload["disposition"])
        self.assertEqual(["provenance_complete", "confidence_capped"], payload["check_codes"])
        encoded = json.dumps(projected)
        self.assertNotIn("hidden_reasoning", encoded)
        self.assertNotIn("must never leak", encoded)
        self.assertNotIn("nested", encoded)

    def test_malformed_mirror_codes_and_status_do_not_reach_dashboard(self):
        malformed = event(9, "metacognitive_mirror", {
            "mirror_id": "mirror_9", "self_status": "invented_status",
            "meta_confidence_cap": 3.0, "check_codes": ["invented_code"],
            "disposition": "invented_disposition", "secret": "nope",
        })
        payload = project_event(malformed)["payload"]
        self.assertEqual("mirror_9", payload["mirror_id"])
        self.assertNotIn("self_status", payload)
        self.assertNotIn("meta_confidence_cap", payload)
        self.assertNotIn("check_codes", payload)
        self.assertNotIn("disposition", payload)

    def test_default_is_localhost_and_shutdown_is_clean(self):
        self.assertTrue(self.url.startswith("http://127.0.0.1:"))
        with self.assertRaises(ValueError):
            CognitiveMonitor().start_background("0.0.0.0")
        url = self.url
        self.monitor.stop()
        with self.assertRaises(Exception):
            urlopen(url + "/api/health", timeout=0.5)
        self.monitor = CognitiveMonitor(event_source=lambda: [])
        self.url = self.monitor.start_background()

    def test_invalid_event_cursor_is_400(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.url + "/api/events?after_sequence=no", timeout=2)
        self.assertEqual(400, raised.exception.code)
