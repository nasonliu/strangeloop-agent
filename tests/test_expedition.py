from datetime import datetime, timedelta, timezone
import unittest

from strangeloop.expedition import (ExpeditionConfig, ExpeditionOutcome,
                                    ExpeditionScheduler, ExpeditionState,
                                    FrontierStatus, Persona, SeedGuidanceSnapshot)
from strangeloop.contracts import FRONTIER_STRATEGY_ARM_VERSION, frontier_strategy_arm_id


NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


class ExpeditionSchedulerTests(unittest.TestCase):
    def build(self, **config):
        return ExpeditionScheduler("stable-test-seed", ExpeditionConfig(**config), NOW)

    @staticmethod
    def experiment_config(**values):
        values.setdefault("authorization_version", "expedition_authorization_v3")
        values.setdefault("experiment_registry_digest", "a" * 64)
        values.setdefault("learner_spec_digest", "b" * 64)
        return ExpeditionConfig(**values)

    def test_seed_is_reproducible_and_rotates_all_personas(self):
        first = self.build()
        second = self.build()
        personas = []
        for number in range(6):
            decision = first.begin_slice(NOW + timedelta(seconds=number))
            mirror = second.begin_slice(NOW + timedelta(seconds=number))
            self.assertEqual(decision.persona, mirror.persona)
            self.assertEqual(decision.task.task_id, mirror.task.task_id)
            self.assertEqual(decision.task.topic_cluster, mirror.task.topic_cluster)
            personas.append(decision.persona)
            first.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                 now=NOW + timedelta(seconds=number))
            second.record_outcome(mirror.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                  now=NOW + timedelta(seconds=number))
        self.assertEqual([Persona(item) for item in ("forager", "diver", "contrarian", "prospector", "bridge", "curator")], personas)

    def test_same_seed_has_identical_public_trajectory_ids(self):
        config = ExpeditionConfig(max_slices=8)
        left = ExpeditionScheduler("replay-seed", config, NOW)
        right = ExpeditionScheduler("replay-seed", config, NOW)
        left_ids, right_ids = [], []
        for number in range(8):
            at = NOW + timedelta(seconds=number)
            left_slice, right_slice = left.begin_slice(at), right.begin_slice(at)
            left_ids.append(left_slice.task.task_id)
            right_ids.append(right_slice.task.task_id)
            left.record_outcome(left_slice.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE, now=at)
            right.record_outcome(right_slice.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE, now=at)
        self.assertEqual(left_ids, right_ids)
        self.assertEqual(left.public_snapshot(), right.public_snapshot())

    def test_different_seed_changes_ids_but_not_hard_boundaries(self):
        config = ExpeditionConfig(total_authorization_seconds=360, max_slice_seconds=120,
                                 max_slices=8, max_frontier_tasks=12)
        left = ExpeditionScheduler("seed-a", config, NOW)
        right = ExpeditionScheduler("seed-b", config, NOW)
        left_slice, right_slice = left.begin_slice(NOW), right.begin_slice(NOW)
        self.assertNotEqual(left_slice.task.task_id, right_slice.task.task_id)
        self.assertEqual(left_slice.persona, right_slice.persona)
        self.assertEqual(left_slice.max_seconds, right_slice.max_seconds)
        left_public, right_public = left.public_snapshot(), right.public_snapshot()
        self.assertEqual(left_public["authorization_seconds"], right_public["authorization_seconds"])
        self.assertEqual(left_public["max_slice_seconds"], right_public["max_slice_seconds"])

    def test_branch_complete_does_not_end_expedition(self):
        scheduler = self.build()
        decision = scheduler.begin_slice(NOW)
        scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE, now=NOW)
        self.assertEqual(ExpeditionState.READY, scheduler.state)
        self.assertIsNotNone(scheduler.begin_slice(NOW + timedelta(seconds=1)))

    def test_terminal_tasks_reclaim_a_bounded_candidate_slot(self):
        """A working-set cap must not end an authorized episode early."""
        scheduler = self.build(max_frontier_tasks=6, max_slices=8)
        completed = set()
        for number in range(6):
            at = NOW + timedelta(seconds=number)
            decision = scheduler.begin_slice(at)
            completed.add(decision.task.task_id)
            scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE, now=at)

        next_decision = scheduler.begin_slice(NOW + timedelta(seconds=6))
        self.assertIsNotNone(next_decision)
        self.assertEqual(ExpeditionState.ACTIVE, scheduler.state)
        self.assertNotIn(next_decision.task.task_id, completed)
        self.assertEqual(6, len(scheduler._tasks))

    def test_empty_search_retries_with_fallback_then_drops_finitely(self):
        scheduler = self.build(max_empty_searches=2, max_task_attempts=4)
        first = scheduler.begin_slice(NOW)
        original_mode = first.task.action_mode
        scheduler.record_outcome(first.task.task_id, ExpeditionOutcome.EMPTY_SEARCH, now=NOW)
        task = scheduler._tasks[first.task.task_id]
        self.assertEqual(FrontierStatus.DEFERRED, task.status)
        self.assertNotEqual(original_mode, task.action_mode)
        # The role order comes back to Forager on slice 7.
        scheduler._slice_number = 6
        task.not_before_slice = 7
        second = scheduler.begin_slice(NOW + timedelta(seconds=1))
        self.assertEqual(first.task.task_id, second.task.task_id)
        scheduler.record_outcome(second.task.task_id, ExpeditionOutcome.EMPTY_SEARCH, now=NOW)
        self.assertEqual(FrontierStatus.DROPPED, scheduler._tasks[first.task.task_id].status)
        self.assertEqual(ExpeditionState.READY, scheduler.state)

    def test_planner_failure_is_not_no_work_and_has_backoff(self):
        scheduler = self.build(max_planner_failures=3)
        decision = scheduler.begin_slice(NOW)
        scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.PLANNER_TRANSIENT_FAILURE, now=NOW)
        task = scheduler._tasks[decision.task.task_id]
        self.assertEqual(1, task.planner_failures)
        self.assertEqual(FrontierStatus.DEFERRED, task.status)
        self.assertEqual(ExpeditionState.READY, scheduler.state)

    def test_http_success_is_not_progress_without_quality_evidence(self):
        scheduler = self.build()
        decision = scheduler.begin_slice(NOW)
        scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.QUALITY_PROGRESS,
                                 quality_score=0.2, quality_signals=(), now=NOW)
        task = scheduler._tasks[decision.task.task_id]
        self.assertEqual(1, task.low_value_results)
        self.assertNotEqual(FrontierStatus.COMPLETE, task.status)

    def test_authorization_expiry_and_sleep_wake_are_host_gated(self):
        scheduler = self.build(total_authorization_seconds=10)
        scheduler.sleep()
        self.assertFalse(scheduler.wake(False, NOW + timedelta(seconds=1)))
        self.assertTrue(scheduler.wake(True, NOW + timedelta(seconds=1)))
        self.assertEqual(ExpeditionState.READY, scheduler.state)
        self.assertIsNone(scheduler.begin_slice(NOW + timedelta(seconds=11)))
        self.assertEqual(ExpeditionState.COMPLETED, scheduler.state)

    def test_quota_retry_waits_without_starting_slice_then_requires_fresh_resume(self):
        scheduler = self.build()
        retry_at = NOW + timedelta(seconds=10)
        scheduler.defer_quota_retry("quota_refresh_transient_retry_pending", retry_at, NOW)
        self.assertEqual(ExpeditionState.WAITING_QUOTA_RETRY, scheduler.state)
        self.assertIsNone(scheduler.begin_slice(NOW + timedelta(seconds=9)))
        self.assertTrue(scheduler.quota_retry_due(retry_at))
        self.assertFalse(scheduler.resume_after_fresh_quota(NOW + timedelta(seconds=9)))
        self.assertTrue(scheduler.resume_after_fresh_quota(retry_at))
        self.assertIsNotNone(scheduler.begin_slice(retry_at))

    def test_quota_retry_is_finite_and_exhaustion_stops_without_sleeping(self):
        scheduler = self.build(max_quota_retry_attempts=2)
        for number in range(2):
            now = NOW + timedelta(seconds=number * 20)
            scheduler.defer_quota_retry("quota_refresh_rejected_fail_closed", now + timedelta(seconds=10), now)
            scheduler.resume_after_fresh_quota(now + timedelta(seconds=10))
        scheduler.defer_quota_retry("quota_refresh_old_or_unknown_fail_closed", NOW + timedelta(seconds=60), NOW + timedelta(seconds=50))
        self.assertEqual(ExpeditionState.STOPPED, scheduler.state)
        self.assertEqual("quota_refresh_retry_exhausted", scheduler.public_snapshot()["stop_reason"])

    def test_quota_refresh_cadence_wait_does_not_consume_transient_retry_budget(self):
        scheduler = self.build(max_quota_retry_attempts=1)
        scheduler.defer_quota_retry("quota_refresh_interval_waiting", NOW + timedelta(seconds=10), NOW)
        snapshot = scheduler.public_snapshot()
        self.assertEqual(0, snapshot["quota_retry"]["attempts"])
        self.assertFalse(scheduler.resume_after_fresh_quota(NOW + timedelta(seconds=9)))
        self.assertTrue(scheduler.resume_after_fresh_quota(NOW + timedelta(seconds=10)))
        self.assertEqual(ExpeditionState.READY, scheduler.state)

    def test_public_snapshot_contains_no_seed_or_network_content(self):
        scheduler = self.build()
        rendered = repr(scheduler.public_snapshot())
        self.assertNotIn("stable-test-seed", rendered)
        self.assertNotIn("http://", rendered)
        self.assertNotIn("https://", rendered)
        self.assertIn("no_network_execution", rendered)

    def test_default_ranker_absence_preserves_persona_rotation(self):
        scheduler = self.build()
        decision = scheduler.begin_slice(NOW)
        self.assertEqual(Persona.FORAGER, decision.persona)
        self.assertEqual("legacy_persona_rotation", decision.selection_reason)
        snapshot = scheduler.candidate_snapshot()
        self.assertEqual("not_configured", snapshot["recommendation_status"])
        self.assertNotIn("stable-test-seed", repr(snapshot))

    def test_active_ranker_can_choose_cross_persona_from_public_descriptors(self):
        class PreferCurator:
            def __init__(self):
                self.candidates = None
                self.context = None

            def rank(self, candidates, state_context):
                self.candidates, self.context = candidates, state_context
                ids = [item["task_id"] for item in candidates]
                curator = [item["task_id"] for item in candidates if item["persona"] == "curator"]
                return curator + [item for item in ids if item not in curator]

        ranker = PreferCurator()
        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(), NOW, ranker=ranker)
        decision = scheduler.begin_slice(NOW)
        self.assertEqual(Persona.CURATOR, decision.persona)
        self.assertEqual("ranker_active_recommendation", decision.selection_reason)
        self.assertEqual(6, len(ranker.candidates))
        self.assertEqual("forager", ranker.context["scheduled_persona"])
        self.assertNotIn("stable-test-seed", repr(ranker.candidates))

    def test_invalid_ranker_recommendation_fails_closed_to_baseline(self):
        class InvalidRanker:
            def rank(self, candidates, state_context):
                del candidates, state_context
                return ("unknown-task",)

        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(), NOW,
                                        ranker=InvalidRanker())
        decision = scheduler.begin_slice(NOW)
        self.assertEqual(Persona.FORAGER, decision.persona)
        self.assertEqual("ranker_invalid_fallback", decision.selection_reason)
        self.assertEqual("ranker_invalid_fallback",
                         scheduler.candidate_snapshot()["recommendation_status"])

    def test_shadow_ranker_is_observed_but_cannot_change_baseline_selection(self):
        class PreferCurator:
            def rank(self, candidates, state_context):
                del state_context
                ids = [item["task_id"] for item in candidates]
                curator = [item["task_id"] for item in candidates if item["persona"] == "curator"]
                return curator + [item for item in ids if item not in curator]

        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(), NOW,
                                        ranker=PreferCurator(), ranker_active=False)
        decision = scheduler.begin_slice(NOW)
        self.assertEqual(Persona.FORAGER, decision.persona)
        self.assertEqual("ranker_shadow_observed_baseline", decision.selection_reason)

    def test_active_ranker_cannot_starve_a_due_persona(self):
        class PreferCurator:
            def rank(self, candidates, state_context):
                del state_context
                ids = [item["task_id"] for item in candidates]
                curator = [item["task_id"] for item in candidates if item["persona"] == "curator"]
                return curator + [item for item in ids if item not in curator]

        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(max_slices=8), NOW,
                                        ranker=PreferCurator())
        # Model an already-expanded frontier: without host-side fairness a
        # ranker could keep consuming these same-persona arms forever.
        for index in range(6):
            scheduler._add_task(Persona.CURATOR, index + 100)
        decisions = []
        for number in range(7):
            decision = scheduler.begin_slice(NOW + timedelta(seconds=number))
            decisions.append(decision)
            scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                     now=NOW + timedelta(seconds=number))
        self.assertEqual(Persona.FORAGER, decisions[6].persona)
        self.assertEqual("fairness_persona_due", decisions[6].selection_reason)

    def test_quota_and_stop_remain_host_controlled_with_ranker(self):
        class UnexpectedRanker:
            def rank(self, candidates, state_context):
                raise AssertionError("ranker must not run while no slice can begin")

        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(), NOW,
                                        ranker=UnexpectedRanker())
        scheduler.defer_quota_retry("quota_refresh_transient_retry_pending", NOW + timedelta(seconds=10), NOW)
        self.assertIsNone(scheduler.begin_slice(NOW + timedelta(seconds=1)))
        scheduler.stop()
        self.assertIsNone(scheduler.begin_slice(NOW + timedelta(seconds=11)))

    def test_experiments_require_v3_and_use_fixed_public_kinds(self):
        with self.assertRaises(ValueError):
            ExpeditionConfig(allowed_experiment_kinds=("frontier_replay",), max_experiments=1)
        with self.assertRaises(ValueError):
            ExpeditionConfig(authorization_version="expedition_authorization_v3",
                             allowed_experiment_kinds=(), max_experiments=1)
        with self.assertRaises(ValueError):
            ExpeditionConfig(authorization_version="expedition_authorization_v3",
                             allowed_experiment_kinds=("unknown",), max_experiments=1)
        with self.assertRaises(ValueError):
            ExpeditionConfig(authorization_version="expedition_authorization_v3",
                             allowed_experiment_kinds=("frontier_replay",), max_experiments=1)
        config = self.experiment_config(allowed_experiment_kinds=("frontier_replay", "td_invariants"),
                                        max_experiments=3)
        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW)
        experiments = [item for item in scheduler.public_snapshot()["tasks"]
                       if item["experiment_kind"] is not None]
        self.assertEqual(3, len(experiments))
        self.assertEqual(["frontier_replay", "td_invariants", "frontier_replay"],
                         [item["experiment_kind"] for item in experiments])
        self.assertTrue(all(item["action_mode"] == "offline_experiment" for item in experiments))
        self.assertNotIn("http", repr(experiments))

    def test_experiment_cadence_is_bounded_and_shadow_auditable(self):
        config = self.experiment_config(allowed_experiment_kinds=("frontier_replay",), max_experiments=1,
                                        max_slices=8)
        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW)
        decisions = []
        for number in range(6):
            decision = scheduler.begin_slice(NOW + timedelta(seconds=number))
            decisions.append(decision)
            scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                     now=NOW + timedelta(seconds=number))
        self.assertTrue(any(item.task.experiment_kind for item in decisions))
        self.assertEqual("experiment_cadence_due", decisions[-1].selection_reason)
        self.assertTrue(any(item.task.experiment_kind is None for item in decisions))

        class WebFirst:
            def rank(self, candidates, state_context):
                del state_context
                web = [item["task_id"] for item in candidates if item["experiment_kind"] is None]
                exp = [item["task_id"] for item in candidates if item["experiment_kind"] is not None]
                return web + exp
        shadow = ExpeditionScheduler("stable-test-seed", config, NOW, ranker=WebFirst(), ranker_active=False)
        for number in range(5):
            decision = shadow.begin_slice(NOW + timedelta(seconds=number))
            shadow.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                  now=NOW + timedelta(seconds=number))
        due = shadow.begin_slice(NOW + timedelta(seconds=5))
        self.assertIsNotNone(due.task.experiment_kind)
        self.assertEqual("experiment_cadence_due_shadow", due.selection_reason)

    def test_experiment_progress_requires_supported_or_refuted_evidence(self):
        config = self.experiment_config(allowed_experiment_kinds=("td_invariants",), max_experiments=1)
        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW)
        for number in range(5):
            decision = scheduler.begin_slice(NOW + timedelta(seconds=number))
            scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                     now=NOW + timedelta(seconds=number))
        experiment = scheduler.begin_slice(NOW + timedelta(seconds=5))
        self.assertIsNotNone(experiment.task.experiment_kind)
        scheduler.record_outcome(experiment.task.task_id, ExpeditionOutcome.EXPERIMENT_INCONCLUSIVE,
                                 quality_score=1.0,
                                 quality_signals=("independent_reproduction",), now=NOW)
        self.assertNotEqual(FrontierStatus.COMPLETE, experiment.task.status)

        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW)
        for number in range(5):
            decision = scheduler.begin_slice(NOW + timedelta(seconds=number))
            scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                     now=NOW + timedelta(seconds=number))
        experiment = scheduler.begin_slice(NOW + timedelta(seconds=5))
        scheduler.record_outcome(experiment.task.task_id, ExpeditionOutcome.EXPERIMENT_REFUTED,
                                 quality_score=.7,
                                 quality_signals=("paired_baseline_treatment", "independent_reproduction"), now=NOW)
        self.assertEqual(FrontierStatus.COMPLETE, experiment.task.status)

    def test_same_experiment_kind_shares_a_stable_strategy_arm(self):
        config = self.experiment_config(allowed_experiment_kinds=("frontier_replay",), max_experiments=3)
        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW)
        experiments = [task for task in scheduler._tasks.values() if task.experiment_kind is not None]
        self.assertEqual(1, len({task.strategy_arm_id for task in experiments}))
        self.assertEqual({FRONTIER_STRATEGY_ARM_VERSION},
                         {task.strategy_version for task in experiments})
        self.assertEqual(3, len({task.task_id for task in experiments}))
        self.assertEqual(frontier_strategy_arm_id("a" * 64, "b" * 64, "frontier_replay"),
                         experiments[0].strategy_arm_id)

    def test_repeated_experiment_groups_are_deterministic_across_replay(self):
        config = self.experiment_config(allowed_experiment_kinds=("frontier_replay", "td_invariants"),
                                        max_experiments=5)
        left = ExpeditionScheduler("stable-test-seed", config, NOW)
        right = ExpeditionScheduler("stable-test-seed", config, NOW)
        def groups(scheduler):
            return [(task.experiment_kind, task.strategy_arm_id) for task in scheduler._tasks.values()
                    if task.experiment_kind is not None]
        self.assertEqual(groups(left), groups(right))
        self.assertEqual(["frontier_replay", "td_invariants", "frontier_replay", "td_invariants",
                          "frontier_replay"], [kind for kind, unused in groups(left)])

    def test_experiment_strategy_arm_isolated_by_kind_and_bound_context(self):
        config = self.experiment_config(allowed_experiment_kinds=("frontier_replay", "td_invariants"),
                                        max_experiments=2)
        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW)
        by_kind = {task.experiment_kind: task.strategy_arm_id for task in scheduler._tasks.values()
                   if task.experiment_kind is not None}
        self.assertNotEqual(by_kind["frontier_replay"], by_kind["td_invariants"])
        changed_context = self.experiment_config(allowed_experiment_kinds=("frontier_replay",),
                                                 max_experiments=1, learner_spec_digest="c" * 64)
        other = ExpeditionScheduler("stable-test-seed", changed_context, NOW)
        replay = next(task for task in other._tasks.values() if task.experiment_kind == "frontier_replay")
        self.assertNotEqual(by_kind["frontier_replay"], replay.strategy_arm_id)

    def test_web_tasks_have_unique_instance_strategy_arms(self):
        scheduler = self.build()
        web = [task for task in scheduler._tasks.values() if task.experiment_kind is None]
        self.assertEqual(len(web), len({task.strategy_arm_id for task in web}))

    def test_experiment_cadence_uses_ranked_eligible_experiment(self):
        class PreferTdInvariants:
            def rank(self, candidates, state_context):
                del state_context
                preferred = [item["task_id"] for item in candidates
                             if item["experiment_kind"] == "td_invariants"]
                return preferred + [item["task_id"] for item in candidates
                                    if item["task_id"] not in preferred]

        config = self.experiment_config(allowed_experiment_kinds=("frontier_replay", "td_invariants"),
                                        max_experiments=2)
        scheduler = ExpeditionScheduler("stable-test-seed", config, NOW, ranker=PreferTdInvariants())
        scheduler._last_experiment_selected_slice = -5
        decision = scheduler.begin_slice(NOW)
        self.assertEqual("td_invariants", decision.task.experiment_kind)
        self.assertEqual("experiment_cadence_due_ranker", decision.selection_reason)

    @staticmethod
    def review_guidance():
        return SeedGuidanceSnapshot(
            seed_ids=("seed_opaque_1",), authority_event_ids=("event_opaque_1",),
            digest="c" * 64,
        )

    def test_guidance_soft_ranks_existing_eligible_web_descriptor_only(self):
        scheduler = self.build()
        first = next(task for task in scheduler._tasks.values()
                     if task.persona == Persona.FORAGER)
        alternate = scheduler._add_task(Persona.FORAGER, 100)
        first.evidence_stance, first.action_mode = "support", "web_search"
        alternate.evidence_stance, alternate.action_mode = "counterevidence", "citation_trace"
        scheduler.set_seed_guidance(self.review_guidance())
        decision = scheduler.begin_slice(NOW)
        self.assertEqual(alternate.task_id, decision.task.task_id)
        self.assertEqual("seed_guidance_request_human_review", decision.selection_reason)
        rendered = repr(scheduler.public_snapshot())
        self.assertIn("seed_guidance_v1", rendered)
        self.assertIn("c" * 64, rendered)
        self.assertNotIn("seed_opaque_1", rendered)
        self.assertNotIn("event_opaque_1", rendered)

    def test_guidance_absence_or_no_preferred_descriptor_preserves_legacy_choice(self):
        ordinary = self.build()
        explicit_none = self.build()
        explicit_none.set_seed_guidance(None)
        self.assertEqual(ordinary.begin_slice(NOW).public_snapshot(),
                         explicit_none.begin_slice(NOW).public_snapshot())

        no_match = self.build()
        no_match.set_seed_guidance(self.review_guidance())
        decision = no_match.begin_slice(NOW)
        self.assertEqual("legacy_persona_rotation", decision.selection_reason)
        self.assertEqual(Persona.FORAGER, decision.persona)

    def test_guidance_cannot_bypass_fairness_or_experiment_cadence(self):
        class PreferCurator:
            def rank(self, candidates, state_context):
                del state_context
                curators = [item["task_id"] for item in candidates if item["persona"] == "curator"]
                return curators + [item["task_id"] for item in candidates if item["task_id"] not in curators]

        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(max_slices=8), NOW,
                                        ranker=PreferCurator())
        scheduler.set_seed_guidance(self.review_guidance())
        for index in range(6):
            task = scheduler._add_task(Persona.CURATOR, index + 100)
            task.evidence_stance, task.action_mode = "limitation", "source_triangulation"
        # Consume six choices.  Slice seven has a due Forager and must remain
        # host-fair even though several cross-persona descriptors are preferred.
        for number in range(6):
            decision = scheduler.begin_slice(NOW + timedelta(seconds=number))
            scheduler.record_outcome(decision.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE,
                                     now=NOW + timedelta(seconds=number))
        fair = scheduler.begin_slice(NOW + timedelta(seconds=6))
        self.assertEqual(Persona.FORAGER, fair.persona)
        self.assertEqual("fairness_persona_due", fair.selection_reason)

        experiments = ExpeditionScheduler("stable-test-seed",
                                           self.experiment_config(
                                               allowed_experiment_kinds=("frontier_replay",),
                                               max_experiments=1), NOW)
        experiments.set_seed_guidance(self.review_guidance())
        for task in experiments._tasks.values():
            if task.experiment_kind is None:
                task.evidence_stance, task.action_mode = "counterevidence", "citation_trace"
        experiments._last_experiment_selected_slice = -5
        due = experiments.begin_slice(NOW)
        self.assertIsNotNone(due.task.experiment_kind)
        self.assertEqual("experiment_cadence_due", due.selection_reason)

    def test_guidance_replay_is_deterministic_and_cannot_change_mid_slice(self):
        left, right = self.build(), self.build()
        for scheduler in (left, right):
            scheduler.set_seed_guidance(self.review_guidance())
            for task in scheduler._tasks.values():
                if task.persona == Persona.FORAGER:
                    task.evidence_stance, task.action_mode = "counterevidence", "citation_trace"
        left_slice, right_slice = left.begin_slice(NOW), right.begin_slice(NOW)
        self.assertEqual(left_slice.public_snapshot(), right_slice.public_snapshot())
        with self.assertRaises(RuntimeError):
            left.set_seed_guidance(None)
        left.record_outcome(left_slice.task.task_id, ExpeditionOutcome.BRANCH_COMPLETE, now=NOW)
        left.set_seed_guidance(None)
        self.assertIsNone(left.seed_guidance_snapshot())

    def test_guidance_never_enters_ranker_input(self):
        class CapturingRanker:
            def __init__(self):
                self.context = None

            def rank(self, candidates, state_context):
                self.context = dict(state_context)
                return [item["task_id"] for item in candidates]

        ranker = CapturingRanker()
        scheduler = ExpeditionScheduler("stable-test-seed", ExpeditionConfig(), NOW, ranker=ranker)
        scheduler.set_seed_guidance(self.review_guidance())
        scheduler.begin_slice(NOW)
        self.assertNotIn("seed_guidance", ranker.context)
        self.assertNotIn("c" * 64, repr(ranker.context))


if __name__ == "__main__":
    unittest.main()
