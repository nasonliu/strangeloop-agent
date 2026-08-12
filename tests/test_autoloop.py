import unittest

from strangeloop.autoloop import (
    CycleResult, FunctionalLoopController, LoopConfig, LoopState, Phase,
    StopReason, TickTrigger,
)


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class AutoLoopTests(unittest.TestCase):
    def test_restart_default_is_paused_and_resume_pause_are_idempotent(self):
        controller = FunctionalLoopController(lambda context: CycleResult())
        self.assertEqual(LoopState.PAUSED, controller.state)
        self.assertEqual(StopReason.PAUSED, controller.stop_reason)
        self.assertTrue(controller.resume())
        self.assertTrue(controller.resume())
        self.assertTrue(controller.pause())
        self.assertFalse(controller.pause())
        self.assertEqual(LoopState.PAUSED, controller.state)

    def test_step_is_bounded_and_returns_only_structured_public_record(self):
        seen = []
        def cycle(context):
            seen.append(context)
            return CycleResult(event_count=context.event_budget, made_progress=True,
                               payload={"outcome": "checked", "count": 1})
        controller = FunctionalLoopController(cycle, LoopConfig(max_events_per_tick=2))
        controller.resume()
        record = controller.step()
        self.assertEqual(1, record.tick_number)
        self.assertEqual(2, record.event_count)
        self.assertTrue(record.made_progress)
        self.assertEqual({"outcome": "checked", "count": 1}, record.payload)
        self.assertEqual(2, seen[0].event_budget)
        self.assertIn(Phase.CYCLE, record.phases)

    def test_salience_preempts_scheduled_work_and_survives_interval_gate(self):
        contexts = []
        clock = FakeClock()
        controller = FunctionalLoopController(
            lambda context: contexts.append(context) or CycleResult(made_progress=True),
            LoopConfig(min_interval=10), clock,
        )
        controller.resume()
        controller.step(TickTrigger.SCHEDULED)
        idle = controller.step(TickTrigger.SCHEDULED)
        self.assertEqual(LoopState.IDLE, idle.state_after)
        self.assertEqual(StopReason.MIN_INTERVAL, idle.stop_reason)
        self.assertTrue(controller.signal_salience({"channel": "audio", "level": 2}))
        record = controller.step(TickTrigger.SCHEDULED)
        self.assertEqual(TickTrigger.SALIENCE, record.trigger)
        self.assertTrue(record.salience_preempted)
        self.assertEqual({"channel": "audio", "level": 2}, contexts[-1].salience)

    def test_tick_and_wall_budgets_are_exact(self):
        clock = FakeClock()
        controller = FunctionalLoopController(
            lambda context: CycleResult(made_progress=True),
            LoopConfig(max_ticks=2, max_wall_seconds=10), clock,
        )
        controller.resume()
        records = controller.run_until_stopped()
        self.assertEqual(2, len(records))
        self.assertEqual(2, controller.ticks_completed)
        self.assertEqual(LoopState.EXHAUSTED, controller.state)
        self.assertEqual(StopReason.MAX_TICKS, controller.stop_reason)

        wall = FunctionalLoopController(
            lambda context: CycleResult(made_progress=True),
            LoopConfig(max_ticks=3, max_wall_seconds=1), clock,
        )
        wall.resume()
        clock.advance(1)
        record = wall.step()
        self.assertEqual(0, wall.ticks_completed)
        self.assertEqual(StopReason.MAX_WALL_SECONDS, record.stop_reason)

    def test_event_budget_violation_and_no_progress_fail_closed(self):
        violating = FunctionalLoopController(
            lambda context: CycleResult(event_count=context.event_budget + 1),
            LoopConfig(max_events_per_tick=1),
        )
        violating.resume()
        record = violating.step()
        self.assertEqual(LoopState.EXHAUSTED, record.state_after)
        self.assertEqual(StopReason.MAX_EVENTS_PER_TICK, record.stop_reason)

        idle = FunctionalLoopController(
            lambda context: CycleResult(), LoopConfig(max_no_progress=2)
        )
        idle.resume()
        idle.step()
        record = idle.step()
        self.assertEqual(LoopState.EXHAUSTED, record.state_after)
        self.assertEqual(StopReason.MAX_NO_PROGRESS, record.stop_reason)

    def test_stop_and_terminal_states_are_idempotent(self):
        controller = FunctionalLoopController(lambda context: CycleResult())
        self.assertTrue(controller.stop())
        self.assertFalse(controller.stop())
        self.assertFalse(controller.resume())
        record = controller.step()
        self.assertEqual(LoopState.STOPPED, record.state_after)
        self.assertEqual(StopReason.EXTERNAL_STOP, record.stop_reason)

    def test_callback_error_stops_without_payload_or_retry(self):
        def boom(context):
            raise RuntimeError("private failure")
        controller = FunctionalLoopController(boom)
        controller.resume()
        record = controller.step()
        self.assertEqual(LoopState.STOPPED, record.state_after)
        self.assertEqual(StopReason.CALLBACK_ERROR, record.stop_reason)
        self.assertEqual({}, record.payload)
        self.assertEqual((Phase.DISPATCH, Phase.CYCLE, Phase.FAILED), record.phases)
        self.assertFalse(controller.resume())

    def test_callback_contract_and_payload_are_strict(self):
        controller = FunctionalLoopController(lambda context: {"not": "a result"})
        controller.resume()
        self.assertEqual(StopReason.CALLBACK_ERROR, controller.step().stop_reason)
        with self.assertRaises(ValueError):
            CycleResult(payload={"bad": object()})
        with self.assertRaises(ValueError):
            FunctionalLoopController(lambda context: CycleResult()).signal_salience({"x": object()})

    def test_config_rejects_boolean_nonfinite_and_zero_budgets(self):
        invalid = (
            {"max_wall_seconds": True},
            {"max_wall_seconds": float("nan")},
            {"max_wall_seconds": float("inf")},
            {"max_wall_seconds": -0.1},
            {"min_interval": False},
            {"min_interval": float("nan")},
            {"min_interval": -1},
            {"max_events_per_tick": 0},
            {"max_events_per_tick": True},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LoopConfig(**kwargs)
        self.assertEqual(1, LoopConfig(max_events_per_tick=1).max_events_per_tick)

    def test_payload_rejects_nonfinite_and_private_reasoning_keys_recursively(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CycleResult(payload={"score": value})
        for key in ("chain_of_thought", "HIDDEN_REASONING", "Private_Reasoning", "scratchpad"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    CycleResult(payload={"public": {key: "do not retain"}})


if __name__ == "__main__":
    unittest.main()
