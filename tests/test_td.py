"""Behavioral tests for bounded TD/RPE preference updates."""

import unittest

from strangeloop.td import (RewardObservation, RewardSource, SAFE_ACTION_CLASSES,
                            SafePreferenceTable, TDConfig, Transition, ValueEstimate,
                            ValueTable)


def transition(identifier, context="ctx", state="s", next_state="n", terminal=False):
    return Transition(identifier, context, state, "respond", next_state, terminal)


def reward(identifier, transition_id, context="ctx", value=1.0, source=RewardSource.USER):
    return RewardObservation(identifier, transition_id, context, value, source, "test-source")


class TDTests(unittest.TestCase):
    def test_delayed_transition_uses_next_state_value_and_terminal_zeros_it(self):
        table = ValueTable(TDConfig(alpha=1.0, gamma=0.5, clip=10.0))
        table.register_transition(transition("first", state="s", next_state="n"))
        table.register_transition(transition("second", state="n", next_state="end", terminal=True))
        table.apply_reward(reward("reward-second", "second", value=1.0))
        update = table.apply_reward(reward("reward-first", "first", value=0.0))
        self.assertEqual(0.5, update.updated_value)
        terminal = ValueTable(TDConfig(alpha=1.0, gamma=0.9, clip=10.0))
        terminal.register_transition(transition("terminal", state="s", next_state="n", terminal=True))
        self.assertEqual(1.0, terminal.apply_reward(reward("reward-terminal", "terminal")).updated_value)

    def test_negative_reward_reverses_preference_value(self):
        table = ValueTable(TDConfig(alpha=1.0, gamma=0.0, clip=10.0))
        table.register_transition(transition("positive", state="preference:respond", terminal=True))
        self.assertEqual(1.0, table.apply_reward(reward("positive-r", "positive", value=1.0)).updated_value)
        table.register_transition(transition("negative", state="preference:respond", terminal=True))
        self.assertEqual(-1.0, table.apply_reward(reward("negative-r", "negative", value=-1.0)).updated_value)

    def test_rejects_nonfinite_and_out_of_range_numbers(self):
        for number in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                TDConfig(alpha=number)
            with self.assertRaises(ValueError):
                ValueEstimate("ctx", "state", number)
            with self.assertRaises(ValueError):
                reward("r", "t", value=number)
        with self.assertRaises(ValueError):
            reward("r", "t", value=1.01)

    def test_clips_prediction_error_before_update(self):
        table = ValueTable(TDConfig(alpha=0.5, gamma=0.0, clip=0.2))
        table.register_transition(transition("clip", terminal=True))
        update = table.apply_reward(reward("clip-r", "clip", value=1.0))
        self.assertEqual(1.0, update.raw_delta)
        self.assertEqual(0.2, update.clipped_delta)
        self.assertEqual(0.1, update.updated_value)

    def test_replay_is_deterministic(self):
        transitions = (transition("a", state="s", next_state="n"),
                       transition("b", state="n", next_state="end", terminal=True))
        rewards = (reward("rb", "b", value=0.4), reward("ra", "a", value=0.1))
        config = TDConfig(alpha=.3, gamma=.8, clip=.9)
        first = ValueTable.replay(transitions, rewards, config)
        second = ValueTable.replay(transitions, rewards, config)
        self.assertEqual(first.export(), second.export())

    def test_rejects_reward_spoofing_duplicate_and_cross_context(self):
        table = ValueTable()
        table.register_transition(transition("one"))
        with self.assertRaises(ValueError):
            reward("model", "one", source="model")
        with self.assertRaises(ValueError):
            reward("system", "one", source="system")
        with self.assertRaises(ValueError):
            reward("self", "one", source="self")
        table.apply_reward(reward("once", "one"))
        with self.assertRaises(ValueError):
            table.apply_reward(reward("once", "one"))
        with self.assertRaises(ValueError):
            table.apply_reward(reward("other-context", "one", context="other"))

    def test_ranking_is_only_a_safe_action_class_preference(self):
        table = SafePreferenceTable()
        ranking = table.rank_action_classes("ctx", ("wait", "respond"))
        self.assertEqual(("respond", "wait"), tuple(item.action_class for item in ranking))
        self.assertNotIn("authorization", table.export())
        self.assertFalse(hasattr(ranking[0], "authorized"))
        with self.assertRaises(ValueError):
            table.rank_action_classes("ctx", ("delete_all_files",))
        self.assertIn("respond", SAFE_ACTION_CLASSES)

    def test_bounds_transition_and_event_storage(self):
        table = ValueTable(max_entries=2, max_events=1)
        table.register_transition(transition("one"))
        with self.assertRaises(ValueError):
            table.apply_reward(reward("r", "one"))
        table = ValueTable(max_entries=2)
        table.register_transition(transition("one"))
        with self.assertRaises(ValueError):
            table.register_transition(transition("two", state="new", next_state="other"))

    def test_transaction_rolls_back_transition_capacity_and_event_log(self):
        table = ValueTable(max_entries=2, max_events=1)
        before = table.export()
        with self.assertRaises(RuntimeError):
            with table.transaction():
                table.register_transition(transition("retryable"))
                raise RuntimeError("downstream resource failed")
        self.assertEqual(before, table.export())

        # Both exact boundaries were reclaimed, including transition ID.
        table.register_transition(transition("retryable"))
        self.assertEqual(2, len(table.export()["estimates"]))
        self.assertEqual(1, len(table.event_log()))

    def test_transaction_rolls_back_reward_value_id_and_event_for_retry(self):
        table = ValueTable(TDConfig(alpha=1.0, gamma=0.0, clip=1.0),
                           max_entries=2, max_events=2)
        table.register_transition(transition("rewardable", terminal=True))
        before = table.export()
        observation = reward("same-reward-id", "rewardable", value=0.75)
        with self.assertRaises(RuntimeError):
            with table.transaction():
                table.apply_reward(observation)
                raise RuntimeError("event-store commit failed")
        self.assertEqual(before, table.export())
        self.assertEqual(0.0, table.estimate("ctx", "s").value)

        # The reward ID, value mutation, and exact event slot all rolled back.
        update = table.apply_reward(observation)
        self.assertEqual(0.75, update.updated_value)
        self.assertEqual(2, len(table.event_log()))

    def test_nested_transaction_restores_each_snapshot(self):
        table = ValueTable(max_entries=4, max_events=2)
        with table.transaction():
            table.register_transition(transition("outer"))
            with self.assertRaises(RuntimeError):
                with table.transaction():
                    table.register_transition(transition(
                        "inner", state="inner-s", next_state="inner-n"))
                    raise RuntimeError("inner failure")
            self.assertEqual(("outer",), tuple(
                event["transition"]["transition_id"] for event in table.event_log()))

        # The caught inner failure released its capacity and ID, while the
        # successful outer transaction remained committed.
        table.register_transition(transition(
            "inner", state="inner-s", next_state="inner-n"))
        self.assertEqual(2, len(table.event_log()))


if __name__ == "__main__":
    unittest.main()
