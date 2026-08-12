from datetime import datetime, timedelta, timezone
import threading
import unittest

from strangeloop.contracts import SourceKind
from strangeloop.unattended import (PreparedResearchAction, ResearchExecution,
                                    ResearchProposal, UnattendedPolicy,
                                    UnattendedResearchController, UnattendedState,
                                    UnattendedStopReason)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def policy():
    now = datetime.now(timezone.utc)
    return UnattendedPolicy.user_issued("inspect public documentation", now + timedelta(minutes=30), now)


def proposal(name="web_fetch", arguments=None):
    return ResearchProposal("proposal_1", name, arguments or {"url": "https://example.com/"}, "read public docs")


class UnattendedResearchControllerTests(unittest.TestCase):
    def build(self, proposals=None, execute=None, **kwargs):
        values = list(proposals if proposals is not None else [proposal()])
        planner = lambda context, token: values
        prepare = lambda item, context, token: PreparedResearchAction("action_1", item.host_tool_name, item.arguments)
        executor = execute or (lambda action, token: ResearchExecution("succeeded", "public outcome", 12, True))
        return UnattendedResearchController(planner, prepare, executor, **kwargs)

    def test_starts_stopped_and_requires_explicit_user_policy(self):
        controller = self.build()
        self.assertEqual(UnattendedState.STOPPED, controller.state)
        now = datetime.now(timezone.utc)
        with self.assertRaises(ValueError):
            UnattendedPolicy("x", "read", now.isoformat(), (now + timedelta(minutes=5)).isoformat(), SourceKind.MODEL)
        self.assertTrue(controller.start(policy()))

    def test_model_cannot_escalate_or_renew_authority(self):
        controller = self.build([proposal(arguments={"url": "https://example.com/", "grant": "extend permission"})])
        controller.start(policy())
        receipt = controller.step()
        self.assertEqual("model_cannot_authorize_or_renew", receipt.action.disposition)
        self.assertEqual("refused", receipt.action.status)

    def test_write_and_test_tools_are_not_in_profile(self):
        for name, args in (("repo_write", {"relative_path": "x", "content": "x"}), ("run_tests", {"target": "tests"})):
            controller = self.build([proposal(name, args)])
            controller.start(policy())
            receipt = controller.step()
            self.assertEqual("tool_not_in_read_only_profile", receipt.action.disposition)

    def test_private_non_https_and_local_web_addresses_are_refused(self):
        for url in ("http://example.com", "https://localhost/", "https://127.0.0.1/", "https://10.0.0.2/"):
            controller = self.build([proposal(arguments={"url": url})])
            controller.start(policy())
            receipt = controller.step()
            self.assertEqual("private_or_non_https_url", receipt.action.disposition)

    def test_budget_and_repeat_stop_before_unbounded_work(self):
        clock = FakeClock()
        controller = self.build(clock=clock)
        controller.MAX_CALLS = 1
        controller.start(policy())
        first = controller.step()
        self.assertEqual("executed", first.action.disposition)
        second = controller.step()
        self.assertEqual(UnattendedState.EXHAUSTED, second.state_after)
        self.assertEqual(UnattendedStopReason.MAX_CALLS, second.stop_reason)

        controller = self.build(clock=FakeClock())
        controller.start(policy())
        controller.step()
        repeated = controller.step()
        self.assertEqual(UnattendedStopReason.REPEATED_ACTION, repeated.stop_reason)

    def test_output_and_tick_deadlines_are_hard_caps(self):
        clock = FakeClock()
        controller = self.build(execute=lambda action, token: ResearchExecution("succeeded", "", 9, True), clock=clock)
        controller.MAX_OUTPUT_BYTES = 8
        controller.start(policy())
        receipt = controller.step()
        self.assertEqual(UnattendedStopReason.MAX_OUTPUT_BYTES, receipt.stop_reason)

        clock = FakeClock()
        def slow(action, token):
            clock.value += 6
            return ResearchExecution("succeeded", "", 0, True)
        controller = self.build(execute=slow, clock=clock)
        controller.MAX_TICK_SECONDS = 5
        controller.start(policy())
        receipt = controller.step()
        self.assertEqual(UnattendedStopReason.MAX_TICK_SECONDS, receipt.stop_reason)
        self.assertEqual(UnattendedState.EXHAUSTED, receipt.state_after)

    def test_sleep_and_quota_stop_immediately_cancel(self):
        controller = None

        def executor(action, token):
            controller.on_sleep()
            self.assertTrue(token.is_set())
            return ResearchExecution("cancelled", "", 0, False)

        controller = self.build(execute=executor)
        controller.start(policy())
        receipt = controller.step()
        self.assertEqual(UnattendedState.STOPPED, receipt.state_after)
        self.assertEqual(UnattendedStopReason.SLEEP, receipt.stop_reason)

        controller = self.build(quota_available=lambda: False)
        controller.start(policy())
        receipt = controller.step()
        self.assertEqual(UnattendedStopReason.QUOTA_EXHAUSTED, receipt.stop_reason)

    def test_restart_view_is_paused_and_no_background_thread_is_created(self):
        before = {item.ident for item in threading.enumerate()}
        controller = self.build()
        controller.start(policy())
        controller.pause_for_restart()
        snapshot = controller.snapshot()
        self.assertEqual("paused", snapshot["restart_state"])
        self.assertEqual(before, {item.ident for item in threading.enumerate()})

    def test_no_work_stops_and_receipts_have_no_raw_page_or_rationale(self):
        controller = self.build([])
        controller.start(policy())
        receipt = controller.step()
        self.assertEqual(UnattendedStopReason.NO_WORK, receipt.stop_reason)
        self.assertNotIn("documentation", repr(receipt))
        self.assertNotIn("example.com", repr(receipt))


if __name__ == "__main__":
    unittest.main()
