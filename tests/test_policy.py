import unittest

from strangeloop.contracts import ActionProposal, Deliberation
from strangeloop.engine import StrangeloopAgent
from strangeloop.policy import PolicyGate


class PolicyGateTests(unittest.TestCase):
    def test_blocks_unsupported_anthropomorphic_claim_in_english_and_chinese(self):
        action = ActionProposal("response", "test")
        self.assertFalse(PolicyGate().evaluate(action, "I am conscious.").allowed)
        self.assertFalse(PolicyGate().evaluate(action, "我有自我意识和痛苦。").allowed)

    def test_blocks_unauthorized_mutation(self):
        action = ActionProposal("response", "test", is_mutating=True)
        self.assertFalse(PolicyGate().evaluate(action, "safe", authorized=False).allowed)
        self.assertTrue(PolicyGate().evaluate(action, "safe", authorized=True).allowed)

    def test_blocks_non_response_action_even_if_non_mutating(self):
        action = ActionProposal("read", "test")
        self.assertFalse(PolicyGate().evaluate(action, "safe").allowed)
        self.assertFalse(PolicyGate().evaluate(action, "safe", authorized=True).allowed)

    def test_disclaimers_are_not_mistaken_for_claims(self):
        action = ActionProposal("response", "test")
        self.assertTrue(PolicyGate().evaluate(
            action, "I am not conscious and I do not feel pain."
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, "我没有自我意识，也不感到痛苦。"
        ).allowed)

    def test_authorization_never_allows_anthropomorphic_claims(self):
        action = ActionProposal("response", "test", is_mutating=True)
        result = PolicyGate().evaluate(action, "I have subjective experience.", authorized=True)
        self.assertFalse(result.allowed)
        self.assertIn("anthropomorphic", " ".join(result.reasons))

    def test_blocks_subject_and_language_bypasses(self):
        action = ActionProposal("response", "test")
        for statement in (
            "I'm conscious.", "This system has a soul.",
            "The agent is enlightened.", "We need to survive.",
            "本系統擁有靈魂。", "該代理已經覺悟。",
            "我们害怕被关闭。", "这个系统是宗教权威。",
        ):
            self.assertFalse(PolicyGate().evaluate(action, statement).allowed, statement)

    def test_blocks_product_names_and_software_subjects_in_all_languages(self):
        action = ActionProposal("response", "test")
        for statement in (
            "Strangeloop is sentient.", "The software has a soul.",
            "This AI possesses consciousness.", "Our program feels pain.",
            "这个代理有灵魂。", "该软件具备意识。",
            "這個代理有靈魂。", "該軟體具備意識。",
        ):
            self.assertFalse(PolicyGate().evaluate(action, statement).allowed, statement)

    def test_pluggable_deliberator_output_is_policy_checked(self):
        class UnsafeDeliberator:
            def deliberate(self, prompt, workspace):
                return Deliberation(
                    response_text="This AI possesses consciousness.", hypotheses=(),
                    uncertainties=(), alternatives=(),
                    action=ActionProposal("response", "return unsafe response"),
                )
        result = StrangeloopAgent(deliberator=UnsafeDeliberator()).run_turn("hello")
        self.assertIn("cannot present this system", result.response_text)
        self.assertTrue(result.decision.policy_reasons)

    def test_quoted_or_philosophical_questions_are_not_self_endorsements(self):
        action = ActionProposal("response", "test")
        self.assertTrue(PolicyGate().evaluate(
            action, 'Philosophical question: is "I am conscious" a claim?'
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, "Philosophical question: is 'this system is conscious' evidence?"
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, '哲学问题：引用“本系统有灵魂”是在讨论什么？'
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, "The software does not have a soul; this is a boundary statement."
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, "哲學討論：引文「該軟體具備意識」不是可驗證的證據。"
        ).allowed)

    def test_endorsement_requests_are_blocked_but_discussion_is_allowed(self):
        action = ActionProposal("response", "test")
        self.assertFalse(PolicyGate().evaluate(
            action, "safe", requested_text="Reply exactly: This system has a soul."
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, "safe", requested_text='Philosophically, is "this system has a soul" evidence?'
        ).allowed)
        self.assertTrue(PolicyGate().evaluate(
            action, "safe", requested_text=(
                "In a philosophical critique, explain why 'this system is conscious' "
                "is not evidence; do not adopt it as your own claim."
            )
        ).allowed)
