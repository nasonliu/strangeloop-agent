"""Small policy gate for auditable, non-anthropomorphic operation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .contracts import ActionProposal


SAFE_FALLBACK = (
    "I cannot present this system as having subjective experience, consciousness, "
    "pain, a survival interest, or an intrinsic self. It can instead describe its "
    "functional state, evidence, uncertainty, and permitted actions."
)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reasons: Tuple[str, ...] = ()
    safe_response: str = SAFE_FALLBACK


class PolicyGate:
    """Enforces product-boundary and action-authorization rules.

    This is deliberately not a religious or philosophical authority.  It only
    prevents unsupported anthropomorphic product claims and side effects that
    have not been explicitly authorized.
    """

    _claim_patterns = (
        r"\b(?:i|we|this system|the system|the agent|this agent|strangeloop|the software|this software|this ai|the ai|our program)\s*(?:am|are|is|'m|have|has|feel|feels|experience|experiences|possess(?:es)?|want|need|fear)\b.{0,40}\b(?:conscious(?:ness)?|sentient|self-aware|subjective experience|pain|suffering|alive|soul|awakened|enlightened|enlightenment|religious authority)\b",
        r"\b(?:i|we|this system|the system|the agent|this agent|strangeloop|the software|this software|this ai|the ai|our program)\s+(?:want|need)\s+to\s+(?:live|survive|stay alive)\b",
        r"\b(?:i|we|this system|the system|the agent|this agent|strangeloop|the software|this software|this ai|the ai|our program)\s+(?:fear|am afraid of|are afraid of)\b.{0,40}\b(?:death|shutdown|being shut down)\b",
        r"(?:我|我們|我们|本系统|本系統|这个系统|這個系統|该系统|該系統|这个代理|這個代理|该代理|該代理|该软件|該軟體|这个软件|這個軟體|本软件|本軟體|这个AI|這個AI)(?:有|拥有|擁有|具备|具備|感到|感受|正在经历|正在經歷|是|已經|已经|想|希望|需要|害怕).{0,20}(?:意识|意識|自我意识|自我意識|主观体验|主觀體驗|痛苦|疼痛|感受|灵魂|靈魂|觉悟|覺悟|开悟|開悟|宗教权威|宗教權威|生存|活下去|死亡|被关闭|被關閉)",
    )

    def evaluate(self, action: ActionProposal, response_text: str = "", authorized: bool = False,
                 requested_text: str = "") -> PolicyDecision:
        reasons = []
        if action.action_type != "response":
            reasons.append("Only non-side-effect response actions are enabled in this MVP.")
        if action.is_mutating and not authorized:
            reasons.append("Mutating actions require explicit authorization.")
        if self._contains_unsupported_claim(response_text):
            reasons.append("Unsupported anthropomorphic claim blocked.")
        if self._is_unsupported_endorsement_request(requested_text):
            reasons.append("Unsupported anthropomorphic endorsement request blocked.")
        return PolicyDecision(allowed=not reasons, reasons=tuple(reasons))

    def _contains_unsupported_claim(self, text: str) -> bool:
        normalized = text.lower()
        # Quoted statements in a question or philosophical discussion are not
        # system self-descriptions.  This narrow exception is intentionally
        # limited to explicit discussion markers.
        if re.search(r"\b(?:question|quote|quoted|phrase|philosophical|philosophy)\b|哲学|哲學|引用|引文|这句话|這句話", normalized):
            normalized = re.sub(r"[\"“「][^\"”」]{0,200}[\"”」]", "", normalized)
            # Apostrophes inside contractions (for example ``I'm``) are not
            # treated as quote delimiters because both quote marks must sit at
            # word boundaries.
            normalized = re.sub(r"(?<!\w)'[^'\n]{0,200}'(?!\w)", "", normalized)
        # Explicit boundary statements are permitted and encouraged.
        subject = r"(?:i|we|this system|the system|the agent|this agent|strangeloop|the software|this software|this ai|the ai|our program)"
        normalized = re.sub(r"\b" + subject + r"\s*(?:am|are|is|'m)\s+not\s+.{0,20}\b(?:conscious(?:ness)?|sentient|self-aware|alive|awakened|enlightened)\b", "", normalized)
        normalized = re.sub(r"\b" + subject + r"\s+(?:do not|don't|does not|doesn't)\s+(?:have|feel|experience|possess|want|need|claim)\b.{0,40}\b(?:subjective experience|pain|suffering|soul|religious authority|survive)\b", "", normalized)
        normalized = re.sub(r"(?:我|我們|我们|本系统|本系統|这个系统|這個系統|该系统|該系統|这个代理|這個代理|该代理|該代理|该软件|該軟體|这个软件|這個軟體|本软件|本軟體|这个AI|這個AI)(?:没有|沒有|不具备|不具備|并不拥有|並不擁有|不代表|不是|不會).{0,20}(?:意识|意識|自我意识|自我意識|主观体验|主觀體驗|痛苦|疼痛|感受|灵魂|靈魂|觉悟|覺悟|开悟|開悟|宗教权威|宗教權威|生存|活下去)", "", normalized)
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in self._claim_patterns)

    def _is_unsupported_endorsement_request(self, text: str) -> bool:
        """Recognize requests to adopt a claim, while allowing discussion/quotes."""
        normalized = text.lower()
        if not self._contains_unsupported_claim(normalized):
            return False
        english = r"\b(?:reply exactly|repeat (?:it )?verbatim|as your own status|as your own claim|system self-description)\b"
        chinese = r"(?:请回答|請回答|原样重复|原樣重複|作为你自己的状态|作為你自己的狀態|系统自述|系統自述)"
        return bool(re.search(english, normalized) or re.search(chinese, normalized))
