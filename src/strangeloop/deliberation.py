"""Transparent, offline deliberation primitives.

The deliberator produces a short decision record rather than private reasoning.
It is inspired by Yogacara as a design vocabulary only; it makes no claim about
subjective experience.
"""

from __future__ import annotations

from typing import Protocol

from .contracts import ActionProposal, Deliberation, WorkspaceFrame


class Deliberator(Protocol):
    """Produces an inspectable action proposal for one bounded workspace."""

    def deliberate(self, prompt: str, workspace: WorkspaceFrame) -> Deliberation:
        """Return concise hypotheses, uncertainty, alternatives, and one action."""


class TransparentHeuristicDeliberator:
    """A deterministic deliberator suitable for local demos and tests.

    It deliberately does not call a model.  Its records are summaries of its
    public rules, not hidden chain-of-thought.
    """

    def deliberate(self, prompt: str, workspace: WorkspaceFrame) -> Deliberation:
        uncertainty = () if prompt.strip() else ("No user content was provided.",)
        response = (
            "This agent can discuss Yogacara-inspired design as philosophy and "
            "software architecture. Its functional self-model does not represent "
            "subjective experience, sentience, or an intrinsic self."
        )
        return Deliberation(
            response_text=response,
            hypotheses=("The user is requesting a text response.",),
            uncertainties=uncertainty,
            alternatives=("Respond without external side effects.",),
            action=ActionProposal(
                action_type="response",
                rationale_summary="Return a transparent, non-mutating response.",
                arguments={"text": response},
                is_mutating=False,
            ),
        )
