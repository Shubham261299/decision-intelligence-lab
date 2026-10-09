"""Behavioral decision contract, independent of providers and evaluation."""

from typing import Protocol, runtime_checkable

from decisionlab.models import DecisionProblem, Recommendation


@runtime_checkable
class DecisionModel(Protocol):
    """Recommend from supplied information without labels or operational actions.

    A decision model can be deterministic or use an LLM. Its identity is distinct
    from a provider's model ID. The execution boundary validates the returned
    recommendation and records failures; policy and verification remain external.
    """

    def decide(self, problem: DecisionProblem) -> Recommendation:
        """Return an original recommendation; invalid output must not be repaired."""
        ...
