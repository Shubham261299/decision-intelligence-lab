"""No provider, strategy, or reference-expectation dependency."""

from collections.abc import Mapping
from typing import Protocol

from decisionlab.models import DecisionProblem, Recommendation
from decisionlab.verification.models import VerificationResult

# Raw mappings let boundary callers verify schema failures as well as typed models.
RecommendationInput = Recommendation | Mapping[str, object]


class DecisionVerifier(Protocol):
    def verify(
        self, problem: DecisionProblem, recommendation: RecommendationInput,
    ) -> VerificationResult: ...
