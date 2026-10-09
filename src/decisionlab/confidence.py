"""Provider-independent interpretation; no probability fitting or decision gates."""

from typing import Literal, Self

from pydantic import model_validator

from decisionlab.models import DomainModel, Probability, Recommendation


class ConfidenceEstimate(DomainModel):
    """Probability of acceptable selection, never of review or abstention."""

    contract_version: Literal["1"] = "1"
    target: Literal["acceptable_selected_option"] = "acceptable_selected_option"
    probability: Probability | None = None
    source: Literal["strategy_reported", "unavailable"] = "unavailable"
    calibration: Literal["uncalibrated", "unavailable"] = "unavailable"

    @model_validator(mode="after")
    def consistent_availability(self) -> Self:
        available = self.probability is not None
        if available != (self.source == "strategy_reported") or available != (
            self.calibration == "uncalibrated"
        ):
            raise ValueError("Confidence availability, source and calibration must agree")
        return self


def selection_confidence(recommendation: Recommendation | None) -> ConfidenceEstimate:
    """Abstention confidence has no defined correctness target in contract v1.

    Its original value remains in Recommendation; no estimate is fabricated.
    Callers supply only accepted recommendations, never rejected candidates.
    """
    if recommendation is None or recommendation.status != "selected":
        return ConfidenceEstimate()
    if recommendation.confidence is None:
        return ConfidenceEstimate()
    return ConfidenceEstimate(
        probability=recommendation.confidence,
        source="strategy_reported", calibration="uncalibrated",
    )
