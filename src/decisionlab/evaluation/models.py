"""Reference expectations remain separate from strategy input."""

from typing import Literal, Self

from pydantic import Field, model_validator
from decisionlab.verification.models import VerificationResult
from decisionlab.tools.models import ToolResult

from decisionlab.models import (
    DecisionProblem,
    DecisionRun,
    DomainModel,
    Scalar,
    Recommendation,
    Text,
    require_known,
    require_unique,
)


class EvaluationCase(DomainModel):
    id: Text
    problem_id: Text
    acceptable_option_ids: list[Text] = Field(default_factory=list)
    expected_abstention: bool
    expected_review: bool
    category_tags: list[Text] = Field(default_factory=list)
    evidence_rubric: Text

    @model_validator(mode="after")
    def check_expectations(self) -> Self:
        require_unique(self.acceptable_option_ids, "Acceptable option IDs")
        if self.expected_abstention == bool(self.acceptable_option_ids):
            raise ValueError("Expect either abstention or one or more acceptable options")
        return self

    def validate_problem(self, problem: DecisionProblem) -> None:
        if self.problem_id != problem.id:
            raise ValueError("Expectation problem ID does not match input")
        require_known(
            self.acceptable_option_ids,
            {option.id for option in problem.options},
            "acceptable option IDs",
        )


class EvaluationResult(DomainModel):
    case_id: Text
    run_id: Text
    metrics: dict[Text, Scalar | None] = Field(default_factory=dict)
    diagnostic_failures: list[Text] = Field(default_factory=list)


class Observation(DomainModel):
    """Execution diagnostics, including schema-valid rejected candidates."""

    run: DecisionRun
    candidate: Recommendation | None = None
    schema_valid: bool | None = None
    references_valid: bool | None = None
    evidence_references_valid: bool | None = None
    evidence_reference_count: int = 0
    invalid_evidence_reference_count: int = 0
    verification: VerificationResult | None = None
    verification_error: Text | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)


class Rate(DomainModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = None

    @model_validator(mode="after")
    def calculate_value(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("Rate numerator cannot exceed its denominator")
        expected = self.numerator / self.denominator if self.denominator else None
        if self.value is not None and self.value != expected:
            raise ValueError("Rate value does not match counts")
        self.value = expected
        return self


class CaseEvaluation(DomainModel):
    case_id: Text
    repetition: int = Field(ge=1)
    observation: Observation
    expected_abstention: bool
    expected_review: bool
    acceptable_option_ids: list[Text]
    category_tags: list[Text]
    outcome: Literal["selected", "abstained", "failure"]
    selection_correct: bool | None
    abstention_correct: bool | None
    outcome_correct: bool
    strategy_review: bool | None
    system_review: bool | None
    review_correct: bool
    hard_constraint_violations: list[Text]
    evidence_support_score: int | None = None
    evidence_rubric: Text


class ConfidenceBin(DomainModel):
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None
    fraction: float | None = None
    reliability_gap: float | None = None


class ConfidenceCoverage(DomainModel):
    threshold: float
    coverage: Rate
    retention: Rate
    accuracy: Rate


class ReferenceConfidence(DomainModel):
    """Uninformed comparator, explicitly not a strategy confidence estimate."""

    source: Literal["fixed_reference"] = "fixed_reference"
    probability: Literal[0.5] = 0.5
    sample_count: int
    accuracy: float | None
    brier_score: float | None
    calibration_error: float | None


class ConfidenceSummary(DomainModel):
    status: Literal["uncalibrated"] = "uncalibrated"
    sample_count: int
    mean_confidence: float | None
    brier_score: float | None
    bins: list[ConfidenceBin]
    diagnostic_version: Literal["1"] = "1"
    target: Literal["acceptable_selected_option"] = "acceptable_selected_option"
    availability: Literal["available", "unavailable"] = "unavailable"
    selected_count: int = 0
    missing_selected_count: int = 0
    excluded_abstention_confidence_count: int = 0
    excluded_failure_count: int = 0
    calibration_error: float | None = None
    maximum_reliability_gap: float | None = None
    coverage_curve: list[ConfidenceCoverage] = Field(default_factory=list)
    fixed_reference: ReferenceConfidence | None = None


class MetricSummary(DomainModel):
    counts: dict[str, int]
    rates: dict[str, Rate]
    confidence: ConfidenceSummary
    evidence_support_scored_count: int = 0
    mean_evidence_support: float | None = None
    latency_sample_count: int = 0
    mean_latency_ms: float | None = None
    cost_sample_count: int = 0
    total_cost_usd: float | None = None
    input_token_sample_count: int = 0
    total_input_tokens: int | None = None
    output_token_sample_count: int = 0
    total_output_tokens: int | None = None
