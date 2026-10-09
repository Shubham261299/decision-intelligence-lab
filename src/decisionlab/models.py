"""Data contracts only: these models do not select or execute decisions."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

Text = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
Scalar = StrictStr | StrictBool | StrictInt | StrictFloat
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class DomainModel(BaseModel):
    """Reject unknown fields and non-finite numbers at every boundary."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def require_known(values: list[str], known: set[str], label: str) -> None:
    missing = set(values) - known
    if missing:
        raise ValueError(f"Unknown {label}: {sorted(missing)}")


class Fact(DomainModel):
    id: Text
    statement: Text
    value: Scalar | None = None
    source: Text


class Constraint(DomainModel):
    id: Text
    description: Text
    kind: Literal["hard", "soft"]


class Option(DomainModel):
    id: Text
    description: Text
    attributes: dict[Text, Scalar] = Field(default_factory=dict)


class Unknown(DomainModel):
    description: Text
    affected_option_ids: list[Text] = Field(default_factory=list)
    material_to_decision: bool


class OptionAssessment(DomainModel):
    option_id: Text
    eligibility: Literal["eligible", "ineligible", "unknown"]
    reasons: Annotated[list[Text], Field(min_length=1)]
    evidence_ids: list[Text] = Field(default_factory=list)
    constraint_ids: list[Text] = Field(default_factory=list)


class Recommendation(DomainModel):
    status: Literal["selected", "abstained"]
    selected_option_id: Text | None = None
    assessments: list[OptionAssessment] = Field(default_factory=list)
    unknowns: list[Unknown] = Field(default_factory=list)
    confidence: Probability | None = None
    explanation: Text
    requested_review: bool = False

    @model_validator(mode="after")
    def check_selection(self) -> Self:
        if (self.status == "selected") != (self.selected_option_id is not None):
            raise ValueError("Only selected recommendations require an option ID")
        require_unique([item.option_id for item in self.assessments], "Assessments")
        return self


class DecisionProblem(DomainModel):
    id: Text
    description: Text
    context: str = ""
    facts: list[Fact] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    options: Annotated[list[Option], Field(min_length=1)]

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        # IDs are namespaced by entity type, matching the typed reference fields.
        for label, ids in (
            ("Fact IDs", [item.id for item in self.facts]),
            ("Constraint IDs", [item.id for item in self.constraints]),
            ("Option IDs", [item.id for item in self.options]),
        ):
            require_unique(ids, label)
        return self

    def validate_recommendation(self, recommendation: Recommendation) -> None:
        """Check references against this input, without judging decision quality."""
        options = {item.id for item in self.options}
        facts = {item.id for item in self.facts}
        constraints = {item.id for item in self.constraints}
        if recommendation.selected_option_id is not None:
            require_known([recommendation.selected_option_id], options, "option IDs")
        for assessment in recommendation.assessments:
            require_known([assessment.option_id], options, "assessment option IDs")
            require_known(assessment.evidence_ids, facts, "evidence IDs")
            require_known(assessment.constraint_ids, constraints, "constraint IDs")
        for unknown in recommendation.unknowns:
            require_known(unknown.affected_option_ids, options, "affected option IDs")


class PolicyResult(DomainModel):
    requires_human_review: bool
    review_reasons: list[Text] = Field(default_factory=list)
    detected_violations: list[Text] = Field(default_factory=list)
    policy_version: Text

    @model_validator(mode="after")
    def check_review(self) -> Self:
        if self.requires_human_review and not self.review_reasons:
            raise ValueError("Required review needs at least one reason")
        if not self.requires_human_review and self.review_reasons:
            raise ValueError("Review reasons require human review")
        return self


class RunMetadata(DomainModel):
    strategy: Text
    strategy_version: Text
    schema_version: Text = "1"
    model_id: Text | None = None
    provider: Text | None = None
    provider_response_id: Text | None = None
    prompt_version: Text | None = None
    dataset_version: Text | None = None
    generation_settings: dict[Text, Scalar] = Field(default_factory=dict)
    retry_count: Annotated[int, Field(strict=True, ge=0)] = 0


class Usage(DomainModel):
    input_tokens: Annotated[int, Field(strict=True, ge=0)] | None = None
    output_tokens: Annotated[int, Field(strict=True, ge=0)] | None = None
    cost_usd: NonNegative | None = None


class RunError(DomainModel):
    code: Text
    message: Text


class ExecutionInfo(DomainModel):
    """Optional execution telemetry, outside the decision strategy interface."""

    metadata: RunMetadata
    usage: Usage | None = None
    latency_ms: NonNegative | None = None
    response_metadata: dict[str, Scalar] = Field(default_factory=dict)


class DecisionRun(DomainModel):
    id: Text
    problem_id: Text
    status: Literal["success", "failure"]
    metadata: RunMetadata
    recommendation: Recommendation | None = None
    policy_result: PolicyResult | None = None
    latency_ms: NonNegative | None = None
    usage: Usage | None = None
    response_metadata: dict[str, Scalar] = Field(default_factory=dict)
    error: RunError | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if self.status == "success":
            if self.recommendation is None or self.policy_result is None:
                raise ValueError("Successful runs require recommendation and policy")
            if self.error is not None:
                raise ValueError("Successful runs cannot contain an error")
        elif (
            self.error is None
            or self.recommendation is not None
            or self.policy_result is not None
        ):
            raise ValueError("Failed runs require an error and no accepted outcome")
        return self
