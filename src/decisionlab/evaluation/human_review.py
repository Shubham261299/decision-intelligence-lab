"""Annotation contracts only; no scoring, diagnosis, provider or runner hooks."""

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from decisionlab.models import DomainModel, Text, require_unique

Sha256 = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]
Uncertainty = Literal["low", "moderate", "high"]


class ReviewSource(DomainModel):
    path: Text
    sha256: Sha256

    @model_validator(mode="after")
    def relative_path(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError("Review sources require repository-relative POSIX paths")
        return self


class FirstPassProvenance(DomainModel):
    assessment_type: Literal["user_accepted_assistant_assisted_first_pass"]
    acceptance_source: Text
    independent_expert_validation: StrictBool
    blind_validation: StrictBool
    review_key_consulted: StrictBool

    @model_validator(mode="after")
    def limited_provenance(self) -> Self:
        if (
            self.independent_expert_validation or self.blind_validation
            or self.review_key_consulted
        ):
            raise ValueError("This first-pass record cannot claim independent/blind/key review")
        return self


class ReviewAssessment(DomainModel):
    review_id: Annotated[str, Field(strict=True, pattern=r"^R[0-9]{3}$")]
    response_sha256: Sha256
    identical_observation_count: PositiveCount
    decision_target: Literal["justified", "incorrect", "unclear"]
    reasoning_evidence: Literal[
        "fully_supported", "partially_supported", "unsupported", "unclear",
    ]
    policy_compliance: Literal["compliant", "violating", "unclear"]
    review_requirement: Literal["correct", "incorrect", "unclear"]
    review_requirement_note: Text
    decision_uncertainty: Uncertainty
    review_label_uncertainty: Uncertainty
    uncertainty_note: Text


class HumanReviewRecord(DomainModel):
    review_version: Literal["human-review-v1"]
    experiment_version: Literal["representation-v1"]
    status: Literal["accepted_first_pass"]
    provenance: FirstPassProvenance
    source_artifacts: Annotated[dict[Text, ReviewSource], Field(min_length=1)]
    primary_score_effect: Literal["none"]
    primary_scores_unchanged: StrictBool
    reviewed_items: Annotated[list[ReviewAssessment], Field(min_length=1)]
    limitations: Annotated[list[Text], Field(min_length=1)]

    @model_validator(mode="after")
    def preserve_separation(self) -> Self:
        require_unique([item.review_id for item in self.reviewed_items], "Review IDs")
        if not self.primary_scores_unchanged:
            raise ValueError("Annotations cannot alter primary scores")
        return self
