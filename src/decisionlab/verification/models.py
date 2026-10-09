"""Typed verification findings, separate from recommendations and ground truth."""

from typing import Literal, Self

from pydantic import Field, model_validator

from decisionlab.models import DomainModel, Text

FindingKind = Literal[
    "invalid_decision", "policy_violation", "insufficient_evidence", "human_review",
]


class VerificationFinding(DomainModel):
    code: Text
    kind: FindingKind
    path: Text | None = None
    option_id: Text | None = None
    evidence_ids: list[Text] = Field(default_factory=list)
    constraint_ids: list[Text] = Field(default_factory=list)


class VerificationResult(DomainModel):
    result_version: Literal["1"] = "1"
    verifier: Text
    verifier_version: Text
    policy_version: Text
    problem_id: Text
    disposition: Literal["accept", "reject", "human_review"]
    decision_validity: Literal["valid", "invalid", "not_assessed"] = "not_assessed"
    structurally_valid: bool
    selected_option_valid: bool | None = None
    references_valid: bool | None = None
    evidence_references_valid: bool | None = None
    evidence_sufficiency: Literal[
        "sufficient", "insufficient", "not_assessed", "not_applicable",
    ] = "not_assessed"
    evidence_support: Literal["not_assessed"] = "not_assessed"
    requires_human_review: bool = False
    policy_violations: list[Text] = Field(default_factory=list)
    findings: list[VerificationFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        invalid = (
            not self.structurally_valid or self.references_valid is False
            or self.evidence_references_valid is False
            or self.selected_option_valid is False or bool(self.policy_violations)
            or any(f.kind == "invalid_decision" for f in self.findings)
        )
        self.decision_validity = (
            "invalid" if invalid else "valid" if self.references_valid is True
            else "not_assessed"
        )
        if invalid and self.disposition != "reject":
            raise ValueError("Invalid decisions must be rejected")
        if self.disposition == "human_review" and not self.requires_human_review:
            raise ValueError("Review disposition requires a review flag")
        if self.disposition != "accept" and not self.findings:
            raise ValueError("Rejection and review require machine-readable findings")
        if self.disposition == "accept" and (
            self.decision_validity != "valid" or self.requires_human_review
            or self.selected_option_valid is not True
            or self.evidence_references_valid is not True
            or self.evidence_sufficiency != "sufficient"
        ):
            raise ValueError("Acceptance requires valid, traceable selection without review")
        return self


class VerificationSummary(DomainModel):
    total: int = 0
    verified: int = 0
    unavailable: int = 0
    errors: int = 0
    accepted: int = 0
    rejected: int = 0
    human_review: int = 0
    policy_violating: int = 0
    insufficient_evidence: int = 0
