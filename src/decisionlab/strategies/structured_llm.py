"""Provider-independent structured generation followed by domain validation."""

import hashlib
from importlib.resources import files
from typing import Literal

from pydantic import ConfigDict, ValidationError

from decisionlab.errors import DecisionExecutionError
from decisionlab.models import (
    DecisionProblem, DomainModel, ExecutionInfo, Recommendation, RunMetadata,
)
from decisionlab.providers.base import StructuredProvider

PROMPT_VERSION = "routing-v1"


class StructuredOutput(DomainModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class StructuredAssessment(StructuredOutput):
    option_id: str
    eligibility: Literal["eligible", "ineligible", "unknown"]
    reasons: list[str]
    evidence_ids: list[str]
    constraint_ids: list[str]


class StructuredUnknown(StructuredOutput):
    description: str
    affected_option_ids: list[str]
    material_to_decision: bool


class StructuredRecommendation(StructuredOutput):
    """All fields required; nullable fields express absence in strict API schemas.

    Domain constraints are applied by Recommendation after this transport schema.
    """

    status: Literal["selected", "abstained"]
    selected_option_id: str | None
    assessments: list[StructuredAssessment]
    unknowns: list[StructuredUnknown]
    confidence: float | None
    explanation: str
    requested_review: bool


class StructuredLLMStrategy:
    name = "structured-llm"
    version = "1"
    prompt_version = PROMPT_VERSION

    def __init__(self, provider: StructuredProvider) -> None:
        self._provider = provider
        self._prompt = files("decisionlab.prompts").joinpath("routing_v1.txt").read_text(encoding="utf-8")
        self.last_execution: ExecutionInfo | None = None

    def decide(self, problem: DecisionProblem) -> Recommendation:
        config = dict(self._provider.public_configuration())
        config["prompt_sha256"] = hashlib.sha256(self._prompt.encode()).hexdigest()
        metadata = RunMetadata(
            strategy=self.name, strategy_version=self.version,
            provider=self._provider.name, model_id=self._provider.model_id,
            prompt_version=self.prompt_version, generation_settings=config,
        )
        self.last_execution = ExecutionInfo(metadata=metadata)
        try:
            response = self._provider.generate(
                instructions=self._prompt, problem_json=problem.model_dump_json(),
                output_model=StructuredRecommendation,
            )
        except DecisionExecutionError:
            self.last_execution.metadata.retry_count = self._provider.retry_count
            raise
        self.last_execution = ExecutionInfo(
            metadata=metadata.model_copy(update={
                "model_id": response.model_id,
                "provider_response_id": response.response_id,
                "retry_count": self._provider.retry_count,
            }), usage=response.usage, latency_ms=response.latency_ms,
            response_metadata=response.response_metadata,
        )
        if response.status == "refused":
            raise DecisionExecutionError("provider_refusal")
        if response.status == "incomplete":
            raise DecisionExecutionError("provider_incomplete")
        if response.status == "malformed":
            raise DecisionExecutionError("provider_malformed_response")
        if response.status == "failed":
            raise DecisionExecutionError("provider_api_error")
        try:
            wire = StructuredRecommendation.model_validate_json(response.content_json or "")
            recommendation = Recommendation.model_validate(wire.model_dump())
        except (ValidationError, ValueError, TypeError):
            raise DecisionExecutionError("invalid_schema") from None
        try:
            problem.validate_recommendation(recommendation)
        except ValueError:
            raise DecisionExecutionError("invalid_references", candidate=recommendation) from None
        return recommendation
