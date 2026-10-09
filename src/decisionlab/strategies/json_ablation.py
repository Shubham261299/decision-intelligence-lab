"""Matched JSON conditions sharing the existing structured validation pipeline."""

import hashlib
import json
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel

from decisionlab.models import Scalar
from decisionlab.providers.base import ProviderResponse
from decisionlab.providers.factory import GenerationProvider
from decisionlab.strategies.structured_llm import (
    StructuredLLMStrategy, StructuredRecommendation,
)

Condition = Literal["unconstrained-json", "schema-enforced-json"]
CONDITIONS: tuple[Condition, ...] = ("unconstrained-json", "schema-enforced-json")


class JSONConditionProvider:
    """Adapt text transport to the same response and validation contract."""

    def __init__(self, provider: GenerationProvider, condition: Condition) -> None:
        if condition not in CONDITIONS:
            raise ValueError("Unknown JSON ablation condition")
        self._provider = provider
        self.condition = condition
        self.name = provider.name
        self.model_id = provider.model_id

    @property
    def retry_count(self) -> int:
        return self._provider.retry_count

    def public_configuration(self) -> dict[str, Scalar]:
        schema = json.dumps(StructuredRecommendation.model_json_schema(), sort_keys=True)
        return {
            **self._provider.public_configuration(),
            "condition": self.condition,
            "experiment_version": "matched-json-v1",
            "contract_version": "recommendation-transport-v1",
            "contract_sha256": hashlib.sha256(schema.encode()).hexdigest(),
        }

    def generate(
        self, *, instructions: str, problem_json: str, output_model: type[BaseModel],
    ) -> ProviderResponse:
        if self.condition == "schema-enforced-json":
            return self._provider.generate(
                instructions=instructions, problem_json=problem_json,
                output_model=output_model,
            )
        response = self._provider.generate_text(
            instructions=instructions, problem_json=problem_json,
        )
        return ProviderResponse(
            status=response.status, content_json=response.content_text,
            model_id=response.model_id, response_id=response.response_id,
            usage=response.usage, latency_ms=response.latency_ms,
            response_metadata=response.response_metadata,
        )


class MatchedJSONStrategy(StructuredLLMStrategy):
    """Only transport enforcement varies; no alternate parser or output repair."""

    prompt_version = "matched-json-v1"

    def __init__(self, provider: GenerationProvider, condition: Condition) -> None:
        super().__init__(JSONConditionProvider(provider, condition))
        self.name = condition
        instruction = files("decisionlab.prompts").joinpath(
            "matched_json_v1.txt"
        ).read_text(encoding="utf-8")
        schema = json.dumps(StructuredRecommendation.model_json_schema(), sort_keys=True)
        self._prompt += "\n" + instruction + "\n" + schema
