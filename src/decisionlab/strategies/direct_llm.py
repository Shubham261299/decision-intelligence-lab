"""Provider-independent plain-text generation with strict domain parsing."""

import hashlib
from importlib.resources import files

from decisionlab.errors import DecisionExecutionError, ErrorCode
from decisionlab.models import DecisionProblem, ExecutionInfo, Recommendation, RunMetadata
from decisionlab.providers.base import TextProvider
from decisionlab.strategies.direct_parser import PARSER_VERSION, parse_recommendation

PROMPT_VERSION = "direct-routing-v1"


class DirectLLMStrategy:
    name = "direct-llm"
    version = "1"

    def __init__(self, provider: TextProvider) -> None:
        self._provider = provider
        self._prompt = files("decisionlab.prompts").joinpath(
            "direct_routing_v1.txt"
        ).read_text(encoding="utf-8")
        self.last_execution: ExecutionInfo | None = None

    def decide(self, problem: DecisionProblem) -> Recommendation:
        config = dict(self._provider.public_configuration())
        config.update({
            "prompt_sha256": hashlib.sha256(self._prompt.encode()).hexdigest(),
            "parser_version": PARSER_VERSION, "output_format": "text",
        })
        metadata = RunMetadata(
            strategy=self.name, strategy_version=self.version,
            provider=self._provider.name, model_id=self._provider.model_id,
            prompt_version=PROMPT_VERSION, generation_settings=config,
        )
        self.last_execution = ExecutionInfo(metadata=metadata)
        try:
            response = self._provider.generate_text(
                instructions=self._prompt, problem_json=problem.model_dump_json(),
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
        failures: dict[str, ErrorCode] = {
            "refused": "provider_refusal", "incomplete": "provider_incomplete",
            "malformed": "provider_malformed_response", "failed": "provider_api_error",
        }
        if response.status in failures:
            raise DecisionExecutionError(failures[response.status])
        recommendation = parse_recommendation(response.content_text or "")
        try:
            problem.validate_recommendation(recommendation)
        except ValueError:
            raise DecisionExecutionError(
                "invalid_references", candidate=recommendation,
            ) from None
        return recommendation
