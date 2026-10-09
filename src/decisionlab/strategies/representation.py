"""Matched representations with prospective capture and unchanged strict validation."""

import hashlib
import json
from importlib.resources import files
from typing import Literal
from urllib.error import HTTPError

from pydantic import JsonValue, ValidationError

from decisionlab.errors import DecisionExecutionError, ErrorCode
from decisionlab.models import DecisionProblem, DomainModel, ExecutionInfo, Recommendation, RunMetadata, Scalar
from decisionlab.providers.base import TextProvider
from decisionlab.providers.ollama import LocalHTTPTransport, LocalTransport
from decisionlab.strategies.direct_parser import ParserExecutionError, parse_recommendation
from decisionlab.strategies.structured_llm import StructuredRecommendation

Representation = Literal["labelled-text", "unconstrained-json"]
REPRESENTATIONS: tuple[Representation, ...] = ("labelled-text", "unconstrained-json")


def digest(text: str | None) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


class RawResponse(DomainModel):
    text: str | None = None
    response_sha256: str | None = None
    provider_status: str | None = None
    envelope: str | None = None
    envelope_sha256: str | None = None
    failure_stage: str | None = None
    validation_issues: list[str] = []


class CaptureLocalTransport:
    """Optional Ollama envelope capture; no request headers or environment values."""

    def __init__(self, transport: LocalTransport | None = None) -> None:
        self._transport = transport if transport is not None else LocalHTTPTransport()
        self.envelope: str | None = None

    def request(self, url: str, payload: dict[str, JsonValue] | None, *, timeout: float) -> str:
        self.envelope = None
        try:
            self.envelope = self._transport.request(url, payload, timeout=timeout)
        except HTTPError as error:
            # A returned error body is data, not a credential-bearing request.
            try:
                self.envelope = error.read().decode("utf-8")
            except (OSError, UnicodeError):
                pass
            raise
        return self.envelope


class RepresentationStrategy:
    version = "1"
    prompt_version = "representation-v1"

    def __init__(self, provider: TextProvider, condition: Representation,
                 capture: CaptureLocalTransport | None = None) -> None:
        if condition not in REPRESENTATIONS:
            raise ValueError("Unknown representation condition")
        self.name = condition
        self._provider = provider
        self._capture = capture
        resources = files("decisionlab.prompts")
        self.policy = resources.joinpath("representation_policy_v1.txt").read_text(encoding="utf-8")
        suffix = resources.joinpath(
            "representation_text_v1.txt" if condition == "labelled-text" else "representation_json_v1.txt"
        ).read_text(encoding="utf-8")
        self.prompt = self.policy + "\n" + suffix
        self.last_execution: ExecutionInfo | None = None
        self.last_raw = RawResponse()

    def configuration(self) -> dict[str, Scalar]:
        return {**self._provider.public_configuration(),
                "provider": self._provider.name, "model_id": self._provider.model_id}

    def decide(self, problem: DecisionProblem) -> Recommendation:
        self.last_raw = RawResponse()
        if self._capture is not None:
            self._capture.envelope = None
        metadata = RunMetadata(
            strategy=self.name, strategy_version=self.version,
            provider=self._provider.name, model_id=self._provider.model_id,
            prompt_version=self.prompt_version,
            generation_settings={
                **self._provider.public_configuration(), "condition": self.name,
                "experiment_version": "representation-v1", "output_enforcement": "none",
                "prompt_sha256": digest(self.prompt) or "", "policy_prompt_sha256": digest(self.policy) or "",
                "contract_version": "recommendation-v1",
                "parser_version": "direct-text-v1" if self.name == "labelled-text" else "strict-json-v1",
            },
        )
        self.last_execution = ExecutionInfo(metadata=metadata)
        try:
            # Both conditions call only the provider's normal-text interface.
            response = self._provider.generate_text(
                instructions=self.prompt, problem_json=problem.model_dump_json(),
            )
        except DecisionExecutionError:
            self.last_execution.metadata.retry_count = self._provider.retry_count
            self.last_raw.failure_stage = "provider_api_failure"
            raise
        finally:
            if self._capture is not None:
                self.last_raw.envelope = self._capture.envelope
                self.last_raw.envelope_sha256 = digest(self.last_raw.envelope)
        self.last_raw.text = response.content_text
        self.last_raw.response_sha256 = digest(response.content_text)
        self.last_raw.provider_status = response.status
        self.last_execution = ExecutionInfo(
            metadata=metadata.model_copy(update={
                "model_id": response.model_id, "provider_response_id": response.response_id,
                "retry_count": self._provider.retry_count,
            }), usage=response.usage, latency_ms=response.latency_ms,
            response_metadata=response.response_metadata,
        )
        statuses: dict[str, ErrorCode] = {
            "refused": "provider_refusal", "incomplete": "provider_incomplete",
            "malformed": "provider_malformed_response", "failed": "provider_api_error",
        }
        if response.status in statuses:
            self.last_raw.failure_stage = (
                "empty_incomplete_refusal" if response.status in ("refused", "incomplete")
                else "provider_api_failure"
            )
            raise DecisionExecutionError(statuses[response.status])
        raw = response.content_text or ""
        if not raw.strip():
            self.last_raw.failure_stage = "empty_incomplete_refusal"
            raise DecisionExecutionError("invalid_schema")
        try:
            if self.name == "labelled-text":
                recommendation = parse_recommendation(raw)
            else:
                try:
                    json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    self.last_raw.failure_stage = "serialization_format_failure"
                    raise DecisionExecutionError("invalid_schema") from None
                wire = StructuredRecommendation.model_validate_json(raw)
                recommendation = Recommendation.model_validate(wire.model_dump())
        except ParserExecutionError as error:
            self.last_raw.failure_stage = error.stage
            self.last_raw.validation_issues = list(error.issues)
            if raw.lstrip().startswith(("```", "{")):
                self.last_raw.failure_stage = "serialization_format_failure"
            raise
        except ValidationError as error:
            self.last_raw.failure_stage = "schema_domain_validation"
            self.last_raw.validation_issues = [
                f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                for item in error.errors(include_input=False, include_context=False)
            ]
            raise DecisionExecutionError("invalid_schema") from None
        try:
            problem.validate_recommendation(recommendation)
        except ValueError:
            self.last_raw.failure_stage = "invalid_option_reference"
            raise DecisionExecutionError("invalid_references", candidate=recommendation) from None
        return recommendation
