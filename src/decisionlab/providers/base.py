"""Provider-neutral contracts for schema-constrained and plain-text generation."""

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from decisionlab.models import DomainModel, NonNegative, Scalar, Text, Usage

ProviderStatus = Literal["completed", "refused", "incomplete", "malformed", "failed"]


class ProviderResponse(DomainModel):
    status: ProviderStatus
    content_json: str | None = None
    model_id: Text
    response_id: Text | None = None
    usage: Usage | None = None
    latency_ms: NonNegative | None = None
    response_metadata: dict[str, Scalar] = Field(default_factory=dict)


class StructuredProvider(Protocol):
    name: str
    model_id: str
    retry_count: int

    def public_configuration(self) -> dict[str, Scalar]: ...

    def generate(
        self, *, instructions: str, problem_json: str, output_model: type[BaseModel],
    ) -> ProviderResponse: ...


class TextProviderResponse(DomainModel):
    status: ProviderStatus
    content_text: str | None = None
    model_id: Text
    response_id: Text | None = None
    usage: Usage | None = None
    latency_ms: NonNegative | None = None
    response_metadata: dict[str, Scalar] = Field(default_factory=dict)


class TextProvider(Protocol):
    name: str
    model_id: str
    retry_count: int

    def public_configuration(self) -> dict[str, Scalar]: ...

    def generate_text(
        self, *, instructions: str, problem_json: str,
    ) -> TextProviderResponse: ...
