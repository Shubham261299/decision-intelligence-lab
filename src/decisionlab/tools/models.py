"""JSON-only tool inputs, outputs and immutable-by-convention provenance."""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from decisionlab.models import DomainModel, Text


class ToolRequest(DomainModel):
    tool_name: Text
    inputs: dict[str, JsonValue] = Field(default_factory=dict)


class ToolOutput(DomainModel):
    data: dict[str, JsonValue]
    source_fact_ids: list[Text] = Field(default_factory=list)
    source_constraint_ids: list[Text] = Field(default_factory=list)


class ToolResult(DomainModel):
    request: ToolRequest
    tool_version: Text | None = None
    status: Literal["success", "failure"]
    evidence_id: Text | None = None
    output: ToolOutput | None = None
    error_code: Text | None = None

    @model_validator(mode="after")
    def check_status(self) -> Self:
        if self.status == "success":
            if self.output is None or self.evidence_id is None or self.tool_version is None or self.error_code is not None:
                raise ValueError("Successful tool calls require output, evidence and version")
        elif self.error_code is None or self.output is not None or self.evidence_id is not None:
            raise ValueError("Failed calls require a code and no accepted evidence")
        return self
