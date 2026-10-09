"""Data-only tool protocol; registration is trusted developer configuration."""

from typing import Protocol, runtime_checkable

from decisionlab.models import DecisionProblem
from decisionlab.tools.models import ToolOutput, ToolRequest, ToolResult


class Tool(Protocol):
    name: str
    version: str

    def execute(self, problem: DecisionProblem, request: ToolRequest) -> ToolOutput: ...


@runtime_checkable
class ToolTrackedStrategy(Protocol):
    last_tool_results: list[ToolResult]
