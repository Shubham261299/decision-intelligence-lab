"""Bounded tools -> evidence -> one generation; verification stays external."""

from pydantic import ValidationError

from decisionlab.errors import DecisionExecutionError
from decisionlab.models import DecisionProblem, ExecutionInfo, Recommendation, RunMetadata
from decisionlab.strategies.tool_generation import (
    RoutingToolGenerator, RoutingToolPlanner, ToolDecisionGenerator, ToolPlanner,
)
from decisionlab.tools.executor import ToolRegistry, routing_tools
from decisionlab.tools.models import ToolResult


class ToolAssistedStrategy:
    name = "tool-assisted"
    version = "1"

    def __init__(
        self, registry: ToolRegistry | None = None, *,
        generator: ToolDecisionGenerator | None = None,
        planner: ToolPlanner | None = None, max_tool_calls: int = 2,
    ) -> None:
        if type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 16:
            raise ValueError("Tool call limit must be an integer from 1 through 16")
        self._registry = registry if registry is not None else routing_tools()
        self._generator = generator if generator is not None else RoutingToolGenerator()
        self._planner = planner if planner is not None else RoutingToolPlanner()
        self.max_tool_calls = max_tool_calls
        self.last_tool_results: list[ToolResult] = []
        self.last_execution: ExecutionInfo | None = None

    def decide(self, problem: DecisionProblem) -> Recommendation:
        self.last_tool_results = []
        self.last_execution = ExecutionInfo(metadata=RunMetadata(
            strategy=self.name, strategy_version=self.version,
            generation_settings={
                "max_tool_calls": self.max_tool_calls,
                "tool_allowlist": ",".join(sorted(self._registry.allowlist)),
                "decision_generator": self._generator.name,
                "decision_generator_version": self._generator.version,
            },
        ))
        problem = DecisionProblem.model_validate(problem.model_dump())
        session = self._registry.session(problem, max_calls=self.max_tool_calls)
        try:
            for request in self._planner.plan(problem.model_copy(deep=True)):
                session.invoke(request)
        finally:
            self.last_tool_results = [r.model_copy(deep=True) for r in session.results]
        try:
            generated = self._generator.generate(
                problem.model_copy(deep=True),
                [r.model_copy(deep=True) for r in self.last_tool_results],
            )
        except DecisionExecutionError:
            raise
        except Exception:
            raise DecisionExecutionError("decision_generation_failure") from None
        try:
            recommendation = Recommendation.model_validate(generated.model_dump(warnings=False))
        except (ValidationError, TypeError, ValueError, AttributeError):
            raise DecisionExecutionError("invalid_schema") from None
        try:
            problem.validate_recommendation(recommendation)
        except ValueError:
            raise DecisionExecutionError("invalid_references", candidate=recommendation) from None
        retrieved_ids = {i for r in self.last_tool_results if r.output is not None
                         for i in r.output.source_fact_ids}
        if any(i not in retrieved_ids for a in recommendation.assessments for i in a.evidence_ids):
            raise DecisionExecutionError("uncollected_tool_evidence")
        retrieved_constraints = {
            i for r in self.last_tool_results if r.output is not None
            for i in r.output.source_constraint_ids
        }
        if any(i not in retrieved_constraints for a in recommendation.assessments for i in a.constraint_ids):
            raise DecisionExecutionError("uncollected_tool_evidence")
        return recommendation
