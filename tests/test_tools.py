import pytest
from pydantic import ValidationError

from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem
from decisionlab.tools.executor import ToolRegistry, routing_tools
from decisionlab.tools.models import ToolOutput, ToolRequest, ToolResult
from decisionlab.tools.routing import FactLookupTool, PolicyLookupTool


def fact_request() -> ToolRequest:
    return ToolRequest(tool_name="fact_lookup", inputs={"fact_ids": ["urgent", "specialist_required"]})


def test_both_tools_and_deterministic_provenance(benchmark: Benchmark) -> None:
    problem = benchmark.problems[0]
    before = problem.model_dump_json()
    first = routing_tools().session(problem)
    fact = first.invoke(fact_request())
    policy = first.invoke(ToolRequest(tool_name="policy_lookup", inputs={"policy_id": "routing-benchmark-1"}))
    assert first.calls == 2 and len(first.results) == 2
    assert fact.status == policy.status == "success"
    assert fact.evidence_id.startswith("tool:fact_lookup:")
    assert policy.evidence_id.startswith("tool:policy_lookup:")
    assert fact.output.source_fact_ids == ["specialist_required", "urgent"]
    assert [f["id"] for f in fact.output.data["facts"]] == ["specialist_required", "urgent"]
    assert policy.output.data["policy"]["policy_version"] == "routing-benchmark-1"
    assert policy.output.source_constraint_ids == ["capacity"]
    assert fact.request == fact_request() and fact.tool_version == "1"
    assert ToolResult.model_validate_json(fact.model_dump_json()) == fact
    assert routing_tools().session(problem).invoke(fact_request()) == fact
    assert problem.model_dump_json() == before
    fact.output.data.clear()
    assert first.results[0].output.data  # Returned objects cannot rewrite trace.


def test_missing_fact_is_successful_lookup_not_tool_failure(benchmark: Benchmark) -> None:
    result = routing_tools().session(benchmark.problems[0]).invoke(
        ToolRequest(tool_name="fact_lookup", inputs={"fact_ids": ["not_supplied"]}),
    )
    assert result.status == "success"
    assert result.output.data == {"facts": [], "missing_fact_ids": ["not_supplied"]}
    assert result.output.source_fact_ids == []


@pytest.mark.parametrize("name,code", [("shell", "tool_unknown"), ("policy_lookup", "tool_not_allowed")])
def test_unknown_and_unallowlisted_rejected(benchmark: Benchmark, name: str, code: str) -> None:
    registry = ToolRegistry([FactLookupTool(), PolicyLookupTool()], allowlist=["fact_lookup"])
    session = registry.session(benchmark.problems[0])
    with pytest.raises(DecisionExecutionError, match=code):
        session.invoke(ToolRequest(tool_name=name))
    assert session.calls == 1 and session.results[0].error_code == code
    assert session.results[0].output is None and session.results[0].evidence_id is None


def test_duplicate_calls_rejected_and_session_closed(benchmark: Benchmark) -> None:
    session = routing_tools().session(benchmark.problems[0])
    session.invoke(fact_request())
    with pytest.raises(DecisionExecutionError, match="tool_duplicate_call"):
        session.invoke(fact_request())
    assert session.calls == 2 and len(session.results) == 2
    with pytest.raises(DecisionExecutionError, match="tool_session_closed"):
        session.invoke(fact_request())
    assert len(session.results) == 2


def test_call_limit_before_execution(benchmark: Benchmark) -> None:
    session = routing_tools().session(benchmark.problems[0], max_calls=1)
    session.invoke(fact_request())
    with pytest.raises(DecisionExecutionError, match="tool_call_limit"):
        session.invoke(ToolRequest(tool_name="policy_lookup", inputs={"policy_id": "routing-benchmark-1"}))
    assert session.calls == 1 and len(session.results) == 2
    assert session.results[-1].output is None


@pytest.mark.parametrize("inputs", [
    {}, {"fact_ids": []}, {"fact_ids": ["urgent", "urgent"]},
    {"fact_ids": ["urgent"], "shell": "arbitrary"}, {"fact_ids": 3},
])
def test_argument_validation(benchmark: Benchmark, inputs: dict[str, object]) -> None:
    session = routing_tools().session(benchmark.problems[0])
    with pytest.raises(DecisionExecutionError, match="tool_invalid_request"):
        session.invoke(ToolRequest.model_validate({"tool_name": "fact_lookup", "inputs": inputs}))
    assert session.results[0].status == "failure"


def test_policy_name_not_dynamic_code_or_path(benchmark: Benchmark) -> None:
    session = routing_tools().session(benchmark.problems[0])
    with pytest.raises(DecisionExecutionError, match="tool_invalid_request"):
        session.invoke(ToolRequest(tool_name="policy_lookup", inputs={"policy_id": "../../secret"}))


@pytest.mark.parametrize("mode,code", [("exception", "tool_failure"), ("invalid", "tool_invalid_result"), ("unknown_ref", "tool_invalid_result")])
def test_tool_failures_sanitized(benchmark: Benchmark, mode: str, code: str) -> None:
    class BrokenTool:
        name = "broken"
        version = "1"

        def execute(self, problem: DecisionProblem, request: ToolRequest) -> ToolOutput:
            if mode == "exception":
                raise RuntimeError("secret-provider-message")
            if mode == "invalid":
                return ToolOutput.model_construct(data="invalid")
            return ToolOutput(data={}, source_fact_ids=["invented"])

    session = ToolRegistry([BrokenTool()], allowlist=["broken"]).session(benchmark.problems[0])
    with pytest.raises(DecisionExecutionError, match=code):
        session.invoke(ToolRequest(tool_name="broken"))
    assert session.results[0].error_code == code
    assert "secret-provider-message" not in session.results[0].model_dump_json()


def test_tool_cannot_mutate_original_problem_or_request(benchmark: Benchmark) -> None:
    class MutatingTool:
        name = "mutating"
        version = "1"

        def execute(self, problem: DecisionProblem, request: ToolRequest) -> ToolOutput:
            problem.facts.clear()
            request.inputs.clear()
            return ToolOutput(data={"found": False})

    problem = benchmark.problems[0]
    request = ToolRequest(tool_name="mutating", inputs={"id": "test"})
    result = ToolRegistry([MutatingTool()], allowlist=["mutating"]).session(problem).invoke(request)
    assert problem.facts and request.inputs == {"id": "test"}
    assert result.request.inputs == request.inputs


@pytest.mark.parametrize("limit", [0, -1, 17, True, 1.5])
def test_invalid_limits(benchmark: Benchmark, limit: int) -> None:
    with pytest.raises(ValueError):
        routing_tools().session(benchmark.problems[0], max_calls=limit)


def test_invalid_registry_and_tool_result_contracts() -> None:
    for tools, allowed in (([], []), ([FactLookupTool()] * 2, ["fact_lookup"]), ([FactLookupTool()], ["shell"])):
        with pytest.raises(ValueError):
            ToolRegistry(tools, allowlist=allowed)
    with pytest.raises(ValidationError):
        ToolResult(request=fact_request(), status="success")
    with pytest.raises(ValidationError):
        ToolResult(request=fact_request(), status="failure", evidence_id="invented")


def test_registry_mapping_immutable_and_identity_validated() -> None:
    registry = routing_tools()
    with pytest.raises(TypeError):
        registry.tools["shell"] = FactLookupTool()
    invalid = FactLookupTool()
    invalid.version = " "
    with pytest.raises(ValueError, match="versions"):
        ToolRegistry([invalid], allowlist=["fact_lookup"])
