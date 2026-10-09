import json

import pytest

from decisionlab.comparison import StrategyEntry, comparison_json, run_comparison
from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem, Recommendation, RunMetadata
from decisionlab.runner import run_benchmark, run_decision
from decisionlab.strategies import DecisionStrategy, RuleBasedStrategy, ToolAssistedStrategy
from decisionlab.strategies.tool_generation import RoutingToolGenerator, RoutingToolPlanner
from decisionlab.tools.executor import ToolRegistry
from decisionlab.tools.models import ToolOutput, ToolRequest, ToolResult
from decisionlab.tools.routing import FactLookupTool, PolicyLookupTool
from decisionlab.verification import RuleBasedVerifier
from test_cli import invoke
from test_comparison import entries


def test_decision_uses_tool_evidence_with_valid_original_fact_refs(benchmark: Benchmark) -> None:
    strategy: DecisionStrategy = ToolAssistedStrategy()
    problem = benchmark.problems[0]
    before = problem.model_dump_json()
    rec = strategy.decide(problem)
    assert rec.selected_option_id == "q-r"
    assert before == problem.model_dump_json()
    problem.validate_recommendation(rec)
    trace = strategy.last_tool_results
    assert len(trace) == 2 and all(r.status == "success" for r in trace)
    cited = {i for a in rec.assessments for i in a.evidence_ids}
    sources = {i for r in trace if r.output for i in r.output.source_fact_ids}
    assert cited == sources == {"urgent", "specialist_required"}
    assert any(trace[0].evidence_id in reason for a in rec.assessments for reason in a.reasons)
    assert RuleBasedVerifier().verify(problem, rec).disposition == "accept"


def test_required_information_plan_is_input_only(benchmark: Benchmark) -> None:
    planner = RoutingToolPlanner()
    for problem in benchmark.problems:
        requests = planner.plan(problem)
        assert len(requests) == 2
        ids = requests[1].inputs["fact_ids"]
        assert {"urgent", "specialist_required"} <= set(ids)
        assert ("approval_granted" in ids) == any(c.id == "approval" and c.kind == "hard" for c in problem.constraints)
        assert ("review_required" in ids) == any(f.id == "review_required" for f in problem.facts)
        assert "acceptable_option_ids" not in json.dumps([r.model_dump() for r in requests])


def test_missing_material_information_abstains_and_verifier_reviews(benchmark: Benchmark) -> None:
    problem = next(p for p in benchmark.problems if any(f.id == "urgent" and f.value is None for f in p.facts))
    strategy = ToolAssistedStrategy()
    rec = strategy.decide(problem)
    assert rec.status == "abstained" and rec.unknowns
    assert all(r.status == "success" for r in strategy.last_tool_results)
    assert RuleBasedVerifier().verify(problem, rec).disposition == "human_review"


def test_ties_and_review_selection_are_generation_not_verification(benchmark: Benchmark) -> None:
    tied = benchmark.problems[8]
    assert RuleBasedStrategy().decide(tied).status == "abstained"
    rec = ToolAssistedStrategy().decide(tied)
    assert rec.selected_option_id == min(benchmark.expectations[tied.id].acceptable_option_ids)
    assert RuleBasedVerifier().verify(tied, rec).disposition == "accept"
    reviewed = next(p for p in benchmark.problems if any(f.id == "review_required" and f.value is True for f in p.facts)
                    and RuleBasedStrategy().decide(p).status == "selected")
    selection = ToolAssistedStrategy().decide(reviewed)
    assert selection.status == "selected" and selection.requested_review
    assert RuleBasedVerifier().verify(reviewed, selection).disposition == "human_review"


@pytest.mark.parametrize("mode,code", [("unknown", "tool_unknown"), ("duplicate", "tool_duplicate_call"), ("limit", "tool_call_limit")])
def test_planned_call_failures_preserve_trace(benchmark: Benchmark, mode: str, code: str) -> None:
    class Planner:
        def plan(self, problem: DecisionProblem) -> list[ToolRequest]:
            if mode == "unknown":
                return [ToolRequest(tool_name="shell")]
            requests = list(RoutingToolPlanner().plan(problem))
            return [requests[0], requests[0]] if mode == "duplicate" else requests

    strategy = ToolAssistedStrategy(planner=Planner(), max_tool_calls=1 if mode == "limit" else 2)
    observed = run_decision(benchmark.problems[0], strategy, RunMetadata(strategy="tool-assisted", strategy_version="1"), "r")
    assert observed.run.status == "failure" and observed.run.error.code == code
    assert observed.run.recommendation is None
    assert observed.tool_results[-1].error_code == code
    assert observed.run.metadata.generation_settings["max_tool_calls"] == (1 if mode == "limit" else 2)


def test_failed_tool_never_generates_or_abstains(benchmark: Benchmark) -> None:
    class FailingFacts(FactLookupTool):
        def execute(self, problem: DecisionProblem, request: ToolRequest) -> ToolOutput:
            raise RuntimeError("secret-error")

    class Generator:
        name = "mock"
        version = "1"
        calls = 0

        def generate(self, problem: DecisionProblem, evidence: list[ToolResult]) -> Recommendation:
            self.calls += 1
            return RuleBasedStrategy().decide(problem)

    generator = Generator()
    strategy = ToolAssistedStrategy(ToolRegistry([FailingFacts(), PolicyLookupTool()], allowlist=["fact_lookup", "policy_lookup"]), generator=generator)
    observed = run_decision(benchmark.problems[0], strategy, RunMetadata(strategy="tool-assisted", strategy_version="1"), "r")
    assert observed.run.error.code == "tool_failure" and generator.calls == 0
    assert len(observed.tool_results) == 2
    assert "secret-error" not in observed.model_dump_json()


def test_generator_only_accepts_authentic_tool_snapshots(benchmark: Benchmark) -> None:
    strategy = ToolAssistedStrategy()
    strategy.decide(benchmark.problems[0])
    trace = [r.model_copy(deep=True) for r in strategy.last_tool_results]
    trace[1].output.data["facts"][0]["value"] = True
    with pytest.raises(DecisionExecutionError, match="tool_invalid_result"):
        RoutingToolGenerator().generate(benchmark.problems[0], trace)
    with pytest.raises(DecisionExecutionError, match="tool_invalid_result"):
        RoutingToolGenerator().generate(benchmark.problems[0], [])


def test_generator_reads_evidence_and_calls_once(benchmark: Benchmark) -> None:
    class RecordingGenerator:
        name = "mock"
        version = "1"
        calls = 0

        def generate(self, problem: DecisionProblem, evidence: list[ToolResult]) -> Recommendation:
            self.calls += 1
            assert len(evidence) == 2 and all(r.output for r in evidence)
            return RoutingToolGenerator().generate(problem, evidence)

    generator = RecordingGenerator()
    strategy = ToolAssistedStrategy(generator=generator)
    assert strategy.decide(benchmark.problems[0]).selected_option_id == "q-r"
    assert generator.calls == 1
    first = strategy.last_tool_results
    strategy.decide(benchmark.problems[0])
    assert strategy.last_tool_results == first and generator.calls == 2


def test_comparison_preserves_existing_scores_and_tool_verification(benchmark: Benchmark) -> None:
    specs, _, _ = entries(benchmark)
    original = run_comparison(benchmark, specs)
    specs, _, _ = entries(benchmark)
    combined = run_comparison(benchmark, [*specs, StrategyEntry("tool-assisted", "1", ToolAssistedStrategy())])
    for name in original.columns:
        assert original.columns[name].report.summary == combined.columns[name].report.summary
    column = combined.columns["tool-assisted"]
    assert column.report.summary.rates["outcome_correctness"].value == 1
    assert column.report.verification_summary.verified == 30
    assert column.report.verification_summary.rejected == 0
    assert all(len(c.observation.tool_results) == 2 for c in column.report.cases)
    assert "tool:fact_lookup:" in comparison_json(combined)


def test_offline_cli_evaluation_and_comparison() -> None:
    evaluation = invoke("evaluate", "--strategy", "tool-assisted", "--json")
    assert evaluation.returncode == 0, evaluation.stderr
    report = json.loads(evaluation.stdout)
    assert report["strategy"] == "tool-assisted"
    assert report["cases"][0]["observation"]["tool_results"]
    comparison = invoke("compare", "--strategies", "rules", "tool-assisted", "--json")
    assert comparison.returncode == 0, comparison.stderr
    assert json.loads(comparison.stdout)["execution_order"] == ["rules", "tool-assisted"]


@pytest.mark.parametrize("mode,code", [
    ("exception", "decision_generation_failure"), ("malformed", "invalid_schema"),
    ("unknown_reference", "invalid_references"),
    ("uncollected", "uncollected_tool_evidence"),
])
def test_generation_failures_are_explicit_not_abstention(benchmark: Benchmark, mode: str, code: str) -> None:
    problem = benchmark.problems[0].model_copy(deep=True)
    from decisionlab.models import Fact
    problem.facts.append(Fact(id="decorative", statement="Irrelevant", value="note", source="note"))

    class Generator:
        name = "mock"
        version = "1"

        def generate(self, original: DecisionProblem, evidence: list[ToolResult]) -> Recommendation:
            if mode == "exception":
                raise RuntimeError("private-error-message")
            rec = RoutingToolGenerator().generate(original, evidence)
            if mode == "malformed":
                rec.selected_option_id = None
            elif mode == "unknown_reference":
                rec.assessments[0].evidence_ids = ["invented"]
            elif mode == "uncollected":
                rec.assessments[0].evidence_ids = ["decorative"]
            return rec

    observed = run_decision(problem, ToolAssistedStrategy(generator=Generator()),
                            RunMetadata(strategy="tool-assisted", strategy_version="1"), "r")
    assert observed.run.status == "failure" and observed.run.error.code == code
    assert observed.run.recommendation is None and len(observed.tool_results) == 2
    assert "private-error-message" not in observed.model_dump_json()


def test_wrong_generation_is_preserved_and_independently_rejected(benchmark: Benchmark) -> None:
    class WrongGenerator:
        name = "wrong-mock"
        version = "1"

        def generate(self, problem: DecisionProblem, evidence: list[ToolResult]) -> Recommendation:
            rec = RoutingToolGenerator().generate(problem, evidence)
            if problem.id == benchmark.problems[0].id:
                rec.selected_option_id = "q-e"
            return rec

    report = run_benchmark(benchmark, ToolAssistedStrategy(generator=WrongGenerator()),
                           strategy_name="tool-assisted", strategy_version="1")
    case = report.cases[0]
    assert case.observation.run.status == "success"
    assert case.observation.run.recommendation.selected_option_id == "q-e"
    assert not case.outcome_correct and "wrong_route" in case.hard_constraint_violations
    assert case.observation.verification.disposition == "reject"


def test_trace_resets_after_success_then_failure(benchmark: Benchmark) -> None:
    class ChangingPlanner:
        calls = 0

        def plan(self, problem: DecisionProblem) -> list[ToolRequest]:
            self.calls += 1
            return list(RoutingToolPlanner().plan(problem)) if self.calls == 1 else [ToolRequest(tool_name="unknown")]

    strategy = ToolAssistedStrategy(planner=ChangingPlanner())
    strategy.decide(benchmark.problems[0])
    with pytest.raises(DecisionExecutionError, match="tool_unknown"):
        strategy.decide(benchmark.problems[0])
    assert len(strategy.last_tool_results) == 1 and strategy.last_tool_results[0].status == "failure"
    assert strategy.last_execution.usage is None


def test_tool_repeated_reports_are_byte_identical(benchmark: Benchmark) -> None:
    from decisionlab.evaluation.report import report_json
    first = run_benchmark(benchmark, ToolAssistedStrategy(), strategy_name="tool-assisted", strategy_version="1", repeats=2)
    second = run_benchmark(benchmark, ToolAssistedStrategy(), strategy_name="tool-assisted", strategy_version="1", repeats=2)
    assert report_json(first) == report_json(second)
    assert first.summary.rates["repeated_decision_agreement"].value == 1
