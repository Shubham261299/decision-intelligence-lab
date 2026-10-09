import pytest

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem, Recommendation, RunMetadata
from decisionlab.runner import run_benchmark, run_decision
from decisionlab.strategies import RuleBasedStrategy

META = RunMetadata(strategy="test", strategy_version="1")


class FixedStrategy:
    def __init__(self, recommendation: Recommendation) -> None:
        self.recommendation = recommendation

    def decide(self, problem: DecisionProblem) -> Recommendation:
        return self.recommendation


def test_strategy_receives_only_an_isolated_problem(benchmark: Benchmark) -> None:
    class MutatingStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            assert isinstance(problem, DecisionProblem)
            assert not hasattr(problem, "acceptable_option_ids")
            problem.facts.clear()
            problem.options.clear()
            return Recommendation(status="abstained", explanation="Test")

    problem = benchmark.problems[0]
    before = problem.model_dump_json()
    observation = run_decision(problem, MutatingStrategy(), META, "test-1")
    assert observation.run.status == "success"
    assert problem.model_dump_json() == before


def test_returned_model_is_revalidated_and_snapshotted(benchmark: Benchmark) -> None:
    rec = Recommendation(status="abstained", explanation="Test")
    observed = run_decision(benchmark.problems[0], FixedStrategy(rec), META, "r")
    rec.explanation = "Later mutation"
    assert observed.run.recommendation is not None
    assert observed.run.recommendation.explanation == "Test"
    rec.status = "selected"
    invalid = run_decision(benchmark.problems[0], FixedStrategy(rec), META, "r2")
    assert invalid.run.status == "failure"
    assert invalid.schema_valid is False
    assert invalid.run.recommendation is None


def test_invalid_evidence_is_rejected_but_preserved_as_diagnostic(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.assessments[0].evidence_ids = ["invented"]
    observation = run_decision(benchmark.problems[0], FixedStrategy(rec), META, "r")
    assert observation.run.status == "failure"
    assert observation.run.recommendation is None
    assert observation.candidate is not None
    assert observation.schema_valid is True
    assert observation.references_valid is False
    assert observation.evidence_references_valid is False
    assert observation.invalid_evidence_reference_count == 1


def test_unknown_option_and_valid_evidence_are_separate(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.selected_option_id = "missing"
    observation = run_decision(benchmark.problems[0], FixedStrategy(rec), META, "r")
    assert observation.references_valid is False
    assert observation.evidence_references_valid is True


def test_exception_is_not_abstention_and_does_not_leak_payload(benchmark: Benchmark) -> None:
    class BrokenStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            raise RuntimeError("secret-token-123")

    observed = run_decision(benchmark.problems[0], BrokenStrategy(), META, "r")
    assert observed.run.status == "failure"
    assert observed.schema_valid is None
    assert observed.candidate is None
    assert "secret-token" not in observed.model_dump_json()


def test_repeated_runs_and_metadata(benchmark: Benchmark) -> None:
    report = run_benchmark(benchmark, RuleBasedStrategy(), strategy_name="rules", strategy_version="1", repeats=2)
    assert len(report.cases) == 60
    assert report.summary.rates["repeated_decision_agreement"].numerator == 30
    assert report.summary.rates["repeated_decision_agreement"].denominator == 30
    assert all(r.observation.run.metadata.dataset_version == "1.0.0" for r in report.cases)


@pytest.mark.parametrize("repeats", [0, -1, True])
def test_invalid_repetition_count(benchmark: Benchmark, repeats: int) -> None:
    with pytest.raises(ValueError):
        run_benchmark(benchmark, RuleBasedStrategy(), strategy_name="rules", strategy_version="1", repeats=repeats)
