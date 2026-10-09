from decisionlab.decision_model import DecisionModel
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem, Recommendation, RunMetadata
from decisionlab.runner import run_benchmark, run_decision
from decisionlab.strategies import RuleBasedStrategy, ToolAssistedStrategy


class AlwaysAbstain:
    """A new decision model needs neither inheritance nor provider metadata."""

    def __init__(self) -> None:
        self.inputs: list[DecisionProblem] = []

    def decide(self, problem: DecisionProblem) -> Recommendation:
        self.inputs.append(problem.model_copy(deep=True))
        problem.facts.clear()
        return Recommendation(status="abstained", explanation="Demonstration abstention")


def test_models_use_shared_contract_without_adapters(benchmark: Benchmark) -> None:
    model: DecisionModel = AlwaysAbstain()
    assert isinstance(model, DecisionModel)
    assert isinstance(RuleBasedStrategy(), DecisionModel)
    assert isinstance(ToolAssistedStrategy(), DecisionModel)
    before = benchmark.model_dump_json()
    report = run_benchmark(benchmark, model, strategy_name="always-abstain", strategy_version="1")
    assert benchmark.model_dump_json() == before
    assert report.summary.counts["abstained"] == 30
    assert report.summary.rates["outcome_correctness"].numerator == 11
    assert report.summary.counts["failed_runs"] == 0
    assert all(run.observation.run.metadata.provider is None for run in report.cases)
    assert model.inputs == benchmark.problems


def test_protocol_membership_is_not_output_validation(benchmark: Benchmark) -> None:
    class Broken:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            return Recommendation.model_construct(status="selected", explanation="Missing selection")

    model: DecisionModel = Broken()
    assert isinstance(model, DecisionModel)
    observation = run_decision(benchmark.problems[0], model,
                              RunMetadata(strategy="broken", strategy_version="1"), "broken-run")
    assert observation.run.status == "failure"
    assert observation.run.error.code == "invalid_schema"
    assert observation.run.recommendation is None
