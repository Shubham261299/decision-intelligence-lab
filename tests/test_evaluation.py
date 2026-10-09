import pytest

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.evaluator import evaluate_observation
from decisionlab.evaluation.metrics import summarize
from decisionlab.evaluation.models import CaseEvaluation
from decisionlab.models import Recommendation, RunMetadata
from decisionlab.runner import run_decision
from decisionlab.strategies import RuleBasedStrategy
from test_runner import FixedStrategy

META = RunMetadata(strategy="test", strategy_version="1")


def evaluate(benchmark: Benchmark, index: int, rec: Recommendation) -> CaseEvaluation:
    problem = benchmark.problems[index]
    observation = run_decision(problem, FixedStrategy(rec), META, f"r-{index}")
    return evaluate_observation(problem, benchmark.expectations[problem.id], observation)


def test_evaluator_never_modifies_result(benchmark: Benchmark) -> None:
    problem = benchmark.problems[0]
    observation = run_decision(problem, RuleBasedStrategy(), META, "r")
    before = observation.model_dump_json()
    result = evaluate_observation(problem, benchmark.expectations[problem.id], observation)
    assert observation.model_dump_json() == before
    assert result.outcome_correct


@pytest.mark.parametrize("option", ["q-r", "q-r2"])
def test_multiple_acceptable_choices(benchmark: Benchmark, option: str) -> None:
    result = evaluate(benchmark, 8, Recommendation(status="selected", selected_option_id=option, explanation="Eligible"))
    assert result.selection_correct is True
    assert result.abstention_correct is None
    assert not result.hard_constraint_violations


def test_abstention_on_tie_is_not_selection_success(benchmark: Benchmark) -> None:
    result = evaluate(benchmark, 8, Recommendation(status="abstained", explanation="Tie"))
    assert result.selection_correct is False
    assert not result.outcome_correct
    assert not result.hard_constraint_violations
    assert not result.review_correct


def test_abstention_correctness_separate_from_selection(benchmark: Benchmark) -> None:
    result = evaluate(benchmark, 10, Recommendation(status="abstained", explanation="Urgency unknown"))
    assert result.selection_correct is None
    assert result.abstention_correct is True
    assert result.review_correct


def test_selected_review_is_independent_of_abstention(benchmark: Benchmark) -> None:
    result = evaluate(benchmark, 15, Recommendation(status="selected", selected_option_id="q-r", explanation="Routine"))
    assert result.selection_correct and result.outcome == "selected"
    assert result.strategy_review is False
    assert result.system_review is True
    assert result.review_correct


def test_wrong_route_violates_policy_without_repair(benchmark: Benchmark) -> None:
    result = evaluate(benchmark, 0, Recommendation(status="selected", selected_option_id="q-e", explanation="Wrong"))
    assert not result.outcome_correct
    assert "wrong_route" in result.hard_constraint_violations
    assert result.observation.run.recommendation.selected_option_id == "q-e"


def test_invalid_references_are_failures_not_abstentions(benchmark: Benchmark) -> None:
    result = evaluate(benchmark, 10, Recommendation(status="selected", selected_option_id="missing", explanation="Wrong"))
    assert result.outcome == "failure"
    assert result.abstention_correct is False
    assert "unknown_selected_option" in result.hard_constraint_violations


def test_existing_citations_do_not_imply_supported_explanation(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.explanation = "The moon is made of cheese, therefore routine routing"
    result = evaluate(benchmark, 0, rec)
    assert result.observation.evidence_references_valid
    assert result.evidence_support_score is None


def test_known_metric_arithmetic_and_confidence(benchmark: Benchmark) -> None:
    good = evaluate(benchmark, 0, Recommendation(status="selected", selected_option_id="q-r", explanation="Correct", confidence=0.8))
    wrong = evaluate(benchmark, 1, Recommendation(status="selected", selected_option_id="q-r", explanation="Wrong", confidence=0.6))
    abstain = evaluate(benchmark, 10, Recommendation(status="abstained", explanation="Missing urgency", confidence=0.99))
    summary = summarize([good, wrong, abstain])
    assert summary.rates["outcome_correctness"].value == pytest.approx(2 / 3)
    assert summary.rates["selection_correctness"].value == 0.5
    assert summary.rates["abstention_correctness"].value == 1
    assert summary.rates["selected_output_accuracy"].value == 0.5
    assert summary.rates["decision_coverage"].value == pytest.approx(2 / 3)
    assert summary.rates["hard_constraint_violation_rate"].value == 0.5
    assert summary.confidence.sample_count == 2
    assert summary.confidence.mean_confidence == 0.7
    assert summary.confidence.brier_score == pytest.approx(0.2)


@pytest.mark.parametrize("confidence,index", [(0.0, 0), (0.2, 1), (0.4, 2), (0.6, 3), (0.8, 4), (1.0, 4)])
def test_confidence_bin_boundaries(benchmark: Benchmark, confidence: float, index: int) -> None:
    result = evaluate(benchmark, 0, Recommendation(status="selected", selected_option_id="q-r", explanation="Correct", confidence=confidence))
    bins = summarize([result]).confidence.bins
    assert bins[index].count == 1
    assert sum(b.count for b in bins) == 1


def test_empty_metrics_have_null_denominators() -> None:
    summary = summarize([])
    assert all(rate.value is None for rate in summary.rates.values())
    assert summary.confidence.brier_score is None
    assert summary.confidence.sample_count == 0


def test_reference_id_mismatch_rejected(benchmark: Benchmark) -> None:
    observation = run_decision(benchmark.problems[0], RuleBasedStrategy(), META, "r")
    with pytest.raises(ValueError):
        evaluate_observation(benchmark.problems[1], benchmark.expectations[benchmark.problems[1].id], observation)


def test_evaluator_does_not_trust_recorded_violation_claims(benchmark: Benchmark) -> None:
    result = evaluate(benchmark, 0, Recommendation(
        status="selected", selected_option_id="q-e", explanation="Wrong route",
    ))
    observation = result.observation
    assert observation.run.policy_result is not None
    observation.run.policy_result.detected_violations = []
    checked = evaluate_observation(
        benchmark.problems[0], benchmark.expectations["route-001"], observation,
    )
    assert "wrong_route" in checked.hard_constraint_violations
    assert observation.run.policy_result.detected_violations == []


def test_failure_does_not_inflate_abstention_or_review_scores(benchmark: Benchmark) -> None:
    from decisionlab.models import DecisionProblem

    class BrokenStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            raise RuntimeError("Failure")

    problem = benchmark.problems[10]
    observed = run_decision(problem, BrokenStrategy(), META, "r")
    result = evaluate_observation(problem, benchmark.expectations[problem.id], observed)
    summary = summarize([result])
    assert summary.counts["abstained"] == 0
    assert summary.counts["failed_runs"] == 1
    assert summary.rates["abstention_correctness"].value == 0
    assert summary.rates["human_review_recall"].value == 0
    assert summary.rates["schema_validity"].value is None
    assert summary.rates["evidence_reference_validity"].value is None
