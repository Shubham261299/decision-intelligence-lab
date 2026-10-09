"""Frozen diagnostic arithmetic, selection-only scope and split isolation."""

import math
from contextlib import nullcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from decisionlab.comparison import comparison_json, comparison_text, run_comparison
from decisionlab.confidence import ConfidenceEstimate, selection_confidence
from decisionlab.evaluation.benchmark import Benchmark, load_benchmark
from decisionlab.evaluation.metrics import summarize
from decisionlab.evaluation.report import report_json, report_text
from decisionlab.models import Recommendation
from decisionlab.runner import run_benchmark
from decisionlab.strategies import RuleBasedStrategy
from decisionlab.strategies import ToolAssistedStrategy
from decisionlab.verification.rules import RuleBasedVerifier
from test_comparison import entries
from test_direct_llm import response
from test_direct_parser import format_recommendation
from test_evaluation import evaluate
from test_structured_llm import response as structured_response


@pytest.mark.parametrize("probability", [0.0, 0.2, 0.999, 1.0])
def test_typed_confidence_contract(probability: float) -> None:
    estimate = ConfidenceEstimate(
        probability=probability, source="strategy_reported", calibration="uncalibrated",
    )
    assert estimate.target == "acceptable_selected_option"
    assert ConfidenceEstimate.model_validate_json(estimate.model_dump_json()) == estimate


@pytest.mark.parametrize("probability", [-0.1, 1.01, math.nan, math.inf, -math.inf, True, "0.5"])
def test_invalid_confidence_contract(probability: object) -> None:
    with pytest.raises(ValidationError):
        ConfidenceEstimate(
            probability=probability, source="strategy_reported", calibration="uncalibrated",
        )
    with pytest.raises(ValidationError):
        Recommendation(status="selected", selected_option_id="q-r",
                       explanation="Invalid confidence", confidence=probability)


@pytest.mark.parametrize("payload", [
    {"probability": 0.5}, {"source": "strategy_reported"},
    {"calibration": "uncalibrated"},
    {"probability": 0.5, "source": "strategy_reported", "calibration": "calibrated"},
])
def test_inconsistent_contract_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ConfidenceEstimate.model_validate(payload)


def test_missing_and_abstention_are_not_invented(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    before = rec.model_dump_json()
    assert selection_confidence(rec) == ConfidenceEstimate()
    assert selection_confidence(None) == ConfidenceEstimate()
    assert rec.model_dump_json() == before
    abstain = Recommendation(status="abstained", explanation="Unknown", confidence=0.9)
    assert selection_confidence(abstain).probability is None
    assert abstain.confidence == 0.9


def test_brier_ece_distribution_and_coverage(benchmark: Benchmark) -> None:
    good = evaluate(benchmark, 0, Recommendation(
        status="selected", selected_option_id="q-r", explanation="Correct", confidence=0.8,
    ))
    wrong = evaluate(benchmark, 1, Recommendation(
        status="selected", selected_option_id="q-r", explanation="Wrong", confidence=0.6,
    ))
    abstain = evaluate(benchmark, 10, Recommendation(
        status="abstained", explanation="Missing", confidence=0.99,
    ))
    missing = evaluate(benchmark, 2, RuleBasedStrategy().decide(benchmark.problems[2]))
    failed = evaluate(benchmark, 0, Recommendation(
        status="selected", selected_option_id="absent", explanation="Invalid", confidence=1.0,
    ))
    summary = summarize([good, wrong, abstain, missing, failed])
    c = summary.confidence
    assert c.sample_count == 2 and c.selected_count == 3
    assert c.missing_selected_count == 1
    assert c.excluded_abstention_confidence_count == c.excluded_failure_count == 1
    assert c.brier_score == pytest.approx(0.2)
    assert c.calibration_error == pytest.approx(0.4)
    assert c.maximum_reliability_gap == pytest.approx(0.6)
    assert [b.count for b in c.bins] == [0, 0, 0, 1, 1]
    assert c.bins[3].accuracy == 0 and c.bins[4].accuracy == 1
    assert c.bins[3].fraction == c.bins[4].fraction == 0.5
    assert c.bins[0].reliability_gap is None
    assert c.coverage_curve[0].coverage.value == 2 / 5
    assert c.coverage_curve[0].retention.value == 1
    assert c.coverage_curve[0].accuracy.value == 0.5
    assert c.coverage_curve[4].coverage.value == 1 / 5
    assert c.coverage_curve[4].accuracy.value == 1
    assert c.coverage_curve[-1].accuracy.value is None
    assert c.fixed_reference.sample_count == 3
    assert c.fixed_reference.brier_score == 0.25
    assert c.fixed_reference.calibration_error == pytest.approx(abs(0.5 - 2 / 3))
    assert failed.observation.candidate.confidence == 1.0
    assert failed.observation.run.recommendation is None


def test_weighted_ece_is_not_unweighted_bin_mean(benchmark: Benchmark) -> None:
    good = evaluate(benchmark, 0, Recommendation(
        status="selected", selected_option_id="q-r", explanation="Correct", confidence=0.8,
    ))
    wrong = evaluate(benchmark, 1, Recommendation(
        status="selected", selected_option_id="q-r", explanation="Wrong", confidence=0.6,
    ))
    assert summarize([good, good, good, wrong]).confidence.calibration_error == pytest.approx(0.3)


def test_calibration_is_not_accuracy(benchmark: Benchmark) -> None:
    cases = [evaluate(benchmark, index, Recommendation(
        status="selected", selected_option_id="q-r", explanation="Fixed estimate", confidence=0.5,
    )) for index in (0, 1)]
    summary = summarize(cases)
    assert summary.rates["selected_output_accuracy"].value == 0.5
    assert summary.confidence.calibration_error == 0
    assert summary.confidence.brier_score == 0.25


@pytest.mark.parametrize("confidence", [1.1, math.nan, True, "0.5"])
def test_invalid_confidence_stays_execution_failure(benchmark: Benchmark, confidence: object) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    # model_copy bypasses validation, representing an invalid strategy output.
    rec = rec.model_copy(update={"confidence": confidence})
    with pytest.warns(UserWarning, match="serializer warnings") if isinstance(
        confidence, str
    ) else nullcontext():
        result = evaluate(benchmark, 0, rec)
    assert result.outcome == "failure"
    assert result.observation.run.error.code == "invalid_schema"
    c = summarize([result]).confidence
    assert c.sample_count == 0 and c.excluded_failure_count == 1
    assert c.brier_score is None


def test_confidence_does_not_gate_verifier_or_review(benchmark: Benchmark) -> None:
    problem = benchmark.problems[15]
    rec = RuleBasedStrategy().decide(problem)
    low = rec.model_copy(update={"confidence": 0.0})
    high = rec.model_copy(update={"confidence": 1.0})
    assert RuleBasedVerifier().verify(problem, low) == RuleBasedVerifier().verify(problem, high)
    for candidate in (low, high):
        result = evaluate(benchmark, 15, candidate)
        assert result.outcome_correct and result.system_review is True


def test_no_samples_are_explicitly_unavailable() -> None:
    c = summarize([]).confidence
    assert c.availability == "unavailable"
    assert c.mean_confidence is c.brier_score is c.calibration_error is None
    assert c.maximum_reliability_gap is None
    assert all(b.fraction is b.accuracy is b.reliability_gap is None for b in c.bins)
    assert all(p.coverage.value is p.retention.value is p.accuracy.value is None
               for p in c.coverage_curve)
    assert c.fixed_reference.sample_count == 0
    assert c.fixed_reference.brier_score is None


def test_frozen_split_calibration_and_mocked_llm_comparison(benchmark: Benchmark) -> None:
    specs, direct, structured = entries(benchmark)
    # Fixed synthetic probabilities author mock outputs only; no threshold fitting.
    for index, problem in enumerate(benchmark.problems):
        rec = RuleBasedStrategy().decide(problem)
        rec.confidence = 0.8 if problem.id in benchmark.manifest.splits["dev"] else 0.6
        direct.events[index] = response(format_recommendation(rec))
        structured.events[index] = structured_response(rec.model_dump_json())
    before = benchmark.model_dump_json()
    report = run_comparison(benchmark, specs)
    assert benchmark.model_dump_json() == before
    direct_report = report.columns["direct-llm"].report
    assert direct_report.summary.confidence == report.columns["structured-llm"].report.summary.confidence
    for split, probability in (("dev", 0.8), ("held_out", 0.6)):
        c = direct_report.by_split[split].confidence
        assert c.sample_count > 0
        assert c.mean_confidence == pytest.approx(probability)
        assert c.brier_score == pytest.approx((1 - probability) ** 2)
        assert c.calibration_error == pytest.approx(1 - probability)
        assert [p.threshold for p in c.coverage_curve] == [0, 0.2, 0.4, 0.6, 0.8, 1]
        selected = [r for r in direct_report.cases
                    if r.observation.run.problem_id in benchmark.manifest.splits[split]
                    and r.outcome == "selected"]
        assert c.sample_count == len(selected)
        for r in selected:
            assert r.observation.verification is not None
    held = load_benchmark(Path("evals"), split="held_out")
    held_report = run_benchmark(held, RuleBasedStrategy(), strategy_name="rules", strategy_version="1")
    assert set(held_report.by_split) == {"held_out"}
    assert held_report.summary == report.columns["rules"].report.by_split["held_out"]
    assert "ECE [held_out]" in comparison_text(report)
    assert "Confidence [held_out]" in report_text(direct_report)
    assert "fixed_reference" in comparison_json(report)
    assert "diagnostic_version" in report_json(direct_report)
    for request in direct.requests:
        assert "acceptable_option_ids" not in request["problem_json"]
    # Confidence differences never change original outcome/review scores.
    assert direct_report.summary.rates == report.columns["rules"].report.summary.rates
    dev = load_benchmark(Path("evals"), split="dev")
    assert set(run_benchmark(dev, RuleBasedStrategy(), strategy_name="rules",
                             strategy_version="1").by_split) == {"dev"}


def test_deterministic_baseline_reference_and_reports(benchmark: Benchmark) -> None:
    first = run_comparison(benchmark, entries(benchmark)[0], repeats=1)
    second = run_comparison(benchmark, entries(benchmark)[0], repeats=1)
    assert comparison_json(first) == comparison_json(second)
    c = first.columns["rules"].report.summary.confidence
    assert c.availability == "unavailable" and c.brier_score is None
    assert c.fixed_reference.sample_count == 16
    assert c.fixed_reference.brier_score == 0.25
    assert c.fixed_reference.calibration_error == 0.5
    assert all(case.observation.run.recommendation.confidence is None
               for case in first.columns["rules"].report.cases)
    tools = run_benchmark(benchmark, ToolAssistedStrategy(), strategy_name="tool-assisted",
                          strategy_version="1")
    assert tools.summary.confidence.availability == "unavailable"
    assert tools.summary.confidence.fixed_reference.sample_count == 19
