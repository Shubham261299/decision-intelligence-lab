import json

import pytest

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.metrics import summarize
from decisionlab.evaluation.models import Rate
from decisionlab.evaluation.report import (
    EvaluationReport, build_report, report_json, report_text, write_report,
)
from decisionlab.runner import run_benchmark
from decisionlab.strategies import RuleBasedStrategy


def baseline(benchmark: Benchmark, repeats: int = 1) -> EvaluationReport:
    return run_benchmark(benchmark, RuleBasedStrategy(), strategy_name="rules", strategy_version="1", repeats=repeats)


def test_full_baseline_results(benchmark: Benchmark) -> None:
    report = baseline(benchmark)
    summary = report.summary
    assert summary.counts["total"] == 30
    assert summary.counts["failed_runs"] == 0
    assert summary.rates["outcome_correctness"].value == 0.9
    assert summary.rates["selection_correctness"].numerator == 16
    assert summary.rates["selection_correctness"].denominator == 19
    assert summary.rates["abstention_correctness"].value == 1
    assert summary.rates["decision_coverage"].numerator == 16
    assert summary.rates["selected_output_accuracy"].value == 1
    assert summary.counts["review_false_positive"] == 3
    assert summary.counts["review_false_negative"] == 0
    assert summary.rates["human_review_precision"].value == 13 / 16
    assert summary.rates["human_review_recall"].value == 1
    assert summary.rates["strategy_review_accuracy"].numerator == 25
    assert summary.rates["human_review_accuracy"].numerator == 27
    assert summary.rates["hard_constraint_violation_rate"].value == 0
    assert summary.rates["schema_validity"].value == 1
    assert summary.rates["reference_validity"].value == 1
    assert summary.rates["evidence_citation_validity"].value == 1
    assert summary.confidence.sample_count == 0
    assert summary.confidence.brier_score is None
    assert summary.mean_latency_ms is None
    assert summary.total_cost_usd is None
    assert report.pair_rates["relevant"].numerator == 4
    assert report.pair_rates["wording"].numerator == 2
    assert report.by_split["dev"].rates["outcome_correctness"].numerator == 18
    assert report.by_split["held_out"].rates["outcome_correctness"].numerator == 9
    assert report.by_category["multiple_acceptable"].rates["outcome_correctness"].value == 0


def test_deterministic_serialization_and_typed_round_trip(benchmark: Benchmark, tmp_path) -> None:
    first, second = baseline(benchmark), baseline(benchmark)
    assert report_json(first) == report_json(second)
    assert EvaluationReport.model_validate_json(report_json(first)) == first
    write_report(first, tmp_path)
    assert (tmp_path / "report.json").read_bytes() == report_json(first).encode()
    assert (tmp_path / "report.txt").read_bytes() == report_text(first).encode()
    assert json.loads(report_json(first))["report_version"] == "1"


def test_report_requires_complete_unique_results(benchmark: Benchmark) -> None:
    report = baseline(benchmark)
    for results in (report.cases[:-1], report.cases + report.cases[:1]):
        with pytest.raises(ValueError):
            build_report(benchmark, results, "rules", "1", 1)


def test_repeated_failures_do_not_count_as_agreement(benchmark: Benchmark) -> None:
    report = baseline(benchmark, 2)
    group = [r for r in report.cases if r.case_id == "route-001"]
    for result in group:
        result.outcome = "failure"
    assert summarize(group).rates["repeated_decision_agreement"].numerator == 0


def test_pair_correctness_requires_both_outcomes(benchmark: Benchmark) -> None:
    report = baseline(benchmark)
    report.cases[0].outcome_correct = False
    rebuilt = build_report(benchmark, report.cases, "rules", "1", 1)
    assert rebuilt.pair_rates["relevant"].numerator == 3


@pytest.mark.parametrize("numerator,denominator,value", [(0, 0, None), (1, 2, 0.5)])
def test_rate_round_trip(numerator: int, denominator: int, value: float | None) -> None:
    rate = Rate(numerator=numerator, denominator=denominator)
    assert rate.value == value
    assert Rate.model_validate_json(rate.model_dump_json()) == rate


def test_inconsistent_rate_is_rejected() -> None:
    with pytest.raises(ValueError):
        Rate(numerator=2, denominator=1)
    with pytest.raises(ValueError):
        Rate(numerator=1, denominator=2, value=1)
