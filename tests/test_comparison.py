import json
from pathlib import Path

import pytest

from decisionlab.comparison import (
    StrategyEntry, comparison_json, comparison_text, run_comparison, write_comparison,
)
from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem, Recommendation, Usage
from decisionlab.strategies import DirectLLMStrategy, RuleBasedStrategy, StructuredLLMStrategy
from test_direct_llm import FakeTextProvider, response as text_response
from test_direct_parser import format_recommendation
from test_structured_llm import FakeProvider, response as structured_response


def entries(
    benchmark: Benchmark, repeats: int = 1,
) -> tuple[list[StrategyEntry], FakeTextProvider, FakeProvider]:
    recommendations = [RuleBasedStrategy().decide(problem)
                       for problem in benchmark.problems for _ in range(repeats)]
    text = FakeTextProvider([text_response(format_recommendation(r)) for r in recommendations])
    structured = FakeProvider([structured_response(r.model_dump_json()) for r in recommendations])
    return [
        StrategyEntry("rules", "1", RuleBasedStrategy()),
        StrategyEntry("direct-llm", "1", DirectLLMStrategy(text), "mocked"),
        StrategyEntry("structured-llm", "1", StructuredLLMStrategy(structured), "mocked"),
    ], text, structured


def test_matched_three_strategy_pipeline_and_no_labels(benchmark: Benchmark) -> None:
    specs, direct, structured = entries(benchmark, repeats=2)
    before = benchmark.model_dump_json()
    report = run_comparison(benchmark, specs, repeats=2)
    assert benchmark.model_dump_json() == before
    assert report.case_ids == [p.id for p in benchmark.problems]
    assert report.execution_order == ["rules", "direct-llm", "structured-llm"]
    assert report.comparability_warnings == []
    assert len(direct.requests) == len(structured.requests) == 60
    for index, problem in enumerate(benchmark.problems):
        for repetition in range(2):
            for requests in (direct.requests, structured.requests):
                request = requests[index * 2 + repetition]
                assert json.loads(request["problem_json"]) == problem.model_dump(mode="json")
                assert "acceptable_option_ids" not in request["problem_json"]
                assert "evidence_rubric" not in request["problem_json"]
    for name, column in report.columns.items():
        summary = column.report.summary
        assert summary.rates["outcome_correctness"].value == 0.9
        assert summary.rates["selection_correctness"].numerator == 32
        assert summary.rates["abstention_correctness"].numerator == 22
        assert summary.rates["human_review_recall"].value == 1
        assert summary.rates["repeated_decision_agreement"].numerator == 30
        assert column.invalid_output_rate.numerator == 0
        assert column.valid_abstentions == 28 and column.unnecessary_abstentions == 6
        assert column.wrong_selected_decisions == 0
        assert column.report.manifest_sha256 == report.manifest_sha256
        if name != "rules":
            assert column.mode == "mocked"
            assert column.metadata_profiles[0].model_id == "model-snapshot"
            assert column.metadata_profiles[0].prompt_version
            assert summary.total_input_tokens == 4800
            assert summary.total_output_tokens == 2400
            assert column.report.cases[0].observation.run.metadata.provider_response_id == "resp-test"
        else:
            assert summary.total_input_tokens is None
        assert summary.mean_evidence_support is None
        assert summary.confidence.brier_score is None


def test_deterministic_serialization_and_files(benchmark: Benchmark, tmp_path: Path) -> None:
    first = run_comparison(benchmark, entries(benchmark)[0])
    second = run_comparison(benchmark, entries(benchmark)[0])
    assert comparison_json(first) == comparison_json(second)
    assert comparison_text(first) == comparison_text(second)
    write_comparison(first, tmp_path)
    assert (tmp_path / "comparison.json").read_bytes() == comparison_json(first).encode()
    assert (tmp_path / "comparison.txt").read_bytes() == comparison_text(first).encode()
    assert "| Overall correctness | 27/30" in comparison_text(first)
    assert "| Execution mode | offline | mocked | mocked |" in comparison_text(first)
    assert "| Repeated agreement | 0/0 (n/a)" in comparison_text(first)


def test_wrong_abstained_violating_and_failed_are_distinct(benchmark: Benchmark) -> None:
    specs, direct, _ = entries(benchmark)
    wrong = RuleBasedStrategy().decide(benchmark.problems[0])
    wrong.selected_option_id = "q-e"
    direct.events[0] = text_response(format_recommendation(wrong))
    direct.events[1] = text_response("I choose q-r.")
    direct.events[2] = DecisionExecutionError("provider_timeout")
    bad_refs = RuleBasedStrategy().decide(benchmark.problems[3])
    bad_refs.assessments[0].evidence_ids = ["invented"]
    direct.events[3] = text_response(format_recommendation(bad_refs))
    report = run_comparison(benchmark, specs)
    column = report.columns["direct-llm"]
    assert column.wrong_selected_decisions == 1
    assert column.valid_abstentions == 14 and column.unnecessary_abstentions == 3
    assert column.invalid_output_rate.numerator == 2
    assert column.format_failure_rate.numerator == 1
    assert column.report.summary.counts["failed_runs"] == 3
    assert column.failure_codes == {"invalid_references": 1, "invalid_schema": 1, "provider_timeout": 1}
    case = column.report.cases[0]
    assert case.outcome == "selected" and not case.outcome_correct
    assert "wrong_route" in case.hard_constraint_violations
    assert column.report.cases[1].outcome == "failure"
    assert column.report.cases[1].observation.run.recommendation is None
    assert column.report.summary.rates["evidence_citation_validity"].value < 1
    assert report.columns["rules"].report.summary.rates["outcome_correctness"].value == 0.9


def test_confidence_cost_and_missing_measurement_counts(benchmark: Benchmark) -> None:
    specs, direct, _ = entries(benchmark)
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.confidence = 0.8
    event = text_response(format_recommendation(rec))
    event.usage = Usage(input_tokens=80, output_tokens=40, cost_usd=0.01)
    direct.events[0] = event
    report = run_comparison(benchmark, specs)
    summary = report.columns["direct-llm"].report.summary
    assert summary.confidence.sample_count == 1
    assert summary.confidence.brier_score == pytest.approx(0.04)
    assert summary.cost_sample_count == 1 and summary.total_cost_usd == 0.01
    assert report.columns["rules"].report.summary.total_cost_usd is None
    assert "| Cost samples | 0 | 1 | 0 |" in comparison_text(report)


def test_repeated_failure_prevents_agreement(benchmark: Benchmark) -> None:
    specs, direct, _ = entries(benchmark, repeats=2)
    direct.events[1] = text_response("ambiguous")
    report = run_comparison(benchmark, specs, repeats=2)
    agreement = report.columns["direct-llm"].report.summary.rates["repeated_decision_agreement"]
    assert agreement.numerator == 29 and agreement.denominator == 30


def test_live_guard_happens_before_any_strategy_execution(benchmark: Benchmark) -> None:
    class Counting:
        calls = 0

        def decide(self, problem: DecisionProblem) -> Recommendation:
            self.calls += 1
            return RuleBasedStrategy().decide(problem)

    strategy = Counting()
    with pytest.raises(ValueError, match="allow-live"):
        run_comparison(benchmark, [StrategyEntry("rules", "1", strategy),
                                  StrategyEntry("llm", "1", strategy, "live")])
    assert strategy.calls == 0
    report = run_comparison(benchmark, [StrategyEntry("mock-live", "1", strategy, "live")], allow_live=True)
    assert strategy.calls == 30 and report.columns["mock-live"].mode == "live"


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_bad_repeat_count(benchmark: Benchmark, repeats) -> None:
    with pytest.raises(ValueError, match="Repeats"):
        run_comparison(benchmark, entries(benchmark)[0], repeats=repeats)


def test_empty_duplicate_and_invalid_identity(benchmark: Benchmark) -> None:
    for specs in ([], [StrategyEntry("rules", "1", RuleBasedStrategy())] * 2,
                  [StrategyEntry(" ", "1", RuleBasedStrategy())],
                  [StrategyEntry("rules", "", RuleBasedStrategy())]):
        with pytest.raises(ValueError):
            run_comparison(benchmark, specs)


def test_strategy_mutation_does_not_change_inputs_for_later_columns(benchmark: Benchmark) -> None:
    class Mutating:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            rec = RuleBasedStrategy().decide(problem)
            problem.facts.clear()
            return rec

    report = run_comparison(benchmark, [StrategyEntry("mutating", "1", Mutating()),
                                      StrategyEntry("rules", "1", RuleBasedStrategy())])
    assert all(p.facts for p in benchmark.problems)
    assert report.columns["rules"].report.summary.rates["outcome_correctness"].value == 0.9


def test_different_model_configuration_is_disclosed(benchmark: Benchmark) -> None:
    specs, _, structured = entries(benchmark)
    structured.model_id = "different-requested-model"
    report = run_comparison(benchmark, specs)
    assert report.comparability_warnings
    assert "profiles differ" in comparison_text(report)


def test_different_acceptable_ties_are_correct_but_not_exact_agreement(benchmark: Benchmark) -> None:
    specs, direct, _ = entries(benchmark, repeats=2)
    index = next(i for i, p in enumerate(benchmark.problems)
                 if len(benchmark.expectations[p.id].acceptable_option_ids) > 1)
    problem = benchmark.problems[index]
    # Labels are used to author mocked responses, never sent in strategy requests.
    options = benchmark.expectations[problem.id].acceptable_option_ids
    for repetition in range(2):
        rec = RuleBasedStrategy().decide(problem)
        rec.status = "selected"
        rec.selected_option_id = options[repetition]
        rec.requested_review = False
        direct.events[index * 2 + repetition] = text_response(format_recommendation(rec))
    report = run_comparison(benchmark, specs, repeats=2)
    column = report.columns["direct-llm"]
    cases = [c for c in column.report.cases if c.observation.run.problem_id == problem.id]
    assert all(c.outcome_correct for c in cases)
    assert column.report.summary.rates["repeated_decision_agreement"].numerator == 29


def test_refusal_is_failure_with_unknown_output_validity(benchmark: Benchmark) -> None:
    specs, direct, _ = entries(benchmark)
    direct.events[0] = text_response("", "refused")
    report = run_comparison(benchmark, specs)
    column = report.columns["direct-llm"]
    assert column.failure_codes == {"provider_refusal": 1}
    assert column.invalid_output_rate.numerator == 0
    assert column.report.cases[0].observation.schema_valid is None
    assert column.report.summary.counts["failed_runs"] == 1


def test_empty_benchmark_invalid_mode_and_input_fingerprint(benchmark: Benchmark) -> None:
    with pytest.raises(ValueError, match="benchmark cases"):
        run_comparison(benchmark.model_copy(update={"problems": []}), entries(benchmark)[0])
    with pytest.raises(ValueError, match="execution mode"):
        run_comparison(benchmark, [StrategyEntry("rules", "1", RuleBasedStrategy(), "invalid")])
    first = run_comparison(benchmark, [StrategyEntry("rules", "1", RuleBasedStrategy())])
    changed = benchmark.model_copy(deep=True)
    changed.problems[0].context += " Decorative text."
    second = run_comparison(changed, [StrategyEntry("rules", "1", RuleBasedStrategy())])
    assert first.input_sha256 != second.input_sha256
    assert len(first.input_sha256) == 64
