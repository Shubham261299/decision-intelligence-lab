import pytest

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import Constraint, Recommendation, Unknown
from decisionlab.policy import enforce_policy, selection_violations


def selected(option_id: str = "q-r") -> Recommendation:
    return Recommendation(
        status="selected", selected_option_id=option_id, explanation="Test selection",
    )


def test_valid_selection_needs_no_review(benchmark: Benchmark) -> None:
    result = enforce_policy(benchmark.problems[0], selected())
    assert not result.requires_human_review
    assert not result.detected_violations


def test_review_on_selection_preserves_recommendation(benchmark: Benchmark) -> None:
    problem = benchmark.problems[15]
    rec = selected()
    before = rec.model_dump_json()
    policy = enforce_policy(problem, rec)
    assert policy.requires_human_review
    assert "explicit_supervisor_review" in policy.review_reasons
    assert rec.model_dump_json() == before
    assert not rec.requested_review
    assert not policy.detected_violations


def test_any_tied_eligible_option_is_safe(benchmark: Benchmark) -> None:
    problem = benchmark.problems[8]
    for option_id in ("q-r", "q-r2"):
        policy = enforce_policy(problem, selected(option_id))
        assert not policy.requires_human_review
        assert not policy.detected_violations


def test_abstention_requires_review_even_without_strategy_flag(benchmark: Benchmark) -> None:
    rec = Recommendation(status="abstained", explanation="No decision")
    policy = enforce_policy(benchmark.problems[0], rec)
    assert policy.requires_human_review
    assert "strategy_abstained" in policy.review_reasons
    assert not policy.detected_violations


@pytest.mark.parametrize("index,option,violation", [
    (0, "q-e", "wrong_route"),
    (7, "q-e", "capacity_exhausted"),
    (10, "q-r", "unresolved_fact:urgent"),
    (12, "q-e", "unresolved_capacity:q-e"),
    (19, "q-r", "approval_not_granted"),
])
def test_selected_hard_violations_trigger_review(
    benchmark: Benchmark, index: int, option: str, violation: str,
) -> None:
    policy = enforce_policy(benchmark.problems[index], selected(option))
    assert violation in policy.detected_violations
    assert policy.requires_human_review


def test_approval_positive_path(benchmark: Benchmark) -> None:
    problem = benchmark.problems[19]
    problem.facts[-1].value = True
    assert not enforce_policy(problem, selected()).detected_violations


def test_unknown_hard_constraint_cannot_be_ignored(benchmark: Benchmark) -> None:
    problem = benchmark.problems[0]
    problem.constraints.append(Constraint(id="security", kind="hard", description="Required"))
    assert "unsupported_hard_constraint:security" in selection_violations(problem, selected())


def test_untrusted_classification_source_requires_review(benchmark: Benchmark) -> None:
    problem = benchmark.problems[0]
    problem.facts[0].source = "stale-note"
    assert "unresolved_fact:urgent" in enforce_policy(problem, selected()).detected_violations


def test_material_unknown_and_requested_review_cannot_be_suppressed(benchmark: Benchmark) -> None:
    rec = selected()
    rec.requested_review = True
    rec.unknowns = [Unknown(description="Unclear evidence", material_to_decision=True)]
    policy = enforce_policy(benchmark.problems[0], rec)
    assert "strategy_requested_review" in policy.review_reasons
    assert "strategy_reported_material_unknown" in policy.review_reasons


def test_unknown_selected_option_is_diagnostic(benchmark: Benchmark) -> None:
    assert "unknown_selected_option" in selection_violations(benchmark.problems[0], selected("missing"))
    with pytest.raises(ValueError):
        enforce_policy(benchmark.problems[0], selected("missing"))
