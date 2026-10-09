"""Offline contract tests, including invalid and ambiguous boundary data."""

import math

import pytest
from pydantic import ValidationError

from decisionlab.evaluation.models import EvaluationCase, EvaluationResult
from decisionlab.models import (
    Constraint,
    DecisionProblem,
    DecisionRun,
    Fact,
    Option,
    OptionAssessment,
    PolicyResult,
    Recommendation,
    RunError,
    RunMetadata,
    Unknown,
    Usage,
)


@pytest.fixture
def problem() -> DecisionProblem:
    return DecisionProblem(
        id="routing-1",
        description="Route a request",
        facts=[Fact(id="urgent", statement="Urgent request", value=True, source="fixture")],
        constraints=[Constraint(id="capacity", description="Queue has capacity", kind="hard")],
        options=[Option(id="expedited", description="Expedited queue", attributes={"capacity": 2})],
    )


def recommendation(**changes: object) -> Recommendation:
    data: dict[str, object] = {
        "status": "selected",
        "selected_option_id": "expedited",
        "explanation": "Urgency supports expedited handling",
        "assessments": [{
            "option_id": "expedited", "eligibility": "eligible",
            "reasons": ["Capacity available"], "evidence_ids": ["urgent"],
            "constraint_ids": ["capacity"],
        }],
    }
    data.update(changes)
    return Recommendation.model_validate(data)


def test_problem_and_recommendation_json_round_trip(problem: DecisionProblem) -> None:
    assert DecisionProblem.model_validate_json(problem.model_dump_json()) == problem
    result = recommendation(confidence=0.8)
    assert Recommendation.model_validate_json(result.model_dump_json()) == result
    problem.validate_recommendation(result)


@pytest.mark.parametrize("field", ["facts", "constraints", "options"])
def test_duplicate_entity_ids_rejected(problem: DecisionProblem, field: str) -> None:
    data = problem.model_dump()
    data[field].append(data[field][0])
    with pytest.raises(ValidationError, match="unique"):
        DecisionProblem.model_validate(data)


def test_problem_requires_available_options(problem: DecisionProblem) -> None:
    with pytest.raises(ValidationError):
        DecisionProblem.model_validate({**problem.model_dump(), "options": []})


@pytest.mark.parametrize("value", ["", "   ", "\n", 123])
def test_invalid_identifiers_rejected(value: object) -> None:
    with pytest.raises(ValidationError):
        Option.model_validate({"id": value, "description": "Queue"})


def test_unknown_fields_rejected_at_nested_boundary(problem: DecisionProblem) -> None:
    data = problem.model_dump()
    data["facts"][0]["secret"] = "unexpected"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DecisionProblem.model_validate(data)


@pytest.mark.parametrize("value", [True, 3, 0.5, "high"])
def test_fact_preserves_scalar_types(value: object) -> None:
    fact = Fact.model_validate({"id": "f", "statement": "Value", "source": "fixture", "value": value})
    assert type(fact.value) is type(value)


@pytest.mark.parametrize("value", [[], {}, math.inf, math.nan])
def test_fact_rejects_non_scalar_or_nonfinite_values(value: object) -> None:
    with pytest.raises(ValidationError):
        Fact.model_validate({"id": "f", "statement": "Value", "source": "fixture", "value": value})


@pytest.mark.parametrize("changes", [
    {"selected_option_id": None},
    {"status": "abstained"},
    {"status": "failure"},
])
def test_invalid_selection_state_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        recommendation(**changes)


def test_abstention_has_no_selection_or_invented_confidence() -> None:
    result = Recommendation(status="abstained", explanation="Missing material fact", requested_review=True)
    assert result.selected_option_id is None
    assert result.confidence is None


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, math.inf])
def test_confidence_rejects_invalid_numbers(confidence: float) -> None:
    with pytest.raises(ValidationError):
        recommendation(confidence=confidence)


@pytest.mark.parametrize("confidence", [0, 1, None])
def test_confidence_boundaries(confidence: float | None) -> None:
    assert recommendation(confidence=confidence).confidence == confidence


@pytest.mark.parametrize("field", ["option_id", "evidence_ids", "constraint_ids"])
def test_assessment_references_resolve(problem: DecisionProblem, field: str) -> None:
    data = recommendation().model_dump()
    data["assessments"][0][field] = "missing" if field == "option_id" else ["missing"]
    with pytest.raises(ValueError, match="Unknown"):
        problem.validate_recommendation(Recommendation.model_validate(data))


def test_selected_option_reference_resolves(problem: DecisionProblem) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        problem.validate_recommendation(recommendation(selected_option_id="missing"))


def test_unknown_affected_options_resolve(problem: DecisionProblem) -> None:
    result = recommendation(unknowns=[Unknown(description="Capacity unclear", affected_option_ids=["missing"], material_to_decision=True)])
    with pytest.raises(ValueError, match="Unknown"):
        problem.validate_recommendation(result)


def test_duplicate_assessments_rejected() -> None:
    assessment = recommendation().assessments[0]
    with pytest.raises(ValidationError, match="unique"):
        recommendation(assessments=[assessment, assessment])


def test_assessment_requires_reason() -> None:
    with pytest.raises(ValidationError):
        OptionAssessment(option_id="o", eligibility="unknown", reasons=[])


def test_structural_validation_does_not_judge_eligibility(problem: DecisionProblem) -> None:
    data = recommendation().model_dump()
    data["assessments"][0]["eligibility"] = "ineligible"
    problem.validate_recommendation(Recommendation.model_validate(data))


@pytest.mark.parametrize("required,reasons", [(True, []), (False, ["Missing evidence"])])
def test_review_reasons_match_review_state(required: bool, reasons: list[str]) -> None:
    with pytest.raises(ValidationError):
        PolicyResult(requires_human_review=required, review_reasons=reasons, policy_version="1")


def test_successful_run_keeps_review_sources_separate() -> None:
    run = DecisionRun(
        id="run-1", problem_id="routing-1", status="success",
        metadata=RunMetadata(strategy="rules", strategy_version="1"),
        recommendation=recommendation(requested_review=False),
        policy_result=PolicyResult(requires_human_review=True, review_reasons=["Policy requires review"], policy_version="1"),
        usage=Usage(input_tokens=0), latency_ms=0,
    )
    assert DecisionRun.model_validate_json(run.model_dump_json()) == run
    assert run.recommendation is not None and not run.recommendation.requested_review
    assert run.policy_result is not None and run.policy_result.requires_human_review


def test_failed_run_is_not_abstention() -> None:
    run = DecisionRun(id="r", problem_id="p", status="failure", metadata=RunMetadata(strategy="llm", strategy_version="1"), error=RunError(code="invalid_output", message="Schema rejected"))
    assert run.recommendation is None


def test_valid_abstention_is_a_successful_execution() -> None:
    run = DecisionRun(
        id="r",
        problem_id="p",
        status="success",
        metadata=RunMetadata(strategy="rules", strategy_version="1"),
        recommendation=Recommendation(
            status="abstained",
            explanation="Missing capacity information",
            requested_review=True,
        ),
        policy_result=PolicyResult(
            requires_human_review=True,
            review_reasons=["Abstained"],
            policy_version="1",
        ),
    )
    assert run.error is None
    assert run.recommendation is not None
    assert run.recommendation.status == "abstained"


def test_mutable_defaults_are_independent() -> None:
    first = Option(id="first", description="First queue")
    second = Option(id="second", description="Second queue")
    first.attributes["capacity"] = 2
    assert second.attributes == {}


@pytest.mark.parametrize("changes", [
    {"status": "success"},
    {"status": "failure"},
    {"status": "failure", "error": {"code": "bad", "message": "Invalid"}, "recommendation": {"status": "abstained", "explanation": "Missing facts"}},
    {"status": "success", "recommendation": {"status": "abstained", "explanation": "Missing facts"}, "policy_result": {"requires_human_review": True, "review_reasons": ["Abstained"], "policy_version": "1"}, "error": {"code": "bad", "message": "Invalid"}},
])
def test_inconsistent_run_outcomes_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DecisionRun.model_validate({"id": "r", "problem_id": "p", "metadata": {"strategy": "rules", "strategy_version": "1"}, **changes})


@pytest.mark.parametrize("data", [{"input_tokens": -1}, {"output_tokens": True}, {"cost_usd": -0.1}, {"cost_usd": math.inf}])
def test_usage_rejects_invalid_values(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Usage.model_validate(data)


def expectation(**changes: object) -> EvaluationCase:
    return EvaluationCase.model_validate({"id": "case-1", "problem_id": "routing-1", "acceptable_option_ids": ["expedited"], "expected_abstention": False, "expected_review": False, "evidence_rubric": "Cite urgency and capacity", **changes})


def test_expectations_link_to_problem_without_leaking_into_input(problem: DecisionProblem) -> None:
    expectation().validate_problem(problem)
    assert "acceptable_option_ids" not in problem.model_dump()
    assert "acceptable_option_ids" not in DecisionProblem.model_json_schema()["properties"]


@pytest.mark.parametrize("changes", [{"problem_id": "other"}, {"acceptable_option_ids": ["missing"]}])
def test_expectation_references_resolve(problem: DecisionProblem, changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        expectation(**changes).validate_problem(problem)


@pytest.mark.parametrize("changes", [{"acceptable_option_ids": []}, {"expected_abstention": True}, {"acceptable_option_ids": ["expedited", "expedited"]}])
def test_invalid_expectations_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        expectation(**changes)


def test_abstention_expectation_and_result_round_trip() -> None:
    case = expectation(expected_abstention=True, acceptable_option_ids=[], expected_review=True)
    assert EvaluationCase.model_validate_json(case.model_dump_json()) == case
    result = EvaluationResult(case_id=case.id, run_id="r", metrics={"valid": True, "accuracy": None}, diagnostic_failures=["Execution failed"])
    assert EvaluationResult.model_validate_json(result.model_dump_json()) == result


def test_schema_generation() -> None:
    for model in (DecisionProblem, Recommendation, PolicyResult, DecisionRun, EvaluationCase, EvaluationResult):
        assert model.model_json_schema()["additionalProperties"] is False
