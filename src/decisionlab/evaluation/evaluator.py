"""Read-only evaluation against reference outcomes; no strategy imports."""

from typing import Literal

from decisionlab.evaluation.models import CaseEvaluation, EvaluationCase, Observation
from decisionlab.models import DecisionProblem
from decisionlab.policy import selection_violations


def evaluate_observation(
    problem: DecisionProblem, expectation: EvaluationCase, observation: Observation,
    repetition: int = 1,
) -> CaseEvaluation:
    expectation.validate_problem(problem)
    run = observation.run
    if run.problem_id != problem.id:
        raise ValueError("Run problem ID does not match evaluator input")
    rec = run.recommendation
    outcome: Literal["selected", "abstained", "failure"] = "failure"
    if run.status == "success" and rec is not None:
        problem.validate_recommendation(rec)
        outcome = rec.status
    selected = outcome == "selected"
    abstained = outcome == "abstained"
    selection_correct = (
        selected and rec is not None
        and rec.selected_option_id in expectation.acceptable_option_ids
    )
    abstention_correct = abstained and expectation.expected_abstention
    system_review = (
        run.policy_result.requires_human_review if run.policy_result is not None else None
    )
    candidate = rec if rec is not None else observation.candidate
    violations = selection_violations(problem, candidate) if candidate is not None else []
    return CaseEvaluation(
        case_id=expectation.id, repetition=repetition, observation=observation,
        expected_abstention=expectation.expected_abstention,
        expected_review=expectation.expected_review,
        acceptable_option_ids=expectation.acceptable_option_ids,
        category_tags=expectation.category_tags,
        outcome=outcome,
        selection_correct=selection_correct if not expectation.expected_abstention else None,
        abstention_correct=abstention_correct if expectation.expected_abstention else None,
        outcome_correct=selection_correct or abstention_correct,
        strategy_review=rec.requested_review if rec is not None else None,
        system_review=system_review,
        review_correct=system_review is not None and system_review == expectation.expected_review,
        hard_constraint_violations=violations,
        evidence_rubric=expectation.evidence_rubric,
    )
