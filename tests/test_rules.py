from pathlib import Path

import pytest

from decisionlab.models import Constraint, DecisionProblem, Fact, Option, Recommendation
from decisionlab.strategies import DecisionStrategy, RuleBasedStrategy


def examples() -> list[DecisionProblem]:
    path = Path(__file__).resolve().parents[1] / "examples" / "routing.jsonl"
    return [DecisionProblem.model_validate_json(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("index,expected", [(0, "routine"), (1, "expedited"), (2, "specialist")])
def test_three_examples(index: int, expected: str) -> None:
    problem = examples()[index]
    strategy: DecisionStrategy = RuleBasedStrategy()
    before = problem.model_dump_json()
    result = strategy.decide(problem)
    assert result.status == "selected"
    assert result.selected_option_id == expected
    assert result.confidence is None
    assert not result.requested_review
    assert len(result.assessments) == len(problem.options)
    problem.validate_recommendation(result)
    assert problem.model_dump_json() == before


@pytest.mark.parametrize("value", [None, "true", 1, 0])
def test_invalid_fact_values_abstain(value: object) -> None:
    data = examples()[0].model_dump()
    data["facts"][0]["value"] = value
    result = RuleBasedStrategy().decide(DecisionProblem.model_validate(data))
    assert result.status == "abstained"
    assert result.requested_review and result.unknowns


def test_missing_fact_abstains() -> None:
    problem = examples()[0]
    problem.facts.pop()
    assert RuleBasedStrategy().decide(problem).status == "abstained"


@pytest.mark.parametrize("attributes", [
    {"route": "routine"},
    {"route": "routine", "capacity": -1},
    {"route": "routine", "capacity": True},
    {"route": "routine", "capacity": "2"},
    {"route": "invalid", "capacity": 2},
    {"route": "routine", "capacity": 2, "unsupported": True},
])
def test_invalid_attributes_abstain(attributes: dict[str, object]) -> None:
    data = examples()[0].model_dump()
    data["options"][0]["attributes"] = attributes
    result = RuleBasedStrategy().decide(DecisionProblem.model_validate(data))
    assert result.status == "abstained" and result.requested_review
    assert result.unknowns


def test_zero_capacity_does_not_fall_back_to_wrong_route() -> None:
    problem = examples()[1]
    problem.options[1].attributes["capacity"] = 0
    result = RuleBasedStrategy().decide(problem)
    assert result.status == "abstained" and result.selected_option_id is None
    assert result.requested_review
    assert all(a.eligibility == "ineligible" for a in result.assessments)


def test_ties_abstain_and_are_order_independent() -> None:
    problem = examples()[0]
    problem.options.append(Option(
        id="routine-2", description="Second routine queue",
        attributes={"route": "routine", "capacity": 3},
    ))
    first = RuleBasedStrategy().decide(problem)
    problem.options.reverse()
    assert first == RuleBasedStrategy().decide(problem)
    assert first.status == "abstained" and first.requested_review
    assert "Multiple" in first.explanation


def test_unknown_hard_constraint_abstains() -> None:
    problem = examples()[0]
    problem.constraints.append(Constraint(
        id="security", description="Security clearance required", kind="hard",
    ))
    result = RuleBasedStrategy().decide(problem)
    assert result.status == "abstained" and result.unknowns


def test_irrelevant_facts_and_soft_constraints_do_not_change_selection() -> None:
    problem = examples()[0]
    problem.facts.append(Fact(
        id="noise", statement="Distracting information", source="synthetic",
        value="irrelevant",
    ))
    problem.constraints.append(Constraint(
        id="preference", description="Unscored preference", kind="soft",
    ))
    result = RuleBasedStrategy().decide(problem)
    assert result.selected_option_id == "routine"
    assert all("noise" not in a.evidence_ids for a in result.assessments)


def test_repeated_calls_are_deterministic() -> None:
    problem = examples()[2]
    strategy = RuleBasedStrategy()
    assert strategy.decide(problem) == strategy.decide(problem)


def test_interface_accepts_an_unrelated_implementation() -> None:
    class OtherStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            return Recommendation(status="abstained", explanation=problem.description)

    strategy: DecisionStrategy = OtherStrategy()
    assert strategy.decide(examples()[0]).status == "abstained"
