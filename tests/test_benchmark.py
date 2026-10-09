"""Validate benchmark artifacts, not strategy performance or evaluation scores."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from decisionlab.evaluation.models import EvaluationCase
from decisionlab.models import DecisionProblem

ROOT = Path(__file__).resolve().parents[1] / "evals"
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
RUBRICS = json.loads(
    (ROOT / "expectations" / "rubrics.json").read_text(encoding="utf-8")
)


def records(folder: str, split: str) -> list[dict]:
    path = ROOT / folder / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


PROBLEMS = {
    item["id"]: item
    for split in ("dev", "held_out")
    for item in records("cases", split)
}
EXPECTATIONS = {
    item["id"]: item
    for split in ("dev", "held_out")
    for item in records("expectations", split)
}


def test_counts_unique_ids_and_exact_split_alignment() -> None:
    assert MANIFEST["case_count"] == 30
    assert len(PROBLEMS) == len(EXPECTATIONS) == 30
    assert set(PROBLEMS) == set(EXPECTATIONS) == set(RUBRICS["cases"])
    assert not set(MANIFEST["splits"]["dev"]) & set(MANIFEST["splits"]["held_out"])
    for split, count in (("dev", 20), ("held_out", 10)):
        inputs = records("cases", split)
        answers = records("expectations", split)
        assert len(inputs) == len(answers) == count
        assert [p["id"] for p in inputs] == MANIFEST["splits"][split]
        assert [e["problem_id"] for e in answers] == MANIFEST["splits"][split]
        assert len({e["id"] for e in answers}) == count


@pytest.mark.parametrize("case_id", sorted(PROBLEMS))
def test_case_contracts_references_and_ground_truth(case_id: str) -> None:
    problem = DecisionProblem.model_validate(PROBLEMS[case_id])
    expectation = EvaluationCase.model_validate(EXPECTATIONS[case_id])
    expectation.validate_problem(problem)
    assert DecisionProblem.model_validate_json(problem.model_dump_json()) == problem
    rubric = RUBRICS["cases"][case_id]
    facts = {fact.id: fact for fact in problem.facts}
    assert set(rubric["required_evidence_ids"]) <= set(facts)
    assert set(rubric["required_constraint_ids"]) <= {
        c.id for c in problem.constraints
    }
    assert bool(rubric["review_reason"]) == expectation.expected_review
    assert rubric["reason"] in expectation.evidence_rubric

    # Integrity oracle for authored labels, independent of RuleBasedStrategy.
    unresolved = any(
        key not in facts or type(facts[key].value) is not bool
        for key in ("urgent", "specialist_required")
    )
    unresolved |= any("capacity" not in o.attributes for o in problem.options)
    approval = next((c for c in problem.constraints if c.id == "approval"), None)
    if approval is not None:
        unresolved |= facts["approval_granted"].value is not True
    eligible = set()
    if not unresolved:
        route = (
            "specialist" if facts["specialist_required"].value
            else "expedited" if facts["urgent"].value else "routine"
        )
        eligible = {
            o.id for o in problem.options
            if o.attributes["route"] == route and o.attributes["capacity"] > 0
        }
    assert set(expectation.acceptable_option_ids) == eligible
    assert expectation.expected_abstention == (not eligible)
    explicit_review = facts.get("review_required")
    assert expectation.expected_review == (
        not eligible or (explicit_review is not None and explicit_review.value is True)
    )


def test_categories_and_nontrivial_outcomes_present() -> None:
    required = {
        "clear", "hard_conflict", "multiple_acceptable", "missing_information",
        "abstention", "human_review", "distractor", "conflicting_evidence",
        "relevant_pair", "wording_pair",
    }
    tags = {tag for item in EXPECTATIONS.values() for tag in item["category_tags"]}
    assert tags == required
    assert sum(e["expected_abstention"] for e in EXPECTATIONS.values()) == 11
    assert sum(e["expected_review"] for e in EXPECTATIONS.values()) == 13
    assert sum(len(e["acceptable_option_ids"]) > 1 for e in EXPECTATIONS.values()) == 3
    # Both partitions exercise selection, abstention, review and ambiguity.
    for split in MANIFEST["splits"].values():
        subset = [EXPECTATIONS[case_id] for case_id in split]
        assert any(e["expected_abstention"] for e in subset)
        assert any(not e["expected_abstention"] for e in subset)
        assert any(len(e["acceptable_option_ids"]) > 1 for e in subset)


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}".lstrip(".")))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            label = child.get("id", str(index)) if isinstance(child, dict) else str(index)
            result.update(flatten(child, f"{prefix}.{label}"))
        return result
    return {prefix: value}


@pytest.mark.parametrize("pair", MANIFEST["pairs"], ids=lambda pair: pair["id"])
def test_pairs_change_only_declared_field_and_never_cross_splits(pair: dict) -> None:
    left_id, right_id = pair["members"]
    assert sum(
        set(pair["members"]) <= set(ids)
        for ids in MANIFEST["splits"].values()
    ) == 1
    left, right = flatten(PROBLEMS[left_id]), flatten(PROBLEMS[right_id])
    changed = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
    assert changed == {"id", pair["changed_field"]}
    left_answers = EXPECTATIONS[left_id]["acceptable_option_ids"]
    right_answers = EXPECTATIONS[right_id]["acceptable_option_ids"]
    if pair["kind"] == "wording":
        assert left_answers == right_answers
    else:
        assert left_answers != right_answers


def test_hashes_freeze_all_data_and_rubric_artifacts() -> None:
    expected_paths = {
        "cases/dev.jsonl", "cases/held_out.jsonl",
        "expectations/dev.jsonl", "expectations/held_out.jsonl",
        "expectations/rubrics.json",
    }
    assert set(MANIFEST["sha256"]) == expected_paths
    for relative, digest in MANIFEST["sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_labels_are_not_part_of_strategy_input_contract() -> None:
    forbidden = {
        "acceptable_option_ids", "expected_abstention", "expected_review",
        "evidence_rubric", "category_tags", "required_evidence_ids", "reason",
    }
    for raw in PROBLEMS.values():
        assert not {
            key.split(".")[-1] for key in flatten(raw)
        } & forbidden
        assert set(raw) == {"id", "description", "context", "facts", "constraints", "options"}
        with pytest.raises(ValidationError):
            DecisionProblem.model_validate({**raw, "expected_review": True})


def test_invalid_reference_label_is_rejected() -> None:
    case_id = "route-001"
    raw = {**EXPECTATIONS[case_id], "acceptable_option_ids": ["nonexistent"]}
    case = EvaluationCase.model_validate(raw)
    with pytest.raises(ValueError, match="Unknown"):
        case.validate_problem(DecisionProblem.model_validate(PROBLEMS[case_id]))


def test_no_duplicate_problems_ignoring_identifiers() -> None:
    normalized = [
        json.dumps({k: v for k, v in item.items() if k != "id"}, sort_keys=True)
        for item in PROBLEMS.values()
    ]
    assert len(set(normalized)) == 30


def test_version_and_manual_rubric_contract() -> None:
    assert MANIFEST["version"] == RUBRICS["version"] == "1.0.0"
    assert MANIFEST["schema_version"] == "1"
    assert MANIFEST["policy_version"] == "routing-benchmark-1"
    assert set(RUBRICS["evidence_support_scale"]) == {"0", "1", "2"}
    assert set(RUBRICS["review_scale"]) == {"0", "1"}
