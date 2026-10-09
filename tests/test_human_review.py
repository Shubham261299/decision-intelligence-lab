"""Portable annotation checks; saved run artifacts are read-only when available."""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from decisionlab.evaluation.human_review import HumanReviewRecord

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/reviews/experiment-3-first-pass-v1.json"


@pytest.fixture
def review_payload() -> dict[str, object]:
    data: object = json.loads(RECORD.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast(dict[str, object], data)


def first_item(payload: dict[str, object]) -> dict[str, object]:
    return cast(list[dict[str, object]], payload["reviewed_items"])[0]


def test_accepted_labels_and_uncertainty(review_payload: dict[str, object]) -> None:
    record = HumanReviewRecord.model_validate(review_payload)
    r003, r022 = record.reviewed_items
    assert [r003.review_id, r022.review_id] == ["R003", "R022"]
    assert r003.decision_target == r022.decision_target == "justified"
    assert r003.policy_compliance == r022.policy_compliance == "compliant"
    assert r003.reasoning_evidence == "partially_supported"
    assert r022.reasoning_evidence == "fully_supported"
    assert r003.review_requirement == "incorrect"
    assert r022.review_requirement == "correct"
    assert r003.decision_uncertainty == r022.decision_uncertainty == "low"
    assert r003.review_label_uncertainty == "moderate"
    assert r022.review_label_uncertainty == "low"
    assert "optional review is not explicitly prohibited" in r003.review_requirement_note
    assert "missing verified urgency" in r022.review_requirement_note
    assert all(item.identical_observation_count == 3 for item in record.reviewed_items)
    assert len(record.reviewed_items) == 2  # Repetitions are not independent judgments.


def test_explicit_provenance_and_score_separation(
    review_payload: dict[str, object],
) -> None:
    record = HumanReviewRecord.model_validate(review_payload)
    assert record.status == "accepted_first_pass"
    assert record.provenance.assessment_type == "user_accepted_assistant_assisted_first_pass"
    assert "User supplied and accepted" in record.provenance.acceptance_source
    assert not record.provenance.independent_expert_validation
    assert not record.provenance.blind_validation
    assert not record.provenance.review_key_consulted
    assert record.primary_score_effect == "none"
    assert record.primary_scores_unchanged
    assert any("not independent blind expert validation" in x for x in record.limitations)
    assert any("not six independent" in x for x in record.limitations)


@pytest.mark.parametrize("field,value", [
    ("decision_target", "successful"),
    ("reasoning_evidence", "partially correct"),
    ("policy_compliance", "accepted"),
    ("review_requirement", "optional"),
    ("decision_uncertainty", "calibrated"),
    ("review_label_uncertainty", 0.5),
    ("identical_observation_count", 0),
    ("identical_observation_count", True),
    ("response_sha256", "not-a-hash"),
    ("review_id", "route-010"),
])
def test_invalid_annotation_values_rejected(
    review_payload: dict[str, object], field: str, value: object,
) -> None:
    first_item(review_payload)[field] = value
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(review_payload)


@pytest.mark.parametrize("field", [
    "independent_expert_validation", "blind_validation", "review_key_consulted",
])
@pytest.mark.parametrize("value", [True, "false"])
def test_provenance_cannot_be_overstated_or_coerced(
    review_payload: dict[str, object], field: str, value: object,
) -> None:
    cast(dict[str, object], review_payload["provenance"])[field] = value
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(review_payload)


@pytest.mark.parametrize("field,value", [
    ("review_version", "human-review-v2"),
    ("experiment_version", "matched-json-v1"),
    ("primary_score_effect", "rescue_failures"),
    ("primary_scores_unchanged", False),
    ("primary_scores_unchanged", "true"),
    ("limitations", []),
    ("reviewed_items", []),
])
def test_version_and_separation_invariants(
    review_payload: dict[str, object], field: str, value: object,
) -> None:
    review_payload[field] = value
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(review_payload)


@pytest.mark.parametrize("scope", ["record", "annotation", "provenance"])
def test_primary_metrics_cannot_be_injected(
    review_payload: dict[str, object], scope: str,
) -> None:
    target = (
        first_item(review_payload) if scope == "annotation"
        else cast(dict[str, object], review_payload["provenance"])
        if scope == "provenance" else review_payload
    )
    target["outcome_correct"] = True
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(review_payload)


def test_duplicate_review_ids_rejected(review_payload: dict[str, object]) -> None:
    items = cast(list[dict[str, object]], review_payload["reviewed_items"])
    items[1]["review_id"] = items[0]["review_id"]
    with pytest.raises(ValidationError, match="Review IDs must be unique"):
        HumanReviewRecord.model_validate(review_payload)


@pytest.mark.parametrize("field", [
    "reasoning_evidence", "review_requirement_note", "uncertainty_note",
])
def test_semantic_dimensions_and_caveats_required(
    review_payload: dict[str, object], field: str,
) -> None:
    del first_item(review_payload)[field]
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(review_payload)


@pytest.mark.parametrize("path", ["/tmp/report.json", "../report.json", "runs\\report.json"])
def test_non_repository_source_paths_rejected(
    review_payload: dict[str, object], path: str,
) -> None:
    sources = cast(dict[str, dict[str, object]], review_payload["source_artifacts"])
    sources["review_packet"]["path"] = path
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(review_payload)


def test_round_trip_does_not_change_annotations(review_payload: dict[str, object]) -> None:
    before = json.dumps(review_payload, sort_keys=True)
    record = HumanReviewRecord.model_validate(review_payload)
    assert HumanReviewRecord.model_validate_json(record.model_dump_json()) == record
    assert json.dumps(review_payload, sort_keys=True) == before


def test_saved_sources_and_response_bindings_when_available(
    review_payload: dict[str, object],
) -> None:
    # runs/ is intentionally ignored. Portable contract tests above always run;
    # original source/hash checks additionally run wherever saved outputs exist.
    record = HumanReviewRecord.model_validate(review_payload)
    for source in record.source_artifacts.values():
        assert Path(source.path).name != "human-review-key.json"
        path = ROOT / source.path
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source.sha256
    packet_path = ROOT / record.source_artifacts["review_packet"].path
    if packet_path.exists():
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        original = {item["review_id"]: item for item in packet["items"]}
        for item in record.reviewed_items:
            raw = original[item.review_id]
            assert item.response_sha256 == hashlib.sha256(raw["raw_response"].encode()).hexdigest()
            assert item.identical_observation_count == raw["identical_observation_count"]
            assert raw["human_annotation"]["decision_target"] == "pending"
    comparison_path = ROOT / record.source_artifacts["primary_comparison"].path
    if comparison_path.exists():
        before = comparison_path.read_bytes()
        comparison = json.loads(before)
        for condition, correct, failures in [
            ("labelled-text", 42, 18), ("unconstrained-json", 48, 0),
        ]:
            summary = comparison["columns"][condition]["report"]["summary"]
            assert summary["rates"]["outcome_correctness"]["numerator"] == correct
            assert summary["counts"]["failed_runs"] == failures
        HumanReviewRecord.model_validate_json(record.model_dump_json())
        assert comparison_path.read_bytes() == before
