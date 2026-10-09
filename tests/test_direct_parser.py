import pytest

from decisionlab.errors import DecisionExecutionError
from decisionlab.models import Recommendation
from decisionlab.strategies.direct_parser import parse_recommendation

TEXT = """Decision: selected
Option: q-r
Confidence: 0.75
Human review: no
Explanation: Routine request with capacity.
Assessments: present
Assessment: q-r | eligible
Reason: Intake facts match and capacity is available.
Evidence: urgent, specialist_required
Constraints: capacity
Unknowns: none"""


def format_recommendation(rec: Recommendation) -> str:
    """Test-only serialization; the application never repairs or re-prompts."""
    lines = [
        f"Decision: {rec.status}", f"Option: {rec.selected_option_id or 'none'}",
        f"Confidence: {rec.confidence if rec.confidence is not None else 'unavailable'}",
        f"Human review: {'yes' if rec.requested_review else 'no'}",
        f"Explanation: {rec.explanation}",
        f"Assessments: {'present' if rec.assessments else 'none'}",
    ]
    for item in rec.assessments:
        lines.extend([
            f"Assessment: {item.option_id} | {item.eligibility}",
            f"Reason: {'; '.join(item.reasons)}",
            f"Evidence: {', '.join(item.evidence_ids) or 'none'}",
            f"Constraints: {', '.join(item.constraint_ids) or 'none'}",
        ])
    lines.append(f"Unknowns: {'present' if rec.unknowns else 'none'}")
    for item in rec.unknowns:
        lines.extend([
            f"Unknown: {item.description}",
            f"Affected options: {', '.join(item.affected_option_ids) or 'none'}",
            f"Material: {'yes' if item.material_to_decision else 'no'}",
        ])
    return "\n".join(lines)


def test_full_selected_contract_and_references() -> None:
    rec = parse_recommendation(TEXT)
    assert rec.status == "selected" and rec.selected_option_id == "q-r"
    assert rec.confidence == 0.75 and rec.requested_review is False
    assert rec.assessments[0].evidence_ids == ["urgent", "specialist_required"]
    assert rec.assessments[0].constraint_ids == ["capacity"]
    assert rec.unknowns == []


def test_abstention_unknowns_and_multiple_blocks() -> None:
    text = TEXT.replace("selected", "abstained").replace("Option: q-r", "Option: none")
    text = text.replace("0.75", "unavailable").replace("Human review: no", "Human review: yes")
    text = text.replace("Unknowns: none", """Assessment: q-e | unknown
Reason: Capacity missing.
Evidence: none
Constraints: none
Unknowns: present
Unknown: Capacity is unknown.
Affected options: q-e
Material: yes
Unknown: Optional note missing.
Affected options: none
Material: no""")
    rec = parse_recommendation(text)
    assert rec.status == "abstained" and rec.selected_option_id is None
    assert rec.confidence is None and rec.requested_review
    assert len(rec.assessments) == len(rec.unknowns) == 2
    assert rec.unknowns[0].material_to_decision
    assert not rec.unknowns[1].material_to_decision


def test_explicit_empty_assessments_and_whitespace() -> None:
    text = TEXT[:TEXT.index("Assessments:")] + "Assessments: none\nUnknowns: none"
    rec = parse_recommendation("\n  " + text.replace("\n", "  \n\n  ") + " \n")
    assert rec.assessments == []


@pytest.mark.parametrize("old,new", [
    ("Decision: selected", "Decision: selected or abstained"),
    ("Decision: selected", "Decision: Selected"),
    ("Option: q-r", "Option: none"),
    ("0.75", "75%"), ("0.75", "high"), ("0.75", "NaN"),
    ("0.75", "1.1"), ("0.75", "-0.1"), ("0.75", "1e-1"),
    ("Human review: no", "Human review: false"),
    ("Explanation: Routine request with capacity.", "Explanation: "),
    ("q-r | eligible", "q-r | eligible | ineligible"),
    ("q-r | eligible", "q-r | maybe"),
    ("Evidence: urgent, specialist_required", "Evidence: urgent,"),
    ("Evidence: urgent, specialist_required", "Evidence: urgent, urgent"),
    ("Evidence: urgent, specialist_required", "Evidence: none, urgent"),
    ("Constraints: capacity", "Constraints: "),
    ("Assessments: present", "Assessments: unknown"),
    ("Assessments: present", "Assessments: none"),
    ("Unknowns: none", "Unknowns: present"),
    ("Unknowns: none", "Unknowns: none\nAdditional explanation."),
])
def test_invalid_ambiguous_values_are_not_repaired(old: str, new: str) -> None:
    with pytest.raises(DecisionExecutionError, match="invalid_schema"):
        parse_recommendation(TEXT.replace(old, new))


@pytest.mark.parametrize("index", range(11))
def test_every_line_is_required(index: int) -> None:
    lines = TEXT.splitlines()
    del lines[index]
    with pytest.raises(DecisionExecutionError, match="invalid_schema"):
        parse_recommendation("\n".join(lines))


@pytest.mark.parametrize("text", [
    "", "I recommend q-r.", "{}", "null", "```text\n" + TEXT + "\n```",
    TEXT + "\nDecision: abstained", "Introductory prose.\n" + TEXT,
    TEXT.replace("Option: q-r", "Confidence: 0.75\nOption: q-r"),
    TEXT.replace("Unknowns: none", "Assessment: q-r | eligible\nReason: duplicate\nEvidence: none\nConstraints: none\nUnknowns: none"),
    TEXT.replace("Unknowns: none", "Unknowns: present\nUnknown: missing\nAffected options: q-r\nMaterial: maybe"),
])
def test_prose_json_duplicates_and_bad_unknowns_fail(text: str) -> None:
    with pytest.raises(DecisionExecutionError, match="invalid_schema"):
        parse_recommendation(text)


@pytest.mark.parametrize("confidence", ["0", "0.0", "1", "1.000", "unavailable"])
def test_confidence_boundaries(confidence: str) -> None:
    assert parse_recommendation(TEXT.replace("0.75", confidence)).confidence == (
        None if confidence == "unavailable" else float(confidence)
    )
