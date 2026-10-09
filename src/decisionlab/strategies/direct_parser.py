"""Strict labelled-text grammar; no repair, inference, or provider dependencies."""

import re

from pydantic import ValidationError

from decisionlab.errors import DecisionExecutionError
from decisionlab.models import OptionAssessment, Recommendation, Unknown

PARSER_VERSION = "direct-text-v1"


class ParserExecutionError(DecisionExecutionError):
    """Diagnostic stage only; primary failure code and parsing remain unchanged."""

    def __init__(self, stage: str, issues: tuple[str, ...] = ()) -> None:
        self.stage = stage
        self.issues = issues
        super().__init__("invalid_schema")


class _Reader:
    def __init__(self, text: str) -> None:
        # Blank lines and surrounding whitespace carry no decision semantics.
        self.lines = [line.strip() for line in text.splitlines() if line.strip()]
        self.position = 0

    def starts(self, label: str) -> bool:
        return self.position < len(self.lines) and self.lines[self.position].startswith(
            f"{label}:"
        )

    def take(self, label: str) -> str:
        if not self.starts(label):
            raise ValueError("Missing or out-of-order field")
        value = self.lines[self.position][len(label) + 1:].strip()
        self.position += 1
        if not value:
            raise ValueError("Empty field")
        return value


def _boolean(value: str) -> bool:
    if value not in ("yes", "no"):
        raise ValueError("Expected yes or no")
    return value == "yes"


def _references(value: str) -> list[str]:
    if value == "none":
        return []
    items = [item.strip() for item in value.split(",")]
    if any(not item or item == "none" for item in items):
        raise ValueError("Ambiguous reference list")
    if len(items) != len(set(items)):
        raise ValueError("Duplicate reference")
    return items


def _present(value: str) -> bool:
    if value not in ("present", "none"):
        raise ValueError("Expected present or none")
    return value == "present"


def parse_recommendation(text: str) -> Recommendation:
    """Require every field and consume the entire response, then validate domain."""
    try:
        reader = _Reader(text)
        status = reader.take("Decision")
        option = reader.take("Option")
        confidence_text = reader.take("Confidence")
        confidence: float | None = None
        if confidence_text != "unavailable":
            if re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", confidence_text) is None:
                raise ValueError("Expected confidence fraction")
            confidence = float(confidence_text)
        review = _boolean(reader.take("Human review"))
        explanation = reader.take("Explanation")
        assessments: list[OptionAssessment] = []
        if _present(reader.take("Assessments")):
            while reader.starts("Assessment"):
                parts = reader.take("Assessment").split("|")
                if len(parts) != 2:
                    raise ValueError("Expected option and eligibility")
                assessments.append(OptionAssessment.model_validate({
                    "option_id": parts[0].strip(), "eligibility": parts[1].strip(),
                    "reasons": [reader.take("Reason")],
                    "evidence_ids": _references(reader.take("Evidence")),
                    "constraint_ids": _references(reader.take("Constraints")),
                }))
            if not assessments:
                raise ValueError("Expected assessments")
        unknowns: list[Unknown] = []
        if _present(reader.take("Unknowns")):
            while reader.starts("Unknown"):
                unknowns.append(Unknown(
                    description=reader.take("Unknown"),
                    affected_option_ids=_references(reader.take("Affected options")),
                    material_to_decision=_boolean(reader.take("Material")),
                ))
            if not unknowns:
                raise ValueError("Expected unknowns")
        if reader.position != len(reader.lines):
            raise ValueError("Unexpected trailing content")
        return Recommendation.model_validate({
            "status": status, "selected_option_id": None if option == "none" else option,
            "confidence": confidence, "requested_review": review,
            "explanation": explanation, "assessments": assessments, "unknowns": unknowns,
        })
    except ValidationError as error:
        issues = tuple(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
            for item in error.errors(include_input=False, include_context=False)
        )
        raise ParserExecutionError("schema_domain_validation", issues) from None
    except (ValueError, TypeError):
        raise ParserExecutionError("parser_failure") from None
