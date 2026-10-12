"""Read-only saved-evidence adapters. No execution, scoring or provider calls."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Generic, TypeVar

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, field_validator

from decisionlab.comparison import ComparisonReport
from decisionlab.evaluation.models import Rate

ROOT = Path(__file__).resolve().parents[1]
Model = TypeVar("Model", bound=BaseModel)
Cell = str | int | float | None

COMPARISONS = {
    "Strategy comparison": {
        "dev": "runs/ollama-first/dev/comparison.json",
        "held_out": "runs/ollama-first/held-out/comparison.json",
    },
    "JSON enforcement": {"dev": "runs/matched-json/dev/comparison.json"},
    "Output representation": {"dev": "runs/representation/dev-20261009/comparison.json"},
}
DIRECTORIES = {
    "Strategy comparison": "runs/ollama-first",
    "JSON enforcement": "runs/matched-json",
    "Output representation": "runs/representation/dev-20261009",
}


class EnforcementPairedAnalysis(BaseModel):
    """Historical Experiment 2 counts; remaining fields belong to its archive."""

    paired_counts: dict[str, Annotated[int, Field(strict=True, ge=0)]]

    @field_validator("paired_counts")
    @classmethod
    def require_recorded_measures(cls, counts: dict[str, int]) -> dict[str, int]:
        required = {"both_correct", "neither_correct", "unconstrained_only_correct",
                    "enforced_only_correct", "both_candidates_available", "identical_complete_candidates"}
        if not required.issubset(counts):
            raise ValueError("Missing paired measures")
        if counts["identical_complete_candidates"] > counts["both_candidates_available"]:
            raise ValueError("Invalid candidate agreement denominator")
        return counts


@dataclass(frozen=True)
class Evidence(Generic[Model]):
    path: str
    data: Model | None
    error: str | None = None


def read_model(root: Path, relative: str, model: type[Model]) -> Evidence[Model]:
    try:
        data = model.model_validate_json((root / relative).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError):
        return Evidence(relative, None, "Unavailable evidence: file missing or unreadable")
    except ValidationError:
        return Evidence(relative, None, "Unavailable evidence: malformed or unsupported report schema")
    return Evidence(relative, data)


def read_document(root: Path, relative: str) -> tuple[dict[str, JsonValue] | None, str | None]:
    """Optional protocols/audits have different schemas; do not guess their fields."""
    try:
        text = (root / relative).read_text(encoding="utf-8-sig")
        return TypeAdapter(dict[str, JsonValue]).validate_json(text), None
    except (OSError, UnicodeError):
        return None, "Unavailable evidence: file missing or unreadable"
    except ValidationError:
        return None, "Unavailable evidence: malformed JSON document"


def load_comparison(root: Path, relative: str) -> Evidence[ComparisonReport]:
    loaded = read_model(root, relative, ComparisonReport)
    if loaded.data is not None:
        report = loaded.data
        expected = {(case_id, rep) for case_id in report.case_ids
                    for rep in range(1, report.repeats + 1)}
        for column in report.columns.values():
            cases = column.report.cases
            actual = {(case.case_id, case.repetition) for case in cases}
            counts = column.report.summary.counts
            if (not expected or actual != expected or len(cases) != len(expected)
                    or counts.get("total") != len(cases)
                    or column.report.split != report.split):
                return Evidence(relative, None, "Unavailable evidence: inconsistent case/split inventory")
    return loaded


def rate_text(rate: Rate | None) -> str:
    if rate is None:
        return "Unavailable"
    if rate.value is None:
        return f"Not assessable ({rate.numerator}/{rate.denominator})"
    return f"{rate.numerator}/{rate.denominator} ({rate.value:.1%})"


def metric_rows(report: ComparisonReport) -> list[dict[str, Cell]]:
    metrics = [
        ("Overall correctness", "outcome_correctness"),
        ("Selection correctness", "selection_correctness"),
        ("Required-abstention correctness", "abstention_correctness"),
        ("Execution failures", "execution_failure_rate"),
        ("Schema validity", "schema_validity"),
        ("Contextual reference validity", "reference_validity"),
        ("Evidence citation validity", "evidence_citation_validity"),
        ("Hard-policy violations", "hard_constraint_violation_rate"),
        ("System-review precision", "human_review_precision"),
        ("System-review recall", "human_review_recall"),
        ("Repeated decision/review agreement", "repeated_decision_agreement"),
    ]
    return [
        {"Metric": label, **{
            name: rate_text(column.report.summary.rates.get(key))
            for name, column in report.columns.items()
        }} for label, key in metrics
    ]


def correctness_chart(report: ComparisonReport) -> list[dict[str, Cell]]:
    return [
        {"Condition": name, "Correctness (%)": rate.value * 100}
        for name, column in report.columns.items()
        if (rate := column.report.summary.rates.get("outcome_correctness")) is not None
        and rate.value is not None
    ]


def case_rows(report: ComparisonReport) -> list[dict[str, Cell]]:
    return [
        {"Condition": name, "Case": case.case_id, "Repeat": case.repetition,
         "Outcome": case.outcome, "Correct": "Yes" if case.outcome_correct else "No",
         "Accepted option": case.observation.run.recommendation.selected_option_id
         if case.observation.run.recommendation else None,
         "Failure": case.observation.run.error.code if case.observation.run.error else "None",
         "Policy violations": ", ".join(case.hard_constraint_violations) or "None",
         "Verifier": case.observation.verification.disposition
         if case.observation.verification else "Unavailable"}
        for name, column in report.columns.items() for case in column.report.cases
    ]
