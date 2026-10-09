"""Read-only diagnostics; explicit target matching never rescues primary failures."""

import json
import re
from typing import Literal

from decisionlab.evaluation.models import CaseEvaluation
from decisionlab.models import DecisionProblem, DomainModel
from decisionlab.strategies.representation import RawResponse, Representation

FailureStage = Literal[
    "none", "provider_api_failure", "empty_incomplete_refusal",
    "serialization_format_failure", "parser_failure", "schema_domain_validation",
    "invalid_option_reference", "ambiguous",
]
DecisionDiagnosis = Literal[
    "not_failed", "explicit_incorrect_decision", "explicit_apparently_correct_rejected",
    "ambiguous",
]


class ExplicitTarget(DomainModel):
    status: Literal["selected", "abstained"]
    option_id: str | None


class Diagnosis(DomainModel):
    rubric_version: str = "representation-diagnosis-v1"
    failure_stage: FailureStage
    validation_issues: list[str]
    decision_diagnosis: DecisionDiagnosis
    explicit_target: ExplicitTarget | None
    benchmark_target_matches: bool | None
    requires_human_judgment: bool
    semantic_support: Literal["not_assessed"] = "not_assessed"
    primary_score_changed: Literal[False] = False
    notes: list[str]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def explicit_target(raw: str | None, condition: Representation) -> ExplicitTarget | None:
    """Extract exact declarations only, never reconstruct a Recommendation.

    Fenced JSON can be inspected diagnostically if the entire response is one
    fenced object. Prose, broken JSON, duplicate declarations and unknown statuses
    are ambiguous. Text targets use exactly one anchored Decision and Option line.
    """
    if raw is None:
        return None
    if condition == "labelled-text":
        decisions = re.findall(r"^Decision:\s*([^\r\n]+)$", raw, re.MULTILINE)
        options = re.findall(r"^Option:\s*([^\r\n]+)$", raw, re.MULTILINE)
        if len(decisions) != 1 or len(options) != 1:
            return None
        status, option = decisions[0].strip(), options[0].strip()
        option_id = None if option == "none" else option
    else:
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
        if fenced is not None:
            text = fenced.group(1)
        try:
            value = json.loads(text, object_pairs_hook=_unique_object)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, dict):
            return None
        status, option_id = value.get("status"), value.get("selected_option_id")
        if "selected_option_id" not in value:
            return None
    if status not in ("selected", "abstained"):
        return None
    if status == "abstained" and option_id is not None:
        return None
    if status == "selected" and (not isinstance(option_id, str) or not option_id.strip()):
        return None
    return ExplicitTarget(status=status, option_id=option_id)


def failure_stage(raw: RawResponse, result: CaseEvaluation) -> FailureStage:
    """Mechanical stages use execution/parser diagnostics, never reference answers."""
    if result.outcome != "failure":
        return "none"
    known: tuple[FailureStage, ...] = (
        "provider_api_failure", "empty_incomplete_refusal", "serialization_format_failure",
        "parser_failure", "schema_domain_validation", "invalid_option_reference",
    )
    return next((stage for stage in known if raw.failure_stage == stage), "ambiguous")


def diagnose(problem: DecisionProblem, result: CaseEvaluation, raw: RawResponse,
             condition: Representation) -> Diagnosis:
    stage = failure_stage(raw, result)
    target = explicit_target(raw.text, condition)
    if stage in ("provider_api_failure", "empty_incomplete_refusal"):
        # Partial/refused/error text cannot establish a final declared decision.
        target = None
    matches: bool | None = None
    if target is not None:
        matches = (result.expected_abstention if target.status == "abstained" else
                   target.option_id in result.acceptable_option_ids)
    classification: DecisionDiagnosis = "not_failed"
    if matches is False:
        classification = "explicit_incorrect_decision"
    elif result.outcome == "failure":
        classification = "explicit_apparently_correct_rejected" if matches is True else "ambiguous"
    notes = [
        "Target matching uses evaluator-only references after execution, not inference-time inputs.",
        "A matching option/abstention does not establish valid review, evidence, reasoning or policy compliance.",
    ]
    if target is not None and target.option_id is not None and target.option_id not in {
        option.id for option in problem.options
    }:
        notes.append("Explicit target names an unavailable option.")
    return Diagnosis(
        failure_stage=stage, validation_issues=list(raw.validation_issues),
        decision_diagnosis=classification, explicit_target=target,
        benchmark_target_matches=matches,
        requires_human_judgment=result.outcome == "failure" or target is None,
        notes=notes,
    )
