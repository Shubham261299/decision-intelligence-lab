"""Matched experiments over existing strategy-independent evaluations."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.models import Rate
from decisionlab.evaluation.report import EvaluationReport
from decisionlab.models import DomainModel, RunMetadata, Text
from decisionlab.runner import run_benchmark
from decisionlab.strategies.base import DecisionStrategy

ExecutionMode = Literal["offline", "mocked", "live"]


@dataclass(frozen=True)
class StrategyEntry:
    name: str
    version: str
    strategy: DecisionStrategy
    mode: ExecutionMode = "offline"


class ComparisonColumn(DomainModel):
    mode: ExecutionMode
    metadata_profiles: list[RunMetadata]
    report: EvaluationReport
    invalid_output_rate: Rate
    format_failure_rate: Rate
    wrong_selected_decisions: int
    valid_abstentions: int
    unnecessary_abstentions: int
    failure_codes: dict[str, int]


class ComparisonReport(DomainModel):
    comparison_version: Text = "1"
    benchmark: Text
    dataset_version: Text
    policy_version: Text
    schema_version: Text
    manifest_sha256: Text
    input_sha256: Text
    case_ids: list[Text]
    split: Text
    repeats: int
    execution_order: list[Text]
    columns: dict[str, ComparisonColumn]
    comparability_warnings: list[str]
    limitations: list[str]


def _column(report: EvaluationReport, mode: ExecutionMode) -> ComparisonColumn:
    profiles: dict[str, RunMetadata] = {}
    invalid = formats = wrong = abstentions = unnecessary = 0
    errors: Counter[str] = Counter()
    for case in report.cases:
        observation = case.observation
        run = observation.run
        # Per-response IDs and retry counts remain in the complete per-case report.
        profile = run.metadata.model_copy(update={
            "provider_response_id": None, "retry_count": 0,
        })
        profiles[json.dumps(profile.model_dump(mode="json"), sort_keys=True)] = profile
        invalid += (
            observation.schema_valid is False or observation.references_valid is False
        )
        if run.error is not None:
            errors[run.error.code] += 1
            formats += run.error.code in (
                "invalid_schema", "provider_malformed_response",
            )
        wrong += case.outcome == "selected" and not case.outcome_correct
        abstentions += case.outcome == "abstained"
        unnecessary += case.outcome == "abstained" and not case.expected_abstention
    n = len(report.cases)
    return ComparisonColumn(
        mode=mode, metadata_profiles=[profiles[key] for key in sorted(profiles)],
        report=report, invalid_output_rate=Rate(numerator=invalid, denominator=n),
        format_failure_rate=Rate(numerator=formats, denominator=n),
        wrong_selected_decisions=wrong, valid_abstentions=abstentions,
        unnecessary_abstentions=unnecessary, failure_codes=dict(sorted(errors.items())),
    )


def run_comparison(
    benchmark: Benchmark, entries: Sequence[StrategyEntry], *, repeats: int = 1,
    allow_live: bool = False,
) -> ComparisonReport:
    """Strategies receive only isolated problems through the existing runner.

    Programmatic callers must declare live entries truthfully; this is an opt-in
    boundary, not a network sandbox. CLI factories control those declarations.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    names = [entry.name for entry in entries]
    if not names or len(names) != len(set(names)) or any(not n.strip() for n in names):
        raise ValueError("Specify nonempty, unique strategy names")
    if any(entry.mode not in ("offline", "mocked", "live") for entry in entries):
        raise ValueError("Unsupported execution mode")
    if any(entry.mode == "live" for entry in entries) and not allow_live:
        raise ValueError("Live comparison requires --allow-live")
    for entry in entries:
        RunMetadata(strategy=entry.name, strategy_version=entry.version)
    if not benchmark.problems:
        raise ValueError("Comparison requires benchmark cases")
    # Snapshot once: each column uses exactly the same inputs, labels and order.
    snapshot = benchmark.model_copy(deep=True)
    columns: dict[str, ComparisonColumn] = {}
    for entry in entries:
        report = run_benchmark(
            snapshot, entry.strategy, strategy_name=entry.name,
            strategy_version=entry.version, repeats=repeats,
        )
        columns[entry.name] = _column(report, entry.mode)
    signatures: set[str] = set()
    for column in columns.values():
        for profile in column.metadata_profiles:
            if profile.provider is not None:
                settings = {
                    k: v for k, v in profile.generation_settings.items()
                    if k not in ("prompt_sha256", "parser_version", "output_format")
                }
                signatures.add(json.dumps({
                    "provider": profile.provider, "model": profile.model_id,
                    "settings": settings,
                }, sort_keys=True))
    input_json = json.dumps(
        [p.model_dump(mode="json") for p in snapshot.problems], sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return ComparisonReport(
        benchmark=snapshot.manifest.benchmark, dataset_version=snapshot.manifest.version,
        policy_version=snapshot.manifest.policy_version,
        schema_version=snapshot.manifest.schema_version,
        manifest_sha256=snapshot.manifest_sha256,
        input_sha256=hashlib.sha256(input_json.encode()).hexdigest(),
        case_ids=[p.id for p in snapshot.problems], split=snapshot.split,
        repeats=repeats, execution_order=names, columns=columns,
        comparability_warnings=[
            "Provider/model/configuration profiles differ; inspect per-run metadata before attribution."
        ] if len(signatures) > 1 else [],
        limitations=[
            "Mocked columns validate plumbing, not LLM accuracy; a single column is not a comparative experiment.",
            "Invalid output counts known schema/reference rejection; API failures with no candidate have unknown output validity.",
            "Format failures include malformed API envelopes; failure codes distinguish them from decision parsing failures.",
            "Valid abstention means accepted execution, not necessarily correct abstention; policy violations can overlap wrong decisions.",
            "Evidence support is manual/unscored; confidence is uncalibrated and conditional on selected outputs with confidence.",
            "Repeated agreement requires identical outcome, option and final review on every repeat; any failed repeat prevents agreement.",
            "Usage/cost include available measurements only, not estimated prices or necessarily all billed retry usage.",
            "Fixed strategy-block execution order is recorded; no randomized order, statistical significance or causal superiority is established.",
            "Small public synthetic benchmark and formatting differences do not establish general decision reliability.",
            "Confidence diagnostics v1 use fixed thresholds, no fitting or held-out tuning; pooled results are descriptive, inspect split summaries.",
            "Constant 0.5 reference measures all accepted selections, not reported strategy confidence; abstentions/failures are excluded from calibration.",
        ],
    )


def comparison_json(report: ComparisonReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False,
    ) + "\n"


def _rate(rate: Rate) -> str:
    percentage = f" ({rate.value:.1%})" if rate.value is not None else " (n/a)"
    return f"{rate.numerator}/{rate.denominator}" + percentage


def comparison_text(report: ComparisonReport) -> str:
    names = report.execution_order
    lines = [
        f"DecisionLab comparison | dataset={report.dataset_version} | split={report.split} | repeats={report.repeats}",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]

    def row(label: str, values: list[str]) -> None:
        lines.append("| " + label + " | " + " | ".join(values) + " |")

    columns = [report.columns[name] for name in names]
    row("Execution mode", [c.mode for c in columns])
    row("Strategy version", [c.report.strategy_version for c in columns])
    row("Model / prompt", [", ".join(sorted({
        f"{p.model_id or 'n/a'} / {p.prompt_version or 'n/a'}"
        for p in c.metadata_profiles
    })) for c in columns])
    for label, key in (
        ("Overall correctness", "outcome_correctness"),
        ("Selection correctness", "selection_correctness"),
        ("Abstention correctness", "abstention_correctness"),
        ("Selected accuracy", "selected_output_accuracy"),
        ("Decision coverage", "decision_coverage"),
        ("Execution failures", "execution_failure_rate"),
        ("Schema validity (assessed)", "schema_validity"),
        ("Reference validity (assessed)", "reference_validity"),
        ("Review precision", "human_review_precision"),
        ("Review recall", "human_review_recall"),
        ("Hard policy violations", "hard_constraint_violation_rate"),
        ("Evidence citation validity", "evidence_citation_validity"),
        ("Evidence reference validity", "evidence_reference_validity"),
        ("Repeated agreement", "repeated_decision_agreement"),
    ):
        row(label, [_rate(c.report.summary.rates[key]) for c in columns])
    row("Known invalid output / all runs", [_rate(c.invalid_output_rate) for c in columns])
    row("Format/envelope failures / all runs", [_rate(c.format_failure_rate) for c in columns])
    for label, key in (
        ("Verifier accepted", "accepted"), ("Verifier rejected", "rejected"),
        ("Verifier review", "human_review"), ("Verifier unavailable", "unavailable"),
        ("Verifier errors", "errors"),
    ):
        row(label, [str(getattr(c.report.verification_summary, key)) for c in columns])
    for label, values in (
        ("Wrong selected decisions", [c.wrong_selected_decisions for c in columns]),
        ("Valid abstentions", [c.valid_abstentions for c in columns]),
        ("Unnecessary abstentions", [c.unnecessary_abstentions for c in columns]),
        ("Confidence samples", [c.report.summary.confidence.sample_count for c in columns]),
        ("Mean confidence (uncalibrated)", [c.report.summary.confidence.mean_confidence for c in columns]),
        ("Brier score (uncalibrated)", [c.report.summary.confidence.brier_score for c in columns]),
        ("Evidence support samples", [c.report.summary.evidence_support_scored_count for c in columns]),
        ("Mean evidence support", [c.report.summary.mean_evidence_support for c in columns]),
        ("Input tokens", [c.report.summary.total_input_tokens for c in columns]),
        ("Input token samples", [c.report.summary.input_token_sample_count for c in columns]),
        ("Output tokens", [c.report.summary.total_output_tokens for c in columns]),
        ("Output token samples", [c.report.summary.output_token_sample_count for c in columns]),
        ("Reported cost USD", [c.report.summary.total_cost_usd for c in columns]),
        ("Cost samples", [c.report.summary.cost_sample_count for c in columns]),
    ):
        row(label, ["n/a" if v is None else str(v) for v in values])
    for split in sorted({split for c in columns for split in c.report.by_split}):
        confidence = [c.report.by_split[split].confidence for c in columns]
        row(f"Confidence availability [{split}]", [s.availability for s in confidence])
        for label, key in (
            ("Confidence samples", "sample_count"),
            ("Missing selected confidence", "missing_selected_count"),
            ("Mean confidence", "mean_confidence"),
            ("Brier", "brier_score"), ("ECE", "calibration_error"),
            ("Maximum reliability gap", "maximum_reliability_gap"),
        ):
            row(f"{label} [{split}]", [
                "n/a" if getattr(s, key) is None else str(getattr(s, key))
                for s in confidence
            ])
        for index in range(6):
            if not any(s.sample_count for s in confidence):
                break
            row(f"Confidence coverage >= {index / 5:g} [{split}]", [
                _rate(s.coverage_curve[index].coverage) for s in confidence
            ])
            row(f"Retained accuracy >= {index / 5:g} [{split}]", [
                _rate(s.coverage_curve[index].accuracy) for s in confidence
            ])
        row(f"Fixed 0.5 reference samples [{split}]", [
            str(s.fixed_reference.sample_count) if s.fixed_reference is not None else "n/a"
            for s in confidence
        ])
        row(f"Fixed 0.5 reference Brier [{split}]", [
            str(s.fixed_reference.brier_score) if s.fixed_reference is not None
            and s.fixed_reference.brier_score is not None else "n/a"
            for s in confidence
        ])
        row(f"Fixed 0.5 reference ECE [{split}]", [
            str(s.fixed_reference.calibration_error) if s.fixed_reference is not None
            and s.fixed_reference.calibration_error is not None else "n/a"
            for s in confidence
        ])
    lines.extend(["", *report.comparability_warnings, *report.limitations])
    return "\n".join(lines) + "\n"


def write_comparison(report: ComparisonReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_bytes(comparison_json(report).encode())
    (directory / "comparison.txt").write_bytes(comparison_text(report).encode())
