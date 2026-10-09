"""Deterministic machine-readable and concise human-readable reports."""

import json
from pathlib import Path
from pydantic import Field

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.metrics import summarize
from decisionlab.evaluation.models import CaseEvaluation, MetricSummary, Rate
from decisionlab.models import DomainModel, Text
from decisionlab.verification.models import VerificationSummary


class PairEvaluation(DomainModel):
    pair_id: Text
    kind: Text
    repetition: int
    members: tuple[str, str]
    correct: bool


class EvaluationReport(DomainModel):
    report_version: Text = "1"
    benchmark: Text
    dataset_version: Text
    schema_version: Text
    policy_version: Text
    manifest_sha256: Text
    artifact_sha256: dict[str, str]
    split: Text
    strategy: Text
    strategy_version: Text
    repeats: int
    summary: MetricSummary
    by_split: dict[str, MetricSummary]
    by_category: dict[str, MetricSummary]
    pairs: list[PairEvaluation]
    pair_rates: dict[str, Rate]
    cases: list[CaseEvaluation]
    limitations: list[str]
    verification_summary: VerificationSummary = Field(default_factory=VerificationSummary)


def build_report(
    benchmark: Benchmark, results: list[CaseEvaluation], strategy_name: str,
    strategy_version: str, repeats: int,
) -> EvaluationReport:
    expected = {(problem.id, rep) for problem in benchmark.problems for rep in range(1, repeats + 1)}
    keys = [(result.observation.run.problem_id, result.repetition) for result in results]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Report requires exactly one result per case and repetition")
    results = sorted(results, key=lambda r: (r.observation.run.problem_id, r.repetition))
    indexed = {(r.observation.run.problem_id, r.repetition): r for r in results}
    pairs: list[PairEvaluation] = []
    for pair in benchmark.manifest.pairs:
        if not all(member in benchmark.expectations for member in pair.members):
            continue
        for repetition in range(1, repeats + 1):
            left, right = [indexed[(member, repetition)] for member in pair.members]
            left_rec, right_rec = left.observation.run.recommendation, right.observation.run.recommendation
            left_key = (left.outcome, left_rec.selected_option_id if left_rec else None)
            right_key = (right.outcome, right_rec.selected_option_id if right_rec else None)
            relation = left_key == right_key if pair.kind == "wording" else left_key != right_key
            pairs.append(PairEvaluation(
                pair_id=pair.id, kind=pair.kind, repetition=repetition, members=pair.members,
                correct=left.outcome_correct and right.outcome_correct and relation,
            ))
    tags = sorted({tag for r in results for tag in r.category_tags})
    return EvaluationReport(
        benchmark=benchmark.manifest.benchmark, dataset_version=benchmark.manifest.version,
        schema_version=benchmark.manifest.schema_version,
        policy_version=benchmark.manifest.policy_version,
        manifest_sha256=benchmark.manifest_sha256,
        artifact_sha256=benchmark.manifest.sha256, split=benchmark.split,
        strategy=strategy_name, strategy_version=strategy_version, repeats=repeats,
        summary=summarize(results),
        by_split={
            split: summarize([r for r in results if r.observation.run.problem_id in ids])
            for split, ids in benchmark.manifest.splits.items()
            if any(r.observation.run.problem_id in ids for r in results)
        },
        by_category={tag: summarize([r for r in results if tag in r.category_tags]) for tag in tags},
        pairs=pairs,
        verification_summary=verification_summary(results),
        pair_rates={kind: Rate(
            numerator=sum(p.correct for p in pairs if p.kind == kind),
            denominator=sum(p.kind == kind for p in pairs),
        ) for kind in ("relevant", "wording")},
        cases=results,
        limitations=[
            "Evidence support and unknown/explanation quality require manual rubric review; scores are null.",
            "Confidence is uncalibrated; absent confidence is not replaced with a probability.",
            "Confidence diagnostics v1 use fixed bins/thresholds; dev and held_out are separate, pooled results descriptive only.",
            "Constant 0.5 reference is not strategy confidence; calibration excludes abstentions and failures, repeats are not independent samples.",
            "Offline timing/cost are unmeasured; live token usage is recorded when supplied, without inferred prices.",
            "Public synthetic holdout is not a blind external benchmark; category counts overlap.",
            "Policy gates review and reports violations without repairing the strategy recommendation.",
        ],
    )


def report_json(report: EvaluationReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False,
    ) + "\n"


def verification_summary(results: list[CaseEvaluation]) -> VerificationSummary:
    verified = [r.observation.verification for r in results
                if r.observation.verification is not None]
    errors = sum(r.observation.verification_error is not None for r in results)
    return VerificationSummary(
        total=len(results), verified=len(verified), errors=errors,
        unavailable=len(results) - len(verified) - errors,
        accepted=sum(v.disposition == "accept" for v in verified),
        rejected=sum(v.disposition == "reject" for v in verified),
        human_review=sum(v.disposition == "human_review" for v in verified),
        policy_violating=sum(bool(v.policy_violations) for v in verified),
        insufficient_evidence=sum(v.evidence_sufficiency == "insufficient" for v in verified),
    )


def report_text(report: EvaluationReport) -> str:
    summary = report.summary
    lines = [
        f"DecisionLab evaluation: {report.strategy} v{report.strategy_version}",
        f"Benchmark {report.dataset_version} | split={report.split} | repeats={report.repeats}",
        f"Cases/runs: {summary.counts['total']} | failures: {summary.counts['failed_runs']}",
    ]
    for label, key in (
        ("Outcome correctness", "outcome_correctness"),
        ("Selection correctness", "selection_correctness"),
        ("Abstention correctness", "abstention_correctness"),
        ("Selected-output accuracy", "selected_output_accuracy"),
        ("Decision coverage", "decision_coverage"),
        ("Human-review precision", "human_review_precision"),
        ("Human-review recall", "human_review_recall"),
        ("Human-review accuracy", "human_review_accuracy"),
        ("Hard-constraint violations", "hard_constraint_violation_rate"),
    ):
        rate = summary.rates[key]
        percentage = f"{rate.value:.1%}" if rate.value is not None else "n/a"
        lines.append(f"{label}: {rate.numerator}/{rate.denominator} ({percentage})")
    missed = sorted({r.case_id for r in report.cases if not r.outcome_correct})
    lines.append("Outcome failures: " + (", ".join(missed) if missed else "none"))
    lines.append(f"Confidence samples: {summary.confidence.sample_count} (uncalibrated)")
    for split, metrics in report.by_split.items():
        confidence = metrics.confidence
        lines.append(
            f"Confidence [{split}]: {confidence.availability}, "
            f"n={confidence.sample_count}, missing={confidence.missing_selected_count}, "
            f"Brier={confidence.brier_score}, ECE={confidence.calibration_error}"
        )
        reference = confidence.fixed_reference
        if reference is not None:
            lines.append(
                f"Fixed 0.5 reference [{split}] (not strategy confidence): "
                f"n={reference.sample_count}, Brier={reference.brier_score}, "
                f"ECE={reference.calibration_error}"
            )
    if summary.input_token_sample_count or summary.output_token_sample_count:
        lines.append(
            f"Reported tokens: input={summary.total_input_tokens}, output={summary.total_output_tokens}"
        )
    lines.append("Evidence support: unscored; apply per-case rubrics manually")
    verification = report.verification_summary
    lines.append(
        f"Verification (diagnostic): accept={verification.accepted}, "
        f"reject={verification.rejected}, review={verification.human_review}, "
        f"unavailable={verification.unavailable}, errors={verification.errors}"
    )
    return "\n".join(lines) + "\n"


def write_report(report: EvaluationReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_bytes(report_json(report).encode("utf-8"))
    (directory / "report.txt").write_bytes(report_text(report).encode("utf-8"))
