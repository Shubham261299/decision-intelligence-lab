"""Counterbalanced dev experiment; unchanged evaluator plus separate raw diagnostics."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

from decisionlab.comparison import ComparisonReport, ExecutionMode, _column, comparison_text
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.models import CaseEvaluation, Rate
from decisionlab.evaluation.report import build_report
from decisionlab.models import DomainModel, Scalar
from decisionlab.representation_diagnosis import Diagnosis, diagnose
from decisionlab.runner import run_benchmark
from decisionlab.strategies.representation import (
    REPRESENTATIONS, RawResponse, Representation, RepresentationStrategy, digest,
)


class ScheduledRun(DomainModel):
    sequence: Annotated[int, Field(ge=1)]
    case_id: str
    repetition: Annotated[int, Field(ge=1)]
    condition: Representation


class ExperimentProtocol(DomainModel):
    version: str = "representation-v1"
    recorded_at: str
    hypothesis: str = "Labelled-text contract compliance explains part of the original pipeline gap."
    independent_variable: str = "Output representation and its required instructions/strict parser"
    controlled_variables: list[str] = [
        "policy instructions", "input serialization", "provider/model/settings", "no schema enforcement",
        "benchmark cases/repetitions", "domain/reference validation", "policy/evaluator/verifier",
    ]
    primary_metrics: list[str] = [
        "overall_correctness", "selection_correctness", "required_abstention_correctness",
        "execution_failures", "serialization_parser_failures", "domain_reference_failures",
        "review_precision_recall", "policy_violations", "paired_agreement", "repeated_agreement",
    ]
    diagnostic_metrics: list[str] = [
        "failed_explicit_apparently_correct_targets", "failed_explicit_incorrect_targets",
        "failed_ambiguous_targets", "failure_stage_counts", "substantive_correctness_not_assessed",
    ]
    validity_threats: list[str] = [
        "Representation necessarily changes instructions/parser; not a pure parser effect.",
        "Small synthetic related cases; repetitions are not independent cases.",
        "Counterbalancing reduces but cannot remove backend/cache/load/time confounding.",
        "Target matching is narrower than full recommendation correctness or semantic support.",
        "Missing raw output on transport errors is unavailable, never empty successful output.",
        "Prompts differ in length and syntax; no claim of equal tokenization.",
    ]
    procedure: list[str] = [
        "Validate frozen dev benchmark and configuration; write this protocol before calls.",
        "Follow declared schedule exactly; alternate first condition by case/repetition parity.",
        "Generate once per scheduled run; preserve raw output/envelope and hashes immediately.",
        "Use existing strict validation, policy, verifier and evaluator without output repair.",
        "Diagnose after scoring; never pass reference answers or diagnoses to strategies.",
        "Do not tune, selectively retry, resume partial runs or execute held-out.",
    ]
    split: str = "dev"
    repeats: int
    planned_generations: int
    mode: ExecutionMode
    benchmark_version: str
    manifest_sha256: str
    benchmark_artifact_sha256: dict[str, str]
    input_sha256: str
    configuration: dict[str, Scalar]
    policy_prompt_sha256: str
    prompt_sha256: dict[str, str]
    source_sha256: dict[str, str]
    schedule: list[ScheduledRun]


class DiagnosedRun(DomainModel):
    scheduled: ScheduledRun
    raw_response: RawResponse
    diagnosis: Diagnosis
    primary_result: CaseEvaluation


class DiagnosticSummary(DomainModel):
    failed_responses: int
    stage_counts: dict[str, int]
    apparent_correct_rejected_targets: int
    explicit_incorrect_targets: int
    ambiguous_targets: int
    human_judgment_required: int
    confirmed_substantively_correct: int | None = None
    confirmed_full_semantically_incorrect: int | None = None


class PairedResult(DomainModel):
    case_id: str
    repetition: int
    both_successful: bool
    same_decision: bool | None
    same_system_review: bool | None
    same_verifier_disposition: bool | None
    labelled_correct: bool
    json_correct: bool
    disagreement: bool


class PairedAnalysis(DomainModel):
    pairs: list[PairedResult]
    decision_agreement: Rate
    both_correct: int
    labelled_only_correct: int
    json_only_correct: int
    neither_correct: int


class RepresentationReport(DomainModel):
    experiment_version: str = "representation-v1"
    protocol: ExperimentProtocol
    comparison: ComparisonReport
    observations: list[DiagnosedRun]
    diagnostics: dict[str, DiagnosticSummary]
    paired: PairedAnalysis


def execution_schedule(benchmark: Benchmark, repeats: int) -> list[ScheduledRun]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    schedule: list[ScheduledRun] = []
    for rep in range(1, repeats + 1):
        for index, problem in enumerate(benchmark.problems):
            order = REPRESENTATIONS if (index + rep - 1) % 2 == 0 else REPRESENTATIONS[::-1]
            for condition in order:
                schedule.append(ScheduledRun(sequence=len(schedule) + 1, case_id=problem.id,
                                             repetition=rep, condition=condition))
    return schedule


def paired_analysis(columns: dict[str, list[CaseEvaluation]]) -> PairedAnalysis:
    indexed = {name: {(c.observation.run.problem_id, c.repetition): c for c in cases}
               for name, cases in columns.items()}
    left, right = [indexed[name] for name in REPRESENTATIONS]
    if set(left) != set(right):
        raise ValueError("Paired cases/repetitions must match")
    pairs = []
    for (case_id, rep), a in sorted(left.items()):
        b = right[(case_id, rep)]
        ar, br = a.observation.run, b.observation.run
        joint = ar.status == br.status == "success"
        same = (ar.recommendation.status, ar.recommendation.selected_option_id) == (
            br.recommendation.status, br.recommendation.selected_option_id
        ) if joint else None
        av, bv = a.observation.verification, b.observation.verification
        same_verifier = av.disposition == bv.disposition if av and bv else None
        pairs.append(PairedResult(
            case_id=case_id, repetition=rep, both_successful=joint, same_decision=same,
            same_system_review=a.system_review == b.system_review if joint else None,
            same_verifier_disposition=same_verifier,
            labelled_correct=a.outcome_correct, json_correct=b.outcome_correct,
            disagreement=(ar.status != br.status or same is False
                          or (ar.error.code if ar.error else None) != (br.error.code if br.error else None)
                          or (joint and a.system_review != b.system_review) or same_verifier is False),
        ))
    return PairedAnalysis(
        pairs=pairs,
        decision_agreement=Rate(numerator=sum(p.same_decision is True for p in pairs),
                                denominator=sum(p.both_successful for p in pairs)),
        both_correct=sum(p.labelled_correct and p.json_correct for p in pairs),
        labelled_only_correct=sum(p.labelled_correct and not p.json_correct for p in pairs),
        json_only_correct=sum(p.json_correct and not p.labelled_correct for p in pairs),
        neither_correct=sum(not p.labelled_correct and not p.json_correct for p in pairs),
    )


def run_representation_experiment(
    benchmark: Benchmark, strategies: dict[Representation, RepresentationStrategy], *,
    repeats: int = 3, mode: ExecutionMode = "live", allow_live: bool = False,
    output: Path | None = None,
) -> RepresentationReport:
    if mode not in ("live", "mocked"):
        raise ValueError("Representation experiment mode must be live or mocked")
    if mode == "live" and not allow_live:
        raise ValueError("Representation experiment requires --allow-live")
    if benchmark.split != "dev" or not benchmark.problems:
        raise ValueError("Representation experiment requires nonempty dev split")
    if set(strategies) != set(REPRESENTATIONS):
        raise ValueError("Specify exactly both representation conditions")
    snapshot = benchmark.model_copy(deep=True)
    schedule = execution_schedule(snapshot, repeats)
    a, b = [strategies[c] for c in REPRESENTATIONS]
    if a.name != "labelled-text" or b.name != "unconstrained-json":
        raise ValueError("Strategy conditions do not match keys")
    if a.configuration() != b.configuration() or a.policy != b.policy:
        raise ValueError("Model/provider/configuration/policy must match")
    if output is not None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory must be new or empty; no overwrite or resume")
        output.mkdir(parents=True, exist_ok=True)
    input_json = json.dumps([p.model_dump(mode="json") for p in snapshot.problems],
                            sort_keys=True, separators=(",", ":"), allow_nan=False)
    package = Path(__file__).parent
    protocol = ExperimentProtocol(
        recorded_at=datetime.now(timezone.utc).isoformat(), repeats=repeats,
        planned_generations=len(schedule), mode=mode,
        benchmark_version=snapshot.manifest.version, manifest_sha256=snapshot.manifest_sha256,
        benchmark_artifact_sha256=snapshot.manifest.sha256,
        input_sha256=digest(input_json) or "", configuration=a.configuration(),
        policy_prompt_sha256=digest(a.policy) or "",
        prompt_sha256={c: digest(s.prompt) or "" for c, s in strategies.items()},
        source_sha256={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in package.rglob("*") if p.suffix in (".py", ".txt")},
        schedule=schedule,
    )
    if output is not None:
        (output / "protocol.json").write_text(protocol.model_dump_json(indent=2) + "\n", encoding="utf-8")
    results: dict[str, list[CaseEvaluation]] = {c: [] for c in REPRESENTATIONS}
    records = []
    problems = {p.id: p for p in snapshot.problems}
    for item in schedule:
        problem = problems[item.case_id]
        strategy = strategies[item.condition]
        # Delegate execution, validation, policy, verification and scoring unchanged.
        single = snapshot.model_copy(update={
            "problems": [problem], "expectations": {problem.id: snapshot.expectations[problem.id]},
        }, deep=True)
        result = run_benchmark(single, strategy, strategy_name=item.condition,
                               strategy_version=strategy.version).cases[0]
        result.repetition = item.repetition
        result.observation.run.id = f"{item.condition}:{problem.id}:{item.repetition}"
        raw = strategy.last_raw.model_copy(deep=True)
        record = DiagnosedRun(scheduled=item, raw_response=raw,
                             diagnosis=diagnose(problem, result, raw, item.condition),
                             primary_result=result)
        records.append(record)
        results[item.condition].append(result)
        if output is not None:
            # Flush each complete observation so interruption does not lose earlier raw outputs.
            with (output / "observations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
                stream.flush()
    columns = {c: _column(build_report(snapshot, cases, c, strategies[c].version, repeats), mode)
               for c, cases in results.items()}
    comparison = ComparisonReport(
        benchmark=snapshot.manifest.benchmark, dataset_version=snapshot.manifest.version,
        policy_version=snapshot.manifest.policy_version, schema_version=snapshot.manifest.schema_version,
        manifest_sha256=snapshot.manifest_sha256, input_sha256=protocol.input_sha256,
        case_ids=[p.id for p in snapshot.problems], split="dev", repeats=repeats,
        execution_order=list(REPRESENTATIONS),
        columns=columns, comparability_warnings=[], limitations=list(protocol.validity_threats),
    )
    diagnostics = {}
    for condition in REPRESENTATIONS:
        failed = [r for r in records if r.scheduled.condition == condition
                  and r.primary_result.outcome == "failure"]
        counts = Counter(r.diagnosis.decision_diagnosis for r in failed)
        diagnostics[condition] = DiagnosticSummary(
            failed_responses=len(failed),
            stage_counts=dict(sorted(Counter(r.diagnosis.failure_stage for r in failed).items())),
            apparent_correct_rejected_targets=counts["explicit_apparently_correct_rejected"],
            explicit_incorrect_targets=counts["explicit_incorrect_decision"],
            ambiguous_targets=counts["ambiguous"],
            human_judgment_required=sum(r.diagnosis.requires_human_judgment for r in failed),
        )
    report = RepresentationReport(protocol=protocol, comparison=comparison, observations=records,
                                  diagnostics=diagnostics, paired=paired_analysis(results))
    if output is not None:
        (output / "experiment.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (output / "comparison.json").write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (output / "paired-analysis.json").write_text(report.paired.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (output / "experiment.txt").write_text(experiment_text(report), encoding="utf-8")
    return report


def experiment_text(report: RepresentationReport) -> str:
    # Generic comparison expects column names here; the exact schedule stays in JSON/protocol.
    table = comparison_text(report.comparison.model_copy(update={"execution_order": list(REPRESENTATIONS)}))
    lines = [table, "Diagnostic counts for FAILED responses only (never rescued scores):"]
    for condition, summary in report.diagnostics.items():
        lines.append(f"{condition}: failed={summary.failed_responses}, "
                     f"apparent-correct target={summary.apparent_correct_rejected_targets}, "
                     f"explicit-incorrect target={summary.explicit_incorrect_targets}, "
                     f"ambiguous={summary.ambiguous_targets}; stages={summary.stage_counts}")
    agreement = report.paired.decision_agreement
    lines.append(f"Paired decision agreement on joint successes: {agreement.numerator}/{agreement.denominator}")
    lines.append("Confirmed full substantive correctness: unavailable; evidence/reasoning require human judgment.")
    lines.append("Exact counterbalanced execution order is recorded in protocol.json and experiment.json.")
    return "\n".join(lines) + "\n"
