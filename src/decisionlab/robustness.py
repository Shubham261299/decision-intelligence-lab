"""Versioned equivalent-input fixtures and read-only stability experiments."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence, get_args

from pydantic import Field

from decisionlab.comparison import StrategyEntry
from decisionlab.evaluation.benchmark import Benchmark, Split, read_records
from decisionlab.evaluation.metrics import mean
from decisionlab.evaluation.models import Observation, Rate
from decisionlab.models import DecisionProblem, DomainModel, Fact, RunMetadata, Text
from decisionlab.runner import run_decision
from decisionlab.verification import RuleBasedVerifier
from decisionlab.verification.models import VerificationResult

Transformation = Literal[
    "wording", "distractor", "fact_order", "option_order", "redundancy", "formatting",
]
TRANSFORMATIONS: tuple[Transformation, ...] = get_args(Transformation)
ARTIFACTS = {"dev.jsonl", "held_out.jsonl"}


class RobustnessCase(DomainModel):
    id: Text
    base_case_id: Text
    split: Literal["dev", "held_out"]
    transformation: Transformation
    problem: DecisionProblem


class RobustnessManifest(DomainModel):
    version: Literal["1"] = "1"
    transformation_version: Literal["routing-invariance-1"] = "routing-invariance-1"
    source_manifest_sha256: Text
    source_dataset_version: Text
    policy_version: Text
    case_count: int = Field(gt=0)
    sha256: dict[str, Text]


class RobustnessDataset(DomainModel):
    manifest: RobustnessManifest
    manifest_sha256: Text
    split: Split
    bases: list[DecisionProblem]
    cases: list[RobustnessCase]


def transform(problem: DecisionProblem, kind: Transformation) -> DecisionProblem:
    """Preserve IDs, typed values, sources, constraints and option attributes.

    This guarantee is scoped to the routing domain, whose options are unordered.
    Keeping problem.id unchanged avoids introducing an ID-based perturbation.
    """
    result = problem.model_copy(deep=True)
    if kind == "wording":
        result.description = "Routing request: " + result.description
        for fact in result.facts:
            fact.statement = "Recorded information: " + fact.statement
    elif kind == "distractor":
        result.facts.append(Fact(
            id="robustness_stationery", statement="The desk stationery is blue; unrelated to routing.",
            value="blue", source="administrative-note",
        ))
    elif kind == "fact_order":
        result.facts.reverse()
    elif kind == "option_order":
        result.options.reverse()
    elif kind == "redundancy":
        result.context += "\nRepeated supplied statements (no additional facts): " + "; ".join(
            f.statement for f in result.facts
        )
    elif kind == "formatting":
        result.description = "\n  " + result.description + "  \n"
        result.context = "\n\n" + result.context + "\n"
        for fact in result.facts:
            fact.statement = "  " + fact.statement + "\n"
    else:
        raise ValueError("Unknown robustness transformation")
    return DecisionProblem.model_validate(result.model_dump())


def load_robustness(root: Path, benchmark: Benchmark, split: Split = "all") -> RobustnessDataset:
    """Check hashes and exact derivation before selecting a split; no labels copied."""
    if split not in ("all", "dev", "held_out") or benchmark.split != "all":
        raise ValueError("Load all source benchmark cases and a supported robustness split")
    raw = (root / "manifest.json").read_bytes()
    manifest = RobustnessManifest.model_validate_json(raw)
    if (manifest.source_manifest_sha256 != benchmark.manifest_sha256
        or manifest.source_dataset_version != benchmark.manifest.version
        or manifest.policy_version != benchmark.manifest.policy_version):
        raise ValueError("Robustness source identity mismatch")
    if set(manifest.sha256) != ARTIFACTS:
        raise ValueError("Unexpected robustness artifacts")
    bases = {p.id: p for p in benchmark.problems}
    cases: list[RobustnessCase] = []
    for partition in ("dev", "held_out"):
        path = root / f"{partition}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.sha256[path.name]:
            raise ValueError("Robustness checksum mismatch")
        for case in read_records(path, RobustnessCase):
            if case.split != partition or case.base_case_id not in benchmark.manifest.splits[partition]:
                raise ValueError("Robustness source/split mismatch")
            if case.id != f"{case.base_case_id}:{case.transformation}":
                raise ValueError("Robustness case ID mismatch")
            if case.problem != transform(bases[case.base_case_id], case.transformation):
                raise ValueError("Robustness transformation does not match versioned derivation")
            cases.append(case)
    expected = {(p.id, t) for p in benchmark.problems for t in TRANSFORMATIONS
                if transform(p, t) != p}
    keys = [(c.base_case_id, c.transformation) for c in cases]
    if len(keys) != len(set(keys)) or set(keys) != expected or len(cases) != manifest.case_count:
        raise ValueError("Robustness coverage/uniqueness mismatch")
    cases = sorted((c for c in cases if split == "all" or c.split == split), key=lambda c: c.id)
    selected_ids = {c.base_case_id for c in cases}
    return RobustnessDataset(
        manifest=manifest, manifest_sha256=hashlib.sha256(raw).hexdigest(), split=split,
        bases=[p.model_copy(deep=True) for p in benchmark.problems if p.id in selected_ids], cases=cases,
    )


class RobustnessPair(DomainModel):
    case_id: Text
    base_case_id: Text
    split: Literal["dev", "held_out"]
    transformation: Transformation
    repetition: int = Field(ge=1)
    base: Observation
    variant: Observation
    state: Literal["stable_decision", "changed_decision", "changed_abstention", "execution_failure", "invalid_output"]
    agreement: bool
    selection_flip: bool | None
    abstention_flip: bool | None
    confidence_delta: float | None
    verifier_disposition_changed: bool | None


def evaluate_pair(case: RobustnessCase, base: Observation, variant: Observation, repetition: int = 1) -> RobustnessPair:
    """No labels, strategy imports, option remapping or recommendation edits."""
    if base.run.problem_id != case.base_case_id or variant.run.problem_id != case.problem.id:
        raise ValueError("Pair problem identity mismatch")
    invalid = any(o.schema_valid is False or o.references_valid is False for o in (base, variant))
    failed = any(o.run.status == "failure" for o in (base, variant))
    left, right = base.run.recommendation, variant.run.recommendation
    abstention_flip = selection_flip = None
    agreement = False
    state: Literal["stable_decision", "changed_decision", "changed_abstention", "execution_failure", "invalid_output"]
    if failed:
        state = "invalid_output" if invalid else "execution_failure"
    else:
        if left is None or right is None:
            raise ValueError("Successful pairs require recommendations")
        abstention_flip = left.status != right.status
        selection_flip = (
            left.selected_option_id != right.selected_option_id
            if left.status == right.status == "selected" else None
        )
        agreement = not abstention_flip and left.selected_option_id == right.selected_option_id
        state = "changed_abstention" if abstention_flip else "stable_decision" if agreement else "changed_decision"
    delta = (
        right.confidence - left.confidence
        if not failed and left is not None and right is not None
        and left.status == right.status == "selected"
        and left.confidence is not None and right.confidence is not None else None
    )
    verification_change = (
        base.verification.disposition != variant.verification.disposition
        if base.verification is not None and variant.verification is not None else None
    )
    return RobustnessPair(
        case_id=case.id, base_case_id=case.base_case_id, split=case.split,
        transformation=case.transformation, repetition=repetition, base=base, variant=variant,
        state=state, agreement=agreement, selection_flip=selection_flip,
        abstention_flip=abstention_flip, confidence_delta=delta,
        verifier_disposition_changed=verification_change,
    )


class RobustnessMetrics(DomainModel):
    pair_count: int
    run_count: int
    states: dict[str, int]
    pair_agreement_rate: Rate
    selection_flip_rate: Rate
    abstention_flip_rate: Rate
    execution_failure_rate: Rate
    invalid_output_rate: Rate
    failed_pair_rate: Rate
    confidence_sample_count: int
    mean_confidence_delta: float | None
    mean_absolute_confidence_change: float | None
    verifier_disposition_change_rate: Rate


def robustness_metrics(pairs: list[RobustnessPair]) -> RobustnessMetrics:
    # Reused base observations count once per scope, not six times as executions.
    runs = {o.run.id: o for p in pairs for o in (p.base, p.variant)}
    confidence = [p.confidence_delta for p in pairs if p.confidence_delta is not None]
    return RobustnessMetrics(
        pair_count=len(pairs), run_count=len(runs),
        states={state: sum(p.state == state for p in pairs) for state in (
            "stable_decision", "changed_decision", "changed_abstention", "execution_failure", "invalid_output",
        )},
        pair_agreement_rate=Rate(numerator=sum(p.agreement for p in pairs), denominator=len(pairs)),
        selection_flip_rate=Rate(numerator=sum(p.selection_flip is True for p in pairs),
                                 denominator=sum(p.selection_flip is not None for p in pairs)),
        abstention_flip_rate=Rate(numerator=sum(p.abstention_flip is True for p in pairs),
                                  denominator=sum(p.abstention_flip is not None for p in pairs)),
        execution_failure_rate=Rate(numerator=sum(o.run.status == "failure" for o in runs.values()), denominator=len(runs)),
        invalid_output_rate=Rate(numerator=sum(o.schema_valid is False or o.references_valid is False for o in runs.values()), denominator=len(runs)),
        failed_pair_rate=Rate(numerator=sum(p.state in ("execution_failure", "invalid_output") for p in pairs), denominator=len(pairs)),
        confidence_sample_count=len(confidence), mean_confidence_delta=mean(confidence),
        mean_absolute_confidence_change=mean([abs(d) for d in confidence]),
        verifier_disposition_change_rate=Rate(numerator=sum(p.verifier_disposition_changed is True for p in pairs),
                                             denominator=sum(p.verifier_disposition_changed is not None for p in pairs)),
    )


class RobustnessColumn(DomainModel):
    mode: Literal["offline", "mocked", "live"]
    summary: RobustnessMetrics
    by_split: dict[str, RobustnessMetrics]
    by_transformation: dict[str, RobustnessMetrics]
    pairs: list[RobustnessPair]


class RobustnessReport(DomainModel):
    report_version: Literal["1"] = "1"
    fixture_manifest: RobustnessManifest
    fixture_manifest_sha256: Text
    input_sha256: Text
    split: Split
    repeats: int
    execution_order: list[str]
    columns: dict[str, RobustnessColumn]
    limitations: list[str]


def _observe(problem: DecisionProblem, entry: StrategyEntry, metadata: RunMetadata, run_id: str) -> Observation:
    observed = run_decision(problem, entry.strategy, metadata, run_id)
    candidate = observed.run.recommendation or observed.candidate
    if candidate is not None:
        try:
            result = RuleBasedVerifier().verify(problem.model_copy(deep=True), candidate.model_copy(deep=True))
            observed.verification = VerificationResult.model_validate(result.model_dump())
        except Exception:
            observed.verification_error = "verifier_error"
    return observed


def run_robustness(dataset: RobustnessDataset, entries: Sequence[StrategyEntry], *, repeats: int = 1, allow_live: bool = False) -> RobustnessReport:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    names = [e.name for e in entries]
    if not names or len(names) != len(set(names)) or any(not n.strip() for n in names):
        raise ValueError("Specify nonempty unique strategy names")
    if any(e.mode not in ("offline", "mocked", "live") for e in entries):
        raise ValueError("Unsupported execution mode")
    if any(e.mode == "live" for e in entries) and not allow_live:
        raise ValueError("Live robustness requires --allow-live")
    snapshot = dataset.model_copy(deep=True)
    if not snapshot.cases:
        raise ValueError("Robustness requires cases")
    base_ids = [p.id for p in snapshot.bases]
    case_ids = [c.id for c in snapshot.cases]
    if len(base_ids) != len(set(base_ids)) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate robustness input identities")
    base_map = {p.id: p for p in snapshot.bases}
    for case in snapshot.cases:
        if case.base_case_id not in base_map or case.problem != transform(
            base_map[case.base_case_id], case.transformation,
        ) or case.problem == base_map[case.base_case_id]:
            raise ValueError("Invalid robustness input derivation")
        if snapshot.split != "all" and case.split != snapshot.split:
            raise ValueError("Robustness split mismatch")
    # Validate all identities before any execution.
    for entry in entries:
        RunMetadata(strategy=entry.name, strategy_version=entry.version)
    columns: dict[str, RobustnessColumn] = {}
    for entry in entries:
        metadata = RunMetadata(strategy=entry.name, strategy_version=entry.version,
                               dataset_version="robustness-" + snapshot.manifest.version)
        pairs: list[RobustnessPair] = []
        for repetition in range(1, repeats + 1):
            bases = {p.id: _observe(p, entry, metadata, f"{entry.name}:base:{p.id}:{repetition}")
                     for p in snapshot.bases}
            for case in snapshot.cases:
                variant = _observe(case.problem, entry, metadata, f"{entry.name}:variant:{case.id}:{repetition}")
                pairs.append(evaluate_pair(case, bases[case.base_case_id], variant, repetition))
        columns[entry.name] = RobustnessColumn(
            mode=entry.mode, summary=robustness_metrics(pairs), pairs=pairs,
            by_split={s: robustness_metrics([p for p in pairs if p.split == s])
                      for s in ("dev", "held_out") if any(p.split == s for p in pairs)},
            by_transformation={t: robustness_metrics([p for p in pairs if p.transformation == t])
                               for t in TRANSFORMATIONS},
        )
    return RobustnessReport(
        fixture_manifest=snapshot.manifest, fixture_manifest_sha256=snapshot.manifest_sha256,
        input_sha256=hashlib.sha256(json.dumps({
            "bases": [p.model_dump(mode="json") for p in snapshot.bases],
            "cases": [c.model_dump(mode="json") for c in snapshot.cases],
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        split=snapshot.split, repeats=repeats, execution_order=names, columns=columns,
        limitations=[
            "Stability is not accuracy or calibration; consistently wrong decisions can agree perfectly.",
            "Exact selected IDs are compared; two acceptable alternatives still constitute a selection flip.",
            "Fixed transformations and public dev/held_out splits are not tuned; pooled results are descriptive.",
            "One base run per case/repetition is reused across six variants; pairs are correlated, not independent samples.",
            "Live stochasticity can cause flips without transformation effects; no identity-control or causal claim is made.",
            "Confidence changes score only two accepted selections with available estimates; confidence remains uncalibrated.",
            "Known invalid outputs overlap execution failures; unknown output validity is not interpreted as valid.",
            "Mocked providers/generators test plumbing, not real LLM robustness; no third provider-backed LLM strategy exists.",
        ],
    )


def robustness_json(report: RobustnessReport) -> str:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False) + "\n"


def robustness_text(report: RobustnessReport) -> str:
    names = report.execution_order
    lines = [f"DecisionLab robustness v1 | split={report.split} | repeats={report.repeats}",
             "| Split / metric | " + " | ".join(names) + " |",
             "| --- | " + " | ".join("---" for _ in names) + " |"]
    lines.append("| Execution mode | " + " | ".join(report.columns[n].mode for n in names) + " |")
    profiles = [{
        (o.run.metadata.strategy_version, o.run.metadata.provider,
         o.run.metadata.model_id, o.run.metadata.prompt_version)
        for p in report.columns[n].pairs for o in (p.base, p.variant)
    } for n in names]
    lines.append("| Strategy version / provider / model / prompt | " + " | ".join(
        "; ".join(sorted(" / ".join(v or "n/a" for v in profile) for profile in group))
        for group in profiles
    ) + " |")
    for split in ("dev", "held_out"):
        if not all(split in report.columns[n].by_split for n in names):
            continue
        metrics = [report.columns[n].by_split[split] for n in names]
        for label, key in (
            ("Pair agreement", "pair_agreement_rate"), ("Selection flips", "selection_flip_rate"),
            ("Abstention flips", "abstention_flip_rate"), ("Execution failures", "execution_failure_rate"),
            ("Known invalid output", "invalid_output_rate"), ("Verifier disposition changes", "verifier_disposition_change_rate"),
        ):
            rates = [getattr(m, key) for m in metrics]
            values = [f"{r.numerator}/{r.denominator} ({r.value:.1%})" if r.value is not None else "n/a" for r in rates]
            lines.append(f"| {split} / {label} | " + " | ".join(values) + " |")
        lines.append(f"| {split} / Mean absolute confidence change (samples) | " + " | ".join(
            f"{m.mean_absolute_confidence_change if m.mean_absolute_confidence_change is not None else 'n/a'} ({m.confidence_sample_count})"
            for m in metrics) + " |")
    lines.extend(["", *report.limitations])
    return "\n".join(lines) + "\n"


def write_robustness(report: RobustnessReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "robustness.json").write_bytes(robustness_json(report).encode())
    (directory / "robustness.txt").write_bytes(robustness_text(report).encode())
