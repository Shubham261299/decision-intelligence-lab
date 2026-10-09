"""Validate frozen benchmark inputs and separate evaluator-only references."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import Field

from decisionlab.evaluation.models import EvaluationCase
from decisionlab.models import DecisionProblem, DomainModel, Text, require_unique
from decisionlab.policy import POLICY_VERSION

Split = Literal["dev", "held_out", "all"]
Model = TypeVar("Model", bound=DomainModel)
ARTIFACTS = {
    "cases/dev.jsonl", "cases/held_out.jsonl", "expectations/dev.jsonl",
    "expectations/held_out.jsonl", "expectations/rubrics.json",
}


class Pair(DomainModel):
    id: Text
    members: tuple[Text, Text]
    kind: Literal["relevant", "wording"]
    changed_field: Text


class Manifest(DomainModel):
    benchmark: Text
    version: Text
    schema_version: Literal["1"]
    policy_version: Text
    case_count: Annotated[int, Field(gt=0)]
    splits: dict[str, list[Text]]
    pairs: list[Pair]
    sha256: dict[str, Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]]


class CaseRubric(DomainModel):
    reason: Text
    required_evidence_ids: list[Text]
    required_constraint_ids: list[Text]
    option_attribute_checks: list[Text]
    review_reason: Text | None


class Rubrics(DomainModel):
    version: Text
    cases: dict[str, CaseRubric]
    evidence_support_scale: dict[str, Text]
    review_scale: dict[str, Text]


class Benchmark(DomainModel):
    manifest: Manifest
    manifest_sha256: Text
    problems: list[DecisionProblem]
    expectations: dict[str, EvaluationCase]
    rubrics: Rubrics
    split: Split


def read_records(path: Path, model: type[Model]) -> list[Model]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"Empty record or file: {path.name}")
    return [model.model_validate_json(line) for line in lines]


def load_benchmark(root: Path, split: Split = "all") -> Benchmark:
    if split not in ("dev", "held_out", "all"):
        raise ValueError("Unsupported split")
    raw_manifest = (root / "manifest.json").read_bytes()
    manifest = Manifest.model_validate_json(raw_manifest)
    if manifest.policy_version != POLICY_VERSION:
        raise ValueError("Unsupported benchmark policy version")
    if set(manifest.sha256) != ARTIFACTS:
        raise ValueError("Manifest must list exactly the supported benchmark artifacts")
    for relative, digest in manifest.sha256.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
    if set(manifest.splits) != {"dev", "held_out"}:
        raise ValueError("Expected development and held-out splits")
    all_problems: list[DecisionProblem] = []
    expectations: dict[str, EvaluationCase] = {}
    for partition in ("dev", "held_out"):
        problems = read_records(root / "cases" / f"{partition}.jsonl", DecisionProblem)
        labels = read_records(root / "expectations" / f"{partition}.jsonl", EvaluationCase)
        ids = [problem.id for problem in problems]
        if ids != manifest.splits[partition] or [e.problem_id for e in labels] != ids:
            raise ValueError("Manifest, input and expectation order/IDs must match")
        require_unique([label.id for label in labels], "Expectation IDs")
        for problem, label in zip(problems, labels, strict=True):
            label.validate_problem(problem)
            expectations[problem.id] = label
        all_problems.extend(problems)
    ids = [problem.id for problem in all_problems]
    require_unique(ids, "Problem IDs across splits")
    require_unique([e.id for e in expectations.values()], "Expectation IDs across splits")
    if len(ids) != manifest.case_count:
        raise ValueError("Manifest case count mismatch")
    rubrics = Rubrics.model_validate_json((root / "expectations/rubrics.json").read_bytes())
    if rubrics.version != manifest.version or set(rubrics.cases) != set(ids):
        raise ValueError("Rubric version/IDs mismatch")
    for problem in all_problems:
        rubric = rubrics.cases[problem.id]
        if not set(rubric.required_evidence_ids) <= {f.id for f in problem.facts}:
            raise ValueError("Unknown rubric evidence reference")
        if not set(rubric.required_constraint_ids) <= {c.id for c in problem.constraints}:
            raise ValueError("Unknown rubric constraint reference")
        if bool(rubric.review_reason) != expectations[problem.id].expected_review:
            raise ValueError("Rubric review requirement mismatch")
    require_unique([pair.id for pair in manifest.pairs], "Pair IDs")
    for pair in manifest.pairs:
        if pair.members[0] == pair.members[1] or sum(
            set(pair.members) <= set(partition) for partition in manifest.splits.values()
        ) != 1:
            raise ValueError("Pair must have distinct members in one split")
    selected = set(ids if split == "all" else manifest.splits[split])
    return Benchmark(
        manifest=manifest, manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        problems=[p for p in all_problems if p.id in selected],
        expectations={key: value for key, value in expectations.items() if key in selected},
        rubrics=rubrics, split=split,
    )
