import hashlib
import json
import shutil
from pathlib import Path

import pytest

from decisionlab.comparison import StrategyEntry
from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem, Recommendation, RunMetadata
from decisionlab.policy import enforce_policy
from decisionlab.robustness import (
    TRANSFORMATIONS, RobustnessDataset, RobustnessReport, evaluate_pair,
    load_robustness, robustness_json, robustness_metrics, robustness_text,
    run_robustness, transform, write_robustness,
)
from decisionlab.runner import run_decision
from decisionlab.strategies import DirectLLMStrategy, RuleBasedStrategy, StructuredLLMStrategy, ToolAssistedStrategy
from decisionlab.tools.models import ToolResult
from decisionlab.verification import RuleBasedVerifier
from test_cli import invoke
from test_direct_llm import FakeTextProvider, response as text_response
from test_direct_parser import format_recommendation
from test_runner import FixedStrategy
from test_structured_llm import FakeProvider, response as structured_response

ROOT = Path("evals/robustness")
META = RunMetadata(strategy="fixture", strategy_version="1")


@pytest.fixture
def dataset(benchmark: Benchmark) -> RobustnessDataset:
    return load_robustness(ROOT, benchmark)


def test_complete_fixtures_preserve_semantics(dataset: RobustnessDataset, benchmark: Benchmark) -> None:
    assert len(dataset.cases) == 179
    assert {s: sum(c.split == s for c in dataset.cases) for s in ("dev", "held_out")} == {"dev": 119, "held_out": 60}
    assert not any(c.base_case_id == "route-012" and c.transformation == "fact_order" for c in dataset.cases)
    bases = {p.id: p for p in benchmark.problems}
    before = benchmark.model_dump_json()
    for case in dataset.cases:
        base, variant = bases[case.base_case_id], case.problem
        assert variant == transform(base, case.transformation)
        assert variant != base
        assert variant.id == base.id
        assert sorted(variant.constraints, key=lambda c: c.id) == sorted(base.constraints, key=lambda c: c.id)
        assert sorted(variant.options, key=lambda o: o.id) == sorted(base.options, key=lambda o: o.id)
        variant_facts = {f.id: f for f in variant.facts}
        assert {f.id for f in base.facts} <= set(variant_facts)
        for f in base.facts:
            changed = variant_facts[f.id]
            assert (changed.value, type(changed.value), changed.source) == (f.value, type(f.value), f.source)
        # Exhaust every existing choice and abstention under domain policy, not
        # merely the strategy's chosen answer. Review and violations must match.
        recommendations = [Recommendation(status="abstained", explanation="Probe")]
        recommendations += [Recommendation(status="selected", selected_option_id=o.id, explanation="Probe") for o in base.options]
        for rec in recommendations:
            assert enforce_policy(base, rec) == enforce_policy(variant, rec)
        benchmark.expectations[base.id].validate_problem(variant)
    assert benchmark.model_dump_json() == before


def test_frozen_sources_and_fixture_hashes(dataset: RobustnessDataset, benchmark: Benchmark) -> None:
    assert dataset.manifest.source_manifest_sha256 == benchmark.manifest_sha256
    for name, digest in dataset.manifest.sha256.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in benchmark.manifest.sha256.items():
        assert hashlib.sha256((Path("evals") / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("mutation", ["bytes", "semantics", "split", "duplicate", "source", "artifact", "version", "count"])
def test_corrupt_fixtures_rejected(benchmark: Benchmark, tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "robustness"
    shutil.copytree(ROOT, root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    path = root / "dev.jsonl"
    lines = path.read_text().splitlines()
    if mutation == "bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation in ("semantics", "split", "duplicate"):
        record = json.loads(lines[0])
        if mutation == "semantics":
            record["problem"]["options"][0]["attributes"]["capacity"] = 999
        elif mutation == "split":
            record["split"] = "held_out"
        else:
            lines[1] = lines[0]
        if mutation != "duplicate":
            lines[0] = json.dumps(record)
        path.write_bytes(("\n".join(lines) + "\n").encode())
        manifest["sha256"]["dev.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "source":
        manifest["source_manifest_sha256"] = "0" * 64
    elif mutation == "artifact":
        manifest["sha256"]["../manifest.json"] = "0" * 64
    elif mutation == "version":
        manifest["transformation_version"] = "unrecognized"
    else:
        manifest["case_count"] = 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_robustness(root, benchmark)


def test_offline_stability_split_reports_and_determinism(dataset: RobustnessDataset, tmp_path: Path) -> None:
    specs = [StrategyEntry("rules", "1", RuleBasedStrategy()),
             StrategyEntry("tool-assisted", "1", ToolAssistedStrategy())]
    before = dataset.model_dump_json()
    first = run_robustness(dataset, specs, repeats=2)
    second = run_robustness(dataset, specs, repeats=2)
    assert dataset.model_dump_json() == before
    assert robustness_json(first) == robustness_json(second)
    assert RobustnessReport.model_validate_json(robustness_json(first)) == first
    for c in first.columns.values():
        assert c.summary.pair_agreement_rate.value == 1
        assert c.summary.pair_count == 358 and c.summary.run_count == 418
        assert c.by_split["dev"].pair_count == 238
        assert c.by_split["held_out"].pair_count == 120
        assert c.summary.execution_failure_rate.value == 0
        assert c.summary.verifier_disposition_change_rate.value == 0
        assert c.summary.confidence_sample_count == 0
        assert c.summary.mean_absolute_confidence_change is None
        assert set(c.by_transformation) == set(TRANSFORMATIONS)
    write_robustness(first, tmp_path)
    assert (tmp_path / "robustness.json").read_bytes() == robustness_json(first).encode()
    assert (tmp_path / "robustness.txt").read_bytes() == robustness_text(first).encode()
    assert "held_out / Pair agreement" in robustness_text(first)
    assert "accuracy" not in first.columns["rules"].summary.model_dump()


class MockGenerator:
    name = "mocked-generator"
    version = "1"

    def __init__(self) -> None:
        self.requests: list[DecisionProblem] = []

    def generate(self, problem: DecisionProblem, evidence: list[ToolResult]) -> Recommendation:
        self.requests.append(problem.model_copy(deep=True))
        assert all(r.status == "success" for r in evidence)
        return RuleBasedStrategy().decide(problem)


def test_mocked_direct_structured_and_tool_generation(dataset: RobustnessDataset) -> None:
    # Execution order is all bases then sorted variants for each repetition.
    inputs = dataset.bases + [c.problem for c in dataset.cases]
    recs = [RuleBasedStrategy().decide(p) for p in inputs]
    direct = FakeTextProvider([text_response(format_recommendation(r)) for r in recs])
    structured = FakeProvider([structured_response(r.model_dump_json()) for r in recs])
    generator = MockGenerator()
    specs = [StrategyEntry("direct-llm", "1", DirectLLMStrategy(direct), "mocked"),
             StrategyEntry("structured-llm", "1", StructuredLLMStrategy(structured), "mocked"),
             StrategyEntry("tool-assisted", "1", ToolAssistedStrategy(generator=generator), "mocked")]
    report = run_robustness(dataset, specs)
    for c in report.columns.values():
        assert c.summary.pair_agreement_rate.value == 1
        assert c.summary.execution_failure_rate.value == 0
        assert c.summary.run_count == 209
    for requests in (direct.requests, structured.requests):
        assert len(requests) == len(inputs)
        for request, problem in zip(requests, inputs, strict=True):
            assert json.loads(request["problem_json"]) == problem.model_dump(mode="json")
            assert "acceptable_option_ids" not in request["problem_json"]
            assert "expected_abstention" not in request["problem_json"]
            assert "transformation" not in json.loads(request["problem_json"])
        metadata = report.columns["direct-llm" if requests is direct.requests else "structured-llm"].pairs[0].variant.run.metadata
        assert metadata.model_id == "model-snapshot" and metadata.prompt_version
    assert generator.requests == inputs


@pytest.mark.parametrize("kind", ["direct", "structured"])
def test_mocked_llm_flips_and_failures_are_recorded(dataset: RobustnessDataset, kind: str) -> None:
    inputs = dataset.bases + [c.problem for c in dataset.cases]
    recs = [RuleBasedStrategy().decide(p) for p in inputs]
    recs[0].confidence = 0.8
    first_variant = len(dataset.bases)
    recs[first_variant].selected_option_id = "q-e"
    recs[first_variant].confidence = 0.6
    if kind == "direct":
        provider = FakeTextProvider([text_response(format_recommendation(r)) for r in recs])
        provider.events[first_variant + 1] = text_response("ambiguous answer")
        provider.events[first_variant + 2] = DecisionExecutionError("provider_timeout")
        strategy = DirectLLMStrategy(provider)
    else:
        provider = FakeProvider([structured_response(r.model_dump_json()) for r in recs])
        provider.events[first_variant + 1] = structured_response("{}")
        provider.events[first_variant + 2] = DecisionExecutionError("provider_timeout")
        strategy = StructuredLLMStrategy(provider)
    report = run_robustness(dataset, [StrategyEntry(kind, "1", strategy, "mocked")])
    column = report.columns[kind]
    assert [p.state for p in column.pairs[:3]] == ["changed_decision", "invalid_output", "execution_failure"]
    assert column.summary.selection_flip_rate.numerator == 1
    assert column.summary.execution_failure_rate.numerator == 2
    assert column.summary.invalid_output_rate.numerator == 1
    assert column.summary.confidence_sample_count == 1
    assert column.summary.mean_confidence_delta == pytest.approx(-0.2)
    assert column.pairs[0].variant.run.recommendation.selected_option_id == "q-e"


@pytest.mark.parametrize("mode,state", [
    ("stable", "stable_decision"), ("option", "changed_decision"),
    ("abstain", "changed_abstention"), ("api", "execution_failure"),
    ("schema", "invalid_output"), ("references", "invalid_output"),
])
def test_distinct_pair_outcomes_and_no_repair(dataset: RobustnessDataset, mode: str, state: str) -> None:
    case = dataset.cases[0]
    base_problem = dataset.bases[0]
    rec = RuleBasedStrategy().decide(base_problem)
    rec.confidence = 0.8
    base = run_decision(base_problem, FixedStrategy(rec), META, "base")
    changed = rec.model_copy(deep=True)
    changed.confidence = 0.6
    if mode == "option":
        changed.selected_option_id = "q-e"
    elif mode == "abstain":
        changed.status, changed.selected_option_id = "abstained", None
    elif mode == "schema":
        changed = changed.model_copy(update={"confidence": 1.1})
    elif mode == "references":
        changed.selected_option_id = "missing"
    class Timeout:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            raise DecisionExecutionError("provider_timeout")
    variant = run_decision(case.problem, Timeout() if mode == "api" else FixedStrategy(changed), META, "variant")
    for problem, observation in ((base_problem, base), (case.problem, variant)):
        candidate = observation.run.recommendation or observation.candidate
        if candidate is not None:
            observation.verification = RuleBasedVerifier().verify(problem, candidate)
    before = (base.model_dump_json(), variant.model_dump_json())
    pair = evaluate_pair(case, base, variant)
    assert pair.state == state
    assert before == (base.model_dump_json(), variant.model_dump_json())
    metrics = robustness_metrics([pair])
    assert metrics.pair_agreement_rate.value == (1 if mode == "stable" else 0)
    assert pair.confidence_delta == pytest.approx(-0.2) if mode in ("stable", "option") else pair.confidence_delta is None
    if mode == "option":
        assert pair.selection_flip is True and pair.verifier_disposition_changed is True
        assert pair.variant.run.recommendation.selected_option_id == "q-e"
    if mode == "abstain":
        assert pair.abstention_flip is True and pair.selection_flip is None
    if mode in ("api", "schema", "references"):
        assert metrics.execution_failure_rate.value == 0.5
        assert pair.abstention_flip is pair.selection_flip is None
        assert metrics.invalid_output_rate.value == (0 if mode == "api" else 0.5)


def test_failures_never_agree_and_acceptable_alternatives_still_flip(dataset: RobustnessDataset) -> None:
    case = next(c for c in dataset.cases if c.base_case_id == "route-009")
    base_problem = next(p for p in dataset.bases if p.id == case.base_case_id)
    left = Recommendation(status="selected", selected_option_id="q-r", explanation="Allowed")
    right = left.model_copy(update={"selected_option_id": "q-r2"})
    a = run_decision(base_problem, FixedStrategy(left), META, "a")
    b = run_decision(case.problem, FixedStrategy(right), META, "b")
    assert evaluate_pair(case, a, b).selection_flip is True
    bad = left.model_copy(update={"selected_option_id": "missing"})
    a = run_decision(base_problem, FixedStrategy(bad), META, "a")
    b = run_decision(case.problem, FixedStrategy(bad), META, "b")
    assert evaluate_pair(case, a, b).agreement is False


def test_cli_splits_live_guard_and_outputs(tmp_path: Path) -> None:
    first = invoke("robustness", "--split", "held_out", "--json", "--output", str(tmp_path))
    assert first.returncode == 0
    payload = json.loads(first.stdout)
    assert set(payload["columns"]["rules"]["by_split"]) == {"held_out"}
    assert payload["columns"]["rules"]["summary"]["pair_count"] == 60
    assert (tmp_path / "robustness.json").read_bytes() == first.stdout.encode()
    second = invoke("robustness", "--split", "held_out", "--json")
    assert first.stdout == second.stdout
    assert invoke("robustness", "--split", "dev").returncode == 0
    for strategy in ("direct-llm", "structured-llm"):
        blocked = invoke("robustness", "--strategies", strategy)
        assert blocked.returncode == 2 and "allow-live" in blocked.stderr
        assert not blocked.stdout
    assert invoke("robustness", "--fixtures", str(tmp_path / "missing")).returncode == 2


def test_runner_guards_before_execution(dataset: RobustnessDataset) -> None:
    class Counting:
        count = 0
        def decide(self, problem: DecisionProblem) -> Recommendation:
            self.count += 1
            return RuleBasedStrategy().decide(problem)
    strategy = Counting()
    with pytest.raises(ValueError, match="allow-live"):
        run_robustness(dataset, [StrategyEntry("live", "1", strategy, "live")])
    for repeats in (0, -1, True, 0.5):
        with pytest.raises(ValueError):
            run_robustness(dataset, [StrategyEntry("rules", "1", strategy)], repeats=repeats)
    for specs in ([], [StrategyEntry("rules", "1", strategy)] * 2,
                  [StrategyEntry("rules", "", strategy)]):
        with pytest.raises(ValueError):
            run_robustness(dataset, specs)
    assert strategy.count == 0
    damaged = dataset.model_copy(deep=True)
    damaged.cases[0].problem.options[0].attributes["capacity"] = 999
    with pytest.raises(ValueError, match="derivation"):
        run_robustness(damaged, [StrategyEntry("rules", "1", strategy)])
    assert strategy.count == 0


def test_empty_metrics_and_missing_diagnostics() -> None:
    metrics = robustness_metrics([])
    assert metrics.pair_agreement_rate.value is None
    assert metrics.verifier_disposition_change_rate.value is None
    assert metrics.mean_confidence_delta is None
