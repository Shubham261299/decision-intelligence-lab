
import pytest

from decisionlab.comparison import comparison_json, comparison_text, run_comparison
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.metrics import summarize
from decisionlab.evaluation.evaluator import evaluate_observation
from decisionlab.evaluation.report import build_report
from decisionlab.models import Constraint, DecisionProblem, Recommendation, RunMetadata, Unknown
from decisionlab.runner import run_benchmark, run_decision
from decisionlab.strategies import RuleBasedStrategy
from decisionlab.verification import DecisionVerifier, RuleBasedVerifier, VerificationResult
from decisionlab.verification.base import RecommendationInput
from test_comparison import entries


def test_valid_contract_and_original_preservation(benchmark: Benchmark) -> None:
    problem = benchmark.problems[0]
    rec = RuleBasedStrategy().decide(problem)
    before = (problem.model_dump_json(), rec.model_dump_json())
    verifier: DecisionVerifier = RuleBasedVerifier()
    result = verifier.verify(problem, rec)
    assert result.disposition == "accept"
    assert result.decision_validity == "valid"
    assert result.structurally_valid and result.selected_option_valid
    assert result.references_valid and result.evidence_references_valid
    assert result.evidence_sufficiency == "sufficient"
    assert result.evidence_support == "not_assessed"
    assert result.policy_violations == [] and not result.requires_human_review
    assert result.verifier == "rules-verifier" and result.verifier_version == "1"
    assert before == (problem.model_dump_json(), rec.model_dump_json())
    assert VerificationResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("changes", [
    {"selected_option_id": None}, {"status": "maybe"}, {"confidence": 2},
    {"explanation": ""}, {"unexpected": "extra"},
    {"assessments": [{"option_id": "q-r", "eligibility": "eligible", "reasons": []}]},
])
def test_raw_invalid_schema_is_rejected(benchmark: Benchmark, changes: dict[str, object]) -> None:
    raw = RuleBasedStrategy().decide(benchmark.problems[0]).model_dump()
    raw.update(changes)
    result = RuleBasedVerifier().verify(benchmark.problems[0], raw)
    assert result.disposition == "reject" and not result.structurally_valid
    assert result.references_valid is None and result.selected_option_valid is None
    assert result.findings[0].code == "invalid_schema"
    assert result.evidence_sufficiency == "not_assessed"


def test_mutated_model_is_revalidated(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.confidence = 9
    assert not RuleBasedVerifier().verify(benchmark.problems[0], rec).structurally_valid
    assert rec.confidence == 9


@pytest.mark.parametrize("field,code", [
    ("selected", "unknown_selected_option"),
    ("assessment", "unknown_assessment_option"),
    ("evidence", "unknown_evidence_reference"),
    ("constraint", "unknown_constraint_reference"),
    ("unknown", "unknown_affected_option"),
])
def test_all_reference_namespaces(benchmark: Benchmark, field: str, code: str) -> None:
    problem = benchmark.problems[0]
    rec = RuleBasedStrategy().decide(problem)
    if field == "selected":
        rec.selected_option_id = "invented"
    elif field == "assessment":
        rec.assessments[0].option_id = "invented"
    elif field == "evidence":
        rec.assessments[0].evidence_ids = ["invented"]
    elif field == "constraint":
        rec.assessments[0].constraint_ids = ["invented"]
    else:
        rec.unknowns = [Unknown(description="Missing", affected_option_ids=["invented"], material_to_decision=True)]
    result = RuleBasedVerifier().verify(problem, rec)
    assert result.disposition == "reject" and result.structurally_valid
    assert result.references_valid is False
    assert any(f.code == code and f.path for f in result.findings)
    assert result.evidence_references_valid is (field != "evidence")
    assert result.selected_option_valid is (field != "selected")


@pytest.mark.parametrize("change,code", [
    ("route", "wrong_route"), ("capacity", "capacity_exhausted"),
    ("fact", "unresolved_fact:urgent"), ("approval", "approval_not_granted"),
    ("unsupported", "unsupported_hard_constraint:compliance"),
])
def test_policy_violations(benchmark: Benchmark, change: str, code: str) -> None:
    problem = benchmark.problems[0].model_copy(deep=True)
    rec = RuleBasedStrategy().decide(problem)
    if change == "route":
        rec.selected_option_id = "q-e"
    elif change == "capacity":
        next(o for o in problem.options if o.id == rec.selected_option_id).attributes["capacity"] = 0
    elif change == "fact":
        next(f for f in problem.facts if f.id == "urgent").value = None
    elif change == "approval":
        problem.constraints.append(Constraint(id="approval", description="Requires approval", kind="hard"))
    else:
        problem.constraints.append(Constraint(id="compliance", description="Manual compliance", kind="hard"))
    result = RuleBasedVerifier().verify(problem, rec)
    assert result.disposition == "reject"
    assert result.structurally_valid and result.selected_option_valid and result.references_valid
    assert code in result.policy_violations
    assert any(f.code == code and f.kind == "policy_violation" for f in result.findings)
    assert result.requires_human_review


@pytest.mark.parametrize("change", ["no_assessment", "no_evidence", "irrelevant_evidence", "no_constraint", "unknown_eligibility"])
def test_insufficient_traceability_requires_review(benchmark: Benchmark, change: str) -> None:
    problem = benchmark.problems[0]
    rec = RuleBasedStrategy().decide(problem)
    selected = next(a for a in rec.assessments if a.option_id == rec.selected_option_id)
    if change == "no_assessment":
        rec.assessments.remove(selected)
    elif change == "no_evidence":
        selected.evidence_ids = []
    elif change == "irrelevant_evidence":
        # One existing reference is not sufficient for the required routing facts.
        selected.evidence_ids = ["urgent"]
    elif change == "no_constraint":
        selected.constraint_ids = []
    else:
        selected.eligibility = "unknown"
    result = RuleBasedVerifier().verify(problem, rec)
    assert result.disposition == "human_review" and result.requires_human_review
    assert result.references_valid and result.evidence_references_valid
    assert result.evidence_sufficiency == "insufficient"
    assert result.policy_violations == []
    assert any(f.kind == "insufficient_evidence" for f in result.findings)


def test_self_reported_ineligible_selection_rejected(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    next(a for a in rec.assessments if a.option_id == rec.selected_option_id).eligibility = "ineligible"
    result = RuleBasedVerifier().verify(benchmark.problems[0], rec)
    assert result.disposition == "reject" and result.policy_violations == []
    assert any(f.code == "selection_assessment_conflict" for f in result.findings)


def test_supervisor_review_on_valid_selection(benchmark: Benchmark) -> None:
    problem = next(p for p in benchmark.problems if any(f.id == "review_required" and f.value is True for f in p.facts)
                   and RuleBasedStrategy().decide(p).status == "selected")
    rec = RuleBasedStrategy().decide(problem)
    result = RuleBasedVerifier().verify(problem, rec)
    assert result.disposition == "human_review" and result.selected_option_valid
    assert result.policy_violations == [] and result.evidence_sufficiency == "sufficient"
    assert any(f.code == "explicit_supervisor_review" for f in result.findings)
    assert any(f.evidence_ids == ["review_required"] for f in result.findings)


@pytest.mark.parametrize("kind", ["requested", "material_unknown"])
def test_review_without_policy_violation(benchmark: Benchmark, kind: str) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    if kind == "requested":
        rec.requested_review = True
    else:
        rec.unknowns = [Unknown(description="Uncertain", affected_option_ids=[rec.selected_option_id], material_to_decision=True)]
    result = RuleBasedVerifier().verify(benchmark.problems[0], rec)
    assert result.disposition == "human_review" and result.policy_violations == []


def test_abstention_is_review_not_invalid_or_reselected(benchmark: Benchmark) -> None:
    problem = benchmark.problems[8]  # Baseline tie abstention is wrong by the evaluator.
    rec = RuleBasedStrategy().decide(problem)
    assert rec.status == "abstained"
    result = RuleBasedVerifier().verify(problem, rec)
    assert result.disposition == "human_review" and result.structurally_valid
    assert result.selected_option_valid is None
    assert result.evidence_sufficiency == "not_applicable"
    assert rec.status == "abstained"


def test_no_semantic_support_claim_for_existing_citations(benchmark: Benchmark) -> None:
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.explanation = "Unsupported arbitrary assertion."
    result = RuleBasedVerifier().verify(benchmark.problems[0], rec)
    assert result.disposition == "accept"
    assert result.evidence_support == "not_assessed"


def test_pipeline_baseline_metrics_unchanged(benchmark: Benchmark) -> None:
    report = run_benchmark(benchmark, RuleBasedStrategy(), strategy_name="rules", strategy_version="1")
    # Reproduce the original execution -> evaluator path without verification.
    original = [evaluate_observation(
        problem, benchmark.expectations[problem.id],
        run_decision(problem, RuleBasedStrategy(), RunMetadata(
            strategy="rules", strategy_version="1", dataset_version=benchmark.manifest.version,
        ), f"rules:{problem.id}:1"),
    ) for problem in benchmark.problems]
    original_report = build_report(benchmark, original, "rules", "1", 1)
    assert report.summary == original_report.summary
    assert report.by_category == original_report.by_category
    assert report.by_split == original_report.by_split
    assert report.pair_rates == original_report.pair_rates
    assert report.summary == summarize(report.cases)
    assert report.verification_summary.verified == 30
    assert report.verification_summary.accepted == 14
    assert report.verification_summary.human_review == 16
    assert report.verification_summary.rejected == 0
    assert all(c.observation.verification for c in report.cases)
    for before, after in zip(original_report.cases, report.cases, strict=True):
        assert before.observation.run == after.observation.run


def test_comparison_verifies_all_strategies_without_redecision(benchmark: Benchmark) -> None:
    specs, direct, structured = entries(benchmark)
    report = run_comparison(benchmark, specs)
    assert len(direct.requests) == len(structured.requests) == 30
    for column in report.columns.values():
        assert column.report.verification_summary.verified == 30
        assert column.report.summary.rates["outcome_correctness"].value == 0.9
    assert "Verifier accepted" in comparison_text(report)
    assert "rules-verifier" in comparison_json(report)


def test_rejected_candidate_verified_but_unparseable_is_unavailable(benchmark: Benchmark) -> None:
    from test_direct_llm import response
    from test_direct_parser import format_recommendation
    specs, direct, _ = entries(benchmark)
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    rec.assessments[0].evidence_ids = ["invented"]
    direct.events[0] = response(format_recommendation(rec))
    direct.events[1] = response("invalid text")
    column = run_comparison(benchmark, specs).columns["direct-llm"]
    assert column.report.verification_summary.verified == 29
    assert column.report.verification_summary.unavailable == 1
    assert column.report.cases[0].observation.verification.disposition == "reject"
    assert column.report.cases[0].outcome == "failure"
    assert column.report.cases[1].observation.verification is None


def test_custom_verifier_isolated_and_failures_do_not_change_scores(benchmark: Benchmark) -> None:
    class BrokenVerifier:
        def verify(self, problem: DecisionProblem, recommendation: RecommendationInput) -> VerificationResult:
            problem.facts.clear()
            if isinstance(recommendation, Recommendation):
                recommendation.selected_option_id = "mutated"
            raise RuntimeError("secret-message")

    report = run_benchmark(benchmark, RuleBasedStrategy(), strategy_name="rules", strategy_version="1", verifier=BrokenVerifier())
    assert report.summary.rates["outcome_correctness"].value == 0.9
    assert report.summary.counts["failed_runs"] == 0
    assert report.verification_summary.errors == 30
    assert "secret-message" not in report.model_dump_json()
    assert report.cases[0].observation.run.recommendation.selected_option_id == "q-r"
    assert all(p.facts for p in benchmark.problems)


def test_mismatched_custom_verifier_result_is_diagnostic_error(benchmark: Benchmark) -> None:
    class WrongProblemVerifier:
        def verify(self, problem: DecisionProblem, recommendation: RecommendationInput) -> VerificationResult:
            result = RuleBasedVerifier().verify(problem, recommendation)
            result.problem_id = "wrong-problem"
            return result

    report = run_benchmark(benchmark, RuleBasedStrategy(), strategy_name="rules", strategy_version="1", verifier=WrongProblemVerifier())
    assert report.verification_summary.errors == 30
    assert report.summary.rates["outcome_correctness"].value == 0.9


def test_contradictory_verification_results_are_rejected(benchmark: Benchmark) -> None:
    from pydantic import ValidationError
    rec = RuleBasedStrategy().decide(benchmark.problems[0])
    valid = RuleBasedVerifier().verify(benchmark.problems[0], rec).model_dump()
    for update in ({"structurally_valid": False}, {"requires_human_review": True},
                   {"references_valid": False}, {"evidence_sufficiency": "insufficient"},
                   {"policy_violations": ["wrong_route"]}):
        with pytest.raises(ValidationError):
            VerificationResult.model_validate({**valid, **update})


def test_evaluation_cli_includes_separate_verifier_output() -> None:
    from test_cli import invoke
    import json
    result = invoke("evaluate", "--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["summary"]["rates"]["outcome_correctness"]["value"] == 0.9
    assert report["verification_summary"]["accepted"] == 14
    assert report["cases"][0]["observation"]["verification"]["disposition"] == "accept"
    human = invoke("compare", "--strategies", "rules")
    assert human.returncode == 0 and "Verifier accepted" in human.stdout
