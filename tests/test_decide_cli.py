"""Standalone execution tests use synthetic inputs, never benchmark expectations."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from decisionlab import cli
from decisionlab.evaluation.models import Observation
from decisionlab.models import DecisionProblem, Fact, Recommendation
from decisionlab.strategies.rules import RuleBasedStrategy
from decisionlab.verification import RuleBasedVerifier

ROOT = Path(__file__).resolve().parents[1]
ROUTINE = ROOT / "examples/decision-routine.json"
MISSING = ROOT / "examples/decision-missing-urgency.json"


@pytest.fixture
def problem() -> DecisionProblem:
    return DecisionProblem.model_validate_json(ROUTINE.read_text(encoding="utf-8"))


def write_problem(tmp_path: Path, problem: DecisionProblem) -> Path:
    path = tmp_path / "problem.json"
    path.write_text(problem.model_dump_json(), encoding="utf-8")
    return path


def test_default_selection_is_offline_and_independent_of_benchmark(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Standalone rules must not load providers or benchmark labels")

    monkeypatch.setattr(cli, "load_benchmark", forbidden)
    monkeypatch.setattr("decisionlab.providers.factory.provider_from_environment", forbidden)
    monkeypatch.setenv("DECISIONLAB_PROVIDER", "invalid-and-irrelevant-for-rules")
    monkeypatch.setenv("OLLAMA_BASE_URL", "invalid-and-irrelevant-for-rules")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)  # No evals/ or saved reports in this working directory.
    assert cli.main(["decide", "--input", str(ROUTINE)]) == 0
    first = capsys.readouterr()
    assert not first.err
    result = Observation.model_validate_json(first.out)
    assert result.run.status == "success"
    assert result.run.recommendation is not None
    assert result.run.recommendation.status == "selected"
    assert result.run.recommendation.selected_option_id == "routine"
    assert result.run.metadata.strategy == "rules"
    assert result.run.metadata.provider is None
    assert result.run.metadata.dataset_version is None
    assert result.run.policy_result is not None
    assert not result.run.policy_result.detected_violations
    assert result.verification is not None and result.verification.disposition == "accept"
    assert "expected_abstention" not in first.out and "acceptable_option_ids" not in first.out
    assert cli.main(["decide", "--input", str(ROUTINE), "--strategy", "rules"]) == 0
    assert capsys.readouterr().out == first.out


def test_module_entrypoint_from_another_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "decisionlab.cli", "decide", "--input", str(ROUTINE)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "DECISIONLAB_PROVIDER": "invalid"},
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0 and not completed.stderr
    assert json.loads(completed.stdout)["run"]["recommendation"]["selected_option_id"] == "routine"


def test_abstention_is_successful_execution_with_review(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["decide", "--input", str(MISSING)]) == 0
    output = capsys.readouterr()
    result = Observation.model_validate_json(output.out)
    assert result.run.status == "success" and result.run.error is None
    assert result.run.recommendation is not None
    assert result.run.recommendation.status == "abstained"
    assert result.run.recommendation.selected_option_id is None
    assert result.run.policy_result is not None and result.run.policy_result.requires_human_review
    assert result.verification is not None and result.verification.disposition == "human_review"
    assert not output.err


@pytest.mark.parametrize("kind", ["malformed", "array", "label_leak", "duplicate", "invalid_value"])
def test_invalid_input_is_sanitized_and_not_executed(
    tmp_path: Path, problem: DecisionProblem, kind: str,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    raw = problem.model_dump()
    if kind == "malformed":
        text = '{"secret-input-123":'
    elif kind == "array":
        text = json.dumps([raw])
    else:
        if kind == "label_leak":
            raw["expected_abstention"] = True
        elif kind == "duplicate":
            raw["options"].append(raw["options"][0])
        else:
            raw["facts"][0]["value"] = {"secret-input-123": "invalid scalar"}
        text = json.dumps(raw)
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")

    def forbidden(*args: object) -> None:
        raise AssertionError("Input validation must precede strategy creation")

    monkeypatch.setattr(cli, "_create_strategy", forbidden)
    assert cli.main(["decide", "--input", str(path)]) == 2
    output = capsys.readouterr()
    assert not output.out and "valid DecisionProblem" in output.err
    assert "secret-input-123" not in output.err


@pytest.mark.parametrize("exists", [False, True])
def test_unreadable_or_non_utf8_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], exists: bool,
) -> None:
    path = tmp_path / "input.json"
    if exists:
        path.write_bytes(b"\xff\xfe")
    assert cli.main(["decide", "--input", str(path)]) == 2
    output = capsys.readouterr()
    assert not output.out and "unable to read" in output.err


def test_utf8_bom_input(
    tmp_path: Path, problem: DecisionProblem, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "input.json"
    path.write_text(problem.model_dump_json(), encoding="utf-8-sig")
    assert cli.main(["decide", "--input", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["recommendation"]["status"] == "selected"


def test_strategy_exception_stays_failure_without_payload_leak(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class BrokenStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            raise RuntimeError("secret-error-123")

    monkeypatch.setattr(cli, "_create_strategy", lambda *args: (BrokenStrategy(), "broken", "1"))
    assert cli.main(["decide", "--input", str(ROUTINE)]) == 1
    output = capsys.readouterr()
    result = Observation.model_validate_json(output.out)
    assert result.run.status == "failure" and result.run.recommendation is None
    assert result.run.error is not None and result.run.error.code == "strategy_exception"
    assert result.verification is None
    assert "secret-error-123" not in output.out + output.err


@pytest.mark.parametrize("invalid", ["schema", "references", "traceability"])
def test_output_validation_and_verifier_findings(
    problem: DecisionProblem, invalid: str,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    rec = RuleBasedStrategy().decide(problem)
    if invalid == "schema":
        rec.selected_option_id = None  # Runner must revalidate mutated models.
    elif invalid == "references":
        rec.assessments[0].evidence_ids = ["invented-evidence"]
    else:
        rec.assessments.clear()

    class FixedStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            assert not hasattr(problem, "acceptable_option_ids")
            return rec

    monkeypatch.setattr(cli, "_create_strategy", lambda *args: (FixedStrategy(), "fixed", "1"))
    before = rec.model_dump_json()
    assert cli.main(["decide", "--input", str(ROUTINE)]) == (0 if invalid == "traceability" else 1)
    result = Observation.model_validate_json(capsys.readouterr().out)
    assert rec.model_dump_json() == before
    if invalid == "schema":
        assert result.schema_valid is False and result.verification is None
        assert result.run.error is not None and result.run.error.code == "invalid_schema"
    elif invalid == "references":
        assert result.references_valid is False and result.run.recommendation is None
        assert result.candidate is not None
        assert result.verification is not None and result.verification.disposition == "reject"
        assert any(f.code == "unknown_evidence_reference" for f in result.verification.findings)
    else:
        assert result.run.status == "success"
        assert result.verification is not None and result.verification.disposition == "human_review"
        assert any(f.code == "incomplete_selection_traceability" for f in result.verification.findings)


def test_policy_rejection_does_not_replace_selected_recommendation(
    tmp_path: Path, problem: DecisionProblem, capsys: pytest.CaptureFixture[str],
) -> None:
    problem.facts[0].source = "unverified-note"
    path = write_problem(tmp_path, problem)
    assert cli.main(["decide", "--input", str(path)]) == 1
    output = capsys.readouterr()
    result = Observation.model_validate_json(output.out)
    assert result.run.status == "success"  # Execution and policy rejection are separate.
    assert result.run.recommendation is not None and result.run.recommendation.status == "selected"
    assert result.run.policy_result is not None
    assert "unresolved_fact:urgent" in result.run.policy_result.detected_violations
    assert result.verification is not None and result.verification.disposition == "reject"
    assert "recommendation rejected" in output.err


def test_valid_selection_can_require_human_review(
    tmp_path: Path, problem: DecisionProblem, capsys: pytest.CaptureFixture[str],
) -> None:
    problem.facts.append(Fact(id="review_required", statement="Supervisor review", value=True, source="supervisor"))
    assert cli.main(["decide", "--input", str(write_problem(tmp_path, problem))]) == 0
    result = Observation.model_validate_json(capsys.readouterr().out)
    assert result.run.recommendation is not None
    assert result.run.recommendation.status == "selected" and not result.run.recommendation.requested_review
    assert result.run.policy_result is not None and result.run.policy_result.requires_human_review
    assert result.verification is not None and result.verification.disposition == "human_review"


def test_verifier_error_is_separate_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError("secret-verifier-123")

    monkeypatch.setattr(RuleBasedVerifier, "verify", broken)
    assert cli.main(["decide", "--input", str(ROUTINE)]) == 1
    output = capsys.readouterr()
    result = Observation.model_validate_json(output.out)
    assert result.run.status == "success" and result.run.recommendation is not None
    assert result.verification_error == "verifier_error"
    assert "secret-verifier-123" not in output.out + output.err


@pytest.mark.parametrize("strategy", ["direct-llm", "structured-llm"])
def test_llm_requires_opt_in_before_provider_creation(
    strategy: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError("Live strategy construction must be explicitly allowed")

    monkeypatch.setattr(cli, "_create_strategy", forbidden)
    assert cli.main(["decide", "--input", str(ROUTINE), "--strategy", strategy]) == 2
    output = capsys.readouterr()
    assert not output.out and "requires --allow-live" in output.err


@pytest.mark.parametrize("strategy", ["direct-llm", "structured-llm"])
def test_live_strategy_wiring_with_mocked_provider_only(
    strategy: str, problem: DecisionProblem,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from test_direct_llm import FakeTextProvider, response as text_response
    from test_direct_parser import format_recommendation
    from test_structured_llm import FakeProvider, response as structured_response

    rec = RuleBasedStrategy().decide(problem)
    provider = (
        FakeTextProvider([text_response(format_recommendation(rec))])
        if strategy == "direct-llm"
        else FakeProvider([structured_response(rec.model_dump_json())])
    )
    monkeypatch.setattr("decisionlab.providers.factory.provider_from_environment", lambda *args: provider)
    assert cli.main([
        "decide", "--input", str(ROUTINE), "--strategy", strategy,
        "--provider", "ollama", "--allow-live",
    ]) == 0
    result = Observation.model_validate_json(capsys.readouterr().out)
    assert result.run.metadata.strategy == strategy and result.run.usage is not None
    assert result.run.metadata.dataset_version is None
    assert json.loads(provider.requests[0]["problem_json"]) == problem.model_dump()


def test_offline_tool_assisted_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["decide", "--input", str(ROUTINE), "--strategy", "tool-assisted"]) == 0
    result = Observation.model_validate_json(capsys.readouterr().out)
    assert result.run.metadata.strategy == "tool-assisted"
    assert result.run.metadata.provider is None and len(result.tool_results) == 2
    assert result.verification is not None and result.verification.disposition == "accept"


def test_configuration_error_does_not_echo_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*args: object) -> None:
        raise ValueError("secret-setting-123")

    monkeypatch.setattr(cli, "_create_strategy", broken)
    assert cli.main([
        "decide", "--input", str(ROUTINE), "--strategy", "direct-llm", "--allow-live",
    ]) == 2
    output = capsys.readouterr()
    assert not output.out and "secret-setting-123" not in output.err
