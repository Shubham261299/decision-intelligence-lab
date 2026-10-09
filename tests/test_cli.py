import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from decisionlab import cli
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import DecisionProblem, Recommendation
from decisionlab.runner import run_benchmark
from conftest import BENCHMARK_ROOT

SOURCE = Path(__file__).resolve().parents[1] / "src"


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "decisionlab.cli", *args],
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        capture_output=True, text=True, check=False,
    )


def test_cli_json_is_clean_and_reproducible(tmp_path: Path) -> None:
    args = ("evaluate", "--benchmark", str(BENCHMARK_ROOT), "--json", "--output", str(tmp_path))
    first, second = invoke(*args), invoke(*args)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == ""
    assert json.loads(first.stdout)["summary"]["counts"]["total"] == 30
    assert (tmp_path / "report.json").exists()


def test_cli_human_output_and_split() -> None:
    result = invoke("evaluate", "--benchmark", str(BENCHMARK_ROOT), "--split", "held_out")
    assert result.returncode == 0
    assert "Cases/runs: 10" in result.stdout
    assert "Outcome correctness: 9/10" in result.stdout


def test_cli_configuration_failures_use_stderr(tmp_path: Path) -> None:
    for args in (
        ("--benchmark", str(tmp_path / "missing")),
        ("--benchmark", str(BENCHMARK_ROOT), "--repeats", "0"),
        ("--benchmark", str(BENCHMARK_ROOT), "--split", "invalid"),
    ):
        result = invoke("evaluate", *args)
        assert result.returncode == 2
        assert not result.stdout
        assert result.stderr


def test_cli_execution_failures_are_nonzero_but_still_reported(
    benchmark: Benchmark, monkeypatch, capsys,
) -> None:
    class BrokenStrategy:
        def decide(self, problem: DecisionProblem) -> Recommendation:
            raise RuntimeError("Broken")

    report = run_benchmark(benchmark, BrokenStrategy(), strategy_name="broken", strategy_version="1")
    monkeypatch.setattr(cli, "run_benchmark", lambda *args, **kwargs: report)
    assert cli.main(["evaluate", "--benchmark", str(BENCHMARK_ROOT), "--json"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["summary"]["counts"]["failed_runs"] == 30
    assert "execution failures" in output.err


def test_evaluator_can_be_imported_without_any_strategy_implementation() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import decisionlab.evaluation.evaluator; assert 'decisionlab.strategies.rules' not in sys.modules"],
        env={**os.environ, "PYTHONPATH": str(SOURCE)}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_llm_requires_explicit_live_opt_in() -> None:
    result = invoke("evaluate", "--strategy", "structured-llm")
    assert result.returncode == 2
    assert "requires --allow-live" in result.stderr
    assert result.stdout == ""


def test_cli_live_missing_configuration_is_safe(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert cli.main(["evaluate", "--benchmark", str(BENCHMARK_ROOT), "--strategy", "structured-llm", "--allow-live"]) == 2
    output = capsys.readouterr()
    assert "must be configured" in output.err
    assert output.out == ""


def test_cli_llm_evaluation_with_mocked_provider(benchmark, monkeypatch, capsys) -> None:
    from decisionlab.providers.openai import OpenAIProvider
    from decisionlab.strategies import RuleBasedStrategy
    from test_structured_llm import FakeProvider, response

    provider = FakeProvider([
        response(RuleBasedStrategy().decide(problem).model_dump_json())
        for problem in benchmark.problems[:20]
    ])
    monkeypatch.setattr(OpenAIProvider, "from_environment", lambda: provider)
    assert cli.main([
        "evaluate", "--benchmark", str(BENCHMARK_ROOT), "--strategy", "structured-llm",
        "--allow-live", "--split", "dev", "--json",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["strategy"] == "structured-llm"
    assert report["cases"][0]["observation"]["run"]["usage"]["input_tokens"] == 80


def test_cli_direct_requires_explicit_live_opt_in() -> None:
    result = invoke("evaluate", "--strategy", "direct-llm")
    assert result.returncode == 2
    assert "requires --allow-live" in result.stderr
    assert not result.stdout


def test_cli_direct_missing_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert cli.main(["evaluate", "--strategy", "direct-llm", "--allow-live"]) == 2
    output = capsys.readouterr()
    assert "must be configured" in output.err and not output.out


@pytest.mark.parametrize("bad_output", [False, True])
def test_cli_direct_mocked_provider_and_reports(
    benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tmp_path: Path, bad_output: bool,
) -> None:
    from decisionlab.providers.openai import OpenAIProvider
    from decisionlab.strategies import RuleBasedStrategy
    from test_direct_llm import FakeTextProvider, response
    from test_direct_parser import format_recommendation

    provider = FakeTextProvider([
        response("I choose routine." if bad_output else format_recommendation(
            RuleBasedStrategy().decide(problem),
        )) for problem in benchmark.problems[:20]
    ])
    monkeypatch.setattr(OpenAIProvider, "from_environment", lambda: provider)
    assert cli.main([
        "evaluate", "--strategy", "direct-llm", "--allow-live", "--split", "dev",
        "--json", "--output", str(tmp_path),
    ]) == (1 if bad_output else 0)
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["strategy"] == "direct-llm"
    assert report["summary"]["counts"]["failed_runs"] == (20 if bad_output else 0)
    assert report["cases"][0]["observation"]["run"]["usage"]["input_tokens"] == 80
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert "DecisionLab evaluation: direct-llm" in (tmp_path / "report.txt").read_text()
    if bad_output:
        assert "execution failures" in output.err
