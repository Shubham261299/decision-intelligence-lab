import io
import json
from pathlib import Path

import pytest

from decisionlab import cli
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.strategies import RuleBasedStrategy
from test_cli import invoke
from test_direct_parser import format_recommendation
from test_llm_pipeline import configure
from test_openai_provider import envelope


def test_offline_cli_default_and_reports(tmp_path: Path) -> None:
    result = invoke("compare", "--repeats", "2", "--json", "--output", str(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["execution_order"] == ["rules"]
    assert report["columns"]["rules"]["report"]["summary"]["counts"]["total"] == 60
    assert report["columns"]["rules"]["report"]["summary"]["rates"]["outcome_correctness"]["value"] == 0.9
    assert json.loads((tmp_path / "comparison.json").read_text()) == report
    assert "Overall correctness" in (tmp_path / "comparison.txt").read_text()


@pytest.mark.parametrize("args", [
    ["--strategies", "rules", "direct-llm", "structured-llm"],
    ["--strategies", "structured-llm"],
    ["--strategies", "rules", "rules"], ["--repeats", "0"],
    ["--benchmark", "missing-benchmark"],
])
def test_cli_configuration_errors(args: list[str]) -> None:
    result = invoke("compare", *args)
    assert result.returncode == 2 and not result.stdout and result.stderr


def test_live_cli_missing_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["compare", "--strategies", "direct-llm", "--allow-live"]) == 2
    assert "must be configured" in capsys.readouterr().err


@pytest.mark.parametrize("bad_text", [False, True])
def test_cli_full_three_strategy_http_mock(
    benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tmp_path: Path, bad_text: bool,
) -> None:
    configure(monkeypatch)
    requests = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        problem_json = payload["input"][0]["content"]
        from decisionlab.models import DecisionProblem
        recommendation = RuleBasedStrategy().decide(DecisionProblem.model_validate_json(problem_json))
        if payload["text"]["format"]["type"] == "text":
            content = "bad text" if bad_text else format_recommendation(recommendation)
        else:
            content = recommendation.model_dump_json()
        return io.BytesIO(envelope(output=[{"type": "message", "content": [
            {"type": "output_text", "text": content},
        ]}]).encode())

    monkeypatch.setattr("decisionlab.providers.openai.urlopen", fake_urlopen)
    assert cli.main([
        "compare", "--strategies", "rules", "direct-llm", "structured-llm",
        "--allow-live", "--split", "dev", "--json", "--output", str(tmp_path),
    ]) == (1 if bad_text else 0)
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["execution_order"] == ["rules", "direct-llm", "structured-llm"]
    assert report["columns"]["direct-llm"]["report"]["summary"]["counts"]["failed_runs"] == (20 if bad_text else 0)
    assert report["columns"]["structured-llm"]["report"]["summary"]["counts"]["failed_runs"] == 0
    assert len(requests) == 40
    assert [r["input"] for r in requests[:20]] == [r["input"] for r in requests[20:]]
    assert all(r["model"] == "requested-model" for r in requests)
    assert "test-secret" not in output.out
    if bad_text:
        assert "execution failures" in output.err
