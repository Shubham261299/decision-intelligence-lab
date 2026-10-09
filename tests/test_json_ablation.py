import json
from pathlib import Path

import pytest

from decisionlab import cli
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import RunMetadata
from decisionlab.providers.ollama import OllamaProvider
from decisionlab.runner import run_decision
from decisionlab.settings import OllamaSettings
from decisionlab.strategies.json_ablation import CONDITIONS, MatchedJSONStrategy
from decisionlab.strategies.rules import RuleBasedStrategy
from decisionlab.strategies.structured_llm import StructuredRecommendation
from test_ollama_provider import FakeTransport, envelope


def test_matched_http_requests_and_metadata(benchmark: Benchmark) -> None:
    problem = benchmark.problems[0]
    recommendation = RuleBasedStrategy().decide(problem)
    payloads = []
    settings = []
    for condition in CONDITIONS:
        transport = FakeTransport([envelope(recommendation.model_dump_json())])
        strategy = MatchedJSONStrategy(
            OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport), condition,
        )
        observation = run_decision(
            problem, strategy, RunMetadata(strategy=condition, strategy_version="1"), "test",
        )
        assert observation.run.status == "success"
        assert observation.run.recommendation == recommendation
        assert observation.schema_valid and observation.references_valid
        metadata = observation.run.metadata
        assert metadata.strategy == condition
        assert metadata.prompt_version == "matched-json-v1"
        assert metadata.provider == "ollama" and metadata.model_id == "qwen2.5:14b"
        assert metadata.generation_settings["condition"] == condition
        assert metadata.generation_settings["contract_version"] == "recommendation-transport-v1"
        assert observation.run.usage.input_tokens == 10
        assert observation.run.latency_ms == 250
        assert len(transport.requests) == 1
        url, payload, timeout = transport.requests[0]
        assert url == "http://127.0.0.1:11434/api/chat" and timeout == 180
        payloads.append(payload)
        settings.append(metadata.generation_settings)
    assert "format" not in payloads[0]
    assert payloads[1]["format"] == StructuredRecommendation.model_json_schema()
    assert payloads[0] == {k: v for k, v in payloads[1].items() if k != "format"}
    assert payloads[0]["options"] == {"temperature": 0, "num_predict": 2000}
    messages = payloads[0]["messages"]
    assert messages[1]["content"] == problem.model_dump_json()
    assert json.loads(messages[0]["content"].splitlines()[-1]) == payloads[1]["format"]
    assert "acceptable_option_ids" not in json.dumps(messages)
    assert {k: v for k, v in settings[0].items() if k != "condition"} == {
        k: v for k, v in settings[1].items() if k != "condition"
    }


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("kind", ["malformed", "fenced", "domain", "extra", "reference"])
def test_validation_failures_are_preserved(benchmark: Benchmark, condition: str, kind: str) -> None:
    problem = benchmark.problems[0]
    content = RuleBasedStrategy().decide(problem).model_dump(mode="json")
    if kind == "domain":
        content["confidence"] = 1.1
    elif kind == "extra":
        content["invented"] = True
    elif kind == "reference":
        content["assessments"][0]["evidence_ids"] = ["invented-fact"]
    raw = json.dumps(content)
    if kind == "malformed":
        raw = "not JSON"
    elif kind == "fenced":
        raw = "```json\n" + raw + "\n```"
    strategy = MatchedJSONStrategy(
        OllamaProvider(OllamaSettings(model="qwen2.5:14b"), FakeTransport([envelope(raw)])),
        condition,
    )
    observed = run_decision(problem, strategy, RunMetadata(strategy=condition, strategy_version="1"), "test")
    assert observed.run.status == "failure" and observed.run.recommendation is None
    assert observed.run.error.code == ("invalid_references" if kind == "reference" else "invalid_schema")
    assert observed.run.metadata.generation_settings["condition"] == condition
    assert observed.run.usage.output_tokens == 5
    if kind == "reference":
        assert observed.schema_valid and observed.references_valid is False
        assert observed.candidate.assessments[0].evidence_ids == ["invented-fact"]
    else:
        assert observed.schema_valid is False


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("event,code", [(TimeoutError(), "provider_timeout"), (envelope(done_reason="length"), "provider_incomplete")])
def test_provider_failures_retain_condition(benchmark: Benchmark, condition: str, event: object, code: str) -> None:
    transport = FakeTransport([event])
    strategy = MatchedJSONStrategy(OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport), condition)
    observed = run_decision(benchmark.problems[0], strategy, RunMetadata(strategy=condition, strategy_version="1"), "test")
    assert observed.run.status == "failure" and observed.run.error.code == code
    assert observed.run.metadata.generation_settings["condition"] == condition
    assert len(transport.requests) == 1


def test_unknown_condition_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        MatchedJSONStrategy(OllamaProvider(OllamaSettings(model="qwen2.5:14b"), FakeTransport([])), "unknown")


def test_cli_opt_in_precedes_provider_loading(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError("Provider must not be loaded")
    monkeypatch.setattr("decisionlab.providers.factory.load_provider_factory", forbidden)
    assert cli.main(["ablate"]) == 2
    assert "--allow-live" in capsys.readouterr().err


@pytest.mark.parametrize("bad_output", [False, True])
def test_cli_mocked_pipeline(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, bad_output: bool) -> None:
    from decisionlab.models import DecisionProblem

    requests = []
    class Transport:
        def request(self, url: str, payload: dict, *, timeout: float) -> str:
            requests.append(payload)
            problem = DecisionProblem.model_validate_json(payload["messages"][1]["content"])
            raw = RuleBasedStrategy().decide(problem).model_dump_json()
            return envelope("not JSON" if bad_output and "format" not in payload else raw)

    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.setattr("decisionlab.providers.ollama.LocalHTTPTransport", Transport)
    assert cli.main(["ablate", "--allow-live", "--split", "dev", "--json", "--output", str(tmp_path)]) == (1 if bad_output else 0)
    report = json.loads(capsys.readouterr().out)
    assert report["execution_order"] == list(CONDITIONS)
    assert len(requests) == 40
    for a, b in zip(requests[:20], requests[20:]):
        assert "format" not in a
        assert a == {k: v for k, v in b.items() if k != "format"}
    for condition in CONDITIONS:
        column = report["columns"][condition]
        cases = column["report"]["cases"]
        assert len(cases) == 20
        assert all(c["observation"]["run"]["metadata"]["generation_settings"]["condition"] == condition for c in cases)
        if bad_output and condition == "unconstrained-json":
            assert column["failure_codes"] == {"invalid_schema": 20}
            assert all(c["observation"]["verification"] is None for c in cases)
        else:
            assert all(c["observation"]["verification"] is not None for c in cases)
            assert column["report"]["summary"]["rates"]["outcome_correctness"]["value"] == 0.9
    assert json.loads((tmp_path / "comparison.json").read_text()) == report
    assert all(c in (tmp_path / "comparison.txt").read_text() for c in CONDITIONS)
