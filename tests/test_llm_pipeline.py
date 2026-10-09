"""Mock the HTTP boundary while exercising adapter, strategy, runner and evaluator."""

import io
import json
from urllib.error import HTTPError

import pytest

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.evaluator import evaluate_observation
from decisionlab.models import RunMetadata
from decisionlab.providers.openai import OpenAIProvider
from decisionlab.runner import run_decision
from decisionlab.strategies import RuleBasedStrategy, StructuredLLMStrategy
from test_openai_provider import envelope


def configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "requested-model")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)


def test_mocked_http_end_to_end(benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    problem = benchmark.problems[0]
    content = RuleBasedStrategy().decide(problem).model_dump_json()
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(envelope(output=[{
            "type": "message", "content": [{"type": "output_text", "text": content}],
        }]).encode())

    monkeypatch.setattr("decisionlab.providers.openai.urlopen", fake_urlopen)
    strategy = StructuredLLMStrategy(OpenAIProvider.from_environment())
    observed = run_decision(problem, strategy, RunMetadata(strategy=strategy.name, strategy_version="1"), "r")
    result = evaluate_observation(problem, benchmark.expectations[problem.id], observed)
    assert result.outcome_correct
    assert observed.run.metadata.model_id == "model-snapshot"
    assert observed.run.usage.input_tokens == 100
    assert json.loads(captured["input"][0]["content"]) == problem.model_dump(mode="json")
    assert "test-secret" not in observed.model_dump_json()
    assert "acceptable_option_ids" not in captured["input"][0]["content"]


def test_failed_api_call_retains_retry_metadata_without_secret_logs(
    benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch, caplog,
) -> None:
    configure(monkeypatch)
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    monkeypatch.setattr("decisionlab.providers.openai.time.sleep", lambda seconds: None)
    calls = []

    def failing_urlopen(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 503, "test-secret server body", {}, None)

    monkeypatch.setattr("decisionlab.providers.openai.urlopen", failing_urlopen)
    strategy = StructuredLLMStrategy(OpenAIProvider.from_environment())
    observed = run_decision(benchmark.problems[0], strategy, RunMetadata(strategy=strategy.name, strategy_version="1"), "r")
    assert observed.run.status == "failure"
    assert observed.run.error.code == "provider_api_error"
    assert observed.run.metadata.retry_count == 1
    assert len(calls) == 2
    assert observed.run.usage is None
    assert "test-secret" not in observed.model_dump_json()
    assert "test-secret" not in caplog.text
