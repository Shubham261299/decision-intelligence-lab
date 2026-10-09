import io
import json
from urllib.error import HTTPError, URLError

import pytest

from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.evaluator import evaluate_observation
from decisionlab.models import RunMetadata
from decisionlab.providers.openai import OpenAIProvider
from decisionlab.runner import run_decision
from decisionlab.settings import OpenAISettings
from decisionlab.strategies import DirectLLMStrategy
from decisionlab.strategies.structured_llm import StructuredRecommendation
from test_direct_parser import TEXT
from test_llm_pipeline import configure
from test_openai_provider import FakeTransport, envelope, provider


def text_envelope(text: str = TEXT) -> str:
    return envelope(output=[{
        "type": "message", "content": [{"type": "output_text", "text": text}],
    }])


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
def test_same_configuration_except_output_enforcement(temperature: float | None) -> None:
    transport = FakeTransport([text_envelope(), envelope()])
    adapter = OpenAIProvider(OpenAISettings(
        api_key="test-secret", model="same-model", temperature=temperature,
        timeout_seconds=15, max_output_tokens=1200, max_retries=1,
    ), transport)
    result = adapter.generate_text(instructions="policy", problem_json='{"id":"case"}')
    adapter.generate(
        instructions="policy", problem_json='{"id":"case"}',
        output_model=StructuredRecommendation,
    )
    direct, structured = transport.payloads
    assert direct["text"] == {"format": {"type": "text"}}
    assert structured["text"]["format"]["type"] == "json_schema"
    assert {k: v for k, v in direct.items() if k != "text"} == {
        k: v for k, v in structured.items() if k != "text"
    }
    assert "tools" not in direct and "response_format" not in direct
    assert result.content_text == TEXT
    assert result.model_id == "model-snapshot" and result.response_id == "resp-test"
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 50


@pytest.mark.parametrize("event,code", [
    (TimeoutError("test-secret"), "provider_timeout"),
    (URLError(TimeoutError("test-secret")), "provider_timeout"),
    (URLError("test-secret"), "provider_connection_error"),
    (ConnectionError("test-secret"), "provider_connection_error"),
    (HTTPError("https://api.openai.com", 401, "test-secret", {}, None), "provider_api_error"),
    (UnicodeError("test-secret"), "provider_malformed_response"),
])
def test_text_transport_errors_are_sanitized(event: Exception, code: str) -> None:
    with pytest.raises(DecisionExecutionError) as captured:
        provider(FakeTransport([event])).generate_text(instructions="policy", problem_json="{}")
    assert captured.value.code == code
    assert "test-secret" not in str(captured.value)


@pytest.mark.parametrize("changes,status", [
    ({"status": "incomplete", "output": []}, "incomplete"),
    ({"output": [{"type": "message", "content": [{"type": "refusal"}]}]}, "refused"),
    ({"output": []}, "malformed"),
    ({"status": "failed", "output": []}, "failed"),
    ({"output": [{"type": "message", "content": [
        {"type": "output_text", "text": TEXT}, {"type": "output_text", "text": TEXT},
    ]}]}, "malformed"),
])
def test_text_statuses_and_usage(changes: dict[str, object], status: str) -> None:
    result = provider(FakeTransport([envelope(**changes)])).generate_text(
        instructions="policy", problem_json="{}",
    )
    assert result.status == status and result.usage.input_tokens == 100


@pytest.mark.parametrize("raw", ["secret not JSON", "{}", envelope(usage={"input_tokens": -1})])
def test_text_malformed_envelope(raw: str) -> None:
    with pytest.raises(DecisionExecutionError, match="provider_malformed_response"):
        provider(FakeTransport([raw])).generate_text(instructions="policy", problem_json="{}")


def test_text_retry_bound_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("decisionlab.providers.openai.time.sleep", lambda seconds: None)
    transport = FakeTransport([TimeoutError(), text_envelope(), text_envelope()])
    adapter = provider(transport, retries=1)
    assert adapter.generate_text(instructions="policy", problem_json="{}").content_text == TEXT
    assert adapter.retry_count == 1
    adapter.generate_text(instructions="policy", problem_json="{}")
    assert adapter.retry_count == 0
    transport = FakeTransport([TimeoutError(), TimeoutError(), text_envelope()])
    with pytest.raises(DecisionExecutionError, match="provider_timeout"):
        provider(transport, retries=1).generate_text(instructions="policy", problem_json="{}")
    assert len(transport.payloads) == 2 and len(transport.events) == 1


@pytest.mark.parametrize("text,expected_status", [(TEXT, "success"), ("I choose q-r.", "failure")])
def test_mocked_http_direct_pipeline(
    benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch, text: str, expected_status: str,
) -> None:
    configure(monkeypatch)
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float) -> io.BytesIO:
        captured.update(json.loads(request.data))
        return io.BytesIO(text_envelope(text).encode())

    monkeypatch.setattr("decisionlab.providers.openai.urlopen", fake_urlopen)
    problem = benchmark.problems[0]
    strategy = DirectLLMStrategy(OpenAIProvider.from_environment())
    observed = run_decision(
        problem, strategy, RunMetadata(strategy=strategy.name, strategy_version="1"), "r",
    )
    result = evaluate_observation(problem, benchmark.expectations[problem.id], observed)
    assert observed.run.status == expected_status
    assert result.outcome_correct is (expected_status == "success")
    assert captured["text"] == {"format": {"type": "text"}}
    assert json.loads(captured["input"][0]["content"]) == problem.model_dump(mode="json")
    assert observed.run.metadata.model_id == "model-snapshot"
    assert observed.run.usage.output_tokens == 50
    assert "test-secret" not in observed.model_dump_json()


def test_text_transport_failure_through_runner(
    benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    configure(monkeypatch)
    transport = FakeTransport([HTTPError("https://api.openai.com", 401, "test-secret", {}, None)])
    # Exercise the real adapter with a fake transport, avoiding network access.
    strategy = DirectLLMStrategy(provider(transport))
    observed = run_decision(benchmark.problems[0], strategy, RunMetadata(
        strategy=strategy.name, strategy_version="1",
    ), "r")
    assert observed.run.error.code == "provider_api_error"
    assert observed.run.usage is None
    assert "test-secret" not in observed.model_dump_json() + caplog.text
