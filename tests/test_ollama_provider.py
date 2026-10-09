import json
from urllib.error import HTTPError, URLError

import pytest
from pydantic import ValidationError

from decisionlab import cli
from decisionlab.errors import DecisionExecutionError
from decisionlab.models import RunMetadata
from decisionlab.providers.factory import load_provider_factory, provider_from_environment
from decisionlab.providers.ollama import LocalHTTPTransport, OllamaProvider, RejectRedirects
from decisionlab.runner import run_decision
from decisionlab.settings import OllamaSettings
from decisionlab.strategies import DirectLLMStrategy, RuleBasedStrategy, StructuredLLMStrategy
from decisionlab.strategies.structured_llm import StructuredRecommendation
from test_direct_parser import format_recommendation


class FakeTransport:
    def __init__(self, events):
        self.events = events
        self.requests = []

    def request(self, url, payload, *, timeout):
        self.requests.append((url, payload, timeout))
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def envelope(content: str = "Hello", **changes) -> str:
    return json.dumps({
        "model": "qwen2.5:14b", "done": True, "done_reason": "stop",
        "created_at": "2026-10-08T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10, "eval_count": 5, "total_duration": 250_000_000,
        "load_duration": 100_000_000, **changes,
    })


@pytest.mark.parametrize("url", ["http://localhost:11434/", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_local_settings(url: str) -> None:
    settings = OllamaSettings(model="qwen2.5:14b", base_url=url)
    assert settings.base_url.startswith("http://")
    assert "api_key" not in settings.model_dump()


@pytest.mark.parametrize("url", [
    "https://example.com", "http://192.168.1.1:11434", "http://evil.example",
    "http://127.0.0.1.evil.example", "http://user:secret@localhost:11434",
    "http://localhost:11434/api", "http://localhost:11434?token=secret",
    "http://localhost:11434#x", "http://localhost:0", "http://localhost:99999",
])
def test_external_or_credential_endpoints_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        OllamaSettings(model="qwen2.5:14b", base_url=url)


def test_environment_selection_and_snapshot(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DECISIONLAB_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    factory = load_provider_factory()
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    assert factory().model_id == "qwen2.5:14b"
    assert provider_from_environment().model_id == "llama3.1:8b"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://user:secret@evil.example")
    with pytest.raises(ValueError, match="Invalid Ollama") as error:
        OllamaSettings.from_environment()
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("model", ["qwen:cloud", "some-cloud-model", "remote/model"])
def test_cloud_model_tags_rejected(model: str) -> None:
    with pytest.raises(ValidationError):
        OllamaSettings(model=model)


def test_text_and_schema_modes_and_metadata() -> None:
    transport = FakeTransport([envelope(), envelope('{}')])
    provider = OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport)
    text = provider.generate_text(instructions="system", problem_json="input")
    structured = provider.generate(instructions="system", problem_json="input", output_model=StructuredRecommendation)
    assert "format" not in transport.requests[0][1]
    assert transport.requests[1][1]["format"] == StructuredRecommendation.model_json_schema()
    for url, payload, timeout in transport.requests:
        assert url == "http://127.0.0.1:11434/api/chat"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "input"}]
        assert payload["options"] == {"temperature": 0, "num_predict": 2000}
        assert timeout == 180
    assert text.content_text == "Hello"
    assert text.latency_ms == structured.latency_ms == 250
    assert text.usage.input_tokens == 10 and text.usage.output_tokens == 5
    assert text.response_metadata["load_duration"] == 100_000_000
    assert text.response_id is None


@pytest.mark.parametrize("kind", ["direct", "structured"])
def test_strategy_validation_provenance_and_usage(benchmark, kind: str) -> None:
    problem = benchmark.problems[0]
    rec = RuleBasedStrategy().decide(problem)
    content = format_recommendation(rec) if kind == "direct" else rec.model_dump_json()
    transport = FakeTransport([envelope(content)])
    provider = OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport)
    strategy = DirectLLMStrategy(provider) if kind == "direct" else StructuredLLMStrategy(provider)
    observed = run_decision(problem, strategy, RunMetadata(strategy=kind, strategy_version="1"), "test")
    assert observed.run.status == "success"
    assert observed.run.metadata.provider == "ollama"
    assert observed.run.metadata.prompt_version
    assert observed.run.latency_ms == 250
    assert observed.run.response_metadata["created_at"] == "2026-10-08T00:00:00Z"
    assert observed.run.usage.input_tokens == 10
    assert json.loads(transport.requests[0][1]["messages"][1]["content"]) == problem.model_dump(mode="json")
    assert "acceptable_option_ids" not in transport.requests[0][1]["messages"][1]["content"]


@pytest.mark.parametrize("content", ["{}", "not JSON", '{"status":"selected","unexpected":1}'])
def test_invalid_structured_content_is_not_repaired(benchmark, content: str) -> None:
    provider = OllamaProvider(OllamaSettings(model="qwen2.5:14b"), FakeTransport([envelope(content)]))
    with pytest.raises(DecisionExecutionError, match="invalid_schema"):
        StructuredLLMStrategy(provider).decide(benchmark.problems[0])


@pytest.mark.parametrize("event,code", [
    (TimeoutError(), "provider_timeout"), (URLError("private"), "provider_connection_error"),
    (URLError(TimeoutError()), "provider_timeout"), (OSError("private"), "provider_connection_error"),
    (HTTPError("http://local", 500, "private", None, None), "provider_api_error"),
    ("not json", "provider_malformed_response"), (envelope(eval_count=-1), "provider_malformed_response"),
])
def test_errors_are_safe_and_never_retried(event, code: str) -> None:
    transport = FakeTransport([event])
    provider = OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport)
    with pytest.raises(DecisionExecutionError) as error:
        provider.generate_text(instructions="secret prompt", problem_json="secret input")
    assert error.value.code == code
    assert "secret" not in str(error.value) and "private" not in str(error.value)
    assert len(transport.requests) == 1 and provider.retry_count == 0


def test_incomplete_and_missing_measurements() -> None:
    provider = OllamaProvider(OllamaSettings(model="qwen2.5:14b"), FakeTransport([
        envelope(done_reason="length"), envelope(done=False),
        envelope(prompt_eval_count=None, eval_count=None, total_duration=None),
    ]))
    assert provider.generate_text(instructions="s", problem_json="p").status == "incomplete"
    assert provider.generate_text(instructions="s", problem_json="p").status == "incomplete"
    response = provider.generate_text(instructions="s", problem_json="p")
    assert response.latency_ms is response.usage is None


def test_preflight_no_generation_and_missing_model() -> None:
    tags = json.dumps({"models": [{"name": "qwen2.5:14b", "digest": "abc"}]})
    transport = FakeTransport([tags, '{"models":[]}'])
    provider = OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport)
    assert provider.preflight()["model_digest"] == "abc"
    assert transport.requests[0][0].endswith("/api/tags") and transport.requests[0][1] is None
    with pytest.raises(ValueError, match="not installed"):
        provider.preflight()
    assert len(transport.requests) == 2


def test_transport_bypasses_proxies_and_rejects_redirects(monkeypatch) -> None:
    import decisionlab.providers.ollama as module
    handlers = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{}'
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:11434/api/chat"
            assert "Authorization" not in request.headers
            return Response()
    def build(*items):
        handlers.extend(items)
        return Opener()
    monkeypatch.setattr(module, "build_opener", build)
    assert LocalHTTPTransport().request("http://127.0.0.1:11434/api/chat", {}, timeout=1) == "{}"
    assert handlers[0].proxies == {}
    assert isinstance(handlers[1], RejectRedirects)
    with pytest.raises(URLError):
        handlers[1].redirect_request(None, None, 302, "redirect", None, "https://cloud.example")


def test_cli_preflight_and_explicit_provider_guard(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    monkeypatch.setattr(OllamaProvider, "preflight", lambda self: {"model": self.model_id, "available": True})
    assert cli.main(["preflight", "--provider", "ollama"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "qwen2.5:14b"
    assert cli.main(["compare", "--provider", "ollama", "--strategies", "direct-llm"]) == 2
    assert "allow-live" in capsys.readouterr().err


@pytest.mark.parametrize("url", ["http://cloud.example/api/chat", "http://127.0.0.1:11434/api/pull"])
def test_transport_rejects_external_and_management_endpoints(url: str) -> None:
    with pytest.raises(ValueError):
        LocalHTTPTransport().request(url, {}, timeout=1)


def test_cli_ollama_comparison_shared_settings_without_key(monkeypatch, capsys) -> None:
    from decisionlab.models import DecisionProblem
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DECISIONLAB_PROVIDER", "openai")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0")
    calls = []
    def request(self, url, payload, *, timeout):
        calls.append((url, payload))
        problem = DecisionProblem.model_validate_json(payload["messages"][1]["content"])
        rec = RuleBasedStrategy().decide(problem)
        return envelope(rec.model_dump_json() if "format" in payload else format_recommendation(rec))
    monkeypatch.setattr(LocalHTTPTransport, "request", request)
    assert cli.main(["compare", "--provider", "ollama", "--strategies", "rules", "direct-llm", "structured-llm",
                     "--allow-live", "--split", "dev", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 40
    assert all(url == "http://127.0.0.1:11434/api/chat" for url, _ in calls)
    assert all("format" not in p for _, p in calls[:20])
    assert all("format" in p for _, p in calls[20:])
    assert report["comparability_warnings"] == []
    for name in ("direct-llm", "structured-llm"):
        column = report["columns"][name]
        assert column["metadata_profiles"][0]["provider"] == "ollama"
        assert column["report"]["summary"]["mean_latency_ms"] == 250
        assert column["report"]["summary"]["total_input_tokens"] == 200
        assert column["report"]["cases"][0]["observation"]["run"]["response_metadata"]["local_validation_required"] is True


def test_cli_preflight_connection_failure_is_safe(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    def fail(self):
        raise DecisionExecutionError("provider_connection_error")
    monkeypatch.setattr(OllamaProvider, "preflight", fail)
    assert cli.main(["preflight"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "provider_connection_error" in output.err
