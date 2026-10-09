import io
import json
from urllib.error import HTTPError, URLError

import pytest
from pydantic import JsonValue

from decisionlab.errors import DecisionExecutionError
from decisionlab.providers.openai import OpenAIProvider, UrllibTransport
from decisionlab.settings import OpenAISettings
from decisionlab.strategies.structured_llm import StructuredRecommendation


def envelope(**changes: object) -> str:
    data = {
        "id": "resp-test", "model": "model-snapshot", "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    }
    data.update(changes)
    return json.dumps(data)


class FakeTransport:
    def __init__(self, events: list[str | Exception]) -> None:
        self.events = events
        self.payloads: list[dict[str, JsonValue]] = []

    def send(self, payload: dict[str, JsonValue], *, api_key: str, timeout: float) -> str:
        assert api_key == "test-secret"
        self.payloads.append(payload)
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def provider(transport: FakeTransport, retries: int = 0) -> OpenAIProvider:
    return OpenAIProvider(OpenAISettings(
        api_key="test-secret", model="model-requested", max_retries=retries,
    ), transport)


def generate(adapter: OpenAIProvider):
    return adapter.generate(
        instructions="Test policy", problem_json='{"id":"test-problem"}',
        output_model=StructuredRecommendation,
    )


def test_request_schema_and_usage_are_normalized() -> None:
    transport = FakeTransport([envelope()])
    adapter = provider(transport)
    response = generate(adapter)
    assert response.model_id == "model-snapshot"
    assert response.response_id == "resp-test"
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 50
    request = transport.payloads[0]
    assert request["model"] == "model-requested"
    assert request["store"] is False
    assert "temperature" not in request
    assert "api_key" not in json.dumps(request)
    schema = request["text"]["format"]
    assert schema["strict"] is True and schema["type"] == "json_schema"
    assert set(schema["schema"]["required"]) == set(schema["schema"]["properties"])
    for nested in schema["schema"]["$defs"].values():
        assert nested["additionalProperties"] is False
        assert set(nested["required"]) == set(nested["properties"])


def test_optional_temperature_is_sent_only_when_configured() -> None:
    transport = FakeTransport([envelope()])
    adapter = OpenAIProvider(OpenAISettings(api_key="test-secret", model="test-model", temperature=0), transport)
    generate(adapter)
    assert transport.payloads[0]["temperature"] == 0


@pytest.mark.parametrize("event,code", [
    (TimeoutError("secret"), "provider_timeout"),
    (URLError(TimeoutError("secret")), "provider_timeout"),
    (URLError("secret"), "provider_connection_error"),
    (ConnectionError("secret"), "provider_connection_error"),
    (HTTPError("https://api.openai.com", 401, "secret", {}, None), "provider_api_error"),
    (HTTPError("https://api.openai.com", 429, "secret", {}, None), "provider_api_error"),
])
def test_transport_errors_are_explicit_and_sanitized(event: Exception, code: str) -> None:
    with pytest.raises(DecisionExecutionError) as captured:
        generate(provider(FakeTransport([event])))
    assert captured.value.code == code
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize("event", [TimeoutError(), URLError("unavailable"), HTTPError("https://api.openai.com", 503, "Unavailable", {}, None)])
def test_transient_retry_is_bounded_and_recorded(event: Exception, monkeypatch) -> None:
    monkeypatch.setattr("decisionlab.providers.openai.time.sleep", lambda seconds: None)
    transport = FakeTransport([event, envelope()])
    adapter = provider(transport, retries=1)
    assert generate(adapter).status == "completed"
    assert len(transport.payloads) == 2
    assert adapter.retry_count == 1


def test_retries_stop_at_limit(monkeypatch) -> None:
    monkeypatch.setattr("decisionlab.providers.openai.time.sleep", lambda seconds: None)
    transport = FakeTransport([TimeoutError(), TimeoutError(), envelope()])
    adapter = provider(transport, retries=1)
    with pytest.raises(DecisionExecutionError, match="provider_timeout"):
        generate(adapter)
    assert len(transport.payloads) == 2
    assert len(transport.events) == 1


def test_nontransient_error_is_not_retried() -> None:
    transport = FakeTransport([HTTPError("https://api.openai.com", 400, "Bad", {}, None), envelope()])
    with pytest.raises(DecisionExecutionError):
        generate(provider(transport, retries=2))
    assert len(transport.payloads) == 1


@pytest.mark.parametrize("raw", ["not-json secret", "{}", envelope(usage={"input_tokens": -1, "output_tokens": 1}), envelope(usage={"input_tokens": True, "output_tokens": 1})])
def test_malformed_api_envelopes_fail(raw: str) -> None:
    with pytest.raises(DecisionExecutionError, match="provider_malformed_response"):
        generate(provider(FakeTransport([raw])))


@pytest.mark.parametrize("changes,status", [
    ({"status": "incomplete", "output": []}, "incomplete"),
    ({"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "Sensitive content"}]}]}, "refused"),
    ({"output": []}, "malformed"),
    ({"status": "failed", "output": []}, "failed"),
])
def test_unsuccessful_model_responses_preserve_usage(changes: dict, status: str) -> None:
    response = generate(provider(FakeTransport([envelope(**changes)])))
    assert response.status == status
    assert response.usage.input_tokens == 100


def test_reasoning_items_are_not_treated_as_output() -> None:
    payload = json.loads(envelope())
    payload["output"].insert(0, {"type": "reasoning", "summary": []})
    assert generate(provider(FakeTransport([json.dumps(payload)]))).content_json == "{}"


def test_multiple_text_outputs_are_not_silently_merged() -> None:
    payload = json.loads(envelope())
    payload["output"][0]["content"].append({"type": "output_text", "text": "{}"})
    assert generate(provider(FakeTransport([json.dumps(payload)]))).status == "malformed"


def test_http_transport_endpoint_headers_and_timeout(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(url=request.full_url, headers=dict(request.header_items()), timeout=timeout, body=json.loads(request.data))
        return io.BytesIO(b'{"status":"completed"}')

    monkeypatch.setattr("decisionlab.providers.openai.urlopen", fake_urlopen)
    assert UrllibTransport().send({"store": False}, api_key="test-secret", timeout=7) == '{"status":"completed"}'
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["timeout"] == 7
