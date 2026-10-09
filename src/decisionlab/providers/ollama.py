"""Local-only Ollama chat adapter; no keys, proxies, redirects or model pulls."""

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from decisionlab.errors import DecisionExecutionError, ErrorCode
from decisionlab.models import Scalar, Text, Usage
from decisionlab.providers.base import ProviderResponse, ProviderStatus, TextProviderResponse
from decisionlab.settings import OllamaSettings


class LocalTransport(Protocol):
    def request(self, url: str, payload: dict[str, JsonValue] | None, *, timeout: float) -> str: ...


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int,
                         msg: str, headers: object, newurl: str) -> None:
        raise URLError("Ollama redirects are disabled")


class LocalHTTPTransport:
    def request(self, url: str, payload: dict[str, JsonValue] | None, *, timeout: float) -> str:
        parsed = urlsplit(url)
        base = OllamaSettings.local_endpoint(f"{parsed.scheme}://{parsed.netloc}")
        if parsed.path not in ("/api/chat", "/api/tags") or parsed.query or parsed.fragment:
            raise ValueError("Unsupported local Ollama endpoint")
        url = base + parsed.path
        request = Request(
            url, data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        # Ignore HTTP_PROXY/HTTPS_PROXY; never follow redirects, even locally.
        opener = build_opener(ProxyHandler({}), RejectRedirects())
        with opener.open(request, timeout=timeout) as response:
            return response.read().decode("utf-8")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    role: str
    content: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    model: Text
    message: ChatMessage
    done: bool
    done_reason: str | None = None
    created_at: str | None = None
    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    total_duration: int | None = Field(default=None, ge=0)
    load_duration: int | None = Field(default=None, ge=0)
    prompt_eval_duration: int | None = Field(default=None, ge=0)
    eval_duration: int | None = Field(default=None, ge=0)


class InstalledModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    name: Text
    digest: Text
    remote_host: str | None = None
    remote_model: str | None = None


class ModelList(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    models: list[InstalledModel]


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: OllamaSettings, transport: LocalTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport if transport is not None else LocalHTTPTransport()
        self.model_id = settings.model
        self.retry_count = 0

    @classmethod
    def from_environment(cls) -> "OllamaProvider":
        return cls(OllamaSettings.from_environment())

    def public_configuration(self) -> dict[str, Scalar]:
        return self._settings.public_configuration()

    def _request(self, path: str, payload: dict[str, JsonValue] | None) -> str:
        try:
            return self._transport.request(self._settings.base_url + path, payload,
                                           timeout=self._settings.timeout_seconds)
        except HTTPError:
            raise DecisionExecutionError("provider_api_error") from None
        except TimeoutError:
            raise DecisionExecutionError("provider_timeout") from None
        except URLError as error:
            code: ErrorCode = "provider_timeout" if isinstance(error.reason, TimeoutError) else "provider_connection_error"
            raise DecisionExecutionError(code) from None
        except OSError:
            raise DecisionExecutionError("provider_connection_error") from None
        except UnicodeError:
            raise DecisionExecutionError("provider_malformed_response") from None

    def preflight(self) -> dict[str, Scalar]:
        """Read installed model metadata only; never download or generate."""
        raw = self._request("/api/tags", None)
        try:
            models = ModelList.model_validate_json(raw).models
        except ValidationError:
            raise DecisionExecutionError("provider_malformed_response") from None
        model = next((m for m in models if m.name == self.model_id), None)
        if model is None or model.remote_host or model.remote_model:
            raise ValueError("Configured local model is not installed locally; no download attempted")
        return {"provider": self.name, "model": model.name, "model_digest": model.digest,
                "base_url": self._settings.base_url, "available": True}

    def generate(self, *, instructions: str, problem_json: str,
                 output_model: type[BaseModel]) -> ProviderResponse:
        return self._generate(instructions, problem_json, output_model.model_json_schema())

    def generate_text(self, *, instructions: str, problem_json: str) -> TextProviderResponse:
        response = self._generate(instructions, problem_json, None)
        return TextProviderResponse(
            status=response.status, content_text=response.content_json,
            model_id=response.model_id, usage=response.usage,
            latency_ms=response.latency_ms, response_metadata=response.response_metadata,
        )

    def _generate(self, instructions: str, problem_json: str,
                  schema: dict[str, JsonValue] | None) -> ProviderResponse:
        payload: dict[str, JsonValue] = {
            "model": self.model_id, "stream": False,
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": problem_json}],
            "options": {"temperature": self._settings.temperature,
                        "num_predict": self._settings.max_output_tokens},
        }
        if schema is not None:
            payload["format"] = schema
        raw = self._request("/api/chat", payload)
        try:
            response = ChatResponse.model_validate_json(raw)
            if response.message.role != "assistant":
                raise ValueError("Unexpected Ollama role")
            status: ProviderStatus = "completed"
            if not response.done or response.done_reason == "length":
                status = "incomplete"
            elif response.done_reason not in (None, "stop"):
                status = "failed"
            metadata: dict[str, Scalar] = {
                k: v for k, v in response.model_dump(exclude={"model", "message"}).items()
                if v is not None
            }
            metadata["latency_source"] = "ollama_total_duration"
            metadata["generation_format"] = "json_schema" if schema is not None else "normal_text"
            metadata["local_validation_required"] = True
            return ProviderResponse(
                status=status, content_json=response.message.content, model_id=response.model,
                usage=Usage(input_tokens=response.prompt_eval_count, output_tokens=response.eval_count)
                if response.prompt_eval_count is not None or response.eval_count is not None else None,
                latency_ms=response.total_duration / 1_000_000 if response.total_duration is not None else None,
                response_metadata=metadata,
            )
        except (ValidationError, ValueError, TypeError):
            raise DecisionExecutionError("provider_malformed_response") from None
