"""OpenAI Responses API HTTP adapter, isolated from strategies and evaluation."""

import json
import socket
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from decisionlab.errors import DecisionExecutionError, ErrorCode
from decisionlab.models import Scalar, Usage
from decisionlab.providers.base import (
    ProviderResponse,
    ProviderStatus,
    TextProviderResponse,
)
from decisionlab.settings import OpenAISettings

ENDPOINT = "https://api.openai.com/v1/responses"


class HTTPTransport(Protocol):
    def send(
        self, payload: dict[str, JsonValue], *, api_key: str, timeout: float,
    ) -> str: ...


class UrllibTransport:
    def send(
        self, payload: dict[str, JsonValue], *, api_key: str, timeout: float,
    ) -> str:
        request = Request(
            ENDPOINT, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")


class APIContent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    text: str | None = None


class APIOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    content: list[APIContent] = Field(default_factory=list)


class APIUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input_tokens: int = Field(strict=True, ge=0)
    output_tokens: int = Field(strict=True, ge=0)


class APIResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    model: str
    status: str
    output: list[APIOutput]
    usage: APIUsage | None = None


class OpenAIProvider:
    name = "openai"

    def __init__(
        self, settings: OpenAISettings, transport: HTTPTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport if transport is not None else UrllibTransport()
        self.model_id = settings.model
        self.retry_count = 0

    @classmethod
    def from_environment(cls) -> "OpenAIProvider":
        return cls(OpenAISettings.from_environment())

    def public_configuration(self) -> dict[str, Scalar]:
        return self._settings.public_configuration()

    def generate(
        self, *, instructions: str, problem_json: str, output_model: type[BaseModel],
    ) -> ProviderResponse:
        payload = self._base_payload(instructions, problem_json)
        payload["text"] = {"format": {
            "type": "json_schema", "name": "decision_recommendation",
            "strict": True, "schema": output_model.model_json_schema(),
        }}
        return self._request(payload)

    def generate_text(
        self, *, instructions: str, problem_json: str,
    ) -> TextProviderResponse:
        payload = self._base_payload(instructions, problem_json)
        payload["text"] = {"format": {"type": "text"}}
        response = self._request(payload)
        return TextProviderResponse(
            status=response.status, content_text=response.content_json,
            model_id=response.model_id, response_id=response.response_id,
            usage=response.usage,
        )

    def _base_payload(
        self, instructions: str, problem_json: str,
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "model": self.model_id, "instructions": instructions,
            "input": [{"role": "user", "content": problem_json}],
            "max_output_tokens": self._settings.max_output_tokens,
            "store": False,
        }
        if self._settings.temperature is not None:
            payload["temperature"] = self._settings.temperature
        return payload

    def _request(self, payload: dict[str, JsonValue]) -> ProviderResponse:
        self.retry_count = 0
        raw = ""
        for attempt in range(self._settings.max_retries + 1):
            self.retry_count = attempt
            try:
                raw = self._transport.send(
                    payload, api_key=self._settings.api_key.get_secret_value(),
                    timeout=self._settings.timeout_seconds,
                )
                break
            except HTTPError as error:
                transient = error.code == 429 or error.code >= 500
                if not transient or attempt == self._settings.max_retries:
                    raise DecisionExecutionError("provider_api_error") from None
            except (TimeoutError, socket.timeout):
                if attempt == self._settings.max_retries:
                    raise DecisionExecutionError("provider_timeout") from None
            except URLError as error:
                if attempt == self._settings.max_retries:
                    code: ErrorCode = (
                        "provider_timeout" if isinstance(error.reason, TimeoutError)
                        else "provider_connection_error"
                    )
                    raise DecisionExecutionError(code) from None
            except OSError:
                if attempt == self._settings.max_retries:
                    raise DecisionExecutionError("provider_connection_error") from None
            except UnicodeError:
                raise DecisionExecutionError("provider_malformed_response") from None
            # Small bounded backoff, no retries for malformed/model-invalid outputs.
            time.sleep(0.25 * (2 ** attempt))
        try:
            response = APIResponse.model_validate_json(raw)
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ) if response.usage is not None else None
            contents = [part for item in response.output if item.type == "message" for part in item.content]
            status: ProviderStatus = "completed"
            if any(part.type == "refusal" for part in contents):
                status = "refused"
            elif response.status == "incomplete":
                status = "incomplete"
            elif response.status != "completed":
                status = "failed"
            texts = [part.text for part in contents if part.type == "output_text" and part.text is not None]
            if status == "completed" and len(texts) != 1:
                status = "malformed"
            return ProviderResponse(
                status=status, content_json=texts[0] if len(texts) == 1 else None,
                model_id=response.model, response_id=response.id, usage=usage,
            )
        except (ValidationError, TypeError, ValueError):
            raise DecisionExecutionError("provider_malformed_response") from None
