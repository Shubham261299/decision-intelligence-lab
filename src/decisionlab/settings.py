"""Explicit environment loading; no configuration work happens at import time."""

import os
import ipaddress
from urllib.parse import urlsplit
from typing import Annotated

from pydantic import ConfigDict, Field, SecretStr, ValidationError, field_validator

from decisionlab.models import DomainModel, Scalar, Text


class OpenAISettings(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    api_key: SecretStr = Field(exclude=True, repr=False)
    model: Text
    timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 30
    max_output_tokens: Annotated[int, Field(ge=256, le=16000)] = 2000
    max_retries: Annotated[int, Field(ge=0, le=2)] = 0
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None

    @classmethod
    def from_environment(cls) -> "OpenAISettings":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "").strip()
        if not key or not model:
            raise ValueError("OPENAI_API_KEY and OPENAI_MODEL must be configured")
        data = {
            "api_key": key, "model": model,
            "timeout_seconds": os.environ.get("OPENAI_TIMEOUT_SECONDS", "30"),
            "max_output_tokens": os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "2000"),
            "max_retries": os.environ.get("OPENAI_MAX_RETRIES", "0"),
            "temperature": os.environ.get("OPENAI_TEMPERATURE") or None,
        }
        try:
            return cls.model_validate(data)
        except ValidationError:
            raise ValueError("Invalid OpenAI environment configuration types or bounds") from None

    def public_configuration(self) -> dict[str, Scalar]:
        config: dict[str, Scalar] = {
            "requested_model": self.model, "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens, "max_retries": self.max_retries,
            "store": False,
        }
        if self.temperature is not None:
            config["temperature"] = self.temperature
        return config


class OllamaSettings(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    base_url: Text = "http://127.0.0.1:11434"
    model: Text
    timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 180
    max_output_tokens: Annotated[int, Field(ge=16, le=16000)] = 2000
    temperature: Annotated[float, Field(ge=0, le=2)] = 0

    @field_validator("base_url")
    @classmethod
    def local_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (parsed.scheme != "http" or not host or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
            raise ValueError("Ollama requires a plain local HTTP base URL")
        if host == "localhost":
            host = "127.0.0.1"  # Avoid DNS configuration directing localhost elsewhere.
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Ollama endpoint must be loopback")
        port = parsed.port if parsed.port is not None else 11434
        if not 1 <= port <= 65535:
            raise ValueError("Invalid Ollama port")
        host = f"[{host}]" if ":" in host else host
        return f"http://{host}:{port}"

    @field_validator("model")
    @classmethod
    def local_model(cls, value: str) -> str:
        if "cloud" in value.lower() or "/" in value or value != value.strip():
            raise ValueError("Use an installed local Ollama model tag")
        return value

    @classmethod
    def from_environment(cls) -> "OllamaSettings":
        model = os.environ.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise ValueError("OLLAMA_MODEL must be configured")
        try:
            return cls.model_validate({
                "model": model,
                "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                "timeout_seconds": os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"),
                "max_output_tokens": os.environ.get("OLLAMA_MAX_OUTPUT_TOKENS", "2000"),
                "temperature": os.environ.get("OLLAMA_TEMPERATURE", "0"),
            })
        except ValidationError:
            raise ValueError("Invalid Ollama local endpoint/model/configuration") from None

    def public_configuration(self) -> dict[str, Scalar]:
        return {
            "base_url": self.base_url, "requested_model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens, "temperature": self.temperature,
            "max_retries": 0, "stream": False,
            "structured_output_capability": "ollama_json_schema_local_validation",
        }
