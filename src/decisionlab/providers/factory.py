"""Explicit environment selection; one immutable settings snapshot per experiment."""

import os
from typing import Callable, Protocol

from decisionlab.providers.base import StructuredProvider, TextProvider


class GenerationProvider(StructuredProvider, TextProvider, Protocol):
    """Provider supporting the existing text and structured contracts."""


def load_provider_factory(name: str | None = None) -> Callable[[], GenerationProvider]:
    selected = name if name is not None else os.environ.get("DECISIONLAB_PROVIDER", "openai")
    if selected == "ollama":
        from decisionlab.providers.ollama import OllamaProvider
        from decisionlab.settings import OllamaSettings

        settings = OllamaSettings.from_environment()
        return lambda: OllamaProvider(settings)
    if selected == "openai":
        from decisionlab.providers.openai import OpenAIProvider
        from decisionlab.settings import OpenAISettings

        openai_settings = OpenAISettings.from_environment()
        return lambda: OpenAIProvider(openai_settings)
    raise ValueError("DECISIONLAB_PROVIDER must be openai or ollama")


def provider_from_environment(name: str | None = None) -> GenerationProvider:
    """Single-provider path preserves the original OpenAI constructor boundary."""
    selected = name if name is not None else os.environ.get("DECISIONLAB_PROVIDER", "openai")
    if selected == "openai":
        from decisionlab.providers.openai import OpenAIProvider

        return OpenAIProvider.from_environment()
    if selected == "ollama":
        from decisionlab.providers.ollama import OllamaProvider

        return OllamaProvider.from_environment()
    raise ValueError("DECISIONLAB_PROVIDER must be openai or ollama")
