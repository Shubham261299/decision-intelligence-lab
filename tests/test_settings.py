import pytest

from decisionlab.settings import OpenAISettings


@pytest.fixture
def configured(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_TIMEOUT_SECONDS", "OPENAI_MAX_OUTPUT_TOKENS", "OPENAI_MAX_RETRIES", "OPENAI_TEMPERATURE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    return monkeypatch


def test_environment_loading_and_secret_exclusion(configured) -> None:
    settings = OpenAISettings.from_environment()
    assert settings.model == "test-model"
    assert "test-secret-value" not in repr(settings)
    assert "test-secret-value" not in settings.model_dump_json()
    assert "api_key" not in settings.public_configuration()
    assert settings.max_retries == 0
    assert "temperature" not in settings.public_configuration()


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "OPENAI_MODEL"])
def test_missing_required_configuration_is_explicit(configured, key: str) -> None:
    configured.delenv(key)
    with pytest.raises(ValueError, match="must be configured"):
        OpenAISettings.from_environment()


@pytest.mark.parametrize("key,value", [
    ("OPENAI_TIMEOUT_SECONDS", "0"), ("OPENAI_TIMEOUT_SECONDS", "nan"),
    ("OPENAI_MAX_RETRIES", "3"), ("OPENAI_MAX_RETRIES", "-1"),
    ("OPENAI_MAX_OUTPUT_TOKENS", "abc"), ("OPENAI_MAX_OUTPUT_TOKENS", "0"),
    ("OPENAI_TEMPERATURE", "3"), ("OPENAI_TEMPERATURE", "inf"),
])
def test_invalid_settings_errors_do_not_echo_values(configured, key: str, value: str) -> None:
    configured.setenv(key, value)
    with pytest.raises(ValueError) as captured:
        OpenAISettings.from_environment()
    assert "test-secret-value" not in str(captured.value)
    assert "environment configuration" in str(captured.value)


def test_explicit_environment_configuration(configured) -> None:
    configured.setenv("OPENAI_TIMEOUT_SECONDS", "10")
    configured.setenv("OPENAI_MAX_RETRIES", "1")
    configured.setenv("OPENAI_TEMPERATURE", "0.2")
    settings = OpenAISettings.from_environment()
    assert settings.public_configuration()["temperature"] == 0.2
    assert settings.timeout_seconds == 10
