"""Paid API smoke tests; never part of ordinary offline test execution."""

import os

import pytest

from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import RunMetadata
from decisionlab.providers.openai import OpenAIProvider
from decisionlab.runner import run_decision
from decisionlab.strategies.structured_llm import PROMPT_VERSION, StructuredLLMStrategy

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"),
        reason="OPENAI_API_KEY and OPENAI_MODEL are required for live tests",
    ),
]


@pytest.mark.parametrize("index", [0, 10], ids=["complete-input", "missing-information"])
def test_live_structured_response_and_runner(benchmark: Benchmark, index: int) -> None:
    strategy = StructuredLLMStrategy(OpenAIProvider.from_environment())
    problem = benchmark.problems[index]
    observation = run_decision(
        problem, strategy,
        RunMetadata(strategy=strategy.name, strategy_version=strategy.version),
        f"live:{problem.id}",
    )
    # API/validation errors fail the smoke test, not silently skip or abstain.
    assert observation.run.status == "success", observation.run.error
    assert observation.run.recommendation is not None
    problem.validate_recommendation(observation.run.recommendation)
    assert observation.run.metadata.prompt_version == PROMPT_VERSION
    assert observation.run.metadata.provider_response_id is not None
    assert observation.run.usage is not None
    assert observation.run.usage.input_tokens > 0
