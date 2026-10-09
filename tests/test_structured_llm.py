import json

import pytest
from pydantic import BaseModel

from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import Recommendation, RunMetadata, Scalar, Usage
from decisionlab.providers.base import ProviderResponse
from decisionlab.runner import run_benchmark, run_decision
from decisionlab.strategies import DecisionStrategy, RuleBasedStrategy, StructuredLLMStrategy
from decisionlab.strategies.structured_llm import (
    PROMPT_VERSION, StructuredAssessment, StructuredRecommendation, StructuredUnknown,
)

META = RunMetadata(strategy="structured-llm", strategy_version="1", dataset_version="1.0.0")


class FakeProvider:
    name = "fake-provider"
    model_id = "requested-model"
    retry_count = 0

    def __init__(self, events: list[ProviderResponse | Exception]) -> None:
        self.events = events
        self.requests: list[dict[str, object]] = []

    def public_configuration(self) -> dict[str, Scalar]:
        return {"requested_model": self.model_id, "max_output_tokens": 2000}

    def generate(
        self, *, instructions: str, problem_json: str, output_model: type[BaseModel],
    ) -> ProviderResponse:
        self.requests.append({"instructions": instructions, "problem_json": problem_json, "output_model": output_model})
        result = self.events.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def response(content: str, status: str = "completed") -> ProviderResponse:
    return ProviderResponse.model_validate({
        "status": status, "content_json": content, "model_id": "model-snapshot",
        "response_id": "resp-test", "usage": {"input_tokens": 80, "output_tokens": 40},
    })


def valid_json(benchmark: Benchmark) -> str:
    return RuleBasedStrategy().decide(benchmark.problems[0]).model_dump_json()


def test_strategy_contract_leakage_boundary_and_input_preservation(benchmark: Benchmark) -> None:
    provider = FakeProvider([response(valid_json(benchmark))])
    strategy: DecisionStrategy = StructuredLLMStrategy(provider)
    problem = benchmark.problems[0]
    before = problem.model_dump_json()
    result = strategy.decide(problem)
    assert isinstance(result, Recommendation)
    problem.validate_recommendation(result)
    assert before == problem.model_dump_json()
    request = provider.requests[0]
    assert request["problem_json"] == before
    assert request["output_model"] is StructuredRecommendation
    assert "route-001" not in request["instructions"]
    assert "acceptable_option_ids" not in request["problem_json"]
    assert "evidence_rubric" not in request["problem_json"]


def test_wire_schema_tracks_domain_fields_and_requires_every_field() -> None:
    assert set(StructuredRecommendation.model_fields) == set(Recommendation.model_fields)
    from decisionlab.models import OptionAssessment, Unknown
    assert set(StructuredAssessment.model_fields) == set(OptionAssessment.model_fields)
    assert set(StructuredUnknown.model_fields) == set(Unknown.model_fields)
    for model in (StructuredRecommendation, StructuredAssessment, StructuredUnknown):
        assert all(field.is_required() for field in model.model_fields.values())


def test_runner_records_model_prompt_configuration_and_usage(benchmark: Benchmark) -> None:
    provider = FakeProvider([response(valid_json(benchmark))])
    strategy = StructuredLLMStrategy(provider)
    observed = run_decision(benchmark.problems[0], strategy, META, "r")
    assert observed.run.status == "success"
    metadata = observed.run.metadata
    assert metadata.provider == "fake-provider"
    assert metadata.model_id == "model-snapshot"
    assert metadata.provider_response_id == "resp-test"
    assert metadata.prompt_version == PROMPT_VERSION
    assert metadata.dataset_version == "1.0.0"
    assert metadata.generation_settings["requested_model"] == "requested-model"
    assert len(metadata.generation_settings["prompt_sha256"]) == 64
    assert observed.run.usage == Usage(input_tokens=80, output_tokens=40)


@pytest.mark.parametrize("changes", [
    {"selected_option_id": None}, {"confidence": 1.5}, {"explanation": "   "},
    {"requested_review": "false"}, {"unexpected": "field"},
])
def test_invalid_outputs_fail_and_keep_usage(benchmark: Benchmark, changes: dict[str, object]) -> None:
    raw = json.loads(valid_json(benchmark))
    raw.update(changes)
    strategy = StructuredLLMStrategy(FakeProvider([response(json.dumps(raw))]))
    observed = run_decision(benchmark.problems[0], strategy, META, "r")
    assert observed.run.status == "failure"
    assert observed.run.error.code == "invalid_schema"
    assert observed.schema_valid is False
    assert observed.run.recommendation is None
    assert observed.run.usage.output_tokens == 40


@pytest.mark.parametrize("content", ["not JSON secret", "{}", "", "null"])
def test_malformed_text_never_becomes_abstention(benchmark: Benchmark, content: str) -> None:
    strategy = StructuredLLMStrategy(FakeProvider([response(content)]))
    with pytest.raises(DecisionExecutionError, match="invalid_schema"):
        strategy.decide(benchmark.problems[0])


def test_invalid_evidence_preserved_for_independent_metrics(benchmark: Benchmark) -> None:
    raw = json.loads(valid_json(benchmark))
    raw["assessments"][0]["evidence_ids"] = ["invented"]
    strategy = StructuredLLMStrategy(FakeProvider([response(json.dumps(raw))]))
    observed = run_decision(benchmark.problems[0], strategy, META, "r")
    assert observed.run.error.code == "invalid_references"
    assert observed.run.recommendation is None
    assert observed.candidate is not None
    assert observed.schema_valid and not observed.references_valid
    assert observed.evidence_references_valid is False
    assert observed.run.usage.input_tokens == 80


def test_valid_abstention_is_distinct_from_failed_generation(benchmark: Benchmark) -> None:
    raw = RuleBasedStrategy().decide(benchmark.problems[10]).model_dump_json()
    strategy = StructuredLLMStrategy(FakeProvider([response(raw)]))
    observed = run_decision(benchmark.problems[10], strategy, META, "r")
    assert observed.run.status == "success"
    assert observed.run.recommendation.status == "abstained"
    assert observed.run.policy_result.requires_human_review


@pytest.mark.parametrize("status,code", [
    ("refused", "provider_refusal"), ("incomplete", "provider_incomplete"),
    ("malformed", "provider_malformed_response"), ("failed", "provider_api_error"),
])
def test_noncompleted_provider_response_is_a_failure(benchmark: Benchmark, status: str, code: str) -> None:
    strategy = StructuredLLMStrategy(FakeProvider([response("", status)]))
    observed = run_decision(benchmark.problems[0], strategy, META, "r")
    assert observed.run.error.code == code
    assert observed.run.recommendation is None
    assert observed.schema_valid is None
    assert observed.run.usage.output_tokens == 40


def test_timeout_resets_previous_usage_and_retains_configuration(benchmark: Benchmark) -> None:
    provider = FakeProvider([response(valid_json(benchmark)), DecisionExecutionError("provider_timeout")])
    strategy = StructuredLLMStrategy(provider)
    first = run_decision(benchmark.problems[0], strategy, META, "r1")
    second = run_decision(benchmark.problems[0], strategy, META, "r2")
    assert first.run.usage.output_tokens == 40
    assert second.run.error.code == "provider_timeout"
    assert second.run.usage is None
    assert second.run.metadata.model_id == "requested-model"
    assert second.run.metadata.prompt_version == PROMPT_VERSION
    assert second.run.metadata.provider_response_id is None


def test_benchmark_execution_uses_same_evaluator_without_labels_in_requests(benchmark: Benchmark) -> None:
    provider = FakeProvider([
        response(RuleBasedStrategy().decide(problem).model_dump_json())
        for problem in benchmark.problems
    ])
    report = run_benchmark(benchmark, StructuredLLMStrategy(provider), strategy_name="structured-llm", strategy_version="1")
    assert report.summary.rates["outcome_correctness"].value == 0.9
    assert len(provider.requests) == 30
    for request, problem in zip(provider.requests, benchmark.problems, strict=True):
        assert json.loads(request["problem_json"]) == problem.model_dump(mode="json")
    assert all(case.observation.run.metadata.prompt_version == PROMPT_VERSION for case in report.cases)
    assert report.summary.input_token_sample_count == 30
    assert report.summary.total_input_tokens == 2400
    assert report.summary.total_output_tokens == 1200


def test_semantic_wrong_choice_reaches_evaluator_and_policy_without_repair(benchmark: Benchmark) -> None:
    raw = json.loads(valid_json(benchmark))
    raw["selected_option_id"] = "q-e"
    observed = run_decision(benchmark.problems[0], StructuredLLMStrategy(FakeProvider([response(json.dumps(raw))])), META, "r")
    assert observed.run.status == "success"
    assert observed.run.recommendation.selected_option_id == "q-e"
    assert "wrong_route" in observed.run.policy_result.detected_violations
