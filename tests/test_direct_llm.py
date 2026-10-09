import json
from importlib.resources import files

import pytest

from decisionlab.errors import DecisionExecutionError
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.models import RunMetadata, Scalar, Usage
from decisionlab.providers.base import TextProviderResponse
from decisionlab.runner import run_benchmark, run_decision
from decisionlab.strategies import DecisionStrategy, DirectLLMStrategy, RuleBasedStrategy
from decisionlab.strategies.direct_llm import PROMPT_VERSION
from test_direct_parser import TEXT, format_recommendation

META = RunMetadata(strategy="direct-llm", strategy_version="1", dataset_version="1.0.0")


class FakeTextProvider:
    name = "fake-provider"
    model_id = "requested-model"
    retry_count = 0

    def __init__(self, events: list[TextProviderResponse | Exception]) -> None:
        self.events = events
        self.requests: list[dict[str, str]] = []

    def public_configuration(self) -> dict[str, Scalar]:
        return {"requested_model": self.model_id, "max_output_tokens": 2000}

    def generate_text(self, *, instructions: str, problem_json: str) -> TextProviderResponse:
        self.requests.append({"instructions": instructions, "problem_json": problem_json})
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def response(text: str = TEXT, status: str = "completed") -> TextProviderResponse:
    return TextProviderResponse.model_validate({
        "status": status, "content_text": text, "model_id": "model-snapshot",
        "response_id": "resp-test", "usage": {"input_tokens": 80, "output_tokens": 40},
    })


def test_identical_routing_policy_instructions() -> None:
    resources = files("decisionlab.prompts")
    structured = resources.joinpath("routing_v1.txt").read_text(encoding="utf-8")
    direct = resources.joinpath("direct_routing_v1.txt").read_text(encoding="utf-8")
    assert structured.split("Return the requested")[0] == direct.split("Return normal")[0]


def test_contract_metadata_and_no_reference_leakage(benchmark: Benchmark) -> None:
    provider = FakeTextProvider([response()])
    strategy: DecisionStrategy = DirectLLMStrategy(provider)
    problem = benchmark.problems[0]
    before = problem.model_dump_json()
    observed = run_decision(problem, strategy, META, "r")
    assert observed.run.status == "success"
    assert problem.model_dump_json() == before
    assert provider.requests[0]["problem_json"] == before
    assert set(provider.requests[0]) == {"instructions", "problem_json"}
    assert "acceptable_option_ids" not in json.dumps(provider.requests)
    assert "evidence_rubric" not in json.dumps(provider.requests)
    assert "route-001" not in provider.requests[0]["instructions"]
    metadata = observed.run.metadata
    assert metadata.provider == "fake-provider" and metadata.model_id == "model-snapshot"
    assert metadata.provider_response_id == "resp-test"
    assert metadata.prompt_version == PROMPT_VERSION
    assert metadata.dataset_version == "1.0.0"
    assert metadata.generation_settings["requested_model"] == "requested-model"
    assert metadata.generation_settings["parser_version"] == "direct-text-v1"
    assert metadata.generation_settings["output_format"] == "text"
    assert len(metadata.generation_settings["prompt_sha256"]) == 64
    assert observed.run.usage == Usage(input_tokens=80, output_tokens=40)


@pytest.mark.parametrize("text", ["", "I choose routine.", "{}", TEXT + "\nAlternatively q-e."])
def test_bad_text_remains_failed_not_abstained(benchmark: Benchmark, text: str) -> None:
    observed = run_decision(
        benchmark.problems[0], DirectLLMStrategy(FakeTextProvider([response(text)])), META, "r",
    )
    assert observed.run.status == "failure"
    assert observed.run.error.code == "invalid_schema"
    assert observed.schema_valid is False
    assert observed.run.recommendation is None
    assert observed.run.usage.output_tokens == 40


@pytest.mark.parametrize("old,new,evidence_valid", [
    ("Option: q-r", "Option: invented", True),
    ("Evidence: urgent, specialist_required", "Evidence: invented", False),
    ("Constraints: capacity", "Constraints: invented", True),
    ("Assessment: q-r", "Assessment: invented", True),
    ("Unknowns: none", "Unknowns: present\nUnknown: missing\nAffected options: invented\nMaterial: yes", True),
])
def test_reference_failures_preserve_candidate(
    benchmark: Benchmark, old: str, new: str, evidence_valid: bool,
) -> None:
    observed = run_decision(benchmark.problems[0], DirectLLMStrategy(
        FakeTextProvider([response(TEXT.replace(old, new))]),
    ), META, "r")
    assert observed.run.error.code == "invalid_references"
    assert observed.run.recommendation is None
    assert observed.candidate is not None
    assert observed.schema_valid and observed.references_valid is False
    assert observed.evidence_references_valid is evidence_valid


@pytest.mark.parametrize("status,code", [
    ("refused", "provider_refusal"), ("incomplete", "provider_incomplete"),
    ("malformed", "provider_malformed_response"), ("failed", "provider_api_error"),
])
def test_provider_statuses_are_failures(benchmark: Benchmark, status: str, code: str) -> None:
    observed = run_decision(benchmark.problems[0], DirectLLMStrategy(
        FakeTextProvider([response(TEXT, status)]),
    ), META, "r")
    assert observed.run.error.code == code
    assert observed.run.recommendation is None and observed.schema_valid is None
    assert observed.run.usage.output_tokens == 40


def test_timeout_clears_previous_usage_and_records_retry(benchmark: Benchmark) -> None:
    provider = FakeTextProvider([response(), DecisionExecutionError("provider_timeout")])
    strategy = DirectLLMStrategy(provider)
    first = run_decision(benchmark.problems[0], strategy, META, "r1")
    provider.retry_count = 1
    second = run_decision(benchmark.problems[0], strategy, META, "r2")
    assert first.run.usage.output_tokens == 40
    assert second.run.error.code == "provider_timeout"
    assert second.run.usage is None
    assert second.run.metadata.model_id == "requested-model"
    assert second.run.metadata.provider_response_id is None
    assert second.run.metadata.retry_count == 1


def test_semantic_error_is_preserved_for_policy(benchmark: Benchmark) -> None:
    observed = run_decision(benchmark.problems[0], DirectLLMStrategy(
        FakeTextProvider([response(TEXT.replace("Option: q-r", "Option: q-e"))]),
    ), META, "r")
    assert observed.run.status == "success"
    assert observed.run.recommendation.selected_option_id == "q-e"
    assert "wrong_route" in observed.run.policy_result.detected_violations


def test_same_benchmark_and_independent_evaluator(benchmark: Benchmark) -> None:
    provider = FakeTextProvider([
        response(format_recommendation(RuleBasedStrategy().decide(problem)))
        for problem in benchmark.problems
    ])
    report = run_benchmark(
        benchmark, DirectLLMStrategy(provider), strategy_name="direct-llm", strategy_version="1",
    )
    assert report.summary.rates["outcome_correctness"].value == 0.9
    assert report.summary.counts["failed_runs"] == 0
    assert report.summary.total_input_tokens == 2400
    assert report.summary.total_output_tokens == 1200
    for request, problem in zip(provider.requests, benchmark.problems, strict=True):
        assert json.loads(request["problem_json"]) == problem.model_dump(mode="json")
    abstentions = [case.observation.run for case in report.cases
                   if case.observation.run.recommendation.status == "abstained"]
    assert abstentions and all(run.policy_result.requires_human_review for run in abstentions)
