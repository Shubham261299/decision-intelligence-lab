import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from decisionlab import cli
from decisionlab.evaluation.benchmark import Benchmark, load_benchmark
from decisionlab.evaluation.evaluator import evaluate_observation
from decisionlab.models import DecisionProblem, RunMetadata
from decisionlab.providers.base import TextProviderResponse
from decisionlab.providers.ollama import OllamaProvider
from decisionlab.representation_diagnosis import diagnose, explicit_target
from decisionlab.representation_experiment import execution_schedule, run_representation_experiment
from decisionlab.runner import run_decision
from decisionlab.settings import OllamaSettings
from decisionlab.strategies.representation import (
    REPRESENTATIONS, CaptureLocalTransport, RepresentationStrategy, digest,
)
from decisionlab.strategies.rules import RuleBasedStrategy
from test_direct_parser import format_recommendation
from test_ollama_provider import FakeTransport, envelope


def dev_benchmark() -> Benchmark:
    return load_benchmark(Path("evals"), "dev")


def single_result(condition: str, raw: str, event: object | None = None):
    benchmark = dev_benchmark()
    problem = benchmark.problems[0]
    transport = CaptureLocalTransport(FakeTransport([event if event is not None else envelope(raw)]))
    strategy = RepresentationStrategy(
        OllamaProvider(OllamaSettings(model="qwen2.5:14b"), transport), condition, transport,
    )
    observation = run_decision(problem, strategy, RunMetadata(strategy=condition, strategy_version="1"), "test")
    result = evaluate_observation(problem, benchmark.expectations[problem.id], observation)
    before = result.model_dump_json()
    diagnosis = diagnose(problem, result, strategy.last_raw, condition)
    assert result.model_dump_json() == before
    return result, strategy, diagnosis


def valid_raw(condition: str) -> str:
    rec = RuleBasedStrategy().decide(dev_benchmark().problems[0])
    return format_recommendation(rec) if condition == "labelled-text" else rec.model_dump_json()


@pytest.mark.parametrize("condition", REPRESENTATIONS)
def test_text_only_generation_raw_capture_and_provenance(condition: str) -> None:
    raw = valid_raw(condition)
    result, strategy, diagnosis = single_result(condition, raw)
    assert result.outcome == "selected" and result.outcome_correct
    assert strategy.last_raw.text == raw
    assert strategy.last_raw.response_sha256 == digest(raw)
    assert strategy.last_raw.envelope == envelope(raw)
    assert strategy.last_raw.envelope_sha256 == digest(envelope(raw))
    assert diagnosis.failure_stage == "none" and diagnosis.decision_diagnosis == "not_failed"
    run = result.observation.run
    assert run.usage.input_tokens == 10 and run.latency_ms == 250
    assert run.metadata.generation_settings["condition"] == condition
    assert run.metadata.generation_settings["output_enforcement"] == "none"
    assert run.metadata.prompt_version == "representation-v1"
    payload = strategy._capture._transport.requests[0][1]
    assert "format" not in payload
    assert payload["messages"][1]["content"] == dev_benchmark().problems[0].model_dump_json()
    assert "acceptable_option_ids" not in json.dumps(payload)


@pytest.mark.parametrize("condition", REPRESENTATIONS)
@pytest.mark.parametrize("kind", ["empty", "format", "domain", "reference", "unknown_option"])
def test_failure_stages_raw_and_no_rescue(condition: str, kind: str) -> None:
    raw = valid_raw(condition)
    expected_stage = {
        "empty": "empty_incomplete_refusal", "format": "serialization_format_failure",
        "domain": "schema_domain_validation", "reference": "invalid_option_reference",
        "unknown_option": "invalid_option_reference",
    }[kind]
    if kind == "empty":
        raw = ""
    elif kind == "format":
        raw = "```" + ("text" if condition == "labelled-text" else "json") + "\n" + raw + "\n```"
    elif condition == "labelled-text":
        if kind == "domain":
            raw = raw.replace("selected", "bogus", 1)
        elif kind == "reference":
            raw = raw.replace("Evidence: urgent", "Evidence: unknown_fact")
        else:
            raw = raw.replace("Option: q-r", "Option: unknown_queue")
    else:
        data = json.loads(raw)
        if kind == "domain":
            data["confidence"] = 1.1
        elif kind == "reference":
            data["assessments"][0]["evidence_ids"] = ["unknown_fact"]
        else:
            data["selected_option_id"] = "unknown_queue"
        raw = json.dumps(data)
    result, strategy, diagnosis = single_result(condition, raw)
    assert result.outcome == "failure" and not result.outcome_correct
    assert result.observation.run.recommendation is None
    assert result.observation.run.error.code == (
        "invalid_references" if kind in ("reference", "unknown_option") else "invalid_schema"
    )
    assert diagnosis.failure_stage == expected_stage
    assert diagnosis.primary_score_changed is False
    assert strategy.last_raw.text == raw
    assert diagnosis.requires_human_judgment
    if kind == "domain":
        assert strategy.last_raw.validation_issues
        assert all("input" not in issue for issue in strategy.last_raw.validation_issues)


@pytest.mark.parametrize("condition", REPRESENTATIONS)
@pytest.mark.parametrize("status", ["refused", "incomplete", "malformed", "failed"])
def test_provider_status_preserves_partial_text(condition: str, status: str) -> None:
    class Provider:
        name = "fake"
        model_id = "qwen2.5:14b"
        retry_count = 0
        def public_configuration(self):
            return {}
        def generate_text(self, **kwargs):
            return TextProviderResponse(status=status, content_text="partial raw text", model_id=self.model_id)
    strategy = RepresentationStrategy(Provider(), condition)
    problem = dev_benchmark().problems[0]
    observed = run_decision(problem, strategy, RunMetadata(strategy=condition, strategy_version="1"), "test")
    assert observed.run.status == "failure"
    assert strategy.last_raw.text == "partial raw text"
    assert strategy.last_raw.provider_status == status
    assert strategy.last_raw.response_sha256 == digest("partial raw text")


@pytest.mark.parametrize("condition", REPRESENTATIONS)
@pytest.mark.parametrize("event,code,body", [
    (TimeoutError(), "provider_timeout", None),
    ("malformed envelope", "provider_malformed_response", "malformed envelope"),
])
def test_provider_failure_availability(condition: str, event: object, code: str, body: str | None) -> None:
    result, strategy, diagnosis = single_result(condition, "unused", event)
    assert result.observation.run.error.code == code
    assert strategy.last_raw.text is None and strategy.last_raw.response_sha256 is None
    assert strategy.last_raw.envelope == body
    assert diagnosis.failure_stage == "provider_api_failure"
    assert diagnosis.decision_diagnosis == "ambiguous"


def test_http_error_body_is_preserved_without_headers() -> None:
    event = HTTPError("http://127.0.0.1:11434/api/chat", 500, "error", None, io.BytesIO(b'{"error":"local failure"}'))
    result, strategy, _ = single_result("labelled-text", "unused", event)
    assert result.observation.run.error.code == "provider_api_error"
    assert strategy.last_raw.envelope == '{"error":"local failure"}'
    assert strategy.last_raw.envelope_sha256 == digest(strategy.last_raw.envelope)


@pytest.mark.parametrize("condition", REPRESENTATIONS)
@pytest.mark.parametrize("correct", [True, False])
def test_explicit_target_diagnosis_is_not_full_semantic_credit(condition: str, correct: bool) -> None:
    raw = valid_raw(condition)
    if not correct:
        if condition == "labelled-text":
            raw = raw.replace("Option: q-r", "Option: q-e")
        else:
            data = json.loads(raw)
            data["selected_option_id"] = "q-e"
            raw = json.dumps(data)
    raw = raw + "\nUnexpected trailing text" if condition == "labelled-text" else "```json\n" + raw + "\n```"
    result, _, diagnosis = single_result(condition, raw)
    assert result.outcome == "failure"
    assert diagnosis.benchmark_target_matches is correct
    assert diagnosis.decision_diagnosis == (
        "explicit_apparently_correct_rejected" if correct else "explicit_incorrect_decision"
    )
    assert diagnosis.semantic_support == "not_assessed" and diagnosis.requires_human_judgment


@pytest.mark.parametrize("raw,condition", [
    ("I recommend q-r.", "labelled-text"),
    ("Decision: selected\nOption: q-r\nOption: q-e", "labelled-text"),
    ('{"status":"selected","selected_option_id":"q-r","selected_option_id":"q-e"}', "unconstrained-json"),
    ('{"status":"selected","selected_option_id":', "unconstrained-json"),
    ('{"status":"abstained","selected_option_id":"q-r"}', "unconstrained-json"),
])
def test_ambiguous_targets_not_inferred(raw: str, condition: str) -> None:
    assert explicit_target(raw, condition) is None


class EchoTransport:
    def __init__(self, requests: list[dict], fail_text: bool = False, output: Path | None = None) -> None:
        self.requests = requests
        self.fail_text = fail_text
        self.output = output

    def request(self, url: str, payload: dict, *, timeout: float) -> str:
        if self.output is not None:
            assert (self.output / "protocol.json").exists()
        self.requests.append(payload)
        assert "format" not in payload
        problem = DecisionProblem.model_validate_json(payload["messages"][1]["content"])
        rec = RuleBasedStrategy().decide(problem)
        text = "Return normal labelled text" in payload["messages"][0]["content"]
        raw = format_recommendation(rec) if text else rec.model_dump_json()
        if text and self.fail_text:
            raw += "\nUnexpected trailing content"
        return envelope(raw)


def make_strategies(requests: list[dict], fail_text: bool = False, output: Path | None = None):
    strategies = {}
    for condition in REPRESENTATIONS:
        capture = CaptureLocalTransport(EchoTransport(requests, fail_text, output))
        strategies[condition] = RepresentationStrategy(
            OllamaProvider(OllamaSettings(model="qwen2.5:14b"), capture), condition, capture,
        )
    return strategies


def test_counterbalanced_schedule_and_matched_pipeline(tmp_path: Path) -> None:
    benchmark = dev_benchmark()
    before = benchmark.model_dump_json()
    requests = []
    report = run_representation_experiment(
        benchmark, make_strategies(requests, output=tmp_path), mode="mocked", output=tmp_path,
    )
    assert benchmark.model_dump_json() == before
    assert len(requests) == len(report.observations) == 120
    assert len(report.protocol.schedule) == 120 and report.protocol.repeats == 3
    first = report.protocol.schedule[::2]
    assert sum(x.condition == "labelled-text" for x in first) == 30
    assert sum(x.condition == "unconstrained-json" for x in first) == 30
    for left, right in zip(requests[::2], requests[1::2]):
        assert left["messages"][1] == right["messages"][1]
        assert left["options"] == right["options"] == {"temperature": 0, "num_predict": 2000}
        assert left["model"] == right["model"] == "qwen2.5:14b"
        assert left["messages"][0]["content"].split("\n\nReturn")[0] == right["messages"][0]["content"].split("\n\nReturn")[0]
    assert report.paired.decision_agreement.numerator == report.paired.decision_agreement.denominator == 60
    assert report.paired.both_correct == 54
    assert report.paired.labelled_only_correct == report.paired.json_only_correct == 0
    for condition in REPRESENTATIONS:
        primary = report.comparison.columns[condition].report
        assert primary.summary.rates["outcome_correctness"].value == 0.9
        assert primary.summary.rates["repeated_decision_agreement"].value == 1
        assert all(c.observation.verification is not None for c in primary.cases)
        assert report.diagnostics[condition].confirmed_substantively_correct is None
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 120
    assert json.loads((tmp_path / "experiment.json").read_text())["protocol"]["mode"] == "mocked"
    assert json.loads((tmp_path / "paired-analysis.json").read_text())["both_correct"] == 54


def test_failed_diagnostics_never_modify_scores(tmp_path: Path) -> None:
    report = run_representation_experiment(dev_benchmark(), make_strategies([], True), repeats=1,
                                           mode="mocked", output=tmp_path)
    column = report.comparison.columns["labelled-text"]
    assert column.report.summary.counts["failed_runs"] == 20
    assert column.report.summary.rates["outcome_correctness"].value == 0
    assert column.failure_codes == {"invalid_schema": 20}
    diagnosis = report.diagnostics["labelled-text"]
    assert diagnosis.apparent_correct_rejected_targets == 18
    assert diagnosis.explicit_incorrect_targets == 2
    assert diagnosis.ambiguous_targets == 0
    assert diagnosis.confirmed_substantively_correct is None
    assert report.comparison.columns["unconstrained-json"].report.summary.rates["outcome_correctness"].value == 0.9
    assert report.paired.decision_agreement.value is None


@pytest.mark.parametrize("kind", ["permission", "split", "repeats", "config", "occupied"])
def test_configuration_guards_precede_calls(tmp_path: Path, kind: str) -> None:
    benchmark = dev_benchmark()
    requests = []
    strategies = make_strategies(requests)
    kwargs = {"mode": "mocked", "output": tmp_path}
    if kind == "permission":
        kwargs["mode"] = "live"
    elif kind == "split":
        benchmark = load_benchmark(Path("evals"), "held_out")
    elif kind == "repeats":
        kwargs["repeats"] = 0
    elif kind == "config":
        strategies["unconstrained-json"]._provider.model_id = "different"
    else:
        (tmp_path / "existing.json").write_text("preserve")
    with pytest.raises(ValueError):
        run_representation_experiment(benchmark, strategies, **kwargs)
    assert not requests


def test_cli_opt_in_and_model_guards(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["representation", "--output", str(tmp_path)]) == 2
    assert "--allow-live" in capsys.readouterr().err
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    assert cli.main(["representation", "--allow-live", "--output", str(tmp_path)]) == 2
    assert "qwen2.5:14b" in capsys.readouterr().err


@pytest.mark.parametrize("bad_text", [False, True])
def test_cli_no_network_mocked_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, bad_text: bool) -> None:
    requests = []
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.setattr("decisionlab.strategies.representation.LocalHTTPTransport",
                        lambda: EchoTransport(requests, bad_text))
    assert cli.main(["representation", "--allow-live", "--repeats", "1", "--json", "--output", str(tmp_path)]) == (1 if bad_text else 0)
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["columns"]["labelled-text"]["report"]["summary"]["counts"]["failed_runs"] == (20 if bad_text else 0)
    assert "raw_response" not in captured.out
    assert len(requests) == 40
