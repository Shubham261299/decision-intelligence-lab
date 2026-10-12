"""Offline presentation integration; no benchmark labels or provider access."""

from pathlib import Path

import pytest

from app import playground
from decisionlab.models import DecisionProblem, Recommendation
from decisionlab.strategies.rules import RuleBasedStrategy


@pytest.mark.parametrize("example,selected,disposition", [
    ("Routine routing", "routine", "accept"),
    ("Missing urgency", None, "human_review"),
])
def test_examples_use_verified_offline_execution(
    example: str, selected: str | None, disposition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Demo must not load providers or benchmark expectations")

    monkeypatch.setattr("decisionlab.providers.factory.provider_from_environment", forbidden)
    monkeypatch.setattr("decisionlab.evaluation.benchmark.load_benchmark", forbidden)
    monkeypatch.setenv("DECISIONLAB_PROVIDER", "invalid")
    text = playground.example_json(example)
    result = playground.decide_json(text)
    assert result.run.status == "success"
    assert result.run.recommendation is not None
    assert result.run.recommendation.selected_option_id == selected
    assert result.run.recommendation.status == ("selected" if selected else "abstained")
    assert result.run.policy_result is not None
    assert result.run.policy_result.requires_human_review == (selected is None)
    assert result.verification is not None
    assert result.verification.disposition == disposition
    assert result.verification.evidence_support == "not_assessed"
    assert result.run.metadata.dataset_version is None
    assert result.model_dump_json() == playground.decide_json(text).model_dump_json()


@pytest.mark.parametrize("text", ['{"sensitive-value":', '{}', '[]'])
def test_bad_input_never_executes(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid inputs must not execute")

    monkeypatch.setattr(playground, "run_verified_decision", forbidden)
    with pytest.raises(ValueError, match="valid DecisionProblem") as error:
        playground.decide_json(text)
    assert "sensitive-value" not in str(error.value)


def test_example_is_allowlisted_and_missing_file_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unknown example"):
        playground.example_json("../../.env")
    monkeypatch.setattr(playground, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Example unavailable"):
        playground.example_json("Routine routing")


def test_strategy_failure_stays_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: RuleBasedStrategy, problem: DecisionProblem) -> Recommendation:
        raise RuntimeError("sensitive-exception")

    monkeypatch.setattr(RuleBasedStrategy, "decide", fail)
    result = playground.decide_json(playground.example_json("Routine routing"))
    assert result.run.status == "failure"
    assert result.run.recommendation is None
    assert result.verification is None
    assert result.run.error is not None and result.run.error.code == "strategy_exception"
    assert "sensitive-exception" not in result.model_dump_json()


def test_reference_failure_is_not_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = DecisionProblem.model_validate_json(playground.example_json("Routine routing"))
    candidate = RuleBasedStrategy().decide(problem)
    candidate.assessments[0].evidence_ids = ["nonexistent"]
    monkeypatch.setattr(RuleBasedStrategy, "decide", lambda self, problem: candidate)
    result = playground.decide_json(problem.model_dump_json())
    assert result.run.status == "failure" and result.run.recommendation is None
    assert result.references_valid is False
    assert result.verification is not None and result.verification.disposition == "reject"
    assert result.candidate is not None and result.candidate.assessments[0].evidence_ids == ["nonexistent"]


def test_policy_violation_remains_separate_from_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    text = playground.example_json("Routine routing")
    candidate = RuleBasedStrategy().decide(DecisionProblem.model_validate_json(text))
    candidate.selected_option_id = "expedited"
    monkeypatch.setattr(RuleBasedStrategy, "decide", lambda self, problem: candidate)
    result = playground.decide_json(text)
    assert result.run.status == "success"
    assert result.run.policy_result is not None and result.run.policy_result.detected_violations
    assert result.verification is not None and result.verification.disposition == "reject"


@pytest.mark.parametrize("example,decision,verifier", [
    ("Routine routing", "routine", "Accept"),
    ("Missing urgency", "Abstained", "Human Review"),
])
def test_rendered_playground(example: str, decision: str, verifier: str) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(playground.ROOT / "app/streamlit_app.py"))
    app.run()
    app.selectbox(key="page").set_value("Decision Playground").run()
    app.radio[0].set_value(example).run()
    app.button[0].click().run()
    assert not app.exception
    summary = next(item.value for item in app.markdown if 'class="decision-summary"' in item.value)
    for value in ["Success", decision, "Required" if decision == "Abstained" else "Not required", verifier]:
        assert f'class="summary-value">{value}</div>' in summary
    assert app.expander[0].label == "Inspect or edit input JSON"
    assert app.expander[0].proto.expanded is False
    table = next(item.value for item in app.markdown if 'class="eligibility-table"' in item.value)
    assert "Reasons" in table and "References" in table
    headings = [item.value for item in app.subheader]
    assert "Unknown or unresolved inputs" in headings
    assert "Policy findings" in headings and "Verifier findings" in headings
    if decision == "Abstained":
        assert any("Human review required" in item.value for item in app.warning)
        assert any("Human Review" in item.value for item in app.warning)
    else:
        assert not app.warning
    assert len(app.json) == 1


def test_rendered_invalid_input() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(playground.ROOT / "app/streamlit_app.py")).run()
    app.selectbox(key="page").set_value("Decision Playground").run()
    app.text_area[0].set_value('{"sensitive-value":')
    app.button[0].click().run()
    assert not app.exception
    assert not app.metric
    assert len(app.error) == 1 and "valid DecisionProblem" in app.error[0].value
    assert "sensitive-value" not in app.error[0].value


def test_eligibility_html_escapes_editable_content(monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    result = playground.decide_json(playground.example_json("Routine routing"))
    assert result.run.recommendation is not None
    result.run.recommendation.assessments[0].reasons = ['<script>alert("test")</script>']
    monkeypatch.setattr(playground, "decide_json", lambda text: result)
    app = testing.AppTest.from_file(str(playground.ROOT / "app/streamlit_app.py")).run()
    app.selectbox(key="page").set_value("Decision Playground").run()
    app.button[0].click().run()
    assert not app.exception
    table = next(item.value for item in app.markdown if 'class="eligibility-table"' in item.value)
    assert "<script>" not in table
    assert "&lt;script&gt;" in table


def test_presentation_distinguishes_execution_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")

    def fail(self: RuleBasedStrategy, problem: DecisionProblem) -> Recommendation:
        raise RuntimeError("sensitive-error")

    monkeypatch.setattr(RuleBasedStrategy, "decide", fail)
    app = testing.AppTest.from_file(str(playground.ROOT / "app/streamlit_app.py")).run()
    app.selectbox(key="page").set_value("Decision Playground").run()
    app.button[0].click().run()
    assert not app.exception
    summary = next(item.value for item in app.markdown if 'class="decision-summary"' in item.value)
    assert '>Failure</div>' in summary and '>Abstained</div>' not in summary
    assert any("Execution failure" in item.value for item in app.error)
    assert all("sensitive-error" not in item.value for item in app.error)


@pytest.mark.parametrize("marker", [
    "![probe](https://example.invalid/image)",
    "[probe](https://example.invalid/link)",
    "<https://example.invalid/autolink>",
])
def test_input_derived_markdown_is_literal(marker: str) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    from app.streamlit_app import plain_markdown

    problem = DecisionProblem.model_validate_json(playground.example_json("Routine routing"))
    # Both a hard-constraint ID (unknown description) and option ID (caption/table)
    # originate in editable input. Run the real offline path, not a repaired output.
    problem.constraints[0].id = marker
    problem.options[0].id = marker
    before = playground.decide_json(problem.model_dump_json()).model_dump_json()
    app = testing.AppTest.from_file(str(playground.ROOT / "app/streamlit_app.py")).run()
    app.selectbox(key="page").set_value("Decision Playground").run()
    app.text_area[0].set_value(problem.model_dump_json())
    app.button[0].click().run()
    assert not app.exception
    assert any(plain_markdown(marker) in item.value for item in app.warning)
    assert any(plain_markdown(marker) in item.value for item in app.caption)
    # Unescaped image/link/autolink syntax never reaches Markdown rendering.
    assert all(marker not in item.value for item in [*app.warning, *app.caption])
    assert before == playground.decide_json(problem.model_dump_json()).model_dump_json()


def test_plain_markdown_preserves_ordinary_text() -> None:
    pytest.importorskip("streamlit")
    from app.streamlit_app import plain_markdown

    assert plain_markdown("Human Review required") == "Human Review required"
    assert plain_markdown("option_a *literal* [notes]") == r"option\_a \*literal\* \[notes\]"
