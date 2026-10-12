"""Saved-evidence presentation tests; no model generation or re-evaluation."""

import json
from pathlib import Path

import pytest

from app import research_data as data
from decisionlab.comparison import ComparisonReport


@pytest.fixture
def report() -> ComparisonReport:
    """Use deterministic existing fixtures via an explicitly synthetic report."""
    from decisionlab.comparison import StrategyEntry, run_comparison
    from decisionlab.evaluation.benchmark import load_benchmark
    from decisionlab.strategies.rules import RuleBasedStrategy

    return run_comparison(load_benchmark(data.ROOT / "evals", "dev"),
                          [StrategyEntry("rules", "1", RuleBasedStrategy(), "offline")])


def write_report(root: Path, report: ComparisonReport) -> Path:
    path = root / "comparison.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    return path


def test_parser_uses_actual_nested_column_schema(tmp_path: Path, report: ComparisonReport) -> None:
    path = write_report(tmp_path, report)
    loaded = data.load_comparison(tmp_path, path.name)
    assert loaded.error is None and loaded.data is not None
    assert loaded.data.model_dump() == report.model_dump()
    rows = data.metric_rows(loaded.data)
    correct = next(row for row in rows if row["Metric"] == "Overall correctness")
    expected = report.columns["rules"].report.summary.rates["outcome_correctness"]
    assert correct["rules"] == data.rate_text(expected)
    assert data.correctness_chart(loaded.data) == [
        {"Condition": "rules", "Correctness (%)": expected.value * 100}
    ]
    assert len(data.case_rows(loaded.data)) == 20


@pytest.mark.parametrize("content", [None, "not json", "[]", "{}"])
def test_missing_or_malformed_report_is_unavailable(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "comparison.json").write_text(content, encoding="utf-8")
    loaded = data.load_comparison(tmp_path, "comparison.json")
    assert loaded.data is None and loaded.error.startswith("Unavailable evidence")


def test_inconsistent_case_count_is_not_displayed(tmp_path: Path, report: ComparisonReport) -> None:
    raw = report.model_dump(mode="json")
    raw["columns"]["rules"]["report"]["summary"]["counts"]["total"] = 999
    (tmp_path / "comparison.json").write_text(json.dumps(raw), encoding="utf-8")
    loaded = data.load_comparison(tmp_path, "comparison.json")
    assert loaded.data is None and "inventory" in loaded.error


def test_zero_denominator_is_not_zero_accuracy() -> None:
    from decisionlab.evaluation.models import Rate

    assert data.rate_text(Rate(numerator=0, denominator=0)) == "Not assessable (0/0)"
    assert data.rate_text(None) == "Unavailable"
    assert data.rate_text(Rate(numerator=0, denominator=4)) == "0/4 (0.0%)"


@pytest.mark.parametrize("content", [None, "[1]", "invalid"])
def test_optional_document_failure(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "protocol.json").write_text(content, encoding="utf-8")
    value, error = data.read_document(tmp_path, "protocol.json")
    assert value is None and error.startswith("Unavailable evidence")


@pytest.mark.parametrize("counts", [{}, {"both_correct": 1}, {"both_correct": "1"}])
def test_incomplete_enforcement_pairs_are_unavailable(tmp_path: Path, counts: dict[str, object]) -> None:
    (tmp_path / "pairs.json").write_text(json.dumps({"paired_counts": counts}), encoding="utf-8")
    loaded = data.read_model(tmp_path, "pairs.json", data.EnforcementPairedAnalysis)
    assert loaded.data is None and loaded.error.startswith("Unavailable evidence")


def test_dashboard_without_local_runs_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    app = testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")).run()
    app.selectbox(key="page").set_value("Research Dashboard").run()
    assert not app.exception
    assert any("Unavailable evidence" in item.value for item in app.warning)
    assert not app.get("arrow_vega_lite_chart")
    assert [button.label for button in app.button] == ["Return to Overview"]
    app.selectbox(key="research-experiment").set_value("Output representation").run()
    assert not app.exception
    assert any("Unavailable evidence" in item.value for item in app.warning)


def test_dashboard_integration_with_synthetic_saved_report(
    tmp_path: Path, report: ComparisonReport, monkeypatch: pytest.MonkeyPatch,
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    path = tmp_path / data.COMPARISONS["Strategy comparison"]["dev"]
    path.parent.mkdir(parents=True)
    path.write_text(report.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(data, "ROOT", tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Research presentation cannot execute a strategy")

    monkeypatch.setattr("decisionlab.runner.run_benchmark", forbidden)
    monkeypatch.setattr("app.playground.decide_json", forbidden)
    app = testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")).run()
    app.selectbox(key="page").set_value("Research Dashboard").run()
    assert not app.exception
    assert any("Artifact-derived metrics" in item.value for item in app.caption)
    assert any('<table class="research-table' in item.value for item in app.markdown)
    app.selectbox(key="split:Strategy comparison").set_value("held_out").run()
    assert not app.exception
    assert any("Unavailable evidence" in item.value for item in app.warning)
    app.selectbox(key="page").set_value("Decision Playground").run()
    assert not app.exception and app.button[0].label == "Run verified decision"


def test_actual_local_evidence_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional archived files are never prerequisites for a fresh checkout."""
    testing = pytest.importorskip("streamlit.testing.v1")
    if not (data.ROOT / "runs/representation/dev-20261009/experiment.json").exists():
        pytest.skip("Local research archive unavailable")
    for splits in data.COMPARISONS.values():
        for relative in splits.values():
            if (data.ROOT / relative).exists():
                assert data.load_comparison(data.ROOT, relative).error is None
    app = testing.AppTest.from_file(str(data.ROOT / "app/streamlit_app.py"), default_timeout=15).run()
    app.selectbox(key="page").set_value("Research Dashboard").run()
    for experiment in data.COMPARISONS:
        app.selectbox(key="research-experiment").set_value(experiment).run()
        assert not app.exception
    assert any("119" in item.value for item in app.markdown)
    assert any("assistant-assisted" in item.value for item in app.warning)
