"""Landing content and native navigation; all tests stay local and offline."""

from pathlib import Path

import pytest

from app import research_data as data

ENTRY = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def test_overview_is_default_and_evidence_absence_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    app = testing.AppTest.from_file(str(ENTRY)).run()
    assert not app.exception
    assert app.selectbox(key="page").value == "Overview"
    assert any("measurable, structured, explainable and reliable" in item.value for item in app.markdown)
    assert any("does not load local raw reports" in item.value for item in app.info)
    assert any("existing repository history" in item.value and "does not remove them" in item.value
               for item in app.info)
    assert not app.dataframe
    assert any("independent blind test" in item.value for item in app.warning)
    assert any("semantic evidence support" in item.value for item in app.markdown)
    assert any("(https://github.com/Shubham261299/decision-intelligence-lab)" in item.value
               for item in app.caption)
    assert all("pending manual confirmation" not in item.value for item in app.caption)


@pytest.mark.parametrize("button,page", [
    ("open-playground", "Decision Playground"),
    ("open-research", "Research Dashboard"),
])
def test_overview_buttons_navigate_without_execution(
    button: str, page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(data, "ROOT", tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Navigation must not execute decisions or experiments")

    monkeypatch.setattr("app.playground.decide_json", forbidden)
    monkeypatch.setattr("decisionlab.runner.run_benchmark", forbidden)
    app = testing.AppTest.from_file(str(ENTRY)).run()
    app.button(key=button).click().run()
    assert not app.exception
    assert app.selectbox(key="page").value == page
    app.selectbox(key="page").set_value("Overview").run()
    assert not app.exception and app.button(key=button)


def test_overview_reports_observations_from_saved_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    # Use an existing archive when available; never generate a replacement experiment.
    relative = data.COMPARISONS["Strategy comparison"]["dev"]
    source = data.ROOT / relative
    if not source.is_file():
        pytest.skip("Local research archive unavailable")
    report = data.load_comparison(data.ROOT, relative).data
    assert report is not None
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/research-findings.md").write_bytes((data.ROOT / "docs/research-findings.md").read_bytes())
    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "software-first-release.json").write_text(
        '{"release_mode":"private-research-review","research_evidence":"available_for_local_review"}',
        encoding="utf-8",
    )
    app = testing.AppTest.from_file(str(ENTRY)).run()
    assert not app.exception
    tables = [item.value for item in app.markdown if '<table class="research-table' in item.value]
    assert len(tables) == 1
    for name, column in report.columns.items():
        assert f'<td>{name}</td>' in tables[0]
        assert data.rate_text(column.report.summary.rates["outcome_correctness"]) in tables[0]
    assert any("Documented interpretation" in item.value for item in app.markdown)


def test_return_to_overview_callback() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ENTRY)).run()
    app.button(key="open-playground").click().run()
    app.button(key="return-overview").click().run()
    assert not app.exception and app.selectbox(key="page").value == "Overview"
