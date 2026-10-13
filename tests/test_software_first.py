"""Public profile withholds results even when historical files exist locally."""

import json
from pathlib import Path

import pytest

from app import research_data as data
from app.release_scope import research_is_withheld


@pytest.mark.parametrize("content", [None, "invalid", "{}", '{"release_mode":"software-first"}'])
def test_default_is_withheld(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "software-first-release.json").write_text(content, encoding="utf-8")
    assert research_is_withheld(tmp_path)


def test_private_local_mode_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "software-first-release.json").write_text(json.dumps({
        "release_mode": "private-research-review", "research_evidence": "available_for_local_review",
    }), encoding="utf-8")
    assert not research_is_withheld(tmp_path)


def test_public_pages_never_load_research(monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Public pages must not read historical evidence")

    monkeypatch.setattr(data, "load_comparison", forbidden)
    monkeypatch.setattr(data, "read_document", forbidden)
    original = data.read_model

    def only_manifest(root: Path, relative: str, model: type) -> object:
        assert relative == "evals/manifest.json"
        return original(root, relative, model)

    monkeypatch.setattr(data, "read_model", only_manifest)
    app = testing.AppTest.from_file(str(data.ROOT / "app/streamlit_app.py")).run()
    assert not app.exception and any("does not load local raw reports" in item.value for item in app.info)
    assert any("does not remove them" in item.value for item in app.info)
    assert not app.get("arrow_vega_lite_chart")
    app.selectbox(key="page").set_value("Research Dashboard").run()
    assert not app.exception and any("does not load local raw reports" in item.value for item in app.info)
    assert any("existing repository history" in item.value and "does not remove them" in item.value
               for item in app.info)
    assert any("existing committed findings and annotations are preserved" in item.value
               for item in app.caption)
    assert all("software-first release" not in item.value for item in [*app.info, *app.caption])
    assert not app.selectbox(key="page").value == "Overview"
    assert not app.get("arrow_vega_lite_chart")


def test_manifest_excludes_review_and_history() -> None:
    manifest = json.loads((data.ROOT / "software-first-release.json").read_text())
    included = {path for paths in manifest["files"].values() for path in paths}
    assert "tests/test_human_review.py" not in included
    assert "docs/reviews/experiment-3-first-pass-v1.json" not in included
    assert "docs/research-findings.md" not in included
    assert not any(path.startswith(("runs/", "docs/research/", ".git/")) for path in included)
    assert all((data.ROOT / path).is_file() for path in included)
