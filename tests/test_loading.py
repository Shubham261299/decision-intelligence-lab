import hashlib
import json
import shutil
from pathlib import Path

import pytest

from decisionlab.evaluation.benchmark import load_benchmark
from conftest import BENCHMARK_ROOT


def copy_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "evals"
    shutil.copytree(BENCHMARK_ROOT, root)
    return root


def save_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def refresh_hash(root: Path, relative: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["sha256"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    save_manifest(root, manifest)


@pytest.mark.parametrize("split,count", [("dev", 20), ("held_out", 10), ("all", 30)])
def test_loading_requested_split(split: str, count: int) -> None:
    benchmark = load_benchmark(BENCHMARK_ROOT, split)
    assert len(benchmark.problems) == len(benchmark.expectations) == count


def test_checksum_mismatch_fails_before_execution(tmp_path: Path) -> None:
    root = copy_benchmark(tmp_path)
    with (root / "cases/dev.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_benchmark(root)


@pytest.mark.parametrize("change", ["count", "duplicate", "pair", "policy", "path", "split"])
def test_invalid_manifest_rejected(tmp_path: Path, change: str) -> None:
    root = copy_benchmark(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    if change == "count":
        manifest["case_count"] = 29
    elif change == "duplicate":
        manifest["splits"]["dev"][1] = manifest["splits"]["dev"][0]
    elif change == "pair":
        manifest["pairs"][0]["members"][1] = "route-021"
    elif change == "policy":
        manifest["policy_version"] = "unrecognized"
    elif change == "path":
        manifest["sha256"]["../../outside.json"] = "0" * 64
    else:
        manifest["splits"]["extra"] = []
    save_manifest(root, manifest)
    with pytest.raises(ValueError):
        load_benchmark(root)


@pytest.mark.parametrize("change", ["label", "evidence", "version", "empty_line"])
def test_semantic_corruption_rejected_even_with_refreshed_checksum(
    tmp_path: Path, change: str,
) -> None:
    root = copy_benchmark(tmp_path)
    relative = "expectations/dev.jsonl" if change in ("label", "empty_line") else "expectations/rubrics.json"
    path = root / relative
    if change == "label":
        lines = path.read_text().splitlines()
        record = json.loads(lines[0])
        record["acceptable_option_ids"] = ["unknown"]
        lines[0] = json.dumps(record)
        path.write_text("\n".join(lines) + "\n")
    elif change == "empty_line":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        raw = json.loads(path.read_text())
        if change == "evidence":
            raw["cases"]["route-001"]["required_evidence_ids"] = ["unknown"]
        else:
            raw["version"] = "other"
        path.write_text(json.dumps(raw))
    refresh_hash(root, relative)
    with pytest.raises(ValueError):
        load_benchmark(root)
