"""Reproduce only robustness artifacts; never write the frozen source benchmark.

Run from the root with PYTHONPATH=src: python evals/build_robustness.py
Changing transformations requires a new dataset/transformation version.
"""

import hashlib
import json
from pathlib import Path

from decisionlab.evaluation.benchmark import load_benchmark
from decisionlab.robustness import RobustnessCase, RobustnessManifest, TRANSFORMATIONS, transform


def main() -> None:
    source = Path(__file__).resolve().parent
    benchmark = load_benchmark(source)
    root = source / "robustness"
    root.mkdir(exist_ok=True)
    hashes: dict[str, str] = {}
    count = 0
    for split in ("dev", "held_out"):
        records = [RobustnessCase.model_validate({
            "id": f"{p.id}:{kind}", "base_case_id": p.id, "split": split,
            "transformation": kind, "problem": transform(p, kind),
        }) for p in benchmark.problems if p.id in benchmark.manifest.splits[split]
            for kind in TRANSFORMATIONS if transform(p, kind) != p]
        data = "".join(json.dumps(r.model_dump(mode="json"), sort_keys=True,
                                  separators=(",", ":"), allow_nan=False) + "\n" for r in records).encode()
        name = f"{split}.jsonl"
        (root / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        count += len(records)
    manifest = RobustnessManifest(
        source_manifest_sha256=benchmark.manifest_sha256,
        source_dataset_version=benchmark.manifest.version,
        policy_version=benchmark.manifest.policy_version, case_count=count, sha256=hashes,
    )
    (root / "manifest.json").write_bytes((json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, indent=2,
    ) + "\n").encode())


if __name__ == "__main__":
    main()
