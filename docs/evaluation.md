# Offline Evaluation Architecture and Metrics

## Boundaries

`evaluation/benchmark.py` loads all frozen artifacts, verifies SHA-256 hashes,
versions, split membership, reference IDs, rubrics, and pair membership, then selects
the requested split. Labels and rubrics remain separate from DecisionProblem inputs.

`runner.py` passes only a deep copy of each problem to a strategy. It revalidates a
snapshot of the output, checks all references, and calls `policy.py`. The policy
uses supplied facts and option attributes; it cannot read labels or strategy code.
It applies `routing-benchmark-1`, including verified-intake provenance, capacity,
approval, explicit supervisor review, abstention review, and material unknowns.
Unsupported hard constraints fail closed. Policy never substitutes a selection or
abstention. A selected output violating a hard rule is retained, flagged, and gated
for human review; it must not be automatically executed.

Malformed schema, unresolved references, and strategy exceptions produce failed
runs. Schema-valid rejected candidates remain diagnostic observations so invalid
citations and attempted hard violations remain measurable. They are never accepted
recommendations. Failed runs have no policy outcome or accepted abstention; consumers
must handle failure explicitly. Error payloads are sanitized rather than exposing
arbitrary exception messages.

`evaluation/evaluator.py` has no strategy imports. It reads the unchanged result
against reference expectations and recomputes hard violations from problem data,
not self-reported eligibility or recorded policy violations. Final human-review
detection is measured separately from the raw strategy flag. `metrics.py` aggregates
the resulting records; `report.py` writes JSON and human-readable text. No component
repairs the strategy result or feeds reference answers back into strategy execution.

## Denominators

Each rate records `numerator`, `denominator`, and `value`; zero denominators yield
null, never invented perfect scores. Counts are run counts when repeats exceed one.
Repeated runs do not create additional independent benchmark cases.

| Metric | Definition |
| --- | --- |
| Outcome correctness | Correct selection or required abstention / all runs; failures count as incorrect |
| Selection correctness | Correct selected option / cases expecting selection; missed selections count as incorrect |
| Abstention correctness | Correct valid abstention / cases expecting abstention; failures count as incorrect |
| Selected-output accuracy | Correct selections / valid selected outputs |
| Decision coverage | Valid selections / all runs |
| Abstention precision | Required valid abstentions / all valid abstentions |
| Schema validity | Schema-valid outputs / outputs whose schema could be checked; exceptions are separately counted |
| Reference validity | Fully valid references / schema-valid candidates |
| Evidence-reference validity | Candidates with no unknown cited fact IDs / schema-valid candidates |
| Evidence-citation validity | Valid fact-reference occurrences / all fact-reference occurrences |
| Hard-constraint violation rate | Schema-valid selected candidates with any violation / all schema-valid selected candidates, including rejected reference failures |
| Human-review accuracy | Correct final review flags / all runs; unavailable flags are incorrect |
| Review precision/recall | TP/(TP+FP), TP/(TP+FN); missing flags on review-required failures count as FN |
| Strategy-review accuracy | Correct raw strategy review flags / all runs |
| Execution failure rate | Failed runs / all runs |
| Repeated-decision agreement | Cases with all successful repeats agreeing on status, option, and final review / cases with multiple repeats |

Review confusion counts include an unavailable count. A failed run in a nonreview
case is not recorded as a true negative; all-case review accuracy still penalizes it.
No hard violation is inferred for a malformed payload whose selection is unavailable.
Reference validity on a candidate with zero citations is vacuously true; explicit
citation counts and zero-citation counts prevent that from implying evidence quality.

Category and split summaries use the same definitions. Categories overlap. Pair
reports require both correct outcomes and the expected change/invariance relation;
they are reported separately, not folded into the outcome score.

## Confidence, Evidence, and Operational Measurements

Only valid selected outcomes with reported confidence enter confidence metrics.
Report sample count, mean confidence, Brier score `(confidence - correctness)^2`,
and five reliability bins of width 0.2 (last bin includes 1). Abstention confidence
and rejected-output confidence are excluded. Confidence remains uncalibrated;
this small benchmark cannot establish calibration.

Diagnostics v1 additionally report weighted ECE, maximum reliability gap, bin
fractions, explicit missing/excluded counts, and fixed-threshold coverage/accuracy
curves. A separately labeled constant .5 reference measures all accepted selections
without imputing strategy confidence. Dev and held_out summaries remain separate;
no thresholds are fitted. See [confidence.md](confidence.md) for formulas,
denominators, strategy audit and interpretation.

Evidence support and specific unknown-handling quality remain manual rubric tasks.
Per-case rubric text accompanies the report; richer required references remain in
the hashed benchmark rubrics. Scores and means are null with zero scored samples.
Valid citation IDs never automatically earn semantic-support credit. No automated
judge or manual-score ingestion has been added.

Wall-clock latency and API usage/cost are not collected in deterministic baseline
runs; availability counts are zero and values null. Aggregation supports timing and
cost when supplied in future run records. No confidence, timing, or costs are fabricated.

## Reproducibility and CLI

The report records dataset, schema, policy, and strategy versions, manifest checksum,
artifact checksums, split, repeats, per-case results, diagnostics, and summaries.
Stable run IDs and sorted JSON with LF endings make repeat baseline runs byte-identical.
It excludes timestamps, host paths, and timing noise. Local reports live under ignored
`runs/`; the benchmark files remain unchanged. Do not pass reports or reference
artifacts into a future strategy prompt.

Run `python -m decisionlab.cli evaluate --benchmark evals --split all --output runs/baseline`
with `PYTHONPATH=src`, or use the installed `decisionlab evaluate` command. Supported
flags include `--json`, `--repeats`, and `--strategy`. Rules remain the offline default;
direct/structured LLM evaluation requires `--allow-live` and environment configuration.
The `compare` command now runs matched strategy evaluations and produces side-by-side
reports; see [comparison.md](comparison.md). Standalone decision commands remain planned.

## Verification Diagnostics

`run_benchmark` also attaches separate post-decision [verifier diagnostics](verification.md).
The evaluator remains read-only and does not use verifier dispositions as correctness
labels or alter any existing metric. Unparsed failures have unavailable verification;
verifier errors are recorded separately from strategy execution failures.

## Structured Strategy Extension

The CLI also supports `--strategy structured-llm --allow-live`. It uses the same
runner, policy, and independent evaluator; no reference artifacts reach the provider.
The core strategy signature is unchanged. Optional execution telemetry records
provider/model, versioned prompt/hash, safe settings, response ID, retries, and usage.
Token totals and availability counts include failures when a valid envelope supplied
usage. Malformed/API failures never count as abstentions.

Baseline runs remain offline and deterministic. Live response IDs, reported usage,
and model outputs vary. Schema-invalid output is distinct from API/refusal/incomplete
failures, whose schema cannot be assessed. Evidence support is unscored and confidence
is uncalibrated. See `structured-llm.md` for settings, opt-in tests, and limitations.
