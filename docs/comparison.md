# Strategy Comparison Experiment

## Running the Experiment

The `compare` command loads one checksum-verified benchmark snapshot, then runs
each selected strategy on exactly the same cases and repetitions. The existing
runner isolates inputs, captures failures, applies policy without repairing
recommendations, and calls the unchanged strategy-independent evaluator. Neither
expectations nor rubric/pair metadata reach a strategy.

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
# Safe default: offline baseline only, no key required.
python -m decisionlab.cli compare --repeats 2 --output runs/comparison-rules
# After exporting API settings: explicit paid three-strategy experiment.
python -m decisionlab.cli compare --strategies rules direct-llm structured-llm --allow-live --split dev --repeats 2 --output runs/comparison-live
```

The live command makes 80 LLM calls before configured retries: 20 cases times two
repeats times two LLM strategies. It reads validated environment settings once and
constructs separate adapters with the same settings/model. Configuration is described
in [structured-llm.md](structured-llm.md); no credentials or secrets enter reports.
Default CLI selection is `rules`; a single column is an offline smoke run, not a
three-strategy experiment. Any LLM selection requires `--allow-live` before calls.

`--output` writes `comparison.json` and a Markdown table in `comparison.txt`.
`--json` emits machine-readable JSON to stdout; otherwise stdout contains the table.
Exit codes are 0 for completed evaluations, including wrong decisions; 1 if any
column has execution failures; 2 for invalid configuration/loading/output errors.
Other columns still finish when a strategy has per-case failures. There is no
command for merging previously saved reports; `compare` runs a matched experiment.

## Reproducibility and Metadata

Comparison schema version 1 records ordered case IDs, split, repeats, execution
order, dataset/schema/policy versions, manifest hash, and a canonical input hash.
Every column embeds its complete EvaluationReport, including artifact hashes,
per-case outputs and diagnostics, category/split summaries, and paired-case results.
Strategy version, provider, requested/returned model, prompt version/hash, safe
settings, response ID, retries, and usage remain available per run. Compact metadata
profiles omit response IDs/retry counts only; the embedded records retain both.

Provider/model/configuration differences produce a comparability warning. This
check excludes expected prompt hash, parser version, and output-format differences;
it is not certification that experiments isolate a causal variable. Strategies run
sequentially in user-selected blocks, with repetitions inside each case. Order is
recorded, not randomized. Serialization is deterministic for identical observations;
live model calls are stochastic and may change across time.

## Interpreting the Metrics

| Metric | Denominator and interpretation |
| --- | --- |
| Overall correctness | All case-repetitions; failed runs count as unsuccessful |
| Selection correctness | Cases expecting selection; any acceptable option earns credit |
| Abstention correctness | Cases requiring abstention; failures never count as abstentions |
| Selected-output accuracy / coverage | Accuracy among successful selections / selections over all runs; read together |
| Execution failures | All runs; inspect failure-code counts to distinguish API, format, and reference errors |
| Known invalid output | All runs; union of explicit schema or reference rejection, counted once per run |
| Format/envelope failure | All runs; `invalid_schema` or `provider_malformed_response`, separated by code |
| Human-review precision/recall | Final policy review TP/(TP+FP), TP/(TP+FN); unavailable review is not a detected positive |
| Hard policy violations | Violating outputs over schema-valid selected candidates, including rejected reference candidates |
| Evidence citation validity | Valid citation occurrences over all citation occurrences, including rejected candidates |
| Evidence reference validity | Outputs with valid references over reference-assessable outputs; empty citations can be structurally valid |
| Evidence support | Manual rubric score/sample count; currently unscored, not inferred from citation validity |
| Confidence | Uncalibrated selected-output sample count, mean, Brier score, and reliability bins; no replacement for missing confidence |
| Repeated agreement | Cases with multiple repetitions; all outcomes, option IDs, and final review flags must match, with no failed repeat |
| Usage/cost | Available token/cost totals with measurement counts; missing is null, not zero cost |

All rates retain numerator, denominator, and nullable value in JSON. A zero
denominator appears as `0/0 (n/a)` in text. API errors without a usable candidate
have unknown output validity; a low known-invalid rate does not establish validity
of unobserved outputs. Inspect schema/reference sample denominators in the embedded
summaries. Costs are never inferred from model names or token counts. Failed retry
attempts may have unknown billed usage. Timing remains unmeasured.

Four distinctions are essential:

- A wrong selection is an accepted, structurally valid recommendation that misses
  the reference options. It can also violate hard policy; the labels overlap.
- A valid abstention is a successful execution with no selected option. It can be
  required and correct or unnecessary and incorrect; both are counted explicitly.
- A policy violation is a separately recorded breach, not a corrected selection.
  Human review can be required even for a correct, compliant selection.
- An execution/format failure has no accepted recommendation. It is never silently
  converted into abstention or removed from correctness denominators.

High repeated agreement can describe consistently wrong decisions. Different
acceptable tied options can lower exact agreement even when all selections are
correct. Confidence diagnostics on a small selected subset do not prove calibration.

## Offline Tests and Example

Programmatic `run_comparison` accepts explicit StrategyEntry objects. Register
mocked provider-backed DirectLLM/StructuredLLM instances with `mode="mocked"` and
RuleBased with `mode="offline"`. Entries declared `live` require `allow_live=True`.
Callers must label modes truthfully; this guard is not a network sandbox. The CLI
always labels its actual provider-backed entries live. Normal pytest execution
blocks unmocked HTTP and needs no API key.

```bash
python -m pytest -q tests/test_comparison.py tests/test_comparison_cli.py --basetemp .pytest_cache/comparison-tests
python -m pytest -q --basetemp .pytest_cache/foundation-tests
```

See [mocked sample results](comparison-results.md) for reproducible local artifacts.
Tests use real strategy classes with fake provider responses, cover matched inputs,
metadata, reporting, failures, mutations, repeated agreement, and CLI opt-in.

## Verification Diagnostics

The optional offline `tool-assisted` column adds [bounded local lookup tools](tool-assisted.md)
and a deterministic generator. Select it with `--strategies rules tool-assisted`;
it needs no live flag or credentials. It uses the same original cases and retains
tool provenance in each observation. Its generation/tie rules differ from the original
rule baseline, so a score gain cannot be attributed to LLM tool use or extra knowledge.

The default benchmark path now adds [verification diagnostics](verification.md) to
each parsed recommendation/candidate. Comparison text includes verifier dispositions
and unavailable/error counts. These are separate from original strategy scores;
reject/review findings never rewrite outcomes or human-review metrics. Unparseable
provider output has no verification result. A verifier accept is not a ground-truth
correctness label or a scored evidence-support judgment.

## Conclusions Not Yet Justified

The separate [robustness experiment](robustness.md) uses equivalent-input variants
to measure decision/abstention flips, failures, confidence changes and verifier
changes. Run it with `decisionlab robustness`; it does not add transformed cases to
ordinary accuracy or calibration reports, and stability is not correctness.

[Confidence diagnostics v1](confidence.md) extend each strategy's split summaries
with distribution/reliability bins, Brier, weighted ECE, maximum gap and fixed
coverage curves. Text displays dev and held_out separately. Missing confidence
remains unavailable; the .5 reference is a separate comparator. Confidence never
changes correctness, review, verifier dispositions or strategy execution. No
held-out threshold tuning or calibrated-confidence claim is made.

No live comparison has been performed. Rule-shaped mocked outputs prove plumbing,
not LLM accuracy, superiority, reliability, evidence quality, or calibration.
The small public synthetic holdout is not an external blinded benchmark. Prompts,
schema tokens, parsing conventions, and fixed call order differ; a future score
difference cannot automatically be attributed solely to structured output or
reasoning quality. No significance tests, randomized trials, uncertainty intervals,
production safety claims, or generalization across domains are established. Do not
tune on held-out labels, discard failed runs, or pick a winner from one metric.
