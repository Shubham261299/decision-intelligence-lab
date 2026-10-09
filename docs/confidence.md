# Confidence and Calibration Experiment

## Contract and Strategy Audit

Confidence contract v1 (`decisionlab.confidence.ConfidenceEstimate`) describes an
estimated probability that a **selected option is acceptable**. In evaluation,
the event is membership in the benchmark's acceptable option IDs. It does not
predict verifier acceptance, evidence quality, human-review need, or abstention
correctness. A selected option can be correct and still require review.

Probabilities are finite numbers in [0, 1]; booleans and numeric strings are
rejected at domain/contract boundaries. `None` means unavailable, never zero.
An estimate records its target, contract version, source (`strategy_reported`
or `unavailable`), and calibration status (`uncalibrated` or `unavailable`).
The original Recommendation remains unchanged during measurement.

| Strategy | Existing production behavior | Interpretation |
| --- | --- | --- |
| RuleBased | No confidence assigned; defaults to `None` | Unavailable |
| DirectLLM | Optional decimal or `unavailable` in versioned text prompt; strict parser | Uncalibrated model estimate |
| StructuredLLM | Required nullable transport field, optional estimate in prompt | Uncalibrated model estimate |
| ToolAssisted | Default deterministic generator leaves `None`; injected generators may report confidence | Unavailable by default; otherwise uncalibrated |

No strategy, prompt, selection rule, benchmark label, review gate, or correctness
formula is changed. Invalid confidence remains an execution/validation failure.

## Metrics and Denominators

Diagnostics v1 score only accepted **selected outputs with reported confidence**.
Let `p` be confidence and `y` equal 1 for an acceptable selected option, else 0.
Wrong selections, including policy-violating accepted outputs, remain samples.
Abstentions have no selection event; their raw confidence is preserved but excluded.
Failed runs and rejected candidates are excluded, never treated as abstentions.

- Distribution: sample count, mean, and five equal-width bins `[0,.2)`, …,
  `[.8,1]`; each bin records count, sample fraction, mean confidence and accuracy.
- Brier score: mean `(p-y)^2`; lower is better for this binary event.
- Reliability gap: absolute difference between bin mean confidence and accuracy.
  ECE (`calibration_error`) is the count-weighted mean gap; maximum gap is also
  reported. Empty bins have null measurements, not artificial perfect calibration.
- Coverage curve: predeclared inclusive thresholds `0,.2,.4,.6,.8,1`.
  Coverage divides retained selections by **all runs**, including failures and
  abstentions. Retention divides by confidence-bearing selections. Accuracy
  divides correct retained selections by retained selections; zero denominator
  yields null. These are diagnostics, not operational acceptance thresholds.
- Availability: selected/missing counts, excluded abstention confidence count,
  and excluded execution failure count accompany the metrics.

A separate **fixed 0.5 reference** is scored on all accepted selections, including
ones without confidence. Its Brier score is 0.25 whenever a selection exists;
ECE is `abs(.5 - selected accuracy)`. This deterministic uninformed comparator
is labeled `fixed_reference`; it never fills Recommendation.confidence. Its
sample population may differ from reported confidence, so compare denominators.
With no selections, its scores are null.

## Development and Held-Out Methodology

The frozen manifest identifies 20 dev and 10 held_out cases. Existing runners
compute diagnostics independently in `by_split.dev.confidence` and
`by_split.held_out.confidence`; JSON also retains descriptive pooled and category
summaries. The bin edges, thresholds, reference probability and formulas are
fixed in diagnostic version 1, independent of labels. No threshold selection,
confidence fitting, prompt tuning or recalibration is performed here.

Use dev only for future fitting/design, freeze that choice before inspecting
held-out performance, then report held-out once for the frozen configuration.
This implementation has no fitted calibrator or threshold optimizer. Strategies
receive only original problem inputs; labels enter the independent evaluator.
Repeated runs count as observations, not independent new problems.

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli compare --strategies rules tool-assisted --split all --output runs/confidence-offline
python -m decisionlab.cli evaluate --strategy rules --split held_out --output runs/confidence-held-out
```

JSON contains bins, curves and reference scores; text shows split-specific Brier,
ECE, availability, and comparison coverage/retained accuracy. Existing live opt-in
requirements still apply to LLM columns. Default tests and the commands above
need no credentials or API calls.

## Interpretation and Limits

Confidence is a claim about uncertainty, not a correctness label. A confidently
wrong selection lowers accuracy and worsens Brier score. Calibration concerns
agreement between probabilities and observed frequencies; accuracy concerns how
often a selected option is acceptable. A constant .5 estimate on half-correct
selections can have zero empirical ECE while selection accuracy is only .5.
Neither calibration nor accuracy establishes sound reasoning or policy compliance.

The real offline rule baseline selects 16/30 cases, all correctly, but reports
no confidence. Its reported Brier/ECE are null; the separate fixed .5 reference
has Brier .25 and ECE .5. Selection-only calibration does not penalize its three
unnecessary tie abstentions; unchanged outcome correctness remains 27/30.

A deterministic test using benchmark route-001 (correct, .8), route-002 (wrong,
.6), and route-011 (correct abstention, .99) yields two calibration samples,
Brier .20, ECE .40, maximum gap .60; the abstention is excluded. These are authored
test probabilities, not observations about an LLM's capabilities.

Thirty public synthetic cases, sparse bins, selective missingness, differing
coverage and correlated repeated runs make estimates unstable. ECE depends on
binning and can look favorable with very few samples. No calibrated-confidence,
generalization, superiority, statistical significance or production-threshold
claim is justified. No live confidence experiment has been run.
