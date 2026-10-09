# Decision Robustness Experiment

## Purpose and Contracts

Robustness v1 measures whether equivalent routing inputs produce the same decision.
`RobustnessCase` identifies a base case, dev/held_out split, transformation, and
derived DecisionProblem. `RobustnessPair` retains both original observations and
typed stability diagnostics. `evaluate_pair` consumes observations, not strategies
or answer labels. `run_robustness` accepts existing provider-independent
`StrategyEntry` objects, including offline, mocked and explicitly permitted live
strategies. No new decision strategy or provider is introduced.

Strategy execution still uses `decide(DecisionProblem) -> Recommendation` through
the existing validation/policy boundary. Verification is a separate post-execution
diagnostic. Original outputs are preserved without repair, normalization, option
remapping, or forced agreement. Explanations, assessment order and citation order
are retained but do not define decision equivalence.

## Separate Frozen Fixtures

`evals/robustness/manifest.json`, `dev.jsonl` and `held_out.jsonl` are separate from
the unchanged benchmark. Version 1 binds the source manifest SHA256, source dataset
and policy versions, transformation version `routing-invariance-1`, and artifact
hashes. The loader validates hashes, exact derivation, unique IDs, complete applicable
coverage and split membership before execution. Artifact paths are allowlisted.

There are **179 variants** from all 30 cases: 119 dev and 60 held_out. Six fixed
transformations apply wherever they change the input:

| Transformation | Change | Preserved routing information |
| --- | --- | --- |
| wording | Prefix request and fact statements with neutral wording | Original statements, values and sources |
| distractor | Add a blue-stationery administrative fact | All original facts; added fact is outside routing policy |
| fact_order | Reverse supplied facts | Fact IDs, values and sources |
| option_order | Reverse options | Stable IDs and all option attributes; no positional priority exists |
| redundancy | Repeat supplied statements in context | No additional claims or typed facts |
| formatting | Add whitespace/newlines to request/context/statements | Original words and structured values |

Each type has 30 variants except fact_order (29): route-012 has one fact, so reversing
it is a no-op and is excluded. Problem IDs remain unchanged to avoid confounding
the transformation with an identifier change; fixture/run IDs distinguish variants.
No original fact source/value, constraint, option ID or attribute changes. Tests
exhaustively compare policy results for every existing option and abstention,
alongside structural field preservation. This proves equivalence under the current
routing policy, not arbitrary natural-language interpretation or future policies.

Rebuild only these artifacts with `$env:PYTHONPATH='src'` followed by
`python evals/build_robustness.py`. This script never writes the source benchmark.
Changing derivation requires a new fixture/transformation version.

## Experiment and Metrics

Each strategy runs one base observation per source case/repetition, then every
sorted variant. One full repetition makes **209 executions per strategy** and
179 pairs. The same snapshot/order is shared across strategies. Base observations
are reused across variants; run-based metrics count them once per report scope.

| Metric | Denominator / meaning |
| --- | --- |
| pair agreement | All pairs; both executions must succeed with identical status and selected option ID |
| selection flip | Pairs where both select; selected option ID changes |
| abstention flip | Pairs where both execute successfully; selected versus abstained status changes |
| execution failure | Distinct executions in scope; any failed run, including invalid output |
| known invalid output | Distinct executions; observed schema/reference rejection; API validity can be unknown |
| failed pair | All pairs; either execution failed |
| confidence change | Two successful selections with available confidence; signed variant-minus-base delta and mean absolute delta, with sample count |
| verifier-disposition change | Pairs with both verifier results available; accept/reject/human_review changes |

Exclusive states are `stable_decision`, `changed_decision`, `changed_abstention`,
`execution_failure`, and `invalid_output`. Known invalid output takes precedence
when either run fails schema/reference validation; underlying run/error records
retain simultaneous API failures. Two failed executions never count as agreement.
Two successful abstentions do agree regardless of whether abstention is correct.
A flip between two acceptable options is recorded as a flip, not necessarily an
accuracy failure. Confidence changes can also accompany selection flips and thus
refer to estimates for different options. Missing confidence is never imputed.

## CLI, Reports and Split Methodology

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli robustness --strategies rules tool-assisted --output runs/robustness-offline
python -m decisionlab.cli robustness --split held_out --json --output runs/robustness-held-out
# Paid calls only after configuring API environment variables:
python -m decisionlab.cli robustness --split dev --strategies rules direct-llm structured-llm --allow-live --output runs/robustness-live-dev
```

The last command makes 278 LLM calls before retries (139 dev executions for each
of two LLM strategies). No live experiment was run during implementation. The
repository has two provider-backed LLM strategies; mocked tests exercise both
and ToolAssisted's injectable generation path, rather than inventing a third LLM.

`robustness.json` records manifest/artifact/input hashes, execution order, repeats,
per-run strategy/model/provider/prompt/configuration/usage metadata, original
observations, pair states, and split/transformation summaries. `robustness.txt`
shows split-specific rates and metadata. `--json` switches stdout format. Exit
codes: 0 completed (including flips), 1 execution failures, 2 configuration/load
errors. Default execution is offline and needs no API key.

Dev variants derive exclusively from the 20 dev sources; held-out variants derive
exclusively from the 10 held-out sources. Reports keep both splits separate.
Transformation design is fixed, not selected using observed strategy performance.
No labels, rubrics, expected answers or fixture metadata enter strategy inputs;
this experiment computes stability without benchmark reference scoring. Future
tuning must use dev only and freeze choices before held-out evaluation. Public
held-out cases are not a blind external test. Pooled metrics are descriptive only.

## Results and Limitations

Offline rules and default ToolAssisted both agree on 179/179 pairs: dev 119/119,
held_out 60/60. Neither has selection/abstention flips, execution failures or
verifier-disposition changes; confidence deltas are unavailable. This leaves
ordinary accuracy unchanged (rules 27/30, tools 30/30), reported separately.

Stability measures sensitivity to these transformations; accuracy measures matching
reference outcomes; calibration measures whether estimated probabilities match
observed correctness frequencies. A consistently wrong or always-abstaining
strategy can be perfectly stable. High stability does not validate evidence,
reasoning, policy compliance, accuracy or confidence calibration.

The fixtures are small, synthetic and deliberately mild. Shared bases and repeated
runs create correlated observations. Live model randomness can produce changes
even for identical inputs; no identity-control experiment, randomized execution
order, confidence intervals or causal attribution is implemented. Exact ID matching
may penalize reasonable tie choices. The semantic checks share the existing policy
implementation and cannot prove it correct. Prompt injection, multilingual wording,
external evidence, and realistic adversarial perturbations remain future experiments.
