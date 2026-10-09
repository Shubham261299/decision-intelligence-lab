# DecisionLab: Interpretation of the First Three Live Experiments

## Scope and Provenance

These are descriptive measurements of local `qwen2.5:14b` on the frozen synthetic
service-routing benchmark v1.0.0. They do not establish general model superiority,
statistical significance, calibrated confidence, or production reliability.

All experiments used temperature 0, 2,000 maximum output tokens, a 180-second
timeout, no provider retries, and `http://127.0.0.1:11434`. Expectations remained
outside strategy inputs. Primary evaluation, policy and verification definitions
were unchanged. Experiments 2 and 3 used only dev; no new held-out run was made.

Experiment 3 completed on 2026-10-09 in
`runs/representation/dev-20261009/`. A pre-existing interrupted 18-observation
attempt in `runs/representation/dev/` was preserved and excluded. The fresh run
executed the entire predeclared schedule, not selective replacement cases.

## What the Three Experiments Establish

| Experiment | Design | Measured outcome | Interpretation |
| --- | --- | --- | --- |
| 1 | Rules vs original DirectLLM vs StructuredLLM; 30 cases, one repetition | Correct: 27/30, 18/30, 21/30. LLM execution failures: 10 vs 6 | A pipeline difference exists, but prompts, representation, parser and enforcement were confounded |
| 2 | Matched unconstrained JSON vs schema-enforced JSON; 20 dev cases, three repetitions | Both 46/60 correct; 59/59 available complete candidates identical; one enforced-arm timeout | No observed decision or contract benefit from enforcement under this particular shared JSON prompt |
| 3 | Matched labelled text vs unconstrained JSON; 20 dev cases, three repetitions | 42/60 vs 48/60 correct; 42/42 jointly successful decision targets identical | Representation/instructions/parser affect usability; successful paired targets provide no evidence of better JSON selection reasoning |

Experiment 2 tests enforcement with its own prompt, including an explicit schema
description. It does not prove enforcement has zero effect under every prompt or
rule out an interaction with the original StructuredLLM prompt. Experiment 3
controls semantic policy instructions and inputs, but necessarily changes output
instructions and parsers together. It is not a pure parser-only intervention.
Cross-experiment score changes cannot be attributed to one variable: prompts,
execution order and backend state differ between experiments.

## Experiment 3: Primary Results

There were 120 scheduled attempts and 119 returned model responses. Both conditions
used normal text generation without Ollama `format` or schema enforcement. The
schedule alternated condition order; each condition was first in 30 of 60 pairs.

| Metric | Labelled text | Unconstrained JSON |
| --- | --- | --- |
| Overall correctness | 42/60 (70%) | 48/60 (80%) |
| Selection correctness, selection-required cases | 33/36 (91.7%) | 36/36 (100%) |
| Required-abstention correctness | 9/24 (37.5%) | 12/24 (50%) |
| Execution failures | 18/60 (30%) | 0/60 |
| Known invalid outputs | 17/60 | 0/60 |
| Schema validity, assessable responses | 45/59 | 60/60 |
| Reference validity, schema-valid candidates | 42/45 | 60/60 |
| Fact-citation validity | 150/159 | 261/261 |
| Selected-output accuracy | 33/33 | 36/48 |
| Policy-violating selected candidates | 3/36 | 12/48 |
| System-review precision / recall | 28.6% / 44.4% | 81.8% / 100% |
| Verifier accept / reject / review / unavailable | 0 / 3 / 42 / 15 | 27 / 12 / 21 / 0 |
| Repeated successful decision/review agreement | 14/20 | 20/20 |
| Input tokens, available samples | 42,587 (59) | 45,519 (60) |
| Output tokens, available samples | 7,893 (59) | 16,453 (60) |
| Mean / median provider-reported latency | 33.60s / 32.91s (59) | 63.81s / 56.49s (60) |

Latency is Ollama's reported total duration, not complete experiment wall time.
The timeout has unavailable latency and usage. Monitoring experienced delays of
unknown cause; timing comparisons are descriptive. Local execution incurs no
cloud API charge, but resource use was not priced. JSON generated substantially
more output tokens; this limits interpretations of its longer latency.

The labelled arm's 100% selected-output accuracy reflects filtering: malformed
wrong outputs disappear from that denominator. Its lower reported policy violation
count likewise does not establish safer underlying decisions. Malformed outputs
cannot be scored for hard violations in the unchanged evaluator.

## Prospective Raw-Response Diagnosis

Mechanical rejection and decision-target correctness are separate axes:

| Labelled-text failure stage | Count | Acceptable explicit targets | Incorrect explicit targets | Unknown |
| --- | --- | --- | --- | --- |
| Strict parser rejection | 11 | 6 | 5 | 0 |
| Schema/domain rejection | 3 | 0 | 3 | 0 |
| Invalid fact references | 3 | 0 | 3 | 0 |
| Provider timeout | 1 | 0 | 0 | 1 |
| Total | 18 | 6 | 11 | 1 |

No failed JSON responses required classification. Its twelve accepted wrong
decisions are still substantive benchmark errors and policy violations.

Thus most labelled execution failures occur at a contract boundary (17/18), but
most available rejected responses also state an incorrect decision (11/17).
"Parser failure" and "wrong decision" are not mutually exclusive explanations.
Six rejected responses have acceptable targets, not confirmed fully correct
recommendations. Diagnostic inspection never repairs output or rescues a score.

| Case, three repetitions each | Labelled-text observations | JSON observations | Diagnostic interpretation |
| --- | --- | --- | --- |
| `route-008` | Routine target twice; empty `Constraints:` triggers parser rejection. Third attempt times out without a response | Routine selection in all repeats | Cross-route fallback violates the urgent-routing policy; missing response remains unknowable |
| `route-010` | Acceptable specialist choice; `Unknowns: none` precedes assessment blocks, rejected for order | Acceptable specialist choice | Three JSON-only correct outcomes arise from text-contract failure; text also requests unnecessary review |
| `route-011` | Required abstention and relevant missing-urgency explanation; empty `Unknowns:` header is rejected | Required abstention | Three further JSON-only correct outcomes arise from text-contract failure; JSON also falsely describes the resolved specialist flag as unknown |
| `route-012` | Invents absent `specialist_required=true`, selects specialist, cites a nonexistent fact | Assumes absent specialist need is false, selects routine, cites only existing urgency | Both incorrectly replace missing information with certainty; valid references do not establish factual support |
| `route-013` | Routine fallback; missing assessment fields trigger parser rejection | Routine fallback, accepted | Both ignore unknown required expedited capacity and the no-fallback policy |
| `route-015` | Selects exhausted specialist queue; repeats literal `eligible or ineligible or unknown`, rejected by domain validation | Selects exhausted specialist queue, accepted | Both subordinate the hard capacity rule to the specialist requirement |

These descriptions are assistant inspection of preserved text and supplied problem
data, not independent human annotations or evidence-support scores.

## Paired Decisions, Review and Repetition

All 42 jointly successful pairs have the same selected option or abstention.
Across 60 pairs: both correct 42, JSON-only correct 6, labelled-only correct 0,
neither correct 12. The six-outcome gap is entirely `route-010` and `route-011`
contract rejection. It is not an observed improvement in those decision targets.

Among all 59 pairs with returned text in both arms, 56 explicitly declare the same
target under the diagnostic rubric. The three differences are `route-012`:
specialist versus routine. Both are wrong because specialist need is missing.
This diagnostic comparison includes rejected outputs and is not primary agreement.

Decision-target agreement does not imply identical Recommendation objects. System
review differs on `route-001`, `002`, `003`, `005`, `006`, `007`, `009`, `017`, and
`018` in every repeat: labelled text requests review, JSON does not. All 59 returned
labelled responses request review. JSON also unnecessarily requests review on
`route-004` and `route-010`; shared policy adds review to its invalid selections on
`route-012` and `route-013`. Review recall therefore includes policy enforcement,
not just the model's own ability to detect review requirements.

Every repetition scores 14/20 for labelled text and 16/20 for JSON. JSON's primary
decisions/review remain stable in all 20 cases; labelled text has all-successful
agreement in 14, with failed repeats disqualifying the other six. Wording changes
within repetitions appear in labelled `route-001` and JSON `route-012`; a timeout
replaces one labelled `route-008` response. Stable primary decisions are not proof
of deterministic text generation or correctness.

## Confidence, Verification and the Rule Baseline

Experiment 2 selected outputs report confidence 1.0 despite errors. Its Brier/ECE
difference is caused by a timeout excluding a wrong selected output, not improved
calibration. Experiment 1's StructuredLLM dev confidence subset excludes some wrong
selections with missing confidence, making its low Brier score misleading if read
as overall reliability.

Experiment 3 supplies no numeric confidence in either arm. Brier score, ECE and
reported-confidence calibration are unavailable. The separate fixed-0.5 reference
is not model confidence and must not be substituted for it.

The verifier independently rejects all twelve schema-valid JSON policy violations.
It also rejects the three text reference-invalid candidates. It cannot verify the
fifteen text outputs without candidates. Schema and reference validity are useful
boundaries, not semantic evidence certification: `route-012` JSON makes unsupported
claims using valid IDs, and `route-011` overstates missing information.

The RuleBased baseline scores 27/30 overall with no execution failures or hard
policy violations. Its three errors are deliberate abstention on acceptable ties.
It provides an interpretable check against the operational rules, not a general
competitor on open-ended reasoning. Experiment 3 did not rerun or change it.

## Human Review and Remaining Uncertainty

The [human review packet](../runs/representation/dev-20261009/human-review.md)
prioritizes rejected, acceptable-target responses R003 and R022. The full JSON
packet groups all 43 unique case/condition/response combinations and preserves
their observation counts. It omits condition labels and reference answers;
representation remains visible from syntax. A separate review key maps items to
case, condition, repetition and automated diagnosis for use after first-pass review.

The accepted first-pass review is recorded separately in
[experiment-3-first-pass-v1.json](reviews/experiment-3-first-pass-v1.json), validated
by the `HumanReviewRecord` Pydantic contract in `evaluation/human_review.py`.
These are **user-accepted, assistant-assisted first-pass assessments**, not
independent blind expert validation. The review key was not opened or used to
revise them. Source-file and response hashes bind the record to the original packet
and primary report; no raw text is copied into this curated documentation artifact.

| Item | Decision target | Reasoning/evidence | Policy | Review requirement | Uncertainty |
| --- | --- | --- | --- | --- | --- |
| R003 | Justified | Partially supported | Compliant | Incorrect: review was not required, although optional review is not explicitly prohibited | Low for routing; moderate for the review label |
| R022 | Justified | Fully supported | Compliant | Correct: missing verified urgency requires abstention and abstention requires review | Low |

The review supports distinguishing a justified decision from a usable output:
R003 and R022 still fail the original parser and retain their original primary
scores. R003 is not confirmed fully correct across all dimensions; unnecessary
review is distinct from violating a routing constraint. R022's accepted semantic
assessment does not repair its empty `Unknowns:` header or establish general
reliability.

Only two deliberately selected distinct responses were reviewed, each appearing
identically in three repetitions. Six observations are not six independent
judgments. These annotations do not cover all rejected responses, the JSON arm,
or a representative sample; they must not be extrapolated into a recovered-score
estimate or full-experiment semantic correctness count. Other semantic judgments
remain unreviewed; the timeout has no response and cannot be adjudicated. The
original packet and audit retain their historical pending status; this separate
versioned record documents the later acceptance without rewriting provenance.

The evidence supports a narrow conclusion: this JSON representation pipeline yields
more usable correct outcomes on this dev benchmark, with no observed advantage in
jointly successful decision targets. It also exposes substantial policy errors.
Enforcement alone showed no measured benefit in Experiment 2. The original gap is
consistent with contract/instruction/parser effects, but its entire causal
decomposition remains unresolved. Parser rigidity, representation-induced generation
behavior and instruction effects are still bundled.

Related synthetic cases and repeats are not independent samples. Dev has already
been examined extensively. Backend state, prompt tokenization, missing responses
and successful-output filtering remain threats. No significance tests, tuning,
held-out extension, model comparison or further experiment was performed here.

## Execution and Artifacts

```powershell
$env:PYTHONPATH = 'src'
$env:DECISIONLAB_PROVIDER = 'ollama'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL = 'qwen2.5:14b'
$env:OLLAMA_MAX_OUTPUT_TOKENS = '2000'
$env:OLLAMA_TEMPERATURE = '0'
python -m decisionlab.cli representation --provider ollama --allow-live --split dev --repeats 3 --output runs/representation/dev-20261009
```

This is the completed command, not a request to rerun. Exit 1 reports recorded
execution failures; all 120 scheduled observations completed. Tests before:
604 passed, 2 skipped (33.94s). Tests after: 604 passed, 2 skipped (20.03s).
Source/prompt/benchmark hashes, response/envelope hashes, final journal equality
and predeclared execution order were verified. No application code changed.

Experiment 3 artifacts: `experiment.json`, `experiment.txt`, `comparison.json`,
`paired-analysis.json`, `protocol.json`, `observations.jsonl`, `audit.json`,
`assistant-review.json`, `human-review.md`, `human-review-packet.json`, and
`human-review-key.json`, all in `runs/representation/dev-20261009/`.
Earlier artifacts remain in `runs/ollama-first/` and `runs/matched-json/`.
The subsequent accepted annotations live in `docs/reviews/experiment-3-first-pass-v1.json`;
`runs/representation/dev-20261009/human-review-finalization-v1.json` records the
separate finalization audit. Neither is imported into scoring or strategy inputs.
