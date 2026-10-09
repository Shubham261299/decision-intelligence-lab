# Experiment 3: Matched Output Representation

## Question and Scope

Does the labelled-text contract/parser pipeline explain the original DirectLLM
versus StructuredLLM gap? Compare `labelled-text` with `unconstrained-json` using
qwen2.5:14b locally. Neither arm sends Ollama `format` or uses JSON mode. This is
a representation-and-parser experiment, not a pure parser or reasoning ablation.
No live run was performed during implementation.

Both arms read one shared, versioned policy resource. Only representation
instructions and strict parser differ. Both request the same decision fields,
one reason per assessment, supplied evidence/constraint IDs, unknowns, optional
confidence, and review. Text uses the existing `direct-text-v1` grammar; JSON uses
StructuredRecommendation then Recommendation. Domain and reference validation,
policy, verifier, evaluator, input serialization, provider/model/settings, cases,
repetitions, and execution permissions remain shared. Existing experiments and
frozen benchmark v1 are unchanged.

## Predeclared Procedure

The specification is in [representation-protocol.json](representation-protocol.json).
Default to dev, three repetitions, 120 attempts. Traverse repetitions, then frozen
case order. For each pair, alternate which condition is first by case index plus
repetition parity. With 20 cases, each condition goes first in 30 of 60 pairs.
Do not tune prompts, repair output, selectively rerun failures, or use held-out.

Before calls, validate the benchmark and a single settings snapshot, reject an
occupied output directory, and write `protocol.json` with the exact schedule,
repetitions, hypothesis, variables, metrics, threats, procedure, model/provider,
settings, prompt/policy/source hashes, input hash and benchmark artifact hashes.
Use a fresh output directory; partial runs cannot be resumed or overwritten.

```powershell
$env:PYTHONPATH = 'src'
$env:DECISIONLAB_PROVIDER = 'ollama'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL = 'qwen2.5:14b'
$env:OLLAMA_TEMPERATURE = '0'
$env:OLLAMA_MAX_OUTPUT_TOKENS = '2000'
python -m decisionlab.cli representation --provider ollama --allow-live --split dev --repeats 3 --output runs/representation/dev
```

This is a future execution command, not a test command. CLI enforces the declared
local model and live opt-in; the strategy/runner also support mocked TextProviders
without Ollama. API credentials are unnecessary. No preflight/generation occurs
at import time. Timeout and other safe settings come from the existing environment
loader and are recorded, not silently adjusted.

## Capture and Primary Scores

`observations.jsonl` is flushed after every completed observation. It includes
exact decoded model text, SHA-256 of UTF-8 text, provider status, available raw HTTP
envelope/error body and hash, validation stages/issues, execution metadata, case,
repetition, sequence, primary outcome and verification. No request headers or
environment secrets are captured. Missing response text is null with a null hash;
an actual empty response is preserved as an empty string with its hash.

Successful, malformed, schema/domain-invalid, reference-invalid and partial/refused
outputs are retained. Returned provider error bodies are retained where available;
timeouts/disconnections may have no body. Capture cannot reconstruct bytes lost
before the provider/transport boundary, and does not invent unavailable usage.
Generic TextProviders preserve their returned text/status; HTTP envelope capture
is an isolated optional local-Ollama transport wrapper. Raw artifacts may contain
input-derived data: keep them local/ignored, never commit or print them as logs.

`experiment.json` embeds the protocol, unchanged evaluation/comparison metrics,
raw observations, diagnostics and paired analysis. `comparison.json` contains
ordinary primary reports; `paired-analysis.json` contains each matched pair and
aggregate agreement/wins. `experiment.txt` prints the comparison and diagnostic
summary without raw payloads. CLI `--json` prints only the comparison. Exit 1
means recorded execution failures; exit 2 means configuration/artifact errors.

Primary metrics retain existing definitions: all-attempt correctness, selection
and required-abstention correctness, failures, review precision/recall, policy
violations, and repeated agreement. Mechanical stage counts additionally separate
serialization/parser failures from domain/reference failures. Paired agreement is
exact selected option or abstention among jointly successful pairs; failures are
reported separately, never counted as agreement. Alternative acceptable tied
options can disagree without either being wrong. Scores never use diagnostic
target extraction. Verifier findings remain separate from correctness.

## Failure-Diagnosis Rubric v1

Two axes prevent syntax and meaning from being conflated. Mechanical stages do
not read expectations:

| Stage | Deterministic rule |
| --- | --- |
| `provider_api_failure` | Adapter exception or malformed/failed provider status; save available envelope |
| `empty_incomplete_refusal` | Empty returned text, explicit incomplete or refused provider status |
| `serialization_format_failure` | JSON decoding fails; text begins with Markdown fence or JSON instead of its required grammar |
| `parser_failure` | Existing labelled-text reader rejects labels/order/lexical values/blocks/trailing content |
| `schema_domain_validation` | Pydantic rejects transport/domain types, values, fields or cross-field invariants |
| `invalid_option_reference` | Original problem rejects option, fact, constraint or affected-option references |
| `ambiguous` | No reliable stage information; do not guess |

Only location/type codes are retained for Pydantic issues, without input values.
The old `invalid_schema` primary code remains unchanged; parser instrumentation
adds stage information without changing its accepted grammar or scores.

The separate decision-target axis inspects exact declarations only. Text requires
one anchored `Decision:` and `Option:` line. JSON requires a complete object with
unique keys and explicit status/option; a single whole-response JSON fence may be
inspected diagnostically. Broken JSON, duplicate declarations, contradictory
status/option pairs and arbitrary prose are ambiguous. This does not produce a
Recommendation. Partial/refused/API-error text is never treated as a final target.

After primary evaluation, compare only that target with evaluator-owned acceptable
options/required abstention. References never enter strategies or prompts:

- `explicit_incorrect_decision`: an explicit target mismatches the benchmark;
  correctness of its entire reasoning is not assessed.
- `explicit_apparently_correct_rejected`: a failed response declares an acceptable
  target but still violates its contract. Primary score remains a failure.
- `ambiguous`: no reliable final target. Human judgment is required.
- `not_failed`: no rejected-output diagnosis; successful wrong targets may still
  be identified separately as explicit incorrect decisions.

Failed-response counts per condition report apparent-correct targets, explicit
incorrect targets, ambiguous targets, mechanical stages and human-review needs.
`confirmed_substantively_correct` and `confirmed_full_semantically_incorrect`
remain null: option matching cannot establish valid evidence, review, explanation,
or compliance. All failed-response diagnoses flag human judgment as required.
Manual reviewers should first inspect anonymized raw responses with original
problems, without condition labels or expectations; compare declared targets
against references afterward. No automated or manual diagnosis rescues primary
scores. Human annotation import/adjudication is not implemented in this increment.

## Threats and Tests

Related synthetic cases and repeated observations are not independent samples.
Counterbalancing reduces order confounding but does not reset/control model cache,
machine load or backend state. Temperature zero does not ensure repeatability.
Representation instructions differ in length and tokenization; this cannot isolate
parser rigidity from generation difficulty by itself. Diagnostic extraction may
miss contradictory free-form explanations: full semantic judgments remain pending.
Correctness on jointly valid outputs has selection bias. Evidence support remains
unscored, confidence uncalibrated, and provider failures can lack payloads/timing.

Mocked tests cover both no-format HTTP paths, shared policy/input/configuration,
counterbalanced order, raw hashes/envelopes/error bodies, every mechanical failure
stage, ambiguous targets, strict rejection without rescue, isolation from labels,
unchanged policy/verifier/evaluation, permissions, overwrite guards, and CLI output.
Normal tests block unmocked HTTP. No new dependencies or agent framework is used.
