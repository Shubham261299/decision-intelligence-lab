# DecisionLab MVP Implementation Plan

## Implementation Status

Experiment 3 implementation, dev execution and the accepted two-item first-pass
diagnostic review are complete:
[matched labelled-text versus unconstrained JSON](representation-experiment.md).
The counterbalanced run completed 120 attempts: labelled text scored 42/60 and
unconstrained JSON 48/60. Both arms used text-only generation with unchanged primary
validation/evaluation/verification. R003 and R022 have versioned user-accepted,
assistant-assisted first-pass annotations; these are not independent blind expert
validation and do not repair failed outputs or change scores. The remaining raw
responses have not received comprehensive human semantic review. See
[research-findings.md](research-findings.md) and the
[accepted review record](reviews/experiment-3-first-pass-v1.json).

**Milestone 13 is complete: standalone decisions and the offline walkthrough.**
`decide --input ... --strategy ...` validates one JSON problem and uses existing
execution, policy and verification contracts. Rules are offline by default;
tool-assisted remains deterministic and LLM generation requires explicit opt-in.
Selection, abstention, failed execution and policy/verifier findings remain separate.
Two documented synthetic inputs and deterministic CLI tests verify these paths.
The [offline walkthrough](offline-quickstart.md) explains setup, representative
output, exit codes and reading existing saved reports without new model calls.

No MVP implementation milestone remains. Optional research extensions below are
not verified completion claims: OpenAI live compatibility, live robustness,
held-out ablations, broader independent semantic review and calibrated confidence
remain unverified/unimplemented as applicable. The framework is limited to synthetic
routing, conservative rule tie abstention and non-LLM bounded tools; verifier
acceptance is not semantic truth certification or production reliability.

Actual implementation order: domain contracts and tests first (Milestone 2), then
the strategy interface, RuleBasedStrategy, and three examples (part of Milestone 3),
then benchmark specification (Milestone 1). The benchmark phase is now complete:
30 versioned synthetic cases, separate expectations and rubrics, a frozen 20/10
development/held-out split, paired-case metadata, hashes, and dataset integrity tests.
See `benchmark.md` for the normative benchmark policy and `routing-baseline.md`
for current baseline behavior. Their tie and review semantics differ explicitly.

The next foundation increment implemented shared deterministic policy, an isolated
execution boundary, a checksum-verified benchmark loader, an independent evaluator,
explicit metrics, deterministic reports, and an offline baseline evaluation CLI.
Milestones 3 and 4 are complete with an intentional baseline limitation: ties still
produce abstention. Shared policy adds review for selected outcomes without changing
the recommendation. The recorded baseline has 27/30 correct outcomes and no execution
failures; see `baseline-results.md`. The strategy is never the ground-truth oracle.

StructuredLLMStrategy is now implemented behind a provider-neutral protocol, with one
OpenAI Responses HTTP adapter, environment settings, a versioned prompt, strict
Pydantic transport/domain validation, safe failure codes, and execution usage metadata.
CLI live evaluation requires explicit opt-in. Mocked unit/pipeline tests pass; two
opt-in live smoke tests exist but have not been executed. A real live benchmark
experiment has been recorded using local Ollama; OpenAI live compatibility remains
unverified. No agent, action execution, or UI exists;
standalone decision commands now use the same validated execution path. See
`structured-llm.md` and `offline-quickstart.md`.

The next increment adds DirectLLMStrategy before CLI expansion or inference-time
verification: a provider-neutral text contract, shared OpenAI transport/settings,
versioned policy-equivalent prompt, strict text-to-Pydantic parser, and CLI support.
Mocked parser/adapter/pipeline tests cover failures without repair. Direct and
structured live benchmark comparisons are recorded in `runs/ollama-first/`;
see `direct-llm.md` and `research-findings.md`.

The strategy comparison increment is now implemented: one benchmark snapshot,
shared runner/evaluator, deterministic side-by-side JSON/text reports, explicit
offline/mocked/live modes, and CLI `compare` with live opt-in. All strategy behavior
and benchmark artifacts remain unchanged. Mocked pipeline results are documented
in `comparison-results.md`; the first live comparison and two matched dev ablations
are now recorded without a general superiority claim.
See `comparison.md` for denominators, metadata, interpretation and limitations.

The next increment implements post-decision verification before any revision loop:
a provider-independent protocol, typed separate findings, deterministic routing
checks, and additive evaluation/comparison diagnostics. Original strategy results,
benchmark labels and existing metrics remain unchanged. See `verification.md`.

The tool-assisted increment now adds typed local tools, explicit registry/allowlist,
per-decision budgets and duplicate prevention, a fixed information plan, explicit
evidence provenance, and a separate deterministic generator. Evaluation/comparison
CLI support remains offline for this strategy. No provider tool loop or new data
source is implemented; see `tool-assisted.md` for controlled-information limitations.

Cross-document references are checked explicitly through
`DecisionProblem.validate_recommendation` and `EvaluationCase.validate_problem`.
The runner now invokes these checks before accepting a result, using an isolated
input copy and a revalidated output snapshot. Rejected schema-valid candidates are
diagnostics, never accepted recommendations or valid abstentions.

Confidence/calibration diagnostics are now implemented after bounded tools:
provider-independent confidence semantics, explicit missingness, Brier, fixed
reliability bins/ECE, coverage curves, and a separately labeled constant .5
reference. Dev and held_out results remain separate with no fitting or tuning.
Strategy decisions and benchmark artifacts are unchanged. See `confidence.md`;
live diagnostics are recorded where confidence is available; fitting calibrated
estimates remains future work. Experiment 3 confidence is unavailable.

Decision robustness is now implemented as a separate versioned, source-hash-bound
fixture dataset and label-free stability runner. Six fixed transformations produce
179 equivalent-input variants; no-op fact reversal is excluded. Dev/held_out,
confidence changes, verifier changes and execution/format failures remain distinct.
See `robustness.md`; ordinary accuracy/calibration and all strategies are unchanged.

The provider-independent `DecisionModel` abstraction is now explicit. Existing
DecisionStrategy extends its unchanged recommendation contract; runners accept
any structural implementation without changing evaluation. `architecture.md`
documents the complete component boundaries and experimental limitations.

The first experiment provider is now local Ollama (`qwen2.5:14b`), behind the
existing generation protocols. OpenAI remains supported. Local-only configuration,
provider selection, read-only preflight, mocked transport tests and one minimal
schema smoke check are complete. The live comparison and both matched dev
experiments are recorded; no held-out ablation or live robustness run has been made.
See `ollama.md` for exact commands, metadata, smoke results and schema limitations.

## Project Objective

Build a small experimental framework for measuring LLMs as decision makers. Compare rule-based, direct-text LLM, and structured LLM strategies using identical problems, explicit constraints, evidence references, and independent evaluation.

Research question: **How can we make LLM-based decision making more measurable, structured, explainable, and reliable?**

The MVP recommends decisions; it does not execute operational actions. Use one synthetic operational-routing domain, predefined options, one LLM provider, bounded local lookup tools, and local JSON artifacts. Exclude a web UI, database, autonomous agents, external action tools, and option generation.

## Architecture

```text
DecisionProblem -> Strategy -> Output validation -> Decision policy -> DecisionRun
                                                                         |
Evaluation expectations --------------------------------------> Evaluator -> Report
```

- Domain models are independent of provider SDKs.
- Strategies receive problem data only; reference answers remain private to the evaluator.
- Validation checks schemas, identifiers, and structural invariants.
- A deterministic policy enforces eligibility and human-review rules outside the model.
- Evaluation measures correctness, evidence, and operational behavior separately.
- A provider adapter handles API calls, timeouts, bounded retries, and usage metadata.

Generation, decision selection, policy enforcement, and evaluation are separate responsibilities. A single model call is not an agent. Schema compliance does not imply a correct decision.

## Incremental Milestones

| Milestone | Deliverable | Completion criteria |
| --- | --- | --- |
| 1. Specify the benchmark — complete after initial coding | Routing rules, case format, rubric, review policy, 30 fixtures | Separate labels and rubrics, hashes, frozen splits, pairs, and dataset tests are present |
| 2. Establish contracts — complete | Minimal Python packaging, Pydantic models, explicit reference validation | Contract tests pass; runner invokes cross-document checks |
| 3. Finish the baseline and shared policy — complete | Existing strategy plus shared validation/review policy | Preserve conservative tie abstention as a measured baseline limitation; enforce review-on-selection and report hard violations without repairing recommendations |
| 4. Implement evaluation — complete | Independent offline evaluator, loader, runner, metrics, JSON/text reports | Baseline recorded; arithmetic, failures, isolation, reproducibility, loader corruption and CLI tests pass |
| 5. Add structured LLM — complete; local live results recorded | Provider-neutral strategy, OpenAI/Ollama adapters, versioned prompt, environment settings and telemetry | Mocked errors/pipeline pass; local qwen2.5:14b benchmark measured; OpenAI live compatibility unverified |
| 6. Add direct LLM baseline — complete; local live results recorded | Plain-text strategy, strict parser, shared provider configuration, versioned prompt, evaluation CLI | Parsing/reference failures preserved and compared with structured output; no repairs |
| 7. Strategy comparison experiment — complete | Matched runner, side-by-side reports, metadata, offline mocked tests and comparison CLI | First local dev/held-out comparison and two dev ablations recorded; frozen cases/metrics preserved; two-item accepted diagnostic review recorded separately |
| 8. Decision verifier — implemented | Typed contract, deterministic rules verifier, policy/reference/traceability checks, separate report diagnostics | Recommendations and baseline metrics preserved; no labels reach verifier; failure, review and insufficiency tests pass |
| 9. Tool-assisted strategy — implemented | Typed allowlisted fact/policy lookups, bounded fixed plan, evidence provenance and separate generation | Deterministic/error/provenance tests pass; existing scores unchanged; verification uses original problem; no agent or extra information source |
| 10. Confidence and calibration experiment — implemented | Typed confidence interpretation, distribution, Brier, reliability/ECE, coverage, fixed reference and split reports | Explicit missingness; no held-out fitting; unchanged scores/decisions; offline arithmetic, isolation and reproducibility tests |
| 11. Decision robustness experiment — implemented | Versioned equivalent-input fixtures, typed pairs, stability metrics, offline/mocked tests and CLI | Original benchmark untouched; transformations preserve routing semantics; split reports and unchanged outputs; live robustness pending |
| 12. Decision Model abstraction — implemented | Shared provider-independent protocol and architecture guide | Existing strategy API/behavior preserved; new structural implementations use unchanged evaluator and execution validation |
| 13. Finish the MVP walkthrough — complete | Standalone decision command, two example inputs, offline quickstart, saved live comparison and limitations | Offline selection/abstention demonstrated; input/output, policy, verification and mocked-provider tests pass; original benchmark and historical results preserved; no new live calls |

Implement one milestone at a time. Establish evaluation cases before tuning model
behavior. This foundation uses the existing environment and adds no dependencies.

## Repository Structure

```text
AGENTS.md
README.md
pyproject.toml
.env.example
src/decisionlab/
    models.py
    validation.py
    policy.py
    runner.py
    cli.py
    comparison.py
    confidence.py
    robustness.py
    verification/
        base.py
        models.py
        rules.py
    settings.py
    strategies/
        base.py
        rules.py
        structured_llm.py
        direct_llm.py
        direct_parser.py
        tool_assisted.py
        tool_generation.py
    tools/
        base.py
        models.py
        routing.py
        executor.py
    prompts/
        routing_v1.txt
        direct_routing_v1.txt
    providers/
        base.py
        openai.py
    evaluation/
        benchmark.py
        evaluator.py
        models.py
        metrics.py
        report.py
tests/
    test_models.py
    test_validation.py
    test_policy.py
    test_rules.py
    test_evaluation.py
    test_cli.py
evals/
    manifest.json
    cases/dev.jsonl
    cases/held_out.jsonl
    expectations/dev.jsonl
    expectations/held_out.jsonl
    expectations/rubrics.json
docs/
    PLANS.md
    benchmark.md
    evaluation.md
    baseline-results.md
    architecture.md
    evaluation.md
    decisions/
```

Create modules only when needed. Keep generated run artifacts in an ignored local directory such as `runs/`. Store versioned prompts with package resources when introducing the LLM strategy.

## Domain Models

Use typed Pydantic boundary models; reject unexpected fields and validate cross-field relationships.

| Model | Core fields |
| --- | --- |
| `DecisionProblem` | ID, description, context, facts, constraints, available options |
| `Fact` | ID, statement, optional typed value, source |
| `Constraint` | ID, description, hard/soft classification |
| `Option` | ID, description, domain attributes |
| `Unknown` | Description, affected option IDs, whether material to selection |
| `OptionAssessment` | Option ID, eligibility assessment, reasons, evidence IDs, constraint IDs |
| `Recommendation` | Selected/abstained status, optional option ID, assessments, unknowns, optional confidence, concise explanation, requested review |
| `PolicyResult` | Required review, review reasons, detected violations, policy version |
| `DecisionRun` | Recommendation, policy result, success/failure status, metadata, timing, usage, error details |
| `EvaluationCase` | Problem ID, acceptable option IDs, expected abstention/review, category tags, evidence rubric |
| `EvaluationResult` | Case/run IDs, individual metric values, diagnostic failures |

Required invariants:

- IDs are unique within a problem; selected options and evidence references must exist.
- A selected recommendation has an option ID; an abstention has none.
- Confidence, when present, is between zero and one.
- Invalid output is an execution failure, not an abstention.
- Model-requested review and policy-required review are recorded separately.
- Human-readable constraints accompany executable domain rules; free-form text alone cannot enforce eligibility.

Define confidence as an estimated probability that the selected option satisfies the benchmark criteria. Treat model-reported confidence as uncalibrated. Do not invent confidence for strategies that lack a meaningful estimate.

## Decision Engine

1. Validate the input problem.
2. Invoke the chosen strategy with the same supplied facts and options.
3. Parse and validate its recommendation, including evidence references.
4. Apply deterministic eligibility and review policy.
5. Record the recommendation and policy outcome without silently replacing a wrong selection.
6. Persist a run artifact; optionally evaluate against separate expectations.

For valid recommendations, require review for hard-constraint violations, material missing information, abstention, or an explicit strategy review request. Execution failures cannot produce an accepted decision. Any future confidence-based threshold must be explicit and evaluated; confidence alone never overrides hard rules.

Require concise evidence-linked reasons, not private chain-of-thought. Record detected unknowns without claiming the engine can identify every missing fact.

## Strategy Interface

Define a small typed Python protocol: `decide(problem: DecisionProblem) -> Recommendation`. Strategy identity/version belongs in runner metadata. Keep the MVP synchronous and single-process.

- `RuleBasedStrategy`: implemented with conservative tie abstention, intentionally retained as a baseline limitation. Benchmark v1 accepts either eligible option on ties. Shared policy flags review and violations separately; it never repairs this selection behavior.
- `StructuredLLMStrategy`: one schema-constrained request, followed by local Pydantic and domain validation.
- `DirectLLMStrategy`: one labelled plain-text request, strict parsing into Recommendation, and the same domain/reference validation; no repairs or schema enforcement at the provider.
- `ToolAssistedStrategy`: bounded fixed-plan local lookups, source-linked evidence, and one separate deterministic generator; original policy/verifier/evaluation remain external.

The runner normalizes typed execution errors into failed runs. Provider usage metadata is captured through the adapter. Strategies share the same output contract and cannot read evaluation expectations. Do not build a plugin registry or general agent framework yet.

## Evaluation Framework

Run each strategy on the same cases and preserve per-case results. Version the dataset, prompt, schema, policy, strategy, model identity, and generation settings; record retry counts, latency, token usage, and cost when available.

Report separately:

- Schema/reference validity and execution failure rates.
- Acceptable selection rate, with explicit denominators; failures count as unsuccessful cases.
- Selection accuracy among selected outputs alongside decision coverage.
- Hard-constraint violations, including violations flagged by policy.
- Human-review precision/recall and abstention behavior.
- Evidence-reference validity and human-scored evidence support.
- Repeated-run agreement, latency, usage, and available cost estimates.

Do not collapse everything into one score. An existing citation does not prove support; use a small manual rubric for evidence quality. Report confidence reliability bins and Brier score only for selected outcomes with confidence, including the sample count. The small dataset supports exploratory findings, not broad reliability claims.

Default evaluations are offline. Live-provider runs require an explicit flag, credentials, and documented expected cost. Keep benchmark evaluation independent of any future inference-time verifier.

## Test Strategy

Use `python -m pytest -q` for the implemented offline suite, including benchmark
integrity tests. Ruff and mypy are configured optional development checks in README;
they were unavailable during initial implementation.

- Unit tests: schema invariants, eligibility rules, policy triggers, baseline choices, and metric arithmetic.
- Integration tests: fixture input through strategy, validation, policy, artifact generation, and report.
- Provider tests: mocked valid output, refusals, malformed responses, timeouts, validation failures, and bounded retries.
- CLI tests: valid JSON output, missing settings, invalid input, live-run opt-in, and exit statuses.
- Regression tests: preserve meaningful failures discovered during evaluation.

Default tests must be deterministic and make no network calls. Prefer coverage of decision-critical behavior over an arbitrary percentage target.

## CLI

Implemented decision and evaluation commands (run after package installation, or use
`python -m decisionlab.cli` with `PYTHONPATH=src`):

```bash
decisionlab decide --input examples/decision-routine.json --strategy rules

decisionlab decide --input examples/decision-missing-urgency.json --strategy rules

decisionlab evaluate --benchmark evals --split all --output runs/baseline
decisionlab evaluate --benchmark evals --split dev --strategy structured-llm --allow-live --output runs/structured-llm
decisionlab evaluate --benchmark evals --split dev --strategy direct-llm --allow-live --output runs/direct-llm
decisionlab compare --benchmark evals --strategies rules --repeats 2 --output runs/comparison-rules
decisionlab compare --benchmark evals --split dev --strategies rules direct-llm structured-llm --allow-live --output runs/comparison-live
```

The following original commands remain planned and unavailable:

```bash
decisionlab evaluate --cases evals/cases/dev.jsonl --expectations evals/expectations/dev.jsonl --strategy rules --output runs/baseline
decisionlab evaluate --cases evals/cases/dev.jsonl --expectations evals/expectations/dev.jsonl --strategy structured-llm --allow-live --output runs/llm
```

Emit machine-readable decision JSON to stdout and diagnostic logs to stderr. A valid abstention is a successful execution; invalid input/configuration and execution failures return nonzero statuses. Evaluation reports must disclose partial failures and return a nonzero status when runs fail. Flag incompatible dataset/policy versions during comparison.

Standalone `decide` also exits 1 on policy/verifier rejection or verifier error,
while preserving the successful execution/recommendation where applicable. Human
review alone exits 0. Its output is the existing Observation, not benchmark scores;
it never loads benchmark expectations. See `offline-quickstart.md` for details.

## Sample Datasets

Implemented `evals/manifest.json`, 20 development and 10 held-out routing inputs,
separate EvaluationCase expectations, and per-case evidence/review rubrics.
Options are routine, expedited, and specialist queues; human review is a separate
flag. Categories include clear choices, hard conflicts, multiple acceptable options,
missing information, abstention, review, distractors, and conflicting evidence.
Four relevant-change pairs and two wording-invariance pairs remain within splits.

The benchmark has 19 selection cases, 11 required abstentions, and 13 review-required
cases (including 2 valid selections). Preserve v1 artifacts and hashes before prompt
tuning. Held-out labels are public; this is a development holdout, not a blind external
benchmark. See `benchmark.md` for policy, rubric, reproducibility, and limitations.

## Documentation

- README: objective, setup, offline quickstart, CLI commands, live-run configuration, and reproducible examples.
- Architecture guide: component boundaries and the distinction between generation, decisions, policy, evaluation, tools, and agents.
- Evaluation guide: dataset construction, rubrics, metric denominators, results, failure examples, and limitations.
- Decision records: significant choices such as benchmark domain and confidence semantics.

Centralize validated environment settings, provide placeholder-only `.env.example`, and never log credentials or sensitive prompts. Keep commits focused and document checks and evaluation results in pull requests.

## Future Extensions

The first local qwen2.5:14b comparison is recorded in `runs/ollama-first/`.
The matched enforcement experiment is completed on dev:
[matched structured-output ablation](matched-json-ablation.md). It holds the JSON
prompt and validation pipeline fixed while varying Ollama schema enforcement;
frozen cases and existing strategy behavior remain unchanged. Experiment 3 is also
complete on dev, with the accepted R003/R022 first-pass review finalized separately.
Broader semantic review, held-out ablations and future experiments are optional
research extensions, not prerequisites for the now-completed standalone CLI milestone.

Add extensions only after the baseline comparison is reproducible:

1. Add compatible saved-report comparison or broader independent semantic review if a new research question requires it; the first rule/direct/structured live comparison is already recorded.
2. Evaluate verifier findings and optionally add bounded revision later; independent benchmark evaluation remains the judge.
3. Provider-backed tool-assisted generation or external evidence acquisition, with explicit capability and information-access comparisons; local deterministic tools are implemented.
4. Candidate option generation with separate validation and evaluation.
5. Larger benchmarks, calibrated confidence, multiple providers, and policy sensitivity analysis.
6. An agent loop only when repeated observation/action steps serve a defined experiment.

A UI, database, deployment, and general orchestration remain outside the MVP. Completion means a reproducible baseline-versus-LLM experiment with visible failures, not guaranteed LLM superiority.
