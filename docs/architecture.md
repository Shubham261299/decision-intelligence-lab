# DecisionLab Architecture

DecisionLab is a local experimentation framework, not an operational action system.
It compares recommendations while keeping generation, policy, verification and
benchmark evaluation separate. All default tests and experiments run offline.

## Decision Model and Strategies

`DecisionModel.decide(DecisionProblem) -> Recommendation` is the provider-independent
behavioral abstraction. RuleBased, DirectLLM, StructuredLLM and ToolAssisted implement
it structurally; no inheritance or adapter is required. `DecisionStrategy` retains
the existing public strategy name and extends this same protocol without additional
methods. Runners accept any DecisionModel, so adding a model does not require
rewriting evaluation. Runtime protocol membership checks the method's presence,
not its semantics or output validity; the runner performs actual validation.

A DecisionModel is not an OpenAI model identifier, provider, verifier or benchmark
judge. Its strategy identity/version belongs in RunMetadata; a provider model ID
is optional. LLM strategies use independent text/structured provider protocols and
capture optional telemetry. Deterministic models need no provider configuration.

OpenAI and [local Ollama](ollama.md) implement both protocols. Explicit provider
selection lives at the CLI/factory boundary, outside strategies. Ollama restricts
HTTP to loopback, disables proxy/redirect forwarding, and needs no credentials.
Optional response metadata/server latency pass through execution telemetry without
changing decision generation or evaluator logic.

## Execution and Evidence

```text
DecisionProblem -> DecisionModel -> schema/reference validation -> shared policy
                                                                   |
                                                              DecisionRun
                                                                   |
                                                      separate DecisionVerifier
                                                                   |
Private reference expectations --------------------------> evaluator -> report
```

The runner passes a deep copy of the problem and validates an output snapshot.
Malformed output, unresolved references and provider failures remain failed runs;
they never become abstentions. Policy records violations and review requirements
without replacing the recommendation. Verification produces a separate result,
including accept/reject/review and evidence traceability findings.

Facts, option/constraint IDs, source citations and bounded tool request/result
provenance remain in original observations. Existing citation IDs demonstrate
traceability, not semantic support. The deterministic verifier checks supplied
routing information; it is not a truth oracle and shares policy implementation.

## Experiment Boundaries

- The hash-verified benchmark stores inputs and private evaluator expectations
  separately. Strategies receive only DecisionProblem, never labels or rubrics.
- Matched comparison uses one snapshot and preserves all failures and metadata.
  Accuracy, abstention, review, policy, evidence and usage remain separate metrics.
- Confidence diagnostics evaluate selected outputs against benchmark outcomes.
  Missing estimates remain unavailable; current reported confidence is uncalibrated.
- Robustness uses separately versioned equivalent-input fixtures and compares
  original paired outputs without labels, normalization or repair.
- Dev and held_out are reported independently. No threshold or strategy tuning
  uses held-out results. Public synthetic holdout is not a blind external test.

ToolAssisted has an allowlisted, bounded fixed plan and a separate deterministic
generator. It is not an autonomous agent and acquires no external information.
Neither policy, verifier nor evaluator calls an LLM. A single provider request is
generation, not an agent loop. No actions, UI, database or orchestration framework
are needed for the current experiments.

## Reproducibility and Claims

Record dataset/artifact hashes, strategy/provider/model/prompt versions, safe
configuration, original observations and available usage. JSON serialization is
deterministic for identical observations; live generation remains stochastic.
Offline/mocked tests prove arithmetic, isolation and failure handling, not real LLM
performance. Live runs require explicit opt-in and environment configuration.

The current routing benchmark supports controlled measurements, not general
reliability, calibrated confidence, causal robustness or LLM superiority claims.
See [evaluation](evaluation.md), [comparison](comparison.md),
[confidence](confidence.md), [verification](verification.md), and
[robustness](robustness.md) for exact contracts, denominators and limitations.
