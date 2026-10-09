# Decision Verification Layer

## Contract and Boundaries

`DecisionVerifier.verify(problem: DecisionProblem, recommendation:
RecommendationInput) -> VerificationResult` is a provider-independent protocol.
RecommendationInput accepts the existing Pydantic Recommendation or a raw mapping
for schema-validation callers. A mutated/model-constructed instance is revalidated
through a fresh snapshot. Inputs must include a valid DecisionProblem; invalid
problems raise validation errors rather than pretending to verify a decision.

`RuleBasedVerifier` version 1 checks the routing domain deterministically. It never
generates options, selects an alternative, calls a provider, repairs text, or reads
benchmark labels. The original recommendation is preserved, with findings returned
in a separate typed Pydantic result. Results record verifier/result/policy versions,
problem ID, disposition, validity dimensions, coded findings, and applicable option,
fact, constraint references and paths.

| Component | Responsibility |
| --- | --- |
| Decision strategy | Produce a recommendation from supplied problem data |
| Policy engine | Apply executable eligibility/review rules to a recommendation |
| Verifier | Inspect an existing recommendation's schema, references, traceability and policy compliance; issue a disposition |
| Evaluator | Score original outcomes against separate benchmark expectations |
| Second LLM | Additional probabilistic generation/judgment, absent from this implementation |

The verifier reuses domain validation and shared `selection_violations`/
`enforce_policy`; it does not copy benchmark truth into an inference-time checker.
Its shared policy dependency also means policy implementation bugs can affect both
layers. This is not an independently proven correctness oracle.

## Validation States and Dispositions

`decision_validity` is `valid`, `invalid`, or `not_assessed` and describes local
contract/policy validity, not benchmark correctness. Separate dimensions record
structural validity, selected-option existence (null for abstention), all-reference
validity, evidence-reference validity, citation completeness, and review requirements.
Policy violations and findings can overlap; they are not mutually exclusive scores.

- **Accept:** structurally valid selected recommendation, known references, sufficient
  routing traceability, no policy violation or review requirement.
- **Reject:** invalid schema/references, unknown selected option, a selected option
  assessed as ineligible, or any deterministic selection-policy violation. Rejection
  takes precedence over review; `requires_human_review` can still be true.
- **Human review:** valid abstention, explicit supervisor/strategy review, material
  unknown, or incomplete traceability without a reject condition.

Selected recommendations must have a selected-option assessment citing `urgent`
and `specialist_required`, plus `approval_granted` when approval is hard-required.
The assessment must cite supplied hard-constraint IDs. Missing assessment, missing
required citations, or unknown self-reported eligibility is `insufficient` evidence
traceability and requires review. Abstentions have `not_applicable` selection
traceability and always require review; no nonexistent facts are fabricated.

`sufficient` means these citations are complete, not that their prose is true.
`evidence_support` remains `not_assessed`. The verifier does not parse explanations
or reasons for factual entailment; a correct citation can accompany an unsupported
assertion. Required-but-missing IDs in incomplete-traceability findings describe
missing requirements, not citations the model actually supplied.

Machine-readable codes include `invalid_schema`, `unknown_selected_option`,
`unknown_evidence_reference`, `unknown_constraint_reference`,
`selection_assessment_conflict`, and `incomplete_selection_traceability`.
Policy codes retain their existing names, such as `wrong_route`, `capacity_exhausted`,
`approval_not_granted`, `unresolved_fact:urgent`, or
`unsupported_hard_constraint:compliance`. Selection without established compliance
is rejected, including unresolved inputs and unsupported hard constraints. Abstention
with those same inputs goes to review. Soft preferences do not establish a ranking.

## Evaluation and Comparison Integration

`run_benchmark` performs verification **after** existing strategy execution/policy
and before reference evaluation. The verifier receives isolated copies of the problem
and recommendation/candidate, never an EvaluationCase, rubric, or expected option.
Reference-invalid schema-valid candidates can be verified diagnostically even though
their strategy run remains failed. API/format failures without a parsed candidate
have null verification; they are unavailable, never accepted or turned into abstention.

Observation adds `verification` and `verification_error`; EvaluationReport adds a
separate `verification_summary` with verified/unavailable/error counts and disposition,
policy-violation, and insufficient-evidence counts. Existing correctness, coverage,
review, confidence, evidence, usage and agreement metrics still score the original
execution, recommendation, and policy outcome. No verifier finding changes them.

A custom verifier can be injected into `run_benchmark`. Its returned result is
revalidated and problem identity checked. Exceptions or invalid results produce a
sanitized `verifier_error`, without changing strategy status or original metrics.
The comparison runner uses the default verifier through this same benchmark path.
Low-level `run_decision` remains an execution-only API; direct callers can invoke
the verifier separately when needed.

Existing `evaluate` and `compare` CLI commands automatically include findings in
JSON and diagnostic verifier summaries in text; no new command or API settings
are required. Exit codes retain execution semantics, so a verifier rejection on a
successful but wrong decision does not become an execution failure. Consumers must
inspect both execution outcomes and verifier diagnostics.

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --strategy rules --json --output runs/verified-baseline
python -m decisionlab.cli compare --strategies rules --repeats 2 --output runs/verified-comparison
```

The baseline still scores 27/30 with zero execution failures. Diagnostic verification
finds 14 accepts and 16 human-review dispositions, zero rejects. Its three tie
abstentions remain benchmark misses; the verifier reviews them rather than selecting
a queue. This demonstrates why verification and evaluation measure different things.
Additive optional report fields keep schema version 1 readable; they do not change
Recommendation or DecisionRun schemas. New report bytes differ from old artifacts.

## Tests and Limits

[ToolAssistedStrategy](tool-assisted.md) now uses the same external verifier against
the original problem. Recommendation citations remain original Fact/Constraint IDs;
tool evidence IDs and source links live in observation tool records. The verifier
does not execute tools or treat their hashes as proof of semantic support.

Run `python -m pytest -q --basetemp .pytest_cache/foundation-tests`. Tests cover
schema/mutation failures, every reference namespace, policy violations, citation
completeness, supervisor/strategy review, abstention, result invariants, preservation,
custom-verifier isolation/errors, CLI reporting, and identical pre-verifier metrics.
All tests are offline; no dependencies or provider calls are introduced.

This verifier is specific to the current routing policy, not an arbitrary natural
language constraint interpreter. No revision loop, automatic evidence-support judge,
confidence calibration, agent, or new strategy is added. Acceptance does not prove
benchmark correctness, explanation truth, completeness of unknowns, or production
safety. Future verifier quality must be evaluated separately before using findings
to change an operational decision or introducing bounded revision.
