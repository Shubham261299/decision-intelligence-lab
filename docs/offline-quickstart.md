# Reproducible Offline Walkthrough

## Run One Decision

Use Python 3.11+ with the project's existing Pydantic 2 dependency installed; see
[README setup](../README.md#development). Run from the repository root. No provider
environment variables, API credentials, Ollama installation or network access are
needed for the default rules path. No new dependencies are introduced.

PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli decide --input examples/decision-routine.json --strategy rules
```

Bash:

```bash
PYTHONPATH=src python -m decisionlab.cli decide --input examples/decision-routine.json --strategy rules
```

After editable installation, `decisionlab decide --input examples/decision-routine.json`
is equivalent; omitting `--strategy` selects rules. All `decide` output is JSON;
there is no need for `--json`. Diagnostics go to stderr. Repeated identical offline
inputs produce identical JSON, without timestamps or invented confidence/usage.

The command returns the existing typed `Observation`: `run` contains execution,
original recommendation, metadata, policy and error; other fields contain validation,
verification and tool diagnostics. Representative **excerpt** (the full output also
includes IDs, assessments, evidence references, validation fields and metadata):

```json
{
  "run": {
    "status": "success",
    "recommendation": {
      "status": "selected",
      "selected_option_id": "routine",
      "confidence": null,
      "explanation": "Selected the only routine queue with available capacity",
      "requested_review": false
    },
    "policy_result": {
      "requires_human_review": false,
      "review_reasons": [],
      "detected_violations": [],
      "policy_version": "routing-benchmark-1"
    },
    "error": null
  },
  "verification": {
    "disposition": "accept",
    "evidence_support": "not_assessed"
  }
}
```

To inspect the complete result interactively in PowerShell:

```powershell
$decisionResult = python -m decisionlab.cli decide --input examples/decision-routine.json | ConvertFrom-Json
$decisionResult.run.recommendation
$decisionResult.run.policy_result
$decisionResult.verification
```

This recommends a route; it does not execute an operational action or calculate
benchmark correctness. A verifier `accept` does not certify real-world truth or
semantic evidence support.

## Input Contract and Abstention

[decision-routine.json](../examples/decision-routine.json) is one UTF-8 JSON object
matching `DecisionProblem`, not JSONL, a list or a benchmark label wrapper. Required
top-level fields are `id`, `description`, and a nonempty `options` list; the example
also supplies `context`, `facts` and `constraints`. Unknown fields and duplicate
entity IDs are rejected. UTF-8 files with a BOM are supported.

For this routing domain, `urgent` and `specialist_required` fact IDs carry Boolean
values, and each option has `route` (`routine`, `expedited`, `specialist`) and
nonnegative integer `capacity`. The shared policy requires classifications from
`verified-intake`; that source string expresses provenance within this synthetic
contract, not an external truth check. Capacity must be positive for selection.
The `capacity` hard constraint prohibits cross-route fallback. See
[routing rules](routing-baseline.md) and [shared policy](evaluation.md).

Missing information is valid problem data, distinct from invalid JSON/schema.
The second example sets verified urgency to null:

```powershell
python -m decisionlab.cli decide --input examples/decision-missing-urgency.json --strategy rules
```

It exits 0 with `run.status="success"`,
`run.recommendation.status="abstained"`, `selected_option_id=null`,
`run.error=null`, policy review required and
`verification.disposition="human_review"`. The unknown identifies urgency.
RuleBased also conservatively abstains on multiple eligible queues; it does not
implement the benchmark's acceptable-choice-on-ties rule.

## Interpret Failures and Findings Separately

| Field | Meaning |
| --- | --- |
| `run.status` | Execution success or failure, not correctness or verifier acceptance |
| `run.recommendation.status` | Selected or abstained, present only on successful execution |
| `run.error` | Sanitized execution failure code/message; failed output is never repaired into abstention |
| `candidate` | Diagnostic output, including schema-valid rejected reference failures; never an accepted recommendation on failed runs |
| `schema_valid`, `references_valid` | Boundary validation; unavailable checks are null |
| `run.policy_result` | Separate violations and required review; never replaces the recommendation |
| `verification` | Separate accept/reject/human-review disposition, machine-readable findings and evidence-reference checks |
| `verification_error` | Verifier execution failure; does not rewrite strategy execution or recommendation |

| Exit code | Meaning for `decide` |
| --- | --- |
| 0 | Successful execution with no policy/verifier rejection; may select, abstain or require human review |
| 1 | Strategy/output execution failure, policy violation, verifier rejection, or verifier error; JSON is still emitted |
| 2 | Input, arguments or configuration invalid/unreadable; no decision JSON is emitted |

A valid selection can still require review. A policy-rejected selection can have
`run.status="success"` and exit 1: generation/validation succeeded but policy or
verification rejected the result. Inspect these dimensions instead of treating
every nonzero exit as the same failure. Benchmark evaluation retains its existing,
separate exit-code and correctness definitions.

## Other Existing Strategies

The deterministic bounded-tool strategy also works offline:

```powershell
python -m decisionlab.cli decide --input examples/decision-routine.json --strategy tool-assisted
```

It includes explicit tool provenance; its generator is deterministic and non-LLM.
`direct-llm` and `structured-llm` are selectable using existing adapters, but require
`--allow-live` and environment configuration before any generation. They are not
part of this offline walkthrough. `--provider openai|ollama` only configures LLM
strategies; rules/tool-assisted ignore provider settings. See [Ollama](ollama.md)
or [structured LLM](structured-llm.md) for configuration. No live calls were made
to implement or verify this milestone; provider wiring tests are mocked.

## Understand the Existing Experiments Without Rerunning Them

Read the portable [research interpretation](research-findings.md) first. It includes
the first strategy comparison, matched schema-enforcement ablation, matched
representation experiment, and the limited accepted first-pass diagnostic review.
Accuracy, parsing, references, policy, review and confidence remain separate.

Local experiment files are intentionally ignored under `runs/`, so a fresh checkout
may not contain them. When available, inspect these saved reports without model
calls, provider configuration or benchmark re-evaluation:

```powershell
Get-Content runs/ollama-first/dev/comparison.txt
Get-Content runs/ollama-first/held-out/comparison.txt
Get-Content runs/matched-json/dev/comparison.txt
Get-Content runs/representation/dev-20261009/experiment.txt
```

Their neighboring JSON reports contain per-case records and metrics. Experiment 3
also preserves raw responses and a protocol/audit. Historical reports are evidence
from those recorded runs, not guaranteed results from rerunning current code.
The benchmark is small and synthetic; neither valid JSON nor stable decisions
establish production reliability. Confidence is optional and uncalibrated, evidence
support remains separate from reference validity, and the two reviewed responses
are not independent blind expert validation.

## Verify the Local Setup

```powershell
python -m pytest -q --basetemp .pytest_cache/offline-tests
```

Default tests block live HTTP; the two opt-in OpenAI tests stay skipped. Tests cover
selection, abstention, input/output failures, policy rejection, review, verifier
errors, mocked provider paths, and execution independent of benchmark files.
No live experiment or additional architecture is required to complete this MVP.
