# Decision Intelligence Lab (DecisionLab)

An experimental framework for evaluating LLMs as structured decision makers.
Currently implemented: Pydantic domain contracts, a provider-independent strategy
protocol, deterministic routing baseline and shared policy, a 30-case benchmark,
an independent evaluator, direct and structured LLM strategies with isolated OpenAI
and local Ollama adapters, a standalone decision/evaluation CLI, verification,
bounded deterministic tools, reproducible experiments, and tests. There is no
agent or action execution. An optional offline Streamlit presentation layer is
available in the [DecisionLab portfolio demo](docs/demo.md).
Overview introduces the framework and research boundaries; the sidebar and overview
actions open the Decision Playground and Research Dashboard for offline decisions and saved research evidence;
missing local reports are labelled unavailable and never regenerated.

## Offline Quickstart

With Python 3.11+ and the declared Pydantic 2 dependency available, run from the
repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli decide --input examples/decision-routine.json --strategy rules
```

Rules are the default; this needs no API key, Ollama, provider settings or network.
The JSON contains `run.status="success"`, a recommendation selecting `routine`,
no policy violations and `verification.disposition="accept"`. Missing information
is demonstrated with `examples/decision-missing-urgency.json`: successful abstention
and required review, not execution failure. After installation, `decisionlab decide`
is equivalent. See the [offline walkthrough](docs/offline-quickstart.md) for the
input contract, representative output, exit codes, other existing strategies and
how to read saved experiments without rerunning them. Setup instructions follow.

## Development

Requires Python 3.11+ and Pydantic 2. Dependencies are declared in
`pyproject.toml`. To prepare a development environment when needed:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[dev]"
```

Run offline tests from the repository root:

```bash
python -m pytest -q
```

Configured optional checks (require the corresponding development tools):

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy
```

## Contracts

`src/decisionlab/models.py` defines problems, recommendations, policy outcomes,
and execution records. `src/decisionlab/evaluation/models.py` defines separate
reference expectations and evaluation records, without scoring behavior.

Models reject unknown fields, invalid selection states, duplicate entity IDs,
and non-finite numbers. Fact values and option attributes support JSON scalar
types. IDs are namespaced by entity type. Confidence is optional and represents
an uncalibrated estimate of benchmark correctness.

Standalone recommendations have no access to a problem. Call
`problem.validate_recommendation(recommendation)` to check option, evidence,
constraint, and unknown references. Call `case.validate_problem(problem)` to
validate reference expectations. These checks validate data relationships, not
eligibility rules or evidence support. Run records link by problem ID; they do
not implicitly perform these cross-document checks.

Models validate construction/deserialization. Treat validated records as
snapshots: after changes, deserialize and validate again rather than mutating
nested collections or relying on unvalidated `model_copy(update=...)`.

Constraints describe requirements; the routing baseline supports the hard
constraint ID `capacity`. The shared benchmark policy separately enforces routing,
capacity, required approval, and human review without changing the recommendation.
Failed runs contain an error and no
accepted recommendation; successful abstentions remain distinct from failures.
Run metadata carries schema version `1`; no persistence migration exists yet.

See [the implementation plan](docs/PLANS.md) and [contributor rules](AGENTS.md).
The MVP milestones, including standalone decisions and the reproducible offline
walkthrough, are implemented. Known routing, evaluation and research limitations
remain explicit in the plan and [research interpretation](docs/research-findings.md).

## Benchmark

The frozen v1 benchmark has 20 development and 10 held-out problems in `evals/cases/`.
Expected outcomes and manual rubrics live separately in `evals/expectations/`.
`evals/manifest.json` records versions, pair membership, and artifact checksums.
Dataset tests validate schemas, labels, split integrity, pairs, and hashes without
scoring strategies. Separate foundation tests exercise execution and scoring.
See [the benchmark specification](docs/benchmark.md).

The benchmark accepts either eligible queue on ties and includes review-required
selections. The baseline still abstains on ties; shared policy adds the required
selection review. The evaluator preserves and measures the original recommendation.

## Offline Evaluation

Run from the repository root in PowerShell using the existing environment:

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --benchmark evals --split all --output runs/baseline
python -m decisionlab.cli evaluate --benchmark evals --split dev --json
python -m decisionlab.cli evaluate --benchmark evals --split all --repeats 2 --output runs/repeated
```

After editable installation, `decisionlab evaluate` is equivalent. `--output` writes
deterministic `report.json` and `report.txt`; omit it to emit only stdout. `--json`
keeps stdout machine-readable. Exit codes: 0 completed evaluation (even with wrong
decisions), 1 recorded execution failures, 2 loading/configuration/output errors.

Tests normally use `python -m pytest -q`. In a restricted environment where the
system temp directory is unavailable, use
`python -m pytest -q --basetemp .pytest_cache/foundation-tests`.

The baseline achieves 27/30 correct outcomes. See [sample results](docs/baseline-results.md)
and [metric definitions and limitations](docs/evaluation.md).

## Structured LLM Evaluation

The provider-neutral strategy uses the same DecisionStrategy signature and evaluator.
The OpenAI adapter uses the standard library; no additional package is needed.
Set `OPENAI_API_KEY` and `OPENAI_MODEL` in your environment. See `.env.example` for
optional limits; `.env` files are not automatically loaded and secrets are excluded
from prompts and report metadata.

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --strategy structured-llm --allow-live --split dev --output runs/structured-llm
```

Live runs require the flag and incur API usage. Reports include model/prompt versions,
safe settings, response IDs, retries, and tokens. API errors, refusal, timeout, malformed
output, and invalid references produce explicit failures, never silent abstention.

Normal tests stay offline. To explicitly run two paid smoke tests with credentials
and a compatible model:

```bash
python -m pytest -q --run-live tests/test_openai_live.py --basetemp .pytest_cache/live-tests
```

OpenAI live API validation remains unverified; local Ollama benchmark results are
recorded in the [research interpretation](docs/research-findings.md). See
[structured strategy documentation](docs/structured-llm.md) for details and limits.

## Direct LLM Comparison

`DirectLLMStrategy` requests labelled normal text without an API-enforced schema.
It parses the complete response into the same Pydantic Recommendation; missing,
ambiguous, or invalid fields cause execution failures without repair. The routing
instructions, benchmark inputs, provider settings, shared policy, and evaluator
match the structured strategy. Output instructions, representation, parsing and
provider enforcement differ; the original comparison does not isolate their effects.

With the same environment settings and compatible model, run separate experiments:

```powershell
python -m decisionlab.cli evaluate --strategy direct-llm --allow-live --split dev --output runs/direct-llm
python -m decisionlab.cli evaluate --strategy structured-llm --allow-live --split dev --output runs/structured-llm
```

Each command makes 20 paid calls before repeats/retries. Compare outcome correctness,
execution failures, coverage, review, violations, and usage together; selected-output
accuracy alone hides failures. Local qwen2.5:14b results are recorded in the
[research interpretation](docs/research-findings.md). See
[the direct strategy and comparison protocol](docs/direct-llm.md).

## Strategy Comparison Experiment

The comparison runner uses one frozen benchmark snapshot for every strategy and
retains independent per-case evaluations. It writes side-by-side JSON/text reports
with correctness, failures, policy/review, evidence, confidence, agreement, usage,
and complete model/prompt metadata. Normal tests use mocked LLM providers.

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli compare --strategies rules --repeats 2 --output runs/comparison-rules
# With environment credentials/settings and explicit permission to make paid calls:
python -m decisionlab.cli compare --strategies rules direct-llm structured-llm --allow-live --split dev --repeats 2 --output runs/comparison-live
```

The second command performs 80 LLM requests before retries. The first local live
comparison and two matched dev ablations are recorded; these example commands are
optional new runs, not the offline quickstart. See [comparison interpretation and limits](docs/comparison.md) and
[the offline mocked example](docs/comparison-results.md). Missing confidence,
evidence-support scores, and costs remain unavailable rather than invented.

## Decision Verification

The provider-independent `RuleBasedVerifier` checks an already-produced recommendation
for schema/reference validity, routing-policy compliance, and citation completeness.
It returns accept/reject/human-review findings separately and never selects or repairs
a decision. Existing `evaluate`/`compare` commands include verifier diagnostics while
retaining original benchmark metrics. See [verification contracts and limitations](docs/verification.md).

## Tool-Assisted Routing

`ToolAssistedStrategy` runs a fixed information plan, two allowlisted in-memory
lookups, and one separate deterministic generator. JSON reports retain requests,
results, evidence IDs and original source-ID links. Existing policy, verifier and
evaluator remain outside generation.

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --strategy tool-assisted --output runs/tool-assisted
python -m decisionlab.cli compare --strategies rules tool-assisted --output runs/comparison-tools
```

This is offline and requires no API key. Its 30/30 deterministic benchmark score
reflects routing/tie rules, not LLM tool-use superiority. See [tool contracts,
provenance, bounds and limitations](docs/tool-assisted.md).

## Rule-Based Routing Examples

`DecisionStrategy` is a typed protocol with
`decide(problem: DecisionProblem) -> Recommendation`. It has no LLM/provider
dependency. `RuleBasedStrategy` implements this contract without network access.

Three synthetic inputs are in `examples/routing.jsonl`: routine, urgent, and urgent
specialist requests. Expected selections are `routine`, `expedited`, and
`specialist`, respectively. Run them from the root without installing the package
in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
@'
from pathlib import Path
from decisionlab.models import DecisionProblem
from decisionlab.strategies import RuleBasedStrategy

strategy = RuleBasedStrategy()
for line in Path("examples/routing.jsonl").read_text().splitlines():
    problem = DecisionProblem.model_validate_json(line)
    print(strategy.decide(problem).model_dump_json(indent=2))
'@ | python -
```

Rules and limitations are specified in [routing-baseline.md](docs/routing-baseline.md).

Confidence diagnostics are included in evaluation and comparison reports, with
separate dev/held-out Brier, ECE, reliability bins and coverage curves. Missing
confidence stays unavailable; a fixed .5 reference is reported separately.
See [the confidence experiment](docs/confidence.md) for the strategy audit,
offline commands, metric definitions and limits of the synthetic benchmark.

The [robustness experiment](docs/robustness.md) measures decision stability across
179 separately versioned wording/order/distractor/redundancy/formatting variants.
Run `python -m decisionlab.cli robustness --strategies rules tool-assisted` with
`PYTHONPATH=src`. Split-specific JSON/text diagnostics stay separate from accuracy
and calibration; live LLM runs require explicit opt-in.

The [architecture guide](docs/architecture.md) defines the provider-independent
`DecisionModel` contract and explains how strategies, policy, verification,
evaluation and experiments remain separate.

[Local Ollama setup](docs/ollama.md) supports `--provider ollama` for LLM evaluation,
comparison and robustness. Export `OLLAMA_MODEL=qwen2.5:14b` and a loopback
`OLLAMA_BASE_URL`; `decisionlab preflight --provider ollama` checks installation
without generating or downloading. No API key is required; generation still needs
`--allow-live`. The existing OpenAI path remains available.

[Experiment 3](docs/representation-experiment.md) compares labelled text with
unconstrained JSON using counterbalanced dev pairs, raw-response capture and
separate failure diagnosis. Its dev run and two-item user-accepted, assistant-assisted
first-pass review are recorded in [research-findings.md](docs/research-findings.md).
Diagnostics never repair failed outputs or alter primary scores.
