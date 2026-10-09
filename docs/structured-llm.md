# Structured LLM Strategy

## Architecture

`StructuredLLMStrategy.decide(problem: DecisionProblem) -> Recommendation` implements
the unchanged DecisionStrategy protocol. It depends on the provider-neutral
`StructuredProvider` contract, not OpenAI code or evaluator expectations.

The versioned `prompts/routing_v1.txt` supplies normative policy, never case answers.
The only per-case input is serialized DecisionProblem. Expected outcomes, rubrics,
category tags, and pair metadata are not passed to providers. Problem prose is
treated as untrusted data, but shared policy still gates outputs; prompt instructions
alone cannot guarantee protection against prompt injection.

The adapter calls the OpenAI Responses API with a strict JSON schema generated from
Pydantic. Transport models require every field, including nullable confidence and
selected option, and forbid additional properties. Local Recommendation validation
then applies domain constraints and verifies option/evidence/constraint references.
The transport model avoids optional defaults and domain-only schema features at the
API boundary. Tests check transport/domain field alignment.

This follows the [official OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
The adapter uses Python's standard HTTP library; no OpenAI SDK or additional package
is required. Provider-specific HTTP payloads and parsing live in `providers/openai.py`.

`DirectLLMStrategy` shares the adapter, environment configuration, routing policy
instructions, domain contract, and evaluator. It requests labelled normal text with
no API-enforced schema and uses a strict local parser. See [the comparison protocol](direct-llm.md)
for the experimental distinction and formatting-related limitations.
The [matched comparison runner](comparison.md) now executes all three strategies
with a common benchmark snapshot and one environment-settings snapshot for live LLMs.

## Environment Configuration

`OpenAIProvider.from_environment()` loads settings explicitly, never at module import.
`.env.example` documents variables; `.env` is ignored and is **not automatically
loaded**. Export variables in your shell before live execution.

| Variable | Requirement/default |
| --- | --- |
| `OPENAI_API_KEY` | Required secret; never logged or serialized |
| `OPENAI_MODEL` | Required ID supporting Responses strict structured outputs |
| `OPENAI_TIMEOUT_SECONDS` | Default 30; greater than 0, at most 120, per attempt |
| `OPENAI_MAX_OUTPUT_TOKENS` | Default 2000; 256–16000 |
| `OPENAI_MAX_RETRIES` | Default 0; 0–2 additional attempts |
| `OPENAI_TEMPERATURE` | Optional; omitted when unset/empty; 0–2 when present |

No model is silently substituted. Unsupported model/settings combinations produce
explicit API errors; some models reject temperature. The endpoint is fixed to the
OpenAI Responses API and `store=false` is sent. Custom gateways and organization or
project headers are outside this increment.

After setting required variables, run from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --strategy structured-llm --allow-live --split dev --output runs/structured-llm
```

The flag is required even when credentials exist. This command performs 20 paid
generation calls, multiplied by repeats and potentially retries. Token usage depends
on inputs and model. No price or spending ceiling is inferred. Configure output
limits/account controls and start with smoke tests before a full development run.
Keep held-out data out of prompt tuning.

## Metadata and Failures

Optional `last_execution: ExecutionInfo` telemetry preserves the core interface and
resets before each call. The runner snapshots provider name, requested/returned model
IDs, response ID, prompt version/hash, nonsecret settings, retry count, and input/output
token usage when available. Runner-owned dataset/schema identity cannot be replaced
by provider telemetry. Usage remains attached to failures when a usable API envelope
was received. Reports aggregate token totals and availability counts without inventing
prices or collecting timings.

Timeouts, connection errors, and HTTP 429/5xx may retry within the configured bound,
with short exponential backoff. Nontransient HTTP errors, malformed envelopes,
invalid JSON/schema/domain output, refusals, and incomplete responses do not retry
or fall back to the rule baseline. Errors use fixed safe messages, never arbitrary
API bodies, refusal text, prompts, or keys.

Invalid structured/domain output raises `invalid_schema`; invalid references raise
`invalid_references` with a schema-valid candidate retained only for diagnostics.
Refusal, incomplete output, timeout, and API errors have explicit provider codes.
All are failures, never fabricated abstentions. Valid explicit abstentions remain
successful recommendations. Semantically wrong but structurally valid choices reach
shared policy and independent evaluation without silent correction.

Usage describes the final returned response; failed transient attempts may have
unknown usage. Socket timeout is per operation/attempt, not a global benchmark
deadline. Instances maintain telemetry for synchronous use and should not be reused
concurrently. Live results and response IDs vary between requests. Confidence remains
uncalibrated, and semantic evidence support remains a manual rubric task.

## Tests and Validation Status

Normal tests need no key and block unmocked provider HTTP. Tests mock transport/provider
behavior and exercise the full adapter → strategy → runner → evaluator pipeline,
including error handling, metadata, domain validation, and secret exclusion:

```bash
python -m pytest -q --basetemp .pytest_cache/foundation-tests
```

Two smoke tests require `--run-live`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. They exercise
complete and missing-information inputs without asserting model accuracy or passing
reference labels to requests. They make two paid requests plus configured retries;
API/validation errors fail visibly:

```bash
python -m pytest -q --run-live tests/test_openai_live.py --basetemp .pytest_cache/live-tests
```

No live API execution was performed in this increment. Account access, model
compatibility, and actual benchmark quality still require configured live validation.
