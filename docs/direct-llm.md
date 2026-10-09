# Direct LLM Strategy

## Experimental Purpose

Compare RuleBased, DirectLLM, and StructuredLLM on the frozen routing benchmark.
DirectLLM is a deliberately less constrained generation baseline: it requests
labelled normal text, with no provider-enforced JSON schema, JSON mode, or tools.
It still needs an explicit response convention to produce measurable decisions.
This is not an arbitrary-prose interpreter.

| Property | DirectLLM | StructuredLLM |
| --- | --- | --- |
| Routing policy instructions | Identical prefix | Identical prefix |
| Per-case input | DecisionProblem JSON only | DecisionProblem JSON only |
| API output enforcement | Plain text | Strict Pydantic JSON schema |
| Local interpretation | Versioned labelled-text parser | Pydantic transport schema |
| Accepted contract | Recommendation plus reference validation | Same |
| Policy, evaluator, dataset | Shared | Shared |

The adapter explicitly sends `text.format.type=text` for direct calls; the
[OpenAI Responses reference](https://developers.openai.com/api/reference/python/resources/responses)
documents plain text and schema-constrained formats. Both modes share model settings,
HTTP transport, bounded retries, status handling, and usage normalization. No new
dependencies or configuration variables are required.

## Architecture and Parsing

`DirectLLMStrategy.decide(problem: DecisionProblem) -> Recommendation` implements
the unchanged strategy protocol. It depends only on the provider-neutral TextProvider
contract; provider and evaluator code stay outside the strategy. Expectations,
rubrics, categories, and pair metadata never enter requests. Only the input problem
and the versioned `prompts/direct_routing_v1.txt` instructions are sent.

The `direct-text-v1` parser requires these case-sensitive fields in order:

```text
Decision: selected
Option: q-r
Confidence: unavailable
Human review: no
Explanation: Verified intake indicates routine routing with available capacity.
Assessments: present
Assessment: q-r | eligible
Reason: Both classification facts are false and capacity is positive.
Evidence: urgent, specialist_required
Constraints: capacity
Unknowns: none
```

Use `Decision: abstained` with `Option: none` for explicit abstention. Confidence is
`unavailable` or a decimal fraction from 0 through 1. Human review and material flags
use `yes`/`no`. Assessments and unknowns use `present` or `none`; present requires
at least one block. Each assessment has exactly the four lines shown, with one reason.
Unknown blocks contain `Unknown: description`, `Affected options: IDs`, and
`Material: yes` or `no`. Reference lists use commas or the exact token `none`.

Blank lines and surrounding whitespace are ignored. Every semantic field must be
explicit; duplicate fields, out-of-order lines, Markdown fences, trailing prose,
ambiguous statuses, percentages, and missing fields fail. No regex decision extraction,
case normalization, default review/confidence inference, repair call, or rule fallback
occurs. Pydantic checks domain invariants; problem validation checks all references.

## Metadata and Failures

Reports preserve provider, requested/returned model identity, response ID, strategy
version, prompt version `direct-routing-v1`, prompt hash, parser version, safe
configuration, retries, and available token usage. Telemetry resets before each call.
API secrets are excluded. Usable envelopes retain usage even when parsing fails.

Invalid text/domain output raises `invalid_schema`, never valid abstention.
Invalid references raise `invalid_references`; the candidate remains diagnostics only.
Provider refusal, timeout, connection/API error, incomplete output, and malformed
envelopes use the same explicit failure codes as StructuredLLM. Semantically wrong
but well-formed decisions reach shared policy and independent evaluation unchanged.

## Running a Fair Comparison

Export environment settings documented in [structured-llm.md](structured-llm.md)
and `.env.example`; environment files are not automatically loaded. Select a model
supporting both Responses modes, keep model and generation settings fixed, and use
identical splits and repeats. Start with development data; do not tune on held-out
cases. For PowerShell, from the repository root:

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --strategy rules --split dev --output runs/rules-dev
python -m decisionlab.cli evaluate --strategy direct-llm --allow-live --split dev --output runs/direct-dev
python -m decisionlab.cli evaluate --strategy structured-llm --allow-live --split dev --output runs/structured-dev
```

Each LLM command makes 20 paid requests before repeats/retries. `--allow-live` is
mandatory even with credentials. Compare JSON reports using dataset/policy hashes,
model identities, settings, and prompt versions. The matched `compare` command now
runs these strategies together; see [the comparison experiment](comparison.md).
Exit codes remain 0 for completed evaluation, 1 for recorded execution failures,
and 2 for input/configuration errors. Both text and JSON reports use existing metrics.

Report outcome correctness, failure/schema/reference rates, selection accuracy
alongside coverage, abstention, human review, violations, confidence sample counts,
and token usage. Never discard failed cases to improve apparent accuracy. Review
evidence support separately with the manual rubric.

## Validation and Limitations

Normal tests require no credentials and block unmocked HTTP. Mocked tests cover the
parser, provider payload parity, retries/errors, references, metadata, benchmark
isolation, full pipeline, and CLI/report output. Run
`python -m pytest -q --basetemp .pytest_cache/foundation-tests`.

No live DirectLLM experiment has been run. Mocked rule-shaped responses test plumbing,
not LLM accuracy. Prompt format instructions, schema tokens, and parsing conventions
are experimental differences alongside provider enforcement; this comparison cannot
isolate reasoning quality alone. The grammar reserves `none` and delimiters, supports
one reason per assessment, and is intended for the existing routing IDs. Extend and
version it explicitly before using incompatible identifiers or formats.

Confidence remains uncalibrated; evidence support is not automatically scored. The
benchmark is small and synthetic. Live outputs are stochastic and deterministic
report serialization does not make model calls reproducible. Retry usage reflects
the final usable response, not necessarily total billed usage. No superiority claim
is supported until actual matched runs are recorded.
