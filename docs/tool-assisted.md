# Tool-Assisted Decision Strategy

## Architecture and Scope

`ToolAssistedStrategy.decide(problem: DecisionProblem) -> Recommendation` implements
the unchanged strategy protocol. It performs one fixed information plan, bounded
tool execution, and one decision-generation call. Verification and benchmark
evaluation remain external. There is no autonomous planning/replanning loop, provider
tool calling, action execution, or agent framework.

```text
Original problem -> required-information plan -> allowlisted tools -> evidence
                                                                    |
                                                        decision generator
                                                                    |
                                                       Recommendation
                                                                    |
                                                 policy / verifier / evaluator
```

The initial generator is deterministic and requires no provider or key. A small
`ToolDecisionGenerator` protocol allows later provider-independent generation with
the same explicit evidence collection, but no new LLM integration is implemented.
The default generator consumes retrieved fact and policy snapshots, then uses
supplied option attributes. Reference expectations, category tags, pair metadata,
and scoring rubrics never enter tools or generation.

| Component | Typed contract |
| --- | --- |
| Tool | `execute(problem, request) -> ToolOutput`, plus name/version |
| ToolRequest | Tool name and JSON-only input mapping |
| ToolOutput | JSON data plus original fact/constraint ID links |
| ToolResult | Original request, tool version, success/failure, evidence ID, output or safe error code |
| ToolPlanner | `plan(problem) -> Sequence[ToolRequest]` |
| ToolDecisionGenerator | `generate(problem, evidence) -> Recommendation`, plus name/version |

## Registered Tools and Required Information

The default registry explicitly registers and allowlists exactly two tools:

- **fact_lookup v1:** retrieve requested Fact snapshots from the current problem;
  return explicit missing IDs without inventing values or failing the tool call.
- **policy_lookup v1:** return the repository-owned routing-policy document and
  supplied Constraint snapshots. Only `routing-benchmark-1` is supported.

The planner requests urgent/specialist classification, approval when hard-required,
and review_required when supplied. It executes policy lookup and one batched fact
lookup: two calls per decision. Missing/null/unverified material information leads
the generator to a valid review-required abstention. Tool execution failure instead
produces an execution failure, never silent abstention or a rule fallback.

Generation supports capacity/approval and specialist precedence, selecting the
lowest eligible option ID on ties. This is intentionally distinct from the original
RuleBasedStrategy's conservative tie abstention. The policy lookup specifies this
deterministic preference; the benchmark accepts any eligible tied option. No labels
are used to select one. Existing strategies remain unchanged.

## Provenance and Evidence References

Each accepted result gets `tool:<name>:<sha256>`, derived from problem ID, canonical
request, tool version, and output. Records preserve actual request inputs/results
and source IDs, with deterministic IDs across identical repetitions. Failures have
codes and no accepted evidence ID/output. Fresh sessions reset trace and limits for
each decision; returned values and generator inputs are copied.

Recommendation.evidence_ids retain their original meaning: supplied Fact IDs, not
tool-record IDs. Constraint references likewise remain original IDs. The strategy
rejects citations not collected by tools. ToolResult source-ID links connect these
citations to their explicit retrieved evidence; policy-result IDs also appear in
assessment reasons. The original problem is not enriched or overwritten. No new
facts are silently admitted into reference validation.

The default generator checks retrieved facts/constraints against the original
snapshots and validates the supported policy document before using them. The tools
add access paths and provenance, not privileged knowledge. The source benchmark
remains the exact same checksum-verified input for every comparison column.

## Bounds and Safety

The registry has explicit names and an explicit allowlist; there is no discovery,
dynamic import, callable input, path/URL resolution, shell, or arbitrary-code tool.
The two built-ins use only in-memory problem data and static policy. Typed argument
models reject extra keys and malformed inputs. Requests carry JSON values only.

Sessions permit 1–16 calls, default 2. Canonical identical requests are rejected
before repeat execution; mapping key order is ignored, but array order remains
part of the request identity. Unknown/nonallowlisted requests and invalid requests
consume attempted-call budget. Exceeding the budget never invokes a tool. The first
blocked/failing request is recorded; the session then closes, with no further trace
growth. A trace may contain the allowed calls plus one blocked limit attempt.

Tool errors, malformed results, duplicate calls, unknown tools, and limit exhaustion
are explicit safe execution codes. Error payloads and arbitrary exception messages
are excluded from reports/logging. Actual requested data is retained in local run
artifacts for provenance; tools have no source of credentials or external data.

Registration is trusted Python developer configuration. The interface and allowlist
do not sandbox malicious Python implementations: register only reviewed, pure tools.
No untrusted user/model can supply or import an implementation. The built-ins have
no filesystem/network/shell capabilities; adding I/O tools would require a separate
capability, timeout, provenance and information-access design.

## Comparison, Verification and CLI

Runner observations now include optional `tool_results`; all accepted and blocked
tool requests survive in JSON, including on failed strategy executions. Existing
model-independent execution telemetry records generator name/version, allowlist,
and call budget. No provider, prompt, token, or cost values are invented for the
deterministic generator. Original strategy scores and verifier behavior are unchanged.

Existing policy/reference checks and DecisionVerifier evaluate the recommendation
against the original problem. Valid citations still do not prove semantic evidence
support. Wrong generated decisions remain wrong and can be rejected by verification;
verification never reselects a queue or rewrites the recommendation.

Run offline from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m decisionlab.cli evaluate --strategy tool-assisted --output runs/tool-assisted
python -m decisionlab.cli compare --strategies rules tool-assisted --repeats 2 --output runs/comparison-tools
```

`tool-assisted` requires no key or live flag. Direct/structured selections still
require `--allow-live`; the same comparison can include all four strategies when
explicitly configured. The default CLI comparison remains rules-only. The existing
DecisionStrategy/Recommendation interfaces and benchmark/evaluator remain unchanged.

On all 30 frozen cases, this deterministic generator achieves 30/30 outcomes and
zero execution failures; the original rule baseline remains 27/30. Its verifier
diagnostics are 17 accepts and 13 reviews. This result demonstrates implementation
alignment with the small routing policy, **not a tool-induced improvement in LLM
reasoning**: tie handling and generation rules differ, and no LLM was called.

## Differences and Limitations

DirectLLM produces normal text; StructuredLLM uses API-enforced schema generation.
This strategy explicitly collects bounded tool evidence before a separate generator
produces the same Recommendation. DecisionVerifier inspects a decision afterwards
and cannot collect information or choose an option. An autonomous agent would
iteratively choose actions/goals; this strategy executes a fixed two-request plan
and one generation without replanning or self-directed actions.

Tests use pure built-ins and deterministic mocks for unknown tools, failures,
duplicates, exhaustion, malformed evidence, provenance, generation errors,
verification and comparison isolation. Run
`python -m pytest -q --basetemp .pytest_cache/foundation-tests`.

No external dependencies, new information sources, LLM tool loop, or action tools
were added. Confidence remains absent/uncalibrated; semantic evidence support is
unscored. The benchmark is small and public. Evidence hashing identifies records,
not signatures or proof of truth. Custom planners/generators/tools are trusted code
and require their own review; there is no wall-clock sandbox for arbitrary callbacks.
Provider-backed tool-assisted generation and genuine missing-information acquisition
remain future experiments requiring explicit access comparisons.
