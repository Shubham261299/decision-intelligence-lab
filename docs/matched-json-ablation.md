# Matched Structured-Output Ablation

## Research Question and Control

Does schema enforcement itself improve usable decision-making? The first live
comparison changed prompts, output formats, and parsing pipelines together, so it
cannot isolate that effect.

`ablate` compares `unconstrained-json` and `schema-enforced-json`. Both use the
existing routing prompt plus versioned `matched_json_v1.txt` instructions and the
same canonical Recommendation transport schema in the system prompt. Both parse
strict JSON through StructuredRecommendation, Recommendation, and original-problem
reference validation. No Markdown stripping, repairs, or alternate parser is used.

The only intended request difference is Ollama's `format` field: absent in the
unconstrained arm, containing that same schema in the enforced arm. Model,
temperature, output limit, timeout, retry policy, messages, input serialization,
local validation, shared policy, evaluator, and verifier are held fixed. Separate
providers share one validated environment-settings snapshot. Existing DirectLLM,
StructuredLLM, and OpenAI behavior is preserved.

## Running Later

No live ablation has been run during implementation. From PowerShell:

```powershell
$env:PYTHONPATH = 'src'
$env:DECISIONLAB_PROVIDER = 'ollama'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL = 'qwen2.5:14b'
$env:OLLAMA_TEMPERATURE = '0'
$env:OLLAMA_MAX_OUTPUT_TOKENS = '2000'
python -m decisionlab.cli ablate --provider ollama --allow-live --split dev --repeats 3 --output runs/matched-json/dev
```

This makes 120 generations: 20 cases × two conditions × three repetitions. Model
choice comes from environment settings; the mechanism does not hardcode or pull a
model. Local endpoint restrictions remain enforced. CLI defaults to dev and one
repetition; `--allow-live` is mandatory. The existing `held_out` split is available,
but its previous results have already informed the hypothesis: it is not fresh
confirmatory evidence. Do not tune on it. Reserve separately specified unseen cases
for future confirmation without changing frozen benchmark v1.

## Artifacts and Metrics

The unchanged comparison runner writes `comparison.json` and `comparison.txt`.
Column/strategy names identify conditions; every run records
`generation_settings.condition`, experiment/contract versions, schema hash, prompt
version/hash, provider/model/settings, usage, latency, outcome, recommendation,
validation diagnostics, and verifier result. Failed outputs remain failed; retained
reference-invalid candidates remain separate from accepted recommendations.

Inspect overall correctness, selection/abstention correctness, schema/reference
failure rates, policy violations, review precision/recall, confidence availability
and calibration, repeated agreement, tokens, and latency. For semantic attribution,
also compare paired cases where both conditions return valid outputs, reporting
that subset's denominator. This subset is descriptive and selected by execution
success; it is not an unbiased estimate of overall reasoning quality.

The generic comparison report flags configuration differences because `condition`
differs intentionally; inspect profiles to ensure no other settings differ.
HTTP-mocked tests assert exact payload equality after removing only `format`.

## Limitations

The 30 synthetic cases and related pairs cannot establish general reliability.
Fixed condition-block execution order, caching, model loading, and backend
nondeterminism can affect results; temperature zero is not a repeatability guarantee.
Schema enforcement changes decoding and can influence content, not just syntax.
Provider schema support does not enforce contextual references or all Pydantic
domain invariants. Evidence support remains unscored; confidence metrics exclude
failures, abstentions, and missing values. Invalid-schema failures do not retain raw
responses, limiting syntax-versus-domain diagnosis. No statistical superiority or
production safety claim follows from this experiment.
