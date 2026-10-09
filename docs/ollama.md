# Local Ollama Provider

## Provider Boundary

DirectLLM and StructuredLLM retain their contracts and prompts. Both use the
existing text/structured protocols, implemented by OpenAIProvider and OllamaProvider.
A factory snapshots environment settings once per comparison; each strategy receives
its own adapter with identical settings. OpenAI remains the default unless `--provider`
or `DECISIONLAB_PROVIDER` selects Ollama. The OpenAI adapter, evaluator and benchmark
are unchanged. No external dependency or API key is required for Ollama.

Ollama uses non-streaming `POST /api/chat` with separate system instructions and
original problem JSON. DirectLLM sends no `format` and retains its strict text parser.
StructuredLLM sends its full Pydantic JSON Schema in `format` and retains strict
transport/domain/reference validation. There is no fallback to text/JSON-only mode,
repair or weaker local validation. See the official [chat API](https://docs.ollama.com/api/chat)
and [structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Ollama schema support is not asserted to provide identical guarantees to OpenAI
strict mode across server/model versions or unsupported schema constructs.
Configuration records `structured_output_capability`; response metadata records
actual generation format and required local validation. Schema validity is not
decision correctness or calibrated confidence.

## Environment Configuration

```powershell
$env:PYTHONPATH = 'src'
$env:DECISIONLAB_PROVIDER = 'ollama'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL = 'qwen2.5:14b'
$env:OLLAMA_TIMEOUT_SECONDS = '180'
$env:OLLAMA_MAX_OUTPUT_TOKENS = '2000'
$env:OLLAMA_TEMPERATURE = '0'
python -m decisionlab.cli preflight --provider ollama
```

`.env.example` documents both providers; `.env` is not loaded automatically. Settings
are loaded explicitly and no network calls happen at import time. Only plain HTTP
loopback URLs without credentials, paths, queries or fragments are accepted.
`localhost` is canonicalized to `127.0.0.1`; IPv6 loopback also works. The transport
bypasses environment proxies and rejects redirects. Cloud-named/remote model tags
are rejected; preflight rejects remote-backed tags when identified by model metadata.
A trusted local daemon is assumed; the client cannot police its own implementation
or outbound behavior.

Preflight reads only [GET /api/tags](https://docs.ollama.com/api/tags) and verifies
the exact installed tag/digest. It generates nothing, installs nothing and never
calls model-management endpoints. Missing models fail without downloading.

## First Experiment Commands — Not Yet Executed

After exporting the configuration above:

```powershell
python -m decisionlab.cli compare --provider ollama --strategies rules direct-llm structured-llm --allow-live --split dev --output runs/ollama-first/dev
python -m decisionlab.cli compare --provider ollama --strategies rules direct-llm structured-llm --allow-live --split held_out --output runs/ollama-first/held-out
```

These make 40 dev and 20 held-out local generations (one repetition, two LLM
strategies). `--allow-live` remains explicit for local generation. Optional
`tool-assisted` uses a deterministic non-LLM generator. Provider selection also
works for `evaluate` and `robustness`. No full benchmark was run for this increment.

## Metadata and Failures

Reports retain provider, requested/returned model, prompt/strategy versions,
recommendations, policy/verifier/evaluation results, and available token counts.
Ollama supplies no response ID; it stays null. Optional `response_metadata` retains
creation time, completion state/reason, format and server timings. Latency uses
server `total_duration` converted from nanoseconds to milliseconds, not client
wall-clock time. Missing measurements stay null. Preflight reports the model digest;
retain it with experiment records because local tags can change.

There are no Ollama retries. Connection failures, timeouts, HTTP errors, malformed
envelopes, incomplete generation and invalid strategy output remain explicit failures.
No raw error bodies, prompts or sensitive environment values are logged. Local
inference has no cloud API billing; local resource consumption is not measured.

## Connectivity and Smoke Result

On 2026-10-08, preflight found `qwen2.5:14b` at `http://127.0.0.1:11434/api/tags`,
digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
Exactly one generation used `http://127.0.0.1:11434/api/chat`: a tiny schema requiring
`status: "ok"`, with a 32-token limit and no benchmark input. It completed and passed
strict Pydantic validation, consuming 36 prompt and 10 output tokens. Server duration
was 13,028.6805 ms, including 10,475.4121 ms model loading.

The local record is `runs/ollama-smoke/report.json`. This demonstrates connectivity
and a minimal schema only, not full Recommendation schema compatibility, decision
accuracy, calibration or text-parser compliance. Mocked HTTP tests cover both
strategies and error paths without requiring a running server. No extra generation,
cloud API call, model download, installation change or benchmark experiment occurred.
