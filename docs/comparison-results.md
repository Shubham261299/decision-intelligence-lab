# Mocked Comparison Results

This is an offline pipeline demonstration, **not an LLM performance experiment**.
The real RuleBased, DirectLLM, and StructuredLLM strategy classes run all 30 frozen
cases twice. Fake LLM providers return the rule recommendations in their respective
text/JSON formats. Provider usage values are synthetic; no API call or charge occurs.

| Metric | RuleBased | Mocked DirectLLM | Mocked StructuredLLM |
| --- | --- | --- | --- |
| Overall correctness | 54/60 | 54/60 | 54/60 |
| Selection correctness | 32/38 | 32/38 | 32/38 |
| Required abstention correctness | 22/22 | 22/22 | 22/22 |
| Execution failures / known invalid outputs | 0/60 / 0/60 | 0/60 / 0/60 | 0/60 / 0/60 |
| Human-review precision / recall | 26/32 / 26/26 | 26/32 / 26/26 | 26/32 / 26/26 |
| Hard policy violations | 0/32 | 0/32 | 0/32 |
| Exact repeated agreement | 30/30 | 30/30 | 30/30 |
| Valid abstentions / unnecessary abstentions | 28 / 6 | 28 / 6 | 28 / 6 |
| Confidence / support samples | 0 / 0 | 0 / 0 | 0 / 0 |
| Input / output tokens | unavailable | 4800 / 2400 (fake) | 4800 / 2400 (fake) |
| Cost | unavailable | unavailable | unavailable |

All columns inherit the baseline's three tie-abstention misses on each repetition.
Valid evidence citations are not evidence of semantic support. Matching mocked scores
do not demonstrate equivalent LLM behavior. See [metric interpretation](comparison.md).

Reproduce the example using the deterministic test fixtures from the repository root:

```powershell
$env:PYTHONPATH = 'src;tests'
@'
from pathlib import Path
from decisionlab.evaluation.benchmark import load_benchmark
from decisionlab.comparison import run_comparison, write_comparison
from test_comparison import entries
benchmark = load_benchmark(Path("evals"))
strategies, _, _ = entries(benchmark, repeats=2)
report = run_comparison(benchmark, strategies, repeats=2)
write_comparison(report, Path("runs/comparison-mocked"))
'@ | python -
```

This developer demonstration requires the existing pytest development environment
because it imports test fixtures. It writes ignored local `comparison.json` and
`comparison.txt`, explicitly marking LLM columns mocked. For a dependency-free
offline CLI smoke run, use `compare --strategies rules --repeats 2`; live LLM results
remain pending and require the explicit command documented in `comparison.md`.
