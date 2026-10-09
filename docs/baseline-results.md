# Rule Baseline Results

Dataset `decisionlab-service-routing` v1.0.0, schema 1, policy `routing-benchmark-1`,
strategy `rules` v1. One repeat on all 30 frozen cases; no benchmark label changes.

Reproduce from the repository root with `PYTHONPATH=src`:

```bash
python -m decisionlab.cli evaluate --benchmark evals --split all --output runs/baseline
```

| Metric | Result |
| --- | --- |
| Outcome correctness | 27/30 (90.0%) |
| Selection correctness | 16/19 (84.2%) |
| Required abstention correctness | 11/11 (100%) |
| Selected-output accuracy | 16/16 (100%) |
| Decision coverage | 16/30 (53.3%) |
| Abstention precision | 11/14 (78.6%) |
| Human-review precision | 13/16 (81.2%) |
| Human-review recall | 13/13 (100%) |
| Human-review accuracy | 27/30 (90%) |
| Raw strategy-review accuracy | 25/30 (83.3%) |
| Hard-constraint violations | 0/16 selected candidates |
| Schema/reference validity | 30/30 |
| Execution failures | 0/30 |
| Development outcome correctness | 18/20 |
| Held-out outcome correctness | 9/10 |
| Relevant-change pairs | 4/4 |
| Wording-invariance pairs | 2/2 |

The three outcome misses are `route-009`, `route-010`, and `route-025`: the strategy
abstains on ties, while the benchmark accepts either eligible queue. Policy does
not repair these missed selections. The abstentions trigger review, causing the
three final review false positives.

Raw strategy review misses `route-016` and `route-028`, both valid selections
requiring supervisor review. Shared policy fixes the final review gate without
changing the original recommendation or its requested-review flag.

Reported confidence has zero samples, so Brier score and reliability-bin estimates
are null. Evidence citation IDs are valid, but semantic evidence support has not
been scored. Timing and cost are unmeasured. High selected-output accuracy must
be read alongside low coverage; these synthetic results do not establish general
decision reliability. Conservative abstention can hide missed opportunities.
