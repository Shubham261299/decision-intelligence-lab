# Rule-Based Routing Baseline

This document describes the existing implementation, not the benchmark oracle.
[Benchmark v1](benchmark.md) accepts either eligible queue on ties and requires
review for explicit supervisor-review facts. The baseline currently abstains on
ties and ignores review facts on selections. Preserve these known differences
until the planned shared-policy increment; do not alter reference labels to fit
the strategy.

## Input Contract

The generic strategy protocol uses only `DecisionProblem` and `Recommendation`.
The routing implementation interprets specific fields through Pydantic
`RoutingFacts` and `RoutingAttributes` models:

- Facts with IDs `urgent` and `specialist_required` must have Boolean values.
  Strings, integers, missing facts, and null values are unresolved inputs.
- Each option has `route` (`routine`, `expedited`, or `specialist`) and a
  nonnegative integer `capacity`. Boolean/string capacities and extra attributes
  are unsupported in version 1.
- The only executable hard constraint ID is `capacity`; its meaning is fixed to
  positive queue capacity. Arbitrary constraint descriptions are not interpreted.
  Other hard constraints cause abstention. Soft constraints are not scored.
- Unrelated facts do not affect the decision. Context and descriptions are
  explanatory text; they do not override typed fact values.

## Selection Rules

1. Specialist requirements take precedence over urgency.
2. Otherwise urgent requests require an expedited queue.
3. Otherwise requests require a routine queue.
4. Only options matching that route with capacity greater than zero are eligible.
   Capacity is mandatory even when no capacity constraint is supplied.
5. Select only when exactly one option is eligible. Abstain for ties or no eligible
   option; never fall back to a different route or silently choose by input order.
6. Any unresolved required input or unsupported hard constraint causes conservative
   abstention, including malformed attributes on a nonselected option.

Abstentions request human review. Selections do not request review automatically;
the future shared policy may impose additional review requirements. Confidence
remains absent because a deterministic rule match is not a calibrated probability.

## Output and Evidence

Every option receives an assessment. Valid-input assessments cite the two routing
fact IDs and the capacity constraint when present; capacity values come from option
attributes, not invented evidence facts. Unresolved-input assessments report unknown
eligibility and material unknowns. References are checked against the problem before
returning. The strategy does not mutate inputs, produce run records, apply shared
policy, or execute routing actions.

## Examples and Validation

`examples/routing.jsonl` contains three synthetic problems whose expected choices
are routine, expedited, and specialist. These are demonstrations, not a frozen
held-out benchmark. Unit tests additionally cover missing/invalid inputs, exhausted
capacity, ties, unsupported constraints, distractors, deterministic behavior, and
substitution of an unrelated protocol implementation.

Run the complete offline suite with `python -m pytest -q`.
