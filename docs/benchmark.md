# Service-Request Routing Benchmark v1.0.0

## Purpose and Scope

Thirty synthetic operational-routing problems test selection, abstention, evidence,
and human review independently. They are authored reference cases, not predictions
from RuleBasedStrategy. The independent evaluator is documented in `evaluation.md`;
benchmark inputs, expectations, rubrics, and hashes remain unchanged.

Select a handling queue; do not execute routing. Options are routine, expedited,
and specialist queues. Human review is an output flag, not a fourth queue option.
This intentionally narrows the initial plan's possible human-triage option.

## Normative Decision and Review Policy

1. Both `urgent` and `specialist_required` must be resolved Boolean values from
   `verified-intake`. Missing/null values require abstention and human review.
   Requiring both is a conservative benchmark rule, including specialist requests.
2. Specialist need takes precedence. Otherwise urgency requires expedited routing;
   otherwise use routine routing. Never substitute another route for lack of capacity.
3. Queue `capacity` must be known and greater than zero. Unknown capacity blocks
   selection; exhausted required queues create a hard-constraint conflict.
4. If multiple queues satisfy the requirements, selecting **any** is acceptable.
   Queue order, capacity magnitude, and names supply no ranking preference.
   A tie alone does not require abstention or review in this benchmark.
5. An `approval` hard constraint requires `approval_granted=true`; unresolved
   approval requires abstention/review. There is no approval-positive case in v1.
6. All abstentions require review. A valid selection also requires review when
   `review_required=true`. Otherwise review is not required.
7. Verified current values outrank superseded `stale-note` statements. Conflicting
   peer reports cannot resolve a null verified value; abstain rather than guess.
   Decorative context and customer status do not affect eligibility.

All cases have valid generic domain schemas. Missing routing attributes are valid
problem data representing unknown information, not execution failures. Invalid
schemas belong in unit tests, not this benchmark. No probabilistic confidence
ground truth is supplied.

## Artifacts and Leakage Boundary

| Artifact | Content |
| --- | --- |
| `evals/cases/dev.jsonl` | 20 strategy inputs, each a DecisionProblem |
| `evals/cases/held_out.jsonl` | 10 strategy inputs |
| `evals/expectations/*.jsonl` | EvaluationCase labels joined by problem ID |
| `evals/expectations/rubrics.json` | Per-case grounds, evidence IDs, constraints, review reasons, manual scales |
| `evals/manifest.json` | Versions, ordered split IDs, pairs, SHA-256 artifact hashes |

Case IDs are neutral sequential IDs. Input files contain no expected answers,
category tags, or scoring instructions. Legitimate input facts such as a supervisor
review requirement are decision evidence, not injected reference labels.
Strategies must receive only deserialized DecisionProblem objects. The future
runner must not pass expectations, rubrics, manifest pair metadata, or this guide
into prompts. Filesystem separation prevents accidental mixing; it is not a security
sandbox against malicious code reading the repository.

The files are static UTF-8 JSONL/JSON with LF line endings. There is no random
generation or sampling at runtime. Verify checksums with the dataset tests. Preserve
bytes, ordered IDs, and splits for v1. Changes to cases, policy, or rubrics require a
new version and refreshed hashes; never silently tune labels to a strategy.
Held-out labels are public for reproducibility, so this is a development holdout,
not a secret or independently administered test set. Avoid tuning to it.

## Case Inventory

| Cases | Purpose | Expected behavior |
| --- | --- | --- |
| 001–002, 003–004 | Urgency and specialist counterfactuals | One changed fact changes the route |
| 005–006 | Routine wording invariance | Routine selection stays unchanged |
| 007–008 | Capacity counterfactual | Expedited selection becomes abstention |
| 009–010 | Multiple acceptable routine/specialist queues | Either matching queue accepted |
| 011–013 | Null urgency, absent specialist fact, unknown capacity | Abstention and review |
| 014–015 | Exhausted queues and specialist/capacity conflict | Abstention and review |
| 016 | Valid routine selection with supervisor review | Select and request review |
| 017–018 | Customer-status distractor and stale contradictory note | Follow verified classification |
| 019–020 | Unresolved conflicting reports and missing approval | Abstention and review |
| 021–022 | Held-out urgency counterfactual | Routine becomes expedited |
| 023–024 | Held-out specialist wording invariance | Specialist selection stays unchanged |
| 025 | Multiple acceptable expedited queues | Either matching queue accepted |
| 026–027 | Unknown specialist need and exhausted routine capacity | Abstention and review |
| 028 | Valid specialist selection with review | Select and request review |
| 029–030 | Stale note versus unresolved peer conflict | Expedited selection versus abstention |

There are 19 selection cases (3 with two acceptable options), 11 required
abstentions, and 13 review-required cases, including 2 valid selections. Categories
overlap. Four relevant-fact pairs and two wording-invariance pairs remain entirely
within their split. Nonpaired scenarios deliberately exercise the same domain rules
across splits; the split does not claim complete semantic independence.

Category counts: clear 13, hard conflicts 5, multiple acceptable options 3, missing
information 7, abstention 11, human review 13, distractors 3, conflicting evidence 4,
relevant-pair members 8, and wording-pair members 4.

## Evaluation Rubric Specification

Record dimensions separately when the evaluator is implemented:

- **Outcome correctness:** selection belongs to the acceptable set, or explicit
  abstention matches an abstention case. Abstaining on a multiple-option selection
  case is an outcome failure, not a hard-constraint violation.
- **Hard constraints:** no selected queue has zero/unknown capacity, wrong route,
  or unresolved required approval. Correct abstention incurs no violation.
- **Review correctness:** compare the final system review flag with `expected_review`.
  Until shared policy exists, strategy-requested review is only a partial signal.
- **Evidence support:** manually score 0 (unsupported/fabricated/contradicted),
  1 (partial, missing required grounds), or 2 (all case-specific grounds supported).
  Per-case required IDs identify supplied evidence; capacity/route are option
  attributes, not Fact IDs. Missing facts must be acknowledged, never cited as if
  present. Constraint references identify rules rather than proving factual support.
- **Unknown handling:** required abstentions identify the specific missing input,
  capacity blocker, or conflict using the per-case rubric; generic uncertainty alone
  earns at most partial evidence credit.
- **Pair consistency:** relevant pairs change the accepted outcome; wording pairs
  retain it. Report paired outcomes separately rather than inflate sample size.

Execution failures are unsuccessful cases and are never recast as abstentions.
Report all-case outcome correctness, selected-output correctness with coverage,
review behavior, and evidence scores separately. Do not set an LLM performance
threshold before observing the baseline. Thirty synthetic cases cannot establish
general reliability or strong confidence calibration.

## Known Baseline Gaps and Next Work

RuleBasedStrategy intentionally abstains on ties and ignores `review_required`.
The current runner exposes failures on cases 009, 010, 025 (selection). Raw strategy
review misses cases 016 and 028; shared deterministic policy now enforces review on
those valid selections without changing them. These are
specification differences to expose, not reasons to rewrite expectations. Unsupported
approval constraints cause conservative abstention; the strategy does not implement
general approval enforcement or source conflict resolution.

Shared validation/policy and the independent evaluator are now implemented. Baseline
tie abstention is deliberately retained and measured; reference labels are unchanged.
See `baseline-results.md` for the first recorded result and `evaluation.md` for
metrics and execution boundaries. Dataset tests check artifacts and reference
integrity only; separate foundation tests run and score strategy results.
