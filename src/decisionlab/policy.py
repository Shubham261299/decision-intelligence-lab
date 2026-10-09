"""Shared deterministic routing policy; no strategy or reference-label imports."""

from decisionlab.models import DecisionProblem, PolicyResult, Recommendation

POLICY_VERSION = "routing-benchmark-1"


def unresolved_inputs(problem: DecisionProblem) -> list[str]:
    facts = {fact.id: fact for fact in problem.facts}
    reasons: list[str] = []
    for key in ("urgent", "specialist_required"):
        fact = facts.get(key)
        if (
            fact is None or type(fact.value) is not bool
            or fact.source != "verified-intake"
        ):
            reasons.append(f"unresolved_fact:{key}")
    for option in sorted(problem.options, key=lambda item: item.id):
        capacity = option.attributes.get("capacity")
        if type(capacity) is not int or capacity < 0:
            reasons.append(f"unresolved_capacity:{option.id}")
        if option.attributes.get("route") not in ("routine", "expedited", "specialist"):
            reasons.append(f"unresolved_route:{option.id}")
    for constraint in sorted(problem.constraints, key=lambda item: item.id):
        if constraint.kind != "hard":
            continue
        if constraint.id == "approval":
            approval = facts.get("approval_granted")
            if approval is None or approval.value is not True:
                reasons.append("approval_not_granted")
        elif constraint.id != "capacity":
            reasons.append(f"unsupported_hard_constraint:{constraint.id}")
    return reasons


def selection_violations(
    problem: DecisionProblem, recommendation: Recommendation,
) -> list[str]:
    """Check actual facts/attributes, never trust self-reported eligibility."""
    if recommendation.status == "abstained":
        return []
    violations = unresolved_inputs(problem)
    option = next(
        (o for o in problem.options if o.id == recommendation.selected_option_id), None
    )
    if option is None:
        return sorted(set(violations + ["unknown_selected_option"]))
    capacity = option.attributes.get("capacity")
    if type(capacity) is int and capacity == 0:
        violations.append("capacity_exhausted")
    facts = {fact.id: fact for fact in problem.facts}
    if not any(reason.startswith("unresolved_fact:") for reason in violations):
        route = (
            "specialist" if facts["specialist_required"].value
            else "expedited" if facts["urgent"].value else "routine"
        )
        if option.attributes.get("route") != route:
            violations.append("wrong_route")
    return sorted(set(violations))


def enforce_policy(
    problem: DecisionProblem, recommendation: Recommendation,
) -> PolicyResult:
    """Return a separate review gate; never repair or replace a recommendation."""
    problem.validate_recommendation(recommendation)
    violations = selection_violations(problem, recommendation)
    reasons = unresolved_inputs(problem) + violations
    if recommendation.status == "abstained":
        reasons.append("strategy_abstained")
    if recommendation.requested_review:
        reasons.append("strategy_requested_review")
    if any(unknown.material_to_decision for unknown in recommendation.unknowns):
        reasons.append("strategy_reported_material_unknown")
    review_fact = next((f for f in problem.facts if f.id == "review_required"), None)
    if review_fact is not None:
        if review_fact.value is True:
            reasons.append("explicit_supervisor_review")
        elif type(review_fact.value) is not bool:
            reasons.append("unresolved_review_requirement")
    reasons = sorted(set(reasons))
    return PolicyResult(
        requires_human_review=bool(reasons), review_reasons=reasons,
        detected_violations=violations, policy_version=POLICY_VERSION,
    )
