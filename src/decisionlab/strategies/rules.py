"""Conservative, deterministic service-request routing baseline."""

from typing import Annotated, Literal

from pydantic import Field, StrictBool, ValidationError

from decisionlab.models import (
    DecisionProblem,
    DomainModel,
    OptionAssessment,
    Recommendation,
    Unknown,
)


class RoutingFacts(DomainModel):
    urgent: StrictBool
    specialist_required: StrictBool


class RoutingAttributes(DomainModel):
    route: Literal["routine", "expedited", "specialist"]
    capacity: Annotated[int, Field(strict=True, ge=0)]


class RuleBasedStrategy:
    """Select one matching queue with capacity; otherwise abstain for review.

    Fact IDs are `urgent` and `specialist_required`; their values must be bools.
    Every option uses RoutingAttributes. Only the hard constraint ID `capacity`
    is supported; capacity is always checked, even without that constraint.
    Other hard constraints cause abstention. Soft constraints are not scored.
    """

    name = "rules"
    version = "1"

    def decide(self, problem: DecisionProblem) -> Recommendation:
        facts_by_id = {fact.id: fact for fact in problem.facts}
        unknowns: list[Unknown] = []
        option_ids = sorted(option.id for option in problem.options)
        for fact_id in ("urgent", "specialist_required"):
            fact = facts_by_id.get(fact_id)
            if fact is None or type(fact.value) is not bool:
                unknowns.append(Unknown(
                    description=f"Fact {fact_id} requires a Boolean value",
                    affected_option_ids=option_ids,
                    material_to_decision=True,
                ))
        for constraint in sorted(problem.constraints, key=lambda item: item.id):
            if constraint.kind == "hard" and constraint.id != "capacity":
                unknowns.append(Unknown(
                    description=f"Unsupported hard constraint: {constraint.id}",
                    affected_option_ids=option_ids,
                    material_to_decision=True,
                ))

        attributes: dict[str, RoutingAttributes] = {}
        for option in sorted(problem.options, key=lambda item: item.id):
            try:
                attributes[option.id] = RoutingAttributes.model_validate(
                    option.attributes
                )
            except ValidationError:
                unknowns.append(Unknown(
                    description=f"Invalid routing attributes for option {option.id}",
                    affected_option_ids=[option.id],
                    material_to_decision=True,
                ))
        if unknowns:
            result = Recommendation(
                status="abstained",
                explanation="Required facts, attributes, or constraints are unresolved",
                unknowns=unknowns,
                requested_review=True,
                assessments=[OptionAssessment(
                    option_id=option_id,
                    eligibility="unknown",
                    reasons=["Cannot establish eligibility with unresolved inputs"],
                ) for option_id in option_ids],
            )
            problem.validate_recommendation(result)
            return result

        facts = RoutingFacts.model_validate({
            "urgent": facts_by_id["urgent"].value,
            "specialist_required": facts_by_id["specialist_required"].value,
        })
        route = (
            "specialist" if facts.specialist_required
            else "expedited" if facts.urgent else "routine"
        )
        assessments: list[OptionAssessment] = []
        eligible: list[str] = []
        constraint_ids = [c.id for c in problem.constraints if c.id == "capacity"]
        for option_id in option_ids:
            attrs = attributes[option_id]
            matches = attrs.route == route
            has_capacity = attrs.capacity > 0
            reasons = [
                f"Required route is {route}; option route is {attrs.route}",
                f"Available capacity is {attrs.capacity}",
            ]
            if matches and has_capacity:
                eligible.append(option_id)
            assessments.append(OptionAssessment(
                option_id=option_id,
                eligibility="eligible" if matches and has_capacity else "ineligible",
                reasons=reasons,
                evidence_ids=["urgent", "specialist_required"],
                constraint_ids=constraint_ids,
            ))
        selected = eligible[0] if len(eligible) == 1 else None
        explanation = (
            f"Selected the only {route} queue with available capacity"
            if selected is not None else
            "Multiple eligible queues; no tie-breaking preference is defined"
            if eligible else "No matching queue has available capacity"
        )
        result = Recommendation(
            status="selected" if selected is not None else "abstained",
            selected_option_id=selected,
            assessments=assessments,
            explanation=explanation,
            requested_review=selected is None,
        )
        problem.validate_recommendation(result)
        return result
