"""One fixed information plan and a deterministic evidence-consuming generator."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from decisionlab.errors import DecisionExecutionError
from decisionlab.models import (
    Constraint, DecisionProblem, DomainModel, Fact, OptionAssessment,
    Recommendation, Text, Unknown,
)
from decisionlab.policy import POLICY_VERSION, unresolved_inputs
from decisionlab.tools.models import ToolRequest, ToolResult
from decisionlab.tools.routing import FactLookupInput, RoutingPolicyDocument


class ToolPlanner(Protocol):
    def plan(self, problem: DecisionProblem) -> Sequence[ToolRequest]: ...


class ToolDecisionGenerator(Protocol):
    name: str
    version: str

    def generate(
        self, problem: DecisionProblem, evidence: Sequence[ToolResult],
    ) -> Recommendation: ...


class RoutingToolPlanner:
    def plan(self, problem: DecisionProblem) -> Sequence[ToolRequest]:
        ids = {"urgent", "specialist_required"}
        if any(c.id == "approval" and c.kind == "hard" for c in problem.constraints):
            ids.add("approval_granted")
        if any(f.id == "review_required" for f in problem.facts):
            ids.add("review_required")
        return [
            ToolRequest(tool_name="policy_lookup", inputs={"policy_id": POLICY_VERSION}),
            ToolRequest(tool_name="fact_lookup", inputs={"fact_ids": sorted(ids)}),
        ]


class FactEvidence(DomainModel):
    facts: list[Fact]
    missing_fact_ids: list[Text]


class PolicyEvidence(DomainModel):
    policy: RoutingPolicyDocument
    constraints: list[Constraint]


class RoutingToolGenerator:
    name = "deterministic-routing-tools"
    version = "1"

    def generate(
        self, problem: DecisionProblem, evidence: Sequence[ToolResult],
    ) -> Recommendation:
        try:
            fact_result = next(r for r in evidence if r.request.tool_name == "fact_lookup")
            policy_result = next(r for r in evidence if r.request.tool_name == "policy_lookup")
            if fact_result.output is None or policy_result.output is None:
                raise ValueError("No accepted evidence")
            facts = FactEvidence.model_validate(fact_result.output.data)
            policy = PolicyEvidence.model_validate(policy_result.output.data)
            requested = FactLookupInput.model_validate(fact_result.request.inputs).fact_ids
            expected = sorted((f for f in problem.facts if f.id in requested), key=lambda f: f.id)
            if facts.facts != expected or facts.missing_fact_ids != sorted(set(requested) - {f.id for f in expected}):
                raise ValueError("Fact evidence is not an input snapshot")
            if fact_result.output.source_fact_ids != [f.id for f in expected]:
                raise ValueError("Fact provenance does not match evidence")
            if policy.policy != RoutingPolicyDocument() or policy.constraints != sorted(problem.constraints, key=lambda c: c.id):
                raise ValueError("Policy evidence is unsupported")
        except (ValidationError, ValueError, TypeError, StopIteration):
            raise DecisionExecutionError("tool_invalid_result") from None
        # Decisions use retrieved facts, not unrestricted access to input facts.
        retrieved = problem.model_copy(update={"facts": facts.facts}, deep=True)
        unresolved = unresolved_inputs(retrieved)
        ids = [f.id for f in facts.facts]
        constraint_ids = [c.id for c in policy.constraints if c.kind == "hard"]
        if unresolved:
            return Recommendation(
                status="abstained", explanation="Retrieved information cannot establish routing eligibility",
                requested_review=True,
                unknowns=[Unknown(description=code, material_to_decision=True,
                                  affected_option_ids=[o.id for o in problem.options]) for code in unresolved],
                assessments=[OptionAssessment(
                    option_id=o.id, eligibility="unknown", reasons=unresolved,
                    evidence_ids=ids, constraint_ids=constraint_ids,
                ) for o in sorted(problem.options, key=lambda o: o.id)],
            )
        values = {f.id: f.value for f in facts.facts}
        route = ("specialist" if values["specialist_required"]
                 else "expedited" if values["urgent"] else "routine")
        eligible: list[str] = []
        assessments: list[OptionAssessment] = []
        for option in sorted(problem.options, key=lambda o: o.id):
            capacity = option.attributes.get("capacity")
            matches = (option.attributes.get("route") == route and type(capacity) is int
                       and capacity >= policy.policy.minimum_capacity)
            if matches:
                eligible.append(option.id)
            assessments.append(OptionAssessment(
                option_id=option.id, eligibility="eligible" if matches else "ineligible",
                reasons=[f"Retrieved policy {policy_result.evidence_id} requires {route} with capacity >= {policy.policy.minimum_capacity}"],
                evidence_ids=ids, constraint_ids=constraint_ids,
            ))
        # The supplied policy document specifies an explicit deterministic tie rule.
        selected = eligible[0] if eligible else None
        return Recommendation(
            status="selected" if selected is not None else "abstained",
            selected_option_id=selected, assessments=assessments,
            explanation="Selected the first eligible queue by ID using retrieved facts and routing policy"
                        if selected else "No queue satisfies the retrieved routing policy",
            requested_review=selected is None or values.get("review_required") is True,
        )
