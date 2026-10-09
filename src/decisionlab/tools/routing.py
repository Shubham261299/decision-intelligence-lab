"""Pure in-memory lookups: no filesystem, network, shell or executable input."""

from pydantic import Field, StrictInt

from decisionlab.models import DecisionProblem, DomainModel, Text, require_unique
from decisionlab.policy import POLICY_VERSION
from decisionlab.tools.models import ToolOutput, ToolRequest


class FactLookupInput(DomainModel):
    fact_ids: list[Text] = Field(min_length=1, max_length=16)


class PolicyLookupInput(DomainModel):
    policy_id: Text


class RoutingPolicyDocument(DomainModel):
    policy_version: Text = POLICY_VERSION
    required_fact_ids: list[Text] = Field(default_factory=lambda: ["urgent", "specialist_required"])
    minimum_capacity: StrictInt = Field(default=1, ge=1)
    specialist_precedence: bool = True
    supported_hard_constraints: list[Text] = Field(default_factory=lambda: ["approval", "capacity"])
    tie_breaker: Text = "lowest_option_id"


class FactLookupTool:
    name = "fact_lookup"
    version = "1"

    def execute(self, problem: DecisionProblem, request: ToolRequest) -> ToolOutput:
        args = FactLookupInput.model_validate(request.inputs)
        require_unique(args.fact_ids, "Requested fact IDs")
        found = sorted((f for f in problem.facts if f.id in args.fact_ids), key=lambda f: f.id)
        ids = [f.id for f in found]
        return ToolOutput(data={
            "facts": [f.model_dump(mode="json") for f in found],
            "missing_fact_ids": sorted(set(args.fact_ids) - set(ids)),
        }, source_fact_ids=ids)


class PolicyLookupTool:
    name = "policy_lookup"
    version = "1"

    def execute(self, problem: DecisionProblem, request: ToolRequest) -> ToolOutput:
        args = PolicyLookupInput.model_validate(request.inputs)
        if args.policy_id != POLICY_VERSION:
            raise ValueError("Unsupported policy")
        constraints = sorted(problem.constraints, key=lambda c: c.id)
        return ToolOutput(data={
            "policy": RoutingPolicyDocument().model_dump(mode="json"),
            "constraints": [c.model_dump(mode="json") for c in constraints],
        }, source_constraint_ids=[c.id for c in constraints])
