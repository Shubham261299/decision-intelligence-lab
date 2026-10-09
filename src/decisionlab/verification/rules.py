"""Deterministic routing checks; no selection, generation or benchmark scoring."""

from pydantic import ValidationError

from decisionlab.models import DecisionProblem, Recommendation
from decisionlab.policy import POLICY_VERSION, enforce_policy, selection_violations
from decisionlab.verification.base import RecommendationInput
from decisionlab.verification.models import VerificationFinding, VerificationResult


class RuleBasedVerifier:
    name = "rules-verifier"
    version = "1"

    def verify(
        self, problem: DecisionProblem, recommendation: RecommendationInput,
    ) -> VerificationResult:
        # Revalidate snapshots: Pydantic instances can be mutated or constructed
        # without validation. Neither supplied object is modified.
        problem = DecisionProblem.model_validate(problem.model_dump())
        try:
            raw = (recommendation.model_dump(warnings=False) if isinstance(recommendation, Recommendation)
                   else dict(recommendation))
            rec = Recommendation.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            return VerificationResult(
                verifier=self.name, verifier_version=self.version,
                policy_version=POLICY_VERSION, problem_id=problem.id,
                disposition="reject", structurally_valid=False,
                findings=[VerificationFinding(code="invalid_schema", kind="invalid_decision")],
            )
        findings: list[VerificationFinding] = []
        options = {o.id for o in problem.options}
        facts = {f.id for f in problem.facts}
        constraints = {c.id for c in problem.constraints}
        selected_valid = None if rec.status == "abstained" else rec.selected_option_id in options

        def bad_reference(code: str, path: str, refs: list[str], known: set[str]) -> None:
            for ref in sorted(set(refs) - known):
                findings.append(VerificationFinding(
                    code=code, kind="invalid_decision", path=path,
                    option_id=ref if "option" in code else None,
                    evidence_ids=[ref] if code == "unknown_evidence_reference" else [],
                    constraint_ids=[ref] if code == "unknown_constraint_reference" else [],
                ))

        if rec.selected_option_id is not None:
            bad_reference("unknown_selected_option", "selected_option_id", [rec.selected_option_id], options)
        for index, assessment in enumerate(rec.assessments):
            path = f"assessments.{index}"
            bad_reference("unknown_assessment_option", path + ".option_id", [assessment.option_id], options)
            bad_reference("unknown_evidence_reference", path + ".evidence_ids", assessment.evidence_ids, facts)
            bad_reference("unknown_constraint_reference", path + ".constraint_ids", assessment.constraint_ids, constraints)
        for index, unknown in enumerate(rec.unknowns):
            bad_reference("unknown_affected_option", f"unknowns.{index}.affected_option_ids", unknown.affected_option_ids, options)
        references_valid = not findings
        evidence_valid = not any(f.code == "unknown_evidence_reference" for f in findings)
        violations = selection_violations(problem, rec)
        for code in violations:
            fact_id = code.split(":", 1)[1] if code.startswith("unresolved_fact:") else None
            constraint_ids = (
                ["capacity"] if code == "capacity_exhausted" and "capacity" in constraints
                else ["approval"] if code == "approval_not_granted" and "approval" in constraints
                else [code.split(":", 1)[1]] if code.startswith("unsupported_hard_constraint:")
                else []
            )
            findings.append(VerificationFinding(
                code=code, kind="policy_violation", option_id=rec.selected_option_id,
                evidence_ids=(
                    sorted({"urgent", "specialist_required"} & facts)
                    if code == "wrong_route"
                    else ["approval_granted"]
                    if code == "approval_not_granted" and "approval_granted" in facts
                    else [fact_id] if fact_id is not None and fact_id in facts else []
                ),
                constraint_ids=constraint_ids,
            ))
        evidence_sufficiency = "not_applicable"
        if rec.status == "selected":
            selected = next((a for a in rec.assessments if a.option_id == rec.selected_option_id), None)
            required_facts = {"urgent", "specialist_required"}
            hard = {c.id for c in problem.constraints if c.kind == "hard"}
            if "approval" in hard:
                required_facts.add("approval_granted")
            cited_facts = set(selected.evidence_ids) if selected is not None else set()
            cited_constraints = set(selected.constraint_ids) if selected is not None else set()
            missing_facts = sorted(required_facts - cited_facts)
            missing_constraints = sorted(hard - cited_constraints)
            if selected is None or missing_facts or missing_constraints or selected.eligibility == "unknown":
                evidence_sufficiency = "insufficient"
                findings.append(VerificationFinding(
                    code="incomplete_selection_traceability", kind="insufficient_evidence",
                    option_id=rec.selected_option_id, evidence_ids=missing_facts,
                    constraint_ids=missing_constraints,
                ))
            else:
                evidence_sufficiency = "sufficient"
            if selected is not None and selected.eligibility == "ineligible":
                findings.append(VerificationFinding(
                    code="selection_assessment_conflict", kind="invalid_decision",
                    option_id=rec.selected_option_id,
                ))
        review = evidence_sufficiency == "insufficient" or bool(violations)
        if references_valid:
            policy = enforce_policy(problem, rec)
            review = review or policy.requires_human_review
            findings.extend(VerificationFinding(
                code=code, kind="human_review",
                evidence_ids=["review_required"] if code in (
                    "explicit_supervisor_review", "unresolved_review_requirement",
                ) else [],
            ) for code in policy.review_reasons)
        else:
            review = review or rec.requested_review or rec.status == "abstained"
        reject = bool(violations) or any(f.kind == "invalid_decision" for f in findings)
        return VerificationResult.model_validate({
            "verifier": self.name, "verifier_version": self.version,
            "policy_version": POLICY_VERSION, "problem_id": problem.id,
            "disposition": "reject" if reject else "human_review" if review else "accept",
            "structurally_valid": True, "selected_option_valid": selected_valid,
            "references_valid": references_valid, "evidence_references_valid": evidence_valid,
            "evidence_sufficiency": evidence_sufficiency, "requires_human_review": review,
            "policy_violations": violations, "findings": findings,
        })
