"""Execution boundary: strategies see isolated inputs, never reference labels."""

import logging

from pydantic import ValidationError

from decisionlab.errors import DecisionExecutionError
from decisionlab.decision_model import DecisionModel

from decisionlab.models import (
    DecisionProblem, DecisionRun, Recommendation, RunError, RunMetadata,
)
from decisionlab.evaluation.models import Observation
from decisionlab.evaluation.benchmark import Benchmark
from decisionlab.evaluation.evaluator import evaluate_observation
from decisionlab.evaluation.report import EvaluationReport, build_report
from decisionlab.policy import enforce_policy
from decisionlab.strategies.base import ExecutionTrackedStrategy
from decisionlab.verification import DecisionVerifier, RuleBasedVerifier
from decisionlab.verification.models import VerificationResult
from decisionlab.tools.base import ToolTrackedStrategy
from decisionlab.tools.models import ToolResult

logger = logging.getLogger(__name__)


def run_decision(
    problem: DecisionProblem, strategy: DecisionModel, metadata: RunMetadata,
    run_id: str,
) -> Observation:
    base: dict[str, object] = {"id": run_id, "problem_id": problem.id, "metadata": metadata}
    tool_results: list[ToolResult] = []

    def capture_execution_info() -> None:
        if isinstance(strategy, ToolTrackedStrategy):
            tool_results[:] = [ToolResult.model_validate(r.model_dump(warnings=False))
                               for r in strategy.last_tool_results]
        if isinstance(strategy, ExecutionTrackedStrategy) and strategy.last_execution is not None:
            info = strategy.last_execution
            recorded = info.metadata.model_dump()
            # Dataset/schema identity comes from the runner, never the provider/model.
            recorded["dataset_version"] = metadata.dataset_version
            recorded["schema_version"] = metadata.schema_version
            base["metadata"] = RunMetadata.model_validate(recorded)
            base["usage"] = info.usage.model_copy(deep=True) if info.usage is not None else None
            base["latency_ms"] = info.latency_ms
            base["response_metadata"] = dict(info.response_metadata)

    def failed(code: str, message: str, **diagnostics: object) -> Observation:
        logger.warning(
            "Decision execution failed",
            extra={"run_id": run_id, "problem_id": problem.id, "failure_code": code},
        )
        run = DecisionRun.model_validate({
            **base, "status": "failure", "error": RunError(code=code, message=message),
        })
        return Observation.model_validate({"run": run, "tool_results": tool_results, **diagnostics})

    try:
        output = strategy.decide(problem.model_copy(deep=True))
    except DecisionExecutionError as error:
        capture_execution_info()
        if error.code == "invalid_references" and error.candidate is not None:
            output = error.candidate
        else:
            return failed(
                error.code, f"Decision execution failed: {error.code}",
                schema_valid=False if error.code == "invalid_schema" else None,
            )
    except Exception as error:
        capture_execution_info()
        # No arbitrary exception payloads, secrets or unstable object reprs in reports.
        return failed("strategy_exception", f"Strategy raised {type(error).__name__}")
    capture_execution_info()
    try:
        raw = output.model_dump() if isinstance(output, Recommendation) else output
        candidate = Recommendation.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return failed("invalid_schema", "Strategy output failed schema validation", schema_valid=False)

    evidence = [ref for a in candidate.assessments for ref in a.evidence_ids]
    known_facts = {fact.id for fact in problem.facts}
    invalid_count = sum(ref not in known_facts for ref in evidence)
    diagnostics = {
        "candidate": candidate, "schema_valid": True,
        "evidence_references_valid": invalid_count == 0,
        "evidence_reference_count": len(evidence),
        "invalid_evidence_reference_count": invalid_count,
    }
    try:
        problem.validate_recommendation(candidate)
    except ValueError:
        return failed(
            "invalid_references", "Strategy output contains unresolved references",
            references_valid=False, **diagnostics,
        )
    policy = enforce_policy(problem, candidate)
    run = DecisionRun.model_validate({
        **base, "status": "success", "recommendation": candidate,
        "policy_result": policy,
    })
    return Observation.model_validate({
        "run": run, "references_valid": True, "tool_results": tool_results, **diagnostics,
    })


def run_verified_decision(
    problem: DecisionProblem, strategy: DecisionModel, metadata: RunMetadata,
    run_id: str, *, verifier: DecisionVerifier | None = None,
) -> Observation:
    """Execute and verify one problem without benchmark loading or scoring."""
    observation = run_decision(problem, strategy, metadata, run_id)
    implementation = verifier if verifier is not None else RuleBasedVerifier()
    candidate = observation.run.recommendation or observation.candidate
    if candidate is not None:
        try:
            verification = implementation.verify(
                problem.model_copy(deep=True), candidate.model_copy(deep=True),
            )
            verified = VerificationResult.model_validate(
                verification.model_dump(warnings=False),
            )
            if verified.problem_id != problem.id:
                raise ValueError("Verification problem ID mismatch")
            observation.verification = verified
        except Exception:
            # Verifier failure cannot change execution outcomes or scores.
            observation.verification_error = "verifier_error"
            logger.warning("Decision verification failed", extra={"run_id": run_id})
    return observation


def run_benchmark(
    benchmark: Benchmark, strategy: DecisionModel, *, strategy_name: str,
    strategy_version: str, repeats: int = 1, verifier: DecisionVerifier | None = None,
) -> EvaluationReport:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    metadata = RunMetadata(
        strategy=strategy_name, strategy_version=strategy_version,
        schema_version=benchmark.manifest.schema_version,
        dataset_version=benchmark.manifest.version,
    )
    implementation = verifier if verifier is not None else RuleBasedVerifier()
    results = []
    for problem in benchmark.problems:
        for repetition in range(1, repeats + 1):
            run_id = f"{strategy_name}:{problem.id}:{repetition}"
            observation = run_verified_decision(
                problem, strategy, metadata, run_id, verifier=implementation,
            )
            results.append(evaluate_observation(
                problem, benchmark.expectations[problem.id], observation, repetition,
            ))
    return build_report(benchmark, results, strategy_name, strategy_version, repeats)
