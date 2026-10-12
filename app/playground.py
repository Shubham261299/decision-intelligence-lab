"""Offline demo integration using the same execution boundary as the CLI."""

from pathlib import Path

from pydantic import ValidationError

from decisionlab.evaluation.models import Observation
from decisionlab.models import DecisionProblem, RunMetadata
from decisionlab.runner import run_verified_decision
from decisionlab.strategies.rules import RuleBasedStrategy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "Routine routing": "decision-routine.json",
    "Missing urgency": "decision-missing-urgency.json",
}


def example_json(name: str) -> str:
    """Read only an explicitly allowlisted synthetic example."""
    if name not in EXAMPLES:
        raise ValueError("Unknown example")
    try:
        return (ROOT / "examples" / EXAMPLES[name]).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ValueError("Example unavailable; check the local project files") from error


def decide_json(text: str) -> Observation:
    """Validate input and execute rules with policy and verification, never labels."""
    try:
        problem = DecisionProblem.model_validate_json(text)
    except ValidationError as error:
        raise ValueError("Input must be valid DecisionProblem JSON; no decision was run") from error
    strategy = RuleBasedStrategy()
    return run_verified_decision(
        problem, strategy,
        RunMetadata(strategy=strategy.name, strategy_version=strategy.version),
        f"demo:{strategy.name}:{problem.id}:1",
    )
