"""No provider SDK, model name, or evaluation answer belongs in this interface."""

from typing import Protocol, runtime_checkable

from decisionlab.decision_model import DecisionModel
from decisionlab.models import ExecutionInfo


class DecisionStrategy(DecisionModel, Protocol):
    """Produce a recommendation without executing actions or enforcing policy."""



@runtime_checkable
class ExecutionTrackedStrategy(Protocol):
    """Optional telemetry capability; DecisionStrategy itself stays unchanged."""

    last_execution: ExecutionInfo | None
