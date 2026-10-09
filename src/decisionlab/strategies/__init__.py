"""Interchangeable strategies operating on provider-independent contracts."""

from decisionlab.strategies.base import DecisionStrategy
from decisionlab.strategies.direct_llm import DirectLLMStrategy
from decisionlab.strategies.rules import RuleBasedStrategy
from decisionlab.strategies.structured_llm import StructuredLLMStrategy
from decisionlab.strategies.tool_assisted import ToolAssistedStrategy

__all__ = [
    "DecisionStrategy",
    "DirectLLMStrategy",
    "RuleBasedStrategy",
    "StructuredLLMStrategy",
    "ToolAssistedStrategy",
]
