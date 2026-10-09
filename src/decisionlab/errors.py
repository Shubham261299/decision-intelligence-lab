"""Typed execution failures with fixed, non-sensitive public messages."""

from typing import Literal

from decisionlab.models import Recommendation

ErrorCode = Literal[
    "provider_timeout", "provider_connection_error", "provider_api_error",
    "provider_refusal", "provider_incomplete", "provider_malformed_response",
    "invalid_schema", "invalid_references",
    "tool_unknown", "tool_not_allowed", "tool_duplicate_call", "tool_call_limit",
    "tool_invalid_request", "tool_failure", "tool_invalid_result", "tool_session_closed",
    "uncollected_tool_evidence", "decision_generation_failure",
]


class DecisionExecutionError(Exception):
    def __init__(
        self, code: ErrorCode, *, candidate: Recommendation | None = None,
    ) -> None:
        self.code = code
        self.candidate = candidate
        super().__init__(f"Decision execution failed: {code}")
