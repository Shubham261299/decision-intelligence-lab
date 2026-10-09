"""Explicit registry and per-decision bounded, fail-closed tool session."""

import hashlib
import json
from collections.abc import Collection, Sequence
from types import MappingProxyType

from pydantic import ValidationError

from decisionlab.errors import DecisionExecutionError, ErrorCode
from decisionlab.models import DecisionProblem
from decisionlab.tools.base import Tool
from decisionlab.tools.models import ToolOutput, ToolRequest, ToolResult
from decisionlab.tools.routing import FactLookupTool, PolicyLookupTool


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool], *, allowlist: Collection[str]) -> None:
        if any(
            not isinstance(t.name, str) or not t.name.strip()
            or not isinstance(t.version, str) or not t.version.strip()
            for t in tools
        ):
            raise ValueError("Tool names and versions must be nonempty strings")
        registered = {tool.name: tool for tool in tools}
        if len(registered) != len(tools) or not registered:
            raise ValueError("Register nonempty, unique tools")
        if not set(allowlist) <= set(registered):
            raise ValueError("Allowlist contains unregistered tools")
        self.tools = MappingProxyType(registered)
        self.allowlist = frozenset(allowlist)

    def session(self, problem: DecisionProblem, *, max_calls: int = 2) -> "ToolSession":
        return ToolSession(self, problem, max_calls=max_calls)


class ToolSession:
    def __init__(self, registry: ToolRegistry, problem: DecisionProblem, *, max_calls: int) -> None:
        if type(max_calls) is not int or not 1 <= max_calls <= 16:
            raise ValueError("Tool call limit must be an integer from 1 through 16")
        self._registry = registry
        self._problem = DecisionProblem.model_validate(problem.model_dump())
        self.max_calls = max_calls
        self.calls = 0
        self.results: list[ToolResult] = []
        self._seen: set[str] = set()
        self._closed = False

    def invoke(self, request: ToolRequest) -> ToolResult:
        request = ToolRequest.model_validate(request.model_dump(warnings=False))
        if self._closed:
            raise DecisionExecutionError("tool_session_closed")
        tool = self._registry.tools.get(request.tool_name)

        def fail(code: ErrorCode) -> ToolResult:
            self.results.append(ToolResult(
                request=request, tool_version=tool.version if tool else None,
                status="failure", error_code=code,
            ))
            self._closed = True
            raise DecisionExecutionError(code)

        if self.calls >= self.max_calls:
            return fail("tool_call_limit")
        self.calls += 1
        if tool is None:
            return fail("tool_unknown")
        if request.tool_name not in self._registry.allowlist:
            return fail("tool_not_allowed")
        key = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        if key in self._seen:
            return fail("tool_duplicate_call")
        self._seen.add(key)
        try:
            supplied = tool.execute(self._problem.model_copy(deep=True), request.model_copy(deep=True))
        except (ValidationError, ValueError):
            return fail("tool_invalid_request")
        except Exception:
            return fail("tool_failure")
        try:
            output = ToolOutput.model_validate(supplied.model_dump(warnings=False))
        except Exception:
            return fail("tool_invalid_result")
        if not set(output.source_fact_ids) <= {f.id for f in self._problem.facts}:
            return fail("tool_invalid_result")
        if not set(output.source_constraint_ids) <= {c.id for c in self._problem.constraints}:
            return fail("tool_invalid_result")
        identity = json.dumps({
            "problem_id": self._problem.id, "request": request.model_dump(mode="json"),
            "tool_version": tool.version, "output": output.model_dump(mode="json"),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result = ToolResult(
            request=request, tool_version=tool.version, status="success", output=output,
            evidence_id=f"tool:{tool.name}:{hashlib.sha256(identity.encode()).hexdigest()}",
        )
        self.results.append(result.model_copy(deep=True))
        return result.model_copy(deep=True)


def routing_tools() -> ToolRegistry:
    return ToolRegistry([FactLookupTool(), PolicyLookupTool()],
                        allowlist=["fact_lookup", "policy_lookup"])
