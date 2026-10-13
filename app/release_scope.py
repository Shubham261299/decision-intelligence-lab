"""Fail-closed presentation policy; independent of decision execution."""

import json
from pathlib import Path


def research_is_withheld(root: Path) -> bool:
    """Only an explicit private-review configuration enables saved findings."""
    try:
        scope = json.loads((root / "software-first-release.json").read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        return True
    return not (
        isinstance(scope, dict)
        and scope.get("release_mode") == "private-research-review"
        and scope.get("research_evidence") == "available_for_local_review"
    )
