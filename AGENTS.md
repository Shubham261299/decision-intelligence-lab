# Repository Guidelines

Decision Intelligence Lab (DecisionLab) has typed contracts, rule, direct-text and structured LLM strategies, isolated provider adapters, shared policy, a frozen benchmark, evaluator/CLI, pytest tests, and an optional Streamlit portfolio under `app/`. Configuration lives in `pyproject.toml`; this project is an initialized Git repository. These rules govern future implementation. Do not scaffold application code or install dependencies unless requested.

## Python & Project Structure

- Use `src/decisionlab/` for package code, `tests/` for tests, `evals/` for evaluation cases and runners, and `docs/` for design decisions. Create directories when needed.
- Separate business logic from LLM providers, persistence, and interfaces. Avoid network calls and configuration loading at import time.
- Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Prefer small functions and explicit imports.
- When configuring tooling, declare Python compatibility and dependencies in `pyproject.toml`; use Ruff for formatting/linting and mypy for type checking. Document commands once configured.

## Type Hints & Pydantic Models

- Type all function parameters and return values. Prefer concrete types; justify unavoidable `Any` usage.
- Use Pydantic models for external inputs, settings, and structured LLM responses. Define required fields, constraints, and handling of unknown fields; validate at boundaries.
- Separate domain calculations from validation models. Version persisted schemas when compatibility matters.

## Structured LLM Outputs

- Request schema-constrained outputs where supported, except the deliberate direct-text baseline; validate every accepted response against its Pydantic domain model. Never repair direct-text parsing failures.
- Handle refusals, malformed output, timeouts, and validation failures explicitly. Bound retries; never silently accept invalid output.
- Version prompts; record model identity, prompt/schema versions, and generation settings for reproducibility.

## Testing & Evaluation-First Development

- Use pytest with `tests/test_*.py` files and `test_*` functions. Cover business rules, validation, and failure paths; mock external services in default tests.
- Before changing LLM behavior, add representative evaluation cases, expected outcomes or scoring rubrics, and acceptance criteria. Establish a baseline and compare results after changes.
- Separate live-provider evaluations from deterministic tests; require explicit execution and document cost. Convert discovered failures into regression cases.

## Logging & Secrets

- Use Python logging with structured fields and request/run identifiers; avoid diagnostic `print` calls. Log actionable failures without secrets or sensitive prompt content.
- Load environment variables through centralized, validated settings. Fail clearly on missing required values. Provide `.env.example` placeholders; ignore `.env`, virtual environments, caches, and local evaluation output.

## Documentation & Git Hygiene

- Keep README setup, commands, and configuration instructions current. Document public interfaces and significant architectural decisions.
- Review diffs, keep commits focused, and use imperative subjects. Never commit credentials, private datasets, or generated artifacts.
- Pull requests must explain behavior changes, linked issues where applicable, checks run, and evaluation results for LLM changes. State checks not run and why.
