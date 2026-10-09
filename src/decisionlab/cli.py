"""Offline decisions and evaluation by default; live generation is opt-in."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, cast

from pydantic import ValidationError

from decisionlab.models import DecisionProblem, RunMetadata
from decisionlab.evaluation.benchmark import Split, load_benchmark
from decisionlab.evaluation.report import report_json, report_text, write_report
from decisionlab.runner import run_benchmark, run_verified_decision
from decisionlab.strategies.rules import RuleBasedStrategy
from decisionlab.strategies.base import DecisionStrategy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decisionlab")
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide", help="Recommend from one DecisionProblem JSON file")
    decide.add_argument("--input", type=Path, required=True)
    decide.add_argument(
        "--strategy", choices=("rules", "direct-llm", "structured-llm", "tool-assisted"),
        default="rules",
    )
    decide.add_argument("--allow-live", action="store_true", help="Explicitly permit LLM generation (local or cloud)")
    evaluate = commands.add_parser("evaluate", help="Evaluate a strategy on the frozen benchmark")
    evaluate.add_argument("--benchmark", type=Path, default=Path("evals"))
    evaluate.add_argument("--split", choices=("dev", "held_out", "all"), default="all")
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--json", action="store_true", help="Emit full JSON report to stdout")
    evaluate.add_argument(
        "--strategy", choices=("rules", "direct-llm", "structured-llm", "tool-assisted"),
        default="rules",
    )
    evaluate.add_argument("--allow-live", action="store_true", help="Explicitly permit LLM generation (local or cloud)")
    compare = commands.add_parser("compare", help="Run matched strategy evaluations")
    compare.add_argument("--benchmark", type=Path, default=Path("evals"))
    compare.add_argument("--split", choices=("dev", "held_out", "all"), default="all")
    compare.add_argument("--repeats", type=int, default=1)
    compare.add_argument("--strategies", nargs="+", default=["rules"],
                         choices=("rules", "direct-llm", "structured-llm", "tool-assisted"))
    compare.add_argument("--allow-live", action="store_true")
    compare.add_argument("--output", type=Path)
    compare.add_argument("--json", action="store_true")
    robustness = commands.add_parser("robustness", help="Measure equivalent-input decision stability")
    robustness.add_argument("--benchmark", type=Path, default=Path("evals"))
    robustness.add_argument("--fixtures", type=Path, default=Path("evals/robustness"))
    robustness.add_argument("--split", choices=("dev", "held_out", "all"), default="all")
    robustness.add_argument("--repeats", type=int, default=1)
    robustness.add_argument("--strategies", nargs="+", default=["rules"],
                            choices=("rules", "direct-llm", "structured-llm", "tool-assisted"))
    robustness.add_argument("--allow-live", action="store_true")
    robustness.add_argument("--output", type=Path)
    robustness.add_argument("--json", action="store_true")
    for command in (decide, evaluate, compare, robustness):
        command.add_argument("--provider", choices=("openai", "ollama"), default=None)
    preflight = commands.add_parser("preflight", help="Check installed local Ollama model; no generation")
    preflight.add_argument("--provider", choices=("ollama",), default="ollama")
    ablate = commands.add_parser("ablate", help="Matched JSON schema-enforcement ablation")
    ablate.add_argument("--benchmark", type=Path, default=Path("evals"))
    ablate.add_argument("--split", choices=("dev", "held_out"), default="dev")
    ablate.add_argument("--repeats", type=int, default=1)
    ablate.add_argument("--provider", choices=("ollama",), default="ollama")
    ablate.add_argument("--allow-live", action="store_true")
    ablate.add_argument("--output", type=Path)
    ablate.add_argument("--json", action="store_true")
    ablate.set_defaults(strategies=["unconstrained-json", "schema-enforced-json"])
    representation = commands.add_parser(
        "representation", help="Matched labelled-text vs unconstrained JSON with raw diagnostics",
    )
    representation.add_argument("--benchmark", type=Path, default=Path("evals"))
    representation.add_argument("--split", choices=("dev",), default="dev")
    representation.add_argument("--repeats", type=int, default=3)
    representation.add_argument("--provider", choices=("ollama",), default="ollama")
    representation.add_argument("--allow-live", action="store_true")
    representation.add_argument("--output", type=Path, required=True)
    representation.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "decide":
        return _decide(args)
    if args.command == "representation":
        return _representation(args)
    if args.command == "preflight":
        from decisionlab.errors import DecisionExecutionError
        from decisionlab.providers.ollama import OllamaProvider

        try:
            result = OllamaProvider.from_environment().preflight()
        except (ValueError, DecisionExecutionError) as error:
            message = error.code if isinstance(error, DecisionExecutionError) else str(error)
            sys.stderr.write(f"decisionlab: {message}\n")
            return 2
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
        return 0
    if args.command in ("compare", "robustness", "ablate"):
        return _compare(args)
    try:
        if args.repeats < 1:
            raise ValueError("Repeats must be a positive integer")
        if args.strategy == "structured-llm" and not args.allow_live:
            raise ValueError("Structured LLM evaluation requires --allow-live")
        if args.strategy == "direct-llm" and not args.allow_live:
            raise ValueError("Direct LLM evaluation requires --allow-live")
        benchmark = load_benchmark(args.benchmark, cast(Split, args.split))
        strategy, strategy_name, strategy_version = _create_strategy(args.strategy, args.provider)
        report = run_benchmark(
            benchmark, strategy, strategy_name=strategy_name,
            strategy_version=strategy_version, repeats=args.repeats,
        )
        if args.output is not None:
            write_report(report, args.output)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"decisionlab: {error}\n")
        return 2
    sys.stdout.write(report_json(report) if args.json else report_text(report))
    if report.summary.counts["failed_runs"]:
        sys.stderr.write("decisionlab: report contains execution failures\n")
        return 1
    return 0


def _create_strategy(name: str, provider: str | None) -> tuple[DecisionStrategy, str, str]:
    """Share existing strategy construction; offline choices never load settings."""
    if name == "structured-llm":
        from decisionlab.providers.factory import provider_from_environment
        from decisionlab.strategies.structured_llm import StructuredLLMStrategy

        return (
            StructuredLLMStrategy(provider_from_environment(provider)),
            StructuredLLMStrategy.name, StructuredLLMStrategy.version,
        )
    if name == "direct-llm":
        from decisionlab.providers.factory import provider_from_environment
        from decisionlab.strategies.direct_llm import DirectLLMStrategy

        return (
            DirectLLMStrategy(provider_from_environment(provider)),
            DirectLLMStrategy.name, DirectLLMStrategy.version,
        )
    if name == "tool-assisted":
        from decisionlab.strategies.tool_assisted import ToolAssistedStrategy

        strategy = ToolAssistedStrategy()
        return strategy, strategy.name, strategy.version
    return RuleBasedStrategy(), RuleBasedStrategy.name, RuleBasedStrategy.version


def _decide(args: argparse.Namespace) -> int:
    if args.strategy in ("direct-llm", "structured-llm") and not args.allow_live:
        sys.stderr.write("decisionlab: LLM generation requires --allow-live\n")
        return 2
    try:
        problem = DecisionProblem.model_validate_json(
            args.input.read_text(encoding="utf-8-sig"),
        )
    except (OSError, UnicodeError):
        sys.stderr.write("decisionlab: unable to read UTF-8 decision input file\n")
        return 2
    except ValidationError:
        # Pydantic exception text can include user input; never echo it to logs.
        sys.stderr.write("decisionlab: input must be one valid DecisionProblem JSON object\n")
        return 2
    try:
        strategy, name, version = _create_strategy(args.strategy, args.provider)
    except ValueError:
        sys.stderr.write("decisionlab: invalid strategy/provider configuration; see environment settings\n")
        return 2
    observation = run_verified_decision(
        problem, strategy, RunMetadata(strategy=name, strategy_version=version),
        f"{name}:{problem.id}:1",
    )
    sys.stdout.write(observation.model_dump_json(indent=2) + "\n")
    if observation.run.status == "failure":
        sys.stderr.write("decisionlab: decision execution failed; see run.error\n")
        return 1
    if observation.verification_error is not None:
        sys.stderr.write("decisionlab: decision verification failed; see verification_error\n")
        return 1
    if (
        observation.run.policy_result is not None
        and observation.run.policy_result.detected_violations
    ) or (
        observation.verification is not None
        and observation.verification.disposition == "reject"
    ):
        sys.stderr.write("decisionlab: recommendation rejected; see policy and verification findings\n")
        return 1
    return 0


def _compare(args: argparse.Namespace) -> int:
    from decisionlab.comparison import (
        StrategyEntry, comparison_json, comparison_text, run_comparison,
        write_comparison,
    )

    try:
        if args.repeats < 1:
            raise ValueError("Repeats must be a positive integer")
        if len(args.strategies) != len(set(args.strategies)):
            raise ValueError("Specify unique strategies")
        live = args.command == "ablate" or any(
            name in ("direct-llm", "structured-llm") for name in args.strategies
        )
        if live and not args.allow_live:
            raise ValueError("Live comparison requires --allow-live")
        benchmark = load_benchmark(
            args.benchmark, "all" if args.command == "robustness" else cast(Split, args.split),
        )
        entries: list[StrategyEntry] = []
        if live:
            from decisionlab.providers.factory import load_provider_factory
            from decisionlab.strategies.direct_llm import DirectLLMStrategy
            from decisionlab.strategies.structured_llm import StructuredLLMStrategy

            # Read once: both adapters use an identical validated settings snapshot.
            create_provider = load_provider_factory(args.provider)
        for name in args.strategies:
            if args.command == "ablate":
                from decisionlab.strategies.json_ablation import Condition, MatchedJSONStrategy

                strategy = MatchedJSONStrategy(create_provider(), cast(Condition, name))
                entries.append(StrategyEntry(name, strategy.version, strategy, "live"))
            elif name == "rules":
                entries.append(StrategyEntry(
                    name, RuleBasedStrategy.version, RuleBasedStrategy(),
                ))
            elif name == "tool-assisted":
                from decisionlab.strategies.tool_assisted import ToolAssistedStrategy

                entries.append(StrategyEntry(name, ToolAssistedStrategy.version, ToolAssistedStrategy()))
            elif name == "direct-llm":
                entries.append(StrategyEntry(
                    name, DirectLLMStrategy.version,
                    DirectLLMStrategy(create_provider()), "live",
                ))
            else:
                entries.append(StrategyEntry(
                    name, StructuredLLMStrategy.version,
                    StructuredLLMStrategy(create_provider()), "live",
                ))
        if args.command == "robustness":
            from decisionlab.robustness import (
                load_robustness, robustness_json, robustness_text,
                run_robustness, write_robustness,
            )

            dataset = load_robustness(args.fixtures, benchmark, cast(Split, args.split))
            robustness_report = run_robustness(
                dataset, entries, repeats=args.repeats, allow_live=args.allow_live,
            )
            if args.output is not None:
                write_robustness(robustness_report, args.output)
            sys.stdout.write(robustness_json(robustness_report) if args.json
                             else robustness_text(robustness_report))
            if any(c.summary.execution_failure_rate.numerator
                   for c in robustness_report.columns.values()):
                sys.stderr.write("decisionlab: robustness contains execution failures\n")
                return 1
            return 0
        report = run_comparison(
            benchmark, entries, repeats=args.repeats, allow_live=args.allow_live,
        )
        if args.output is not None:
            write_comparison(report, args.output)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"decisionlab: {error}\n")
        return 2
    sys.stdout.write(comparison_json(report) if args.json else comparison_text(report))
    if any(c.report.summary.counts["failed_runs"] for c in report.columns.values()):
        sys.stderr.write("decisionlab: comparison contains execution failures\n")
        return 1
    return 0


def _representation(args: argparse.Namespace) -> int:
    from decisionlab.providers.ollama import OllamaProvider
    from decisionlab.representation_experiment import experiment_text, run_representation_experiment
    from decisionlab.settings import OllamaSettings
    from decisionlab.strategies.representation import (
        REPRESENTATIONS, CaptureLocalTransport, RepresentationStrategy,
    )

    try:
        if not args.allow_live:
            raise ValueError("Representation experiment requires --allow-live")
        if args.repeats < 1:
            raise ValueError("Repeats must be a positive integer")
        benchmark = load_benchmark(args.benchmark, "dev")
        settings = OllamaSettings.from_environment()
        if settings.model != "qwen2.5:14b":
            raise ValueError("This declared experiment requires OLLAMA_MODEL=qwen2.5:14b")
        strategies = {}
        for condition in REPRESENTATIONS:
            capture = CaptureLocalTransport()
            strategies[condition] = RepresentationStrategy(
                OllamaProvider(settings, capture), condition, capture,
            )
        report = run_representation_experiment(
            benchmark, strategies, repeats=args.repeats, allow_live=True, output=args.output,
        )
    except (OSError, ValueError) as error:
        sys.stderr.write(f"decisionlab: {error}\n")
        return 2
    # Raw model text/envelopes remain in local artifacts, never stdout/logs.
    if args.json:
        sys.stdout.write(report.comparison.model_dump_json(indent=2) + "\n")
    else:
        sys.stdout.write(experiment_text(report))
    if any(c.report.summary.counts["failed_runs"] for c in report.comparison.columns.values()):
        sys.stderr.write("decisionlab: representation report contains execution failures\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
