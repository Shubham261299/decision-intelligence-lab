from pathlib import Path

import pytest

from decisionlab.evaluation.benchmark import Benchmark, load_benchmark

BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "evals"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live", action="store_true", default=False,
                     help="Opt in to paid API integration tests; environment credentials required")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-live"):
        for item in items:
            if item.get_closest_marker("live") is not None:
                item.add_marker(pytest.mark.skip(reason="Live tests require explicit --run-live"))


@pytest.fixture(autouse=True)
def block_live_http_in_unit_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live") is not None:
        return

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Live HTTP is disabled in normal tests")

    monkeypatch.setattr("decisionlab.providers.openai.urlopen", blocked)
    monkeypatch.setattr("decisionlab.providers.ollama.build_opener", blocked)


@pytest.fixture
def benchmark() -> Benchmark:
    return load_benchmark(BENCHMARK_ROOT)
