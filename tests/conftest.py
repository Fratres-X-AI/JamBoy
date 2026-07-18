"""Pytest configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gpu", action="store_true", help="Enable GPU-marked tests")
    parser.addoption("--runslow", action="store_true", help="Run slow stress tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA")
    config.addinivalue_line("markers", "slow: long-running stress test")


@pytest.fixture
def use_gpu(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--gpu"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
