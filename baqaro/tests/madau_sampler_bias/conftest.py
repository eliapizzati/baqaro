"""Mark every test in this directory ``slow``.

These layers are a Monte-Carlo *investigation*, not regression gates: each draws
1–2 million sampler columns, and a single one runs for minutes. They also collect
FIRST (this directory sorts before ``test_*.py``), so a bare ``pytest`` used to
spend many minutes here before reaching a single gate — measured: 9 tests in
7.5 min, against 149 gates in 3m22s for the whole rest of the suite.

``pyproject.toml`` therefore deselects ``slow`` by default. Run them explicitly:

    pytest -m slow baqaro/tests/madau_sampler_bias

Marking happens here rather than as a decorator on each test so that a new layer
is covered automatically.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Attach the ``slow`` marker to every test collected from this directory."""
    for item in items:
        if "madau_sampler_bias" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
