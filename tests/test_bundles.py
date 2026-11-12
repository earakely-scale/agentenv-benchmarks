"""Every benchmark's bundle parses as agent-env reads it, with a description and the framework it needs, and is
registered as an entry point, so ``agent-env run <name>`` runs it once the package is installed."""
from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import pytest
from agent_env.bundle.parse import parse_bundle

import agentenv_benchmarks.bundles

BUNDLES = sorted(path for path in Path(agentenv_benchmarks.bundles.__file__).parent.iterdir()
                 if path.is_dir() and not path.name.startswith(("_", ".")))


def test_there_is_a_bundle():
    assert [path.name for path in BUNDLES] == ["online-mind2web"]


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda path: path.name)
def test_each_bundle_parses_with_a_description_and_the_framework_it_needs(bundle):
    parsed = parse_bundle(bundle)

    assert parsed.description and parsed.requires and parsed.requires[0].startswith("agentenv-framework>=")


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda path: path.name)
def test_each_bundle_is_registered_as_an_entry_point(bundle):
    registered = {point.name: point.value for point in entry_points(group="agent_env.bundles")}

    assert registered.get(bundle.name) == "agentenv_benchmarks.bundles"
