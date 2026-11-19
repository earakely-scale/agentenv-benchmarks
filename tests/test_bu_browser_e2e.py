"""bu-browser end to end, opt in (``uv run pytest -m docker tests/test_bu_browser_e2e.py``): OpenEnv episodes on the
site's real image, built and run by agent-env's local Docker provider, Playwright MCP 0.0.85 driving a local Chromium,
and a fake Browser Use API, so no key or spend. Each episode's browser starts at its first tool call and is stopped
by the run's teardown, whether the episode ended, was abandoned or never called a tool. Needs Docker, uv and network
for the image build and the openenv package; takes several minutes."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import agentenv_benchmarks.bundles

pytestmark = pytest.mark.docker

HERE = Path(__file__).parent
BUNDLE = Path(agentenv_benchmarks.bundles.__file__).parent / "bu-browser"
CHROMIUM = "chromedp/headless-shell:stable"
OPENENV = "openenv==0.14.0"
HASH = "3721b75f6c501523694ef58cd7a3911c032cdabddbb38bca02709a249465c68f"


def test_each_episode_starts_its_browser_at_its_first_call_and_teardown_stops_it(tmp_path):
    if shutil.which("docker") is None or shutil.which("uv") is None:
        pytest.skip("needs docker and uv")
    bridge = _docker("network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}").strip()
    with socket.socket() as probe:
        if probe.connect_ex((bridge, 9222)) == 0:
            pytest.skip(f"port 9222 on {bridge}, which the local Chromium listens on, is taken")
    port, calls, chromium = _free_port(), tmp_path / "calls.jsonl", f"bu-browser-e2e-{os.getpid()}"
    bundle = tmp_path / "bu-browser"
    shutil.copytree(BUNDLE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
    settings = bundle / "envs" / "bu-browser-site" / "env.toml"
    settings.write_text(settings.read_text() + f'api_url = "http://{bridge}:{port}"\n')
    # Its CDP port, which needs no credential, only on Docker's bridge, where the site's container reaches it.
    _docker("run", "-d", "--rm", "--name", chromium, "-p", f"{bridge}:9222:9222", CHROMIUM)
    fake = subprocess.Popen([sys.executable, str(HERE / "e2e" / "fake_browser_use.py"), str(calls), str(port),
                             f"http://{bridge}:9222"])
    try:
        (tmp_path / "home").mkdir()
        environment = {**os.environ, "HOME": str(tmp_path / "home"), "BROWSER_USE_API_KEY": "e2e-key",
                       "LITELLM_BASE_URL": "http://127.0.0.1:9", "LITELLM_API_KEY": "e2e",
                       "UV_CACHE_DIR": subprocess.run(["uv", "cache", "dir"], capture_output=True, text=True,
                                                      check=True).stdout.strip()}
        episode = subprocess.run(["uv", "run", "--with", OPENENV, "python", str(HERE / "e2e" / "bu_browser_episode.py"),
                                  str(bundle), str(calls)], env=environment, capture_output=True, text=True,
                                 timeout=1800, check=False)
    finally:
        fake.terminate()
        fake.wait(10)
        subprocess.run(["docker", "rm", "-f", chromium], capture_output=True, check=False)
    assert episode.returncode == 0, episode.stderr[-4000:]
    seen = {point["at"]: point for point in map(json.loads, (line for line in episode.stdout.splitlines()
                                                             if line.startswith('{"at"')))}
    keys = {entry.get("key") for entry in map(json.loads, calls.read_text().splitlines())}

    assert seen["episode 1 reset"]["created"] == []  # no browser before the first call
    first = seen["episode 1 ended"]
    # The navigate result holds the page's own text (its URL doesn't), inline, not a link to a snapshot file.
    assert first["saw_page"] and not first["linked"] and first["done"] and first["reward"] == 1.0
    assert "browser_navigate" in first["tools"]
    # browser_evaluate is hidden until a task's tool access names it, which the example task's doesn't.
    assert {"browser_close", "browser_run_code_unsafe", "browser_evaluate"}.isdisjoint(first["tools"])
    assert first["created"] == first["stopped"] == ["e2e1-0123456789abcdef"]
    running = seen["episode 2 running"]
    assert running["stopped"] == ["e2e1-0123456789abcdef"]
    # Playwright MCP runs as pw, not as the site's PID 1 (root), which holds the key.
    assert running["mcp_uids"] and running["pid1_uid"] is not None
    assert running["pid1_uid"] not in running["mcp_uids"]
    # Its browser's CDP address, a credential, is in its environment, not on its command line.
    assert running["mcp_args_name_cdp"] is False
    both = ["e2e1-0123456789abcdef", "e2e2-0123456789abcdef"]
    assert seen["episode 2 abandoned"]["created"] == seen["episode 2 abandoned"]["stopped"] == both
    assert seen["session closed"]["created"] == seen["session closed"]["stopped"] == both
    assert keys == {"e2e-key", None}


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True, timeout=600).stdout


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
