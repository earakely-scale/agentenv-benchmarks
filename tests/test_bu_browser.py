"""bu-browser: its example task plays as an OpenEnv episode scored by its own verifier, its key can only be a secret
reference, its env type stops the browser as it closes, and its site starts, serves, scrubs, resets and stops a Browser
Use Cloud browser, here a fake Browser Use API and a fake Playwright MCP, so no key, network or Docker is needed."""
from __future__ import annotations

import asyncio
import contextlib
import getpass
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import tomllib
from collections.abc import Awaitable
from pathlib import Path
from unittest.mock import MagicMock

import agent_env
import httpx
import pytest
import uvicorn
from agent_env.env.env import DeployedEnv, Env
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv, live_problems
from agent_env.env.registry import get_env_registry
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.task import EpisodeSpec, Task, teardown
from agent_env.task.episode.rules import works_in_sandbox
from agent_env.utils.card_naming import website_source_from_source
from agentenv_protocol import HIDDEN_BY_DEFAULT_META_KEY
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import agentenv_benchmarks.bundles
from agentenv_benchmarks.bu_bench import cloud as bu_bench_cloud
from agentenv_benchmarks.bu_browser import env as bu_env

BUNDLE = Path(agentenv_benchmarks.bundles.__file__).parent / "bu-browser"
SITE = BUNDLE / "envs" / "bu-browser-site"
TASK = BUNDLE / "tasks" / "pypi-browser-use-wheel-sha256.json"
FAKE_PLAYWRIGHT_MCP = Path(__file__).with_name("fake_playwright_mcp.py")

# The site's modules import each other by name, as the image runs them from one folder; no __pycache__ is left in the
# folder, which is the image's build context.
sys.path.insert(0, str(SITE))
no_bytecode_before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
import browser_site
import cloud as site_cloud
import playwright_mcp

sys.path.remove(str(SITE))
sys.dont_write_bytecode = no_bytecode_before

MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
READY = "/agentenv/ext/website/ready"
TOOLS = [{"name": name, "description": name, "inputSchema": schema, "annotations": {"readOnlyHint": False}}
         for name, schema in (
             ("browser_navigate", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}),
             ("browser_environment", {"type": "object", "properties": {}}),
             ("browser_evaluate", {"type": "object", "properties": {"function": {"type": "string"}}}),
             ("browser_click", {"type": "object", "properties": {"ref": {"type": "string"}}}),
             ("browser_crash", {"type": "object", "properties": {}}),
             ("browser_hang", {"type": "object", "properties": {}}),
             ("browser_take_screenshot", {"type": "object", "properties": {}}),
             ("browser_close", {"type": "object", "properties": {}}),
             ("browser_run_code_unsafe", {"type": "object", "properties": {"code": {"type": "string"}}}))]
HASH = "3721b75f6c501523694ef58cd7a3911c032cdabddbb38bca02709a249465c68f"


def test_the_example_task_is_an_episode_on_the_browser_alone_scored_by_its_verifier():
    task = Task.from_dict({"id": "pypi", "version": None, "steps": json.loads(TASK.read_text())})
    spec = EpisodeSpec.from_task(task)
    [deploy] = [step for step in task.steps if step.type == "deploy_agent"]
    access = [(step.env_id, step.action, step.role, step.tools) for step in task.steps
              if step.type == "modify_env_tool_access"]
    reward, problem = _reward_of()(task, None)

    assert spec.trainer_agents == (deploy.agent_name,)
    assert deploy.env_ids == ["bu-browser"] and not works_in_sandbox(task.steps, deploy.agent_name)
    assert access == [("bu-browser", "disable", "default", ["list_website_urls"])]
    assert problem is None and reward({"grade": 1.0}) == 1.0


@pytest.mark.parametrize("reply, scored", [(HASH, True), (f" {HASH.upper()}\n", True),
                                           (f"The hash is {HASH}.", False),
                                           (f"{HASH} or b1e0d7a9a8e1f2c3d4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7", False)],
                         ids=["the hash", "the hash spaced and upper case", "the hash in a sentence", "two hashes"])
def test_the_example_task_scores_only_a_reply_that_is_the_hash(reply, scored):
    task = Task.from_dict({"id": "pypi", "version": None, "steps": json.loads(TASK.read_text())})
    [grade] = [step for step in task.steps if step.type == "agent_prompt_response_verifier"]

    assert [grade._eval_criterion(0, reply, criterion)["passed"] for criterion in grade.criteria] == [scored]


def test_the_bundle_runs_no_bu_bench_step_or_task():
    steps = [step for path in (BUNDLE / "tasks").glob("*.json") for step in json.loads(path.read_text())]

    assert {step["type"] for step in steps}.isdisjoint({"bu_cloud_browser", "bu_findings_verifier"})
    assert not any(path.name.startswith("bu2-") for path in (BUNDLE / "tasks").iterdir())


def test_the_key_is_a_secret_setting_so_only_a_reference_to_it_is_accepted():
    source = website_source_from_source(str(SITE / "Dockerfile"), str(SITE))
    declared = tomllib.loads((SITE / "env.toml").read_text())
    domains = declared["domains"]

    assert source.secret_settings == ["api_key"]
    assert declared["settings"]["api_key"] == "secret:BROWSER_USE_API_KEY"
    assert live_problems(False, domains, [], declared["settings"], source.secret_settings) == []
    assert live_problems(False, domains, [], {**declared["settings"], "api_key": "bu_0123456789"},
                         source.secret_settings)


def test_the_bu_browser_type_is_this_packages_multi_env():
    assert get_env_registry()["bu_browser"] is bu_env.BuBrowserEnv
    assert issubclass(bu_env.BuBrowserEnv, MultiEnv)


def test_the_sites_cloud_client_is_bu_benchs_byte_for_byte():
    assert (SITE / "cloud.py").read_bytes() == Path(bu_bench_cloud.__file__).read_bytes()


@pytest.mark.parametrize("stop_fails", [False, True], ids=["stopped", "stop fails"])
def test_closing_stops_the_sites_browser_for_good_then_closes_the_env(monkeypatch, stop_fails):
    calls, deployed = _closing(monkeypatch, answer=RuntimeError("gateway gone") if stop_fails else None)
    env = _env()
    env._deployed = deployed

    asyncio.run(env.close())

    assert calls == [("invoke", browser_site.CLOSE_URI, "close", "bu-browser-site"), ("close",)]
    # The extension call waits for the site's slowest stop: every request timing out, with the waits between them.
    slowest = site_cloud.STOP_ATTEMPTS * site_cloud.STOP_REQUEST_S + site_cloud.STOP_RETRY_S * sum(
        range(1, site_cloud.STOP_ATTEMPTS))
    assert deployed.timeouts == [int(bu_env.STOP_TIMEOUT_S)] and deployed.timeouts[0] >= slowest


def test_closing_gives_up_on_a_stop_that_hangs_and_still_closes(monkeypatch):
    calls, deployed = _closing(monkeypatch, answer=asyncio.sleep(60))
    monkeypatch.setattr(bu_env, "STOP_TIMEOUT_S", 0.05)
    env = _env()
    env._deployed = deployed

    asyncio.run(env.close())

    assert calls == [("invoke", browser_site.CLOSE_URI, "close", "bu-browser-site"), ("close",)]


def test_a_deployed_env_is_not_deployed_again_until_closed(monkeypatch):
    """A second deploy of one env object would leave close() reaching only the new site, and AgentEnv would end the
    first deployment's sandbox without its close: its browser would bill to its timeout. deploy_env still sees the
    arguments MultiEnv's deploy takes."""
    from agent_env.task_step.task_steps.deploy_env import _takes

    deploys = []

    async def deploy(self, **kwargs):
        deploys.append(kwargs)
        self._env_provider = object()

    monkeypatch.setattr(MultiEnv, "deploy", deploy)
    calls, deployed = _closing(monkeypatch)
    env = _env()

    asyncio.run(env.deploy(ttl_seconds=60))
    env._deployed = deployed
    with pytest.raises(RuntimeError, match="already deployed"):
        asyncio.run(env.deploy(ttl_seconds=60))

    assert deploys == [{"ttl_seconds": 60}] and calls == []
    assert _takes(env.deploy, "child_envs") and not _takes(env.deploy, "env_params")


def test_playwright_mcp_runs_as_pw_with_no_groups_and_its_browser_off_its_command_line():
    """Not as the site's root, which holds the Browser Use key in its settings; and the browser's CDP address, a
    credential, not where any user can read it (ps, /proc/<pid>/cmdline): in its environment."""
    command = playwright_mcp.command(9000, Path("/home/pw/playwright-output/x"))

    assert command[:4] == ["setpriv", "--reuid=pw", "--regid=pw", "--clear-groups"]
    assert "--cdp-endpoint" not in command
    assert playwright_mcp.environment("ws://browser")["PLAYWRIGHT_MCP_CDP_ENDPOINT"] == "ws://browser"


def test_closing_an_env_never_deployed_only_closes(monkeypatch):
    calls, _ = _closing(monkeypatch)

    asyncio.run(_env().close())

    assert calls == [("close",)]


def test_the_site_starts_its_browser_at_the_first_call_serves_its_tools_and_stops_it_on_reset_and_shutdown(
        site, browser_use):
    with TestClient(site.create_app()) as client:
        assert _ready(client)["ready"] is True
        created_when_ready = list(browser_use.created)
        mcp = _mcp(client)
        listed = {tool["name"]: bool((tool.get("_meta") or {}).get(HIDDEN_BY_DEFAULT_META_KEY))
                  for tool in _rpc(client, mcp, "tools/list")["tools"]}
        navigated = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))
        [(first, asked)] = browser_use.created
        environment = json.loads(_text(_call(client, mcp, "browser_environment")))
        reset = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        stopped_by_reset, running_after_reset = list(browser_use.stopped), site.playwright.running
        again = _text(_call(client, mcp, "browser_navigate", url="https://x/"))
        [_, (second, _)] = browser_use.created

    # No browser before a call: none runs, or counts down its timeout, before the run records the env.
    assert created_when_ready == []
    assert browser_use.keys == {"bu-test-key"}
    assert asked == {"proxyCountryCode": "us", "timeout": 30, "enableRecording": False}
    # browser_evaluate is listed hidden, which the gateway hides until tool access names it; browser_close, which can
    # end the cloud browser, and run_code_unsafe aren't listed.
    assert listed == {"browser_navigate": False, "browser_environment": False, "browser_evaluate": True,
                      "browser_click": False, "browser_crash": False, "browser_hang": False,
                      "browser_take_screenshot": False}
    assert navigated.startswith("navigated to https://pypi.org/ over <cdp>") and first not in navigated
    assert "cdp.test" not in navigated
    assert {"AGENTENV_WEBSITE_SETTINGS", "BROWSER_USE_API_KEY"}.isdisjoint(environment) and "USER" in environment
    assert reset.json().get("result") == {} and stopped_by_reset == [first] and not running_after_reset
    assert again.startswith("navigated to https://x/ over <cdp>")
    assert browser_use.stopped == [first, second] and not site.playwright.running


@pytest.mark.parametrize("how", ["discovery 404", "discovery 502, API stopped"])
def test_a_browser_browser_use_ended_is_said_to_have_ended_and_the_next_call_gets_a_fresh_one(site, browser_use, how):
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        [(first, _)] = browser_use.created
        failed_while_running = _call(client, mcp, "browser_click", ref="e1")
        stopped_while_running = list(browser_use.stopped)
        # its timeout_minutes ran out, say
        (browser_use.ended if how == "discovery 404" else browser_use.stopped_by_browser_use).add(first)
        failed_once_ended = _call(client, mcp, "browser_click", ref="e1")
        stopped_once_ended = list(browser_use.stopped)
        again = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))
        [_, (second, _)] = browser_use.created

    # A tool's own error on a running browser is the agent's to handle, and the browser stays.
    assert failed_while_running["isError"] and "has been closed (clicking e1)" in failed_while_running["content"][0]["text"]
    assert stopped_while_running == []
    assert failed_once_ended["isError"] and "the browser ended during browser_click" in failed_once_ended["content"][0]["text"]
    assert stopped_once_ended == [first]
    assert again.startswith("navigated to https://pypi.org/ over <cdp>") and second != first


@pytest.mark.parametrize("answer", [503, "hang"], ids=["CDP discovery 503", "CDP discovery timeout"])
def test_a_browser_whose_check_says_nothing_stays_and_the_call_gets_its_own_error(site, browser_use, monkeypatch,
                                                                                  answer):
    monkeypatch.setattr(playwright_mcp, "PROBE_S", 0.5)
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        browser_use.probe = answer
        failed = _call(client, mcp, "browser_click", ref="e1")
        stopped = list(browser_use.stopped)
        again = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))

    # Neither a 5xx nor a probe that times out says Browser Use ended the browser: it stays, and the agent gets the
    # tool's own error.
    assert failed["isError"] and "has been closed (clicking e1)" in failed["content"][0]["text"]
    assert stopped == [] and len(browser_use.created) == 1 and again.startswith("navigated to https://pypi.org/")


def test_a_browser_whose_cdp_discovery_refuses_connections_has_ended(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(playwright_mcp, "PROBE_RETRY_S", 0.0)
    gone = SimpleNamespace(cdp_url=f"http://127.0.0.1:{_free_port()}")  # nothing listens there

    assert asyncio.run(playwright_mcp._ended(gone)) is True



def test_one_refused_connection_is_not_enough_to_call_the_browser_ended(monkeypatch):
    """A connection refused once and then answered is a blip: the browser is left running."""
    from types import SimpleNamespace

    answers = iter([ConnectionRefusedError(), httpx.Response(200, json={})])
    real = httpx.AsyncClient

    def reply(request):
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise httpx.ConnectError("refused", request=request) from answer
        return answer

    monkeypatch.setattr(playwright_mcp, "PROBE_RETRY_S", 0.0)
    monkeypatch.setattr(playwright_mcp.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(reply), **kw))

    assert asyncio.run(playwright_mcp._ended(SimpleNamespace(cdp_url="http://browser.invalid"))) is False

def test_the_first_result_on_the_browser_that_replaced_one_playwright_mcp_lost_says_the_pages_are_gone(site, browser_use):
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        before = _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        _call(client, mcp, "browser_crash")
        time.sleep(2)  # it exits 0.2 s after it answers
        replaced = _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        after = _call(client, mcp, "browser_navigate", url="https://pypi.org/")

    assert [part["text"].split("\n")[0] for part in replaced["content"]] == [
        playwright_mcp.REPLACED, "navigated to https://pypi.org/ over <cdp>"]
    assert len(before["content"]) == len(after["content"]) == 1 and len(browser_use.created) == 2


def test_an_actions_result_carries_the_page_it_left_not_a_link_to_a_file_the_model_cant_open(site, browser_use,
                                                                                            tmp_path):
    """Playwright MCP writes each action's page snapshot to a file and answers with a link; the model has no file tool."""
    with TestClient(site.create_app()) as client:
        navigated = _text(_call(client, _mcp(client), "browser_navigate", url="https://pypi.org/"))

    assert 'heading "Snapshot of the page"' in navigated and "[Snapshot](" not in navigated
    # The page's text is scrubbed of the CDP address too, and its file is gone.
    assert "cdp.test" not in navigated and navigated.count("<cdp>") == 2
    assert list((tmp_path / "output").iterdir()) == []


def test_a_file_one_task_saved_is_gone_before_a_later_tasks_browser_can_upload_it(site, browser_use, tmp_path):
    """Playwright MCP keeps screenshots, PDFs and downloads in its output folder, and browser_file_upload sends any file
    it can read: each Playwright MCP gets a folder of its own, deleted once it has exited."""
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        saved = Path(_text(_call(client, mcp, "browser_take_screenshot")))
        kept_until_reset = saved.is_file()
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        kept_after_reset = saved.exists() or saved.parent.exists()
        later = Path(_text(_call(client, mcp, "browser_take_screenshot")))

    assert kept_until_reset and not kept_after_reset and later.parent != saved.parent
    assert list((tmp_path / "output").iterdir()) == []


def test_a_snapshot_link_that_is_a_symlink_or_fifo_is_left_as_a_link_and_not_read(monkeypatch, tmp_path):
    """OUTPUT is the pw user's: the root site must not inline what a link there points at (its settings hold the key),
    nor wait on a FIFO."""
    from mcp.types import CallToolResult, TextContent

    output, secret = tmp_path / "output", tmp_path / "settings.json"
    output.mkdir()
    secret.write_text("bu-test-key")
    (output / "page-1.yml").symlink_to(secret)
    os.mkfifo(output / "page-2.yml")
    text = "- [Snapshot](./page-1.yml)\n- [Snapshot](./page-2.yml)"

    inlined = playwright_mcp._inlined(CallToolResult(content=[TextContent(type="text", text=text)]), output)

    assert inlined.content[0].text == text and sorted(p.name for p in output.iterdir()) == ["page-1.yml", "page-2.yml"]


def test_a_reset_sends_the_browsers_stop_before_waiting_on_playwright_mcp_to_exit(site, browser_use, monkeypatch):
    monkeypatch.setattr(playwright_mcp, "STOP_S", 2.0)
    monkeypatch.setattr(playwright_mcp, "command", lambda port, output: [
        sys.executable, str(FAKE_PLAYWRIGHT_MCP), "--port", str(port), "--ignore-term"])
    with TestClient(site.create_app()) as client:
        _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")
        asked = time.monotonic()
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        answered = time.monotonic()

    [(browser, stopped_at)] = browser_use.stopped_at
    # Playwright MCP ignores SIGTERM, so the reset waits STOP_S for it; the stop was sent before that wait.
    assert answered - asked >= 2.0 and stopped_at - asked < 1.0 and browser == browser_use.created[0][0]


def test_a_browser_whose_stops_failed_during_a_reset_is_stopped_by_close(site, browser_use, monkeypatch):
    """Browser Use's API was down for the reset's three stops: the browser is not forgotten, and close, with the API
    back, stops it."""
    monkeypatch.setattr(site_cloud, "STOP_REQUEST_S", 0.3)
    monkeypatch.setattr(site_cloud, "STOP_RETRY_S", 0.0)
    with TestClient(site.create_app()) as client:
        _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")
        [(browser, _)] = browser_use.created
        browser_use.hanging_stops = site_cloud.STOP_ATTEMPTS
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        unstopped_after_reset = [b.id for b in site_cloud.UNSTOPPED]
        client.portal.call(site.playwright.close)
        stopped_by_close = list(browser_use.stopped)

    assert unstopped_after_reset == [browser]
    assert stopped_by_close == [browser] and site_cloud.UNSTOPPED == []


def test_a_call_playwright_mcp_never_answers_is_refused_and_the_next_gets_a_fresh_browser(site, browser_use,
                                                                                          monkeypatch):
    """Playwright MCP stuck on a call would hold every later call as long: it goes, with its browser, and the call that
    may have acted is not replayed."""
    monkeypatch.setattr(playwright_mcp, "CALL_TIMEOUT_S", 1.0)
    monkeypatch.setattr(playwright_mcp, "STOP_S", 1.0)
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        hung = _call(client, mcp, "browser_hang")
        stopped_by_the_hang = list(browser_use.stopped)
        again = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))

    [(first, _), _] = browser_use.created  # the call after it ran on a second browser
    assert hung["isError"] and "didn't answer browser_hang within 1 s" in hung["content"][0]["text"]
    assert stopped_by_the_hang == [first] and again.startswith("navigated to https://pypi.org/ over <cdp>")


def test_a_browser_that_fails_to_start_is_refused_to_the_call_and_leaves_nothing(site, browser_use):
    browser_use.refuse = True
    with TestClient(site.create_app()) as client:
        ready = _ready(client)
        refused = _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")

    assert ready["ready"] is True
    assert refused["isError"] and "returned 429" in refused["content"][0]["text"]
    assert browser_use.created == [] and not site.playwright.running



def test_an_empty_key_fails_readiness(site, browser_use, monkeypatch):
    monkeypatch.setenv("AGENTENV_WEBSITE_SETTINGS", json.dumps({"api_url": browser_use.url}))
    with TestClient(browser_site.BuBrowser().create_app()) as client:
        ready = _ready(client)

    assert ready["ready"] is False and "api_key setting is empty" in " ".join(ready["problems"])


def test_playwright_mcp_failing_to_start_stops_the_browser_it_was_given(site, browser_use, monkeypatch):
    monkeypatch.setattr(playwright_mcp, "command", lambda port, output: [sys.executable, "-c", "raise SystemExit(3)"])
    with TestClient(site.create_app()) as client:
        refused = _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")

    assert refused["isError"] and "exited at start with 3" in refused["content"][0]["text"]
    assert browser_use.stopped == [browser_use.created[0][0]] and not site.playwright.running


def test_a_reset_while_the_browser_is_still_being_created_stops_it_once_created(site, browser_use):
    browser_use.delay_s = 1.0
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        calling = threading.Thread(target=_call, args=(client, mcp, "browser_navigate"), kwargs={"url": "https://x/"})
        calling.start()
        while not browser_use.asked:
            time.sleep(0.01)
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        stopped_by_reset = list(browser_use.stopped)
        calling.join(10)

    assert stopped_by_reset == [browser_use.created[0][0]] and len(browser_use.created) == 1
    assert browser_use.stopped == stopped_by_reset


def test_a_reset_cancelled_while_a_browser_is_being_created_still_leaves_close_to_stop_it(site, browser_use):
    """The reset's wait for the browser being created is cut short (its caller gave up): that browser's stop must go
    on, and close must wait for it, or Browser Use bills it until its timeout."""
    browser_use.delay_s = 1.0

    async def cut_short_reset():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(site.playwright.stop(), 0.3)

    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        calling = threading.Thread(target=_call, args=(client, mcp, "browser_navigate"), kwargs={"url": "https://x/"})
        calling.start()
        while not browser_use.asked:
            time.sleep(0.01)
        client.portal.call(cut_short_reset)
        client.portal.call(site.playwright.close)
        stopped_when_closed = list(browser_use.stopped)
        calling.join(10)

    assert [browser for browser, _ in browser_use.created] == stopped_when_closed and len(stopped_when_closed) == 1


def test_closing_while_a_call_replaces_a_lost_browser_waits_for_the_old_browsers_stop(site, browser_use, monkeypatch):
    """Playwright MCP died, so the next call stops its browser before starting a fresh one; that stop request hangs and
    is retried as teardown's close cancels the call's start. close must not return before the stop has gone through:
    the container goes next, and the browser with nothing left to stop it."""
    monkeypatch.setattr(site_cloud, "STOP_REQUEST_S", 0.5)
    monkeypatch.setattr(site_cloud, "STOP_RETRY_S", 0.1)
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        _call(client, mcp, "browser_crash")
        time.sleep(2)  # it exits 0.2 s after it answers
        browser_use.hanging_stops = 1
        answers = []
        calling = threading.Thread(target=lambda: answers.append(_call(client, mcp, "browser_navigate",
                                                                       url="https://x/")))
        calling.start()
        while not browser_use.hung:
            time.sleep(0.01)
        time.sleep(0.2)  # the call's start is waiting on that stop
        client.portal.call(site.playwright.close)
        stopped_when_closed = list(browser_use.stopped)
        calling.join(10)

    [(first, _)] = browser_use.created
    assert stopped_when_closed == [first] and not site.playwright.running
    assert answers[0]["isError"] and "reset before browser_navigate" in answers[0]["content"][0]["text"]


def test_teardown_closing_a_deployed_bu_browser_stops_its_sites_browser_for_good(site, browser_use, monkeypatch):
    """agent-env's teardown closes the env a run's record names (``teardown._close``), as every run's teardown does
    before it terminates sandboxes, here with the real site served on loopback behind the record's card. Only Docker
    is stood in for: the record's sandbox, which the env reattaches and terminates last. Browser Use's first two stop
    requests hang past their wait, as a slow API's would: the third still goes through before the sandbox goes."""
    terminated: list[tuple[str, list[str]]] = []

    class Sandbox:
        sandbox_id, mode = "sandbox-1", None

        async def terminate(self):
            terminated.append((self.sandbox_id, list(browser_use.stopped)))

    monkeypatch.setattr(site_cloud, "STOP_REQUEST_S", 0.5)
    monkeypatch.setattr(site_cloud, "STOP_RETRY_S", 0.1)

    async def reattach(self, env, deployed):
        return Sandbox()

    declared = tomllib.loads((BUNDLE / "envs" / "bu-browser" / "env.toml").read_text())
    [site_name] = declared["website_envs"]

    def bu_browser(cls, id, version=None):
        assert (id, version) == (declared["name"], 1)
        website = WebsiteEnv(id=site_name, version=1, server_docker_image_artifact=MagicMock(),
                             environment_name=site_name, domains=["bu-browser.localhost"])
        return bu_env.BuBrowserEnv(id=declared["name"], version=1, mcp_server_envs=[], website_envs=[website],
                                   browser_envs=[])

    monkeypatch.setattr(Env, "get", classmethod(bu_browser))
    monkeypatch.setattr(EnvironmentGatewayProvider, "_reattach", reattach)
    with _served(site.create_app()) as url, httpx.Client(base_url=url, timeout=30) as client:
        _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")
        card = client.get("/.well-known/agent-env.json").json()
        record = DeployedEnv.from_dict({
            "env_id": declared["name"], "env_version": 1, "env_provider_type": "gateway", "gateway_url": url,
            "sandbox_id": "sandbox-1", "sandbox_type": "local_docker",
            "environment_card_url": f"{url}/.well-known/agent-env.json",
            "environment_card": {"name": declared["name"], "protocolVersion": "1.0", "url": "/agentenv",
                                 "children_environments": [{**card, "environment_name": site_name}]}})
        [(browser, _)] = browser_use.created
        running = list(browser_use.stopped)

        browser_use.hanging_stops = 2
        asyncio.run(teardown._close(record, teardown.TERMINATE_TIMEOUT_SECONDS))
        # A call that comes after teardown's stop, before the container is removed, starts no browser that nothing
        # would stop; a reset_env's data/reset, by contrast, leaves the next call free to start one.
        late = _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")

        assert running == [] and browser_use.stopped == [browser] and not site.playwright.running
        assert late["isError"] and "the env has closed" in late["content"][0]["text"]
        assert len(browser_use.created) == 1 and browser_use.hung == 2 and terminated == [("sandbox-1", [browser])]


class FakeBrowserUse:
    """Browser Use's API as cloud.py calls it: create a browser, its CDP discovery, stop it; each call recorded."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.asked = 0
        self.created: list[tuple[str, dict]] = []
        self.stopped: list[str] = []
        self.keys: set[str] = set()
        self.stopped_at: list[tuple[str, float]] = []
        self.ended: set[str] = set()
        #: Browsers Browser Use stopped itself: CDP discovery 502, the API's status "stopped" (seen live, Nov 19).
        self.stopped_by_browser_use: set[str] = set()
        self.refuse = False
        self.delay_s = 0.0
        #: Stop requests still to hang (answered after the client gave up), and how many did.
        self.hanging_stops = 0
        self.hung = 0
        #: What CDP discovery answers, once set: that status, or with "hang" nothing in time.
        self.probe: int | str | None = None

    def app(self) -> Starlette:
        async def create(request):
            self.asked += 1
            self.keys.add(request.headers.get("x-browser-use-api-key"))
            if self.refuse:
                return JSONResponse({"detail": "too many concurrent browsers"}, 429)
            browser = f"b{len(self.created) + 1}-0123456789abcdef"
            self.created.append((browser, await request.json()))
            await asyncio.sleep(self.delay_s)  # created, and answered late: a client that gave up never learns its id
            return JSONResponse({"id": browser, "cdpUrl": f"{self.url}/cdp/{browser}"})

        async def version(request):
            if request.path_params["id"] in self.stopped_by_browser_use:
                return JSONResponse({"detail": "bad gateway"}, 502)  # what Browser Use's CDP gateway answers then
            if request.path_params["id"] in self.ended:
                return JSONResponse({"detail": "not found"}, 404)
            if self.probe == "hang":
                await asyncio.sleep(5)
            elif self.probe:
                return JSONResponse({"detail": "unavailable"}, self.probe)
            return JSONResponse({"webSocketDebuggerUrl": f"ws://cdp.test/devtools/{request.path_params['id']}"})

        async def status(request):
            browser = request.path_params["id"]
            return JSONResponse({"id": browser, "status": "stopped" if browser in self.stopped_by_browser_use
                                 else "active"})

        async def stop(request):
            if self.hanging_stops:
                self.hanging_stops -= 1
                self.hung += 1
                await asyncio.sleep(site_cloud.STOP_REQUEST_S * 2)
                return JSONResponse({"detail": "too slow"}, 504)
            self.stopped.append(request.path_params["id"])
            self.stopped_at.append((request.path_params["id"], time.monotonic()))
            return JSONResponse({"id": request.path_params["id"]})

        return Starlette(routes=[Route("/browsers", create, methods=["POST"]),
                                 Route("/cdp/{id}/json/version", version),
                                 Route("/browsers/{id}", stop, methods=["PATCH"]),
                                 Route("/browsers/{id}", status, methods=["GET"])])


@pytest.fixture
def browser_use():
    fake = FakeBrowserUse("")
    with _served(fake.app()) as fake.url:
        yield fake


@contextlib.contextmanager
def _served(app):
    """``app`` served on a loopback port, lifespan included, until the block ends; yields its URL."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(30)


@pytest.fixture
def site(browser_use, tmp_path, monkeypatch):
    manifest = tmp_path / "playwright-mcp-tools.json"
    manifest.write_text(json.dumps({"version": "test", "tools": TOOLS}))
    monkeypatch.setattr(playwright_mcp, "TOOLS", manifest)
    monkeypatch.setattr(playwright_mcp, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(playwright_mcp, "USER", getpass.getuser())  # who owns each output folder: pw in the image
    monkeypatch.setattr(playwright_mcp, "LOG", tmp_path / "playwright-mcp.log")
    monkeypatch.setattr(playwright_mcp, "command", lambda port, output: [
        sys.executable, str(FAKE_PLAYWRIGHT_MCP), "--port", str(port)])
    monkeypatch.setattr(site_cloud, "UNSTOPPED", [])
    monkeypatch.setattr(site_cloud, "API", site_cloud.API)  # the site sets both from its settings
    monkeypatch.setattr(site_cloud, "api_key", site_cloud.api_key)
    monkeypatch.setenv("AGENTENV_WEBSITE_SETTINGS", json.dumps({"api_key": "bu-test-key", "api_url": browser_use.url}))
    monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-test-key")
    playwright_mcp.manifest.cache_clear()
    yield browser_site.BuBrowser()
    playwright_mcp.manifest.cache_clear()


def _closing(monkeypatch, answer: BaseException | Awaitable | None = None) -> tuple[list[tuple], object]:
    """What closing does, recorded, and a deployment whose extension calls answer ``answer``."""
    calls: list[tuple] = []

    class Deployed:
        def __init__(self) -> None:
            self.timeouts: list[int] = []

        async def invoke(self, uri, method, *, environment_name, timeout=30):  # agent-env's default timeout
            calls.append(("invoke", uri, method, environment_name))
            self.timeouts.append(timeout)
            if isinstance(answer, BaseException):
                raise answer
            if answer is not None:
                await answer

    async def closing(self):
        calls.append(("close",))

    monkeypatch.setattr(MultiEnv, "close", closing)
    return calls, Deployed()


def _env() -> bu_env.BuBrowserEnv:
    site = WebsiteEnv(id="bu-browser-site", version=1, server_docker_image_artifact=MagicMock(),
                      environment_name="bu-browser-site", domains=["bu-browser.localhost"])
    return bu_env.BuBrowserEnv(id="bu-browser", version=1, mcp_server_envs=[], website_envs=[site], browser_envs=[])


def _reward_of():
    """openenv's ``reward_of``, which ``build_app`` scores a task by, read from its file: importing its package needs
    the openenv extra, which this package doesn't install."""
    spec = importlib.util.spec_from_file_location("agentenv_openenv_reward",
                                                  Path(agent_env.__file__).parent / "openenv" / "reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward_of


def _ready(client: TestClient) -> dict:
    """The site's readiness once its boot has said either way."""
    deadline = time.monotonic() + 30
    while True:
        ready = client.get(READY).json()
        if ready["ready"] or any("returned False" not in problem for problem in ready["problems"]):
            return ready
        assert time.monotonic() < deadline, ready
        time.sleep(0.05)


def _mcp(client) -> dict:
    first = client.post("/mcp", headers=MCP_HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    headers = {**MCP_HEADERS, "mcp-session-id": first.headers["mcp-session-id"]}
    client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


def _rpc(client, headers: dict, method: str, params: dict | None = None) -> dict:
    response = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": method,
                                                          "params": params or {}})
    text = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        text = [line[len("data:"):] for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)["result"]


def _call(client, headers: dict, tool: str, **arguments) -> dict:
    return _rpc(client, headers, "tools/call", {"name": tool, "arguments": arguments})


def _text(result: dict) -> str:
    assert not result.get("isError"), result
    return result["content"][0]["text"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
