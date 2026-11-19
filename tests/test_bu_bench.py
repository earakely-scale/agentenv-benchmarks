import asyncio
import base64
import dataclasses
import importlib.util
import io
import json
import os
import re
import textwrap
from pathlib import Path

import httpx
import pytest

from agentenv_benchmarks.bu_bench import cloud, evidence, steps, tasks

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src" / "agentenv_benchmarks" / "bundles" / "bu-bench"
TOOLS = ROOT / "tools" / "bu_bench"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tasks_decrypt_the_frozen_benchmark():
    assert len(tasks.task_ids()) == 200
    assert tasks.instruction("bu2-140").strip()


def test_bundle_tasks_carry_no_instruction_text():
    mini = re.findall(r'"(bu2-\d+)"', (BUNDLE / "evals" / "mini-20.toml").read_text())
    assert len(mini) == 20
    full = (BUNDLE / "evals" / "bu-bench-v21.toml").read_text()
    assert sorted(p.stem for p in BUNDLE.glob("tasks/*.json")) == tasks.task_ids()
    assert all(f'"{t}"' in full for t in tasks.task_ids())
    for path in BUNDLE.glob("tasks/*.json"):
        text = path.read_text()
        assert tasks.instruction(path.stem)[:40] not in text
        steps_ = json.loads(text)
        assert [s["type"] for s in steps_] == ["deploy_agent", "bu_cloud_browser", "prompt_agent",
                                                "collect_artifacts", "bu_findings_verifier"]
        assert steps_[2]["prompt"] == "<bu_task>"


def test_steps_round_trip():
    for cls, data in ((steps.BuCloudBrowserTaskStep, {"id": "b", "task_id": "bu2-001"}),
                      (steps.BuFindingsVerifierTaskStep, {"id": "g", "task_id": "bu2-001", "prompt_id": "solve"})):
        step = cls.from_dict(data)
        assert cls.from_dict(step.to_dict()).to_dict() == step.to_dict()


def test_codex_steps_keep_tools_and_messages_and_drop_reasoning():
    events = {"format": "codex-exec-json/1", "events": [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "secret plan"}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "browser-harness <<'PY'",
                                            "aggregated_output": "title: Example"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "FINAL ANSWER: Example"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 10}},
    ]}
    fmt, found, usage = evidence.steps_and_usage(events)
    assert fmt == "codex-exec-json/1"
    assert found == ["$ browser-harness <<'PY'\ntitle: Example", "FINAL ANSWER: Example"]
    assert usage["input_tokens"] == 1000 and usage["cached_input_tokens"] == 600
    assert round(evidence.model_usd(usage), 6) == round((400 * 1 + 600 * 0.1 + 10 * 5) / 1e6, 6)


def test_browser_use_steps_pair_calls_with_results():
    messages = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "task"}]},
        {"kind": "response", "usage": {"input_tokens": 50, "cache_read_tokens": 10, "output_tokens": 5},
         "parts": [{"part_kind": "thinking", "content": "hmm"},
                   {"part_kind": "tool-call", "tool_name": "python", "args": {"code": "1+1"}, "tool_call_id": "c1"}]},
        {"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": "python", "content": "2",
                                       "tool_call_id": "c1"}]},
        {"kind": "response", "usage": {"input_tokens": 60, "cache_read_tokens": 50, "output_tokens": 7},
         "parts": [{"part_kind": "text", "content": "done"}]},
    ]
    fmt, found, usage = evidence.steps_and_usage({"events": messages})
    assert fmt == "browser-use-messages/1"
    assert found == ["thinking: hmm", 'python: {"code": "1+1"}\nresult: 2', "text: done"]
    assert usage == {"input_tokens": 110, "cached_input_tokens": 60, "output_tokens": 12, "model_requests": 2}


def test_images_never_reach_the_judge_as_base64():
    assert evidence.text_of({"type": "image", "data": "A" * 1000}) == "[image shown to the agent]"
    assert "AAAA" not in evidence.text_of(json.dumps({"data": "A" * 1000}))


def test_output_files_open_only_the_judges_share_and_split_its_cap():
    """The judge reads the first 50 files by name, 600 kB of text between them; a file past them is never opened, and a
    big one is read in chunks, whatever its size, into the same head and tail."""
    big = "é" + "x" * 2_000_000 + "€"  # multi-byte characters at both ends and across every chunk boundary
    opened = []

    def opener(name, raw):
        return lambda: opened.append(name) or io.BytesIO(raw)

    named = {"a.csv": big.encode(), "b.bin": b"\xff\xfe", "c.txt": b"ok \xe2\x82",  # c ends inside a character
             **{f"d{n:02}.txt": b"d" for n in range(60)}}
    files = evidence.output_files([(name, opener(name, raw)) for name, raw in reversed(named.items())])

    limit = 600_000 // 50
    assert [f["name"] for f in files] == sorted(named)[:50] == sorted(opened)
    assert files[0] == {"name": "a.csv", "size": len(big.encode()), "clipped": True,
                        "text": big[:limit // 3] + "\n...[middle omitted]...\n" + big[-(limit - limit // 3):]}
    assert files[1]["text"] is None and files[2]["text"] is None and files[2]["size"] == 5
    assert files[3] == {"name": "d00.txt", "size": 1, "text": "d", "clipped": False}


BROWSER_ID = "0f6c1a52-9d0e-4b8a-a1f7-3c2d5e6f7a8b"
WS = f"wss://{BROWSER_ID}.cdp.browser-use.com/devtools/browser/11111111-2222-3333-4444-555555555555"


def test_scrub_removes_every_form_of_the_cdp_address():
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")
    leaked = {"out": f"BU_CDP_WS={WS}", "nested": [f"connect to {BROWSER_ID}.cdp.browser-use.com failed",
                                                     f"id {BROWSER_ID}"]}
    text = json.dumps(cloud.scrub_json(leaked, browser.secrets))
    assert BROWSER_ID not in text and "cdp.browser-use.com" not in text and browser.label == BROWSER_ID[:8]
    assert BROWSER_ID not in repr(browser)


def test_api_errors_do_not_name_the_browser(monkeypatch):
    def reply(request):
        return httpx.Response(404, json={"detail": "nope"}, request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setenv(cloud.KEY_NAME, "k")
    try:
        asyncio.run(cloud.request("GET", f"/browsers/{BROWSER_ID}"))
    except RuntimeError as error:
        assert BROWSER_ID not in str(error) and "404" in str(error)
    else:
        raise AssertionError("expected an error")


def test_screenshot_errors_are_scrubbed(tmp_path):
    shots = cloud.Screenshots("wss://" + BROWSER_ID + ".cdp.browser-use.com/x", tmp_path)

    async def boom(label):
        raise OSError(f"cannot reach {BROWSER_ID}.cdp.browser-use.com")

    shots._capture = boom

    async def capture_often():
        for n in range(50):
            await shots.capture(f"t{n}")

    asyncio.run(capture_often())
    assert shots.status["errors"] and BROWSER_ID not in shots.status["errors"][0]
    assert shots.status["captures"] == 50 and len(shots.status["errors"]) == 20  # the last 20, however long it runs


def test_the_final_frame_is_saved_after_the_capture_reached_its_limit(tmp_path):
    """The judge reads the frames in order: the last must be the page the agent left, however long the task ran."""
    shots = cloud.Screenshots(WS, tmp_path, limit=3)
    frames = iter(range(1000))

    async def send(method, params, session_id=None):
        return {"data": base64.b64encode(f"frame {next(frames)}".encode()).decode()}

    async def page():
        return "page-1"

    shots._ws, shots._sessions, shots._send, shots._page = object(), {"page-1": "s"}, send, page

    async def run():
        for n in range(5):
            await shots.capture(f"t{n}")
        return await shots.stop()

    status = asyncio.run(run())
    assert shots.saved == ["0001-t0.jpg", "0002-t1.jpg", "0003-final.jpg"] and status["saved"] == 3
    assert (tmp_path / "0003-final.jpg").read_bytes() == b"frame 2"


def test_the_capture_ends_at_the_browsers_time_limit(tmp_path):
    shots = cloud.Screenshots(WS, tmp_path, interval=0.01, seconds=0.05)
    taken = []

    async def capture(label):
        taken.append(label)

    shots.capture = capture

    async def run():
        shots.start()
        await asyncio.sleep(0.5)
        return shots._loop.done()

    assert asyncio.run(run()) and 0 < len(taken) < 10


def codex_tools_agent(monkeypatch):
    """The Codex arms' agent module, imported over AgentEnv's own Codex agent sources (the image's base). Those are
    AgentEnv's private layout, not its API: a version that moves or renames them skips these tests, saying so."""
    import sys

    import agent_env

    codex = Path(agent_env.__file__).parent / "agents" / "agents" / "codex"
    if not (codex / "a2a_server.py").is_file():
        pytest.skip(f"this AgentEnv has no Codex agent sources at {codex}, which the Codex arms' image builds on")
    monkeypatch.syspath_prepend(str(codex))
    monkeypatch.syspath_prepend(str(BUNDLE / "artifacts" / "codex-browser-tools"))
    monkeypatch.delitem(sys.modules, "codex_tools_agent", raising=False)
    try:
        return importlib.import_module("codex_tools_agent")
    except ImportError as error:
        ours = {path.stem for path in codex.glob("*.py")}
        if error.name in ours or str(error.name).startswith("agentenv_protocol"):
            pytest.skip(f"this AgentEnv's Codex agent sources lack what the Codex arms' agent imports: {error}")
        raise


def test_every_codex_arm_has_an_agent_on_the_one_image(monkeypatch):
    import tomllib

    agent = codex_tools_agent(monkeypatch)
    tools = {}
    for path in BUNDLE.glob("agents/codex-*/agent.toml"):
        config = tomllib.loads(path.read_text())
        assert config["image"] == "codex-browser-tools", path
        tools[config["default_env_vars"]["BU_BENCH_TOOL"]] = path.parent.name
    assert set(tools) == set(agent.TOOLS) and tools["browser-harness"] == "codex-bu-cli"


def test_mcp_arms_name_the_server_their_prompt_names_and_parse_as_toml(monkeypatch):
    import re
    import tomllib

    agent = codex_tools_agent(monkeypatch)
    monkeypatch.setattr(agent, "BROWSE_HOME", Path("/nonexistent"))
    monkeypatch.setattr(agent, "browse_env", lambda: {"OPENCODE_BROWSER_PORT": "1"})
    for tool in agent.TOOLS:
        overrides = agent.mcp_overrides(agent.mcp_server(tool, WS, "gpt-6.1-terra", "medium"))
        named = re.search(r"the `([a-z_-]+)` MCP tools", agent.compose_prompt("t", tool))
        if named is None:  # a CLI tool: no MCP server
            assert overrides == [], tool
            continue
        config = {}
        for override in overrides:
            key, value = override.split("=", 1)
            config[key] = tomllib.loads(f"v = {value}")["v"]
        assert config[f"mcp_servers.{named.group(1)}.command"], tool


@pytest.mark.parametrize("headers", ['{"x-litellm-customer-id": "c1", "x-team": "t\\"1"}', None], ids=["set", "unset"])
def test_stagehands_model_calls_get_the_endpoints_extra_headers(monkeypatch, headers):
    """Codex hands an MCP server only the environment its config names: AgentEnv's LITELLM_EXTRA_HEADERS reaches
    Stagehand's model client only through it."""
    import tomllib

    agent = codex_tools_agent(monkeypatch)
    if headers is None:
        monkeypatch.delenv("LITELLM_EXTRA_HEADERS", raising=False)
    else:
        monkeypatch.setenv("LITELLM_EXTRA_HEADERS", headers)
    [env] = [tomllib.loads(f"v = {override.split('=', 1)[1]}")["v"]
             for override in agent.mcp_overrides(agent.mcp_server("stagehand", WS, "gpt-6.1-terra", "medium"))
             if override.startswith("mcp_servers.stagehand.env=")]

    assert env.get("STAGEHAND_HEADERS") == headers


def test_codex_turn_keeps_mcp_config_and_scrubs_the_browser(monkeypatch):
    agent = codex_tools_agent(monkeypatch)
    seen = {}

    async def fake_codex(*, prompt, config_overrides, event_callback, **kwargs):
        seen.update(prompt=prompt, overrides=config_overrides)
        event_callback({"type": "item.completed", "item": {"type": "mcp_tool_call", "arguments": {"url": WS}}})
        return {"response": WS}  # not a dataclass: returned as is

    events = []
    monkeypatch.setattr(agent, "_run_codex", fake_codex)
    monkeypatch.setattr(agent, "_cdp_url", WS)
    agent._mcp[:] = agent.mcp_overrides(agent.mcp_server("playwright-mcp", WS))
    asyncio.run(agent.run_codex(prompt="task", event_callback=events.append))
    assert BROWSER_ID not in json.dumps(events) and seen["prompt"].endswith("Task:\ntask\n")
    assert 'web_search="disabled"' in seen["overrides"]
    assert any(o.startswith("mcp_servers.playwright.args=") and BROWSER_ID in o for o in seen["overrides"])


def test_codex_turn_result_and_bridge_logs_never_carry_the_browser(monkeypatch, caplog):
    """The framework sends the result's warnings in the turn's data and its error as the failure, and Codex's bridge logs
    both before run_codex returns them; a tool's error text can carry the address."""
    import logging

    agent = codex_tools_agent(monkeypatch)
    import codex_bridge

    async def fake_codex(**kwargs):
        logging.getLogger("codex_bridge").info("codex CLI emitted %d non-fatal notice(s): %s", 1, f"cannot reach {WS}")
        return codex_bridge.CodexResult(response=f"FINAL ANSWER: {WS}", tool_call_count=1, session_id="s",
                                        error=f"turn failed at {WS}", error_detail=WS,
                                        warnings=[f'{{"message": "connect {BROWSER_ID}.cdp.browser-use.com refused"}}'],
                                        trajectory=[{"type": "error", "message": WS}])

    monkeypatch.setattr(agent, "_run_codex", fake_codex)
    monkeypatch.setattr(agent, "_cdp_url", WS)
    with caplog.at_level(logging.INFO, logger="codex_bridge"):
        result = asyncio.run(agent.run_codex(prompt="task"))

    assert BROWSER_ID not in json.dumps(dataclasses.asdict(result)) and caplog.records
    assert BROWSER_ID not in caplog.text and result.response == "FINAL ANSWER: <cdp>" and result.tool_call_count == 1


def test_repository_carries_no_decrypted_task_text():
    """Task text, rubrics and the canary live only in the encrypted benchmark file; no other file may carry them."""
    data = tasks._benchmark()
    needles = {data["canary"]: "canary"} if data.get("canary") else {}
    for task in data["tasks"]:
        needles[task["canary"]] = f"{task['id']} canary"
        for field in ("task", "rubric", "summary", "title"):
            text = " ".join(str(task.get(field) or "").split())
            if len(text) >= 20:  # every 40 characters, and the tail, so any 80 copied characters hold a whole needle
                for needle in {text[i:i + 40] for i in range(0, len(text) - 39, 40)} | {text[-40:]}:
                    needles[needle] = f"{task['id']} {field}"
    long = {needle: where for needle, where in needles.items() if len(needle) == 40}
    short = {needle: where for needle, where in needles.items() if len(needle) != 40}
    skip = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}
    # The judge's files are the public benchmark's own, checked byte for byte against it as it runs; its prompt shares
    # the rubrics' template text.
    public = {tasks.JUDGE_DIR / name for name in json.loads((tasks.JUDGE_DIR / "SOURCE.json").read_text())["files"]}
    leaks = []
    # Every file this benchmark's change adds or touches.
    ours = [ROOT / "src" / "agentenv_benchmarks" / "bu_bench", BUNDLE, TOOLS, ROOT / "tests",
            ROOT / "src" / "agentenv_benchmarks" / "bu_browser", BUNDLE.parent / "bu-browser", ROOT / "README.md",
            ROOT / "NOTICE"]
    for path in (p for top in ours for p in ([top] if top.is_file() else top.rglob("*"))):
        if path.is_dir() or skip & set(path.relative_to(ROOT).parts) or path in public:
            continue
        text = " ".join(path.read_bytes().decode("utf-8", "replace").split())
        windows = {text[i:i + 40] for i in range(len(text) - 39)}
        leaks += [f"{path.relative_to(ROOT)}: {long[needle]}" for needle in windows & long.keys()]
        leaks += [f"{path.relative_to(ROOT)}: {where}" for needle, where in short.items() if needle in text]
    assert not leaks, leaks[:5]  # where, never the text itself: CI logs can be public


def test_browser_handoff_sends_the_agents_control_token(monkeypatch):
    """Agents on agentenv-framework-protocol >= 0.1.582 answer 401 on cdp-config without their control token."""
    from types import SimpleNamespace

    from agent_env.a2a_agent.a2a_agent import A2AAgent

    seen = []
    real = httpx.AsyncClient

    def reply(request):
        seen.append(request)
        return httpx.Response(200, json={}, request=request)

    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")

    async def start(**kwargs):
        return browser

    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setattr(steps.cloud.CloudBrowser, "start", start)
    monkeypatch.setattr(steps.cloud.Screenshots, "start", lambda self: None)
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_CDP_CONFIG, "params": {"endpoint": "/ext/cdp"}}]}}
    step = steps.BuCloudBrowserTaskStep("b", 1, "bu2-140")
    for token, header in [("tok", {"agentenv-control-token": "tok"}), (None, {})]:
        agent = SimpleNamespace(agent_name=step.agent_name, a2a_url="http://agent", a2a_card=card,
                                control_headers=header)
        context = SimpleNamespace(metadata={}, deployed_agents=[agent])
        asyncio.run(step.execute(context))
        assert seen[-1].url.path == "/ext/cdp"
        assert seen[-1].headers.get("agentenv-control-token") == token
        steps._LIVE.clear()


def test_preflight_names_a_missing_key_before_any_agent_or_browser_starts(monkeypatch):
    """A run with no Browser Use or judge key stops at preflight, not after an agent turn and a cloud browser."""
    for name in (cloud.KEY_NAME, *steps.JUDGE_KEY_NAMES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(steps, "_judge_key", lambda: "")
    monkeypatch.setattr(cloud, "api_key", lambda: (_ for _ in ()).throw(RuntimeError("set BROWSER_USE_API_KEY")))
    browser = steps.BuCloudBrowserTaskStep("browser", 1, "bu2-140")
    verifier = steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")
    assert "BROWSER_USE_API_KEY" in " ".join(browser.preflight())
    assert "BU_BENCH_JUDGE_API_KEY" in " ".join(verifier.preflight())
    monkeypatch.setattr(steps, "_judge_key", lambda: "k")
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    assert browser.preflight() == [] and verifier.preflight() == []


def test_bundle_tasks_are_what_make_bundle_writes(tmp_path, monkeypatch):
    """The committed task files and evals are make_bundle.py's output, byte for byte."""
    make_bundle = load_script("make_bundle")
    fake = tmp_path / "bu-bench"
    monkeypatch.setattr(make_bundle, "BUNDLE", fake)
    make_bundle.main()
    for kind in ("tasks", "evals"):
        written = {p.name: p.read_text() for p in (fake / kind).iterdir()}
        committed = {p.name: p.read_text() for p in (BUNDLE / kind).iterdir()}
        assert written == committed, kind


class FakeBrowserUseApi:
    """cloud.request as Browser Use answers it: every call recorded, the first ``failing_stops`` stops a 503."""

    def __init__(self, failing_stops: int = 0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.failing_stops = failing_stops

    async def __call__(self, method, path, body=None, timeout=60):
        self.calls.append((method, path))
        if method == "PATCH" and self.failing_stops:
            self.failing_stops -= 1
            raise RuntimeError("Browser Use API PATCH /browsers returned 503")
        return {}

    @property
    def stops(self) -> int:
        return sum(method == "PATCH" for method, _ in self.calls)


#: A Codex turn's trajectory of one step, its answer.
TRAJECTORY = json.dumps([{"type": "item.completed", "item": {"type": "agent_message", "text": "FINAL ANSWER: done"}}]
                        ).encode()


def verifying(monkeypatch, tmp_path, judgment, failing_stops=0, evidence_dir=None, handle="h1", artifacts=None,
              trajectory=TRAJECTORY):
    """A live browser as bu_cloud_browser leaves it, the agent's answered turn, its delivered files (``artifacts``,
    name: bytes, or None for one whose download fails) in a faked object store, Browser Use's API faked, the judge
    answering ``judgment`` (None: the real judge runs), and the verifier step with the context it runs in. The stored
    ``trajectory`` is None for one whose read fails."""
    from types import SimpleNamespace

    import agent_env.config
    from agent_env.task_step.context import PromptResponse

    objects = {"mem://trajectory": trajectory, **{f"mem://out/{name}": raw for name, raw in (artifacts or {}).items()}}

    class Store:
        def open(self, url):
            if objects[url] is None:
                raise OSError("object store unavailable")
            return io.BytesIO(objects[url])

        def get(self, url):
            return self.open(url).read()

    monkeypatch.setattr(agent_env.config, "get_config",
                        lambda: SimpleNamespace(get_object_store_at=lambda url: Store()))
    api = FakeBrowserUseApi(failing_stops)
    monkeypatch.setattr(cloud, "request", api)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: real_sleep(0))
    if evidence_dir is None:
        monkeypatch.delenv(steps.EVIDENCE_DIR_ENV, raising=False)
    else:
        monkeypatch.setenv(steps.EVIDENCE_DIR_ENV, str(evidence_dir))
    judged = []

    async def judge(self, workspace):
        judged.append(workspace)
        return judgment

    if judgment is not None:
        monkeypatch.setattr(steps.BuFindingsVerifierTaskStep, "_judge", judge)
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")
    shots = cloud.Screenshots("ws://127.0.0.1:9/devtools/browser/x", tmp_path / "shots")  # nothing answers there
    monkeypatch.setattr(steps, "_LIVE", {handle: steps._Live(browser, shots, tmp_path / "shots", 0.0)})
    answered = PromptResponse(prompt_id="solve", response="FINAL ANSWER: done",
                              agent_trajectory_object_url="mem://trajectory")
    context = SimpleNamespace(metadata={steps.STATE_KEY: {"handle": handle, "browser": browser.label},
                                        "run_group_id": "group-1",
                                        "artifacts": {name: f"mem://out/{name}" for name in artifacts or {}}},
                              prompt_responses=[answered], deployed_agents=[], instance_id=None)
    return steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve"), context, api, judged


SCORED = {"status": "scored", "score": 0.75, "rubric": {"a": True, "b": False}, "reason": "ok"}


def test_a_graded_task_stops_its_browser_once_and_drops_its_temporary_evidence(monkeypatch, tmp_path):
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)

    asyncio.run(step.execute(context))

    verification = context.metadata["verifications"]["findings"]
    assert verification["score"] == 0.75 and api.stops == 1 and steps._LIVE == {}
    # The workspace held the decrypted task and its rubric.
    assert not judged[0].exists() and verification["evidence_dir"] is None


def test_a_judge_error_stops_the_browser_once_records_no_score_and_keeps_the_evidence(monkeypatch, tmp_path):
    step, context, api, judged = verifying(monkeypatch, tmp_path, {"status": "error", "reason": "judge 500"})

    with pytest.raises(RuntimeError, match="did not score"):
        asyncio.run(step.execute(context))

    assert "score" not in context.metadata["verifications"]["findings"] and api.stops == 1 and steps._LIVE == {}
    assert (judged[0] / "result.json").exists()


def test_an_evidence_folder_that_cant_be_written_still_stops_the_browser(monkeypatch, tmp_path):
    (tmp_path / "not-a-folder").write_text("")
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED, evidence_dir=tmp_path / "not-a-folder")

    with pytest.raises(OSError):
        asyncio.run(step.execute(context))

    assert api.stops == 1 and steps._LIVE == {} and judged == [] and "verifications" not in context.metadata


@pytest.mark.parametrize("failing_stops, stops, left", [(1, 2, False), (3, 3, True)],
                         ids=["one 503, then stopped", "every stop fails"])
def test_a_failed_stop_is_retried_and_a_browser_never_stopped_is_left_to_the_exit_hook(
        monkeypatch, tmp_path, failing_stops, stops, left):
    step, context, api, _ = verifying(monkeypatch, tmp_path, SCORED, failing_stops=failing_stops)
    patched = []
    monkeypatch.setattr(steps.httpx, "patch", lambda url, **kwargs: patched.append((url, kwargs["json"])))
    monkeypatch.setattr(cloud, "api_key", lambda: "k")

    asyncio.run(step.execute(context))
    steps._stop_leftovers()

    assert api.stops == stops and bool(steps._LIVE) is left
    assert patched == ([(f"{cloud.API}/browsers/{BROWSER_ID}", {"action": "stop"})] if left else [])


def test_a_task_whose_browser_is_gone_is_not_judged(monkeypatch, tmp_path):
    """Its screenshots went with it: a judgment would score the agent on no evidence."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    steps._LIVE.clear()

    with pytest.raises(RuntimeError, match="no live browser"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.calls == [] and "verifications" not in context.metadata


def test_a_turn_that_left_no_response_is_not_judged(monkeypatch, tmp_path):
    """A turn that timed out, or whose agent died or never got the prompt, leaves no response: judging an empty reply
    would score the agent on what the run lost."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    context.prompt_responses = []

    with pytest.raises(RuntimeError, match="left no response"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.stops == 1 and steps._LIVE == {} and "verifications" not in context.metadata


def test_a_delivered_file_that_cant_be_read_is_not_judged_as_missing(monkeypatch, tmp_path):
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED, artifacts={"a.csv": b"1,2", "b.csv": None})

    with pytest.raises(RuntimeError, match="reading a delivered file failed"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.stops == 1 and "verifications" not in context.metadata


@pytest.mark.parametrize("failed, judged_anyway", [("collect_artifacts", False), ("prompt_agent", True)],
                         ids=["collect failed", "the turn failed"])
def test_a_failed_collect_is_not_judged_as_no_files_and_a_failed_turn_is_judged_on_its_response(
        monkeypatch, tmp_path, failed, judged_anyway):
    """Both steps run with fail_task_on_error=false. A collect that failed (its sandbox died, an upload failed) kept only
    the files it uploaded first; a turn that failed left the response it is judged on, as the published runs were. The
    failure is recorded as AgentEnv's runner records it."""
    from agent_env.task import Task
    FailedStep = pytest.importorskip("agent_env.task.outcome").FailedStep

    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    task = Task.from_dict({"id": "bu2-140", "version": None,
                           "steps": json.loads((BUNDLE / "tasks" / "bu2-140.json").read_text())})
    step = next(s for s in task.steps if s.type == steps.BuFindingsVerifierTaskStep.type)
    failing = next(s for s in task.steps if s.type == failed)
    context.metadata["failed_steps"] = [FailedStep.from_exception(
        failing, RuntimeError("sandbox gone"), started_at_utc="2026-11-19T00:00:00Z", duration_seconds=1.0,
        is_fatal=False).to_dict()]

    if judged_anyway:
        asyncio.run(step.execute(context))
        assert context.metadata["verifications"]["findings"]["score"] == 0.75
    else:
        with pytest.raises(RuntimeError, match="step collect, which this verifier depends on, failed") as raised:
            asyncio.run(step.execute(context))
        assert judged == [] and "verifications" not in context.metadata
        assert Path(re.search(r"kept in (\S+)\)", str(raised.value)).group(1)).is_dir()
    assert api.stops == 1


@pytest.mark.parametrize("stored", [None, b"[]", b'[{"type": "thread.started", "thread_id": "t"}]',
                                    b'[{"type": "agentenv.usage", "usage": {"input_tokens": 9}, "model_requests": 1}]',
                                    json.dumps({"format": "browser-use-messages/1", "trajectory": [
                                        {"kind": "response", "parts": [{"part_kind": "thinking", "content": "hm"}]}]
                                                }).encode()],
                         ids=["lost", "empty", "control events only", "usage only (the CLI's events lost)",
                              "thinking only, which the judge drops"])
def test_a_turn_whose_trajectory_cant_be_read_or_is_empty_is_not_judged_as_zero_steps(monkeypatch, tmp_path, stored):
    """Neither the stored copy nor the agent (gone here) gives the trajectory, or the one stored has no step the judge
    reads: it would see no steps at all. No published run had none."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED, trajectory=stored)

    with pytest.raises(RuntimeError, match="no trajectory with a step the judge reads could be read"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.stops == 1 and "verifications" not in context.metadata


def test_the_verifiers_object_store_reads_leave_other_tasks_running(monkeypatch, tmp_path):
    """The runner runs every task's steps on one event loop: a slow store read must not stall the others."""
    import time
    from types import SimpleNamespace

    import agent_env.config

    step, context, _, _ = verifying(monkeypatch, tmp_path, SCORED, artifacts={"a.csv": b"1,2"})

    reads = []

    class SlowStore:
        def open(self, url):
            started = time.monotonic()
            time.sleep(0.3)
            reads.append((started, time.monotonic()))
            return io.BytesIO(TRAJECTORY if url == "mem://trajectory" else b"1,2")

        def get(self, url):
            return self.open(url).read()

    monkeypatch.setattr(agent_env.config, "get_config",
                        lambda: SimpleNamespace(get_object_store_at=lambda url: SlowStore()))
    ticks = []

    async def other_task():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0)

    async def run():
        other = asyncio.ensure_future(other_task())
        await step.execute(context)
        other.cancel()

    asyncio.run(run())
    assert len(reads) == 2 and all(any(start < tick < end for tick in ticks) for start, end in reads)


def test_two_runs_of_a_task_in_one_run_group_keep_their_own_evidence(monkeypatch, tmp_path):
    """An explorer batch runs a task several times under one run group id."""
    for n, handle in enumerate(["a" * 32, "b" * 32]):
        step, context, _, _ = verifying(monkeypatch, tmp_path / str(n), SCORED, evidence_dir=tmp_path / "evidence",
                                        handle=handle)
        asyncio.run(step.execute(context))
    assert len(list((tmp_path / "evidence").iterdir())) == 2
    assert all((folder / "result.json").exists() for folder in (tmp_path / "evidence").iterdir())


def fake_judge(monkeypatch, tmp_path, code: str) -> None:
    """The judge replaced by ``code``, a Python script run as the real one is, its paths in EVAL_*."""
    script = tmp_path / "fake_judge.py"
    script.write_text("import json, os, time\nfrom pathlib import Path\n" + textwrap.dedent(code))
    monkeypatch.setattr(steps, "JUDGE_RUN", script)
    monkeypatch.setenv("BU_BENCH_JUDGE_API_KEY", "k")


@pytest.mark.parametrize("answers, runs, status", [(["error", "scored"], 2, "scored"),
                                                   (["error", "error", "error", "scored"], 3, "error")],
                         ids=["one transient error", "errs every time"])
def test_a_judge_error_is_asked_again_before_the_task_is_given_up(monkeypatch, tmp_path, answers, runs, status):
    fake_judge(monkeypatch, tmp_path, f"""
        runs = Path(os.environ["EVAL_WORKSPACE"]) / "runs"
        runs.write_text((runs.read_text() if runs.exists() else "") + "x")
        answer = {answers!r}[len(runs.read_text()) - 1]
        Path(os.environ["EVAL_JUDGMENT_PATH"]).write_text(json.dumps({{"status": answer, "score": 0.5}}))
    """)
    monkeypatch.setattr(steps, "JUDGE_RETRY_S", 0)
    step = steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")

    assert asyncio.run(step._judge(tmp_path))["status"] == status and len((tmp_path / "runs").read_text()) == runs


def test_grading_cancelled_kills_the_judge_it_runs(monkeypatch, tmp_path):
    """The run gave up on the task (its timeout, Ctrl-C): a judge left running would go on asking the model, billed."""
    step, context, _, _ = verifying(monkeypatch, tmp_path, None)
    fake_judge(monkeypatch, tmp_path, """
        (Path(os.environ["EVAL_WORKSPACE"]) / "pid").write_text(str(os.getpid()))
        time.sleep(60)
    """)
    monkeypatch.setenv(steps.EVIDENCE_DIR_ENV, str(tmp_path / "evidence"))

    async def grade_then_cancel() -> bool:
        grading = asyncio.ensure_future(step.execute(context))
        while not (pid := next((p.read_text() for p in (tmp_path / "evidence").glob("*/pid")), "")):
            await asyncio.sleep(0.05)
        grading.cancel()
        await asyncio.gather(grading, return_exceptions=True)
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True  # killed and reaped
        os.kill(int(pid), 9)
        return False

    assert asyncio.run(grade_then_cancel())


def test_the_real_judge_reads_the_verifiers_evidence_up_to_its_model_call(monkeypatch, tmp_path):
    """No stand-in: the vendored judge checks its sources against SOURCE.json, finds the task, matches its instruction
    and reads the evidence the verifier wrote, then fails only at the model, which nothing answers here."""
    step, context, _, _ = verifying(monkeypatch, tmp_path, None)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BU_BENCH_JUDGE_API_KEY", "dry-run")
    monkeypatch.setattr(steps, "JUDGE_ATTEMPTS", 1)

    with pytest.raises(RuntimeError, match="did not score"):
        asyncio.run(step.execute(context))

    judgment = context.metadata["verifications"]["findings"]["report"]["judgment"]
    assert judgment["status"] == "error" and judgment["reason"].startswith("ModelProviderError: Connection error")


@pytest.mark.parametrize("answer, failing_stops, left", [(401, 0, False), (500, 0, False), (500, 3, True)],
                         ids=["handoff 401", "handoff 500", "handoff 500, every stop fails"])
def test_a_failed_browser_handoff_stops_the_browser_or_leaves_it_to_the_exit_hook(monkeypatch, answer, failing_stops,
                                                                                  left):
    from types import SimpleNamespace

    from agent_env.a2a_agent.a2a_agent import A2AAgent

    real = httpx.AsyncClient
    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda request: httpx.Response(answer, request=request)), **kw))
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")

    async def start(**kwargs):
        return browser

    monkeypatch.setattr(steps.cloud.CloudBrowser, "start", start)
    api = FakeBrowserUseApi(failing_stops)
    monkeypatch.setattr(cloud, "request", api)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: real_sleep(0))
    monkeypatch.setattr(steps, "_LIVE", {})
    patched = []
    monkeypatch.setattr(steps.httpx, "patch", lambda url, **kwargs: patched.append((url, kwargs["json"])))
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_CDP_CONFIG, "params": {"endpoint": "/ext/cdp"}}]}}
    step = steps.BuCloudBrowserTaskStep("b", 1, "bu2-140")
    agent = SimpleNamespace(agent_name=step.agent_name, a2a_url="http://agent", a2a_card=card, control_headers={})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(step.execute(SimpleNamespace(metadata={}, deployed_agents=[agent])))
    steps._stop_leftovers()

    assert api.stops == (cloud.STOP_ATTEMPTS if left else 1) and bool(steps._LIVE) is left
    assert patched == ([(f"{cloud.API}/browsers/{BROWSER_ID}", {"action": "stop"})] if left else [])


@pytest.mark.parametrize("recorded, delivered", [("per step and flat", ["a.csv"]), ("flat", ["b.csv"]),
                                                 ("neither", None)])
def test_the_delivered_files_are_those_the_verifiers_own_collect_recorded(monkeypatch, tmp_path, recorded, delivered):
    """collect_artifacts records its files under its step id, and the latest collect's under the legacy flat key; a
    collect that succeeded and recorded neither must not be judged as no files."""
    step, context, _, judged = verifying(monkeypatch, tmp_path, SCORED, artifacts={"a.csv": b"1", "b.csv": b"2"})
    grade = json.loads((BUNDLE / "tasks" / "bu2-140.json").read_text())[-1]
    step = steps.BuFindingsVerifierTaskStep.from_dict({**grade, "version": 1})
    metadata = {key: value for key, value in context.metadata.items() if key != "artifacts"}
    if recorded != "neither":
        metadata["artifacts"] = {"b.csv": "mem://out/b.csv"}  # as a later collect, not this verifier's, left it
    if recorded == "per step and flat":
        metadata["collected_artifacts"] = {"collect": {"artifacts": {"a.csv": "mem://out/a.csv"}}}
    context.metadata = metadata

    if delivered is None:
        with pytest.raises(RuntimeError, match="step collect, which this verifier depends on, recorded no collected"):
            asyncio.run(step.execute(context))
        assert judged == []
    else:
        asyncio.run(step.execute(context))
        assert context.metadata["verifications"]["findings"]["report"]["delivered_files"] == delivered


def test_the_recording_is_kept_as_a_file_artifact(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from agent_env.artifact.artifacts.file import FileArtifact
    from agent_env.artifact.store import reset_artifact_store
    from agent_env.config import configure, reset_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    video = os.urandom(3 << 20)
    real = httpx.AsyncClient
    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=video)), **kw))
    configure()
    reset_artifact_store()
    try:
        context = SimpleNamespace(metadata={}, instance_id=None)
        kept = asyncio.run(steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")._keep_recording(
            {"recordingUrl": "https://recordings.example/r.mp4?X-Amz-Signature=s"}, context))
        artifact = FileArtifact.get(kept["artifact_id"])

        assert artifact.load() == video and kept["bytes"] == len(video) and artifact.content_type == "video/mp4"
        assert context.metadata["recordings"]["grade"][0]["bytes"] == len(video)
    finally:
        reset_config()
        reset_artifact_store()


def test_httpx_request_logs_name_no_browser_and_no_signed_link(monkeypatch, caplog):
    """httpx logs each request's URL at INFO, which agent-env up logs at: a browser's id or CDP host there reaches it."""
    import logging

    recording = "https://recordings.example/r.mp4?X-Amz-Signature=s3cr3t"

    def reply(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": BROWSER_ID, "cdpUrl": f"https://{BROWSER_ID}.cdp.browser-use.com"})
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"webSocketDebuggerUrl": WS})
        return httpx.Response(200, json={"id": BROWSER_ID, "recordingUrl": recording})

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setenv(cloud.KEY_NAME, "k")

    async def browser_lifetime():
        browser = await cloud.CloudBrowser.start()
        await browser.stop()
        session = await browser.session()
        async with httpx.AsyncClient() as client:
            await client.get(cloud.recording_urls(session)[0])

    with caplog.at_level(logging.INFO, logger="httpx"):
        asyncio.run(browser_lifetime())

    assert len([record for record in caplog.records if record.name == "httpx"]) == 5
    assert BROWSER_ID not in caplog.text and "cdp.browser-use.com" not in caplog.text and "s3cr3t" not in caplog.text


def test_a_recording_link_that_redirects_logs_neither_signature(monkeypatch, caplog):
    """A recording link may redirect to another host, whose link is signed too: httpx logs every hop's URL."""
    import logging
    from types import SimpleNamespace

    from agent_env.artifact.artifacts.file import FileArtifact

    def reply(request):
        if request.url.host == "recordings.example":
            return httpx.Response(302, headers={"location": "https://cdn.elsewhere.example/r.mp4?X-Amz-Signature=hop2"})
        return httpx.Response(200, content=b"mp4")

    real = httpx.AsyncClient
    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setattr(FileArtifact, "put", lambda artifact_id, file_path, description: SimpleNamespace(
        id=artifact_id, version=1, content_type="video/mp4"))
    monkeypatch.setattr(cloud, "RECORDING_HOSTS", set())
    session = {"recordingUrl": "https://recordings.example/r.mp4?X-Amz-Signature=hop1"}

    with caplog.at_level(logging.INFO, logger="httpx"):
        kept = asyncio.run(steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")._keep_recording(
            session, SimpleNamespace(metadata={}, instance_id=None)))

    assert kept["bytes"] == 3 and len([record for record in caplog.records if record.name == "httpx"]) == 2
    assert "hop1" not in caplog.text and "hop2" not in caplog.text and "cdn.elsewhere.example" in caplog.text


def test_logged_urls_hide_every_cdp_host_form():
    import logging

    record = logging.LogRecord("httpx", logging.INFO, __file__, 1, "HTTP Request: GET %s", (
        f"https://{BROWSER_ID}.cdp.eu.browser-use.com/json and https://{BROWSER_ID}.cdp0.browser-use.com/json",), None)
    cloud._scrub_logged_urls(record)
    assert BROWSER_ID not in record.getMessage() and record.getMessage().count("<cdp>") == 2


def test_a_browser_whose_start_failed_and_stop_failed_is_left_to_the_exit_hook(monkeypatch):
    calls = []

    async def api(method, path, body=None, timeout=60):
        calls.append(method)
        if method == "POST":
            return {"id": BROWSER_ID, "cdpUrl": ""}  # no cdpUrl: the start fails after the browser exists
        raise RuntimeError("Browser Use API PATCH /browsers returned 503")

    patched = []
    monkeypatch.setattr(cloud, "request", api)
    monkeypatch.setattr(cloud, "STOP_RETRY_S", 0.0)
    monkeypatch.setattr(cloud, "UNSTOPPED", [])
    monkeypatch.setattr(steps, "_LIVE", {})
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    monkeypatch.setattr(steps.httpx, "patch", lambda url, **kw: patched.append(url))
    try:
        asyncio.run(cloud.CloudBrowser.start())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the start to fail")
    steps._stop_leftovers()
    assert calls.count("PATCH") == cloud.STOP_ATTEMPTS and [b.id for b in cloud.UNSTOPPED] == [BROWSER_ID]
    assert patched == [f"{cloud.API}/browsers/{BROWSER_ID}"]


def test_a_browser_whose_start_failed_and_whose_stop_was_cancelled_is_left_to_the_exit_hook(monkeypatch):
    async def api(method, path, body=None, timeout=60):
        if method == "POST":
            return {"id": BROWSER_ID, "cdpUrl": ""}  # no cdpUrl: the start fails after the browser exists
        await asyncio.sleep(60)  # the stop hangs, and the run is cancelled meanwhile

    monkeypatch.setattr(cloud, "request", api)
    monkeypatch.setattr(cloud, "UNSTOPPED", [])

    async def cancelled_during_the_stop():
        starting = asyncio.ensure_future(cloud.CloudBrowser.start())
        await asyncio.sleep(0.1)
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)

    asyncio.run(cancelled_during_the_stop())
    assert [browser.id for browser in cloud.UNSTOPPED] == [BROWSER_ID]


@pytest.mark.parametrize("failing_stops, left", [(0, False), (3, True)], ids=["stopped", "every stop fails"])
def test_a_browser_step_cancelled_before_browser_use_answered_stops_the_browser_it_made(monkeypatch, failing_stops,
                                                                                      left):
    """Cancelled (a task timeout) while the create is in flight, the step still learns the browser's id once Browser
    Use answers, and stops it or leaves it to the exit hook, rather than letting it bill for its 70 minutes."""
    from types import SimpleNamespace

    from agent_env.a2a_agent.a2a_agent import A2AAgent

    api = FakeBrowserUseApi(failing_stops)
    answered = None

    async def late_api(method, path, body=None, timeout=60):
        if method == "POST":
            api.calls.append((method, path))
            await answered.wait()
            return {"id": BROWSER_ID, "cdpUrl": f"https://{BROWSER_ID}.cdp.browser-use.com"}
        return await api(method, path, body, timeout)

    monkeypatch.setattr(cloud, "request", late_api)
    monkeypatch.setattr(cloud, "STOP_RETRY_S", 0.0)
    monkeypatch.setattr(cloud, "UNSTOPPED", [])
    monkeypatch.setattr(steps, "_LIVE", {})
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    patched = []
    monkeypatch.setattr(steps.httpx, "patch", lambda url, **kw: patched.append(url))
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_CDP_CONFIG, "params": {"endpoint": "/ext/cdp"}}]}}
    step = steps.BuCloudBrowserTaskStep("b", 1, "bu2-140")
    agent = SimpleNamespace(agent_name=step.agent_name, a2a_url="http://agent", a2a_card=card, control_headers={})

    async def cancelled_then_answered():
        nonlocal answered
        answered = asyncio.Event()
        running = asyncio.ensure_future(step.execute(SimpleNamespace(metadata={}, deployed_agents=[agent])))
        while not api.calls:
            await asyncio.sleep(0)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        answered.set()
        for _ in range(1000):  # until whatever the cancel left running has ended
            if asyncio.all_tasks() == {asyncio.current_task()}:
                break
            await asyncio.sleep(0)
        return running

    assert asyncio.run(cancelled_then_answered()).cancelled()
    steps._stop_leftovers()
    assert api.stops == (cloud.STOP_ATTEMPTS if left else 1)
    assert [b.id for b in cloud.UNSTOPPED] == ([BROWSER_ID] if left else [])
    assert patched == ([f"{cloud.API}/browsers/{BROWSER_ID}"] if left else [])


def test_a_browser_create_waits_minutes_for_its_answer_and_a_stop_does_not(monkeypatch):
    """A create answered after the client gave up makes a browser whose id nobody has, billing until its timeout."""
    read_timeouts = {}

    def reply(request):
        read_timeouts[request.method] = request.extensions["timeout"]["read"]
        if request.method == "POST":
            return httpx.Response(200, json={"id": BROWSER_ID, "cdpUrl": f"https://{BROWSER_ID}.cdp.browser-use.com"})
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"webSocketDebuggerUrl": WS})
        return httpx.Response(200, json={})

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setenv(cloud.KEY_NAME, "k")

    async def start_and_stop():
        await (await cloud.CloudBrowser.start()).stop()

    asyncio.run(start_and_stop())
    assert read_timeouts["POST"] >= 300 and read_timeouts["PATCH"] == cloud.STOP_REQUEST_S


def test_a_stop_whose_answer_trickles_in_still_ends_within_its_budget(monkeypatch):
    """httpx's timeout bounds each read, not the request: an answer that keeps sending must not outlast the close."""
    import time

    async def trickle():
        for _ in range(100):
            await asyncio.sleep(0.05)
            yield b" "

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=trickle())), **kw))
    monkeypatch.setenv(cloud.KEY_NAME, "k")
    monkeypatch.setattr(cloud, "STOP_REQUEST_S", 0.3)
    monkeypatch.setattr(cloud, "STOP_RETRY_S", 0.0)
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS)

    started = time.monotonic()
    asyncio.run(browser.stop())

    assert time.monotonic() - started < cloud.STOP_ATTEMPTS * 0.3 + 1 and not browser.stopped


def test_the_exit_hooks_stops_end_by_its_deadline_and_one_that_trickles_holds_back_no_other(monkeypatch):
    """httpx's timeout bounds each read: a stop answer that keeps sending must neither hold the exit nor the stops of
    the browsers after it. Here Browser Use answers the first browser's stop a byte at a time, on a real socket."""
    import socketserver
    import threading
    import time

    stopped, done = [], threading.Event()

    class BrowserUse(socketserver.BaseRequestHandler):
        def handle(self):
            request = self.request.recv(65536).decode()
            path = request.split(" ")[1]
            if path.endswith("/browsers/trickling"):
                self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")
                while not done.wait(0.05):
                    self.request.sendall(b" ")
                return
            stopped.append(path.rsplit("/", 1)[1])
            self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), BrowserUse)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(cloud, "API", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    monkeypatch.setattr(steps, "EXIT_STOP_S", 0.5)
    monkeypatch.setattr(steps, "_LIVE", {"h1": steps._Live(cloud.CloudBrowser(id="trickling", ws_url=WS), None,
                                                           Path("."), 0.0)})
    monkeypatch.setattr(cloud, "UNSTOPPED", [cloud.CloudBrowser(id="unstopped", ws_url=WS)])
    try:
        started = time.monotonic()
        steps._stop_leftovers()
        took = time.monotonic() - started
    finally:
        done.set()
        server.shutdown()
        server.server_close()

    assert stopped == ["unstopped"] and took < 0.5 + 0.5


def test_the_log_filter_leaves_requests_to_other_hosts_as_they_are():
    """It is process-wide: a bundle run beside BU Bench keeps its own log lines."""
    import logging

    other = 'HTTP Request: GET https://huggingface.co/api/models/x?revision=main "HTTP/1.1 200 OK"'
    ours = f'HTTP Request: PATCH https://api.browser-use.com/api/v3/browsers/{BROWSER_ID}?x=1 "HTTP/1.1 200 OK"'
    records = [logging.LogRecord("httpx", logging.INFO, __file__, 1, message, None, None) for message in (other, ours)]
    for record in records:
        cloud._scrub_logged_urls(record)

    assert records[0].getMessage() == other
    assert records[1].getMessage() == 'HTTP Request: PATCH https://api.browser-use.com/api/v3/browsers/<id>?<redacted> ' \
                                      '"HTTP/1.1 200 OK"'


def test_a_cancelled_recording_upload_keeps_its_file_until_the_upload_has_read_it(monkeypatch, tmp_path):
    """The upload runs in a thread, which a cancelled wait for it leaves running."""
    import threading
    from types import SimpleNamespace

    from agent_env.artifact.artifacts.file import FileArtifact

    video = os.urandom(1 << 16)
    release, read = threading.Event(), []

    def put(artifact_id, file_path, description):
        release.wait(10)
        read.append(Path(file_path).read_bytes())
        return SimpleNamespace(id=artifact_id, version=1, content_type="video/mp4")

    real = httpx.AsyncClient
    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=video)), **kw))
    monkeypatch.setattr(FileArtifact, "put", put)
    monkeypatch.setattr(steps.tempfile, "tempdir", str(tmp_path))
    step = steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")

    async def cancelled_while_uploading():
        keeping = asyncio.ensure_future(step._keep_recording(
            {"recordingUrl": "https://recordings.example/r.mp4"}, SimpleNamespace(metadata={}, instance_id=None)))
        await asyncio.sleep(0.2)
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)
        release.set()

    asyncio.run(cancelled_while_uploading())  # its end waits for the upload's thread
    assert read == [video] and list(tmp_path.iterdir()) == []


def test_a_key_with_stray_whitespace_is_sent_trimmed_and_never_echoed(monkeypatch, caplog):
    """A key stored with a trailing newline must not reach an error text (h11 refuses the header and names its value)."""
    sent = []

    def reply(request):
        sent.append(request.headers["x-browser-use-api-key"])
        return httpx.Response(200, json={"id": BROWSER_ID})

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setattr(cloud, "api_key", lambda: "bu_SECRETKEY123\n")
    asyncio.run(cloud.request("GET", f"/browsers/{BROWSER_ID}"))
    assert sent == ["bu_SECRETKEY123"]

    def refuse(request):
        raise httpx.LocalProtocolError("Illegal header value b'bu_SECRETKEY123\\x00'")

    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(refuse), **kw))
    try:
        asyncio.run(cloud.request("GET", f"/browsers/{BROWSER_ID}"))
    except RuntimeError as error:
        assert "SECRETKEY" not in str(error) and BROWSER_ID not in str(error) and "LocalProtocolError" in str(error)
    else:
        raise AssertionError("expected the request to fail")
