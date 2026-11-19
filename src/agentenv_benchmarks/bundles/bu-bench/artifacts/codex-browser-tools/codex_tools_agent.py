"""AgentEnv's Codex agent with one browser tool, set up as the harness bake-off's arm for that tool was
(Browser Use's eval platform, harness codex-browser-tools: setup.sh pins, run.py tool setup and compose_prompt).

``BU_BENCH_TOOL`` (from the agent's agent.toml) picks the tool: browser-harness (the Browser Use CLI), agent-browser,
playwright-mcp, chrome-devtools-mcp, browser-use (browser-use's own MCP server), stagehand or opencode-browse.

Changes to the agent this image is built on, and nothing else:
- It declares ``cdp-config/v1``, so a task hands it a browser, and each turn points the tool at that browser: a CLI
  tool through its environment and only its command on PATH, an MCP tool as a stdio server in the turn's Codex config,
  started as the bake-off started it. With no browser the turn fails rather than start one of its own.
- Each turn's message is the bake-off prompt for the tool: its rules, then the task the step sends.
- Each turn's Codex runs as the bake-off's did: no web search tool and no repository instructions.
- The browser's CDP address, which alone reaches it, is scrubbed from the events the trajectory keeps, from every text
  the turn's result carries (its reply, errors and warnings) and from what Codex's bridge logs, in case a command or a
  tool prints it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import a2a_server
from agentenv_protocol.a2a_agent import CDP_CONFIG_V1, TaskRequest, TaskResult, enable
from agentenv_protocol.a2a_agent.framework import _AGENT_DEFINITION

TOOLS = ("browser-harness", "agent-browser", "playwright-mcp", "chrome-devtools-mcp", "browser-use", "stagehand",
         "opencode-browse")
TOOL = os.environ.get("BU_BENCH_TOOL", "browser-harness")
if TOOL not in TOOLS:
    raise SystemExit(f"BU_BENCH_TOOL must be one of {', '.join(TOOLS)}, not {TOOL!r}")
AGENT_NAME = {"browser-harness": "codex-bu-cli", "browser-use": "codex-browser-use-mcp"}.get(TOOL, f"codex-{TOOL}")

WORKSPACE = Path(os.environ.get("CODEX_WORKSPACE_DIR", "/app"))
SKILL = Path("/opt/bh/SKILL.md")
TOOLS_DIR = Path("/opt/tools")
NPM = TOOLS_DIR / "npm" / "node_modules"
NODE = "/usr/local/bin/node"
SESSION = "agentenv-task"  # one task per container
OVERRIDES = ('web_search="disabled"', "project_doc_max_bytes=0")
# Kept out of the workspace: /app is the agent's, and its snapshot carries it.
AGENT_BROWSER_CONFIG = Path("/tmp/agent-browser.json")
STAGEHAND_KEY = Path("/tmp/stagehand.key")
BROWSE_HOME = Path("/tmp/browse-home")
SECRET_ENV = ("LITELLM_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY")

TOOL_RULES = {
    "browser-harness": f"Use Browser Harness, the `browser-harness` command, for every web interaction. First read its "
                       f"instructions at {SKILL}.",
    "agent-browser": "Use Agent Browser, the `agent-browser` command, for every web interaction. "
                     "First run `agent-browser skills get core` and follow it.",
    "playwright-mcp": "Use the `playwright` MCP tools for every web interaction.",
    "chrome-devtools-mcp": "Use the `chrome-devtools` MCP tools for every web interaction.",
    "browser-use": "Use the `browser_use` MCP tools for every web interaction.",
    "stagehand": "Use the `stagehand` MCP tools for every web interaction.",
    "opencode-browse": "Use the `browse` MCP tools for every web interaction.",
}


def compose_prompt(task: str, tool: str = TOOL) -> str:
    """codex-browser-tools compose_prompt for ``tool``, byte for byte, with this image's paths."""
    rules = [
        TOOL_RULES[tool],
        ("Get website content only through that browser tool. Do not use web search, curl, wget, requests, "
         "another browser, or another browser automation library."),
        ("A cloud browser is already running and connected to that tool. Do not start, stop, close, or replace "
         "the browser."),
        ("This message and the browser tool's own documentation are the only instructions. Do not look for or "
         "read any README.md or AGENTS.md."),
        f"Work only inside {WORKSPACE}. Save generated deliverables under {WORKSPACE / 'agent_outputs'}.",
        "Do not ask questions. Make the best deterministic attempt with the available page state.",
        "End with `FINAL ANSWER: <your result>`; include every value the task asks for.",
    ]
    lines = "\n".join(f"- {rule}" for rule in rules)
    return f"You are completing one browser benchmark task in an isolated CI VM.\n\nRules:\n{lines}\n\nTask:\n" \
        f"{task}\n"


_cdp_url = ""  # this turn's browser; one task per container


def scrub(text: str, url: str | None = None) -> str:
    """``text`` without the CDP address, its host or the host's first label (a cloud browser's id)."""
    url = _cdp_url if url is None else url
    host = urlparse(url).hostname or ""
    for secret in sorted({url, host, host.split(".")[0] if host.count(".") >= 2 else ""}, key=len, reverse=True):
        if len(secret) >= 8:
            text = text.replace(secret, "<cdp>")
    return text


def mcp_server(tool: str, ws_url: str, model: str = "", effort: str = "") -> dict | None:
    """The tool's MCP server, as the bake-off's mcp_config gave it: defaults except the browser and file locations."""
    if tool == "playwright-mcp":
        return {"name": "playwright", "command": NODE, "env": {}, "args": [
            str(NPM / "@playwright" / "mcp" / "cli.js"), "--cdp-endpoint", ws_url,
            "--output-dir", str(WORKSPACE / "playwright-output")]}
    if tool == "chrome-devtools-mcp":
        return {"name": "chrome-devtools", "command": NODE, "env": {}, "args": [
            str(NPM / "chrome-devtools-mcp" / "out" / "Default" / "gen" / "src" / "bin" / "chrome-devtools-mcp.js"),
            "--wsEndpoint", ws_url, "--workspace", str(WORKSPACE), "--no-usage-statistics"]}
    if tool == "browser-use":
        # browser-use's own MCP server (browser_exec / browser_screenshot) drives the browser through its bundled
        # Browser Harness daemon, pinned by BU_CDP_WS. browser_login and browser_feedback act outside the task.
        runtime = f"/tmp/bu-{hashlib.sha256(str(WORKSPACE).encode()).hexdigest()[:12]}"
        return {"name": "browser_use", "command": "/opt/bu/bin/browser-use", "args": ["--cli-mcp"],
                "disabled_tools": ["browser_login", "browser_feedback"],
                "env": {"BU_CDP_WS": ws_url, "BU_NAME": "eval-browser-use", "BH_OPEN_LIVE_URL": "0",
                        "BH_TAB_MARKER": "0", "BH_HOME": str(WORKSPACE / ".browser-use-bh"),
                        "HOME": str(WORKSPACE / ".browser-use-home"), "BH_RUNTIME_DIR": runtime}}
    if tool == "stagehand":
        # Stagehand's own act/extract/observe calls use the turn's model and effort through AgentEnv's model endpoint;
        # the key is read from a file outside the workspace, never argv or Codex config.
        env = {"STAGEHAND_CDP_URL": ws_url, "STAGEHAND_MODEL_KEY_FILE": str(STAGEHAND_KEY),
               "STAGEHAND_LOG_DIR": str(WORKSPACE / "stagehand-logs"), "STAGEHAND_MODEL": f"openai/{model}",
               "STAGEHAND_REASONING_EFFORT": effort}
        if base := os.environ.get("LITELLM_BASE_URL", ""):
            env["STAGEHAND_BASE_URL"] = base.rstrip("/") + "/openai/v1"
        if headers := os.environ.get("LITELLM_EXTRA_HEADERS", ""):  # Codex hands an MCP server only what env names
            env["STAGEHAND_HEADERS"] = headers
        return {"name": "stagehand", "command": NODE, "args": [str(TOOLS_DIR / "stagehand" / "stagehand_mcp.mjs")],
                "env": env}
    if tool == "opencode-browse":
        # browse's MCP server talks to the relay start_browse runs; browse_shim.mjs stands in for its extension.
        # settings edits the (absent) extension's options; secrets_* manage the user's stored site secrets.
        return {"name": "browse", "command": NODE, "args": [str(NPM / "@opencode" / "browse" / "dist" / "cli.mjs"),
                                                            "mcp"],
                "disabled_tools": ["settings", "secrets_list", "secrets_status", "secrets_refresh", "secrets_run"],
                "env": browse_env()}
    return None


def mcp_overrides(mcp: dict | None) -> list[str]:
    """The bake-off's ``--config`` entries for the server, as ``-c`` overrides."""
    if mcp is None:
        return []
    name = mcp["name"]
    out = [f"mcp_servers.{name}.command={json.dumps(mcp['command'])}",
           f"mcp_servers.{name}.args={json.dumps(mcp['args'])}",
           f"mcp_servers.{name}.startup_timeout_sec=60",
           f"mcp_servers.{name}.tool_timeout_sec=120"]
    if mcp["env"]:
        out.append(f"mcp_servers.{name}.env={{" + ", ".join(f"{k}={json.dumps(v)}" for k, v in mcp["env"].items())
                   + "}")
    if mcp.get("disabled_tools"):
        out.append(f"mcp_servers.{name}.disabled_tools={json.dumps(mcp['disabled_tools'])}")
    return out


def browse_env() -> dict[str, str]:
    """browse's isolated state: one root, the port start_browse chose, no updates, and no relay of its own."""
    return {"OPENCODE_BROWSER_HOME": str(BROWSE_HOME),
            "OPENCODE_BROWSER_PORT": (BROWSE_HOME / "port").read_text().strip(),
            "OPENCODE_BROWSER_AUTO_UPDATE": "off", "OPENCODE_BROWSER_UPDATE_CHECK": "0",
            "OPENCODE_BROWSER_AUTOSTART": "0", "OPENCODE_BROWSER_MCP_DIRECT": "1"}


def tool_env() -> dict[str, str]:
    """This process's environment less model keys: what a tool process started here gets."""
    return {key: value for key, value in os.environ.items() if key not in SECRET_ENV}


def start_browse(ws_url: str) -> list[subprocess.Popen]:
    """browse's relay, then browse_shim.mjs standing in for its Chrome extension on the browser's CDP socket."""
    package = NPM / "@opencode" / "browse"
    BROWSE_HOME.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    (BROWSE_HOME / "port").write_text(str(port))
    env = tool_env()
    # Files the agent saves from execute land in the workspace; agent sessions never expire mid-task.
    env.update(browse_env(), OPENCODE_BROWSER_FILE_ROOTS=str(WORKSPACE), OPENCODE_BROWSER_IDLE_TAB_TTL_MS="0")
    relay = subprocess.Popen([NODE, str(package / "dist" / "cli.mjs"), "serve"], env=env, stdin=subprocess.DEVNULL,
                             stdout=(BROWSE_HOME / "relay.log").open("wb"), stderr=subprocess.STDOUT)
    shim_log = BROWSE_HOME / "shim.log"
    processes = [relay]
    deadline = time.monotonic() + 30
    try:
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/version", timeout=2):
                    break
            except OSError:
                if relay.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("browse relay did not start: " + tail(BROWSE_HOME / "relay.log")) from None
                time.sleep(0.2)
        shim = subprocess.Popen([NODE, str(TOOLS_DIR / "browse_shim.mjs")],
                                env={**env, "BROWSE_SHIM_CDP_URL": ws_url, "BROWSE_SHIM_PACKAGE": str(package)},
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=shim_log.open("wb"))
        processes.insert(0, shim)
        while '"relay.paired"' not in shim_log.read_text():
            if shim.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("browse shim did not pair with the relay: " + tail(shim_log))
            time.sleep(0.2)
    except BaseException:
        stop_processes(processes)
        raise
    return processes


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def tail(path: Path, limit: int = 600) -> str:
    try:
        return scrub(path.read_text(errors="replace")[-limit:])
    except OSError:
        return "(no log)"


def set_cli_env(tool: str, ws_url: str) -> None:
    """A CLI tool's settings, in this process's environment, which each Codex turn and its commands inherit; as the
    bake-off's tool_environment set them. Only the tool's own command goes on PATH."""
    if tool not in ("browser-harness", "agent-browser"):
        return
    os.environ["PATH"] = f"{TOOLS_DIR / 'path' / tool}:{os.environ['PATH']}"
    if tool == "browser-harness":
        os.environ.update({
            "BU_NAME": SESSION, "BU_CDP_WS": ws_url, "BH_HOME": str(WORKSPACE / ".browser-harness"),
            "BH_RUNTIME_DIR": f"/tmp/bh-{SESSION}", "BH_TMP_DIR": str(WORKSPACE / "browser-harness-log"),
            "BH_AGENT_WORKSPACE": str(WORKSPACE / ".browser-harness-workspace"), "BH_OPEN_LIVE_URL": "0",
            # The horse marker rewrites page titles for humans watching; the agent should see the real title.
            "BH_TAB_MARKER": "0",
        })
    else:
        # The core skill tells agents to mint their own session ids; the config-file CDP endpoint applies to every
        # session name, so none of them launches a browser of its own.
        AGENT_BROWSER_CONFIG.write_text(json.dumps({"cdp": ws_url}))
        AGENT_BROWSER_CONFIG.chmod(0o600)
        os.environ.update({"AGENT_BROWSER_SESSION": SESSION, "AGENT_BROWSER_CONFIG": str(AGENT_BROWSER_CONFIG)})
        done = subprocess.run(["agent-browser", "connect", ws_url], cwd=WORKSPACE, env=tool_env(),
                              capture_output=True, text=True, timeout=60, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"agent-browser connect failed: {scrub(done.stderr.strip()[:500])}")


def stop_tool(tool: str) -> None:
    """The bake-off's stop_tool: a CLI tool's daemon is stopped after the turn (best effort)."""
    command = {"agent-browser": ["agent-browser", "close"],
               "browser-harness": ["/opt/bh/bin/python", "-c",
                                   f"from browser_harness.admin import restart_daemon; restart_daemon({SESSION!r})"],
               }.get(tool)
    if command is not None:
        try:
            subprocess.run(command, cwd=WORKSPACE, env=tool_env(), capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            pass


_mcp: list[str] = []  # this turn's MCP server overrides
_run_codex = a2a_server.run_codex


async def run_codex(*, prompt: str, config_overrides=(), event_callback=None, **kwargs):
    def record(event: dict) -> None:
        event_callback(json.loads(scrub(json.dumps(event))))

    try:
        result = await _run_codex(prompt=compose_prompt(prompt), config_overrides=[*config_overrides, *OVERRIDES,
                                                                                    *_mcp],
                                  event_callback=record if event_callback else None, **kwargs)
    except Exception as error:  # noqa: BLE001 - re-raised with the browser address scrubbed
        raise RuntimeError(scrub(str(error))) from None
    if dataclasses.is_dataclass(result):  # its texts: the reply, errors, warnings (sent in the turn's data) and events
        result = dataclasses.replace(result, **{
            field.name: json.loads(scrub(json.dumps(value))) for field in dataclasses.fields(result)
            if field.init and isinstance(value := getattr(result, field.name), (str, list))})
    return result


class _ScrubbedLog(logging.Filter):
    """Codex's bridge logs the turn's errors and warnings before run_codex returns them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = scrub(record.getMessage()), None
        return True


logging.getLogger("codex_bridge").addFilter(_ScrubbedLog())  # the module a2a_server imports run_codex from


a2a_server.run_codex = run_codex


CodexConfig = a2a_server.CodexConfig


class CodexToolAgent(a2a_server.CodexAgent):
    async def run(self, request: TaskRequest[CodexConfig]) -> TaskResult:
        global _cdp_url
        endpoint = request.cdp_endpoint()
        ws_url = _cdp_url = endpoint["ws_url"]
        (WORKSPACE / "agent_outputs").mkdir(parents=True, exist_ok=True)
        processes: list[subprocess.Popen] = []
        try:
            set_cli_env(TOOL, ws_url)
            if TOOL == "opencode-browse":
                processes = start_browse(ws_url)
            if TOOL == "stagehand":
                STAGEHAND_KEY.touch(mode=0o600)
                STAGEHAND_KEY.write_text(os.environ.get("LITELLM_API_KEY", ""))
            _mcp[:] = mcp_overrides(mcp_server(TOOL, ws_url, request.config.model or "", request.config.effort or ""))
        except Exception as error:  # noqa: BLE001 - the turn fails as an infra error, scrubbed
            stop_processes(processes)
            message = scrub(f"{TOOL} setup failed: {error}")[:2000]
            return TaskResult.builder().failed("infra_error", message, error_type="infra_error").add_text(
                message).build()
        try:
            return await super().run(request)
        finally:
            stop_tool(TOOL)
            stop_processes(processes)
            STAGEHAND_KEY.unlink(missing_ok=True)


_definition = getattr(a2a_server.CodexAgent, _AGENT_DEFINITION)
setattr(CodexToolAgent, _AGENT_DEFINITION, dataclasses.replace(
    _definition,
    identity=dataclasses.replace(_definition.identity, name=AGENT_NAME,
                                 description=f"Codex CLI with {TOOL} as its browser (BU Bench harness bake-off arm)."),
    extensions=(*_definition.extensions, enable(CDP_CONFIG_V1, description=f"The browser {TOOL} drives.")),
))

app = a2a_server.build_app(CodexToolAgent())
AGENT_CARD = app.state.agentenv_a2a.card
