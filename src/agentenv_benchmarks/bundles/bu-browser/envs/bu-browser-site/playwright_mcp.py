"""Playwright MCP on a Browser Use Cloud browser: its tools, as the manifest the image's build wrote lists them, and one
browser, one Playwright MCP process attached to it over CDP and one MCP session on that process, started together and
stopped together.

Adapted from agentenv-framework 0.9.1573's page driver (src/agent_env/drivers/envs/page/playwright_mcp.py, Apache-2.0):
its process start and stop, its session held open by a task of its own, and its environment allowlist, cut down to
one session and given the cloud browser to own. Playwright MCP runs as the unprivileged user ``pw``, so nothing an
agent's tools reach can read the site's settings, which hold the Browser Use API key.

``python playwright_mcp.py --write-manifest`` is the image build's step: it checks the installed Playwright MCP is
VERSION and writes TOOLS from its own ``tools/list``, so the tools served can't drift from the version installed.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import cloud
import httpx
from agentenv_protocol import ForwardedTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

#: The Playwright MCP the image installs, the same pin as BU Bench's codex-browser-tools image.
VERSION = "0.0.85"
HERE = Path(__file__).resolve().parent
CLI = HERE / "node_modules" / "@playwright" / "mcp" / "cli.js"
TOOLS = HERE / "playwright-mcp-tools.json"
#: Runs the agent's JavaScript in the page: hidden until a task's tool access names it, as the page driver hides it.
HIDDEN_TOOLS = frozenset({"browser_evaluate"})
#: Never served: browser_run_code_unsafe runs the agent's code inside this container, beside the site that holds the
#: key; browser_close, on a browser attached over CDP, can end Browser Use's browser and its session with it.
DROPPED_TOOLS = frozenset({"browser_run_code_unsafe", "browser_close"})
#: Who Playwright MCP runs as, and all it keeps of the site's environment.
USER = "pw"
KEPT_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "TZ")
#: Root's folder where each Playwright MCP gets an output folder of its own, deleted once that one has exited: nothing
#: a task saved (a screenshot, a PDF, a download) is left for a later task's browser_file_upload to send.
OUTPUT = Path("/home") / USER / "playwright-output"
LOG = Path(tempfile.gettempdir()) / "bu-browser-playwright-mcp.log"
#: How Playwright MCP answers an action (navigate, click, type...): the page's snapshot written to a file in its output
#: folder and only a link to it, which a model with no file tool can't open. --snapshot-mode has no inline choice.
SNAPSHOT_LINK = re.compile(r"^- \[Snapshot\]\(\./(page-[^/\\()]+\.yml)\)$", re.MULTILINE)
START_S = 30.0
CALL_TIMEOUT_S = 300.0
STOP_S = 10.0
POLL_S = 0.1
#: How long a check that Browser Use still runs the browser waits for its CDP discovery.
PROBE_S = 10.0
#: The wait before a check whose connection was refused asks once more.
PROBE_RETRY_S = 1.0
#: What a call's result starts with when its browser replaced one that ended without a reset.
REPLACED = ("The browser ended and this call ran on a fresh one: the pages, tabs and anything signed in before it are "
            "gone.")

logger = logging.getLogger(__name__)


@functools.cache
def manifest() -> list[dict]:
    return json.loads(TOOLS.read_text(encoding="utf-8"))["tools"]


def command(port: int, output: Path) -> list[str]:
    """Playwright MCP on loopback ``port``, as ``pw`` with no groups, ended by the kernel if the site dies first. Its
    other flags are BU Bench's Playwright MCP arm's; the browser it attaches to is in its environment."""
    return ["setpriv", f"--reuid={USER}", f"--regid={USER}", "--clear-groups", "--pdeathsig", "TERM", "--",
            shutil.which("node") or "node", str(CLI), "--host", "127.0.0.1", "--port", str(port), "--ignore-roots",
            "--output-dir", str(output)]


def environment(endpoint: str) -> dict[str, str]:
    """The browser at ``endpoint`` to attach to, never one of its own, as --cdp-endpoint would name it: a credential,
    which a command line shows any user (ps, /proc/<pid>/cmdline) and an environment only its own."""
    return {**{name: os.environ[name] for name in KEPT_ENVIRONMENT if name in os.environ},
            "HOME": str(OUTPUT.parent), "USER": USER, "PLAYWRIGHT_MCP_CDP_ENDPOINT": endpoint}


class PlaywrightMcp:
    """The browser ``start_browser`` starts (a ``cloud.CloudBrowser``), Playwright MCP on it and one session there.
    ``start`` brings all three up once, shared by every caller, and again after ``stop`` or once one has ended, which
    the first result on the new browser says (REPLACED); ``stop`` ends them, a start under way included, and starts
    nothing, so a reset leaves no browser running; ``close`` stops them for good: no call starts a browser after it."""

    def __init__(self, start_browser: Callable[[], Awaitable[Any]]) -> None:
        self._start_browser = start_browser
        self._lock = asyncio.Lock()
        self._starting: asyncio.Task | None = None
        self.browser: Any = None
        self._process: asyncio.subprocess.Process | None = None
        self._output: Path | None = None
        self._session: _Session | None = None
        self._orphans: list[asyncio.Task] = []
        self.closed = False

    def tools(self) -> list[ForwardedTool]:
        return [ForwardedTool.listed(t, functools.partial(self.call, t["name"]),
                                     hidden_by_default=t["name"] in HIDDEN_TOOLS)
                for t in manifest() if t["name"] not in DROPPED_TOOLS]

    @property
    def running(self) -> bool:
        return self.browser is not None or self._process is not None

    async def start(self) -> _Session:
        async with self._lock:
            if self.closed:
                raise RuntimeError("the env has closed, so no browser starts")
            if self._starting is None or self._starting.done() and not self._up():
                ended = self._starting is not None and _succeeded(self._starting)  # a browser this one replaces
                self._starting = asyncio.create_task(self._start(REPLACED if ended else None))
            starting = self._starting
        return await asyncio.shield(starting)

    async def stop(self, held: _Session | None = None) -> None:
        """Stop everything, or with ``held`` only while that session is still the one running."""
        async with self._lock:
            if held is not None and self._session is not held:
                return
            if cloud.UNSTOPPED:  # alongside the rest, so close's time goes to both at once
                self._orphans.append(asyncio.ensure_future(_stop_unstopped()))
            starting, self._starting = self._starting, None
            if starting is not None:
                starting.cancel()
                self._orphans.append(starting)
                await asyncio.shield(asyncio.gather(starting, return_exceptions=True))
            await self._close()
            # Shielded and listed until done: a stop cancelled here leaves them running, for close to wait on.
            while pending := [task for task in self._orphans if not task.done()]:
                await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
            self._orphans = []

    async def close(self) -> None:
        """Stop everything for good, as the run tears the env down: a call that comes late starts no browser, which
        nothing would stop once the container is removed."""
        self.closed = True
        await self.stop()

    async def call(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """``tool`` called on Playwright MCP, its text scrubbed of the browser's CDP address, or refused saying why.
        An error from a browser Browser Use has ended (its timeout, a crash) stops what is left of it and says so, so
        the next call starts a fresh browser rather than failing on the dead one for the rest of the episode."""
        try:
            held = await self.start()
        except asyncio.CancelledError:
            if (current := asyncio.current_task()) is not None and current.cancelling():
                raise
            return _refused(f"the browser was reset before {tool} ran")
        except Exception as e:  # noqa: BLE001 - the agent is told why, as the page driver tells it
            # cloud.CloudBrowser.start's errors name the browser by its label only, never its CDP address.
            return _refused(f"the browser didn't start: {type(e).__name__}: {e}")
        called = asyncio.ensure_future(held.session.call_tool(
            tool, arguments, read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT_S)))
        try:
            await asyncio.wait({called, held.owner}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            called.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await called
            raise
        if not called.done():
            called.cancel()
            return _refused(f"the browser was reset or Playwright MCP ended during {tool}, which may have acted")
        try:
            result = _inlined(called.result(), held.output)
        except Exception as e:  # noqa: BLE001 - the call may have acted: refused, not retried
            if _timed_out(e):  # Playwright MCP is stuck on it, and every later call on it would wait as long
                await self.stop(held)
                return _refused(f"Playwright MCP didn't answer {tool} within {CALL_TIMEOUT_S:.0f} s, which may have "
                                "acted; the browser was stopped, its pages are gone, and the next browser tool call "
                                "starts a fresh browser")
            return _refused(cloud.scrub(f"Playwright MCP didn't answer {tool}: {type(e).__name__}: {e}", held.secrets))
        if result.isError and await _ended(held.browser):
            await self.stop(held)
            return _refused(f"the browser ended during {tool} (Browser Use stopped it at its time limit, or it "
                            "crashed); its pages are gone, and the next browser tool call starts a fresh browser")
        result = _scrubbed(result, held.secrets)
        if notice := held.notice:
            held.notice = None
            result = result.model_copy(update={"content": [TextContent(type="text", text=notice), *result.content]})
        return result

    def _up(self) -> bool:
        starting = self._starting
        return (starting is not None and starting.done() and not starting.cancelled() and starting.exception() is None
                and self.browser is not None and not self.browser.stopped
                and self._process is not None and self._process.returncode is None
                and self._session is not None and not self._session.owner.done())

    async def _start(self, notice: str | None) -> _Session:
        await self._close()  # what an ended start left: a stopped browser, an exited process
        browser_start = asyncio.ensure_future(self._start_browser())
        try:
            self.browser = await asyncio.shield(browser_start)
        except asyncio.CancelledError:
            # Cancelled mid-request, Browser Use may create the browser anyway: wait for it, then stop it.
            self._orphans.append(asyncio.ensure_future(_stop_once_started(browser_start)))
            raise
        try:
            port = _free_port()
            OUTPUT.mkdir(parents=True, exist_ok=True)
            self._output = Path(tempfile.mkdtemp(dir=OUTPUT))
            shutil.chown(self._output, USER)
            log = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)  # it can name the CDP address
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *command(port, self._output), cwd=self._output, env=environment(self.browser.ws_url),
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=log)
            finally:
                os.close(log)
            await _listening(port, self._process)
            self._session = await _Session.open(f"http://127.0.0.1:{port}/mcp", self.browser, self._output, notice)
        except BaseException:
            await self._close()
            raise
        logger.info("cloud browser %s is up with Playwright MCP %s", self.browser.label, VERSION)
        return self._session

    async def _close(self) -> None:
        """Stop the browser, its request sent first, alongside ending the session and Playwright MCP: a reset's time
        goes to the stop, which BU bills until, and not to waiting on a process before it is sent."""
        session, process, browser, output = self._session, self._process, self.browser, self._output
        self._session, self._process, self.browser, self._output = None, None, None, None
        stopping = asyncio.ensure_future(_stop(browser)) if browser is not None else None
        try:
            if session is not None:
                session.close.set()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.gather(session.owner, return_exceptions=True), STOP_S)
            if process is not None and process.returncode is None:
                process.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), STOP_S)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        finally:
            if process is not None and process.returncode is None:
                process.kill()  # cancelled before it was ended: never leave Playwright MCP running
            if output is not None:
                shutil.rmtree(output, ignore_errors=True)  # links in it are removed, never followed
            if stopping is not None:
                try:
                    await asyncio.shield(stopping)
                except asyncio.CancelledError:  # the stop goes on: ``stop`` waits for it with the other orphans
                    self._orphans.append(stopping)
                    raise


def _succeeded(task: asyncio.Task) -> bool:
    return task.done() and not task.cancelled() and task.exception() is None


async def _ended(browser: Any) -> bool:
    """Whether Browser Use no longer runs ``browser``. Its CDP discovery, which needs no key, settles it when it answers
    (404 or 410: ended; a success: running) or refuses the connection twice. Anything else, a timeout or a 5xx (Browser
    Use's CDP gateway answers 502 for a browser it stopped), is settled by Browser Use's API: the browser's status. When
    that fails too, the browser is left running."""
    for attempt in range(2):
        if attempt:
            await asyncio.sleep(PROBE_RETRY_S)
        try:
            async with httpx.AsyncClient(timeout=PROBE_S) as client:
                status = (await client.get(f"{browser.cdp_url}/json/version")).status_code
        except httpx.ConnectError as error:
            if _refused_connection(error):
                continue
            break
        except httpx.HTTPError:
            break
        if status in (404, 410):
            return True
        if status < 400:
            return False
        break
    else:
        return True
    try:
        state = str((await browser.session()).get("status") or "").lower()
    except Exception:  # noqa: BLE001 - no answer either way: the browser is left running
        return False
    return bool(state) and state != "active"


def _timed_out(error: Exception) -> bool:
    """Whether a call failed waiting for its answer: mcp's request timeout (408), or the HTTP read under it."""
    return (isinstance(error, McpError) and error.error.code == httpx.codes.REQUEST_TIMEOUT
            or isinstance(error, (TimeoutError, httpx.TimeoutException)))


def _refused_connection(error: BaseException) -> bool:
    while error is not None:
        if isinstance(error, ConnectionRefusedError):
            return True
        error = error.__cause__ or error.__context__
    return False


async def _stop_once_started(browser_start: asyncio.Future) -> None:
    try:
        browser = await browser_start
    except Exception:  # noqa: BLE001 - a start that failed stopped what it had made (cloud.CloudBrowser.start)
        return
    await _stop(browser)


async def _stop(browser: Any) -> None:
    """Stop ``browser``; one whose stops all failed goes to cloud.UNSTOPPED, which every ``PlaywrightMcp.stop`` tries
    again, close's last."""
    await browser.stop()
    if browser.stopped:
        logger.info("cloud browser %s stopped", browser.label)
    else:
        cloud.UNSTOPPED.append(browser)


async def _stop_unstopped() -> None:
    """Stop again each browser whose stops all failed, as a failed start or an earlier stop left it, and drop those
    that went through."""
    await asyncio.gather(*(browser.stop() for browser in list(cloud.UNSTOPPED)))
    for browser in [browser for browser in cloud.UNSTOPPED if browser.stopped]:
        cloud.UNSTOPPED.remove(browser)
        logger.info("cloud browser %s stopped", browser.label)


class _Session:
    """One MCP session on Playwright MCP, held open by a task of its own until ``close`` is set, since mcp's client
    contexts are anyio task groups, which only the task that entered one can leave."""

    def __init__(self, url: str, browser: Any, output: Path, notice: str | None) -> None:
        self.url, self.browser, self.output, self.notice = url, browser, output, notice
        self.secrets: list[str] = list(browser.secrets)
        self.close = asyncio.Event()
        self.session: ClientSession | None = None
        self.owner: asyncio.Task | None = None

    @classmethod
    async def open(cls, url: str, browser: Any, output: Path, notice: str | None) -> _Session:
        held = cls(url, browser, output, notice)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        held.owner = asyncio.create_task(held._own(ready))
        try:
            held.session = await asyncio.wait_for(ready, START_S)
        except BaseException:
            held.close.set()
            held.owner.cancel()
            raise
        return held

    async def _own(self, ready: asyncio.Future) -> None:
        try:
            timeout = httpx.Timeout(CALL_TIMEOUT_S, connect=START_S)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http, \
                    streamable_http_client(self.url, http_client=http) as (read, write, _), \
                    ClientSession(read, write) as session:
                await session.initialize()
                if ready.done():
                    return
                ready.set_result(session)
                await self.close.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            elif not isinstance(e, asyncio.CancelledError):
                logger.warning("the session on Playwright MCP ended: %s", type(e).__name__)
            if not isinstance(e, Exception):
                raise


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _listening(port: int, process: asyncio.subprocess.Process) -> None:
    deadline = asyncio.get_running_loop().time() + START_S
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"Playwright MCP exited at start with {process.returncode}; its log is {LOG}")
        with contextlib.suppress(OSError):
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        await asyncio.sleep(POLL_S)
    raise RuntimeError(f"Playwright MCP didn't listen within {START_S:.0f} s")


def _inlined(result: CallToolResult, output: Path) -> CallToolResult:
    """``result`` with each page snapshot Playwright MCP linked to in its ``output`` folder in place of its link, as it
    renders one inline, and the file deleted: the model sees the page it acted on, and the folder doesn't grow with
    every action. Only a regular file is read, never what a link there, which the pw user writes, points at: this root
    process reads it."""
    def snapshot(link: re.Match) -> str:
        path = output / link.group(1)
        try:
            # No link followed, and no wait on a FIFO's writer.
            with open(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), encoding="utf-8") as file:
                if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                    return link.group(0)
                page = file.read()
        except OSError:
            return link.group(0)
        path.unlink(missing_ok=True)
        return f"```yaml\n{page.rstrip()}\n```"

    content = [part.model_copy(update={"text": SNAPSHOT_LINK.sub(snapshot, part.text)})
               if isinstance(part, TextContent) else part for part in result.content]
    return result.model_copy(update={"content": content})


def _scrubbed(result: CallToolResult, secrets: list[str]) -> CallToolResult:
    content = [part.model_copy(update={"text": cloud.scrub(part.text, secrets)}) if isinstance(part, TextContent)
               else part for part in result.content]
    structured = cloud.scrub_json(result.structuredContent, secrets) if result.structuredContent else None
    return result.model_copy(update={"content": content, "structuredContent": structured})


def _refused(message: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])


def write_manifest() -> None:
    """Check the installed Playwright MCP is VERSION, and write TOOLS from its ``tools/list``, run attached to no
    browser, as agentenv-framework's scripts/upgrade_playwright_mcp.py lists it for the page driver."""
    node = shutil.which("node") or "node"
    said = subprocess.run([node, str(CLI), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if said.split()[-1] != VERSION:
        raise SystemExit(f"Playwright MCP says {said!r}; this site serves {VERSION}'s tools")
    messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "bu-browser-build", "version": "0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    # stdin stays open until the answer is read: Playwright MCP exits at its end, before a long answer is all written.
    with subprocess.Popen([node, str(CLI), "--cdp-endpoint", "http://127.0.0.1:9/"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as server:
        timer = threading.Timer(120, server.kill)
        timer.start()
        try:
            server.stdin.write("".join(json.dumps(message) + "\n" for message in messages))
            server.stdin.flush()
            tools = next(answer["result"]["tools"] for answer in map(json.loads, server.stdout)
                         if answer.get("id") == 2)
        finally:
            timer.cancel()
            server.stdin.close()
    if not any(t["name"] == "browser_navigate" for t in tools):
        raise SystemExit(f"Playwright MCP {VERSION} lists no browser_navigate: {[t['name'] for t in tools]}")
    TOOLS.write_text(json.dumps({"version": VERSION, "tools": tools}, indent=1) + "\n", encoding="utf-8")
    print(f"Playwright MCP {VERSION}: {len(tools)} tools, serving "
          f"{len([t for t in tools if t['name'] not in DROPPED_TOOLS])}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-manifest"]:
        raise SystemExit("usage: python playwright_mcp.py --write-manifest")
    write_manifest()
