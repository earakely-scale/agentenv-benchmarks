"""A stand-in for Playwright MCP in tests/test_bu_browser.py: an MCP server on loopback ``--port`` whose tools say
which CDP endpoint it was given (PLAYWRIGHT_MCP_CDP_ENDPOINT) and which environment variables it has, as the real one
would be given them, link a page snapshot as the real one's actions do, fail as a tool does, never answer, or end the
server. With ``--ignore-term`` it outlives a SIGTERM, as a hung Playwright MCP would."""
import asyncio
import contextlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP


@contextlib.asynccontextmanager
async def lifespan(app):
    if "--ignore-term" in sys.argv:  # here, since uvicorn sets its own handler as it starts
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    yield


endpoint = os.environ["PLAYWRIGHT_MCP_CDP_ENDPOINT"]
server = FastMCP("fake-playwright-mcp", host="127.0.0.1", port=int(sys.argv[sys.argv.index("--port") + 1]),
                 lifespan=lifespan)


@server.tool()
def browser_navigate(url: str) -> str:
    """As Playwright MCP 0.0.85 answers an action: the page's snapshot written to a file in its working directory (its
    output folder), and only a link to it."""
    snapshot = f"page-{time.time_ns()}.yml"
    Path(snapshot).write_text(f'- heading "Snapshot of the page" [level=1] [ref=e1]\n- paragraph [ref=e2]: {endpoint}\n')
    return f"navigated to {url} over {endpoint}\n### Snapshot\n- [Snapshot](./{snapshot})"


@server.tool()
def browser_take_screenshot() -> str:
    """Saves a file in its working directory (its output folder), as the real one's screenshots, PDFs and downloads."""
    saved = Path(f"page-{time.time_ns()}.png").resolve()
    saved.write_bytes(b"\x89PNG")
    return str(saved)


@server.tool()
def browser_environment() -> str:
    return json.dumps(sorted(os.environ))


@server.tool()
def browser_click(ref: str) -> str:
    raise RuntimeError(f"Target page, context or browser has been closed (clicking {ref})")


@server.tool()
async def browser_hang() -> str:
    """Never answers, as a Playwright MCP stuck on a page would."""
    await asyncio.sleep(3600)
    return "late"


@server.tool()
def browser_crash() -> str:
    """Answers, then the server dies, as Playwright MCP would on a crash."""
    threading.Timer(0.2, os._exit, [1]).start()
    return "crashing"


server.run(transport="streamable-http")
