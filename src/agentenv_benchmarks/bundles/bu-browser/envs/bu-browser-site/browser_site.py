"""A Browser Use Cloud browser as a website env that is an MCP server: Playwright MCP's tools, served on the browser.

The site starts a cloud browser, and Playwright MCP on it, at the first tool call, so no browser runs, or counts down
its ``timeout_minutes``, before the run has recorded the env it would have to stop; it is ready once its key is set.
``data/reset`` stops the browser and starts nothing: the next tool call starts a fresh one. Its ``close`` extension,
which the run's teardown calls (``bu_browser``), stops it for good: no call starts one after it. The server's shutdown,
SIGTERM included (once no tool call is in flight), stops it too, and the browser's own ``timeout_minutes`` ends it if nothing else does.

``api_key`` is a ``SecretStr`` setting, so its env.toml value is a ``secret:`` reference agentenv resolves only as it
writes the site's container; it stays in this process, and Playwright MCP, as another user, never sees it.
"""
from __future__ import annotations

import contextlib

import cloud
import playwright_mcp
from agentenv_protocol import add_data, extension, get_data, reset_data
from agentenv_protocol.website import WebsiteEnvironment, ready_check, website
from pydantic import BaseModel, SecretStr
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route

API_URL = "https://api.browser-use.com/api/v3"
#: The extension the run's teardown stops the browser for good with (agentenv_benchmarks.bu_browser.env).
CLOSE_URI = "urn:agentenv-benchmarks:bu-browser-close/v1"


async def _page(request) -> HTMLResponse:
    return HTMLResponse("<!doctype html><title>bu-browser</title><p>A Browser Use Cloud browser, driven through this "
                        "site's MCP tools at /mcp.</p>")


@website(domains=["bu-browser.localhost"], clock="real", builtin_routes=False, ready_timeout_s=180)
class BuBrowser(WebsiteEnvironment):
    class Settings(BaseModel):
        api_key: SecretStr = SecretStr("")
        api_url: str = API_URL
        proxy_country_code: str = "us"
        #: Browser Use ends the browser after this long, counted from the first tool call: at least a task's turns.
        timeout_minutes: int = 30

    #: The one page, at /: the browser is Browser Use's, reached only through the tools.
    app = Starlette(routes=[Route("/", _page)])

    def __init__(self) -> None:
        super().__init__()
        # cloud.py is BU Bench's, unchanged: it reads its API and key from these, here the site's settings.
        cloud.API = self.settings.api_url.rstrip("/")
        cloud.api_key = lambda: self.settings.api_key.get_secret_value().strip()
        self.playwright = playwright_mcp.PlaywrightMcp(self._start_browser)

    async def _start_browser(self) -> cloud.CloudBrowser:
        return await cloud.CloudBrowser.start(proxy_country_code=self.settings.proxy_country_code or None,
                                              timeout_minutes=self.settings.timeout_minutes, enable_recording=False)

    def forwarded_tools(self) -> list:
        return self.playwright.tools()

    def create_app(self):
        """The site's app, whose lifespan also stops the browser at shutdown, which uvicorn runs on SIGTERM before it
        exits."""
        app = super().create_app()
        served = app.router.lifespan_context

        @contextlib.asynccontextmanager
        async def lifespan(scope_app):
            async with served(scope_app):
                try:
                    yield
                finally:
                    await self.playwright.stop()

        app.router.lifespan_context = lifespan
        return app

    @ready_check
    async def api_key_set(self) -> bool:
        """The key is set. It is first used at the first tool call, whose result says why if no browser starts."""
        if not self.settings.api_key.get_secret_value().strip():
            raise RuntimeError("the api_key setting is empty; set it to secret:BROWSER_USE_API_KEY")
        return True

    @reset_data
    async def stop_browser(self) -> None:
        """Stop the browser and Playwright MCP, starting nothing; the next tool call starts a fresh browser."""
        await self.playwright.stop()

    @extension(CLOSE_URI, description="Stop the browser for good: tool calls after it start none.")
    async def close(self) -> None:
        await self.playwright.close()

    @get_data
    def export(self) -> list:
        """Nothing: what a task leaves is in its replies, and the browser is gone with its reset."""
        return []

    @add_data
    def load(self, parts: list) -> None:
        if parts:
            raise ValueError("bu-browser-site takes no data")


if __name__ == "__main__":
    BuBrowser().serve()
