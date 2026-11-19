"""The ``bu_browser`` env type: a multi env (the bu-browser bundle's ``bu-browser``) that stops its site's Browser Use
Cloud browser when it closes.

A run's teardown closes each env it deployed before it terminates any sandbox, so ``close`` reaches the site through
the gateway with its ``close`` extension, which stops the browser for good, whatever provider the sandbox is on: a local
container's teardown is ``docker rm -f``, which gives the site no SIGTERM to stop it on. Not ``data/reset``, which a
``reset_env`` step sends too and after which a late tool call starts a fresh browser that nothing would stop. It tries
once, waiting at most STOP_TIMEOUT_S, and never raises: the browser's own timeout ends one it couldn't stop.

One env object is deployed once until it is closed: a second deploy would leave ``close`` reaching only the new site,
and AgentEnv would end the earlier one's sandbox without its ``close``, its browser billing to its timeout.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import ClassVar

from agent_env.env.envs.multi_env import MultiEnv

logger = logging.getLogger(__name__)

#: The site's stop takes at most 63 s (its STOP_ATTEMPTS requests of STOP_REQUEST_S and the waits between them,
#: cloud.py), plus a create still in flight at teardown, which is waited for so its browser can be stopped; teardown
#: gives each env's close 120 s. Also the extension call's own timeout, whose default is 30 s.
STOP_TIMEOUT_S = 115.0
#: The site's extension that stops its browser for good (envs/bu-browser-site/browser_site.py).
CLOSE_URI = "urn:agentenv-benchmarks:bu-browser-close/v1"


class BuBrowserEnv(MultiEnv):
    type: ClassVar[str] = "bu_browser"
    description = ("A multi env whose website serves Playwright MCP's tools on a Browser Use Cloud browser, which it "
                   "stops as it closes")

    @functools.wraps(MultiEnv.deploy)  # its signature, which deploy_env reads to know which arguments it takes
    async def deploy(self, *args, **kwargs):
        if self._env_provider is not None:
            raise RuntimeError(f"env {self.id} is already deployed: close it first, or deploy a new env object (a "
                               "second deploy would leave the first deployment's browser running to its timeout)")
        return await super().deploy(*args, **kwargs)

    async def close(self) -> None:
        if self._deployed is not None:
            try:
                async with asyncio.timeout(STOP_TIMEOUT_S):
                    for site in self.website_envs:
                        await self._deployed.invoke(CLOSE_URI, "close", environment_name=site.environment_name,
                                                    timeout=int(STOP_TIMEOUT_S))
            except Exception as error:  # noqa: BLE001 - the browser's own timeout is the backstop
                logger.warning("env %s: stopping its cloud browser failed (%s: %s); its timeout will end it",
                               self.id, type(error).__name__, error)
        await super().close()
