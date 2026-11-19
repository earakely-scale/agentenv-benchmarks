"""Browser Use Cloud browsers: create, connect, screenshot, stop.

The request is the one the harness bake-off sent for every arm (Browser Use's eval platform, harness codex-browser-tools):
POST /api/v3/browsers {proxyCountryCode, timeout, enableRecording}, then the CDP WebSocket from cdpUrl/json/version,
PATCH {action: stop} when done, and the billed cost from GET /browsers/{id}.

A running browser's CDP address is its credential: ``https://<browser id>.cdp.browser-use.com`` answers
``/json/version`` without a key, so the full browser id is one too. Only ``label`` (its first 8 characters) is logged
or stored, error text is scrubbed of the address before it is kept, and the live-view link, which embeds the
WebSocket URL, is never kept at all.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

#: Browsers whose stop never went through, listed until it does: whoever still can (an exit hook, a close) stops them.
UNSTOPPED: list = []

#: What httpx's request log line (INFO, logger ``httpx``) must not keep: a browser's CDP host, its id in the API's
#: path, and any query string, where a signed link such as the recording's carries its signature.
LOGGED_URL_SECRETS = ((re.compile(r"[\w-]+\.cdp[\w.-]*\.browser-use\.com"), "<cdp>"),  # also cdp.eu., cdp0.
                      (re.compile(r"/browsers/[^/?#\s\"]+"), "/browsers/<id>"),
                      (re.compile(r"\?[^\s\"]*"), "?<redacted>"))
#: Hosts of the recording links this process read: their requests are scrubbed too.
RECORDING_HOSTS: set[str] = set()


def _scrub_logged_urls(record: logging.LogRecord) -> bool:
    """httpx logs every request's URL at INFO, the level ``agent-env up`` logs at. Only requests to Browser Use, its
    configured API or a recording link are scrubbed: the filter is process-wide, and other requests are not ours."""
    message = record.getMessage()
    if not any(host and host in message for host in ("browser-use.com", urlparse(API).hostname, *RECORDING_HOSTS)):
        return True
    for pattern, replacement in LOGGED_URL_SECRETS:
        message = pattern.sub(replacement, message)
    record.msg, record.args = message, None
    return True


logging.getLogger("httpx").addFilter(_scrub_logged_urls)

API = os.environ.get("BROWSER_USE_API_URL", "https://api.browser-use.com/api/v3")
KEY_NAME = "BROWSER_USE_API_KEY"
#: Stop requests per browser, the wait for each, and the wait before each retry (times the attempt): one 503 must not
#: leave it billing. All of them take at most 63 s, inside bu-browser's 115 s close.
STOP_ATTEMPTS = 3
STOP_REQUEST_S = 20.0
STOP_RETRY_S = 1.0
#: Browser Use can take minutes to answer a create, and a client that gives up never learns the id of the browser it
#: made, which then bills until its timeout: the answer gets 5 minutes.
CREATE_TIMEOUT = httpx.Timeout(60, read=300)


def api_key() -> str:
    """The key from the environment, else agent-env's secret store under the same name."""
    key = os.environ.get(KEY_NAME, "")
    if not key:
        try:
            from agent_env.config import get_config

            key = get_config().get_secret_store().get(KEY_NAME) or ""
        except Exception:  # noqa: BLE001 - no framework or no secret: the error below says what to set
            key = ""
    if not key:
        raise RuntimeError(f"{KEY_NAME} is not set (environment or agent-env secret store)")
    return key


def secrets_of(*urls: str) -> list[str]:
    """What must not be kept from these CDP addresses: each URL, its host, and its first label (the browser id)."""
    found: list[str] = []
    for url in urls:
        host = urlparse(url).hostname or ""
        found += [url, host, host.split(".")[0] if host.count(".") >= 2 else ""]
    return sorted({s for s in found if len(s) >= 8}, key=len, reverse=True)


def scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "<cdp>")
    return text


def scrub_json(value: Any, secrets: list[str]) -> Any:
    """``value`` with every secret replaced in any string inside it (JSON round trip, so non-JSON values stringify)."""
    if not secrets:
        return value
    return json.loads(scrub(json.dumps(value, default=str), secrets))


async def request(method: str, path: str, body: dict[str, Any] | None = None,
                  timeout: float | httpx.Timeout = 60) -> dict[str, Any]:
    segment = path.strip('/').split('/')[0]
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, f"{API}{path}", json=body,
                                            headers={"X-Browser-Use-API-Key": api_key().strip()})
    except httpx.HTTPError as error:
        # Its text can name the URL (the full browser id) or a header value it refused: the key itself.
        raise RuntimeError(f"Browser Use API {method} /{segment} failed ({type(error).__name__})") from None
    if response.is_error:
        # httpx's own error names the URL, and /browsers/{id} carries the full browser id.
        raise RuntimeError(f"Browser Use API {method} /{segment} returned {response.status_code}")
    return response.json() if response.content else {}


@dataclass
class CloudBrowser:
    id: str = field(repr=False)  # with the host it names, enough to reach the browser
    ws_url: str = field(repr=False)  # a credential
    cdp_url: str = field(default="", repr=False)  # also a credential
    stopped: bool = False

    @property
    def label(self) -> str:
        """The id as far as it is safe to log: not enough to reach the browser."""
        return self.id[:8]

    @property
    def secrets(self) -> list[str]:
        return sorted({*secrets_of(self.ws_url, self.cdp_url), self.id} - {""}, key=len, reverse=True)

    @classmethod
    async def start(cls, *, proxy_country_code: str | None = "us", timeout_minutes: int = 70,
                    enable_recording: bool = True) -> CloudBrowser:
        creating = asyncio.ensure_future(request("POST", "/browsers", {"proxyCountryCode": proxy_country_code,
                                                                       "timeout": timeout_minutes,
                                                                       "enableRecording": enable_recording},
                                                 timeout=CREATE_TIMEOUT))
        try:
            created = await asyncio.shield(creating)
        except asyncio.CancelledError:
            # Browser Use may already have made it: the answer is still read, and that browser stopped.
            abandoned = asyncio.ensure_future(_stop_created(creating))
            _ABANDONED.add(abandoned)
            abandoned.add_done_callback(_ABANDONED.discard)
            raise
        cdp_url = str(created.get("cdpUrl") or "").rstrip("/")
        browser = cls(id=str(created["id"]), ws_url="", cdp_url=cdp_url)
        try:
            if not cdp_url:
                raise RuntimeError("Browser Use did not return cdpUrl")
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    version = (await client.get(f"{cdp_url}/json/version")).json()
            except Exception as error:  # noqa: BLE001 - its text can name the address
                raise RuntimeError(f"cloud browser {browser.label}: CDP discovery failed "
                                   f"({type(error).__name__})") from None
            browser.ws_url = str(version.get("webSocketDebuggerUrl") or "")
            if not browser.ws_url:
                raise RuntimeError(f"cloud browser {browser.label} did not expose webSocketDebuggerUrl")
        except BaseException:
            await _stop_listed(browser)
            raise
        return browser

    async def stop(self) -> None:
        """PATCH {action: stop}, retried; ``stopped`` says whether it went through. Never raises: the browser's own
        timeout ends one this could not stop."""
        for attempt in range(1, STOP_ATTEMPTS + 1):
            if self.stopped:
                return
            try:
                # httpx's timeout bounds each read, not the request: an answer that trickles in would outlast it.
                async with asyncio.timeout(STOP_REQUEST_S):
                    await request("PATCH", f"/browsers/{self.id}", {"action": "stop"}, timeout=STOP_REQUEST_S)
                self.stopped = True
            except Exception as error:  # noqa: BLE001
                logger.warning("stopping cloud browser %s failed (%s), attempt %d of %d%s", self.label,
                               scrub(str(error) or type(error).__name__, self.secrets), attempt, STOP_ATTEMPTS,
                               "" if attempt < STOP_ATTEMPTS else "; its timeout will end it")
                if attempt < STOP_ATTEMPTS:
                    await asyncio.sleep(STOP_RETRY_S * attempt)

    async def session(self) -> dict[str, Any]:
        """The session as Browser Use reports it: costs once stopped, recording when enabled."""
        return await request("GET", f"/browsers/{self.id}")


#: Creates whose caller was cancelled before Browser Use answered, each waiting to stop the browser it made.
_ABANDONED: set[asyncio.Task] = set()


async def _stop_created(creating: asyncio.Future) -> None:
    try:
        browser = CloudBrowser(id=str((await creating)["id"]), ws_url="")
    except Exception:  # noqa: BLE001 - no answer, no browser to stop
        return
    await _stop_listed(browser)


async def _stop_listed(browser: CloudBrowser) -> None:
    UNSTOPPED.append(browser)  # listed first: a stop cancelled midway leaves it to the exit hook
    await browser.stop()
    if browser.stopped and browser in UNSTOPPED:  # else whoever stopped it meanwhile dropped it
        UNSTOPPED.remove(browser)


def costs(session: dict[str, Any]) -> dict[str, float]:
    return {"browser_usd": float(session.get("browserCost") or 0), "proxy_usd": float(session.get("proxyCost") or 0),
            "proxy_mb": float(session.get("proxyUsedMb") or 0)}


def recording_urls(session: dict[str, Any]) -> list[str]:
    """Whatever recording links the session carries; the field name is not part of a documented contract."""
    urls = []
    for key, value in session.items():
        if "record" in key.lower():
            values = value if isinstance(value, list) else [value]
            urls += [v for v in values if isinstance(v, str) and v.startswith("http")]
    RECORDING_HOSTS.update(urlparse(url).hostname or "" for url in urls)
    return urls


class Screenshots:
    """After-action evidence for the judge, taken over the env's own CDP connection, never the agent's.

    The agent drives the browser directly, so the env cannot see its actions; it captures the active page every
    ``interval`` seconds, for at most ``seconds`` (the browser's own time limit), and keeps a frame only when it
    differs from the last one, up to ``limit`` frames, the last of which is kept for the final one. Fail-open: errors
    are counted, never raised into the task.
    """

    def __init__(self, ws_url: str, directory: Path, interval: float = 2.0, limit: int = 400,
                 seconds: float = 70 * 60) -> None:
        self.ws_url, self.directory, self.interval, self.limit, self.seconds = ws_url, directory, interval, limit, seconds
        self._secrets = secrets_of(ws_url)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.saved: list[str] = []
        self.status: dict[str, Any] = {"captures": 0, "saved": 0, "duplicates": 0, "errors": []}
        self._ids = itertools.count(1)
        self._ws: Any = None
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._loop: asyncio.Task | None = None
        self._sessions: dict[str, str] = {}
        self._urls: dict[str, str] = {}
        self._active = ""
        self._last = ""

    def start(self) -> None:
        self._loop = asyncio.create_task(self._run())

    async def stop(self) -> dict[str, Any]:
        """One last capture, then close. The final frame is what the judge sees last."""
        if self._loop is not None:
            self._loop.cancel()
            await asyncio.gather(self._loop, return_exceptions=True)
        await self.capture("final")
        await self._close()
        return self.status

    async def _run(self) -> None:
        deadline = time.monotonic() + self.seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(self.interval)
            await self.capture(f"t{int(time.time())}")
        await self._close()

    async def capture(self, label: str) -> None:
        # The last of ``limit`` frames is the final one's, so the judge always sees the page the agent left.
        if len(self.saved) >= self.limit - (label != "final"):
            return
        self.status["captures"] += 1
        try:
            await asyncio.wait_for(self._capture(label), timeout=15)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            errors = self.status["errors"]
            errors.append(f"{label}:{type(error).__name__}:{scrub(str(error), self._secrets)[:120]}")
            del errors[:-20]
            self._sessions.pop(self._active, None)
            await self._close()

    async def _capture(self, label: str) -> None:
        if self._ws is None:
            import websockets

            self._ws = await websockets.connect(self.ws_url, max_size=64 * 1024 * 1024, open_timeout=15)
            self._reader = asyncio.create_task(self._read())
        target = await self._page()
        if not target:
            return
        if target not in self._sessions:
            attached = await self._send("Target.attachToTarget", {"targetId": target, "flatten": True})
            self._sessions[target] = attached["sessionId"]
        shot = await self._send("Page.captureScreenshot", {"format": "jpeg", "quality": 80}, self._sessions[target])
        data = base64.b64decode(shot["data"])
        digest = hashlib.sha256(data).hexdigest()
        if digest == self._last:
            self.status["duplicates"] += 1
            return
        self._last = digest
        name = f"{len(self.saved) + 1:04d}-{label}.jpg"
        (self.directory / name).write_bytes(data)
        self.saved.append(name)
        self.status["saved"] = len(self.saved)

    async def _page(self) -> str:
        """The page the agent last navigated or opened, as the bake-off harness picked it."""
        targets = (await self._send("Target.getTargets", {})).get("targetInfos", [])
        pages = [t for t in targets if t.get("type") == "page"
                 and not str(t.get("url", "")).startswith(("chrome://", "devtools://", "chrome-extension://"))]
        changed = [str(t["targetId"]) for t in pages if self._urls.get(str(t["targetId"])) != t.get("url")]
        self._urls = {str(t["targetId"]): str(t.get("url", "")) for t in pages}
        if changed:
            self._active = changed[-1]
        elif self._active not in self._urls:
            real = [tid for tid, url in self._urls.items() if url != "about:blank"]
            self._active = (real or list(self._urls) or [""])[-1]
        return self._active

    async def _send(self, method: str, params: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        message_id = next(self._ids)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        payload = {"id": message_id, "method": method, "params": params}
        if session_id:
            payload["sessionId"] = session_id
        await self._ws.send(json.dumps(payload))
        try:
            reply = await asyncio.wait_for(future, timeout=10)
        finally:
            self._pending.pop(message_id, None)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error'].get('message')}")
        return reply.get("result") or {}

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                future = self._pending.get(message.get("id"))
                if future is not None and not future.done():
                    future.set_result(message)
        except Exception:  # noqa: BLE001, S110 - a dropped socket fails the pending sends, and the next capture reconnects
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("CDP socket closed"))

    async def _close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001, S110 - closing a dead socket
                pass
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        self._ws, self._reader, self._sessions = None, None, {}
