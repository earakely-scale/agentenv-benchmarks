"""The shopping admin site class, against a stand-in for the image's Magento on loopback: its session signs an admin in
by the auto-login header and gives the browser Magento's own admin cookie, at its path and for its lifetime, never a
cookie Magento deleted; its card names the header as its sign-in header and declares a real clock and no built-in
routes; and it is ready only once the image's boot init says [OK] and Magento's sign-in page names the site."""
from __future__ import annotations

import importlib.util
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agentenv_protocol.website import WEBSITE_EXTENSION_URI
from starlette.testclient import TestClient

from agentenv_benchmarks.webarena_verified import BUNDLE

SITE_FILE = BUNDLE / "envs" / "shopping-admin" / "shopping_admin.py"
NAME = "shopping-admin.localhost"


def _site_module():
    spec = importlib.util.spec_from_file_location("shopping_admin", SITE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Magento:
    """What the site asks of the image's Magento: /admin signs in a request carrying the auto-login header, setting the
    admin cookie and deleting the form key, and otherwise is the sign-in page, naming the base URL it was set to."""

    def __init__(self) -> None:
        self.base_url = f"http://{NAME}/"
        self.seen: list[dict] = []
        magento = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                magento.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                if self.headers.get("X-M2-Admin-Auto-Login") == "admin:admin1234":
                    self._send(302, b"", [
                        ("Location", f"http://{NAME}/admin/admin/dashboard/"),
                        ("Set-Cookie", ("admin=s3ss10n; expires=Tue, 17-Nov-2026 03:06:45 GMT; Max-Age=360000; "
                                        f"path=/admin; domain={NAME}; HttpOnly; SameSite=Lax")),
                        ("Set-Cookie", ("form_key=deleted; expires=Thu, 01-Jan-1970 00:00:01 GMT; Max-Age=0; "
                                        f"path=/admin; domain={NAME}; SameSite=Lax"))])
                else:
                    page = f'<form action="{magento.base_url}admin/" id="login-form"></form>'.encode()
                    self._send(200, page, [("Content-Type", "text/html; charset=UTF-8")])

            def _send(self, status: int, body: bytes, headers: list[tuple[str, str]]) -> None:
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def magento() -> Iterator[Magento]:
    served = Magento()
    yield served
    served.server.shutdown()
    served.server.server_close()


@pytest.fixture
def site(magento, tmp_path, monkeypatch):
    module = _site_module()
    monkeypatch.setattr(module.ShoppingAdmin, "upstream", magento.url)
    monkeypatch.setattr(module, "INIT_LOG", tmp_path / "env-ctrl-init.log")
    with TestClient(module.ShoppingAdmin().create_app(), headers={"X-Forwarded-Host": NAME}) as client:
        yield client, module


def _website_params(client: TestClient) -> dict:
    card = client.get("/.well-known/agent-env.json").json()
    return next(ext for ext in card["capabilities"]["extensions"] if ext["uri"] == WEBSITE_EXTENSION_URI)["params"]


def test_its_session_is_magentos_admin_cookie_at_its_path_for_its_lifetime_and_never_a_deleted_one(site, magento):
    client, _ = site

    session = client.post("/agentenv/ext/website/session", json={"user": "admin"}).json()

    assert session == {"cookies": [{"name": "admin", "value": "s3ss10n", "path": "/admin", "max_age": 360000.0,
                                    "httpOnly": True, "secure": False, "sameSite": "Lax"}], "local_storage": []}
    asked = magento.seen[-1]
    assert (asked["path"], asked["headers"]["host"], asked["headers"]["x-m2-admin-auto-login"]) == (
        "/admin", NAME, "admin:admin1234")


def test_its_card_names_the_auto_login_header_as_its_sign_in_header_with_a_real_clock_and_no_built_in_routes(site):
    client, _ = site

    params = _website_params(client)

    assert (params["domains"], params["sign_in_headers"], params["clock"]) == (
        [NAME], ["X-M2-Admin-Auto-Login"], "real")
    assert "session_path" not in params and "session_cookie" not in params


def test_a_browser_request_carrying_the_auto_login_header_reaches_magento_without_it(site, magento):
    client, _ = site

    client.get("/admin", headers={"X-M2-Admin-Auto-Login": "admin:admin1234"})

    assert "x-m2-admin-auto-login" not in magento.seen[-1]["headers"] and magento.seen[-1]["path"] == "/admin"


@pytest.mark.parametrize(("init_log", "base_url", "ready", "problem"), [
    (None, f"http://{NAME}/", False, "boot_init_done returned False"),
    ("[OK]", "http://localhost:7780/", False, "base_url_set returned False"),
    ("Failed: setup:store-config:set exited 1", f"http://{NAME}/", False, "the boot init says"),
    ("[OK]", f"http://{NAME}/", True, None),
])
def test_it_is_ready_once_the_boot_init_says_ok_and_magentos_sign_in_page_names_the_site(
        site, magento, init_log, base_url, ready, problem):
    client, module = site
    if init_log is not None:
        module.INIT_LOG.write_text(init_log)
    magento.base_url = base_url

    answer = client.get("/agentenv/ext/website/ready").json()

    assert answer["ready"] is ready, answer
    assert problem is None or any(problem in found for found in answer["problems"]), answer
