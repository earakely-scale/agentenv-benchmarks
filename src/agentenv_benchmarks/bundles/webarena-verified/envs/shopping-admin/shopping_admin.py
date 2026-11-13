"""WebArena-Verified's shopping admin as a website env: a ForwardingSite in front of the image's Magento on port 80,
signed in by the image's admin auto-login header, which only this site's own request sends."""
from __future__ import annotations

import time
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from pathlib import Path

from agentenv_protocol.website import ForwardingSite, ready_check, session_state, website

NAME = "shopping-admin.localhost"
AUTO_LOGIN = "X-M2-Admin-Auto-Login"
CREDENTIALS = "admin:admin1234"
SESSION_DEFAULT_S = 24 * 3600
#: Where the image's boot init (env-ctrl init: base URL, password policy, cache flush) says how it went: [OK] once done.
INIT_LOG = Path("/tmp/env-ctrl-init.log")


@website(domains=[NAME], builtin_routes=False, clock="real", ready_timeout_s=900)
class ShoppingAdmin(ForwardingSite):
    upstream = "http://127.0.0.1:80"
    ready_path = "/admin"
    sign_in_headers = (AUTO_LOGIN,)

    @ready_check
    def boot_init_done(self) -> bool:
        """The image's boot init finished, so Magento has its base URL and a warm cache."""
        said = INIT_LOG.read_text(errors="replace").strip() if INIT_LOG.exists() else ""
        if said and "[OK]" not in said:
            raise RuntimeError(f"the boot init says {said[-300:]!r}")
        return "[OK]" in said

    @ready_check
    async def base_url_set(self) -> bool:
        """Magento's sign-in page names this site, so the image's boot init has set the base URL."""
        answer = await self.ask_app("/admin", timeout=8)
        if answer.status_code == 200:
            return f"http://{NAME}/" in answer.text
        return answer.headers.get("location", "").startswith(f"http://{NAME}/")

    @session_state
    async def admin(self, user):
        answer = await self.ask_app("/admin", headers={AUTO_LOGIN: CREDENTIALS})
        cookies = []
        for header in answer.headers.get_list("set-cookie"):
            for morsel in SimpleCookie(header).values():
                if _max_age(morsel) <= 0:
                    continue
                cookies.append({"name": morsel.key, "value": morsel.value, "path": morsel["path"] or "/",
                                "max_age": _max_age(morsel), "httpOnly": bool(morsel["httponly"]),
                                "secure": bool(morsel["secure"]), **({"sameSite": morsel["samesite"].capitalize()}
                                                                    if morsel["samesite"] else {})})
        if not any(cookie["name"] == "admin" for cookie in cookies):
            raise RuntimeError(f"the auto-login request answered {answer.status_code} without an admin session cookie")
        return {"cookies": cookies, "local_storage": []}


def _max_age(morsel) -> float:
    if morsel["max-age"]:
        return float(morsel["max-age"])
    if morsel["expires"]:
        try:
            return max(0.0, parsedate_to_datetime(morsel["expires"].replace("-", " ")).timestamp() - time.time())
        except (TypeError, ValueError):
            pass
    return float(SESSION_DEFAULT_S)


if __name__ == "__main__":
    ShoppingAdmin().serve()
