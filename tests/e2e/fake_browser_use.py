"""A fake Browser Use API for tests/test_bu_browser_e2e.py: every browser it creates is the one local Chromium at
``cdp_url``; each create and stop is appended to ``log`` as a JSON line.

    python fake_browser_use.py <log> <port> <cdp_url>
"""
import json
import sys

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

LOG, PORT, CDP_URL = sys.argv[1], int(sys.argv[2]), sys.argv[3]
created = 0


def log(entry: dict) -> None:
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


async def create(request):
    global created
    created += 1
    browser = f"e2e{created}-0123456789abcdef"
    log({"call": "create", "id": browser, "key": request.headers.get("x-browser-use-api-key"),
         "body": await request.json()})
    return JSONResponse({"id": browser, "cdpUrl": CDP_URL})


async def stop(request):
    log({"call": "stop", "id": request.path_params["id"], "body": await request.json()})
    return JSONResponse({})


uvicorn.run(Starlette(routes=[Route("/browsers", create, methods=["POST"]),
                              Route("/browsers/{id}", stop, methods=["PATCH"])]),
            host="0.0.0.0", port=PORT, log_level="warning")
