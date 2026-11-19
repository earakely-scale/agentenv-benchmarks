"""OpenEnv episodes on a copy of the bu-browser bundle, this script playing the model, for
tests/test_bu_browser_e2e.py: the site's real image on agent-env's local Docker provider, a fake Browser Use API
(fake_browser_use.py, which logs to ``calls``) and a local Chromium. Prints one JSON line per checkpoint: what was
created and stopped so far.

    python bu_browser_episode.py <bundle> <calls>
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from agent_env.openenv import build_app
from fastapi.testclient import TestClient

HASH = "3721b75f6c501523694ef58cd7a3911c032cdabddbb38bca02709a249465c68f"
#: A page whose text is the hash, which its URL doesn't hold: its script writes it reversed back, so only the page's
#: own content in the result can show it.
PAGE = "data:text/html," + quote(f"<title>pypi</title><p id=h></p><script>document.getElementById('h').textContent="
                                 f"'{HASH[::-1]}'.split('').reverse().join('')</script>")
assert HASH not in PAGE
root, calls = Path(sys.argv[1]), Path(sys.argv[2])


def browsers() -> dict:
    entries = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return {"created": [e["id"] for e in entries if e["call"] == "create"],
            "stopped": [e["id"] for e in entries if e["call"] == "stop"]}


def checkpoint(label: str, **seen) -> None:
    print(json.dumps({"at": label, **browsers(), **seen}), flush=True)


def settled(label: str, **seen) -> None:
    """checkpoint once every browser created so far is stopped, or after 60 s: OpenEnv tears an episode down in the
    background."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        seen_so_far = browsers()
        if sorted(seen_so_far["stopped"]) == sorted(seen_so_far["created"]):
            break
        time.sleep(0.5)
    checkpoint(label, **seen)


def playwright_mcp_users() -> dict:
    """The uid Playwright MCP runs as, that of PID 1 of its container (the site's), and whether its command line names
    the browser's CDP address, as the host's ps sees them."""
    def docker(*args: str) -> str:
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout

    for container in docker("ps", "-q").split():
        rows = [row.split(None, 2) for row in docker("top", container, "-o", "pid,uid,args").splitlines()[1:]]
        mcp = [uid for _, uid, args in rows if "@playwright/mcp" in args]
        if mcp:
            init = docker("inspect", "-f", "{{.State.Pid}}", container).strip()
            return {"mcp_uids": mcp, "pid1_uid": next((uid for pid, uid, _ in rows if pid == init), None),
                    "mcp_args_name_cdp": any("/devtools/" in args for _, _, args in rows if "@playwright/mcp" in args)}
    return {"mcp_uids": [], "pid1_uid": None, "mcp_args_name_cdp": None}


def step(ws, data: dict) -> dict:
    ws.send_json({"type": "step", "data": data})
    return ws.receive_json()


def reset(ws) -> None:
    ws.send_json({"type": "reset", "data": {"split": "all", "index": 0}})
    ws.receive_json()


with TestClient(build_app(root)) as client, client.websocket_connect("/ws") as ws:
    reset(ws)
    checkpoint("episode 1 reset")
    tools = sorted(t["name"] for t in step(ws, {"type": "list_tools"})["data"]["observation"]["tools"])
    navigated = step(ws, {"type": "call_tool", "tool_name": "browser_navigate",
                          "arguments": {"url": PAGE}})
    navigated_text = json.dumps(navigated["data"]["observation"].get("result"))
    ended = step(ws, {"type": "call_tool", "tool_name": "end_turn", "arguments": {"response": HASH}})
    settled("episode 1 ended", tools=tools, saw_page=HASH in navigated_text, linked="[Snapshot](" in navigated_text,
            done=ended["data"].get("done"), reward=ended["data"].get("reward"))
    reset(ws)
    step(ws, {"type": "call_tool", "tool_name": "browser_navigate", "arguments": {"url": "data:text/html,x"}})
    checkpoint("episode 2 running", **playwright_mcp_users())
    reset(ws)  # abandons episode 2 mid-turn; episode 3 calls no tool
    settled("episode 2 abandoned")
settled("session closed")
