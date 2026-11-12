"""The Online-Mind2Web task template: every arm on the same browser, chromium alone and headful, set once here, its
agent given the website in the prompt from a blank tab, its run recorded as frames after each burst of the agent's
actions, which grading reads, and as a video of the screen, and its run's record naming the dataset row and revision it
came from and the browser as deployed."""
from __future__ import annotations

import json
from pathlib import Path

from . import dataset
from .dataset import Task

#: The website browser every arm drives: agent-env's chromium, deployed alone.
ENV = "chromium"
#: The browser for every arm, set through its env_params rather than by any agent: the design's defaults, with dialogs
#: accepted, each set explicitly so the run's deploy record shows them all.
SETTINGS = {
    "start_urls": ["about:blank"],
    "window": {"width": 1280, "height": 800},
    "dialogs": "accept",
    "geolocation": "deny",
    "notifications": "deny",
    "locale": "en-US",
    "time_zone": "UTC",
}
#: How long an agent has for one task.
PROMPT_TIMEOUT_S = 1800
#: Each arm's agent, by the alias agent-env's agents bundle gives it; ``agent-env run --agent`` runs another on an
#: arm's template, Codex on ``tools`` or a Stagehand agent on ``cdp``.
ARMS = {"cdp": "browser-use", "tools": "claude-code"}
#: The tools the ``tools`` arm sees: Playwright MCP 0.0.84's, as the browser computer serves them shown, so the arm
#: drives the browser through Playwright MCP alone and not its computer_* pixels or page tools.
BROWSER_TOOLS = (
    "browser_click", "browser_close", "browser_console_messages", "browser_drag", "browser_drop",
    "browser_emulate_media", "browser_file_upload", "browser_fill_form", "browser_find", "browser_handle_dialog",
    "browser_hover", "browser_mouse_click_xy", "browser_mouse_down", "browser_mouse_drag_xy", "browser_mouse_move_xy",
    "browser_mouse_up", "browser_mouse_wheel", "browser_navigate", "browser_navigate_back", "browser_network_request",
    "browser_network_requests", "browser_press_key", "browser_resize", "browser_select_option", "browser_snapshot",
    "browser_tabs", "browser_take_screenshot", "browser_type", "browser_wait_for",
)
#: The role an agent's MCP session has when its deploy names none, which tool access is set for.
ROLE = "default"
#: The tasks a sample run takes, one per level (Apple, easy; Trader Joe's, medium; Target, hard), on sites agent-env's
#: pre-flight sweep of the start sites saw load from both an AWS VM and Modal.
SAMPLE = ("d71be72aa25c3eab8eea47a0e60382e2", "b7258ee05d75e6c50673a59914db412e_110325",
          "6ebde509dca8f15c0fa1bd74f071e8d6")
#: The prompt's own id, which the answer is found under in a run's record.
PROMPT_ID = "answer"
#: The run_code result the run's own record is kept under.
SETUP_ID = "setup"
#: The browser recording a run's frames are under, in its record's ``browser_recordings``; a frame missing fails it.
FRAMES = "frames"


def name(task: Task, arm: str) -> str:
    return f"{arm}-{task.task_id}"


def prompt(task: Task) -> str:
    """``Go to {website} and {task}``, as Browser Use's own runner gives it, the task's first word lowercased unless
    it's an initialism or a name such as ``I``, ``IKEA`` or ``iPhone``."""
    first, _, rest = task.task.partition(" ")
    if len(first) > 1 and first[0].isupper() and first[1:].islower():
        first = first.lower()
    return f"Go to {task.website} and {first}{' ' + rest if rest else ''}"


def steps(task: Task, arm: str) -> list[dict]:
    """The task's steps on ``arm``."""
    record = {"dataset": dataset.REPO, "revision": dataset.REVISION, "sha256": dataset.SHA256, "task_id": task.task_id,
              "website": task.website, "task": task.task, "level": task.level,
              "reference_length": task.reference_length, "arm": arm}
    browser = {"id": "browser", "type": "deploy_env", "env_id": ENV, "env_params": {"app_settings": {ENV: SETTINGS}}}
    setup = {"id": SETUP_ID, "type": "run_code", "env_id": ENV, "script_artifact_id": "record_setup",
             "result_id": SETUP_ID, "args": record, "depends_on": ["browser"]}
    recording = {"id": "record", "type": "start_computer_recording", "env_id": ENV, "name": "run",
                 "depends_on": ["browser"]}
    frames = {"id": FRAMES, "type": "start_browser_recording", "env_id": ENV, "name": FRAMES, "depends_on": ["browser"]}
    if arm == "cdp":
        ways_in = [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": ARMS[arm], "env_ids": []},
                   {"id": "connect", "type": "browser_connect_cdp", "env_id": ENV, "depends_on": ["browser", "agent"]}]
    else:
        ways_in = [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": ARMS[arm], "env_ids": [ENV]},
                   {"id": "hide", "type": "modify_env_tool_access", "env_id": ENV, "action": "disable", "role": ROLE,
                    "tools": ["*"], "depends_on": ["browser"]},
                   {"id": "show", "type": "modify_env_tool_access", "env_id": ENV, "action": "enable", "role": ROLE,
                    "tools": list(BROWSER_TOOLS), "depends_on": ["hide"]}]
    ready = [step["id"] for step in ways_in if step["id"] != "hide"] + [SETUP_ID, "record", FRAMES]
    ask = {"id": "task", "type": "prompt_agent", "prompt_id": PROMPT_ID, "prompt": prompt(task),
           "timeout_seconds": PROMPT_TIMEOUT_S, "depends_on": ready}
    end = {"id": "end", "type": "end_computer_recording", "name": "run", "depends_on": ["task"]}
    end_frames = {"id": f"end_{FRAMES}", "type": "end_browser_recording", "name": FRAMES, "depends_on": ["task"]}
    return [browser, setup, recording, frames, *ways_in, ask, end, end_frames]


def render(steps: list[dict]) -> str:
    """A task file as agent-env's shipped tasks are written: a step a line."""
    return "[\n" + ",\n".join(f"  {json.dumps(step, ensure_ascii=False)}" for step in steps) + "\n]\n"


def write(tasks: list[Task], bundle: Path) -> list[str]:
    """Every task on every arm written into ``bundle``'s ``tasks/``, with an eval per arm and a sample eval, and any
    task file this template no longer makes removed. The names written."""
    folder = bundle / "tasks"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for arm in ARMS:
        for task in tasks:
            (folder / f"{name(task, arm)}.json").write_text(render(steps(task, arm)), encoding="utf-8")
            written.append(name(task, arm))
    for stale in folder.glob("*.json"):
        if stale.stem not in written and stale.stem.split("-", 1)[0] in ARMS:
            stale.unlink()
    by_id = {task.task_id: task for task in tasks}
    evals = {arm: [name(task, arm) for task in tasks] for arm in ARMS}
    evals["sample"] = [name(by_id[task_id], arm) for arm in ARMS for task_id in SAMPLE]
    (bundle / "evals").mkdir(exist_ok=True)
    for eval_name, names in evals.items():
        (bundle / "evals" / f"{eval_name}.toml").write_text(
            "tasks = [\n" + "".join(f'    "{n}",\n' for n in names) + "]\n", encoding="utf-8")
    return written
