"""The WebArena-Verified task template: each task a multi env of the shopping admin site, wrapped as a website env, and
agent-env's chromium, the browser signed in to the site server-side before the agent gets its address, its start tabs
the task's start URLs as the site serves them, the agent Browser Use over CDP, given WebArena-Verified's own prompt, and
the multi env snapshotted after its turn, which carries the browser's ``network.har`` out for grading."""
from __future__ import annotations

import json
from pathlib import Path

from . import dataset, prompt
from .dataset import Task

#: The multi env every task deploys, and the site and browser it holds, by their names in the bundle.
OFFICE, SITE, BROWSER = "office", "shopping-admin", "chromium"
#: The site's address, which the dataset's placeholder stands in for, and the admin the browser is signed in as.
ADMIN_URL, USER = "http://shopping-admin.localhost/admin", "admin"
#: The browser, set through its env_params rather than by any agent, each setting explicit so the run's deploy record
#: shows them all; the start URLs are each task's.
SETTINGS = {
    "window": {"width": 1280, "height": 800},
    "dialogs": "accept",
    "geolocation": "deny",
    "notifications": "deny",
    "locale": "en-US",
    "time_zone": "UTC",
}
#: The agent, by the alias agent-env's agents bundle gives it; ``agent-env run --agent`` runs another on the template.
AGENT = "browser-use"
#: How long the agent has for one task.
PROMPT_TIMEOUT_S = 1800
#: The prompt's own id, which the answer is found under in a run's record.
PROMPT_ID = "answer"
#: The snapshot step, whose environment universe holds the browser's data/get, ``network.har`` among it.
SNAPSHOT_ID = "snapshot"
#: A task's name: the site's, then its dataset task id.
PREFIX = "shopping-admin-"


def name(task: Task) -> str:
    return f"{PREFIX}{task.task_id}"


def task_id(name: str) -> int:
    """The dataset task id of the task named ``name``, bundle-qualified or not."""
    return int(name.rsplit("/", 1)[-1].removeprefix(PREFIX).split("@", 1)[0])


def start_urls(task: Task) -> list[str]:
    """The task's start URLs, its placeholder made the site's address."""
    return [url.replace(dataset.PLACEHOLDER, ADMIN_URL) for url in task.start_urls]


def steps(task: Task) -> list[dict]:
    """The task's steps: the multi env deployed, the browser signed in, which loads its start tabs again signed in,
    Browser Use deployed and handed the browser's CDP address, prompted, and the multi env snapshotted."""
    chromium = {"env_params": {"app_settings": {BROWSER: {**SETTINGS, "start_urls": start_urls(task)}}}}
    return [
        {"id": OFFICE, "type": "deploy_env", "env_id": OFFICE, "child_envs": {BROWSER: chromium}},
        {"id": "sign_in", "type": "browser_sign_in", "env_id": OFFICE, "site": SITE, "user": USER,
         "depends_on": [OFFICE]},
        {"id": "agent", "type": "deploy_agent", "a2a_agent_id": AGENT, "env_ids": []},
        {"id": "connect", "type": "browser_connect_cdp", "env_id": OFFICE, "depends_on": ["sign_in", "agent"]},
        {"id": "task", "type": "prompt_agent", "prompt_id": PROMPT_ID, "prompt": prompt.render(task.intent, start_urls(task)),
         "timeout_seconds": PROMPT_TIMEOUT_S, "depends_on": ["agent", "connect"]},
        {"id": SNAPSHOT_ID, "type": "snapshot_env", "env_id": OFFICE, "depends_on": ["task"]},
    ]


def sample(tasks: list[Task]) -> list[Task]:
    """One task per intent template, the first the dataset lists of each, in the dataset's order."""
    seen: set[int] = set()
    return [task for task in tasks if not (task.intent_template_id in seen or seen.add(task.intent_template_id))]


def render(steps: list[dict]) -> str:
    """A task file as agent-env's shipped tasks are written: a step a line."""
    return "[\n" + ",\n".join(f"  {json.dumps(step, ensure_ascii=False)}" for step in steps) + "\n]\n"


def write(tasks: list[Task], bundle: Path) -> list[str]:
    """Every task written into ``bundle``'s ``tasks/``, with the ``all`` eval and the ``sample`` eval, one task per
    intent template, and any task file this template no longer makes removed. The names written."""
    folder = bundle / "tasks"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for task in tasks:
        (folder / f"{name(task)}.json").write_text(render(steps(task)), encoding="utf-8")
        written.append(name(task))
    for stale in folder.glob(f"{PREFIX}*.json"):
        if stale.stem not in written:
            stale.unlink()
    evals = {"all": written, "sample": [name(task) for task in sample(tasks)]}
    (bundle / "evals").mkdir(exist_ok=True)
    for eval_name, names in evals.items():
        (bundle / "evals" / f"{eval_name}.toml").write_text(
            "tasks = [\n" + "".join(f'    "{n}",\n' for n in names) + "]\n", encoding="utf-8")
    return written
