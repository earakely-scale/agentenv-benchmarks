"""Online-Mind2Web's v2 result format: a folder per attempt at a task, ``result.json`` and a ``trajectory/`` of
screenshots, checked as the benchmark's consumers check it, against its schema (``schema_v2.json``, copied from the
benchmark's repository at ``SCHEMA_COMMIT``) and the rules its README adds."""
from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema

SCHEMA_VERSION = "online-mind2web-v2"
#: The benchmark's commit the schema is copied from.
SCHEMA_COMMIT = "1a8db93168d5d7870402b7f453160e1c557153c8"
SCHEMA = json.loads((Path(__file__).with_name("schema_v2.json")).read_text(encoding="utf-8"))
RESULT = "result.json"
TRAJECTORY = "trajectory"
#: The terminal action, and how it names the agent's answer.
COMPLETE, ANSWER = "TASK_COMPLETE", "TASK_COMPLETE -> ANSWER:"
_STATUS = re.compile(r"\| (SUCCESS|FAILED)\s*$")


def problems(folder: Path) -> list[str]:
    """Why the attempt in ``folder`` isn't a valid v2 result, as the schema and its README's rules have it; none for
    one that is."""
    try:
        result = json.loads((folder / RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{RESULT}: {e}"]
    found = [f"{RESULT}: {'/'.join(map(str, error.absolute_path)) or '(top)'}: {error.message}"
             for error in jsonschema.Draft202012Validator(SCHEMA).iter_errors(result)]
    if found:
        return found
    steps = result["action_history"]
    names = [step["screenshot"] for step in steps]
    for index, step in enumerate(steps):
        if step["step"] != index:
            found.append(f"step {index} is numbered {step['step']}")
        if not (folder / TRAJECTORY / step["screenshot"]).is_file():
            found.append(f"step {index}'s screenshot {step['screenshot']} isn't in {TRAJECTORY}/")
        suffix = _STATUS.search(step["action"])
        if step.get("action_status") is not None and (suffix is None or suffix[1] != step["action_status"]):
            found.append(f"step {index}'s action_status {step['action_status']} isn't its action's suffix")
    if names != sorted(names) or len(set(names)) != len(names):
        found.append("the screenshots' names don't sort into step order, one each")
    last = steps[-1]["action"]
    if not last.startswith(COMPLETE):
        found.append(f"the last step's action doesn't start with {COMPLETE}")
    elif (answer := result.get("agent_final_answer")) is not None and (
            not last.startswith(ANSWER) or " ".join(last[len(ANSWER):].split()) != " ".join(answer.split())):
        found.append("agent_final_answer isn't the last step's answer")
    return found
