"""WebArena-Verified's shopping admin tasks at the one release the bundle is generated from: the dataset that release
ships, checked against its pinned sha256, so one release's results are never mixed with another's. The package pins
its own dependencies exactly (pydantic among them), so it is read through ``uvx`` in an environment of its own."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

VERSION = "1.2.3"
PACKAGE = f"webarena-verified=={VERSION}"
#: The dataset and the hard subset, in the package's ``assets/dataset``, and their sha256 at VERSION.
FILE, SHA256 = "webarena-verified.json", "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30"
HARD_FILE = "subsets/webarena-verified-hard.json"
HARD_SHA256 = "3b0a4df231bb5a0c642215e521c3fa97701a384f52a734dc2db8f617ad0591a7"
#: The site the bundle covers, as the dataset names it, and the placeholder its URLs stand in for it by.
SITE, PLACEHOLDER = "shopping_admin", "__SHOPPING_ADMIN__"
#: The evaluator that grades a task's final answer, which every task has; any other grades its network log.
ANSWER_EVALUATOR = "AgentResponseEvaluator"
LOCATE = "import pathlib, webarena_verified; print(pathlib.Path(webarena_verified.__file__).parent / 'assets' / 'dataset')"


@dataclass(frozen=True)
class Task:
    """One of the site's single-site tasks, as the dataset gives it, and whether it's in the hard subset and graded on
    its network log too."""

    task_id: int
    intent: str
    intent_template_id: int
    start_urls: tuple[str, ...]
    revision: int
    hard: bool
    network_graded: bool

    @classmethod
    def from_row(cls, row: dict, hard: set[int]) -> Task:
        return cls(task_id=row["task_id"], intent=row["intent"], intent_template_id=row["intent_template_id"],
                   start_urls=tuple(row["start_urls"]), revision=row["revision"], hard=row["task_id"] in hard,
                   network_graded=any(check["evaluator"] != ANSWER_EVALUATOR for check in row["eval"]))


def locate() -> Path:
    """The dataset folder of the package at VERSION, which ``uvx`` installs in an environment of its own."""
    found = subprocess.run(["uvx", "--from", PACKAGE, "python", "-I", "-c", LOCATE], capture_output=True, text=True,
                           check=True)
    return Path(found.stdout.strip().splitlines()[-1])


def checked(path: Path, sha256: str) -> bytes:
    """``path``'s bytes; ValueError when they aren't the ones pinned."""
    data = path.read_bytes()
    if (found := hashlib.sha256(data).hexdigest()) != sha256:
        raise ValueError(f"{path} has sha256 {found}, not {PACKAGE}'s {sha256}")
    return data


def tasks(folder: Path) -> list[Task]:
    """The site's single-site tasks, from the dataset folder ``folder``, in the dataset's order."""
    hard = set(json.loads(checked(folder / HARD_FILE, HARD_SHA256))["task_ids"])
    return [Task.from_row(row, hard) for row in json.loads(checked(folder / FILE, SHA256)) if row["sites"] == [SITE]]
