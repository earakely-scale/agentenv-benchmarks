"""BU Bench V2.1 tasks, read from the public benchmark's encrypted file at run time.

github.com/browser-use/benchmark ships the 200 tasks and rubrics as BU_Bench_V2.enc, keyed by sha256("BU_Bench_V2")
to keep the text away from crawlers, and asks that decrypted tasks never be published. So the bundle's task files
name only a task id: the setup step decrypts the instruction in memory and hands it to the prompt as the seed value
``bu_task``. Rubrics are never sent to the agent or put in task files; the judge writes them, with the canary, into
the run's evidence folder, which stays private.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from pathlib import Path

from cryptography.fernet import Fernet

JUDGE_DIR = Path(__file__).resolve().parent / "judge" / "findings-v21"
BENCHMARK = "BU_Bench_V2"


@functools.cache
def _benchmark() -> dict:
    path = JUDGE_DIR / "upstream" / f"{BENCHMARK}.enc"
    source = json.loads((JUDGE_DIR / "SOURCE.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["files"][f"upstream/{BENCHMARK}.enc"]:
        raise ValueError("BU_Bench_V2.enc differs from the frozen public benchmark")
    key = base64.urlsafe_b64encode(hashlib.sha256(BENCHMARK.encode()).digest())
    data = json.loads(Fernet(key).decrypt(base64.b64decode(path.read_text())))
    return data if isinstance(data, dict) else {"tasks": data}


@functools.cache
def _instructions() -> dict[str, str]:
    tasks = _benchmark()["tasks"]
    return {str(t.get("task_id", t.get("id"))): str(t.get("confirmed_task", t.get("task"))) for t in tasks}


def instruction(task_id: str) -> str:
    """The task's instruction, exactly as the judge's frozen copy has it."""
    try:
        return _instructions()[task_id]
    except KeyError:
        raise KeyError(f"{task_id} is not a BU Bench V2.1 task") from None


def task_ids() -> list[str]:
    return sorted(_instructions())
