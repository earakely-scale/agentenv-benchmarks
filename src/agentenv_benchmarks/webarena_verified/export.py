"""WebArena-Verified's exporter: a run of one of the bundle's tasks as the files WebArena-Verified's evaluator grades
offline, from the run's own record:

    python -m agentenv_benchmarks.webarena_verified.export OUT INSTANCE_ID...

Each run is written to ``OUT/<task_id>/``:

- ``network.har``, the browser's network log, as the run's snapshot of its multi env after the agent's turn saved it;
- ``agent_response.json``, the JSON object WebArena-Verified's own extractor finds in the agent's final message
  (``upstream/extract_agent_response.py``, copied unchanged from the release and run as its harness runs it), or that
  message as text when it finds none, which the evaluator then scores as a wrong answer;
- ``agentenv.json``, which names the run, its model, the task, the release and the dataset it came from, and whether
  the extractor found an answer.

A run with no snapshot, or no ``network.har`` in it, isn't exported, nor one whose task isn't the bundle's."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_env.artifact import Artifact
from agent_env.store.routing import namespace_routing
from agent_env.task.store import find_task_instance

from . import dataset, template

#: WebArena-Verified's own extractor, as ``examples/evaluation/extract_agent_response.py`` holds it at the release.
EXTRACTOR = Path(__file__).resolve().parent / "upstream" / "extract_agent_response.py"
EXTRACTOR_SHA256 = "29c37c590440e97f7a53eba00db5dd3bd7011a62870b737f32a7ce3d96fbc32b"
#: The file in the browser's data/get the evaluator reads as the run's network trace.
HAR_FILE = "network.har"
AGENT_RESPONSE, RECORD = "agent_response.json", "agentenv.json"


class NotExportable(Exception):
    """A run that can't be exported, saying why."""


@dataclass
class Run:
    """A run of one of the bundle's tasks: the task it ran, the agent's final message, its model and the error it
    ended with, and the browser's network log as the run's snapshot saved it."""

    instance_id: str
    task_name: str
    answer: str | None
    model: str | None
    agent_error: str | None
    har: bytes
    snapshot: dict


def extracted(answer: str | None) -> dict | None:
    """The JSON object WebArena-Verified's own extractor finds in ``answer``, run as a script as its harness runs it,
    with each field it expects set; None when it finds none."""
    if (found := hashlib.sha256(EXTRACTOR.read_bytes()).hexdigest()) != EXTRACTOR_SHA256:
        raise ValueError(f"{EXTRACTOR} has sha256 {found}, not the release's {EXTRACTOR_SHA256}")
    if not answer:
        return None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent_response.txt"
        path.write_text(answer, encoding="utf-8")
        ran = subprocess.run([sys.executable, "-I", str(EXTRACTOR), str(path)], capture_output=True, text=True,
                             timeout=60, check=False)
    return json.loads(ran.stdout) if ran.returncode == 0 else None


def write(run: Run, out: Path) -> Path:
    """The run written as ``out/<task_id>/``; refused when that folder is there already, as for a second run of one
    task."""
    task_id = template.task_id(run.task_name)
    folder = out / str(task_id)
    if folder.exists():
        raise NotExportable(f"{folder} is there already; export each run of a task into a folder of its own")
    folder.mkdir(parents=True)
    response = extracted(run.answer)
    (folder / HAR_FILE).write_bytes(run.har)
    (folder / AGENT_RESPONSE).write_text(
        json.dumps(response if response is not None else (run.answer or ""), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (folder / RECORD).write_text(json.dumps({
        "instance_id": run.instance_id, "task": run.task_name, "task_id": task_id, "model": run.model,
        "agent_error": run.agent_error, "answer_extracted": response is not None, "release": dataset.PACKAGE,
        "dataset_sha256": dataset.SHA256, "snapshot": run.snapshot}, indent=1) + "\n", encoding="utf-8")
    return folder


def load(instance_id: str) -> Run:
    """The run ``instance_id`` from the configured stores, or a bundle run's from the local ones."""
    with namespace_routing():
        instance = find_task_instance(instance_id)
        if instance is None or not instance.context:
            raise NotExportable(f"no run {instance_id} is recorded")
        name = str(instance.task_id)
        try:
            template.task_id(name)
        except ValueError:
            raise NotExportable(f"run {instance_id} ran {name}, not one of the bundle's tasks") from None
        context = instance.context
        snapshot = ((context.get("metadata") or {}).get("env_snapshotted_universes") or {}).get(template.SNAPSHOT_ID)
        if not snapshot:
            raise NotExportable(f"run {instance_id} has no {template.SNAPSHOT_ID} of its multi env")
        universe = Artifact.get(snapshot["id"], version=snapshot["version"])
        hars = [file for environment in universe.get_environment_artifacts()
                for path, file in environment.get_file_artifacts().items() if Path(path).name == HAR_FILE]
        if not hars:
            raise NotExportable(f"run {instance_id}'s snapshot holds no {HAR_FILE}")
        har = hars[0].load()
    answered = next((r for r in context.get("prompt_responses") or [] if r.get("prompt_id") == template.PROMPT_ID),
                    {})
    return Run(instance_id=instance_id, task_name=name, answer=(answered.get("response") or "").strip() or None,
               model=answered.get("model"), agent_error=answered.get("error_code"), har=har,
               snapshot={"id": snapshot["id"], "version": snapshot["version"]})


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder each run is written into, one folder per task")
    parser.add_argument("instance_ids", nargs="+", metavar="INSTANCE_ID", help="a run's instance id")
    args = parser.parse_args(argv)
    failed = 0
    for instance_id in args.instance_ids:
        try:
            print(f"{instance_id}: {write(load(instance_id), args.out)}")
        except NotExportable as e:
            failed += 1
            print(f"{instance_id}: not exported: {e}", file=sys.stderr)
    if failed:
        raise SystemExit(f"{failed} of {len(args.instance_ids)} runs weren't exported")


if __name__ == "__main__":
    main(sys.argv[1:])
