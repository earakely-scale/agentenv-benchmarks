"""WebArena-Verified's grader: every run exported into OUT graded by WebArena-Verified's own evaluator, at the bundle's
release, in an environment of its own (its exact pins can't share ours), from each run's ``network.har`` and
``agent_response.json``:

    python -m agentenv_benchmarks.webarena_verified.grade OUT

It writes ``OUT/grades.jsonl``, a line a run: the evaluator's result (its task, score, status, each evaluator's result,
its error, and the release's version and checksums) and the run as its ``agentenv.json`` names it. A run the evaluator
errors on is listed apart, unscored, not counted as a wrong answer. It needs ``uv``, which runs the release by
``uvx``."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import dataset, export, template

GRADES = "grades.jsonl"
#: What a result's status is when the evaluator couldn't grade the run, which leaves it unscored.
ERROR = "error"
#: The script the release runs: each run folder of OUT graded, its result a JSON line.
SCRIPT = """
import json, pathlib, sys
from webarena_verified import WebArenaVerified
from webarena_verified.types.config import WebArenaVerifiedConfig
out, site, url = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
wav = WebArenaVerified(config=WebArenaVerifiedConfig(environments={site: {"urls": [url]}}))
for folder in sorted((p for p in out.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
    response = json.loads((folder / "agent_response.json").read_text(encoding="utf-8"))
    result = wav.evaluate_task(task_id=int(folder.name), agent_response=response,
                               network_trace=folder / "network.har")
    print(json.dumps(result.model_dump(mode="json")), flush=True)
"""


def results(out: Path) -> list[dict]:
    """The evaluator's result for each run folder of ``out``, in task order."""
    ran = subprocess.run(["uvx", "--from", dataset.PACKAGE, "python", "-I", "-c", SCRIPT, str(out), dataset.SITE,
                          template.ADMIN_URL], capture_output=True, text=True, check=False)
    if ran.returncode != 0:
        raise SystemExit(f"{dataset.PACKAGE} couldn't grade {out}: {ran.stderr[-2000:]}")
    return [json.loads(line) for line in ran.stdout.splitlines() if line.startswith("{")]


def grade(out: Path) -> dict:
    """Every run in ``out`` graded into ``out/grades.jsonl``; how many were scored, their mean score, and the task ids
    of those left unscored."""
    lines = []
    for result in results(out):
        record = out / str(result["task_id"]) / export.RECORD
        run = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
        lines.append({**result, "run": run})
    (out / GRADES).write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    scored = [line for line in lines if line["status"] != ERROR]
    return {"scored": len(scored), "mean_score": sum(line["score"] for line in scored) / len(scored) if scored else None,
            "unscored": [line["task_id"] for line in lines if line["status"] == ERROR]}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder the runs were exported into")
    args = parser.parse_args(argv)
    summary = grade(args.out)
    mean = "none" if summary["mean_score"] is None else f"{summary['mean_score']:.3f}"
    print(f"{summary['scored']} runs scored, mean {mean}; {len(summary['unscored'])} unscored "
          f"(evaluator errors): {summary['unscored']}; {args.out / GRADES}")


if __name__ == "__main__":
    main(sys.argv[1:])
