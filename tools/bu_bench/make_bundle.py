#!/usr/bin/env python3
"""Write the bu-bench bundle's task and eval files: all 200 BU Bench V2.1 tasks, plus the smoke-2 and mini-20 subsets.

Each task file names only its BU Bench id; the instruction is decrypted at run time (see tasks.py). Every task runs
the same five steps, so a harness is changed with `agent-env run ... --agent NAME` and nothing else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentenv_benchmarks.bu_bench import tasks

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "src" / "agentenv_benchmarks" / "bundles" / "bu-bench"
SMOKE = ["bu2-140", "bu2-149"]  # the bake-off's frozen smoke pair
DEFAULT_AGENT = "codex-bu-cli"


def task(task_id: str) -> list[dict]:
    return [
        {"id": "deploy", "type": "deploy_agent", "a2a_agent_id": DEFAULT_AGENT, "env_ids": []},
        {"id": "browser", "type": "bu_cloud_browser", "task_id": task_id, "depends_on": ["deploy"]},
        {"id": "solve", "type": "prompt_agent", "prompt_id": "solve", "prompt": "<bu_task>", "effort": "medium",
         "timeout_seconds": 3600, "fail_task_on_error": False, "depends_on": ["browser"]},
        {"id": "collect", "type": "collect_artifacts", "base_path": "/app/agent_outputs",
         "fail_task_on_error": False, "depends_on": ["solve"]},
        {"id": "grade", "type": "bu_findings_verifier", "task_id": task_id, "prompt_id": "solve",
         "judge_model": "gpt-5.6-luna", "reasoning_effort": "xhigh", "depends_on": ["solve", "collect"]},
    ]


#: 20 tasks spread over task length, none an infra error in either of two full reference runs on our platform.
MINI_20 = ["bu2-034", "bu2-046", "bu2-047", "bu2-051", "bu2-055", "bu2-058", "bu2-061", "bu2-095", "bu2-103", "bu2-104", "bu2-129", "bu2-140", "bu2-159", "bu2-161", "bu2-164", "bu2-165", "bu2-167", "bu2-177", "bu2-183", "bu2-193"]


def main() -> int:
    (BUNDLE / "tasks").mkdir(parents=True, exist_ok=True)
    (BUNDLE / "evals").mkdir(parents=True, exist_ok=True)
    for old in (BUNDLE / "tasks").glob("*.json"):
        old.unlink()
    every = tasks.task_ids()
    for task_id in every:
        steps = task(task_id)
        lines = ",\n".join("  " + json.dumps(step) for step in steps)
        (BUNDLE / "tasks" / f"{task_id}.json").write_text(f"[\n{lines}\n]\n")
    for name, ids, note in (("smoke-2", SMOKE, "The bake-off's two smoke tasks."),
                            ("mini-20", MINI_20, "20 BU Bench V2.1 tasks, spread over task length, picked from two full reference runs."),
                            ("bu-bench-v21", every, "All 200 BU Bench V2.1 tasks (the harness bake-off's set).")):
        listed = ", ".join(json.dumps(t) for t in ids)
        (BUNDLE / "evals" / f"{name}.toml").write_text(f"# {note}\ntasks = [{listed}]\n")
    print(f"wrote {len(every)} tasks, 3 evals", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
