#!/usr/bin/env python3
"""Frozen public V2.1 findings judge on staged Browser Code evidence.

Upstream task validation, prompt, schemas, image selection, SDK, and scoring are
pinned in SOURCE.json. Browser Code screenshots are after-action, explicitly
labeled. SDK request/response objects are saved; these are not HTTP wire bytes.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "upstream"))

from evaluation import create_judge, judge_config, judge_trace, load_tasks  # noqa: E402


def verify_source():
    source = json.loads((ROOT / "SOURCE.json").read_text())
    for name, expected in source["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen judge source drift: {name}")
    return source


def output_text(files):
    if files is None:
        return None
    if not files:
        return "The agent's outputs/ deliverables directory is empty: the agent saved no files."
    parts = []
    for file in files:
        body = file.get("text")
        content = body if body is not None else "(binary or oversized: content not shown)"
        clipped = ' clipped="true"' if file.get("clipped") else ""
        parts.append(f'<file name="{file["name"]}" bytes="{file["size"]}"{clipped}>\n{content}\n</file>')
    return "\n".join(parts)


def evidence(workspace, result):
    metadata = result.get("metadata") or {}
    shots = [base64.b64encode((workspace / name).read_bytes()).decode()
             for name in metadata.get("judge_screenshots", [])]
    mapping = metadata.get("judge_screenshot_steps")
    return {
        "final_result": result.get("final_output") or "",
        "agent_steps": metadata.get("steps") or [],
        "screenshots_b64": shots,
        "screenshot_steps": mapping or None,
        "screenshot_timing": "after",
        "output_files_text": output_text(metadata.get("output_files")),
    }


async def main():
    workspace = Path(os.environ["EVAL_WORKSPACE"])
    output = Path(os.environ["EVAL_JUDGMENT_PATH"])
    started = time.monotonic()
    model = os.environ.get("EVAL_JUDGE_MODEL", "gpt-5.6-luna")
    try:
        source = verify_source()
        options = json.loads(os.environ.get("EVAL_JUDGE_OPTIONS_JSON") or "{}")
        if set(options) - {"reasoning_effort"}:
            raise ValueError("unknown V2.1 judge option")
        supplied = json.loads(Path(os.environ["EVAL_TASK_PATH"]).read_text())
        task = next(t for t in load_tasks() if t["task_id"] == supplied["task_id"])
        if supplied["confirmed_task"] != task["confirmed_task"]:
            raise ValueError("instruction differs from frozen public task")
        result = json.loads(Path(os.environ["EVAL_RESULT_PATH"]).read_text())
        trace = evidence(workspace, result)
        os.environ["OPENAI_API_KEY"] = os.environ["EVAL_JUDGE_API_KEY"]
        llm = create_judge(model=model, reasoning_effort=options.get("reasoning_effort", "xhigh"))
        config = judge_config("BU_Bench_V2", llm)
        config["screenshot_timing"] = "after_action"
        (workspace / "judge-config.json").write_text(json.dumps(config, indent=2))
        details = await judge_trace(task, trace, llm, artifact_dir=workspace / "judge-recording")
        judgment = details["judgement"]
        payload = {
            "status": "scored", "score": details["score"],
            "reason": f'{details["earned_weight"]}/100 rubric weight met'
                      + (", zeroed for reward hacking or canary leak" if details["rh_zeroed"] else ""),
            "failure_class": "reward-hacking" if details["rh_zeroed"] else
                             "infra-error" if judgment["infra_error"] else "",
            "evidence_checked": ["result.json", "judge-recording/judge_input.json",
                                 "judge-recording/judge_response.json"],
            "rubric": {f["item"]: f["status"] == "met" for f in judgment["findings"]},
            "additional_scores": {"raw_rubric_score": details["raw_score"]},
            "model": model, "duration_seconds": time.monotonic() - started,
        }
        (workspace / "judge_trace.json").write_text(json.dumps({
            "task_id": task["task_id"], "weights": task["weights"],
            "task_sha": task["task_sha"], "rubric_sha": task["rubric_sha"],
            "weights_sha": task["weights_sha"], "source": source,
            "score_details": details, "judgement": judgment,
            "prompt": {"agent_reasoning_included": False, "config": config},
        }, indent=2) + "\n")
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return 0
    except Exception as error:
        output.write_text(json.dumps({
            "status": "error", "score": 0, "reason": f"{type(error).__name__}: {error}",
            "failure_class": "runtime-failure", "evidence_checked": [],
            "rubric": {}, "model": model, "duration_seconds": time.monotonic() - started,
        }, indent=2) + "\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
