"""The Hub release of graded runs: a row per run, the Space's pass file with each step on its sheet of frames and its
WebJudge score, the card's results, and a refusal to build anything holding what looks like a credential."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from agentenv_benchmarks.online_mind2web import judge, v2

TOOL = Path(__file__).resolve().parent.parent / "tools" / "online_mind2web_release.py"
#: A task the bundle has, on the tools arm.
TASK = "005be9dd91c95669d6ddde9ae667125c"


def _release():
    spec = importlib.util.spec_from_file_location("online_mind2web_release", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exported(root: Path, answer: str = "20 kg.") -> Path:
    folder = root / "tools" / TASK
    (folder / v2.TRAJECTORY).mkdir(parents=True)
    steps = []
    for index, height in enumerate((50, 60, 40)):
        Image.new("RGB", (100, height), "white").save(folder / v2.TRAJECTORY / f"{index:04d}.jpg")
        steps.append({"step": index, "screenshot": f"{index:04d}.jpg", "url": "https://www.qatarairways.com/",
                      "action": f"action {index}", "action_status": "SUCCESS" if index < 2 else None, "thought": None})
    (folder / v2.RESULT).write_text(json.dumps({
        "schema_version": v2.SCHEMA_VERSION, "task": "Find the baggage allowance.", "task_id": TASK,
        "agent_final_answer": answer, "reference_length": 4, "action_history": steps}))
    (folder / "agentenv.json").write_text(json.dumps({"instance_id": "run-1", "arm": "tools", "model": "haiku",
                                                      "revision": "eacad896", "recording_id": "rec-1"}))
    (folder.parent / judge.GRADES).write_text(json.dumps({
        "task_id": TASK, "grade": "success", "response": 'Status: "success"', "key_points": "1. Economy.",
        "scores": [1, 4, 5], "judge": {"snapshot": judge.SNAPSHOT},
        "reported": [{"model": judge.SNAPSHOT, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}]}) + "\n")
    return root


def test_a_pass_is_built_as_rows_the_spaces_pass_file_and_the_cards_results(tmp_path, monkeypatch):
    release = _release()
    monkeypatch.setattr(release, "PER_SHEET", 2)
    out = tmp_path / "out"
    release.build(out, {"p1": _exported(tmp_path / "exports")}, {"p1": "The first pass."})

    (row,) = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert row["grade"] == "success" and row["level"] == "easy" and row["actions"] == 2
    assert (row["judge_prompt_tokens"], row["judge_completion_tokens"]) == (10, 2)
    assert "steps" not in row and row["export"] == "exports/p1-tools.tar.gz"
    (run,) = json.loads((out / "viewer" / "p1.json").read_text())["runs"]
    assert [(s["sheet"], s["y"], s["h"], s["score"]) for s in run["steps"]] == [
        (0, 0, 400, 1), (0, 400, 480, 4), (1, 0, 320, 5)]
    for sheet in run["sheets"]:
        with Image.open(out / sheet["path"]) as image:
            assert image.size == (release.WIDTH, sheet["height"])
    assert (out / "exports" / "p1-tools.tar.gz").stat().st_size
    index = json.loads((out / "viewer" / "index.json").read_text())
    assert index["passes"] == [{"name": "p1", "note": "The first pass.", "runs": 1,
                                "graded": {"tools": {"success": 1}, "tools/easy": {"success": 1}}}]
    assert "| p1 | Claude Code with Playwright MCP's browser_* tools | 1 | 1 | 100% | 0 |" in (out / "README.md").read_text()


def test_a_run_holding_what_looks_like_a_credential_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="credential"):
        _release().build(tmp_path / "out", {"p1": _exported(tmp_path / "exports", answer="key sk-" + "a" * 24)}, {})
