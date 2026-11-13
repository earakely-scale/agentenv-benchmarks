"""The Online-Mind2Web exporter: a run's frames and action log as the benchmark's v2 result, a step per burst of the
agent's actions in grammar A, a browser_* call over several bursts one step with its last frame, the start frame left
out and the last the TASK_COMPLETE step; what it refuses; and the checks a v2 result must pass."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentenv_benchmarks.online_mind2web import export, v2

RECORD = {"dataset": "osunlp/Online-Mind2Web", "revision": "eacad896", "sha256": "7ded", "task_id": "t1",
          "website": "https://shop.test/", "task": "Find a standing desk.", "level": "easy", "reference_length": 4,
          "arm": "tools", "browser": {"chromium": "156.0.8078.4"}}


def _frame(index: int, kind: str, after: list[int] = (), url: str = "https://shop.test/") -> dict:
    return {"index": index, "kind": kind, "at": "t", "after": list(after), "tab": "T1", "url": url,
            "path": f"frames/{index:04d}.jpg", "content_type": "image/jpeg", "uploaded": True}


def _line(seq: int, **action) -> dict:
    return {"seq": seq, "at": "t", "action": {"acts": True, "tab": "T1", "ok": True, **action}}


def _run(frames: list[dict], log: list[dict], answer: str | None = "A desk at $199.") -> export.Run:
    return export.Run(instance_id="run-1", record=RECORD, answer=answer,
                      manifest={"recording_id": "rec-1", "frames_on": True, "frames": frames}, log=log,
                      frame=lambda path: f"jpeg {path}".encode(), model="claude")


def _tools_run(**kwargs) -> export.Run:
    """Claude Code with Playwright MCP: a navigate whose commands span two bursts, a snapshot, a click that fails,
    typing that submits."""
    log = [
        _line(3, via="mcp", who="default", tool="browser_navigate", args={"url": "https://shop.test/"}),
        _line(4, via="cdp", who="playwright-mcp", method="Page.navigate", **{"for": 3}),
        _line(5, via="cdp", who="playwright-mcp", method="Runtime.evaluate", **{"for": 3}),
        _line(6, via="mcp", who="default", tool="browser_snapshot", args=None),
        _line(7, via="cdp", who="playwright-mcp", method="Runtime.callFunctionOn", **{"for": 6}),
        _line(8, via="mcp", who="default", tool="browser_click", args={"target": "e12", "element": "Desks link"},
              ok=False),
        _line(9, via="cdp", who="playwright-mcp", method="Input.dispatchMouseEvent", **{"for": 8}),
        _line(10, via="mcp", who="default", tool="browser_type",
              args={"target": "e3", "element": "Search", "text": "standing desk", "submit": True}),
        _line(11, via="cdp", who="playwright-mcp", method="Input.insertText", **{"for": 10}),
    ]
    frames = [_frame(0, "first", url="about:blank"), _frame(1, "action", [4]), _frame(2, "action", [5, 3]),
              _frame(3, "action", [7, 6]), _frame(4, "action", [9, 8]), _frame(5, "action", [11, 10]),
              _frame(6, "last")]
    return _run(frames, log, **kwargs)


def test_a_tools_run_is_a_step_per_browser_call_with_the_frame_its_call_ended_on(tmp_path):
    folder = export.write(_tools_run(), tmp_path)
    result = json.loads((folder / v2.RESULT).read_text())
    steps = result["action_history"]

    assert [(s["screenshot"], s["action"], s["action_status"]) for s in steps] == [
        ("0000.jpg", "page -> NAVIGATE -> go to https://shop.test/ | SUCCESS", "SUCCESS"),
        ("0001.jpg", "WAIT page -> read the page's accessibility snapshot", None),
        ("0002.jpg", "CLICK [aria-ref='e12'] -> click Desks link | FAILED", "FAILED"),
        ("0003.jpg", "TYPE [aria-ref='e3'] -> type 'standing desk' into Search, then press Enter | SUCCESS", "SUCCESS"),
        ("0004.jpg", "TASK_COMPLETE -> ANSWER: A desk at $199.", None)]
    assert (folder / v2.TRAJECTORY / "0000.jpg").read_bytes() == b"jpeg frames/0002.jpg"
    assert (folder / v2.TRAJECTORY / "0004.jpg").read_bytes() == b"jpeg frames/0006.jpg"
    assert (result["task_id"], result["agent_final_answer"], result["reference_length"]) == ("t1", "A desk at $199.", 4)
    assert all(s["thought"] is None for s in steps) and not v2.problems(folder)
    assert json.loads((folder / "agentenv.json").read_text())["revision"] == "eacad896"


def test_a_cdp_run_reads_its_input_events_as_the_actions_they_make():
    log = [
        _line(4, via="cdp", who="default-agent", method="Target.createTarget", params={"url": "about:blank"}),
        _line(5, via="cdp", who="default-agent", method="Page.navigate", params={"url": "https://shop.test/"}),
        _line(6, via="cdp", who="default-agent", method="Runtime.evaluate", params={"expression": "1"}),
        _line(7, via="cdp", who="default-agent", method="Runtime.evaluate", params={"expression": "2"}),
        _line(8, via="cdp", who="default-agent", method="Input.dispatchMouseEvent",
              params={"type": "mouseMoved", "x": 640.4, "y": 300}),
        _line(9, via="cdp", who="default-agent", method="Input.dispatchMouseEvent",
              params={"type": "mousePressed", "x": 640.4, "y": 300, "button": "left", "clickCount": 1}),
        _line(10, via="cdp", who="default-agent", method="Input.dispatchMouseEvent",
              params={"type": "mouseReleased", "x": 640.4, "y": 300}),
        _line(11, via="cdp", who="default-agent", method="Input.insertText", params={"text": "desk"}),
        _line(12, via="cdp", who="default-agent", method="Input.dispatchKeyEvent",
              params={"type": "keyDown", "key": "s", "text": "s"}),
        _line(13, via="cdp", who="default-agent", method="Input.dispatchKeyEvent", params={"type": "keyUp", "key": "s"}),
        _line(14, via="cdp", who="default-agent", method="Input.dispatchKeyEvent",
              params={"type": "rawKeyDown", "key": "Enter"}),
        _line(15, via="cdp", who="default-agent", method="Input.dispatchMouseEvent",
              params={"type": "mouseWheel", "x": 1, "y": 1, "deltaY": -400}),
    ]
    frames = [_frame(0, "first"), _frame(1, "action", [4, 5, 6, 7]), _frame(2, "action", [8, 9, 10]),
              _frame(3, "action", [11, 12, 13, 14]), _frame(4, "action", [15]), _frame(5, "last")]

    assert [action for _, action, _ in export.steps(_run(frames, log).manifest, log)] == [
        "page -> NAVIGATE -> open a new tab; then go to https://shop.test/; then run a script on the page | SUCCESS",
        "CLICK coords(640, 300) -> click | SUCCESS",
        "TYPE page -> type 'desks'; then press Enter | SUCCESS",
        "SCROLL page -> scroll up by 400px | SUCCESS"]


def test_an_action_still_going_when_the_run_ended_is_said_to_be_and_counts_toward_no_status():
    log = [
        _line(3, via="mcp", who="default", tool="browser_navigate", args={"url": "https://shop.test/"}),
        _line(4, via="mcp", who="default", tool="browser_click", args={"target": "e5", "element": "Desks"},
              unfinished=True),
        _line(5, via="mcp", who="default", tool="browser_type", args={"target": "e3", "element": "Search", "text": "x"},
              ok=None, unfinished=True),
    ]
    frames = [_frame(0, "first"), _frame(1, "action", [3, 4]), _frame(2, "action", [5]), _frame(3, "last")]

    assert [(action, status) for _, action, status in export.steps(_run(frames, log).manifest, log)] == [
        ("page -> NAVIGATE -> go to https://shop.test/; then click Desks (unfinished at the end) | SUCCESS", "SUCCESS"),
        ("TYPE [aria-ref='e3'] -> type 'x' into Search (unfinished at the end)", None)]


def test_a_run_without_an_answer_completes_saying_so(tmp_path):
    folder = export.write(_tools_run(answer=None), tmp_path)
    result = json.loads((folder / v2.RESULT).read_text())

    assert result["agent_final_answer"] is None and not v2.problems(folder)
    assert result["action_history"][-1]["action"] == "TASK_COMPLETE -> the agent gave no answer"


def test_a_run_missing_a_frame_or_exported_twice_into_one_folder_is_refused(tmp_path):
    missing = _tools_run()
    missing.manifest["frames"][3]["uploaded"] = False
    with pytest.raises(export.NotExportable, match=r"missing frames \[3\]"):
        export.write(missing, tmp_path)

    export.write(_tools_run(), tmp_path)
    with pytest.raises(export.NotExportable, match="there already"):
        export.write(_tools_run(), tmp_path)


def test_a_result_is_checked_as_the_benchmarks_consumers_check_it(tmp_path):
    folder = export.write(_tools_run(), tmp_path)
    path = folder / v2.RESULT
    result = json.loads(path.read_text())

    def broken(change) -> list[str]:
        document = json.loads(json.dumps(result))
        change(document)
        path.write_text(json.dumps(document))
        return v2.problems(folder)

    assert any("numbered 7" in p for p in broken(lambda d: d["action_history"][1].update(step=7)))
    assert any("isn't in trajectory/" in p for p in broken(lambda d: d["action_history"][1].update(
        screenshot="0009.jpg")))
    assert any("action_status" in p for p in broken(lambda d: d["action_history"][2].update(action_status="SUCCESS")))
    assert any("last step's answer" in p for p in broken(lambda d: d.update(agent_final_answer="Another desk.")))
    assert any("thought" in p for p in broken(lambda d: d["action_history"][0].pop("thought")))
    assert any("TASK_COMPLETE" in p for p in broken(lambda d: d["action_history"].pop()))


def test_a_run_is_loaded_from_its_record_its_frames_recording_and_its_answer(monkeypatch):
    objects = {"s3://b/manifest.json": json.dumps({"recording_id": "rec-1", "frames_on": True, "frames": []}).encode(),
               "s3://b/log.jsonl": b'{"seq": 1, "at": "t", "action": {"via": "cdp"}}\n',
               "s3://b/frames/0001.jpg": b"jpeg"}
    files = {"manifest.json": "m", "log.jsonl": "l", "frames-0001.jpg": "f"}
    urls = {"m": "s3://b/manifest.json", "l": "s3://b/log.jsonl", "f": "s3://b/frames/0001.jpg"}
    context = {"metadata": {
        "browser_recordings": {"frames": {"state": "ended", "artifact": {"id": "u", "version": 1}, "missing_frames": 0,
                                          "log_uploaded": True, "log_missing": 0}},
        "script_results": {"setup": RECORD}},
        "prompt_responses": [{"prompt_id": "answer", "response": " A desk. ", "model": "claude"}]}

    def get(artifact_id, version=None):
        return SimpleNamespace(file_artifact_ids=files) if artifact_id == "u" else SimpleNamespace(
            object_url=urls[artifact_id])

    monkeypatch.setattr(export, "find_task_instance", lambda instance_id: SimpleNamespace(context=context))
    monkeypatch.setattr(export.Artifact, "get", get)
    monkeypatch.setattr(export, "get_config", lambda: SimpleNamespace(
        get_object_store=lambda: SimpleNamespace(get=objects.__getitem__)))
    run = export.load("run-1")

    assert (run.record, run.answer, run.model, run.log) == (RECORD, "A desk.", "claude",
                                                           [{"seq": 1, "at": "t", "action": {"via": "cdp"}}])
    assert run.frame("frames/0001.jpg") == b"jpeg"
    context["metadata"]["browser_recordings"]["frames"]["missing_frames"] = 2
    with pytest.raises(export.NotExportable, match="incomplete"):
        export.load("run-1")


def test_a_run_whose_agent_failed_is_loaded_from_the_recording_its_teardown_ended(monkeypatch):
    base = "file:///store/recordings/ns/rec-1"
    manifest = {"recording_id": "rec-1", "ended_by": "teardown", "error": None, "log": "log.jsonl",
                "log_uploaded": True, "log_missing": [], "frames": [{"path": "frames/0000.jpg", "uploaded": True}]}
    objects = {f"{base}/manifest.json": json.dumps(manifest).encode(),
               f"{base}/log.jsonl": b'{"seq": 1, "at": "t", "action": {"via": "cdp"}}\n',
               f"{base}/frames/0000.jpg": b"jpeg"}
    entry = {"state": "open", "namespace": "file:///store/recordings/ns", "recording_id": "rec-1"}
    context = {"metadata": {"browser_recordings": {"frames": entry}, "script_results": {"setup": RECORD}},
               "prompt_responses": [{"prompt_id": "answer", "response": "Out of steps.", "model": "claude",
                                     "error_code": "browser_use.max_steps"}]}
    monkeypatch.setattr(export, "find_task_instance", lambda instance_id: SimpleNamespace(context=context))
    monkeypatch.setattr(export, "get_config", lambda: SimpleNamespace(
        get_object_store=lambda: SimpleNamespace(get=objects.__getitem__)))
    run = export.load("run-1")

    assert (run.answer, run.agent_error, run.manifest["ended_by"]) == ("Out of steps.", "browser_use.max_steps",
                                                                       "teardown")
    assert run.frame("frames/0000.jpg") == b"jpeg"
    manifest["log_missing"] = [3]
    objects[f"{base}/manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(export.NotExportable, match="incomplete"):
        export.load("run-1")
    del objects[f"{base}/manifest.json"]
    with pytest.raises(export.NotExportable, match="didn't end"):
        export.load("run-1")
