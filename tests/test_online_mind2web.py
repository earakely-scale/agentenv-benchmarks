"""The Online-Mind2Web bundle: every committed task is what the template makes from the dataset row its run records,
each of the dataset's 300 tasks once per arm at the pinned revision, every arm on the same browser, the evals the
arms' tasks, and the run's record of its setup read from the browser's card."""
from __future__ import annotations

import importlib.util
import io
import json
import tomllib

import pytest

from agentenv_benchmarks.online_mind2web import BUNDLE, dataset, template
from agentenv_benchmarks.online_mind2web.dataset import Task

TASKS = sorted((BUNDLE / "tasks").glob("*.json"))
RECORD_SETUP = BUNDLE / "artifacts" / "record_setup" / "run.py"


def _committed() -> dict[str, list[dict]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in TASKS}


def _recorded(steps: list[dict]) -> dict:
    return next(step for step in steps if step["id"] == template.SETUP_ID)["args"]


def _task(record: dict) -> Task:
    return Task(task_id=record["task_id"], task=record["task"], website=record["website"], level=record["level"],
                reference_length=record["reference_length"])


def test_every_committed_task_is_what_the_template_makes_from_the_row_it_records():
    for path in TASKS:
        record = _recorded(json.loads(path.read_text(encoding="utf-8")))
        assert path.read_text(encoding="utf-8") == template.render(template.steps(_task(record), record["arm"])), path


def test_each_of_the_300_tasks_runs_once_per_arm_at_the_pinned_revision():
    records = [_recorded(steps) for steps in _committed().values()]
    by_arm = {arm: {r["task_id"] for r in records if r["arm"] == arm} for arm in template.ARMS}

    assert len(records) == 300 * len(template.ARMS) and len(by_arm["cdp"]) == 300 and by_arm["cdp"] == by_arm["tools"]
    assert {(r["dataset"], r["revision"], r["sha256"]) for r in records} == {
        (dataset.REPO, dataset.REVISION, dataset.SHA256)}
    assert {r["level"] for r in records} == set(dataset.LEVELS)
    assert sorted(path.stem for path in TASKS) == sorted(template.name(_task(r), r["arm"]) for r in records)


def test_every_arm_runs_on_one_browser_set_by_the_task_and_starts_from_a_blank_tab():
    browsers = {json.dumps(steps[0], sort_keys=True) for steps in _committed().values()}

    assert len(browsers) == 1
    browser = json.loads(browsers.pop())
    assert (browser["type"], browser["env_id"]) == ("deploy_env", "chromium")
    assert browser["env_params"]["app_settings"]["chromium"] == {
        "start_urls": ["about:blank"], "window": {"width": 1280, "height": 800}, "dialogs": "accept",
        "geolocation": "deny", "notifications": "deny", "locale": "en-US", "time_zone": "UTC"}


def test_each_arm_reaches_the_browser_its_own_way_and_is_asked_the_same_prompt():
    tasks = _committed()
    cdp, tools = tasks["cdp-d71be72aa25c3eab8eea47a0e60382e2"], tasks["tools-d71be72aa25c3eab8eea47a0e60382e2"]
    by_id = {name: {step["id"]: step for step in steps} for name, steps in (("cdp", cdp), ("tools", tools))}

    assert by_id["cdp"]["agent"]["a2a_agent_id"] == "browser-use" and by_id["cdp"]["agent"]["env_ids"] == []
    assert by_id["cdp"]["connect"]["type"] == "browser_connect_cdp"
    assert by_id["tools"]["agent"]["a2a_agent_id"] == "claude-code" and by_id["tools"]["agent"]["env_ids"] == ["chromium"]
    assert (by_id["tools"]["hide"]["tools"], by_id["tools"]["show"]["tools"]) == (["*"], list(template.BROWSER_TOOLS))
    assert not {"browser_evaluate", "browser_run_code_unsafe"} & set(template.BROWSER_TOOLS)
    assert by_id["cdp"]["task"]["prompt"] == by_id["tools"]["task"]["prompt"] == \
        "Go to https://www.apple.com/ and find technical specs for the latest Macbook Air on Apple."
    for steps in (by_id["cdp"], by_id["tools"]):
        assert {"setup", "record", "frames"} <= set(steps["task"]["depends_on"])
        assert steps["end"]["depends_on"] == steps["end_frames"]["depends_on"] == ["task"]


def test_each_run_records_frames_for_grading_and_a_video_of_the_screen():
    for steps in _committed().values():
        by_id = {step["id"]: step for step in steps}
        assert (by_id["frames"]["type"], by_id["frames"]["name"], by_id["end_frames"]["name"]) == (
            "start_browser_recording", "frames", "frames")
        assert "fail_task_on_error" not in by_id["frames"] and "fail_task_on_error" not in by_id["end_frames"]
        assert (by_id["record"]["type"], by_id["end"]["type"]) == ("start_computer_recording", "end_computer_recording")


@pytest.mark.parametrize("task, prompted", [
    ("Find the hours.", "find the hours."),
    ("IKEA: find a desk.", "IKEA: find a desk."),
    ("iPhone deals.", "iPhone deals."),
    ("I need a flight.", "I need a flight."),
    ("Browse", "browse"),
])
def test_the_prompt_lowercases_the_tasks_first_word_unless_its_an_initialism_or_a_name(task, prompted):
    row = Task(task_id="t", task=task, website="https://a.test/", level="easy", reference_length=1)
    assert template.prompt(row) == f"Go to https://a.test/ and {prompted}"


def test_an_eval_per_arm_runs_its_300_tasks_and_the_sample_one_per_level_on_each_arm():
    evals = {path.stem: tomllib.loads(path.read_text())["tasks"] for path in (BUNDLE / "evals").glob("*.toml")}
    levels = {_recorded(steps)["task_id"]: _recorded(steps)["level"] for steps in _committed().values()}

    assert set(evals) == {*template.ARMS, "sample", "agreement"}
    for arm in template.ARMS:
        assert sorted(evals[arm]) == sorted(path.stem for path in TASKS if path.stem.startswith(f"{arm}-"))
    assert evals["sample"] == [f"{arm}-{task_id}" for arm in template.ARMS for task_id in template.SAMPLE]
    assert sorted(levels[task_id] for task_id in template.SAMPLE) == sorted(dataset.LEVELS)


def test_the_agreement_sample_is_30_tasks_on_both_arms_by_level_with_none_on_a_blocked_site():
    evals = {path.stem: tomllib.loads(path.read_text())["tasks"] for path in (BUNDLE / "evals").glob("*.toml")}
    tasks = {path.stem: _task(_recorded(steps)) for path, steps in zip(TASKS, _committed().values())}
    sampled = [tasks[name] for name in evals["agreement"]]
    by_arm = {arm: [t.task_id for t, name in zip(sampled, evals["agreement"]) if name.startswith(f"{arm}-")]
              for arm in template.ARMS}
    picked = [tasks[f"cdp-{task_id}"] for task_id in by_arm["cdp"]]

    assert len(evals["agreement"]) == 60 and by_arm["cdp"] == by_arm["tools"] and len(set(by_arm["cdp"])) == 30
    assert [t.level for t in picked].count("easy") == 8 and [t.level for t in picked].count("medium") == 14
    assert not any(template.blocked(t) for t in picked)
    every = {tasks[name].task_id: tasks[name] for name in tasks if name.startswith("cdp-")}
    assert sum(template.blocked(t) for t in every.values()) == 21
    assert [t.task_id for t in template.agreement(list(every.values()))] == by_arm["cdp"]


def test_writing_from_the_dataset_writes_every_task_and_eval_and_drops_what_it_no_longer_makes(tmp_path):
    rows = [Task(task_id=task_id, task="Find it.", website="https://a.test/", level=level, reference_length=3)
            for task_id, level in zip(template.SAMPLE, dataset.LEVELS)]
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "cdp-gone.json").write_text("[]")
    (tmp_path / "tasks" / "kept.json").write_text("[]")

    written = template.write(rows, tmp_path)

    assert sorted(path.stem for path in (tmp_path / "tasks").glob("*.json")) == sorted([*written, "kept"])
    assert len(written) == 3 * len(template.ARMS)
    assert sorted(path.stem for path in (tmp_path / "evals").glob("*.toml")) == sorted([*template.ARMS, "sample"])


def _record_setup():
    spec = importlib.util.spec_from_file_location("record_setup", RECORD_SETUP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_run_records_its_dataset_row_and_the_browser_its_card_names(monkeypatch):
    module = _record_setup()
    card = {"name": "chromium", "children_environments": [
        {"name": "page"},
        {"name": "chromium", "capabilities": {"extensions": [{"uri": "urn:agentenv:browser/v1", "params": {
            "chromium": "156.0.8078.4", "clock": "real", "mode": "headful",
            "window": {"width": 1280, "height": 800}, "methods": {"log": {}}}}]}}]}
    asked = []

    def urlopen(request, timeout):
        asked.append((request.full_url, request.headers))
        return io.BytesIO(json.dumps(card).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    recorded = module.run({"args": {"task_id": "t1", "arm": "cdp"},
                           "env": {"url": "http://127.0.0.1:18765", "headers": {"Authorization": "Bearer x"}}})

    assert recorded == {"task_id": "t1", "arm": "cdp", "browser": {
        "chromium": "156.0.8078.4", "clock": "real", "mode": "headful", "window": {"width": 1280, "height": 800}}}
    assert asked == [("http://127.0.0.1:18765/.well-known/agent-env.json", {"Authorization": "Bearer x"})]
    with pytest.raises(ValueError, match="no website browser"):
        module.browser({"name": "desk"})
