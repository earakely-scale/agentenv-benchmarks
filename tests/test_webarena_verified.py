"""The WebArena-Verified bundle: its 182 tasks are what the template makes from the dataset its pinned release ships,
each the multi env signed in before Browser Use gets the browser, prompted with the release's own prompt and snapshotted
after its turn; the exporter writes what the release's evaluator grades, the answer found by the release's own
extractor; and grading lists apart the runs the evaluator couldn't grade."""
from __future__ import annotations

import json
import shutil
import tomllib
from collections import Counter
from types import SimpleNamespace

import pytest

from agentenv_benchmarks.webarena_verified import BUNDLE, dataset, export, grade, prompt, template
from agentenv_benchmarks.webarena_verified.dataset import Task

TASK = Task(task_id=7, intent="Show the orders of Grace Nguyen", intent_template_id=3,
            start_urls=("__SHOPPING_ADMIN__/sales/order/",), revision=2, hard=False, network_graded=True)
HAR = b'{"log": {"version": "1.2", "entries": []}}'


def _committed() -> dict[str, list[dict]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (BUNDLE / "tasks").glob("*.json")}


def _eval(name: str) -> list[str]:
    return tomllib.loads((BUNDLE / "evals" / f"{name}.toml").read_text(encoding="utf-8"))["tasks"]


@pytest.mark.skipif(shutil.which("uvx") is None, reason=f"uvx reads the dataset {dataset.PACKAGE} ships")
def test_every_committed_task_is_what_the_template_makes_from_the_pinned_release(tmp_path):
    tasks = dataset.tasks(dataset.locate())
    written = template.write(tasks, tmp_path)

    assert (len(tasks), len({task.intent_template_id for task in tasks})) == (182, 41)
    assert (sum(task.hard for task in tasks), sum(task.network_graded for task in tasks)) == (55, 94)
    assert {name: json.loads((tmp_path / "tasks" / f"{name}.json").read_text()) for name in written} == _committed()
    for name in ("all", "sample"):
        assert (tmp_path / "evals" / f"{name}.toml").read_text() == (BUNDLE / "evals" / f"{name}.toml").read_text()


def test_the_all_eval_runs_every_task_and_the_sample_one_task_per_intent_template():
    committed = _committed()

    assert sorted(_eval("all")) == sorted(committed) and len(committed) == 182
    assert len(_eval("sample")) == 41 and set(_eval("sample")) <= set(committed)
    assert template.sample([TASK, Task(**{**TASK.__dict__, "task_id": 8}), Task(**{**TASK.__dict__, "task_id": 9,
                                                                                 "intent_template_id": 4})]) == [
        TASK, Task(**{**TASK.__dict__, "task_id": 9, "intent_template_id": 4})]


def test_each_task_signs_the_browser_in_before_browser_use_gets_it_and_snapshots_the_multi_env_after_its_turn():
    for name, steps in _committed().items():
        by_id = {step["id"]: step for step in steps}
        chromium = by_id["office"]["child_envs"]["chromium"]["env_params"]["app_settings"]["chromium"]
        sign_in, connect, task, snapshot = by_id["sign_in"], by_id["connect"], by_id["task"], by_id["snapshot"]

        assert [step["type"] for step in steps] == ["deploy_env", "browser_sign_in", "deploy_agent",
                                                    "browser_connect_cdp", "prompt_agent", "snapshot_env"], name
        assert by_id["office"]["env_id"] == "office" and by_id["agent"]["a2a_agent_id"] == "browser-use"
        assert chromium["start_urls"] and all(url.startswith(template.ADMIN_URL) for url in chromium["start_urls"])
        assert (sign_in["site"], sign_in["user"], sign_in["depends_on"]) == ("shopping-admin", "admin", ["office"])
        assert connect["depends_on"] == ["sign_in", "agent"] and task["depends_on"] == ["agent", "connect"]
        assert snapshot["depends_on"] == ["task"] and template.task_id(name) >= 0
        assert all(url in task["prompt"] for url in chromium["start_urls"]) and "{{" not in task["prompt"], name


def test_a_tasks_start_tabs_are_its_start_urls_as_the_site_serves_them_and_its_prompt_the_releases():
    by_id = {step["id"]: step for step in template.steps(TASK)}
    rendered = by_id["task"]["prompt"]

    assert by_id["office"]["child_envs"]["chromium"]["env_params"]["app_settings"]["chromium"]["start_urls"] == [
        "http://shopping-admin.localhost/admin/sales/order/"]
    assert "- **Task Objective:** `Show the orders of Grace Nguyen`" in rendered
    assert "  - http://shopping-admin.localhost/admin/sales/order/\n" in rendered
    assert rendered.startswith("You are an autonomous web agent operating in Merchant Admin Portal.")
    assert template.task_id("webarena-verified/shopping-admin-7") == template.task_id("shopping-admin-7") == 7


def test_the_prompt_is_refused_when_its_bytes_arent_the_releases(tmp_path, monkeypatch):
    changed = tmp_path / "shopping_admin.md"
    changed.write_text(prompt.template() + "\nBe quick.\n", encoding="utf-8")
    monkeypatch.setattr(prompt, "FILE", changed)

    with pytest.raises(ValueError, match="not https://github.com/ServiceNow/webarena-verified"):
        prompt.render("anything", [template.ADMIN_URL])


def test_the_answer_is_the_first_json_object_the_releases_own_extractor_finds_or_none():
    two = 'Example: {"task_type": "NAVIGATE"}\nFinal: {"task_type": "RETRIEVE", "status": "SUCCESS"}'

    assert export.extracted(two) == {"task_type": "NAVIGATE", "status": "", "retrieved_data": "", "error_details": ""}
    assert export.extracted("```json\n{\"status\": \"NOT_FOUND_ERROR\"}\n```")["status"] == "NOT_FOUND_ERROR"
    assert export.extracted("I couldn't find it") is None and export.extracted(None) is None


def test_the_extractor_is_refused_when_its_bytes_arent_the_releases(tmp_path, monkeypatch):
    changed = tmp_path / "extract_agent_response.py"
    changed.write_bytes(export.EXTRACTOR.read_bytes() + b"\n")
    monkeypatch.setattr(export, "EXTRACTOR", changed)

    with pytest.raises(ValueError, match="not the release's"):
        export.extracted('{"status": "SUCCESS"}')


def _run(answer: str | None) -> export.Run:
    return export.Run(instance_id="i-1", task_name="webarena-verified/shopping-admin-7", answer=answer,
                      model="claude-haiku-5-5", agent_error=None, har=HAR, snapshot={"id": "u", "version": 1})


def test_a_run_is_written_as_the_files_the_evaluator_grades_and_a_second_run_of_a_task_is_refused(tmp_path):
    folder = export.write(_run('Done.\n{"task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": ["x"]}'),
                          tmp_path)
    text = export.write(_run("no answer"), tmp_path / "other")

    assert folder == tmp_path / "7" and (folder / "network.har").read_bytes() == HAR
    assert json.loads((folder / "agent_response.json").read_text())["retrieved_data"] == ["x"]
    assert json.loads((text / "agent_response.json").read_text()) == "no answer"
    record = json.loads((folder / "agentenv.json").read_text())
    assert (record["task_id"], record["model"], record["answer_extracted"], record["release"]) == (
        7, "claude-haiku-5-5", True, dataset.PACKAGE)
    assert json.loads((text / "agentenv.json").read_text())["answer_extracted"] is False
    with pytest.raises(export.NotExportable, match="is there already"):
        export.write(_run("again"), tmp_path)


class _File:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def load(self) -> bytes:
        return self.data


def _instance(task_id: str, metadata: dict, responses: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, context={"metadata": metadata, "prompt_responses": responses})


@pytest.mark.parametrize(("instance", "files", "why"), [
    (None, {}, "no run i-1 is recorded"),
    (_instance("online-mind2web/cdp-1", {}, []), {}, "not one of the bundle's tasks"),
    (_instance("webarena-verified/shopping-admin-7", {}, []), {}, "has no snapshot"),
    (_instance("webarena-verified/shopping-admin-7", {"env_snapshotted_universes": {"snapshot": {"id": "u",
                                                                                                 "version": 2}}}, []),
     {"browser-state.json": b"{}"}, "holds no network.har"),
])
def test_a_run_with_no_snapshot_or_no_network_har_in_it_isnt_exported(monkeypatch, instance, files, why):
    environment = SimpleNamespace(get_file_artifacts=lambda: {path: _File(data) for path, data in files.items()})
    monkeypatch.setattr(export, "find_task_instance", lambda instance_id: instance)
    monkeypatch.setattr(export.Artifact, "get", lambda id, version=None: SimpleNamespace(
        get_environment_artifacts=lambda: [environment]))

    with pytest.raises(export.NotExportable, match=why):
        export.load("i-1")


def test_a_run_is_loaded_from_its_snapshots_network_har_and_its_answer(monkeypatch):
    snapshot = {"id": "u", "version": 3}
    instance = _instance("webarena-verified/shopping-admin-7", {"env_snapshotted_universes": {"snapshot": snapshot}},
                         [{"prompt_id": "answer", "response": " {} ", "model": "claude-haiku-5-5"}])
    site = SimpleNamespace(get_file_artifacts=dict)
    browser = SimpleNamespace(get_file_artifacts=lambda: {"network.har": _File(HAR)})
    monkeypatch.setattr(export, "find_task_instance", lambda instance_id: instance)
    monkeypatch.setattr(export.Artifact, "get", lambda id, version=None: SimpleNamespace(
        get_environment_artifacts=lambda: [site, browser]))

    run = export.load("i-1")

    assert (run.task_name, run.answer, run.model, run.har, run.snapshot) == (
        "webarena-verified/shopping-admin-7", "{}", "claude-haiku-5-5", HAR, snapshot)


def test_grading_keeps_each_runs_result_beside_its_record_and_lists_the_ones_the_evaluator_errored_on(
        tmp_path, monkeypatch):
    for task_id, record in ((7, {"instance_id": "i-7"}), (9, {"instance_id": "i-9"})):
        (tmp_path / str(task_id)).mkdir()
        (tmp_path / str(task_id) / "agentenv.json").write_text(json.dumps(record))
    results = [{"task_id": 7, "status": "success", "score": 1.0}, {"task_id": 8, "status": "failure", "score": 0.0},
               {"task_id": 9, "status": "error", "score": 0.0, "error_msg": "Unknown trace format"}]
    monkeypatch.setattr(grade, "results", lambda out: results)

    summary = grade.grade(tmp_path)

    lines = [json.loads(line) for line in (tmp_path / "grades.jsonl").read_text().splitlines()]
    assert summary == {"scored": 2, "mean_score": 0.5, "unscored": [9]}
    assert [(line["task_id"], line["run"]) for line in lines] == [(7, {"instance_id": "i-7"}), (8, {}),
                                                                  (9, {"instance_id": "i-9"})]
    assert Counter(line["status"] for line in lines) == {"success": 1, "failure": 1, "error": 1}
