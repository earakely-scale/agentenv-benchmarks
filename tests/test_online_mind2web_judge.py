"""WebJudge on exported runs: the benchmark's own code, unchanged, grading a run as its run.py grades a v2 result,
each grade naming the judge's code and model and the snapshot that answered; a call that fails, a screenshot left
unscored, an empty verdict or another snapshot making the grade an error, never a failure."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from agentenv_benchmarks.online_mind2web import judge, v2

#: The judge's files as the benchmark's repository holds them at judge.JUDGE_COMMIT.
UPSTREAM = {"utils.py": "57e9252170173f86583ccfbcb3077d76b779c9f8719826efb902242383e94af7",
            "webjudge_online_mind2web.py": "e3cd499b7c1fc92cbd51d3c4216c95ebab774c2c2cd998c5ad6dad94710780c4"}


class Endpoint:
    """An OpenAI-compatible endpoint answering WebJudge's three kinds of call: key points, a screenshot's score and the
    verdict; ``score`` and ``verdict`` are its answers, ``fails`` makes every call raise."""

    def __init__(self, score="**Reasoning**: The cart shows the desk.\n**Score**: 4",
                 verdict='Thoughts: The desk is in the cart.\nStatus: "success"', model=judge.SNAPSHOT, fails=False):
        self.score, self.verdict, self.model, self.fails = score, verdict, model, fails
        self.asked: list[dict] = []

    def __call__(self, api_key=None, base_url=None):
        self.base_url = base_url
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def create(self, **kwargs):
        self.asked.append(kwargs)
        if self.fails:
            raise RuntimeError("the endpoint is down")
        system = kwargs["messages"][0]["content"]
        if "identify the key points" in system:
            content = "**Key Points**:\n1. Find a standing desk"
        elif "whether an image contains" in system:
            content = self.score
        else:
            content = self.verdict
        return SimpleNamespace(model=self.model, usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def exported(tmp_path):
    folder = tmp_path / "t1"
    (folder / v2.TRAJECTORY).mkdir(parents=True)
    for name in ("0000.jpg", "0001.jpg"):
        Image.new("RGB", (64, 40), "white").save(folder / v2.TRAJECTORY / name)
    (folder / v2.RESULT).write_text(json.dumps({
        "schema_version": v2.SCHEMA_VERSION, "task": "Find a standing desk.", "task_id": "t1",
        "agent_final_answer": "A desk.", "reference_length": 3, "action_history": [
            {"step": 0, "screenshot": "0000.jpg", "url": "https://shop.test/",
             "action": "page -> NAVIGATE -> go to https://shop.test/ | SUCCESS", "action_status": "SUCCESS",
             "thought": None},
            {"step": 1, "screenshot": "0001.jpg", "url": "https://shop.test/", "action":
             "TASK_COMPLETE -> ANSWER: A desk.", "action_status": None, "thought": None}]}))
    (folder / "agentenv.json").write_text(json.dumps({"instance_id": "run-1", "arm": "tools", "revision": "eacad896"}))
    return folder


def _graded(folder, monkeypatch, endpoint: Endpoint, snapshot=judge.SNAPSHOT) -> dict:
    monkeypatch.setattr(judge, "OpenAI", endpoint)
    engine = judge.Engine(judge.MODEL, "key", "https://models.test/v1", snapshot)
    return judge.grade(folder, engine)


def test_the_judges_code_is_the_benchmarks_own_unchanged():
    folder = judge.Path(judge.__file__).with_name("webjudge")
    assert {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in UPSTREAM} == UPSTREAM


def test_a_run_is_graded_as_the_benchmarks_run_py_grades_it_naming_the_judge_and_its_model(exported, monkeypatch):
    endpoint = Endpoint()
    graded = _graded(exported, monkeypatch, endpoint)

    assert (graded["grade"], graded["predicted_label"], graded["scores"]) == ("success", 1, [4, 4])
    assert (graded["task_id"], graded["instance_id"], graded["arm"], graded["revision"]) == (
        "t1", "run-1", "tools", "eacad896")
    assert graded["judge"] == {"code": judge.JUDGE_CODE, "mode": "WebJudge_Online_Mind2Web_eval", "score_threshold": 3,
                               "model": "gpt-6-luna", "snapshot": judge.SNAPSHOT, "max_tokens": judge.MAX_TOKENS}
    assert len(endpoint.asked) == 4 and {call["max_tokens"] for call in endpoint.asked} == {judge.MAX_TOKENS}
    assert {call["model"] for call in endpoint.asked} == {"gpt-6-luna"} and endpoint.base_url == "https://models.test/v1"


def test_a_failure_is_graded_failure(exported, monkeypatch):
    graded = _graded(exported, monkeypatch, Endpoint(verdict='Thoughts: No desk.\nStatus: "failure"'))
    assert (graded["grade"], graded["predicted_label"]) == ("failure", 0)


@pytest.mark.parametrize("endpoint, why", [
    (Endpoint(verdict=""), "no verdict"),
    (Endpoint(verdict="Thoughts: I can't tell."), "no verdict"),
    (Endpoint(score=""), "got no score"),
    (Endpoint(fails=True), "the endpoint is down"),
    (Endpoint(model="gpt-6-luna-2027-01-01"), "not gpt-6-luna-2026-09-22"),
])
def test_a_judge_that_fails_or_leaves_a_call_unanswered_makes_the_grade_an_error_not_a_failure(
        exported, monkeypatch, endpoint, why):
    graded = _graded(exported, monkeypatch, endpoint)
    assert graded["grade"] == "error" and why in graded["error"] and "predicted_label" not in graded


def test_the_command_grades_every_run_and_fails_when_one_is_an_error(exported, monkeypatch, capsys):
    monkeypatch.setenv("LITELLM_API_KEY", "key")
    monkeypatch.setattr(judge, "OpenAI", Endpoint())
    judge.main([str(exported.parent)])
    (line,) = [json.loads(text) for text in (exported.parent / judge.GRADES).read_text().splitlines()]
    assert line["grade"] == "success" and line["reported"][0]["model"] == judge.SNAPSHOT
    assert "1 success, 0 failure, 0 error" in capsys.readouterr().out

    monkeypatch.setattr(judge, "OpenAI", Endpoint(verdict=""))
    with pytest.raises(SystemExit, match="1 of 1 runs weren't graded"):
        judge.main([str(exported.parent)])
