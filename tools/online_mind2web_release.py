"""Online-Mind2Web's results on the Hub: a dataset of every run's export, WebJudge grade and frames, and the Space that
views it.

    python tools/online_mind2web_release.py build OUT --pass NAME=EXPORTS [--pass ...] [--note NAME=TEXT ...]
    python tools/online_mind2web_release.py upload OUT [--dataset REPO] [--space REPO]

EXPORTS holds `tools/` and `cdp/`, each a folder `export` wrote and `judge` graded: a folder per task and
`webjudge.jsonl`, and `settings.json`, the pass's models and their settings: for each arm its agent, version, model,
effort, thinking, output and step limits, and how each was found, and the judge's model, snapshot, reasoning effort,
max tokens and temperature. `build` writes OUT as the dataset's files, refusing a run whose model or judge snapshot
isn't its pass's, and anything that looks like a credential; `upload` creates the dataset and the Space, public, and
uploads each in one commit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from agentenv_benchmarks.online_mind2web import BUNDLE, dataset, judge

DATASET, SPACE = "earakely-scale/OnlineMind2Web-AgentEnv", "earakely-scale/OnlineMind2Web-AgentEnv"
SPACE_SOURCE = Path(__file__).resolve().parent.parent / "space" / "online-mind2web"
ARMS = {"tools": "Claude Code with Playwright MCP's browser_* tools", "cdp": "Browser Use over CDP"}
#: A run's frames, as the Space shows them: this wide, stacked into sheets of this many, well under JPEG's 65,535 px.
WIDTH, QUALITY, PER_SHEET = 800, 68, 24
#: What each pass's settings.json says, for each arm and for the judge.
AGENT_KEYS = ("agent", "version", "drives", "model", "model_name", "effort", "thinking", "max_output_tokens",
              "step_limit", "time_limit_s", "how")
JUDGE_KEYS = ("model", "model_name", "snapshot", "reasoning_effort", "max_tokens", "temperature", "score_threshold",
              "mode", "code", "how")
#: What a credential looks like, which no published file may hold.
#: Each starts at a word's start, so a word that holds one, as a URL's ``ask-construction-co`` holds ``sk-``, isn't one.
SECRETS = re.compile(r"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|ASIA[0-9A-Z]{16})|"
                     r"Bearer [A-Za-z0-9._-]{20,}|LITELLM_API_KEY=")


def build(out: Path, passes: dict[str, Path], notes: dict[str, str]) -> None:
    (out / "viewer").mkdir(parents=True, exist_ok=True)
    (out / "exports").mkdir(exist_ok=True)
    rows, index = [], []
    for name, root in passes.items():
        settings = _settings(root / "settings.json")
        runs = []
        for arm in ARMS:
            folder = root / arm
            if not folder.is_dir():
                continue
            if arm not in settings["agents"]:
                raise SystemExit(f"{root / 'settings.json'} names no agent for the {arm} arm")
            grades = {json.loads(line)["task_id"]: json.loads(line)
                      for line in (folder / judge.GRADES).read_text().splitlines() if line.strip()}
            for task in sorted(p.parent for p in folder.glob("*/result.json")):
                runs.append(_run(name, arm, task, grades.get(task.name) or {}, settings, out))
            _tar(folder, out / "exports" / f"{name}-{arm}.tar.gz")
        (out / "viewer" / f"{name}.json").write_text(json.dumps({"pass": name, "note": notes.get(name), "runs": runs}))
        rows += [{key: value for key, value in run.items() if key not in ("steps", "width", "sheets")}
                 for run in runs]
        index.append({"name": name, "note": notes.get(name), "runs": len(runs), "graded": _rates(runs),
                      "settings": settings})
    (out / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "viewer" / "index.json").write_text(json.dumps({
        "generated": datetime.now(UTC).isoformat(timespec="seconds"), "dataset": dataset.REPO,
        "revision": dataset.REVISION, "passes": index}))
    (out / "README.md").write_text(_card(index))
    _scan(out, passes.values())
    print(f"{len(rows)} runs in {len(index)} passes built into {out}")


def _settings(path: Path) -> dict:
    """A pass's models and settings, each arm's and the judge's, refused when one leaves a key out."""
    if not path.is_file():
        raise SystemExit(f"{path} is missing: a pass says which models ran it and at what effort")
    settings = json.loads(path.read_text())
    for arm, agent in (settings.get("agents") or {}).items():
        if missing := [key for key in AGENT_KEYS if key not in agent]:
            raise SystemExit(f"{path}: the {arm} agent leaves out {', '.join(missing)}")
    if missing := [key for key in JUDGE_KEYS if key not in (settings.get("judge") or {})]:
        raise SystemExit(f"{path}: the judge leaves out {', '.join(missing)}")
    return settings


def _run(name: str, arm: str, folder: Path, graded: dict, settings: dict, out: Path) -> dict:
    result = json.loads((folder / "result.json").read_text())
    meta = json.loads((folder / "agentenv.json").read_text())
    agent, judged = settings["agents"][arm], settings["judge"]
    if meta.get("model") != agent["model"]:
        raise SystemExit(f"{folder} ran {meta.get('model')}, not its pass's {agent['model']}")
    ran = graded.get("judge") or {}
    if graded and (ran.get("snapshot"), ran.get("max_tokens")) != (judged["snapshot"], judged["max_tokens"]):
        raise SystemExit(f"{folder} was graded on {ran.get('snapshot')} with {ran.get('max_tokens')} tokens, not its "
                         f"pass's {judged['snapshot']} with {judged['max_tokens']}")
    setup = next(s for s in json.loads((BUNDLE / "tasks" / f"{arm}-{result['task_id']}.json").read_text())
                 if s["id"] == "setup")["args"]
    stem = f"viewer/{name}/{arm}-{result['task_id']}"
    sheets, steps = [], []
    history = result["action_history"]
    for first in range(0, len(history), PER_SHEET):
        images, y = [], 0
        for step in history[first:first + PER_SHEET]:
            with Image.open(folder / "trajectory" / step["screenshot"]) as image:
                image = image.convert("RGB")
                height = round(image.height * WIDTH / image.width)
                images.append(image.resize((WIDTH, height), Image.LANCZOS))
            steps.append({"action": step["action"], "url": step.get("url"), "status": step.get("action_status"),
                          "sheet": len(sheets), "y": y, "h": height})
            y += height
        sheet = Image.new("RGB", (WIDTH, y), "white")
        offset = 0
        for image in images:
            sheet.paste(image, (0, offset))
            offset += image.height
        path = f"{stem}-{len(sheets)}.jpg"
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out / path, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        sheets.append({"path": path, "height": y})
    if len(scores := graded.get("scores") or []) == len(steps):
        for step, score in zip(steps, scores):
            step["score"] = score
    usage = [r.get("usage") or {} for r in graded.get("reported") or []]
    return {"pass": name, "arm": arm, "agent": f"{agent['agent']} {agent['version']}", "agent_model": agent["model"],
            "agent_effort": agent["effort"], "agent_thinking": agent["thinking"], "task_id": result["task_id"],
            "task": result["task"], "website": setup["website"], "level": setup["level"],
            "reference_length": result["reference_length"], "actions": len(result["action_history"]) - 1,
            "answer": result.get("agent_final_answer"), "grade": graded.get("grade") or "ungraded",
            "verdict": graded.get("response") or graded.get("error"), "key_points": graded.get("key_points"),
            "scores": graded.get("scores"), "judge_model": judged["model"], "judge_snapshot": ran.get("snapshot"),
            "judge_effort": judged["reasoning_effort"],
            "judge_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "judge_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "instance_id": meta.get("instance_id"), "recording_id": meta.get("recording_id"),
            "dataset_revision": meta.get("revision"), "export": f"exports/{name}-{arm}.tar.gz",
            "width": WIDTH, "sheets": sheets, "steps": steps}


def _rates(runs: list[dict]) -> dict:
    """Per arm, and per arm and level, how many runs WebJudge graded each way."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for run in runs:
        counts[run["arm"]][run["grade"]] += 1
        counts[f"{run['arm']}/{run['level']}"][run["grade"]] += 1
    return {key: dict(value) for key, value in sorted(counts.items())}


def _tar(folder: Path, path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)


def _scan(out: Path, roots) -> None:
    """Refuse to publish a file that holds anything shaped like a credential, the LiteLLM key in use or this
    machine's home folder."""
    needles = [value for value in (os.environ.get("LITELLM_API_KEY"), str(Path.home())) if value]
    found = []
    for base in [out, *roots]:
        for path in base.rglob("*"):
            if path.suffix in (".json", ".jsonl", ".md", ".txt") and path.is_file():
                text = path.read_text(errors="replace")
                if SECRETS.search(text) or any(needle in text for needle in needles):
                    found.append(str(path))
    if found:
        raise SystemExit(f"refusing to publish: {len(found)} files hold what looks like a credential: {found[:5]}")


def _card(index: list[dict]) -> str:
    rows = []
    for item in index:
        for arm, agent in ARMS.items():
            graded = item["graded"].get(arm)
            if not graded:
                continue
            total = sum(graded.values())
            rows.append(f"| {item['name']} | {agent} | {total} | {graded.get('success', 0)} | "
                        f"{graded.get('success', 0) / total:.0%} | {graded.get('error', 0)} |")
    notes = "\n".join(f"- **{item['name']}:** {item['note']}" for item in index if item.get("note"))
    models, judges, hows = [], [], []
    for names, settings in _by_settings(index):
        passes = ", ".join(names)
        for arm, agent in settings["agents"].items():
            models.append(f"| {passes} | `{arm}` | {agent['agent']} {agent['version']} | {agent['model_name']} "
                          f"(`{agent['model']}`) | **{agent['effort']}** | {agent['thinking']} | "
                          f"{agent['max_output_tokens']:,} | {agent['step_limit'] or 'none'} | {agent['time_limit_s']} s |")
            hows.append(f"- **`{arm}`{'' if len(index) == 1 else ', ' + passes}:** {agent['how']}")
        j = settings["judge"]
        judges.append(f"| {passes} | {j['model_name']} (`{j['snapshot']}`) | **{j['reasoning_effort']}** | "
                      f"{j['max_tokens']:,} | {j['temperature']} | {j['score_threshold']} |")
        hows.append(f"- **The judge{'' if len(index) == 1 else ', ' + passes}:** {j['how']}")
    headline = _headline(index)
    return f"""---
license: cc-by-4.0
pretty_name: Online-Mind2Web on agent-env
tags:
- web-agents
- browser-agents
- online-mind2web
- webjudge
- agent-env
configs:
- config_name: runs
  data_files: runs.jsonl
---

# Online-Mind2Web on agent-env

Runs of [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web)'s 300 live-web tasks on
[agent-env](https://github.com/scaleapi/agentenv-framework)'s own browser, graded by WebJudge. Browse them in the
[viewer Space](https://huggingface.co/spaces/{SPACE}).

The tasks come from [`{dataset.REPO}`](https://huggingface.co/datasets/{dataset.REPO}) at revision `{dataset.REVISION}`,
as the [Online-Mind2Web bundle](https://github.com/earakely-scale/agentenv-benchmarks) runs them.

{headline}

## Models and effort

| Pass | Arm | Agent | Model | Effort | Thinking | Max output tokens | Step limit | Time limit |
|---|---|---|---|---|---|---:|---:|---:|
{chr(10).join(models)}

| Pass | Judge | Reasoning effort | Max tokens | Temperature | Score threshold |
|---|---|---|---:|---:|---:|
{chr(10).join(judges)}

{chr(10).join(hows)}

## Results

| Pass | Arm | Runs | Success | Rate | Judge errors |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

{notes}

## How the runs were made

- **Browser:** agent-env's `chromium` computer, deployed alone and headful, on local Docker. Every arm uses the same
  browser, set up the same way: a 1280×800 window, a blank start tab, dialogs accepted, en-US, UTC.
- **Arms:** `tools` is Claude Code driving Playwright MCP's `browser_*` tools. `cdp` is Browser Use with its own CDP
  connection to the browser. Each is prompted `Go to {{website}} and {{task}}`.
- **Recording:** the browser records a frame of the tab the agent acted in after each burst of its actions, keyed to
  its action log. A run with a frame or log line missing fails as an infrastructure failure and isn't here.
- **Grading:** WebJudge, Online-Mind2Web's own judge, its code unchanged from
  [`{judge.JUDGE_CODE}`]({'https://github.com/OSU-NLP-Group/Online-Mind2Web/tree/' + judge.JUDGE_COMMIT}), in its
  `WebJudge_Online_Mind2Web_eval` mode on `{judge.MODEL}` (snapshot `{judge.SNAPSHOT}`). Answers may take 8,192 tokens,
  not the 512 its code asks for, since a reasoning model's verdict comes back empty at 512. No run was hand-labelled.
- **Sites that block:** 21 tasks on 9 sites (apartments.com, petfinder, imdb, backcountry, microcenter, boardgamegeek,
  marriott, uniqlo, ticketmaster) were blocked by the sites themselves in agent-env's pre-flight sweep. They stay in,
  and a run there usually fails.

## Files

- `runs.jsonl`: a row per run: its pass, arm, agent and version, model, effort and thinking, task, website, level,
  the agent's answer, WebJudge's model, snapshot, effort, grade, verdict and key points, and the run's ids.
- `exports/<pass>-<arm>.tar.gz`: each run in Online-Mind2Web's v2 result format (`result.json` and `trajectory/`),
  WebJudge's grades (`webjudge.jsonl`), and `agentenv.json`, which names the run. Regrade with
  `python -m agentenv_benchmarks.online_mind2web.judge`.
- `viewer/`: what the Space shows: a pass's runs with their steps, each run's frames stacked into sheets, and
  `index.json`, each pass's results and its models and settings, as the tables above give them.

## License and attribution

The task text, websites, levels and reference lengths are Online-Mind2Web's, by the OSU NLP Group, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Xue et al., "An Illusion of Progress? Assessing the
Current State of Web Agents" (2025), and Deng et al., "Mind2Web: Towards a Generalist Agent for the Web" (NeurIPS
2023). The frames are screenshots of public websites as the agents saw them, kept for reviewing the runs; the sites'
content stays their owners'.
"""


def _by_settings(index: list[dict]) -> list[tuple[list[str], dict]]:
    """The passes grouped by their settings, each group's names with the settings they share, in pass order."""
    groups: dict[str, tuple[list[str], dict]] = {}
    for item in index:
        groups.setdefault(json.dumps(item["settings"], sort_keys=True), ([], item["settings"]))[0].append(item["name"])
    return list(groups.values())


def _headline(index: list[dict]) -> str:
    """Each arm's model and effort and the judge's, once for passes that share them."""
    thinking = {"adaptive": ", adaptive thinking", "off": ", no reasoning"}
    lines = []
    for names, settings in _by_settings(index):
        agents = "; ".join(f"`{arm}`, {a['agent']} on {a['model_name']} (`{a['model']}`) at effort **{a['effort']}**"
                           f"{thinking.get(a['thinking'], '')}" for arm, a in settings["agents"].items())
        j = settings["judge"]
        judged = f"Judge: WebJudge on {j['model_name']} (`{j['snapshot']}`) at reasoning effort **{j['reasoning_effort']}**."
        lines.append((names, f"{agents}. {judged}"))
    if len(lines) == 1:
        return f"**{'Every pass' if len(index) > 1 else index[0]['name']}:** {lines[0][1]}"
    return "\n".join(f"- **{', '.join(names)}:** {line}" for names, line in lines)


def upload(out: Path, dataset_repo: str, space_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(dataset_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=dataset_repo, repo_type="dataset", folder_path=str(out),
                      commit_message=f"Online-Mind2Web runs, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")
    api.create_repo(space_repo, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=str(SPACE_SOURCE),
                      commit_message="Online-Mind2Web viewer")
    print(f"https://huggingface.co/datasets/{dataset_repo}\nhttps://huggingface.co/spaces/{space_repo}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    building = sub.add_parser("build")
    building.add_argument("out", type=Path)
    building.add_argument("--pass", dest="passes", action="append", required=True, metavar="NAME=EXPORTS")
    building.add_argument("--note", action="append", default=[], metavar="NAME=TEXT")
    uploading = sub.add_parser("upload")
    uploading.add_argument("out", type=Path)
    uploading.add_argument("--dataset", default=DATASET)
    uploading.add_argument("--space", default=SPACE)
    args = parser.parse_args(argv)
    if args.command == "build":
        passes = dict(item.split("=", 1) for item in args.passes)
        build(args.out, {name: Path(root) for name, root in passes.items()},
              dict(item.split("=", 1) for item in args.note))
    else:
        upload(args.out, args.dataset, args.space)


if __name__ == "__main__":
    main(sys.argv[1:])
