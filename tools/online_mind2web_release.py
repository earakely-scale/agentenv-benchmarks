"""Online-Mind2Web's results on the Hub: a dataset of every run's export, WebJudge grade and frames, and the Space that
views it.

    python tools/online_mind2web_release.py build OUT --pass NAME=EXPORTS [--pass ...] [--note NAME=TEXT ...]
    python tools/online_mind2web_release.py upload OUT [--dataset REPO] [--space REPO]

EXPORTS holds `tools/` and `cdp/`, each a folder `export` wrote and `judge` graded: a folder per task and
`webjudge.jsonl`. `build` writes OUT as the dataset's files, refusing anything that looks like a credential; `upload`
creates the dataset and the Space, public, and uploads each in one commit.
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
#: What a credential looks like, which no published file may hold.
SECRETS = re.compile(r"sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|Bearer [A-Za-z0-9._-]{20,}|"
                     r"LITELLM_API_KEY=|ASIA[0-9A-Z]{16}")


def build(out: Path, passes: dict[str, Path], notes: dict[str, str]) -> None:
    (out / "viewer").mkdir(parents=True, exist_ok=True)
    (out / "exports").mkdir(exist_ok=True)
    rows, index = [], []
    for name, root in passes.items():
        runs = []
        for arm in ARMS:
            folder = root / arm
            if not folder.is_dir():
                continue
            grades = {json.loads(line)["task_id"]: json.loads(line)
                      for line in (folder / judge.GRADES).read_text().splitlines() if line.strip()}
            for task in sorted(p.parent for p in folder.glob("*/result.json")):
                runs.append(_run(name, arm, task, grades.get(task.name) or {}, out))
            _tar(folder, out / "exports" / f"{name}-{arm}.tar.gz")
        (out / "viewer" / f"{name}.json").write_text(json.dumps({"pass": name, "note": notes.get(name), "runs": runs}))
        rows += [{key: value for key, value in run.items() if key not in ("steps", "width", "sheets")}
                 for run in runs]
        index.append({"name": name, "note": notes.get(name), "runs": len(runs), "graded": _rates(runs)})
    (out / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "viewer" / "index.json").write_text(json.dumps({
        "generated": datetime.now(UTC).isoformat(timespec="seconds"), "dataset": dataset.REPO,
        "revision": dataset.REVISION, "judge": {"code": judge.JUDGE_CODE, "model": judge.MODEL,
                                                "snapshot": judge.SNAPSHOT}, "passes": index}))
    (out / "README.md").write_text(_card(index))
    _scan(out, passes.values())
    print(f"{len(rows)} runs in {len(index)} passes built into {out}")


def _run(name: str, arm: str, folder: Path, graded: dict, out: Path) -> dict:
    result = json.loads((folder / "result.json").read_text())
    meta = json.loads((folder / "agentenv.json").read_text())
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
    return {"pass": name, "arm": arm, "agent": ARMS[arm], "model": meta.get("model"), "task_id": result["task_id"],
            "task": result["task"], "website": setup["website"], "level": setup["level"],
            "reference_length": result["reference_length"], "actions": len(result["action_history"]) - 1,
            "answer": result.get("agent_final_answer"), "grade": graded.get("grade") or "ungraded",
            "verdict": graded.get("response") or graded.get("error"), "key_points": graded.get("key_points"),
            "scores": graded.get("scores"), "judge_snapshot": (graded.get("judge") or {}).get("snapshot"),
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

- `runs.jsonl`: a row per run: its pass, arm, model, task, website, level, the agent's answer, WebJudge's grade,
  verdict and key points, and the run's ids.
- `exports/<pass>-<arm>.tar.gz`: each run in Online-Mind2Web's v2 result format (`result.json` and `trajectory/`),
  WebJudge's grades (`webjudge.jsonl`), and `agentenv.json`, which names the run. Regrade with
  `python -m agentenv_benchmarks.online_mind2web.judge`.
- `viewer/`: what the Space shows: a pass's runs with their steps, and each run's frames stacked into sheets.

## License and attribution

The task text, websites, levels and reference lengths are Online-Mind2Web's, by the OSU NLP Group, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Xue et al., "An Illusion of Progress? Assessing the
Current State of Web Agents" (2025), and Deng et al., "Mind2Web: Towards a Generalist Agent for the Web" (NeurIPS
2023). The frames are screenshots of public websites as the agents saw them, kept for reviewing the runs; the sites'
content stays their owners'.
"""


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
