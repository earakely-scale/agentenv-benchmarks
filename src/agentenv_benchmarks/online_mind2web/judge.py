"""WebJudge, Online-Mind2Web's judge, grading exported runs (``export``) offline with a pinned, named model:

    python -m agentenv_benchmarks.online_mind2web.judge OUT [--model gpt-6-luna]

It runs the benchmark's own WebJudge code, ``webjudge/``, copied unchanged from its repository at ``JUDGE_COMMIT``: the
same key points, per-screenshot scores and verdict prompt, read as ``src/run.py`` reads a v2 result in its
``WebJudge_Online_Mind2Web_eval`` mode. The model is reached through an OpenAI-compatible endpoint, ``LITELLM_BASE_URL``
and ``LITELLM_API_KEY`` as agent-env's own, else ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``.

Each run's grade is written to ``OUT/webjudge.jsonl``, a line per run naming the judge's code, mode and model, and the
model the endpoint answered with. A judge that fails, or answers a call with no score or no verdict, which WebJudge's
own code would read as a failure, makes the run's grade an error, an infrastructure failure, never a failure."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

from openai import OpenAI

from . import v2

#: The benchmark's commit the judge's code is copied from, and its mode and threshold, as its ``script/eval.sh`` runs.
JUDGE_COMMIT = "1a8db93168d5d7870402b7f453160e1c557153c8"
JUDGE_CODE = f"OSU-NLP-Group/Online-Mind2Web@{JUDGE_COMMIT}"
MODE = "WebJudge_Online_Mind2Web_eval"
SCORE_THRESHOLD = 3
#: The model WebJudge's authors recommend for the benchmark's automatic evaluation, at its default reasoning effort,
#: and the snapshot every answer must come from, so a grade never mixes snapshots.
MODEL, SNAPSHOT = "gpt-6-luna", "gpt-6-luna-2026-09-22"
#: How many tokens each of the judge's answers may take, its reasoning included. WebJudge's code asks for 512, written
#: for models that don't reason, which leaves a reasoning model's verdict empty; nothing else of its calls changes.
MAX_TOKENS = 8192
GRADES = "webjudge.jsonl"
_VERDICT = re.compile(r"status:\s*\**\s*\"?(success|failure)", re.IGNORECASE)


def _webjudge() -> tuple[ModuleType, ModuleType]:
    """The judge's ``utils`` and ``webjudge_online_mind2web``, which imports ``utils`` by that name, as its ``src``
    holds them."""
    folder = Path(__file__).with_name("webjudge")
    utils = _module("utils", folder / "utils.py")
    previous = sys.modules.get("utils")
    sys.modules["utils"] = utils
    try:
        return utils, _module("webjudge_online_mind2web", folder / "webjudge_online_mind2web.py")
    finally:
        if previous is None:
            sys.modules.pop("utils")
        else:
            sys.modules["utils"] = previous


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"agentenv_benchmarks.online_mind2web.webjudge.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UTILS, WEBJUDGE = _webjudge()


class Engine(UTILS.OpenaiEngine):
    """WebJudge's model engine on the given endpoint, its answers up to ``MAX_TOKENS``, keeping each one's reported
    model and usage; ``snapshot`` is the model every answer must come from, when one is pinned."""

    def __init__(self, model: str, api_key: str, base_url: str | None, snapshot: str | None = None) -> None:
        super().__init__(api_key=api_key, model=model)
        self.snapshot = snapshot
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.reported: list[dict] = []
        create = self.client.chat.completions.create

        def recording(**kwargs):
            response = create(**kwargs)
            usage = getattr(response, "usage", None)
            self.reported.append({"model": response.model, "usage": usage.model_dump() if usage else None})
            return response

        self.client.chat.completions.create = recording

    def generate(self, messages, max_new_tokens=None, **kwargs):
        return super().generate(messages, max_new_tokens=MAX_TOKENS, **kwargs)


def grade(folder: Path, engine: Engine) -> dict:
    """The run in ``folder`` graded as ``src/run.py`` grades a v2 result: its success or failure, or why there is
    neither."""
    result = json.loads((folder / v2.RESULT).read_text(encoding="utf-8"))
    run = json.loads((folder / "agentenv.json").read_text(encoding="utf-8")) if (folder / "agentenv.json").is_file() else {}
    steps = result["action_history"]
    actions = [step["action"] for step in steps]
    screenshots = [str(folder / v2.TRAJECTORY / step["screenshot"]) for step in steps]
    line = {"task_id": result["task_id"], "instance_id": run.get("instance_id"), "arm": run.get("arm"),
            "revision": run.get("revision"), "judge": {
                "code": JUDGE_CODE, "mode": MODE, "score_threshold": SCORE_THRESHOLD, "model": engine.model,
                "snapshot": engine.snapshot, "max_tokens": MAX_TOKENS}}
    try:
        messages, _, _, record, key_points = asyncio.run(WEBJUDGE.WebJudge_Online_Mind2Web_eval(
            result["task"], actions, screenshots, engine, SCORE_THRESHOLD))
        response = engine.generate(messages)[0]
    except Exception as e:  # noqa: BLE001 - any failure of the judge's is an infrastructure failure
        return {**line, "grade": "error", "error": f"{type(e).__name__}: {e}"[:500]}
    line.update(key_points=key_points, scores=[entry["Score"] for entry in record], response=response)
    if engine.snapshot and (others := sorted({r["model"] for r in engine.reported} - {engine.snapshot})):
        return {**line, "grade": "error", "error": f"answered by {', '.join(others)}, not {engine.snapshot}"}
    if unscored := [index for index, entry in enumerate(record) if not entry.get("Response") or not entry["Score"]]:
        return {**line, "grade": "error", "error": f"screenshots {unscored} got no score"}
    if not response or not (verdict := _VERDICT.search(response)):
        return {**line, "grade": "error", "error": "the judge gave no verdict"}
    predicted = UTILS.extract_predication(response, MODE)
    if (verdict[1].lower() == "success") != bool(predicted):
        return {**line, "grade": "error", "error": "the judge's verdict didn't read as WebJudge reads it"}
    return {**line, "grade": "success" if predicted else "failure", "predicted_label": predicted}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder export wrote, a folder per task")
    parser.add_argument("--model", default=MODEL, help=f"the judge's model (default {MODEL})")
    parser.add_argument("--snapshot", help=f"the snapshot every answer must come from (default {SNAPSHOT} for {MODEL})")
    parser.add_argument("--workers", type=int, default=4, help="runs graded at once")
    args = parser.parse_args(argv)
    snapshot = args.snapshot or (SNAPSHOT if args.model == MODEL else None)
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set LITELLM_API_KEY (or OPENAI_API_KEY) for the judge's model")
    base_url = os.environ.get("LITELLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    folders = sorted(path.parent for path in args.out.glob(f"*/{v2.RESULT}"))
    if not folders:
        raise SystemExit(f"{args.out} holds no exported run")

    def graded(folder: Path) -> dict:
        engine = Engine(args.model, api_key, base_url, snapshot)
        line = grade(folder, engine)
        return {**line, "reported": engine.reported}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        lines = list(pool.map(graded, folders))
    (args.out / GRADES).write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    counts = {grade: sum(line["grade"] == grade for line in lines) for grade in ("success", "failure", "error")}
    print(f"{args.out / GRADES}: {counts['success']} success, {counts['failure']} failure, {counts['error']} error, "
          f"by {args.model} ({', '.join(sorted({r['model'] for line in lines for r in line['reported']})) or 'no call'})")
    for line in lines:
        if line["grade"] == "error":
            print(f"  {line['task_id']}: {line['error']}", file=sys.stderr)
    if counts["error"]:
        raise SystemExit(f"{counts['error']} of {len(lines)} runs weren't graded: an infrastructure failure")


if __name__ == "__main__":
    main(sys.argv[1:])
