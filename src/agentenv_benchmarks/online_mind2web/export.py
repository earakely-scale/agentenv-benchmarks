"""Online-Mind2Web's exporter: a run of one of the bundle's tasks as the benchmark's v2 result, the folder WebJudge
grades, from the browser's own record of the run rather than the agent's account of it:

    python -m agentenv_benchmarks.online_mind2web.export OUT INSTANCE_ID...

Each step is a frame the browser took after a burst of the agent's actions, with the benchmark's grammar A for that
burst: the agent's ``browser_*`` calls on the ``tools`` arm, its CDP commands on ``cdp``, read from the action log the
frame is keyed to. A call that spans several bursts is one step, with the frame of its last. The start frame, a blank
tab, isn't a step; the last frame is the ``TASK_COMPLETE`` step, with the agent's answer. Thoughts are null.

A run is written to ``OUT/<task_id>/``, beside ``agentenv.json``, which names the run, its arm and the dataset revision
it ran, and the error its agent ended with, if any; a run whose frames recording didn't end complete isn't exported. A
run whose agent failed, running out of steps say, is exported as any other: its recording, left open by the run and
ended by its teardown, is read from where the browser wrote it."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_env.artifact import Artifact
from agent_env.config import get_config
from agent_env.store.routing import namespace_routing
from agent_env.task.store import find_task_instance

from . import template, v2

#: Steps the judge reads as observations: reads, waits and settings that change nothing on the page.
WAIT = "WAIT"
#: The verbs whose target comes first in grammar A, ``page -> VERB -> ...``, as the benchmark's dictionary writes them.
PAGE_VERBS = ("NAVIGATE", "GO_BACK", "GO_FORWARD", "REFRESH")
_READS = {
    "browser_snapshot": "read the page's accessibility snapshot",
    "browser_take_screenshot": "take a screenshot",
    "browser_find": "find text on the page",
    "browser_console_messages": "read the console",
    "browser_network_requests": "list the page's network requests",
    "browser_network_request": "read a network request",
    "browser_resize": "resize the window",
    "browser_emulate_media": "emulate a media feature",
}


@dataclass
class Run:
    """A run of one of the bundle's tasks: its record (``script_results.setup``), the agent's answer, its frames
    recording's manifest and log, and each frame's bytes by its path in the manifest."""

    instance_id: str
    record: dict
    answer: str | None
    manifest: dict
    log: list[dict]
    frame: Callable[[str], bytes]
    model: str | None = None
    #: The error the agent ended with, as its prompt response names it, such as running out of steps.
    agent_error: str | None = None


@dataclass
class _Item:
    verb: str
    target: str
    description: str
    ok: bool = True
    #: Still going when the run ended, as the recording's log marks such an action.
    unfinished: bool = False


class NotExportable(Exception):
    """A run that can't be exported, saying why."""


def steps(manifest: dict, log: list[dict]) -> list[tuple[dict, str, str | None]]:
    """Each action step's frame, its action in grammar A and its status, in order."""
    entries = {line["seq"]: line["action"] for line in log if "action" in line}
    unseen = _unseen(entries)
    frames = [frame for frame in manifest["frames"] if frame["kind"] == "action"]
    covered = [_covered(frame, entries, unseen) for frame in frames]
    last_of = {}
    for index, keys in enumerate(covered):
        for key in keys:
            if key[0] == "mcp":
                last_of[key] = index
    found = []
    for index, (frame, keys) in enumerate(zip(frames, covered)):
        mine = [key for key in keys if key[0] == "cdp" or last_of[key] == index]
        if mine:
            items = [item for key in sorted(mine, key=_seq) if (item := _item(key, entries)) is not None]
            found.append((frame, *_action(_merged(items))))
    return found


def result(run: Run) -> tuple[dict, list[tuple[str, bytes]]]:
    """The run's ``result.json`` and its screenshots, by name."""
    if missing := [f["index"] for f in run.manifest["frames"] if not f.get("uploaded")]:
        raise NotExportable(f"its recording is missing frames {missing}")
    last = next((frame for frame in reversed(run.manifest["frames"]) if frame["kind"] == "last"), None)
    if last is None:
        raise NotExportable("its recording has no last frame")
    answered = f"{v2.ANSWER} {run.answer}" if run.answer else f"{v2.COMPLETE} -> the agent gave no answer"
    history, shots = [], []
    for index, (frame, action, status) in enumerate([*steps(run.manifest, run.log), (last, answered, None)]):
        name = f"{index:04d}.{frame['path'].rsplit('.', 1)[-1]}"
        history.append({"step": index, "screenshot": name, "url": frame.get("url"), "action": action,
                        "action_status": status, "thought": None})
        shots.append((name, run.frame(frame["path"])))
    record = run.record
    return {"schema_version": v2.SCHEMA_VERSION, "task": record["task"], "task_id": record["task_id"],
            "agent_final_answer": run.answer or None, "reference_length": record["reference_length"],
            "action_history": history}, shots


def write(run: Run, out: Path) -> Path:
    """The run written as ``out/<task_id>/``, checked as the benchmark's consumers check it; refused when that folder
    is there already, as for a second run of one task."""
    document, shots = result(run)
    folder = out / document["task_id"]
    if folder.exists():
        raise NotExportable(f"{folder} is there already; export each run of a task into a folder of its own")
    (folder / v2.TRAJECTORY).mkdir(parents=True)
    for name, data in shots:
        (folder / v2.TRAJECTORY / name).write_bytes(data)
    (folder / v2.RESULT).write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    record = run.record
    (folder / "agentenv.json").write_text(json.dumps({
        "instance_id": run.instance_id, "arm": record.get("arm"), "model": run.model,
        "dataset": record.get("dataset"), "revision": record.get("revision"), "sha256": record.get("sha256"),
        "level": record.get("level"), "browser": record.get("browser"),
        "recording_id": run.manifest.get("recording_id"), "recording_ended_by": run.manifest.get("ended_by"),
        "agent_error": run.agent_error, "schema_commit": v2.SCHEMA_COMMIT}, indent=1) + "\n",
        encoding="utf-8")
    if found := v2.problems(folder):
        raise NotExportable(f"{folder} doesn't validate: {'; '.join(found)}")
    return folder


def load(instance_id: str) -> Run:
    """The run ``instance_id`` from the configured stores, or a bundle run's from the local ones."""
    with namespace_routing():
        instance = find_task_instance(instance_id)
        if instance is None or not instance.context:
            raise NotExportable(f"no run {instance_id} is recorded")
        context = instance.context
        metadata = context.get("metadata") or {}
        entry = (metadata.get("browser_recordings") or {}).get(template.FRAMES) or {}
        store = get_config().get_object_store()
        if entry.get("state") == "ended" and "artifact" in entry:
            if entry.get("missing_frames") or not entry.get("log_uploaded") or entry.get("log_missing"):
                raise NotExportable(f"run {instance_id}'s frames recording is incomplete")
            universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
            urls = {name: Artifact.get(file_id).object_url for name, file_id in universe.file_artifact_ids.items()}
        elif entry.get("state") == "open" and entry.get("namespace") and entry.get("recording_id"):
            urls = _left_open(store, entry, instance_id)
        else:
            raise NotExportable(f"run {instance_id}'s frames recording didn't end: {entry.get('error')}")
        manifest = json.loads(store.get(urls["manifest.json"]))
        log = [json.loads(line) for line in store.get(urls["log.jsonl"]).decode().splitlines() if line.strip()]
    answered = next((r for r in context.get("prompt_responses") or [] if r.get("prompt_id") == template.PROMPT_ID),
                    {})
    record = (metadata.get("script_results") or {}).get(template.SETUP_ID)
    if not record:
        raise NotExportable(f"run {instance_id} has no {template.SETUP_ID} record")

    def frame(path: str) -> bytes:
        with namespace_routing():
            return store.get(urls[path.replace("/", "-")])

    return Run(instance_id=instance_id, record=record, answer=(answered.get("response") or "").strip() or None,
               manifest=manifest, log=log, frame=frame, model=answered.get("model"),
               agent_error=answered.get("error_code"))


def _left_open(store, entry: dict, instance_id: str) -> dict[str, str]:
    """The files of a frames recording its run left open, as a run whose agent failed leaves it, which teardown ended
    in the browser: read from its namespace, as the browser wrote them, when its manifest says it ended complete."""
    base = f"{entry['namespace'].rstrip('/')}/{entry['recording_id']}"
    try:
        manifest = json.loads(store.get(f"{base}/manifest.json"))
    except Exception as e:  # noqa: BLE001 - any store's not-found
        raise NotExportable(f"run {instance_id}'s frames recording didn't end: {type(e).__name__}") from None
    if not manifest.get("ended_by") or manifest.get("error"):
        raise NotExportable(f"run {instance_id}'s frames recording didn't end: {manifest.get('error')}")
    if not manifest.get("log_uploaded") or manifest.get("log_missing"):
        raise NotExportable(f"run {instance_id}'s frames recording is incomplete")
    urls = {"manifest.json": f"{base}/manifest.json", "log.jsonl": f"{base}/{manifest.get('log') or 'log.jsonl'}"}
    return urls | {frame["path"].replace("/", "-"): f"{base}/{frame['path']}" for frame in manifest["frames"]
                   if frame.get("path")}


def _unseen(entries: dict[int, dict]) -> set[int]:
    """The CDP commands of tabs no one saw, which aren't the task's steps: a tab opened and closed with nothing done in
    it, as browser-use opens one for itself and closes it when it drives the browser's own, and a tab command on a tab
    already gone, which changed nothing, as its stop closing that tab again."""
    unseen, opened = set(), {}
    for seq in sorted(entries):
        entry = entries[seq]
        method, tab = entry.get("method") or "", entry.get("tab")
        if method.startswith("Target.") and entry.get("ok") is False and "No target" in (entry.get("error") or ""):
            unseen.add(seq)
        elif method == "Target.createTarget" and entry.get("ok", True) is not False:
            opened[tab] = seq
        elif method == "Target.closeTarget" and tab in opened:
            unseen |= {opened.pop(tab), seq}
        elif entry.get("acts") and tab in opened:
            del opened[tab]
    return unseen


def _covered(frame: dict, entries: dict[int, dict], unseen: set[int] = frozenset()) -> set[tuple[str, int]]:
    """What the frame follows: the agent's ``browser_*`` calls, by the seq they began at, and its own CDP commands,
    but those of tabs no one saw."""
    keys = set()
    for seq in frame.get("after") or []:
        entry = entries.get(seq)
        if entry is None or seq in unseen:
            continue
        if entry.get("via") == "mcp":
            keys.add(("mcp", seq))
        elif "for" in entry:
            keys.update(("mcp", call) for call in (entry["for"] if isinstance(entry["for"], list) else [entry["for"]]))
        else:
            keys.add(("cdp", seq))
    return keys


def _seq(key: tuple[str, int]) -> int:
    return key[1]


def _item(key: tuple[str, int], entries: dict[int, dict]) -> _Item | None:
    entry = entries.get(key[1]) or {}
    item = _from_call(entry) if key[0] == "mcp" else _from_command(entry)
    if item is not None:
        item.ok = entry.get("ok", True) is not False
        item.unfinished = bool(entry.get("unfinished"))
    return item


def _target(args: dict) -> str:
    return f"[aria-ref='{args['target']}']" if args.get("target") else "page"


def _named(args: dict) -> str:
    return args.get("element") or "the element"


def _from_call(entry: dict) -> _Item:
    """A ``browser_*`` call, as the log names it with its arguments."""
    tool, args = entry.get("tool") or "", entry.get("args") or {}
    if tool in _READS:
        return _Item(WAIT, "page", _READS[tool])
    if tool == "browser_navigate":
        return _Item("NAVIGATE", "page", f"go to {args.get('url', 'a page')}")
    if tool == "browser_navigate_back":
        return _Item("GO_BACK", "page", "go back")
    if tool == "browser_click":
        how = "double-click" if args.get("doubleClick") else f"{args['button']}-click" if args.get("button") in (
            "right", "middle") else "click"
        return _Item("CLICK", _target(args), f"{how} {_named(args)}")
    if tool == "browser_hover":
        return _Item("HOVER", _target(args), f"hover over {_named(args)}")
    if tool == "browser_type":
        text = f" {args['text']!r}" if "text" in args else ""
        return _Item("TYPE", _target(args), f"type{text} into {_named(args)}" + (
            ", then press Enter" if args.get("submit") else ""))
    if tool == "browser_fill_form":
        fields = args.get("fields") or []
        return _Item("TYPE", "page", f"fill {len(fields) or 'the'} form field{'s' if len(fields) != 1 else ''}")
    if tool == "browser_select_option":
        values = ", ".join(map(repr, args.get("values") or [])) or "an option"
        return _Item("SELECT", _target(args), f"select {values} in {_named(args)}")
    if tool == "browser_press_key":
        return _Item("PRESS_KEY", "page", f"press {args.get('key', 'a key')}")
    if tool in ("browser_mouse_click_xy", "browser_mouse_move_xy"):
        verb = "CLICK" if tool == "browser_mouse_click_xy" else "HOVER"
        return _Item(verb, f"coords({args.get('x')}, {args.get('y')})", "click" if verb == "CLICK" else "move the pointer")
    if tool == "browser_mouse_wheel":
        return _scroll(args.get("deltaY", 0))
    if tool in ("browser_mouse_drag_xy", "browser_drag", "browser_drop", "browser_mouse_down", "browser_mouse_up"):
        return _Item("CLICK", "page", {"browser_mouse_down": "press the mouse button",
                                       "browser_mouse_up": "release the mouse button"}.get(tool, "drag and drop"))
    if tool == "browser_file_upload":
        return _Item("CLICK", "page", f"choose {len(args.get('paths') or [])} files to upload")
    if tool == "browser_handle_dialog":
        return _Item("CLICK", "page", "accept the dialog" if args.get("accept") else "dismiss the dialog")
    if tool == "browser_tabs":
        action = args.get("action")
        if action in (None, "list"):
            return _Item(WAIT, "page", "list the tabs")
        return _Item("NAVIGATE", "page", {"new": "open a new tab", "close": "close a tab"}.get(
            action, f"switch to tab {args.get('index')}"))
    if tool == "browser_wait_for":
        return _Item(WAIT, "page", "wait for the page")
    if tool == "browser_close":
        return _Item(WAIT, "page", "close the page")
    return _Item(WAIT, "page", f"call {tool}")


def _from_command(entry: dict) -> _Item | None:
    """A CDP command of the agent's own; None for the part of an input event that is no action of its own, a release
    or a key's text."""
    method, params = entry.get("method") or "", entry.get("params") or {}
    if method == "Page.navigate":
        return _Item("NAVIGATE", "page", f"go to {params.get('url', 'a page')}")
    if method == "Page.reload":
        return _Item("REFRESH", "page", "reload the page")
    if method == "Page.navigateToHistoryEntry":
        return _Item("GO_BACK", "page", "go to another page in the tab's history")
    if method == "Target.createTarget":
        url = params.get("url")
        return _Item("NAVIGATE", "page", "open a new tab" + (f" at {url}" if url and url != "about:blank" else ""))
    if method in ("Target.activateTarget", "Page.bringToFront"):
        return _Item("NAVIGATE", "page", "switch to another tab")
    if method == "Target.closeTarget":
        return _Item("NAVIGATE", "page", "close the tab")
    if method == "Input.dispatchMouseEvent":
        kind, at = params.get("type"), f"coords({_int(params.get('x'))}, {_int(params.get('y'))})"
        if kind == "mouseWheel":
            return _scroll(params.get("deltaY", 0))
        if kind == "mousePressed":
            how = "double-click" if (params.get("clickCount") or 1) > 1 else f"{params['button']}-click" if params.get(
                "button") in ("right", "middle") else "click"
            return _Item("CLICK", at, how)
        return _Item("HOVER", at, "move the pointer") if kind == "mouseMoved" else None
    if method == "Input.dispatchTouchEvent":
        return _Item("CLICK", "page", "tap the page")
    if method == "Input.insertText":
        return _Item("TYPE", "page", f"type {params.get('text', '')!r}")
    if method == "Input.dispatchKeyEvent":
        key, kind = params.get("key") or "", params.get("type")
        if kind not in ("keyDown", "rawKeyDown"):
            return None
        if params.get("text") and len(key) == 1:
            return _Item("TYPE", "page", f"type {params['text']!r}")
        return _Item("PRESS_KEY", "page", f"press {key or 'a key'}")
    if method in ("Runtime.evaluate", "Runtime.callFunctionOn"):
        return _Item(WAIT, "page", "run a script on the page")
    return _Item(WAIT, "page", f"send {method}")


def _scroll(delta: Any) -> _Item:
    delta = _int(delta)
    return _Item("SCROLL", "page", f"scroll {'down' if delta >= 0 else 'up'} by {abs(delta)}px")


def _int(value: Any) -> int:
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return 0


def _merged(items: list[_Item]) -> list[_Item]:
    """The burst's items as one reads them: a pointer move into the click at the same place, keys typed one after
    another into one text, and repeats into the first, failing when any of them failed."""
    merged: list[_Item] = []
    for item in items:
        previous = merged[-1] if merged else None
        if previous and (previous.verb, item.verb) == ("HOVER", "CLICK") and previous.target == item.target:
            merged[-1] = item
        elif previous and _typed(previous) and _typed(item):
            previous.description, previous.ok = previous.description[:-1] + item.description[6:], previous.ok and item.ok
        elif same := next((m for m in merged if (m.verb, m.target, m.description) ==
                           (item.verb, item.target, item.description)), None):
            same.ok, same.unfinished = same.ok and item.ok, same.unfinished and item.unfinished
        else:
            merged.append(item)
    return merged or [_Item(WAIT, "page", "wait")]


def _typed(item: _Item) -> bool:
    """Whether ``item`` is text a CDP agent typed on the page, ``type '...'``, which the next keys extend."""
    return (item.verb, item.target) == ("TYPE", "page") and item.description.startswith("type '") and \
        item.description.endswith("'")


def _action(items: list[_Item]) -> tuple[str, str | None]:
    """The step's action, its first action's verb and target and what it and the rest did, an action still going when
    the run ended said to be, and its status, from its finished actions: none for a step that only observes or whose
    actions none finished, else FAILED when any of them failed."""
    acting = [item for item in items if item.verb != WAIT]
    primary = (acting or items)[0]
    described = [f"{item.description} (unfinished at the end)" if item.unfinished else item.description
                 for item in [primary, *(item for item in items if item is not primary)]]
    head = f"{primary.target} -> {primary.verb}" if primary.verb in PAGE_VERBS else f"{primary.verb} {primary.target}"
    finished = [item for item in acting if not item.unfinished]
    if not finished:
        return f"{head} -> {'; then '.join(described)}", None
    status = "SUCCESS" if all(item.ok for item in finished) else "FAILED"
    return f"{head} -> {'; then '.join(described)} | {status}", status


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder each run is written into, one folder per task")
    parser.add_argument("instance_ids", nargs="+", metavar="INSTANCE_ID", help="a run's instance id")
    args = parser.parse_args(argv)
    failed = 0
    for instance_id in args.instance_ids:
        try:
            print(f"{instance_id}: {write(load(instance_id), args.out)}")
        except NotExportable as e:
            failed += 1
            print(f"{instance_id}: not exported: {e}", file=sys.stderr)
    if failed:
        raise SystemExit(f"{failed} of {len(args.instance_ids)} runs weren't exported")


if __name__ == "__main__":
    main(sys.argv[1:])
