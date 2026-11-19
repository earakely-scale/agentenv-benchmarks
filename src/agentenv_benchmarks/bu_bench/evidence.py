"""Judge evidence from an AgentEnv trajectory: the step strings and token usage the findings judge and the cost report
read, for the two trajectory formats this bundle's agents write.

- ``codex-exec-json/1`` (Codex): ``codex exec --json`` events. Steps are tool calls with their output and the
  agent's messages, as the bake-off harness built them from the Codex rollout; reasoning is left out.
- ``browser-use-messages/1`` (browser-use 1.0): Pydantic AI messages. Steps are each tool call with its result and
  the model's text, as the browser-use eval adapter built them; thinking is kept with a ``thinking:`` prefix, which
  the judge drops.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import re
from collections.abc import Callable
from typing import Any, BinaryIO

BASE64_BLOB = re.compile(r'"[A-Za-z0-9+/=\\]{400,}"')
TOOL_OUTPUT_CHARS = 20_000
PRICES = {"input": 1.0, "cached": 0.10, "output": 5.0}  # $/1M tokens: the bake-off's cost basis
#: Event types that are no step of the agent's: Codex's thread and turn events, AgentEnv's usage record.
CONTROL_EVENTS = ("thread.", "turn.", "agentenv.")
#: Steps the frozen findings judge drops before it reads the trajectory (upstream/findings_judge.py).
JUDGE_HIDDEN = ("thinking:", "reasoning:")


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"...[at least {len(text) - limit} characters omitted from the middle]..."
    budget = limit - len(marker)
    head = budget // 3
    return text[:head] + marker + text[-(budget - head):]


def text_of(value: Any) -> str:
    """Text a model saw from one tool result; images are marked, never dropped silently or inlined."""
    if isinstance(value, str):
        return BASE64_BLOB.sub('"[image shown to the agent]"', value)
    if isinstance(value, dict):
        if value.get("type") in {"image", "input_image"} or value.get("kind") == "binary" or "image_url" in value:
            return "[image shown to the agent]"
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "content" in value:
            return text_of(value["content"])
        return text_of(json.dumps(value, default=str))
    if isinstance(value, list):
        return "\n".join(text_of(part) for part in value)
    return text_of(json.dumps(value, default=str))


def events(trajectory: Any) -> list[dict]:
    """The event list inside a stored trajectory, whichever envelope wraps it."""
    if isinstance(trajectory, list):
        return [event for event in trajectory if isinstance(event, dict)]
    if isinstance(trajectory, dict):
        for key in ("events", "trajectory", "entries", "messages", "items"):
            if key in trajectory:
                return events(trajectory[key])
    return []


def _usage() -> dict[str, int]:
    return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "model_requests": 0}


def codex_steps(items: list[dict]) -> tuple[list[str], dict[str, int]]:
    steps: list[str] = []
    summed, reported = _usage(), None
    for event in items:
        kind = event.get("type")
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                summed[key] += int(event["usage"].get(key) or 0)
        elif kind == "agentenv.usage" and isinstance(event.get("usage"), dict):
            reported = event  # the agent's own total, which also covers a turn the CLI was stopped in
        if kind != "item.completed" or not isinstance(event.get("item"), dict):
            continue
        item = event["item"]
        item_type = item.get("type")
        if item_type == "command_execution":
            output = text_of(item.get("aggregated_output") or "")
            steps.append(f"$ {item.get('command')}\n{clip(output, TOOL_OUTPUT_CHARS)}")
        elif item_type == "mcp_tool_call":
            result = item.get("result") if item.get("result") is not None else item.get("error")
            steps.append(f"$ {item.get('server')}.{item.get('tool')} {json.dumps(item.get('arguments'))}\n"
                         f"{clip(text_of(result), TOOL_OUTPUT_CHARS)}")
        elif item_type == "agent_message" and item.get("text"):
            steps.append(str(item["text"]))
        elif item_type == "file_change":
            steps.append(f"$ apply_patch {json.dumps(item.get('changes'))[:2000]}")
        elif item_type == "web_search":
            steps.append(f"$ web_search {json.dumps(item.get('query'))}")
        elif item_type == "error" and item.get("message"):
            steps.append(f"error: {item['message']}")
    usage = summed
    if reported is not None:
        usage = _usage()
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            usage[key] = int((reported.get("usage") or {}).get(key) or 0)
        usage["model_requests"] = int(reported.get("model_requests") or 0)
    return steps, usage


def browser_use_steps(messages: list[dict]) -> tuple[list[str], dict[str, int]]:
    steps: list[str] = []
    usage = _usage()
    calls: dict[str, tuple[str, Any]] = {}
    for message in messages:
        if message.get("kind") == "response":
            usage["model_requests"] += 1
            counts = message.get("usage") or {}
            usage["input_tokens"] += int(counts.get("input_tokens") or 0)
            usage["cached_input_tokens"] += int(counts.get("cache_read_tokens") or 0)
            usage["output_tokens"] += int(counts.get("output_tokens") or 0)
        for part in message.get("parts") or []:
            kind = part.get("part_kind")
            if kind == "thinking" and part.get("content"):
                steps.append(f"thinking: {clip(str(part['content']), 4000)}")
            elif kind == "text" and message.get("kind") == "response" and part.get("content"):
                steps.append(f"text: {clip(str(part['content']), 4000)}")
            elif kind == "tool-call":
                args = part.get("args")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                calls[str(part.get("tool_call_id"))] = (str(part.get("tool_name")), args)
            elif kind in {"tool-return", "retry-prompt"}:
                name, args = calls.pop(str(part.get("tool_call_id")), (str(part.get("tool_name")), None))
                result = text_of(part.get("content"))
                prefix = "error: " if kind == "retry-prompt" else ""
                steps.append(f"{name}: {clip(json.dumps(args, default=str), 2000)}\n"
                             f"result: {prefix}{clip(result, TOOL_OUTPUT_CHARS)}")
    for name, args in calls.values():  # a call the run ended before answering
        steps.append(f"{name}: {clip(json.dumps(args, default=str), 2000)}\nresult: (none: the run ended)")
    return steps, usage


def steps_and_usage(trajectory: Any) -> tuple[str, list[str], dict[str, int]]:
    """(format, steps, usage) for a stored trajectory: the format its envelope declares, else the one its events show.
    Control events (Codex's thread and turn events, AgentEnv's usage record) are never steps."""
    items = events(trajectory)
    declared = trajectory.get("format") if isinstance(trajectory, dict) else None
    if declared == "browser-use-messages/1" or declared is None and any(
            "parts" in item and item.get("kind") in {"request", "response"} for item in items):
        return ("browser-use-messages/1", *browser_use_steps(items))
    if declared == "codex-exec-json/1" or declared is None and any(
            str(item.get("type", "")).startswith(("item.", *CONTROL_EVENTS)) for item in items):
        return ("codex-exec-json/1", *codex_steps(items))
    return (str(declared or "unknown"), [clip(text_of(item), TOOL_OUTPUT_CHARS) for item in items
                                         if not str(item.get("type", "")).startswith(CONTROL_EVENTS)], _usage())


def judge_visible(steps: list[str]) -> list[str]:
    """The steps the findings judge reads: it drops each one that starts with a JUDGE_HIDDEN prefix."""
    return [step for step in steps if not step.lstrip().startswith(JUDGE_HIDDEN)]


def model_usd(usage: dict[str, int]) -> float:
    cached = usage.get("cached_input_tokens", 0)
    uncached = max(0, usage.get("input_tokens", 0) - cached)
    output = usage.get("output_tokens", 0)
    return (uncached * PRICES["input"] + cached * PRICES["cached"] + output * PRICES["output"]) / 1e6


def output_files(files: list[tuple[str, Callable[[], BinaryIO]]]) -> list[dict[str, Any]]:
    """Delivered files as the findings judge reads them: the first 50 by name, text when UTF-8, split evenly under its
    600 kB cap. Only those 50 are opened, each streamed, holding no more than its share of the cap whatever its size."""
    files = sorted(files, key=lambda file: file[0])[:50]
    limit = 600_000 // max(1, len(files))
    out = []
    for name, opened in files:
        with contextlib.closing(opened()) as stream:
            out.append({"name": name, **_capped_text(stream, limit)})
    return out


def _capped_text(stream: BinaryIO, limit: int) -> dict[str, Any]:
    """The stream's size, and its UTF-8 text clipped to its first third and last two thirds of ``limit`` characters
    (None when it isn't UTF-8), read 1 MB at a time."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    size, chars, head, tail, text_ok = 0, 0, "", "", True
    while True:
        chunk = stream.read(1 << 20)
        size += len(chunk)
        if text_ok:
            try:
                piece = decoder.decode(chunk, final=not chunk)
            except UnicodeDecodeError:
                text_ok = False
            else:
                chars += len(piece)
                head += piece[:limit + 1 - len(head)]
                tail = (tail + piece)[-(limit - limit // 3):]
        if not chunk:
            break
    if not text_ok:
        return {"size": size, "text": None, "clipped": False}
    if chars <= limit:
        return {"size": size, "text": head, "clipped": False}
    return {"size": size, "text": head[: limit // 3] + "\n...[middle omitted]...\n" + tail, "clipped": True}
