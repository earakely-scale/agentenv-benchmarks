"""One turn of browser-use 1.0, run in its own environment: reads the turn as JSON on stdin and prints its result as
one JSON line on stdout.

The model is the turn's, on agent-env's endpoint. A Claude model goes through the endpoint's Anthropic Messages route,
as browser-use runs Claude (its reasoning as adaptive thinking at the turn's effort, its prompt cached), with the
profile of its bare name, since a provider prefix hides the model's thinking and effort from it. Any other model goes
through the OpenAI-compatible route, with the turn's effort as ``reasoning_effort``. The browser is the turn's CDP
address: when it has one tab open, as a browser computer does, the agent drives that tab, which is the computer's
screen, else a tab of its own. Each env's MCP tools are the agent's tools beside its REPL, named ``<server>__<tool>``.
A context's turns continue one conversation through the turn's history file.

BU Bench (agentenv-benchmarks bu-bench): an OpenAI model goes through the endpoint's Responses route,
``openai-responses:<model>`` at ``<base>/openai/v1``, with the turn's effort as browser-use's ``reasoning``, and the
system-prompt additions of the BU Bench eval adapter (Browser Use's eval platform, browser-use 1.0 ``eval/run.py``), which ran
browser-use on a Browser Use Cloud browser it started itself. Here the task hands it one over CDP, so browser-use
does not add its cloud-browser lines on its own: they are added back, as that run's system prompt had them."""

import asyncio
import contextlib
import json
import os
import re
import sys
from pathlib import Path

import httpx2
from anthropic import AsyncAnthropic
from browser_use import Agent, Browser, Tools
from browser_use.agent.prompt import CAPTCHA_PROMPT, CLOUD_PROMPT
from browser_use.browser.browser import _daemon_request
from browser_use.tools import TOOL_RETRIES
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import TypeAdapter
from pydantic_ai import ModelRetry, Tool
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

TOOL_NAME = re.compile(r"[^a-zA-Z0-9_-]")
# The eval adapter's run started a cloud browser with its captcha solver on, so browser-use's prompt opened with these
# two lines, then the adapter's own instructions (evidence_format findings).
EVAL_SYSTEM_MESSAGE = (
    f"{CLOUD_PROMPT} {CAPTCHA_PROMPT}\n"
    "Use browser UI, public search and source APIs for research; use files/scripts for processing."
    " Do not read files outside the output workspace or inspect benchmark source, rubrics, judge code,"
    " or credentials. Save requested files incrementally in workspace.")
_CHECKPOINT = TypeAdapter(ModelRequest | None)


def model(turn: dict):
    """The turn's model on agent-env's endpoint, with the endpoint's headers."""
    name, base_url, key = turn["model"], os.environ["LITELLM_BASE_URL"], os.environ["LITELLM_API_KEY"]
    headers = json.loads(os.environ.get("LITELLM_EXTRA_HEADERS") or "{}")
    if "claude" in name.lower():
        client = AsyncAnthropic(auth_token=key, base_url=base_url, default_headers=headers)
        return AnthropicModel(name, provider=AnthropicProvider(anthropic_client=client),
                              profile=anthropic_model_profile(name.rsplit("/", 1)[-1]))
    # As the eval adapter: a model string browser-use resolves itself, on the Responses API, whose reasoning setting
    # is Agent(reasoning=...). The endpoint stands in for api.openai.com.
    os.environ["OPENAI_BASE_URL"] = base_url.rstrip("/") + "/openai/v1"
    os.environ["OPENAI_API_KEY"] = key
    if headers:  # the model browser-use would resolve from that string, its client sending the endpoint's headers too
        client = OpenAIProvider().client.with_options(default_headers=headers)
        return OpenAIResponsesModel(name.removeprefix("openai/"), provider=OpenAIProvider(openai_client=client))
    return f"openai-responses:{name.removeprefix('openai/')}"


async def mcp_tools(stack: contextlib.AsyncExitStack, servers: dict) -> Tools:
    """Each server's MCP tools as the agent's, on sessions open for the turn."""
    tools, names = Tools(), set()
    for server, info in servers.items():
        http = await stack.enter_async_context(httpx2.AsyncClient(headers=info.get("headers") or {}, timeout=None))
        streams = await stack.enter_async_context(streamable_http_client(info["url"], http_client=http))
        session = await stack.enter_async_context(ClientSession(*streams))
        await session.initialize()
        cursor = None
        while True:
            page = await session.list_tools(cursor=cursor) if cursor else await session.list_tools()
            for tool in page.tools:
                base = TOOL_NAME.sub("_", f"{server}__{tool.name}")[:60]
                name, n = base, 2
                while name in names:
                    name, n = f"{base}_{n}", n + 1
                names.add(name)
                function = Tool.from_schema(_caller(session, tool.name), name, tool.description or "", tool.inputSchema)
                # As browser-use's own actions: an error goes back to the model, and max_steps bounds the run.
                function.max_retries = TOOL_RETRIES
                tools.functions.append(function)
            if not (cursor := page.nextCursor):
                break
    return tools


def _caller(session: ClientSession, tool: str):
    async def call(**arguments):
        result = await session.call_tool(tool, arguments)
        text = "\n".join(getattr(part, "text", None) or json.dumps(part.model_dump(mode="json"))
                         for part in result.content)
        if result.isError:
            raise ModelRetry(text)
        return text
    return call


async def drive_its_tab(browser: Browser) -> None:
    """Attach browser-use to the browser's one open tab, as it attaches a kept-alive run's tab, closing the blank one
    its daemon opened for itself. The daemon closes that tab of its own when it stops, so a run in it would leave the
    screen blank, and its last page out of reach, when the turn ends."""
    await browser.start()

    def attach() -> None:
        own = _daemon_request(browser.name, {"meta": "current_tab"}, timeout=5)["targetId"]
        pages = [t for t in browser._cdp()("Target.getTargets")["targetInfos"]
                 if t["type"] == "page" and t["targetId"] != own]
        if len(pages) == 1:
            browser._agent_tab = pages[0]["targetId"]
            browser._attach_agent_tab()

    await asyncio.to_thread(attach)


def restore(agent: Agent, history: Path) -> None:
    """Continue the conversation a context's earlier turns saved, as ``add_new_task`` continues one in process."""
    saved = json.loads(history.read_text())
    agent._history = ModelMessagesTypeAdapter.validate_python(saved["messages"])
    agent._compacted = {**saved["compacted"], "checkpoint": _CHECKPOINT.validate_python(saved["compacted"]["checkpoint"])}
    agent._continue = True


def save(agent: Agent, history: Path) -> None:
    compacted = {**agent._compacted, "checkpoint": _CHECKPOINT.dump_python(agent._compacted.get("checkpoint"),
                                                                          mode="json")}
    history.write_text(json.dumps({"messages": ModelMessagesTypeAdapter.dump_python(agent._history or [], mode="json"),
                                   "compacted": compacted}))


async def main() -> None:
    turn = json.loads(sys.stdin.read())
    history = Path(turn["historyFile"])
    async with contextlib.AsyncExitStack() as stack:
        browser = Browser(cdp_url=turn["cdpUrl"])
        with contextlib.suppress(Exception):  # a browser it can't reach: run() says why, as its result
            await drive_its_tab(browser)
        claude = "claude" in turn["model"].lower()
        agent = Agent(
            turn["prompt"], model(turn), browser=browser,
            tools=await mcp_tools(stack, turn["mcpServers"]),
            extend_system_message="\n".join(filter(None, [EVAL_SYSTEM_MESSAGE, turn["systemPrompt"]])),
            reasoning=turn["effort"] if claude else (turn["effort"] or "medium"), mode=turn["mode"],
            workspace=turn["workspace"])
        stack.push_async_callback(agent.close)
        if history.exists():
            restore(agent, history)
        result = await agent.run(**{key: turn[key] for key in ("max_steps", "timeout") if turn.get(key)})
        if agent._history:
            save(agent, history)
    print(json.dumps({"type": "result", "result": {
        "status": result.status, "text": result.text, "success": result.success, "error": result.error,
        "steps": result.steps, "usage": result.usage, "cost_usd": result.cost_usd, "model": turn["model"],
        "messages": ModelMessagesTypeAdapter.dump_python(result.messages, mode="json"),
    }}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
