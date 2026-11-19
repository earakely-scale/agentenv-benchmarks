A Browser Use Cloud browser as an env a trainer's model drives through Playwright MCP's tools, for RL on AgentEnv.

# bu-browser

Not BU Bench. BU Bench's 200 tasks (`bu-bench`) are an evaluation and must stay out of training. This bundle is the
browser env for RL tasks of your own, plus one example task.

## What it is

`bu-browser` (env type `bu_browser`, from `agentenv_benchmarks.bu_browser`) is a multi env with one website and no
browser of the run's own:

- `bu-browser-site` (`envs/bu-browser-site/browser_site.py`) starts one Browser Use Cloud browser at the first tool
  call, attaches Playwright MCP 0.0.85 to it over CDP, and serves Playwright MCP's tools as its own MCP tools. The
  gateway serves them to the agent like any MCP env's tools. The site is ready once its key is set; a bad key or an
  account at its concurrency limit shows up as the first tool call's error.
- If Browser Use ends the browser mid-episode (its `timeout_minutes`, a crash), the next failing call says so and the
  call after it starts a fresh browser. A call fails that way when the browser's CDP discovery answers 404 or 410 or
  refuses the connection twice. On any other answer (a timeout, a 5xx) the site asks Browser Use's API for the
  browser's status and treats any status but `active` as ended (no status counts as running). Only when that also
  fails is the browser left running, and the call returns the tool's own error. If Playwright MCP dies, the next call starts a fresh browser and its
  result says the earlier pages are gone.
- `browser_evaluate` is hidden until a task's tool access enables it. `browser_run_code_unsafe` is never served,
  because it would run the agent's code in the site's container, and neither is `browser_close`, which can end the
  Browser Use browser.
- An action's result (navigate, click, type...) carries the page's snapshot inline: Playwright MCP writes it to a file
  and answers with a link, which a model with no file tool can't open, so the site puts the file's text in its place.
- Text in tool results is scrubbed of the browser's CDP address, which is itself a credential.

In an OpenEnv episode (`agent_env.openenv.build_app`), the task's `deploy_agent` becomes a proxy, and the model gets
the tools of the envs in its `env_ids` plus `end_turn`. A task that lists `bu-browser` there gets this browser.

## Key

The site reads its key from `[settings] api_key = "secret:BROWSER_USE_API_KEY"`. AgentEnv resolves it from the
configured secret store (`[stores.secret]`: local, AWS or GCP) only when it writes the site's container. Only the site
process has it. Playwright MCP runs as another user with an allowlisted environment.

```bash
export BROWSER_USE_API_KEY=...   # the local secret store reads the environment; a cloud store holds it by this name
agent-env run bu-browser --eval example
```

Each deploy logs `env_deploy_tools_probe ... status=missing ... missing=bu-browser-site` at WARNING: the site's card
lists `browser_evaluate`, which the gateway hides until a task enables it. That is expected.

## Serve for RL

```bash
pip install "agentenv-framework[openenv]" git+https://github.com/earakely-scale/agentenv-benchmarks
export BROWSER_USE_API_KEY=... OPENENV_TOKEN=...
```

```python
import os

import uvicorn
from agent_env.openenv import build_app

uvicorn.run(build_app("bu-browser", token=os.environ["OPENENV_TOKEN"]), host="0.0.0.0", port=8000)
```

Clients send `Authorization: Bearer $OPENENV_TOKEN`. Without a token, anyone who reaches the server starts Browser
Use browsers on your key.

## Cost and cleanup

One episode uses one browser. The browser is stopped when:

- the run tears down (`bu_browser` closes by calling the site's `close` extension, which stops the browser for good:
  a tool call that arrives after it starts none), which covers the final `end_turn`, an OpenEnv reset, an abandoned
  or timed-out episode, a closed session and server shutdown;
- a `reset_env` step runs (`data/reset`: the next tool call starts a fresh browser);
- the site's server gets SIGTERM while no tool call is in flight (uvicorn runs the shutdown that stops the browser
  only after open requests finish, which a stop grace of a few seconds may not wait for).

A failed stop request is retried twice. Each request waits at most 20 s, and the run's teardown waits up to 115 s for
the stop. A browser whose three stop requests all failed is stopped again by the next reset or close. A browser still being created at teardown is waited for and then stopped; if its create and stop together
take longer than that, it runs to its timeout. If none of these happens, Browser Use ends the browser after
`timeout_minutes` (30 by default), counted from the first tool call. That is the case when:

- the server process or its container is killed outright;
- the run's teardown skips `close()`: a retried `deploy_env` span's earlier deployment, and an env on a provider that
  is not one of agent-env's built-in env providers, are not closed by it.

A trainer script that deploys one env object again before closing it gets an error: AgentEnv would end the earlier
deployment's sandbox without its `close`, leaving its browser to its timeout. Close it first, or deploy a new env
object.

Keep `timeout_minutes` at or above the task's turn timeouts.
`build_app` serves up to 8 episodes at once by default (`max_concurrent_envs`), so that is up to 8 browsers. Check
your Browser Use account's concurrency limit before you scale.

The site's container must reach `api.browser-use.com` and `*.cdp.browser-use.com`.

Rollback: end every running bu-browser episode before you revert or uninstall this package. Teardown stops a
deployment's browser through the `bu_browser` env type this package registers; a deployment torn down without it is
not closed through the site, and its browser runs to `timeout_minutes`.

## Example task

`pypi-browser-use-wheel-sha256` asks for the SHA256 of `browser_use-1.0.3-py3-none-any.whl`, as shown on PyPI. That
file is immutable and was uploaded on 2026-11-12, so the answer comes from the page, not from memory. A regex
verifier scores the reply 1 only when the whole reply is the hash, else 0, and that score is the episode's reward. It runs on the live pypi.org, so it is a
smoke test, not a training set.

## Tests

`tests/test_bu_browser.py` runs in CI with a fake Browser Use API and a fake Playwright MCP, including agent-env's
teardown closing a deployed record of this env. `uv run pytest -m docker tests/test_bu_browser_e2e.py` (opt in, a few
minutes, Docker) plays OpenEnv episodes on the real site image with a local Chromium and a fake Browser Use API.
