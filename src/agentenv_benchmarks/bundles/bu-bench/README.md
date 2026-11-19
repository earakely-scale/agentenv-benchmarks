BU Bench V2.1's 200 hard live-web tasks, each on a fresh Browser Use Cloud browser, graded by its frozen findings judge, with the 8 harnesses of Browser Use's harness bake-off.

# BU Bench V2.1

The tasks are [BU Bench V2.1](https://github.com/browser-use/benchmark)'s 200, most of them from
[Odysseys2](https://odysseysbench.com) (Carnegie Mellon University and Browser Use) plus some of Browser Use's own.
They ship encrypted: each task file here names a task id only, and `bu_cloud_browser` decrypts its instruction at run
time from the benchmark's `.enc`. Please don't publish decrypted tasks or rubrics, or train on them. A test fails if
any file of this bundle holds task, rubric or canary text. A run's records hold the decrypted instruction too (the
context's seed value `bu_task`, the stored prompt, the agents' trajectories in agent-env's stores): don't publish them
either.

## Run

Install this repository as a plugin first (it is not on PyPI, so a bare `agent-env plugin add agentenv-benchmarks` can't
work): `agent-env plugin add 'agentenv-benchmarks @ git+https://github.com/earakely-scale/agentenv-benchmarks'`, or
`uv run` from a checkout.

Needs a local Docker daemon (or an agent-env sandbox that builds Dockerfiles): the first run builds the agents'
images, pulling `ghcr.io/scaleapi/agentenv-codex` and the npm and PyPI tools (top-level versions pinned). A run stops before any agent or
browser starts when the Browser Use key or the judge key is missing. Keys are checked for presence only, not validity:
run `smoke-2` first. `--judge-model` is ignored: the judge is frozen at gpt-5.6-luna, as the published scores were
graded.

```bash
export BROWSER_USE_API_KEY=...       # cloud.browser-use.com: one fresh cloud browser per task
export LITELLM_BASE_URL=... LITELLM_API_KEY=...       # the agents' model endpoint
export BU_BENCH_JUDGE_API_KEY=...    # OpenAI key for the judge (gpt-5.6-luna); OPENAI_BASE_URL is honoured
agent-env run bu-bench --eval smoke-2 --model gpt-6.1-terra                              # 2 tasks, Codex + Browser Use CLI
agent-env run bu-bench --eval mini-20 --model gpt-6.1-terra --agent browser-use-104rc1
agent-env run bu-bench --eval bu-bench-v21 --model gpt-6.1-terra --agent codex-playwright-mcp   # all 200
```

Evals: `smoke-2` (bu2-140, bu2-149), `mini-20` (20 tasks spread over task length),
`bu-bench-v21` (all 200). Any agent that takes a browser over `cdp-config/v1` runs with `--agent`.

## What every task does

`deploy_agent`, then `bu_cloud_browser`, `prompt_agent` (the decrypted task), `collect_artifacts` (the files the agent
wrote under `/app/agent_outputs`) and `bu_findings_verifier`.

| Piece | What it does |
|---|---|
| `bu_cloud_browser` step | Creates a Browser Use Cloud browser (`POST /api/v3/browsers {proxyCountryCode: us, timeout: 70, enableRecording: true}`), posts its CDP WebSocket to the agent's `cdp-config/v1`, starts the env's own screenshot capture (2 s, deduplicated, at most 400 frames, the last kept for the final one, its own CDP connection), and decrypts the task instruction into the seed value `bu_task`. |
| `bu_findings_verifier` step | Takes a final screenshot, stops the browser (always, retrying a failed stop; plus an exit hook and the browser's 70 min timeout), stores the cloud recording MP4 as a file artifact, turns the agent's trajectory into judge steps, and runs the frozen judge (`agentenv_benchmarks/bu_bench/judge/findings-v21`, browser-use/benchmark@1e0e2f1e, gpt-5.6-luna xhigh) unchanged as a subprocess. Writes `verifications.findings` with the continuous score and one row per rubric item. The judge runs up to 3 times while it errs. A judge that still errs, a task whose browser this process no longer holds, a turn that left no response (it timed out, its agent died, or the prompt never reached it), a failed `collect_artifacts` (it keeps only the files it uploaded before it failed) or one that recorded no files, a trajectory that can be read neither from the store nor from the agent or that has no step the judge reads (it drops `thinking:` and `reasoning:` steps), and a delivered file that can't be read record no score and fail the step: never a zero. A failed turn that left a response is still judged on it. |
| Evidence | The judge's inputs and outputs (`task.json` with the decrypted task, `result.json`, screenshots, `judgment.json`, `report.json`, and the judge's own `judge-config.json`, `judge_trace.json` and `judge-recording/`, which hold the task's rubric and canary) go to `$BU_BENCH_EVIDENCE_DIR/<task>-<run group>-<browser handle>` and stay there. Without it they go to a temporary folder, deleted once the task is scored and kept when the judge errs (the error names the folder), so it can be rejudged. Keep the folder private and never upload it. Known AgentEnv limitation: when `collect_artifacts` can't list `/app/agent_outputs` (a permission or I/O error, not a missing folder), agentenv-framework 0.9.1573 logs `Failed to enumerate /app/agent_outputs: ...` at WARNING and records the step as a success with no files, the same record an agent that wrote nothing leaves, so the task is judged with no delivered files. Nothing in the run's context tells the two apart; that WARNING with an error other than `No such file or directory` is the only sign. `report.json`'s `model_usd` prices the agent's token usage at GPT-6.1 Terra's rates (evidence.py `PRICES`), whatever model ran, and counts only the model calls in the agent's own trajectory: the model calls Stagehand makes itself (`codex-stagehand`) are not in it. Rejudge with: `EVAL_WORKSPACE=$dir EVAL_JUDGMENT_PATH=$dir/judgment.json EVAL_TASK_PATH=$dir/task.json EVAL_RESULT_PATH=$dir/result.json EVAL_JUDGE_API_KEY=$BU_BENCH_JUDGE_API_KEY EVAL_JUDGE_OPTIONS_JSON='{"reasoning_effort":"xhigh"}' uv run python src/agentenv_benchmarks/bu_bench/judge/findings-v21/run.py`. |
| `agents/codex-*` | AgentEnv's published Codex agent + Codex CLI 0.166.0 + one browser tool (table below), all on `artifacts/codex-browser-tools`. Declares `cdp-config/v1`; each turn gets the bake-off prompt and its tool pointed at the task's browser. |
| `agents/browser-use-104rc1` | AgentEnv's browser-use agent, on browser-use 1.0.4rc1, with the BU Bench eval adapter's model route (Responses API, `reasoning=medium`) and system-prompt additions. |
| Tasks | Each task file names a task id only. The instruction is decrypted from the benchmark's `.enc` at run time. Rubrics are never sent to the agent or put in task files; the evidence folder holds the decrypted task, rubric and canary, and the stored verification names each rubric item by its id. Do not publish decrypted tasks or rubrics: a test fails if any file here carries task, rubric or canary text. |
| Credentials | A running cloud browser's CDP address reaches it with no other key, and its host is the browser id. Only the id's first 8 characters are logged or stored; the address is scrubbed from evidence, agent replies, errors and warnings, the Codex arms' bridge logs and both agents' trajectories. Files the agent writes are collected as it wrote them and may hold the address; it stops being a credential once the verifier stops the browser. |
| Cleanup | The verifier stops the browser; the exit hook stops any this `agent-env run` process started and did not stop. Any task that ends before its verifier runs (a persistent runner's cancel, or a fatal failure after the browser step) leaves its browser to the exit hook, which waits at most 20 s for its stops, or to Browser Use's 70 min limit. |

## Arms

The 8 entries of the harness bake-off: Codex CLI 0.166.0 with each of 7 browser tools, plus browser-use's own agent.
Same task steps, cloud browser, model and judge for all; `--agent` is the only change.

| `--agent` | Harness | Tool version (top-level pinned) | How the tool reaches the browser |
|---|---|---|---|
| `codex-bu-cli` (default) | Codex + Browser Use CLI | browser-harness @c2420943 | `BU_CDP_WS`; `browser-harness` on PATH |
| `codex-browser-use-mcp` | Codex + Browser Use MCP (`browser_exec`) | browser-use 1.0.3 `--cli-mcp` | MCP server env `BU_CDP_WS` |
| `codex-playwright-mcp` | Codex + Playwright MCP | @playwright/mcp 0.0.85 | `--cdp-endpoint` |
| `codex-chrome-devtools-mcp` | Codex + Chrome DevTools MCP | chrome-devtools-mcp 1.13.0 | `--wsEndpoint` |
| `codex-agent-browser` | Codex + agent-browser | agent-browser 0.42.0 | `agent-browser connect` + config `cdp` |
| `codex-stagehand` | Codex + Stagehand (small MCP server) | @browserbasehq/stagehand 3.7.3 | `localBrowserLaunchOptions.cdpUrl` |
| `codex-opencode-browse` | Codex + OpenCode browse | @opencode/browse 0.0.16 | relay + a shim for its extension |
| `browser-use-104rc1` | browser-use 1.0.4rc1 agent | browser-use 1.0.4rc1 | `cdp-config/v1` |

The 7 Codex agents are one image (`artifacts/codex-browser-tools`, every tool installed at the bake-off's top-level
pins, checked at build time; their transitive npm and PyPI dependencies are not locked) and differ only in their `agent.toml` (`BU_BENCH_TOOL`). Each turn sets up only its tool, as
the bake-off did, and sends the bake-off's prompt with that tool's rule line.

## Results

All 8 arms on all 200 tasks, run by `agent-env run` (agentenv-framework 0.9.1536) on GitHub Actions, Nov 17-18 2026:
GPT-6.1 Terra at medium effort, the findings judge on gpt-5.6-luna xhigh, one run per arm, **1600/1600 scored**. Infra
failures (a model rate limit inside the stream, a stalled turn) were rerun by task id; judge errors were rejudged on the
run's saved evidence, never scored 0. The verifier still scores a turn that ended in an agent error, as these runs were
scored: a task whose `report.agent_error` is set was an infra failure in our runs and was rerun by task id. 29 runs
(14 of them Stagehand) reached the 400-screenshot cap, which then dropped the final frame; this branch fixes that, and
those 29 were rerun with the fix on Nov 19 (scores below include them; no arm moved more than 0.4 points). The
reference is the same 200 tasks on Browser Use's own eval platform; the diff is paired by task with a bootstrap 95% CI.
Cost and time columns are from the Nov 17-18 runs.

| harness | score [95% CI] | reference | paired diff [95% CI] | model $/task | cloud $/task | min/task |
|---|---|---|---|---|---|---|
| browser-use 1.0.4rc1 agent | 78.0 [74.8, 81.0] | 76.1 | +1.8 [-1.0, +4.7] | 0.24 | 0.29 | 6.2 |
| Codex + Browser Use MCP | 73.9 [70.4, 77.2] | 71.7 | +2.2 [-1.2, +5.6] | 0.43 | 0.26 | 7.5 |
| Codex + Playwright MCP | 71.7 [68.2, 75.1] | 73.5 | -1.8 [-5.0, +1.3] | 0.49 | 0.38 | 7.5 |
| Codex + Chrome DevTools MCP | 71.4 [67.8, 74.7] | 70.3 | +1.0 [-2.0, +4.0] | 0.52 | 0.29 | 7.8 |
| Codex + Browser Use CLI | 69.7 [65.8, 73.6] | 70.5 | -0.7 [-3.9, +2.5] | 0.55 | 0.32 | 8.5 |
| Codex + agent-browser | 69.5 [65.9, 73.0] | 69.6 | 0.0 [-3.4, +3.5] | 0.68 | 0.29 | 7.5 |
| Codex + OpenCode browse | 66.7 [63.0, 70.5] | 62.4 | +4.3 [+1.1, +7.6] | 0.44 | 0.20 | 6.6 |
| Codex + Stagehand | 62.0 [58.3, 65.7] | 62.8 | -0.8 [-4.5, +2.9] | 1.26 | 0.36 | 16.6 |

For 7 of 8 harnesses there is no statistically detectable difference between agent-env and Browser Use's platform
(paired 95% CI contains 0). Codex + OpenCode browse scored 4.3 points higher here [+1.1, +7.6]; not yet explained. A smoke of 2 tasks on
agentenv-framework 0.9.1573, the version this repository's lock pins, scored 100 on both for `browser-use-104rc1`
and `codex-playwright-mcp`.

## Tools

`tools/bu_bench/make_bundle.py` writes the task files and evals from the benchmark; a test checks they match.
