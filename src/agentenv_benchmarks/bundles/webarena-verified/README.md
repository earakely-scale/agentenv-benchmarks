WebArena-Verified's 182 single-site shopping admin tasks, on its own Magento image wrapped as a website env, with Browser Use over CDP, graded offline by WebArena-Verified's own evaluator.

# WebArena-Verified: shopping admin

The tasks are [WebArena-Verified](https://github.com/ServiceNow/webarena-verified)'s (Apache-2.0), every single-site
task of its shopping admin site in the dataset release 1.2.3 ships (`webarena-verified.json`, sha256 `d6527566…`):
182 tasks across 41 intent templates, 55 of them in its hard subset, 94 graded on the network log as well as the
answer. `agentenv_benchmarks.webarena_verified` generates them.

## Run

```bash
agent-env run webarena-verified --eval sample --model claude-haiku-5-5   # 41 tasks, one per intent template
agent-env run webarena-verified --eval all --model claude-haiku-5-5      # all 182
agent-env run webarena-verified --task shopping-admin-0 --model claude-haiku-5-5
```

The site's image is amd64 alone and about 1.2 GB compressed, so a run needs an amd64 machine for local Docker, or
`modal_vm`. The agent needs a model endpoint (`[model]` in agent-env's config), and every result names the model it
ran on.

## What every task does

| Step | What it does |
| --- | --- |
| `office` | Deploys the `office` multi env: the `shopping-admin` site and agent-env's `chromium`, its start tabs the task's start URLs, `__SHOPPING_ADMIN__` made `http://shopping-admin.localhost/admin`, in a 1280×800 window, en-US, UTC. |
| `sign_in` | Signs the browser in as `admin` through the site's `website/v1` session, which signs in server-side with the image's admin auto-login header; the start tabs load again, signed in. |
| `agent` / `connect` | Deploys Browser Use and hands it a CDP address to the browser, through the multi env's gateway. |
| `task` | Prompts it with WebArena-Verified's own shopping admin prompt, rendered with the task's intent and start URLs. Its final message is the prompt's response, `answer`. |
| `snapshot` | Snapshots the multi env after the agent's turn: the browser's data/get, `network.har` among it, saved with the run. |

## The site

`envs/shopping-admin` is WebArena-Verified's own `shopping_admin` image, pinned by digest, with agentenv's site server
(`shopping_admin.py`, a `ForwardingSite`) as one more supervisord program beside its Magento:

- It forwards every request but agentenv's own to Magento on port 80, as it came, under the site's name.
- The image's boot init sets Magento's base URL to `http://shopping-admin.localhost/` at each start; the site is ready
  once that init says `[OK]` and Magento's sign-in page names the site.
- Its session signs an admin in by one server-side request carrying `X-M2-Admin-Auto-Login`, which the forwarder drops
  from every browser request, and the browser's network log takes its value out of.
- Every run deploys afresh, so each starts on the image's own data; the site takes no load and can't reset.

## Export and grade

`agentenv_benchmarks.webarena_verified.export` writes runs as the files WebArena-Verified's evaluator grades, by the
instance ids `agent-env run` prints, and `grade` grades them with the release's own evaluator:

```bash
python -m agentenv_benchmarks.webarena_verified.export results/sample-1 INSTANCE_ID...
python -m agentenv_benchmarks.webarena_verified.grade results/sample-1
```

Each run becomes `results/sample-1/<task_id>/`:

- `network.har`, the browser's network log from the run's snapshot;
- `agent_response.json`, the JSON object WebArena-Verified's own extractor (`upstream/extract_agent_response.py`, copied
  unchanged from the release) finds in the agent's final message, or that message as text when it finds none, which
  the evaluator scores as a wrong answer;
- `agentenv.json`, which names the run, its model, the task, the release and the dataset's sha256.

A run with no snapshot, or no `network.har` in it, isn't exported. Grading writes `results/sample-1/grades.jsonl`, a
line a run with its score, status and each evaluator's result, and lists apart, unscored, any run the evaluator errors
on. It runs the release through `uvx`, since its exact pins can't share agent-env's environment, so it needs `uv`.

## Regenerate

```bash
uv run --extra generate python -m agentenv_benchmarks.webarena_verified
```

It reads the dataset from the release's own package, which `uvx` installs, checks each file's sha256, and renders the
prompt with Jinja as WebArena-Verified's own agents do.
