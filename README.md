# agentenv-benchmarks

Public agent benchmarks as [agent-env](https://github.com/scaleapi/agentenv-framework) bundles. Each benchmark runs
with `agent-env run <name>`, on agent-env's own environments, against the agents agent-env ships.

| Benchmark | Bundle | Tasks | Arms | Source |
| --- | --- | --- | --- | --- |
| [Online-Mind2Web](src/agentenv_benchmarks/bundles/online-mind2web/README.md) | `online-mind2web` | 300 live-web tasks on 147 sites | Browser Use over CDP; Claude Code with Playwright MCP | [`osunlp/Online-Mind2Web`](https://huggingface.co/datasets/osunlp/Online-Mind2Web) @ `eacad896` |

## Install and run

```bash
uv tool install agentenv-framework --with git+https://github.com/earakely-scale/agentenv-benchmarks
agent-env run online-mind2web --eval sample
```

`agent-env run` with no bundle lists the installed ones. From a checkout, a bundle also runs by its folder:

```bash
git clone https://github.com/earakely-scale/agentenv-benchmarks && cd agentenv-benchmarks
uv sync
uv run agent-env run src/agentenv_benchmarks/bundles/online-mind2web --eval sample
```

The agents need a model endpoint: `[model] base_url` and `api_key` in agent-env's `.agentenv/config.toml`, or
`LITELLM_BASE_URL` and `LITELLM_API_KEY`. agent-env's [docs](https://www.agentenvframework.com/docs) cover its
configuration, sandboxes and stores.

## Layout

```
src/agentenv_benchmarks/
  bundles/<name>/          a benchmark's bundle: envs, agents, artifacts, tasks/ and evals/, as agent-env reads them
  <name with underscores>/ the code that generates its tasks from its source dataset, when they're generated
tests/                     checks every bundle parses and its tasks are what its generator makes
```

## Adding a benchmark

1. Put its bundle in `src/agentenv_benchmarks/bundles/<name>/`, with a `README.md` whose first line describes it and a
   `bundle.toml` naming the agentenv-framework it needs.
2. If its tasks come from a dataset, add `src/agentenv_benchmarks/<name>/` with the dataset pinned to a revision and a
   `__main__` that writes the tasks, as `online_mind2web` does, and commit what it writes.
3. Register it in `pyproject.toml` under `[project.entry-points."agent_env.bundles"]`:
   `<name> = "agentenv_benchmarks.bundles"`.
4. Add it to the table above and its tests to `tests/`.

## Develop

```bash
uv sync --group dev
uv run ruff check
uv run pytest
```

Regenerating Online-Mind2Web's tasks needs a Hugging Face login that has accepted the dataset's terms:

```bash
hf auth login
uv run --extra generate python -m agentenv_benchmarks.online_mind2web
```

## License

The code is Apache-2.0. Each benchmark's tasks keep their dataset's license and attribution; see [NOTICE](NOTICE).
