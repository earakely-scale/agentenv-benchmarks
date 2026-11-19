# Codex + one browser tool: AgentEnv's published Codex agent (agentenv-framework 0.9.1536) with the harness bake-off's
# Codex CLI and every browser tool it compared, at the versions its setup.sh pinned (Browser Use's eval
# platform, harness codex-browser-tools). One image; each agent in agents/codex-* picks its tool with BU_BENCH_TOOL.
FROM ghcr.io/scaleapi/agentenv-codex@sha256:e3703ce5c2c292fc7dbe0a3b14121c1122d55aaf18d23d98ea2dbda21e23a828
USER root
ARG CODEX_VERSION=0.166.0
ARG BROWSER_HARNESS_REF=c2420943526d41b9af6182d5c5f11b9cad506ab9
ARG BROWSER_USE_VERSION=1.0.3
ARG PLAYWRIGHT_MCP_VERSION=0.0.85
ARG CHROME_DEVTOOLS_MCP_VERSION=1.13.0
ARG AGENT_BROWSER_VERSION=0.42.0
ARG BROWSE_VERSION=0.0.16
ARG STAGEHAND_VERSION=3.7.3
ARG UV_VERSION=0.9.5
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN npm install --prefix /opt/codex-cli --no-audit --no-fund --save-exact --loglevel=error "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && ln -sf /opt/codex-cli/node_modules/.bin/codex /usr/local/bin/codex \
    && codex --version | grep -q "${CODEX_VERSION}"
# Browser Use CLI (browser-harness), and browser-use for its own MCP server (browser-use --cli-mcp); both kept off
# PATH: a turn puts only its own tool's command there.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}" \
    && uv venv --python 3.12.11 /opt/bh \
    && uv pip install --python /opt/bh/bin/python \
       "browser-harness @ git+https://github.com/browser-use/browser-harness@${BROWSER_HARNESS_REF}" \
    && curl -fsSL "https://raw.githubusercontent.com/browser-use/browser-harness/${BROWSER_HARNESS_REF}/SKILL.md" \
       -o /opt/bh/SKILL.md \
    && uv venv --python 3.12.11 /opt/bu \
    && uv pip install --python /opt/bu/bin/python "browser-use==${BROWSER_USE_VERSION}" \
    && chmod -R a+rX /opt/uv-python /opt/bh /opt/bu
# The npm tools, exact versions.
RUN npm install --prefix /opt/tools/npm --no-fund --no-audit --save-exact --loglevel=error \
       "@playwright/mcp@${PLAYWRIGHT_MCP_VERSION}" \
       "chrome-devtools-mcp@${CHROME_DEVTOOLS_MCP_VERSION}" \
       "agent-browser@${AGENT_BROWSER_VERSION}" \
       "@opencode/browse@${BROWSE_VERSION}" \
    && npm cache clean --force && chmod -R a+rX /opt/tools
# Stagehand v3 behind the bake-off's small MCP server, from its lock file.
COPY stagehand/ /opt/tools/stagehand/
RUN npm ci --prefix /opt/tools/stagehand --ignore-scripts --no-fund --no-audit --loglevel=error && npm cache clean --force \
    && chmod -R a+rX /opt/tools/stagehand
COPY browse_shim.mjs /opt/tools/
# One folder per tool with only that tool's command, for the turn's PATH.
RUN mkdir -p /opt/tools/path/browser-harness /opt/tools/path/agent-browser \
    && ln -s /opt/bh/bin/browser-harness /opt/tools/path/browser-harness/browser-harness \
    && ln -s /opt/tools/npm/node_modules/.bin/agent-browser /opt/tools/path/agent-browser/agent-browser
# Codex runs commands in a login shell (`/bin/sh -lc`), and Debian's /etc/profile resets PATH, dropping the folder
# the agent prepends. Put it back there so the bare command works, as on the bake-off's Ubuntu runner.
RUN printf '%s\n' 'case "${BU_BENCH_TOOL:-}" in browser-harness|agent-browser) PATH="/opt/tools/path/$BU_BENCH_TOOL:$PATH";; esac' \
       > /etc/profile.d/bu-bench-tool.sh
# What was installed, checked against the pins; a mismatch fails the build.
RUN set -eu; bin=/opt/tools/npm/node_modules/.bin; { \
      echo "codex $(codex --version)"; \
      echo "node $(node --version)"; \
      echo "browser-harness ${BROWSER_HARNESS_REF}"; \
      echo "browser-harness-cli $(/opt/bh/bin/browser-harness --version)"; \
      echo "browser-harness-skill $(sha256sum /opt/bh/SKILL.md | cut -d ' ' -f 1)"; \
      echo "playwright-mcp $("$bin/playwright-mcp" --version)"; \
      echo "chrome-devtools-mcp $("$bin/chrome-devtools-mcp" --version)"; \
      echo "agent-browser $("$bin/agent-browser" --version)"; \
      echo "opencode-browse $(node /opt/tools/npm/node_modules/@opencode/browse/dist/cli.mjs --version)"; \
      echo "stagehand $(node -p "require('/opt/tools/stagehand/node_modules/@browserbasehq/stagehand/package.json').version")"; \
      echo "stagehand-lock $(sha256sum /opt/tools/stagehand/package-lock.json | cut -d ' ' -f 1)"; \
      echo "browse-shim $(sha256sum /opt/tools/browse_shim.mjs | cut -d ' ' -f 1)"; \
      echo "npm-lock $(sha256sum /opt/tools/npm/package-lock.json | cut -d ' ' -f 1)"; \
      uv pip freeze --python /opt/bh/bin/python | sed 's/^/bh-python /'; \
      uv pip freeze --python /opt/bu/bin/python | sed 's/^/bu-python /'; \
    } > /opt/tools/versions.txt; cat /opt/tools/versions.txt; \
    grep -q "^codex codex-cli ${CODEX_VERSION}$" /opt/tools/versions.txt; \
    grep -q "^playwright-mcp Version ${PLAYWRIGHT_MCP_VERSION}$" /opt/tools/versions.txt; \
    grep -q "^chrome-devtools-mcp ${CHROME_DEVTOOLS_MCP_VERSION}$" /opt/tools/versions.txt; \
    grep -q "^agent-browser agent-browser ${AGENT_BROWSER_VERSION}$" /opt/tools/versions.txt; \
    grep -q "^bu-python browser-use==${BROWSER_USE_VERSION}$" /opt/tools/versions.txt; \
    grep -q "^stagehand ${STAGEHAND_VERSION}$" /opt/tools/versions.txt; \
    grep -q "^opencode-browse ${BROWSE_VERSION}$" /opt/tools/versions.txt
COPY codex_tools_agent.py /opt/agent/
USER agent
WORKDIR /app
CMD ["uvicorn", "codex_tools_agent:app", "--app-dir", "/opt/agent", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "5"]
