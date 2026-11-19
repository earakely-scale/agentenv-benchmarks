# bu-browser-site: agentenv's site server (as root, holding the settings) and Playwright MCP 0.0.85 (as pw, holding
# nothing but the browser's CDP address). Node is agentenv-framework's own pinned base (its playwright_mcp env).
FROM public.ecr.aws/docker/library/node:22-slim@sha256:813a7480f28fdadac1f7f5c824bcdad435b5bc1322a5968bbbdef8d058f9dff4
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/bu-browser/venv \
    && /opt/bu-browser/venv/bin/pip install --no-cache-dir "agentenv-framework-protocol==0.1.588" "mcp==1.30.0" \
       "uvicorn==0.54.0" "httpx==0.28.1" \
    && useradd --create-home --shell /usr/sbin/nologin pw \
    && install -d /home/pw/playwright-output
WORKDIR /opt/bu-browser
# No browser download: Playwright MCP attaches to the cloud browser over --cdp-endpoint.
RUN npm install --prefix /opt/bu-browser --no-audit --no-fund --save-exact --loglevel=error "@playwright/mcp@0.0.85" \
    && npm cache clean --force
COPY cloud.py playwright_mcp.py browser_site.py /opt/bu-browser/
# Checks the installed Playwright MCP is the version playwright_mcp.py names, and writes the tools it serves.
RUN venv/bin/python playwright_mcp.py --write-manifest
EXPOSE 8000
CMD ["/opt/bu-browser/venv/bin/python", "/opt/bu-browser/browser_site.py"]
