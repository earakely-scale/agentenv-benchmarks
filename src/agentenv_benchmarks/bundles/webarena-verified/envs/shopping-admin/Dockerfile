# WebArena-Verified's shopping admin image, pinned by digest, with agentenv's site server beside its Magento as one more
# supervisord program. The image's boot init sets Magento's base URL from WA_ENV_CTRL_EXTERNAL_SITE_URL at each start.
FROM am1n3e/webarena-verified-shopping_admin@sha256:d0531dd27ed98d0c459ff9e88118bf2ed8b660b0ed99c38837db46c065a5be13
ENV WA_ENV_CTRL_EXTERNAL_SITE_URL=http://shopping-admin.localhost/
RUN python3 -m venv /opt/agentenv/venv \
    && /opt/agentenv/venv/bin/pip install --no-cache-dir "agentenv-framework-protocol==0.1.547" "uvicorn>=0.30"
COPY shopping_admin.py /opt/agentenv/shopping_admin.py
COPY agentenv-site.ini /etc/supervisor.d/agentenv-site.ini
EXPOSE 8000
