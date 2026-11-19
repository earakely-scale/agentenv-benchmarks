# browser-use 1.0.4rc1 as an AgentEnv agent; see NOTICE for what differs from AgentEnv's own browser-use agent.
FROM mirror.gcr.io/library/python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
WORKDIR /app
COPY requirements.txt browser-use-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m venv /opt/browser-use \
    && /opt/browser-use/bin/pip install --no-cache-dir -r browser-use-requirements.txt \
    && useradd --create-home agent && mkdir -p /app/agent_outputs && chown -R agent /app
COPY a2a_server.py agent.py ./
USER agent
# As the BU Bench eval adapter set them: page titles left as they are, no live-view link.
ENV ANONYMIZED_TELEMETRY=false BH_TELEMETRY=0 BH_TAB_MARKER=0 BH_OPEN_LIVE_URL=0 PYDANTIC_AI_NO_BANNER=1
EXPOSE 8000
CMD ["python3", "a2a_server.py"]
