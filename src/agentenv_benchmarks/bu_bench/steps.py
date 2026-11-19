"""The two task steps BU Bench adds to AgentEnv.

``bu_cloud_browser`` starts a Browser Use Cloud browser, hands its CDP address to the agent through the agent's
``cdp-config/v1`` (as the built-in ``browser_connect_cdp`` step does for AgentEnv's own Chromium), starts the
screenshot capture the judge reads, and puts the task's instruction into the seed value ``bu_task`` the prompt names.

``bu_findings_verifier`` stops the capture and the browser (always, even when the agent failed), keeps the
recording, builds the judge evidence from the agent's reply, its trajectory and the delivered files (scrubbed of the
browser's CDP address, which is its credential), and runs the frozen BU Bench V2.1 findings judge unchanged, as a
subprocess. The score is the judge's continuous one, 0 to 1. A judge error is asked again; one that persists, a turn
that left no response, a failed step it depends on other than the turn's own (collect_artifacts), a trajectory it
can't read or with no step the judge reads and a delivered file it can't read fail the step and record no score: never
a zero.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import httpx
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep

from agentenv_benchmarks.bu_bench import cloud, evidence, tasks

logger = logging.getLogger(__name__)

STATE_KEY = "bu_cloud_browser"
SEED_KEY = "bu_task"
JUDGE_RUN = tasks.JUDGE_DIR / "run.py"
JUDGE_TIMEOUT_S = 1800
#: Judge runs per task while it errs (a timeout is not retried), and the wait before each retry (times the attempt).
JUDGE_ATTEMPTS = 3
JUDGE_RETRY_S = 5.0
#: Where each task's judge inputs and outputs are kept after the run. Unset, they go to a temporary folder, deleted
#: once the task is graded and kept when it isn't.
EVIDENCE_DIR_ENV = "BU_BENCH_EVIDENCE_DIR"
JUDGE_KEY_NAMES = ("BU_BENCH_JUDGE_API_KEY", "OPENAI_API_KEY")
#: All the judge subprocess keeps of this process's environment: its paths, locale, proxy and CA settings.
JUDGE_ENVIRONMENT = {"PATH", "HOME", "LANG", "TMPDIR", "OPENAI_BASE_URL", "SSL_CERT_FILE", "SSL_CERT_DIR",
                     "REQUESTS_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy",
                     "no_proxy"}
#: Not deliverables, as the browser-use eval adapter skipped them: browser-use's own REPL cells and logs, installs.
SKIPPED_PARTS = {".browser-use", "node_modules", ".git", "screenshots"}


@dataclass
class _Live:
    browser: cloud.CloudBrowser
    screenshots: cloud.Screenshots
    shots_dir: Path
    started: float


#: Browsers this process started and has not stopped, by handle. Both steps run in the runner's process.
_LIVE: dict[str, _Live] = {}
#: How long the exit hook waits for all its stops, sent at once: httpx's timeout bounds each read, so an answer that
#: trickles in would otherwise hold the exit, and the stops after it, without end.
EXIT_STOP_S = 20.0


@atexit.register
def _stop_leftovers() -> None:
    """A run that ends before its verifier (Ctrl-C, a crash) still stops its browsers. Each stop runs on a thread of its
    own (asyncio can't resolve a host once the interpreter is shutting down), waited for until EXIT_STOP_S has passed."""
    browsers = [live.browser for live in _LIVE.values()] + list(cloud.UNSTOPPED)
    threads = [threading.Thread(target=_stop_at_exit, args=(browser,), daemon=True) for browser in browsers]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + EXIT_STOP_S
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


def _stop_at_exit(browser: cloud.CloudBrowser) -> None:
    try:
        httpx.patch(f"{cloud.API}/browsers/{browser.id}", json={"action": "stop"}, timeout=EXIT_STOP_S,
                    headers={"X-Browser-Use-API-Key": cloud.api_key().strip()})
    except Exception:  # noqa: BLE001, S110 - the browser's own timeout ends it
        pass


def _control_headers(agent) -> dict[str, str]:
    """The agent's control token header: agents on agentenv-framework-protocol >= 0.1.582 answer 401 on extension
    routes without it. AgentEnv 0.9.1536 has no token, so nothing is sent there."""
    return dict(getattr(agent, "control_headers", None) or {})


def _judge_key() -> str:
    """The judge's OpenAI key: BU_BENCH_JUDGE_API_KEY, else OPENAI_API_KEY, else agent-env's secret store."""
    key = next((os.environ[name] for name in JUDGE_KEY_NAMES if os.environ.get(name)), "")
    if not key:
        try:
            from agent_env.config import get_config

            key = get_config().get_secret_store().get("OPENAI_API_KEY") or ""
        except Exception:  # noqa: BLE001 - no framework or no secret: the caller says what to set
            key = ""
    return key


def _agent(context: TaskStepContext, agent_name: str):
    agent = next((a for a in context.deployed_agents if a.agent_name == agent_name), None)
    if agent is None:
        raise RuntimeError(f"agent {agent_name!r} isn't deployed in this run")
    return agent


class BuCloudBrowserTaskStep(TaskStep):
    type: ClassVar[str] = "bu_cloud_browser"

    def __init__(self, id: str, version: int | None, task_id: str, agent_name: str | None = None,
                 name: str = "bu-cloud", proxy_country_code: str | None = "us", timeout_minutes: int = 70,
                 enable_recording: bool = True, screenshot_interval_seconds: float = 2.0,
                 depends_on: list | None = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.task_id = task_id
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.name = name
        self.proxy_country_code = proxy_country_code
        self.timeout_minutes = timeout_minutes
        self.enable_recording = enable_recording
        self.screenshot_interval_seconds = screenshot_interval_seconds

    def config_problems(self) -> list[str]:
        problems = []
        if not isinstance(self.timeout_minutes, int) or not 1 <= self.timeout_minutes <= 240:
            problems.append(f"step {self.id!r}: timeout_minutes must be 1..240")
        interval = self.screenshot_interval_seconds
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval < 0.5:
            problems.append(f"step {self.id!r}: screenshot_interval_seconds must be a number >= 0.5")
        return problems

    def preflight(self) -> list[str]:
        try:
            tasks.instruction(self.task_id)
        except Exception as error:  # noqa: BLE001
            return [f"step {self.id!r}: {error}"]
        try:
            cloud.api_key()
        except Exception as error:  # noqa: BLE001 - says what to set, before any agent or browser starts
            return [f"step {self.id!r}: {error}"]
        return []

    def to_dict(self) -> dict:
        return {**super().to_dict(), "task_id": self.task_id, "agent_name": self.agent_name, "name": self.name,
                "proxy_country_code": self.proxy_country_code, "timeout_minutes": self.timeout_minutes,
                "enable_recording": self.enable_recording,
                "screenshot_interval_seconds": self.screenshot_interval_seconds}

    @classmethod
    def from_dict(cls, data: dict) -> BuCloudBrowserTaskStep:
        return cls(**cls._base_from_dict(data), task_id=data["task_id"], agent_name=data.get("agent_name"),
                   name=data.get("name", "bu-cloud"), proxy_country_code=data.get("proxy_country_code", "us"),
                   timeout_minutes=data.get("timeout_minutes", 70),
                   enable_recording=data.get("enable_recording", True),
                   screenshot_interval_seconds=data.get("screenshot_interval_seconds", 2.0))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        context.metadata.setdefault("seed", {})[SEED_KEY] = tasks.instruction(self.task_id)
        agent = _agent(context, self.agent_name)
        cdp_config = A2AAgent.find_extension(agent.a2a_card or {}, A2AAgent.EXT_CDP_CONFIG)
        if cdp_config is None or not agent.a2a_url:
            raise RuntimeError(f"agent {agent.agent_name!r} can't take a browser: its card has no "
                               f"{A2AAgent.EXT_CDP_CONFIG}")
        handle, browser = uuid.uuid4().hex, None
        shots_dir = Path(tempfile.mkdtemp(prefix=f"bu-shots-{self.task_id}-"))
        try:
            browser = await cloud.CloudBrowser.start(proxy_country_code=self.proxy_country_code,
                                                     timeout_minutes=self.timeout_minutes,
                                                     enable_recording=self.enable_recording)
            # Held for the exit hook before the handoff: a browser this step fails to stop is still stopped at exit.
            screenshots = cloud.Screenshots(browser.ws_url, shots_dir, interval=self.screenshot_interval_seconds,
                                            seconds=self.timeout_minutes * 60)
            _LIVE[handle] = _Live(browser, screenshots, shots_dir, time.time())
            async with httpx.AsyncClient(timeout=60) as client:
                added = await client.post(agent.a2a_url + cdp_config["params"]["endpoint"],
                                          headers=_control_headers(agent), json={
                    "name": self.name, "http_url": browser.cdp_url, "ws_url": browser.ws_url,
                    "metadata": {"browser": "browser-use-cloud", "browser_label": browser.label}})
            added.raise_for_status()
            screenshots.start()
        except BaseException:
            if browser is not None:
                await browser.stop()
            if browser is None or browser.stopped:  # else the exit hook tries again
                _LIVE.pop(handle, None)
                shutil.rmtree(shots_dir, ignore_errors=True)
            raise
        context.metadata[STATE_KEY] = {"handle": handle, "browser": browser.label, "task_id": self.task_id,
                                       "started_at": time.time()}
        logger.info(f"agent {agent.agent_name!r} has Browser Use Cloud browser {browser.label} as {self.name!r}")
        return context


class BuFindingsVerifierTaskStep(TaskStep):
    type: ClassVar[str] = "bu_findings_verifier"

    def __init__(self, id: str, version: int | None, task_id: str, prompt_id: str,
                 agent_name: str | None = None, verifier_id: str = "findings", judge_model: str = "gpt-5.6-luna",
                 reasoning_effort: str = "xhigh", keep_recording: bool = True,
                 depends_on: list | None = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.task_id = task_id
        self.prompt_id = prompt_id
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.verifier_id = verifier_id
        self.judge_model = judge_model
        self.reasoning_effort = reasoning_effort
        self.keep_recording = keep_recording

    def preflight(self) -> list[str]:
        if _judge_key():
            return []
        return [(f"step {self.id!r}: no judge key: set BU_BENCH_JUDGE_API_KEY or OPENAI_API_KEY (or OPENAI_API_KEY in "
                 "agent-env's secret store)")]

    def to_dict(self) -> dict:
        return {**super().to_dict(), "task_id": self.task_id, "prompt_id": self.prompt_id,
                "agent_name": self.agent_name, "verifier_id": self.verifier_id, "judge_model": self.judge_model,
                "reasoning_effort": self.reasoning_effort, "keep_recording": self.keep_recording}

    @classmethod
    def from_dict(cls, data: dict) -> BuFindingsVerifierTaskStep:
        return cls(**cls._base_from_dict(data), task_id=data["task_id"], prompt_id=data["prompt_id"],
                   agent_name=data.get("agent_name"), verifier_id=data.get("verifier_id", "findings"),
                   judge_model=data.get("judge_model", "gpt-5.6-luna"),
                   reasoning_effort=data.get("reasoning_effort", "xhigh"),
                   keep_recording=data.get("keep_recording", True))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        state = context.metadata.get(STATE_KEY) or {}
        handle = state.get("handle", "")
        live = _LIVE.get(handle)
        if live is None:
            # No browser step ran, or its screenshots and browser are gone (an earlier attempt of this step stopped
            # them, or another process started it): a judgment now would score the agent on no evidence.
            raise RuntimeError(f"{self.task_id}: no live browser for this task's handle, so nothing to judge")
        report: dict[str, Any] = {"task_id": self.task_id, "browser": state.get("browser")}
        secrets = live.browser.secrets
        session: dict[str, Any] = {}
        try:
            report["screenshots"] = await live.screenshots.stop()
            shot_names = list(live.screenshots.saved)
            workspace = self._workspace(context, handle)
            (workspace / "judge_screenshots").mkdir(exist_ok=True)
            for name in shot_names:
                shutil.copy2(live.shots_dir / name, workspace / "judge_screenshots" / name)
        finally:
            await live.browser.stop()
            if live.browser.stopped:  # else the exit hook tries again, and a retry of this step still has them
                _LIVE.pop(handle, None)
                shutil.rmtree(live.shots_dir, ignore_errors=True)
            report["browser_seconds"] = round(time.time() - live.started, 1)
            try:
                await asyncio.sleep(2)  # costs and the recording link appear once the session has stopped
                session = await live.browser.session()
            except Exception as error:  # noqa: BLE001 - cost lookup never fails the task
                logger.warning("reading cloud browser session %s failed (%s)", live.browser.label,
                               cloud.scrub(str(error), secrets))
        report["cloud"] = cloud.costs(session)
        if self.keep_recording:
            report["recording"] = await self._keep_recording(session, context)

        response = next((r for r in reversed(context.prompt_responses) if r.prompt_id == self.prompt_id), None)
        if response is None:
            # The turn never answered (it timed out, its agent died or the prompt never reached it): judging an empty
            # reply would score the agent on what the run lost, not on what it did.
            raise RuntimeError(f"{self.task_id}: the agent's turn {self.prompt_id!r} left no response, so nothing to "
                               f"judge (its screenshots are kept in {workspace})")
        if lost := self._failed_dependencies(context):
            # A failed collect_artifacts keeps only the files it uploaded before it failed: judging now would score the
            # agent on files the run lost. The turn's own failure is judged on the response it left, as published.
            raise RuntimeError(f"{self.task_id}: step {lost}, which this verifier depends on, failed, so nothing is "
                               f"judged (its screenshots are kept in {workspace})")
        final_output = cloud.scrub(response.response or "", secrets)
        report["agent_error"] = ({"type": response.error_type, "code": response.error_code,
                                  "message": cloud.scrub(response.error_message or "", secrets)[:2000]}
                                 if response.error_type else None)
        trajectory, report["trajectory_source"] = await self._trajectory(context, response)
        trajectory = cloud.scrub_json(trajectory, secrets)
        (workspace / "trajectory.json").write_text(json.dumps(trajectory, default=str))
        fmt, steps, usage = evidence.steps_and_usage(trajectory)
        if not evidence.judge_visible(steps):
            # Lost, empty, only control events or only thinking (which the judge drops): no turn that answered took no
            # step at all, so the judge would see steps the run lost.
            raise RuntimeError(f"{self.task_id}: no trajectory with a step the judge reads could be read (source: "
                               f"{report['trajectory_source']}), so nothing is judged (its screenshots are kept in "
                               f"{workspace})")
        files = cloud.scrub_json(await asyncio.to_thread(self._delivered, context, response), secrets)
        report.update(trajectory_format=fmt, steps=len(steps), usage=usage, model_usd=evidence.model_usd(usage),
                      delivered_files=[f["name"] for f in files])
        (workspace / "task.json").write_text(json.dumps({"task_id": self.task_id,
                                                         "confirmed_task": tasks.instruction(self.task_id)}))
        (workspace / "result.json").write_text(json.dumps({
            "final_output": final_output,
            "metadata": {"steps": steps, "judge_screenshots": [f"judge_screenshots/{n}" for n in shot_names],
                         "output_files": files}}))
        judgment = await self._judge(workspace)
        report["judgment"] = {k: judgment.get(k) for k in ("status", "score", "reason", "failure_class", "model")}
        (workspace / "report.json").write_text(json.dumps(report, indent=1, default=str))

        verification: dict[str, Any] = {
            "verifier_type": self.type, "judge": "findings-v21 (browser-use/benchmark@1e0e2f1e)",
            "judge_model": self.judge_model, "reasoning_effort": self.reasoning_effort,
            "results": [{"id": item, "result": met, "score": 1.0 if met else 0.0}
                        for item, met in (judgment.get("rubric") or {}).items()],
            "reason": judgment.get("reason"), "failure_class": judgment.get("failure_class"),
            "report": report, "evidence_dir": str(workspace),
        }
        if judgment.get("status") != "scored":
            context.metadata.setdefault("verifications", {})[self.verifier_id] = verification
            raise RuntimeError(f"findings judge did not score {self.task_id}: {judgment.get('reason')} "
                               f"(its evidence is kept in {workspace})")
        verification["score"] = float(judgment["score"])
        if not os.environ.get(EVIDENCE_DIR_ENV):  # a temporary folder holding the decrypted task, rubric and canary
            shutil.rmtree(workspace, ignore_errors=True)
            verification["evidence_dir"] = None
        context.metadata.setdefault("verifications", {})[self.verifier_id] = verification
        return context

    def _workspace(self, context: TaskStepContext, handle: str) -> Path:
        """``<task>-<run group>-<browser handle>``: one task's runs in one group (an explorer batch) each get their own."""
        root = os.environ.get(EVIDENCE_DIR_ENV)
        run = str(context.metadata.get("run_group_id") or uuid.uuid4().hex)[:12]
        if root:
            path = Path(root) / f"{self.task_id}-{run}-{handle[:8]}"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(tempfile.mkdtemp(prefix=f"bu-judge-{self.task_id}-"))

    async def _trajectory(self, context: TaskStepContext, response) -> tuple[Any, str]:
        """The turn's trajectory as prompt_agent stored it, else asked of the agent itself, inline: the stored copy
        goes through the runner's grant server, and a failed fetch there is only logged."""
        url = response.agent_trajectory_object_url or next(
            (u for u in reversed(response.target_agent_per_turn_trajectory_object_urls or []) if u), None)
        if url:
            from agent_env.config import get_config

            try:
                return json.loads(await asyncio.to_thread(get_config().get_object_store_at(url).get, url)), "stored"
            except Exception:
                logger.exception("reading the stored trajectory of %s failed", self.task_id)
        try:
            agent = _agent(context, self.agent_name)
            extension = A2AAgent.find_extension(agent.a2a_card or {}, A2AAgent.EXT_TRAJECTORY)
            _, path = A2AAgent.operation(extension, "get")
            async with httpx.AsyncClient(timeout=120) as client:
                fetched = await client.post(agent.a2a_url + path, headers=_control_headers(agent),
                                            json={"context_id": response.a2a_context_id})
            fetched.raise_for_status()
            answer = fetched.json()  # with the format it declares, which the stored copy doesn't keep
            return {"format": answer.get("format"), "trajectory": answer.get("trajectory") or []}, "agent"
        except Exception:
            logger.exception("fetching the trajectory of %s from the agent failed", self.task_id)
            return [], "none"

    def _failed_dependencies(self, context: TaskStepContext) -> str:
        """The steps this one depends on (every earlier one, with no ``depends_on``) that failed, other than a prompt
        step, whose turn is judged on its response; a failure a retry recovered from is no failure."""
        from agent_env.task_step.task_steps.prompt_agent import is_prompt_step

        needed = None if self.depends_on is None else {dependency.task_step_id for dependency in self.depends_on}
        return ", ".join(str(failure.get("step_id")) for failure in context.metadata.get("failed_steps") or []
                         if isinstance(failure, dict) and not failure.get("retried")
                         and (needed is None or failure.get("step_id") in needed)
                         and not is_prompt_step({"type": failure.get("step_type")}))

    def _delivered(self, context: TaskStepContext, response) -> list[dict[str, Any]]:
        """The judge's share of the delivered files. One it can't read fails the step: judging without it would score
        the agent on files the run lost."""
        collected = self._collected(context, response)
        if not isinstance(collected, dict):
            return []
        from agent_env.config import get_config

        delivered = [(str(name), url) for name, url in collected.items()
                     if not any(part in SKIPPED_PARTS for part in Path(str(name)).parts)]
        try:
            return evidence.output_files([(name, lambda url=url: get_config().get_object_store_at(url).open(url))
                                          for name, url in delivered])
        except Exception as error:
            raise RuntimeError(f"{self.task_id}: reading a delivered file failed ({type(error).__name__}: {error}), "
                               "so nothing is judged") from error

    def _collected(self, context: TaskStepContext, response) -> Any:
        """The delivered files, name: object URL, as the collect step this verifier depends on recorded them: its own
        entry in ``collected_artifacts``, else the legacy flat ``artifacts`` (the latest collect's). The verifier's
        named dependencies are its turn and its collect: one of those that succeeded and recorded neither fails the
        step, as judging no files would score the agent on files the run lost."""
        by_step = context.metadata.get("collected_artifacts") or {}
        collects = list(by_step) if self.depends_on is None else [
            dependency.task_step_id for dependency in self.depends_on
            if dependency.task_step_id not in {response.step_id, self.prompt_id}]
        for step_id in reversed(collects):
            entry = by_step.get(step_id)
            if isinstance(entry, dict) and "artifacts" in entry:
                return entry["artifacts"] or {}
        if "artifacts" in context.metadata:
            return context.metadata["artifacts"] or {}
        if self.depends_on is not None and collects:
            raise RuntimeError(f"{self.task_id}: step {', '.join(collects)}, which this verifier depends on, recorded "
                               "no collected files, so nothing is judged")
        return {}

    async def _keep_recording(self, session: dict[str, Any], context: TaskStepContext) -> dict[str, Any]:
        """The cloud browser's own MP4, stored as a file artifact; its signed link expires in an hour."""
        urls = cloud.recording_urls(session)
        if not urls:
            return {"available": bool(session.get("recordingAvailable")), "artifact_id": None}
        folder = Path(tempfile.mkdtemp(prefix=f"bu-recording-{self.task_id}-"))
        uploading = threading.Event()
        try:
            # Streamed through a file both ways, never held whole in memory: a long task's MP4 is hundreds of MB.
            video = folder / "recording.mp4"

            async def scrubbed(request: httpx.Request) -> None:
                """Each hop's host joins the log filter's before httpx logs it: a redirect's link is signed too."""
                cloud.RECORDING_HOSTS.add(request.url.host)

            async with httpx.AsyncClient(timeout=120, follow_redirects=True,
                                         event_hooks={"request": [scrubbed]}) as client, \
                    client.stream("GET", urls[0]) as answer:
                answer.raise_for_status()
                with video.open("wb") as out:
                    async for chunk in answer.aiter_bytes(1 << 20):
                        out.write(chunk)
            from agent_env.artifact.artifacts.file import FileArtifact

            # Under the run's instance id, so a local run keeps it in the local store, as collect_artifacts does.
            artifact_id = f"{context.instance_id}-recording" if context.instance_id else \
                f"bu-bench-recording-{self.task_id}-{uuid.uuid4().hex[:8]}"
            size = video.stat().st_size

            def upload() -> Any:
                """Deletes the file once uploaded: a thread can't be cancelled, so a cancelled wait for it leaves it
                reading the file."""
                uploading.set()
                try:
                    return FileArtifact.put(artifact_id, file_path=str(video),
                                            description=f"Browser Use Cloud recording of {self.task_id}")
                finally:
                    shutil.rmtree(folder, ignore_errors=True)

            artifact = await asyncio.to_thread(upload)
            context.metadata.setdefault("recordings", {})[self.id] = [
                {"name": "recording.mp4", "artifact_id": artifact.id, "version": artifact.version,
                 "bytes": size, "content_type": artifact.content_type}]
            return {"available": True, "artifact_id": artifact.id, "bytes": size}
        except Exception as error:  # noqa: BLE001 - the recording never fails the task
            # Not logger.exception: httpx's error text names the signed recording link.
            logger.warning("keeping the recording of %s failed (%s)", self.task_id, type(error).__name__)
            return {"available": True, "artifact_id": None, "error": type(error).__name__}
        finally:
            if not uploading.is_set():  # else the upload deletes it
                shutil.rmtree(folder, ignore_errors=True)

    async def _judge(self, workspace: Path) -> dict[str, Any]:
        """The judge's judgment, asked up to JUDGE_ATTEMPTS times while it errs: one failed judge request must not
        lose the task, and its evidence is still on disk. Cancelled, it kills the judge it runs: no judge goes on
        asking the model for a task nobody waits on."""
        key = _judge_key()
        if not key:
            return {"status": "error", "reason": "no judge key: set BU_BENCH_JUDGE_API_KEY or OPENAI_API_KEY"}
        env = {name: value for name, value in os.environ.items() if name in JUDGE_ENVIRONMENT}
        env.update(EVAL_WORKSPACE=str(workspace), EVAL_JUDGMENT_PATH=str(workspace / "judgment.json"),
                   EVAL_TASK_PATH=str(workspace / "task.json"), EVAL_RESULT_PATH=str(workspace / "result.json"),
                   EVAL_JUDGE_MODEL=self.judge_model,
                   EVAL_JUDGE_OPTIONS_JSON=json.dumps({"reasoning_effort": self.reasoning_effort}),
                   EVAL_JUDGE_API_KEY=key, PYTHONDONTWRITEBYTECODE="1")
        judgment: dict[str, Any] = {}
        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            if attempt > 1:
                logger.warning("judging %s failed (%s), attempt %d of %d", self.task_id, judgment.get("reason"),
                               attempt - 1, JUDGE_ATTEMPTS)
                await asyncio.sleep(JUDGE_RETRY_S * (attempt - 1))
            (workspace / "judgment.json").unlink(missing_ok=True)
            with (workspace / "judge.log").open("a") as log:
                judge = await asyncio.create_subprocess_exec(sys.executable, str(JUDGE_RUN), env=env, stdout=log,
                                                             stderr=asyncio.subprocess.STDOUT)
                try:
                    await asyncio.wait_for(judge.wait(), JUDGE_TIMEOUT_S)
                except TimeoutError:
                    return {"status": "error", "reason": f"judge timed out after {JUDGE_TIMEOUT_S}s"}
                finally:
                    if judge.returncode is None:
                        judge.kill()
                        await asyncio.shield(judge.wait())
            try:
                judgment = json.loads((workspace / "judgment.json").read_text())
            except (OSError, ValueError) as error:
                judgment = {"status": "error", "reason": f"judge wrote no judgment: {error}"}
            if judgment.get("status") != "error":
                break
        return judgment
