# BU Bench V2.1 findings judge, frozen

The judge of browser-use/benchmark at commit 1e0e2f1ed12d3dfbdfab6b7bff415ec090ff9f74: all 200 task contracts,
validation, prompt, schema, screenshot selection and scoring, in `upstream/`. `SOURCE.json` verifies their bytes and
those of the browser-use 0.11.5 LLM adapter in `vendor/` before every grading.

In this bundle, `bu_cloud_browser` decrypts a task's instruction from `upstream/BU_Bench_V2.enc` at run time, and
`bu_findings_verifier` runs `run.py` as a subprocess on the run's evidence. The judge decrypts the rubric itself and
checks that the instruction matches; rubrics are never sent to the agent or put in task files.

The judge is gpt-5.6-luna, xhigh, 32768 completion tokens, a 300-second request timeout and two SDK retries. It
excludes reasoning from its input. Screenshot labels say after-action. Delivered file text comes from the files the
agent wrote; binary or clipped files stay marked, not presumed inspected.

The schema requires one finding per rubric item, including the reason for a not-assessable one. The code computes
weighted credit and applies the reward-hacking/canary zero. A judge error records no score, never a zero.

Each grading writes judge-config.json, judge_trace.json and judge-recording/ (judge_input.json, judge_response.json)
into the run's evidence. They hold the decrypted task, its rubric and canary: keep them private, never upload them.
The verification names each rubric item by its id.
