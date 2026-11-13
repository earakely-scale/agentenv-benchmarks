"""WebArena-Verified (ServiceNow/webarena-verified, Apache-2.0): its shopping admin site's 182 single-site tasks as an
agent-env bundle, each a multi env of the site, wrapped as a website env, and chromium, generated from the dataset
WebArena-Verified 1.2.3 ships and graded offline by its own evaluator."""
from __future__ import annotations

from pathlib import Path

#: The bundle the tasks are written into, which ``agent-env run webarena-verified`` runs.
BUNDLE = Path(__file__).resolve().parent.parent / "bundles" / "webarena-verified"
