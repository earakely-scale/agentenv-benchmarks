"""Online-Mind2Web (osunlp/Online-Mind2Web, CC-BY-4.0): 300 live-web tasks on 147 sites, as an agent-env bundle,
each task once per arm, generated from the dataset's file at one pinned revision."""
from __future__ import annotations

from pathlib import Path

#: The bundle the tasks are written into, which ``agent-env run online-mind2web`` runs.
BUNDLE = Path(__file__).resolve().parent.parent / "bundles" / "online-mind2web"
