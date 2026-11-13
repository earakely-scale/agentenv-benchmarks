"""WebArena-Verified's own prompt for the shopping admin site, copied from the release the bundle is generated from and
checked against its sha256 there, rendered for each task with its intent and its start URLs as the site serves them,
by Jinja as WebArena-Verified's own agents render it (``examples/agents/utils.py``: no autoescape, the trailing newline
kept). Only generating the bundle renders it, which the ``generate`` extra's jinja2 does."""
from __future__ import annotations

import hashlib
from pathlib import Path

#: The template, as ``examples/prompts/shopping_admin.md`` holds it at the release's tag.
SOURCE = "https://github.com/ServiceNow/webarena-verified/blob/v1.2.3/examples/prompts/shopping_admin.md"
FILE = Path(__file__).resolve().parent / "prompts" / "shopping_admin.md"
SHA256 = "2a2822161e8065e5d67ba106db6b0f9de9e1aa45663dd8bde8d91d755bf2f1c9"


def template() -> str:
    """The template, refused when its bytes aren't the pinned ones."""
    data = FILE.read_bytes()
    if (found := hashlib.sha256(data).hexdigest()) != SHA256:
        raise ValueError(f"{FILE} has sha256 {found}, not {SOURCE}'s {SHA256}")
    return data.decode("utf-8")


def render(intent: str, start_urls: list[str]) -> str:
    """The prompt for a task: the template rendered with its ``INTENT`` and ``START_URLS``."""
    from jinja2 import Environment, StrictUndefined

    environment = Environment(autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined)
    return environment.from_string(template()).render(INTENT=intent, START_URLS=start_urls)
