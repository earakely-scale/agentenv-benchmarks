"""Online-Mind2Web's tasks at the one revision the bundle is generated from: the dataset's file, checked against its
pinned sha256, so a revision's results are never mixed with another's."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO = "osunlp/Online-Mind2Web"
REVISION = "eacad896a84dc5b65e29b0b06e4699ab0544d701"
FILE = "Online_Mind2Web.json"
SHA256 = "7dedf381531d423dc0fc48d21dc1d425655dd75f38b10211a880fa09102f257f"
LEVELS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Task:
    """One of the dataset's tasks, as its file gives it."""

    task_id: str
    task: str
    website: str
    level: str
    reference_length: int

    @classmethod
    def from_row(cls, row: dict) -> Task:
        return cls(task_id=row["task_id"], task=row["confirmed_task"], website=row["website"], level=row["level"],
                   reference_length=row["reference_length"])


def download() -> Path:
    """The dataset's file at REVISION, through the Hugging Face Hub, which asks for a login that has accepted the
    dataset's terms; refused when its bytes aren't the ones pinned."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION))
    if (found := hashlib.sha256(path.read_bytes()).hexdigest()) != SHA256:
        raise ValueError(f"{REPO}@{REVISION}/{FILE} has sha256 {found}, not the pinned {SHA256}")
    return path


def tasks(path: Path) -> list[Task]:
    """The dataset's tasks, from its file at ``path``, in the file's order."""
    return [Task.from_row(row) for row in json.loads(path.read_text(encoding="utf-8"))]
