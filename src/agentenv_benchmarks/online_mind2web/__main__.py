"""Write the Online-Mind2Web bundle's tasks and evals from the dataset at its pinned revision:

    uv run --extra generate python -m agentenv_benchmarks.online_mind2web [--file Online_Mind2Web.json]

It downloads the file through the Hugging Face Hub, which needs a login that has accepted the dataset's terms
(``hf auth login``), unless ``--file`` names a copy; either way the file's sha256 must be the pinned one."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from . import BUNDLE, dataset, template


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file", type=Path, help="a copy of the dataset's file, instead of downloading it")
    args = parser.parse_args(argv)
    path = args.file or dataset.download()
    if (found := hashlib.sha256(path.read_bytes()).hexdigest()) != dataset.SHA256:
        raise SystemExit(f"{path} has sha256 {found}, not {dataset.REPO}@{dataset.REVISION}'s {dataset.SHA256}")
    written = template.write(dataset.tasks(path), BUNDLE)
    print(f"wrote {len(written)} tasks and {len(list((BUNDLE / 'evals').glob('*.toml')))} evals into {BUNDLE}, "
          f"from {dataset.REPO}@{dataset.REVISION[:8]}")


if __name__ == "__main__":
    main(sys.argv[1:])
