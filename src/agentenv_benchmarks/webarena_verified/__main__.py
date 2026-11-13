"""Write the WebArena-Verified bundle's tasks and evals from the dataset its pinned release ships:

    uv run --extra generate python -m agentenv_benchmarks.webarena_verified [--folder DATASET_FOLDER]

It finds the dataset in the release's package, which ``uvx`` installs in an environment of its own, unless
``--folder`` names a copy of its ``assets/dataset``; either way each file's sha256 must be the pinned one."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import BUNDLE, dataset, template


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folder", type=Path, help="a copy of the package's assets/dataset, instead of uvx's")
    args = parser.parse_args(argv)
    tasks = dataset.tasks(args.folder or dataset.locate())
    written = template.write(tasks, BUNDLE)
    print(f"wrote {len(written)} tasks and {len(list((BUNDLE / 'evals').glob('*.toml')))} evals into {BUNDLE}, "
          f"from {dataset.PACKAGE}")


if __name__ == "__main__":
    main(sys.argv[1:])
