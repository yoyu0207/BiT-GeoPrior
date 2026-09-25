"""Fetch the exact official revisions used by recent baselines."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SOURCES = {
    "rschange": {
        "repository": "https://github.com/xwmaxwma/rschange.git",
        "revision": "a3bc6ffc99d74c7a6a92a69f9471af2293dbff2a",
    },
    "EdgeRefNet": {
        "repository": "https://github.com/Wafaa-Hima/EdgeRefNet.git",
        "revision": "ba6f872fd077dccb6e53418d9e88c65839b4ded4",
    },
}


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "baseline_sources")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        destination = source_root / name
        revision = source["revision"]
        if not (destination / ".git").exists():
            run(["git", "clone", source["repository"], str(destination)])
        run(["git", "fetch", "origin", revision], cwd=destination)
        run(["git", "checkout", "--detach", revision], cwd=destination)
        actual = run(["git", "rev-parse", "HEAD"], cwd=destination)
        if actual != revision:
            raise RuntimeError(f"Expected {revision}, found {actual}")
        print(f"Prepared {name} at {destination} ({actual})")


if __name__ == "__main__":
    main()
