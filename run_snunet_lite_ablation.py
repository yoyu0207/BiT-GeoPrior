"""Queue one SNUNet-CD lite/no-ECAM ablation after OOF controls finish."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--wait_status", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--poll_seconds", type=int, default=60)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "pipeline_status.json"
    write_json(status_path, {"status": "waiting", "stage": "oof_controls"})
    while True:
        upstream = (
            json.loads(args.wait_status.read_text(encoding="utf-8"))
            if args.wait_status.exists() else {}
        )
        if upstream.get("stage") == "completed":
            break
        time.sleep(args.poll_seconds)

    run_name = "SNUNetCDLiteNoECAM_seed42"
    run_dir = args.output_root / run_name
    run_dir.mkdir(exist_ok=True)
    command = [
        str(args.python), "train.py",
        "--model", "SNUNetCDLiteNoECAM",
        "--data_root", str(args.data_root),
        "--output_root", str(args.output_root),
        "--run_name", run_name,
        "--lr", "0.0005",
        "--epochs", "200",
        "--patience", "30",
        "--batch_size", "8",
        "--seed", "42",
        "--amp",
    ]
    write_json(run_dir / "command.json", command)
    write_json(status_path, {"status": "running", "stage": "training", "run": run_name})
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    summary = run_dir / "summary.json"
    final_status = "completed" if result.returncode == 0 and summary.exists() else "failed"
    write_json(status_path, {
        "status": final_status,
        "stage": "completed" if final_status == "completed" else "failed",
        "run": run_name,
        "returncode": result.returncode,
        "summary": str(summary),
    })


if __name__ == "__main__":
    main()
