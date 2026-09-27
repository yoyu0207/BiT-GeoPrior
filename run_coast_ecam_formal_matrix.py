"""Run the confirmed three-seed COAST ECAM/control experiment matrix."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"D:\develop\miniconda3\envs\geo\python.exe")
DATA_ROOT = Path(r"D:\yoyu\SA_Identification\dataset_patches_2020_2024")
OUTPUT_ROOT = ROOT / "coast_ecam_formal_revision"
SEEDS = (42, 1337, 3407)

COMMON = [
    "--data_root", str(DATA_ROOT),
    "--output_root", str(OUTPUT_ROOT),
    "--epochs", "200",
    "--patience", "0",
    "--batch_size", "8",
    "--amp",
]

COAST_COMMON = [
    "--model", "BiT_Online_Boundary",
    "--lr", "0.00005",
    "--prior_dir", "spatial_prior_gwda_oof",
    "--alpha", "0.1",
    "--spg_lr", "0.0001",
    "--spg_gamma_lr", "0.0001",
    "--boundary_weight", "0.2",
]

EXPERIMENTS = (
    ("COAST_ECAM", COAST_COMMON),
    ("COAST_ECAM_shuffled", COAST_COMMON + ["--prior_control", "shuffled"]),
    ("COAST_ECAM_random_per_epoch", COAST_COMMON + [
        "--prior_control", "random_per_epoch"]),
    ("COAST_ECAM_prior_only_gating", [
        *COAST_COMMON[:-8],
        "--alpha", "0",
        "--spg_lr", "0.0001",
        "--spg_gamma_lr", "0.0001",
        "--boundary_weight", "0.2",
        "--prior_only_gating",
    ]),
    ("COAST_ECAM_no_gating", COAST_COMMON + ["--no_gating"]),
    ("COAST_ECAM_zero", COAST_COMMON + ["--prior_control", "zero"]),
    ("COAST_ECAM_constant", COAST_COMMON + ["--prior_control", "constant"]),
    ("SNUNetCDLiteNoECAM", [
        "--model", "SNUNetCDLiteNoECAM",
        "--lr", "0.0005",
    ]),
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def completed_summary(run_dir: Path) -> dict | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if summary.get("epochs_completed") != 200:
        return None
    if not summary.get("test_metrics"):
        return None
    return summary


def safe_reset(run_dir: Path) -> None:
    resolved_root = OUTPUT_ROOT.resolve()
    resolved_run = run_dir.resolve()
    if resolved_run.parent != resolved_root:
        raise RuntimeError(f"Refusing to reset unexpected path: {resolved_run}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def update_status(payload: dict) -> None:
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    write_json(OUTPUT_ROOT / "pipeline_status.json", payload)


def collect_rows() -> list[dict]:
    rows = []
    for experiment, _ in EXPERIMENTS:
        for seed in SEEDS:
            run_name = f"{experiment}_seed{seed}"
            summary = completed_summary(OUTPUT_ROOT / run_name)
            if summary is None:
                continue
            metrics = summary["test_metrics"]
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "best_val_f1": summary["best_val_f1"],
                "epochs_completed": summary["epochs_completed"],
                "test_precision": metrics["Precision"],
                "test_recall": metrics["Recall"],
                "test_f1": metrics["F1"],
                "test_iou": metrics["IoU"],
                "test_oa": metrics["OA"],
            })
    return rows


def aggregate() -> None:
    rows = collect_rows()
    fields = [
        "experiment", "seed", "best_val_f1", "epochs_completed",
        "test_precision", "test_recall", "test_f1", "test_iou", "test_oa",
    ]
    with (OUTPUT_ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    metric_names = [
        "best_val_f1", "test_precision", "test_recall",
        "test_f1", "test_iou", "test_oa",
    ]
    for experiment, _ in EXPERIMENTS:
        subset = [row for row in rows if row["experiment"] == experiment]
        if not subset:
            continue
        item = {"n": len(subset), "seeds": [row["seed"] for row in subset]}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None)
        grouped[experiment] = item
    write_json(OUTPUT_ROOT / "aggregate_results.json", grouped)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    failures_path = OUTPUT_ROOT / "failures.json"
    failures = (
        json.loads(failures_path.read_text(encoding="utf-8"))
        if failures_path.exists() else []
    )
    total = len(EXPERIMENTS) * len(SEEDS)
    index = 0
    for experiment, experiment_args in EXPERIMENTS:
        for seed in SEEDS:
            index += 1
            run_name = f"{experiment}_seed{seed}"
            run_dir = OUTPUT_ROOT / run_name
            if completed_summary(run_dir) is not None:
                aggregate()
                update_status({
                    "status": "running", "stage": "training",
                    "current": run_name, "progress": f"{index}/{total}",
                    "action": "skipped_completed",
                })
                continue

            safe_reset(run_dir)
            command = [
                str(PYTHON), "train.py", *COMMON,
                "--run_name", run_name,
                "--seed", str(seed),
                *experiment_args,
            ]
            write_json(run_dir / "command.json", command)
            update_status({
                "status": "running", "stage": "training",
                "current": run_name, "progress": f"{index}/{total}",
                "command": command,
            })
            started = time.time()
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode != 0 or completed_summary(run_dir) is None:
                failure = {
                    "experiment": experiment,
                    "seed": seed,
                    "run": run_name,
                    "returncode": result.returncode,
                    "console_log": str(run_dir / "console.log"),
                    "elapsed_seconds": time.time() - started,
                }
                write_json(run_dir / "failure.json", failure)
                failures.append(failure)
                write_json(failures_path, failures)
            aggregate()

    aggregate()
    completed = len(collect_rows())
    update_status({
        "status": "completed_with_failures" if failures else "completed",
        "stage": "completed",
        "completed_runs": completed,
        "expected_runs": total,
        "failure_count": len(failures),
        "failures": failures,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
