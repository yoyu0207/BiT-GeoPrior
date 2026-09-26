"""Run the final COAST prior-target controls with five paired seeds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


SEEDS = [42, 1337, 2025, 3407, 9001]
CONTROLS = [
    {"name": "COAST_zero_SPGopt", "prior_control": "zero"},
    {"name": "COAST_constant_SPGopt", "prior_control": "constant"},
    {
        "name": "COAST_random_per_epoch_SPGopt",
        "prior_control": "random_per_epoch",
    },
]
COAST_NAME = "COAST_SPGopt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument(
        "--output_root", type=Path,
        default=Path("prior_control_revision"),
    )
    parser.add_argument(
        "--coast_root", type=Path,
        default=Path("spg_optimization_revision"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--smoke_only", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def command_for(
    args: argparse.Namespace,
    spec: dict,
    seed: int,
    run_name: str,
    epochs: int | None = None,
) -> list[str]:
    return [
        str(args.python), "train.py", "--model", "BiT_Online",
        "--data_root", str(args.data_root),
        "--output_root", str(args.output_root),
        "--run_name", run_name,
        "--epochs", str(epochs or args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed),
        "--amp",
        "--prior_dir", "spatial_prior_gwda_train_only",
        "--spg_lr", "0.0001",
        "--spg_gamma_lr", "0.0001",
        "--prior_tag", spec["name"],
        "--alpha", "0.2",
        "--prior_control", spec["prior_control"],
    ]


def run_one(args: argparse.Namespace, spec: dict, seed: int) -> dict:
    run_name = f"{spec['name']}_seed{seed}"
    run_dir = args.output_root / run_name
    summary = run_dir / "summary.json"
    if summary.exists():
        return {"run": run_name, "status": "skipped_complete"}
    run_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(args, spec, seed, run_name)
    write_json(run_dir / "command.json", command)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0 or not summary.exists():
        return {
            "run": run_name,
            "status": "failed",
            "returncode": result.returncode,
            "log": str(run_dir / "console.log"),
        }
    return {"run": run_name, "status": "complete"}


def bootstrap_ci(values: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(20000)
    ])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def row_from_summary(experiment: str, seed: int, summary: dict) -> dict:
    gamma = summary["best_checkpoint_gamma"]
    row = {
        "experiment": experiment,
        "seed": seed,
        "best_val_f1": summary["best_val_f1"],
        "epochs_completed": summary["epochs_completed"],
        "gamma1": gamma["spg1"],
        "gamma2": gamma["spg2"],
        "max_abs_gamma": gamma["max_abs"],
    }
    fidelity = summary.get("test_prior_fidelity") or {}
    row["test_prior_mse"] = fidelity.get("mse")
    row["test_prior_pearson_r"] = fidelity.get("pearson_r")
    for metric, value in (summary.get("test_metrics") or {}).items():
        row[f"test_{metric.lower()}"] = value
    return row


def aggregate(args: argparse.Namespace) -> dict:
    rows = []
    for spec in CONTROLS:
        for seed in SEEDS:
            path = args.output_root / f"{spec['name']}_seed{seed}" / "summary.json"
            if path.exists():
                rows.append(row_from_summary(
                    spec["name"], seed,
                    json.loads(path.read_text(encoding="utf-8")),
                ))
    if rows:
        columns = list(rows[0])
        with (args.output_root / "all_runs.csv").open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    aggregates = []
    for spec in CONTROLS:
        selected = [r for r in rows if r["experiment"] == spec["name"]]
        result = {"experiment": spec["name"], "n": len(selected)}
        for key in (list(rows[0]) if rows else []):
            if key in {"experiment", "seed"}:
                continue
            values = [
                row[key] for row in selected
                if isinstance(row.get(key), (int, float))
            ]
            if values:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        aggregates.append(result)

    coast = {}
    for seed in SEEDS:
        path = args.coast_root / f"{COAST_NAME}_seed{seed}" / "summary.json"
        if path.exists():
            coast[seed] = row_from_summary(
                COAST_NAME, seed,
                json.loads(path.read_text(encoding="utf-8")),
            )

    significance = {}
    for spec in CONTROLS:
        control = {row["seed"]: row for row in rows
                   if row["experiment"] == spec["name"]}
        paired = [seed for seed in SEEDS if seed in coast and seed in control]
        if len(paired) != len(SEEDS):
            continue
        significance[spec["name"]] = {}
        for metric in ("test_f1", "test_iou"):
            reference = np.asarray([coast[s][metric] for s in paired])
            comparison = np.asarray([control[s][metric] for s in paired])
            difference = reference - comparison
            significance[spec["name"]][metric] = {
                "mean_paired_difference": float(difference.mean()),
                "paired_bootstrap_95_ci": bootstrap_ci(difference),
                "paired_t_p": float(ttest_rel(reference, comparison).pvalue),
                "wilcoxon_p": float(wilcoxon(difference).pvalue),
            }

    payload = {
        "aggregate": aggregates,
        "paired_significance_vs_COAST": significance,
    }
    write_json(args.output_root / "aggregate_results.json", payload)
    return payload


def smoke_test(args: argparse.Namespace) -> None:
    smoke_root = args.output_root / "_smoke"
    original_root = args.output_root
    args.output_root = smoke_root
    failures = []
    for spec in CONTROLS:
        run_name = f"smoke_{spec['prior_control']}"
        run_dir = smoke_root / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        command = command_for(args, spec, 42, run_name, epochs=1)
        command.extend(["--skip_test", "--patience", "0"])
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=Path(__file__).parent,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        if result.returncode != 0:
            failures.append({"control": spec["prior_control"], "log": str(log)})
    args.output_root = original_root
    if failures:
        raise RuntimeError(f"Control smoke test failed: {failures}")
    shutil.rmtree(smoke_root)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "pipeline_status.json"
    status = {
        "created_at": now(),
        "status": "running",
        "configuration": {
            "seeds": SEEDS,
            "controls": CONTROLS,
            "spg_lr": 1e-4,
            "alpha": 0.2,
            "model_lr": 5e-5,
        },
        "runs": [],
        "failures": [],
    }
    write_json(status_path, status)
    try:
        smoke_test(args)
        status["smoke_test"] = "complete"
        write_json(status_path, status)
        if args.smoke_only:
            status["status"] = "smoke_complete"
            status["completed_at"] = now()
            write_json(status_path, status)
            return
        for spec in CONTROLS:
            for seed in SEEDS:
                outcome = run_one(args, spec, seed)
                status["runs"].append(outcome)
                if outcome["status"] == "failed":
                    status["failures"].append(outcome)
                write_json(status_path, status)
                aggregate(args)
        payload = aggregate(args)
        counts = {item["experiment"]: item["n"] for item in payload["aggregate"]}
        status["counts"] = counts
        status["status"] = (
            "complete" if all(counts.get(spec["name"]) == len(SEEDS)
                              for spec in CONTROLS)
            else "partial"
        )
    except Exception as exc:
        status["status"] = "pipeline_failed"
        status["pipeline_error"] = repr(exc)
        raise
    finally:
        status["completed_at"] = now()
        write_json(status_path, status)


if __name__ == "__main__":
    main()
