"""Tune and run the formal models against spatial-block OOF GWDA priors.

The runner never overwrites legacy results.  Alpha selection uses validation
metrics only and passes --skip_test.  Individual failed runs are recorded and
do not prevent the remaining experiment matrix from continuing.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


SEEDS = [42, 1337, 2025, 3407, 9001]
ALPHAS = [0.05, 0.10, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("oof_gwda_revision"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def command_for(
    args: argparse.Namespace,
    model: str,
    run_name: str,
    seed: int,
    *,
    alpha: float | None = None,
    skip_test: bool = False,
    prior_control: str | None = None,
    no_gating: bool = False,
) -> list[str]:
    command = [
        str(args.python),
        "train.py",
        "--model",
        model,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--run_name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(seed),
        "--amp",
        "--prior_dir",
        "spatial_prior_gwda_oof",
        "--lr",
        "0.00005",
    ]
    if model == "BiT_Online":
        command.extend([
            "--prior_tag",
            "COAST_OOF",
            "--alpha",
            str(alpha),
            "--spg_lr",
            "0.0001",
            "--spg_gamma_lr",
            "0.0001",
        ])
    if skip_test:
        command.append("--skip_test")
    if prior_control:
        command.extend(["--prior_control", prior_control])
    if no_gating:
        command.append("--no_gating")
    return command


def run_one(args: argparse.Namespace, command: list[str], run_name: str) -> dict:
    run_dir = args.output_root / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return {"run": run_name, "status": "completed", "resumed": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "command.json", command)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode == 0 and summary_path.exists():
        return {"run": run_name, "status": "completed", "returncode": 0}
    failure = {
        "run": run_name,
        "status": "failed",
        "returncode": result.returncode,
        "log": str(run_dir / "console.log"),
    }
    write_json(run_dir / "failure.json", failure)
    return failure


def mean_sd(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
    }


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(args: argparse.Namespace, selected_alpha: float, failures: list[dict]) -> None:
    rows: list[dict] = []
    for experiment in ("COAST_OOF", "BiT_GWDA_OOF"):
        for seed in SEEDS:
            summary = load_summary(args.output_root / f"{experiment}_seed{seed}" / "summary.json")
            if summary is None:
                continue
            row = {
                "experiment": experiment,
                "seed": seed,
                "best_val_f1": summary["best_val_f1"],
                "epochs_completed": summary.get("epochs_completed"),
            }
            for metric, value in (summary.get("test_metrics") or {}).items():
                row[f"test_{metric.lower()}"] = value
            rows.append(row)
    columns = sorted({key for row in rows for key in row})
    if rows:
        with (args.output_root / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    aggregate_rows = []
    for name in ("COAST_OOF", "BiT_GWDA_OOF"):
        selected = [row for row in rows if row["experiment"] == name]
        result: dict = {"experiment": name, "n": len(selected)}
        for metric in ("best_val_f1", "test_f1", "test_iou", "test_precision", "test_recall"):
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                result[metric] = mean_sd(values)
        aggregate_rows.append(result)

    legacy_locations = {
        "COAST_OOF": (Path("spg_optimization_revision"), "COAST_SPGopt_seed{seed}"),
        "BiT_GWDA_OOF": (Path("revision_experiments_gwda"), "BiT_GWDA_seed{seed}"),
    }
    comparisons = {}
    current_by_key = {(row["experiment"], row["seed"]): row for row in rows}
    for experiment, (legacy_root, pattern) in legacy_locations.items():
        comparisons[experiment] = {}
        for metric in ("test_f1", "test_iou"):
            new_values, old_values, paired_seeds = [], [], []
            for seed in SEEDS:
                current = current_by_key.get((experiment, seed))
                legacy = load_summary(legacy_root / pattern.format(seed=seed) / "summary.json")
                if current is None or legacy is None or metric not in current:
                    continue
                legacy_metric = {"test_f1": "F1", "test_iou": "IoU"}[metric]
                old_metric = (legacy.get("test_metrics") or {}).get(legacy_metric)
                if old_metric is None:
                    continue
                new_values.append(float(current[metric]))
                old_values.append(float(old_metric))
                paired_seeds.append(seed)
            if new_values:
                difference = np.asarray(new_values) - np.asarray(old_values)
                try:
                    wilcoxon_p = float(wilcoxon(difference).pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
                comparisons[experiment][metric] = {
                    "paired_seeds": paired_seeds,
                    "oof_minus_legacy_mean": float(difference.mean()),
                    "paired_t_p": float(ttest_rel(new_values, old_values).pvalue)
                    if len(new_values) > 1 else None,
                    "wilcoxon_p": wilcoxon_p if len(new_values) > 1 else None,
                }

    write_json(
        args.output_root / "aggregate_results.json",
        {
            "selected_alpha_validation_only": selected_alpha,
            "aggregate": aggregate_rows,
            "paired_comparison_with_legacy": comparisons,
            "failures": failures,
        },
    )


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    prior_metadata = args.data_root / "spatial_prior_gwda_oof" / "gwda_metadata.json"
    if not prior_metadata.exists():
        raise FileNotFoundError(f"OOF GWDA metadata not found: {prior_metadata}")
    started = time.time()
    failures: list[dict] = []
    status_path = args.output_root / "pipeline_status.json"

    write_json(status_path, {"status": "running", "stage": "alpha_selection"})
    alpha_scores = {}
    for alpha in ALPHAS:
        tag = str(alpha).replace(".", "p")
        run_name = f"alpha_{tag}_val_seed42"
        command = command_for(args, "BiT_Online", run_name, 42, alpha=alpha, skip_test=True)
        result = run_one(args, command, run_name)
        if result["status"] == "failed":
            failures.append(result)
            continue
        summary = load_summary(args.output_root / run_name / "summary.json")
        alpha_scores[str(alpha)] = float(summary["best_val_f1"])
        write_json(status_path, {
            "status": "running",
            "stage": "alpha_selection",
            "alpha_scores": alpha_scores,
            "failures": failures,
        })
    if not alpha_scores:
        raise RuntimeError("All validation-only alpha runs failed")
    selected_alpha = float(max(alpha_scores, key=alpha_scores.get))
    write_json(args.output_root / "alpha_selection.json", {
        "criterion": "highest validation F1; test set not loaded",
        "scores": alpha_scores,
        "selected_alpha": selected_alpha,
    })

    write_json(status_path, {
        "status": "running",
        "stage": "formal_training",
        "selected_alpha": selected_alpha,
        "failures": failures,
    })
    for experiment, model in (("COAST_OOF", "BiT_Online"), ("BiT_GWDA_OOF", "BiT_GWDA")):
        for seed in SEEDS:
            run_name = f"{experiment}_seed{seed}"
            command = command_for(
                args,
                model,
                run_name,
                seed,
                alpha=selected_alpha if model == "BiT_Online" else None,
            )
            result = run_one(args, command, run_name)
            if result["status"] == "failed":
                failures.append(result)
            write_json(status_path, {
                "status": "running",
                "stage": "formal_training",
                "selected_alpha": selected_alpha,
                "last_run": result,
                "failures": failures,
            })
            aggregate(args, selected_alpha, failures)

    aggregate(args, selected_alpha, failures)
    completed = sum(
        (args.output_root / f"{experiment}_seed{seed}" / "summary.json").exists()
        for experiment in ("COAST_OOF", "BiT_GWDA_OOF")
        for seed in SEEDS
    )
    write_json(status_path, {
        "status": "completed" if completed == 10 else "completed_with_failures",
        "stage": "completed",
        "selected_alpha": selected_alpha,
        "formal_runs_completed": completed,
        "formal_runs_expected": 10,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
