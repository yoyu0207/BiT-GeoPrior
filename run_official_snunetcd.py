"""Validation-tune and run the pinned official SNUNet-CD over five seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


SEEDS = [42, 1337, 2025, 3407, 9001]
PILOT_SEED = 42
LR_CANDIDATES = [5e-5, 1e-4, 5e-4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("official_snunetcd_revision"))
    parser.add_argument("--coast-root", type=Path,
                        default=Path("spg_optimization_revision"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def command_for(args: argparse.Namespace, run_name: str, seed: int,
                lr: float, skip_test: bool = False) -> list[str]:
    command = [
        str(args.python), "train.py", "--model", "SNUNetCDOfficial",
        "--data_root", str(args.data_root),
        "--output_root", str(args.output_root),
        "--run_name", run_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed), "--lr", str(lr), "--amp",
        "--deterministic_warn_only",
        "--prior_dir", "no_prior_for_official_snunetcd",
    ]
    if skip_test:
        command.append("--skip_test")
    return command


def write_status(root: Path, payload: dict) -> None:
    (root / "pipeline_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: list[str], run_dir: Path,
                failures: list[dict]) -> bool:
    if (run_dir / "summary.json").exists():
        print(f"[skip] {run_dir.name}", flush=True)
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8")
    print(f"[run] {run_dir.name}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=Path(__file__).parent,
            stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode == 0:
        return True
    failure = {
        "run": run_dir.name,
        "return_code": result.returncode,
        "log": str(run_dir / "console.log"),
    }
    failures.append(failure)
    (run_dir / "failure.json").write_text(
        json.dumps(failure, indent=2), encoding="utf-8")
    print(f"[failed] {run_dir.name}; continuing", flush=True)
    return False


def select_lr(args: argparse.Namespace, failures: list[dict]) -> float | None:
    candidates = []
    for lr in LR_CANDIDATES:
        label = f"{lr:.0e}".replace("-", "m")
        run_name = f"SNUNetCDOfficial_lr_{label}_val_seed{PILOT_SEED}"
        run_dir = args.output_root / run_name
        if not run_command(
                command_for(args, run_name, PILOT_SEED, lr, skip_test=True),
                run_dir, failures):
            continue
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8"))
        candidates.append({
            "lr": lr,
            "best_val_f1": summary["best_val_f1"],
            "epochs_completed": summary["epochs_completed"],
            "test_metrics_used": False,
        })
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: row["best_val_f1"])
    (args.output_root / "lr_selection.json").write_text(
        json.dumps({
            "selection_rule": "highest validation F1; test split was not loaded",
            "pilot_seed": PILOT_SEED,
            "candidates": candidates,
            "selected_lr": selected["lr"],
        }, indent=2), encoding="utf-8")
    print(f"[selected] LR={selected['lr']:.1e}", flush=True)
    return selected["lr"]


def bootstrap_ci(values: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(20000)
    ])
    return [float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975))]


def aggregate(args: argparse.Namespace, selected_lr: float) -> None:
    rows = []
    for seed in SEEDS:
        path = args.output_root / f"SNUNetCDOfficial_seed{seed}" / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "experiment": "SNUNetCDOfficial",
            "seed": seed,
            "selected_lr": selected_lr,
            "best_val_f1": summary["best_val_f1"],
            "epochs_completed": summary["epochs_completed"],
        }
        for metric, value in summary["test_metrics"].items():
            row[f"test_{metric.lower()}"] = value
        rows.append(row)
    if not rows:
        return
    with (args.output_root / "all_runs.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"experiment": "SNUNetCDOfficial", "n": len(rows)}
    for key in rows[0]:
        if key in {"experiment", "seed"}:
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_sd"] = (
            float(values.std(ddof=1)) if len(values) > 1 else None)

    significance = {}
    if len(rows) == len(SEEDS):
        official = {row["seed"]: row for row in rows}
        coast = {}
        for seed in SEEDS:
            path = args.coast_root / f"COAST_SPGopt_seed{seed}" / "summary.json"
            if path.exists():
                coast[seed] = json.loads(path.read_text(encoding="utf-8"))
        if len(coast) == len(SEEDS):
            for metric, summary_key in (("f1", "F1"), ("iou", "IoU")):
                coast_values = np.asarray([
                    coast[seed]["test_metrics"][summary_key] for seed in SEEDS])
                official_values = np.asarray([
                    official[seed][f"test_{metric}"] for seed in SEEDS])
                difference = coast_values - official_values
                significance[f"test_{metric}"] = {
                    "mean_COAST_minus_official_SNUNetCD": float(difference.mean()),
                    "paired_bootstrap_95_ci": bootstrap_ci(difference),
                    "paired_t_p": float(ttest_rel(
                        coast_values, official_values).pvalue),
                    "wilcoxon_p": float(wilcoxon(difference).pvalue),
                }
    (args.output_root / "aggregate_results.json").write_text(
        json.dumps({
            "aggregate": result,
            "paired_significance_vs_COAST": significance,
        }, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.patience = 0
        args.output_root = args.output_root / "smoke"
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    write_status(args.output_root, {"status": "running", "failures": failures})
    selected_lr = select_lr(args, failures)
    if selected_lr is None:
        write_status(args.output_root, {
            "status": "failed_before_formal_runs", "failures": failures})
        raise SystemExit(1)
    if args.smoke:
        write_status(args.output_root, {
            "status": "smoke_complete", "selected_lr": selected_lr,
            "failures": failures})
        return
    for seed in SEEDS:
        run_name = f"SNUNetCDOfficial_seed{seed}"
        run_command(
            command_for(args, run_name, seed, selected_lr),
            args.output_root / run_name, failures)
        aggregate(args, selected_lr)
        write_status(args.output_root, {
            "status": "running", "selected_lr": selected_lr,
            "completed_seeds": [
                candidate for candidate in SEEDS
                if (args.output_root / f"SNUNetCDOfficial_seed{candidate}"
                    / "summary.json").exists()],
            "failures": failures,
        })
    aggregate(args, selected_lr)
    completed = [
        seed for seed in SEEDS
        if (args.output_root / f"SNUNetCDOfficial_seed{seed}"
            / "summary.json").exists()]
    write_status(args.output_root, {
        "status": "complete" if len(completed) == len(SEEDS)
        else "complete_with_failures",
        "selected_lr": selected_lr,
        "completed_seeds": completed,
        "failures": failures,
    })


if __name__ == "__main__":
    main()
