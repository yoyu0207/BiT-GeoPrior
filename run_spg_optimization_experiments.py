"""Tune SPG learning rate on validation data, then rerun key GWDA controls."""

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
SPG_LR_CANDIDATES = [1e-4, 5e-4, 1e-3]
PRIOR_DIR = "spatial_prior_gwda_train_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path,
                        default=Path("spg_optimization_revision"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def run_command(command: list[str], run_dir: Path) -> None:
    summary = run_dir / "summary.json"
    if summary.exists():
        print(f"[skip] {run_dir.name}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8")
    print(f"[run] {run_dir.name}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=Path(__file__).parent,
            stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{run_dir.name} failed; see {run_dir / 'console.log'}")


def base_command(args: argparse.Namespace, run_name: str, seed: int,
                 spg_lr: float, skip_test: bool = False) -> list[str]:
    command = [
        str(args.python), "train.py", "--model", "BiT_Online",
        "--data_root", str(args.data_root),
        "--output_root", str(args.output_root),
        "--run_name", run_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed), "--amp",
        "--prior_dir", PRIOR_DIR,
        "--spg_lr", str(spg_lr),
        "--spg_gamma_lr", str(spg_lr),
    ]
    if skip_test:
        command.append("--skip_test")
    return command


def run_pilots(args: argparse.Namespace) -> float:
    pilot_rows = []
    for lr in SPG_LR_CANDIDATES:
        label = f"{lr:.0e}".replace("-", "m")
        run_name = f"SPG_lr_{label}_val_seed{PILOT_SEED}"
        command = base_command(args, run_name, PILOT_SEED, lr, skip_test=True)
        command.extend(["--prior_tag", f"SPG_lr_{label}", "--alpha", "0.1"])
        run_dir = args.output_root / run_name
        run_command(command, run_dir)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        pilot_rows.append({
            "spg_lr": lr,
            "seed": PILOT_SEED,
            "best_val_f1": summary["best_val_f1"],
            "gamma1": summary["best_checkpoint_gamma"]["spg1"],
            "gamma2": summary["best_checkpoint_gamma"]["spg2"],
            "max_abs_gamma": summary["best_checkpoint_gamma"]["max_abs"],
            "val_prior_mse": summary["val_prior_fidelity"]["mse"],
            "val_prior_pearson_r": summary["val_prior_fidelity"]["pearson_r"],
            "test_metrics_used": False,
        })
    selected = max(pilot_rows, key=lambda row: row["best_val_f1"])
    payload = {
        "selection_rule": "highest validation F1; test split was not loaded",
        "pilot_seed": PILOT_SEED,
        "candidates": pilot_rows,
        "selected_spg_lr": selected["spg_lr"],
    }
    (args.output_root / "spg_lr_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[selected] SPG LR={selected['spg_lr']:.1e}", flush=True)
    return selected["spg_lr"]


def key_matrix() -> list[dict]:
    return [
        {"name": "OEP_BiT_SPGopt", "prior_tag": "OEP_SPGopt", "alpha": 0.0},
        {"name": "COAST_SPGopt", "prior_tag": "COAST_SPGopt", "alpha": 0.1},
        {"name": "COAST_shuffled_SPGopt", "prior_tag": "COAST_shuffled_SPGopt",
         "alpha": 0.1, "prior_control": "shuffled"},
        {"name": "COAST_random_SPGopt", "prior_tag": "COAST_random_SPGopt",
         "alpha": 0.1, "prior_control": "random"},
        {"name": "COAST_no_gating_SPGopt", "prior_tag": "COAST_no_gating_SPGopt",
         "alpha": 0.1, "no_gating": True},
    ]


def run_key_matrix(args: argparse.Namespace, spg_lr: float) -> list[dict]:
    specs = key_matrix()
    matrix = {
        "seeds": SEEDS,
        "selected_spg_lr": spg_lr,
        "experiments": specs,
    }
    (args.output_root / "experiment_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8")
    for spec in specs:
        for seed in SEEDS:
            run_name = f"{spec['name']}_seed{seed}"
            command = base_command(args, run_name, seed, spg_lr)
            command.extend([
                "--prior_tag", spec["prior_tag"],
                "--alpha", str(spec["alpha"]),
            ])
            if spec.get("prior_control"):
                command.extend(["--prior_control", spec["prior_control"]])
            if spec.get("no_gating"):
                command.append("--no_gating")
            run_command(command, args.output_root / run_name)
            aggregate(args.output_root, specs)
    return specs


def bootstrap_ci(values: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, len(values), replace=True).mean() for _ in range(20000)
    ])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(output_root: Path, specs: list[dict]) -> None:
    rows = []
    for spec in specs:
        for seed in SEEDS:
            path = output_root / f"{spec['name']}_seed{seed}" / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            gamma = summary["best_checkpoint_gamma"]
            row = {
                "experiment": spec["name"],
                "seed": seed,
                "best_val_f1": summary["best_val_f1"],
                "epochs_completed": summary["epochs_completed"],
                "gamma1": gamma["spg1"],
                "gamma2": gamma["spg2"],
                "max_abs_gamma": gamma["max_abs"],
                "val_prior_mse": summary["val_prior_fidelity"]["mse"],
                "val_prior_pearson_r": summary["val_prior_fidelity"]["pearson_r"],
            }
            for metric, value in summary["test_metrics"].items():
                row[f"test_{metric.lower()}"] = value
            fidelity = summary.get("test_prior_fidelity") or {}
            row["test_prior_mse"] = fidelity.get("mse")
            row["test_prior_pearson_r"] = fidelity.get("pearson_r")
            rows.append(row)
    if not rows:
        return
    columns = list(rows[0])
    with (output_root / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for spec in specs:
        selected = [row for row in rows if row["experiment"] == spec["name"]]
        result = {"experiment": spec["name"], "n": len(selected)}
        for key in columns:
            if key in {"experiment", "seed"}:
                continue
            values = [row[key] for row in selected if isinstance(row.get(key), (int, float))]
            if values:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        aggregates.append(result)

    by_name_seed = {(row["experiment"], row["seed"]): row for row in rows}
    significance = {}
    target = "COAST_SPGopt"
    if all((target, seed) in by_name_seed for seed in SEEDS):
        for comparator in [spec["name"] for spec in specs if spec["name"] != target]:
            if not all((comparator, seed) in by_name_seed for seed in SEEDS):
                continue
            significance[comparator] = {}
            for metric in ("test_f1", "test_iou"):
                coast = np.asarray([by_name_seed[(target, seed)][metric] for seed in SEEDS])
                other = np.asarray([by_name_seed[(comparator, seed)][metric] for seed in SEEDS])
                difference = coast - other
                significance[comparator][metric] = {
                    "mean_paired_difference": float(difference.mean()),
                    "paired_bootstrap_95_ci": bootstrap_ci(difference),
                    "paired_t_p": float(ttest_rel(coast, other).pvalue),
                    "wilcoxon_p": float(wilcoxon(difference).pvalue),
                }
    (output_root / "aggregate_results.json").write_text(
        json.dumps({"aggregate": aggregates, "paired_significance": significance},
                   indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_lr = run_pilots(args)
    specs = run_key_matrix(args, selected_lr)
    aggregate(args.output_root, specs)


if __name__ == "__main__":
    main()
