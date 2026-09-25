"""Validation-tune 2025/2026 baselines and run each over five seeds."""

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
MODELS = ["STeInFormer", "EdgeRefNet"]
LR_CANDIDATES = [5e-5, 1e-4, 5e-4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path,
                        default=Path("recent_baseline_revision"))
    parser.add_argument("--spg_output_root", type=Path,
                        default=Path("spg_optimization_revision"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def run_command(command: list[str], run_dir: Path) -> None:
    if (run_dir / "summary.json").exists():
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


def command_for(args: argparse.Namespace, model: str, run_name: str,
                seed: int, lr: float, skip_test: bool = False) -> list[str]:
    command = [
        str(args.python), "train.py", "--model", model,
        "--data_root", str(args.data_root),
        "--output_root", str(args.output_root),
        "--run_name", run_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed),
        "--lr", str(lr),
        "--amp",
        "--deterministic_warn_only",
        "--prior_dir", "no_prior_for_recent_baselines",
    ]
    if skip_test:
        command.append("--skip_test")
    return command


def select_learning_rates(args: argparse.Namespace) -> dict[str, float]:
    selections = {}
    payload = {
        "selection_rule": "highest validation F1; test split was not loaded",
        "pilot_seed": PILOT_SEED,
        "models": {},
    }
    for model in MODELS:
        candidates = []
        for lr in LR_CANDIDATES:
            label = f"{lr:.0e}".replace("-", "m")
            run_name = f"{model}_lr_{label}_val_seed{PILOT_SEED}"
            run_dir = args.output_root / run_name
            run_command(
                command_for(
                    args, model, run_name, PILOT_SEED, lr, skip_test=True),
                run_dir,
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8"))
            candidates.append({
                "lr": lr,
                "best_val_f1": summary["best_val_f1"],
                "epochs_completed": summary["epochs_completed"],
                "test_metrics_used": False,
            })
        selected = max(candidates, key=lambda row: row["best_val_f1"])
        selections[model] = selected["lr"]
        payload["models"][model] = {
            "candidates": candidates,
            "selected_lr": selected["lr"],
        }
        print(f"[selected] {model} LR={selected['lr']:.1e}", flush=True)
    (args.output_root / "lr_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return selections


def run_formal(args: argparse.Namespace,
               learning_rates: dict[str, float]) -> None:
    matrix = {
        "seeds": SEEDS,
        "models": MODELS,
        "selected_learning_rates": learning_rates,
        "spatial_split": "spatial_split_manifest.csv",
        "gwda_used": False,
        "sources": {
            "STeInFormer": {
                "year": 2025,
                "journal": "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing",
                "repository": "xwmaxwma/rschange",
                "revision": "a3bc6ffc99d74c7a6a92a69f9471af2293dbff2a",
            },
            "EdgeRefNet": {
                "year": 2026,
                "journal": "IEEE Transactions on Geoscience and Remote Sensing",
                "repository": "Wafaa-Hima/EdgeRefNet",
                "revision": "ba6f872fd077dccb6e53418d9e88c65839b4ded4",
            },
        },
        "input_adaptation": (
            "All recent baselines were initialized from scratch to match "
            "COAST; first convolutions accept the same eight-channel input"
        ),
        "EdgeRefNet_compatibility_fixes": [
            "remove one accidental leading space before the first import",
            "use the actual 16x16 fourth-stage feature size for 256x256 inputs",
            "upsample the fourth-stage edge feature by 8x to match the first stage",
            "remove an unused malformed model-factory block and debug prints",
        ],
    }
    (args.output_root / "experiment_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8")
    for model in MODELS:
        for seed in SEEDS:
            run_name = f"{model}_seed{seed}"
            run_command(
                command_for(
                    args, model, run_name, seed, learning_rates[model]),
                args.output_root / run_name,
            )
            aggregate(args)


def _bootstrap_ci(values: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(20000)
    ])
    return [float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975))]


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            path = args.output_root / f"{model}_seed{seed}" / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "experiment": model,
                "seed": seed,
                "best_val_f1": summary["best_val_f1"],
                "epochs_completed": summary["epochs_completed"],
            }
            for metric, value in summary["test_metrics"].items():
                row[f"test_{metric.lower()}"] = value
            rows.append(row)
    if not rows:
        return
    columns = list(rows[0])
    with (args.output_root / "all_runs.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for model in MODELS:
        selected = [row for row in rows if row["experiment"] == model]
        result = {"experiment": model, "n": len(selected)}
        for key in columns:
            if key in {"experiment", "seed"}:
                continue
            values = [row[key] for row in selected]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None)
        aggregates.append(result)

    significance = {}
    coast = {}
    for seed in SEEDS:
        path = args.spg_output_root / f"COAST_SPGopt_seed{seed}" / "summary.json"
        if path.exists():
            coast[seed] = json.loads(path.read_text(encoding="utf-8"))
    if len(coast) == len(SEEDS):
        by_key = {(row["experiment"], row["seed"]): row for row in rows}
        for model in MODELS:
            if not all((model, seed) in by_key for seed in SEEDS):
                continue
            significance[model] = {}
            for metric in ("f1", "iou"):
                coast_values = np.asarray([
                    coast[seed]["test_metrics"][metric.upper()]
                    for seed in SEEDS
                ])
                baseline_values = np.asarray([
                    by_key[(model, seed)][f"test_{metric}"]
                    for seed in SEEDS
                ])
                difference = coast_values - baseline_values
                significance[model][f"test_{metric}"] = {
                    "mean_paired_difference": float(difference.mean()),
                    "paired_bootstrap_95_ci": _bootstrap_ci(difference),
                    "paired_t_p": float(ttest_rel(
                        coast_values, baseline_values).pvalue),
                    "wilcoxon_p": float(wilcoxon(difference).pvalue),
                }
    (args.output_root / "aggregate_results.json").write_text(
        json.dumps({
            "aggregate": aggregates,
            "paired_significance_vs_COAST": significance,
        }, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    learning_rates = select_learning_rates(args)
    run_formal(args, learning_rates)
    aggregate(args)


if __name__ == "__main__":
    main()
