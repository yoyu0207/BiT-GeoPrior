"""Run and aggregate the leakage-safe GWDA revision experiment matrix."""

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


def experiment_matrix() -> list[dict]:
    clean_prior = "spatial_prior_gwda_train_only"
    return [
        {"name": "FCSiamDiff", "model": "FCSiamDiff"},
        {"name": "SNUNet", "model": "SNUNet"},
        {"name": "ChangeFormer", "model": "ChangeFormer"},
        {"name": "BiT", "model": "BiT"},
        {"name": "BiT_GWDA", "model": "BiT_GWDA", "prior_dir": clean_prior},
        {"name": "BiT_GWDA_shuffled", "model": "BiT_GWDA", "prior_dir": clean_prior,
         "prior_control": "shuffled"},
        {"name": "BiT_GWDA_random", "model": "BiT_GWDA", "prior_dir": clean_prior,
         "prior_control": "random"},
        {"name": "OEP_BiT", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "OEP", "alpha": 0.0},
        {"name": "COAST", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "COAST", "alpha": 0.1},
        {"name": "COAST_shuffled", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "COAST_shuffled", "alpha": 0.1, "prior_control": "shuffled"},
        {"name": "COAST_random", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "COAST_random", "alpha": 0.1, "prior_control": "random"},
        {"name": "COAST_no_gating", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "COAST_no_gating", "alpha": 0.1, "no_gating": True},
        {"name": "COAST_alpha005_val", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "COAST_alpha005", "alpha": 0.05, "skip_test": True},
        {"name": "COAST_alpha020_val", "model": "BiT_Online", "prior_dir": clean_prior,
         "prior_tag": "COAST_alpha020", "alpha": 0.2, "skip_test": True},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("revision_experiments_gwda"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Optional experiment names to run.")
    return parser.parse_args()


def build_command(args, spec: dict, seed: int, run_name: str) -> list[str]:
    command = [
        str(args.python), "train.py", "--model", spec["model"],
        "--data_root", str(args.data_root),
        "--output_root", str(args.output_root),
        "--run_name", run_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed), "--amp",
    ]
    for key in ("prior_dir", "prior_tag", "prior_control"):
        if key in spec:
            command.extend([f"--{key}", str(spec[key])])
    if "alpha" in spec:
        command.extend(["--alpha", str(spec["alpha"])])
    if spec.get("no_gating"):
        command.append("--no_gating")
    if spec.get("skip_test"):
        command.append("--skip_test")
    return command


def run_all(args, specs: list[dict]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "experiment_matrix.json").write_text(
        json.dumps({"seeds": args.seeds, "experiments": specs}, indent=2),
        encoding="utf-8")
    for spec in specs:
        for seed in args.seeds:
            run_name = f'{spec["name"]}_seed{seed}'
            run_dir = args.output_root / run_name
            summary = run_dir / "summary.json"
            if summary.exists():
                print(f"[skip] {run_name}", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(args, spec, seed, run_name)
            (run_dir / "command.json").write_text(
                json.dumps(command, indent=2), encoding="utf-8")
            print(f"[run] {run_name}", flush=True)
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=Path(__file__).parent,
                    stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{run_name} failed; see {run_dir / 'console.log'}")
            aggregate(args.output_root, specs, args.seeds)


def bootstrap_ci(values: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, len(values), replace=True).mean() for _ in range(20000)
    ])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(output_root: Path, specs: list[dict], seeds: list[int]) -> None:
    rows = []
    for spec in specs:
        for seed in seeds:
            path = output_root / f'{spec["name"]}_seed{seed}' / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "experiment": spec["name"], "seed": seed,
                "best_val_f1": summary["best_val_f1"],
                "epochs_completed": summary.get("epochs_completed"),
            }
            for metric, value in (summary.get("test_metrics") or {}).items():
                row[f"test_{metric.lower()}"] = value
            rows.append(row)
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with (output_root / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for name in sorted({row["experiment"] for row in rows}):
        selected = [row for row in rows if row["experiment"] == name]
        result = {"experiment": name, "n": len(selected)}
        for key in columns:
            values = [row[key] for row in selected if key in row and isinstance(row[key], (int, float))]
            if key == "seed" or not values:
                continue
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate_rows.append(result)

    significance = {}
    by_name_seed = {(row["experiment"], row["seed"]): row for row in rows}
    if all(("COAST", seed) in by_name_seed for seed in seeds):
        for comparator in ("BiT", "BiT_GWDA", "OEP_BiT", "COAST_shuffled",
                           "COAST_random", "COAST_no_gating"):
            if not all((comparator, seed) in by_name_seed for seed in seeds):
                continue
            significance[comparator] = {}
            for metric in ("test_f1", "test_iou"):
                coast = np.asarray([by_name_seed[("COAST", seed)][metric] for seed in seeds])
                other = np.asarray([by_name_seed[(comparator, seed)][metric] for seed in seeds])
                difference = coast - other
                try:
                    wilcoxon_p = float(wilcoxon(difference).pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
                significance[comparator][metric] = {
                    "coast_minus_comparator_mean": float(difference.mean()),
                    "paired_bootstrap_95_ci": bootstrap_ci(difference),
                    "paired_t_p": float(ttest_rel(coast, other).pvalue),
                    "wilcoxon_p": wilcoxon_p,
                }
    payload = {"aggregate": aggregate_rows, "paired_significance": significance}
    (output_root / "aggregate_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = experiment_matrix()
    if args.only:
        requested = set(args.only)
        specs = [spec for spec in specs if spec["name"] in requested]
        missing = requested - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"Unknown experiment names: {sorted(missing)}")
    run_all(args, specs)
    aggregate(args.output_root, specs, args.seeds)


if __name__ == "__main__":
    main()
