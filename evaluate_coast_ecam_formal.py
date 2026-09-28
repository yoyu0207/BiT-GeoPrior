"""Evaluate the completed COAST-ECAM formal matrix from frozen checkpoints."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from dataset import CDDataset
from evaluate_application_metrics import (
    REPRODUCTION_TOLERANCE,
    benchmark_efficiency,
    calibration_metrics,
    collect_predictions,
    evaluate_threshold,
    load_manifest,
    region_metrics,
    select_validation_threshold,
    write_csv,
)
from train import build_model, set_global_seed


PROJECT = Path(__file__).resolve().parent
RUN_ROOT = PROJECT / "coast_ecam_formal_revision"
OUTPUT_ROOT = PROJECT / "results/revision_2026/coast_ecam_formal"
DATA_ROOT = Path(r"D:\yoyu\SA_Identification\dataset_patches_2020_2024")
SEEDS = (42, 1337, 3407)
EXPERIMENTS = (
    "COAST_ECAM",
    "COAST_ECAM_shuffled",
    "COAST_ECAM_random_per_epoch",
    "COAST_ECAM_prior_only_gating",
    "COAST_ECAM_no_gating",
    "COAST_ECAM_zero",
    "COAST_ECAM_constant",
    "SNUNetCDLiteNoECAM",
)
METRICS = (
    "precision", "recall", "f1", "iou", "oa", "boundary_f1",
    "patch_sampled_area_relative_bias", "patch_area_fraction_mae_pp",
)


def add_absolute_area_error(metrics: dict) -> dict:
    metrics["patch_sampled_area_absolute_relative_error"] = abs(
        metrics["patch_sampled_area_relative_bias"])
    return metrics


def mean_sd(values: list[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), (float(array.std(ddof=1)) if len(array) > 1 else None)


def flatten_result(result: dict) -> dict:
    row = {
        "experiment": result["experiment"],
        "seed": result["seed"],
        "validation_selected_threshold": result["validation_selected_threshold"],
        "pr_auc": result["pr_auc"],
        **result["calibration"],
        "reproduction_f1_difference": result["reproduction_check"]["f1_difference"],
        "reproduction_iou_difference": result["reproduction_check"]["iou_difference"],
        "reproduction_passes": result["reproduction_check"]["passes_tolerance"],
    }
    for prefix, block in (("fixed_0p5", result["fixed_0p5"]),
                          ("val_threshold", result["validation_threshold_test"])):
        for key, value in block.items():
            if isinstance(value, (int, float)):
                row[f"{prefix}_{key}"] = value
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for experiment in EXPERIMENTS:
        selected = [row for row in rows if row["experiment"] == experiment]
        if not selected:
            continue
        result = {"experiment": experiment, "n": len(selected),
                  "seeds": ",".join(str(row["seed"]) for row in selected)}
        numeric = [key for key, value in selected[0].items()
                   if key not in {"experiment", "seed", "reproduction_passes"}
                   and isinstance(value, (int, float))]
        for key in numeric:
            result[f"{key}_mean"], result[f"{key}_sd"] = mean_sd(
                [float(row[key]) for row in selected])
        output.append(result)
    return output


def paired_tests(rows: list[dict]) -> list[dict]:
    lookup = {(row["experiment"], int(row["seed"])): row for row in rows}
    output = []
    specs = {
        "fixed_0p5_f1": True,
        "fixed_0p5_iou": True,
        "fixed_0p5_boundary_f1": True,
        "fixed_0p5_patch_area_fraction_mae_pp": False,
        "fixed_0p5_patch_sampled_area_relative_bias": False,
        "pr_auc": True,
    }
    for comparator in EXPERIMENTS[1:]:
        common = [seed for seed in SEEDS
                  if ("COAST_ECAM", seed) in lookup and (comparator, seed) in lookup]
        for metric, higher_is_better in specs.items():
            target = np.asarray([lookup[("COAST_ECAM", seed)][metric] for seed in common])
            other = np.asarray([lookup[(comparator, seed)][metric] for seed in common])
            difference = target - other
            try:
                wilcoxon_p = float(wilcoxon(difference).pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            output.append({
                "target": "COAST_ECAM", "comparator": comparator,
                "metric": metric, "n": len(common),
                "higher_is_better": higher_is_better,
                "target_mean": float(target.mean()),
                "comparator_mean": float(other.mean()),
                "target_minus_comparator": float(difference.mean()),
                "oriented_advantage": float(
                    difference.mean() if higher_is_better else -difference.mean()),
                "paired_t_p": float(ttest_rel(target, other).pvalue),
                "wilcoxon_p": wilcoxon_p,
            })
    return output


def spatial_block_bootstrap(region_rows: list[dict], iterations: int = 10000) -> list[dict]:
    """Resample held-out blocks, aggregate pixel counts, then recompute metrics."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in region_rows:
        if row["group_type"] == "spatial_block":
            grouped[(row["experiment"], row["group_name"])].append(row)
    rng = np.random.default_rng(20260928)
    output = []
    for experiment in EXPERIMENTS:
        names = sorted(name for exp, name in grouped if exp == experiment)
        if not names:
            continue
        indices = rng.integers(0, len(names), size=(iterations, len(names)))
        samples = defaultdict(list)
        for sampled_indices in indices:
            sampled_names = [names[index] for index in sampled_indices]
            seed_metrics = defaultdict(list)
            for seed in SEEDS:
                selected = []
                for name in sampled_names:
                    matches = [row for row in grouped[(experiment, name)]
                               if int(row["seed"]) == seed]
                    if matches:
                        selected.append(matches[0])
                tp = sum(float(row["tp"]) for row in selected)
                fp = sum(float(row["fp"]) for row in selected)
                fn = sum(float(row["fn"]) for row in selected)
                f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
                iou = tp / max(tp + fp + fn, 1.0)
                relative_area_error = abs((tp + fp) - (tp + fn)) / max(tp + fn, 1.0)
                patch_count = sum(float(row["n_patches"]) for row in selected)
                area_mae = sum(
                    float(row["patch_area_fraction_mae_pp"])
                    * float(row["n_patches"]) for row in selected
                ) / max(patch_count, 1.0)
                seed_metrics["f1"].append(f1)
                seed_metrics["iou"].append(iou)
                seed_metrics["absolute_relative_area_error"].append(
                    relative_area_error)
                seed_metrics["patch_area_fraction_mae_pp"].append(area_mae)
            for metric, values in seed_metrics.items():
                samples[metric].append(float(np.mean(values)))
        point_rows = [row for name in names for row in grouped[(experiment, name)]]
        for metric, values in samples.items():
            boot = np.asarray(values, dtype=float)
            if metric == "f1":
                point = np.mean([
                    2 * sum(float(row["tp"]) for row in point_rows if int(row["seed"]) == seed)
                    / max(sum(2 * float(row["tp"]) + float(row["fp"]) + float(row["fn"])
                              for row in point_rows if int(row["seed"]) == seed), 1.0)
                    for seed in SEEDS])
            elif metric == "iou":
                point = np.mean([
                    sum(float(row["tp"]) for row in point_rows if int(row["seed"]) == seed)
                    / max(sum(float(row["tp"]) + float(row["fp"]) + float(row["fn"])
                              for row in point_rows if int(row["seed"]) == seed), 1.0)
                    for seed in SEEDS])
            elif metric == "absolute_relative_area_error":
                point = np.mean([
                    abs(sum(float(row["fp"]) - float(row["fn"])
                            for row in point_rows if int(row["seed"]) == seed))
                    / max(sum(float(row["tp"]) + float(row["fn"])
                              for row in point_rows if int(row["seed"]) == seed), 1.0)
                    for seed in SEEDS])
            else:
                point = np.mean([
                    sum(float(row["patch_area_fraction_mae_pp"])
                        * float(row["n_patches"])
                        for row in point_rows if int(row["seed"]) == seed)
                    / max(sum(float(row["n_patches"])
                              for row in point_rows if int(row["seed"]) == seed), 1.0)
                    for seed in SEEDS])
            output.append({
                "experiment": experiment, "metric": metric,
                "n_spatial_blocks": len(names), "bootstrap_iterations": iterations,
                "point_estimate": float(point),
                "bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
            })
    return output


def summary_markdown(aggregates: list[dict], efficiencies: dict) -> str:
    lines = [
        "# COAST-ECAM formal experiment summary", "",
        "All values are mean +/- sample SD over seeds 42, 1337, and 3407. "
        "The independent test split is evaluated at the fixed 0.5 threshold.", "",
        "| Experiment | Precision | Recall | F1 | IoU | OA | Boundary F1 (2 px) | Relative area bias | Absolute relative area error | Area MAE (pp) | PR-AUC | Params (M) | Latency (ms) | Peak memory (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def cell(key: str, digits: int = 4) -> str:
            mean = row[f"{key}_mean"]
            sd = row[f"{key}_sd"]
            return f"{mean:.{digits}f} +/- {sd:.{digits}f}"
        speed = efficiencies.get(row["experiment"], {})
        lines.append(
            f"| {row['experiment']} | {cell('fixed_0p5_precision')} | "
            f"{cell('fixed_0p5_recall')} | {cell('fixed_0p5_f1')} | "
            f"{cell('fixed_0p5_iou')} | {cell('fixed_0p5_oa')} | "
            f"{cell('fixed_0p5_boundary_f1')} | "
            f"{cell('fixed_0p5_patch_sampled_area_relative_bias')} | "
            f"{cell('fixed_0p5_patch_sampled_area_absolute_relative_error')} | "
            f"{cell('fixed_0p5_patch_area_fraction_mae_pp', 3)} | "
            f"{cell('pr_auc')} | {speed.get('total_parameters', 0) / 1e6:.2f} | "
            f"{speed.get('fp32_batch1_latency_ms_mean', float('nan')):.2f} | "
            f"{speed.get('peak_allocated_memory_mb', float('nan')):.1f} |")
    lines.extend([
        "", "Boundary F1 uses a symmetric 2-pixel tolerance (approximately 20 m).",
        "Area bias is signed relative error over patch-sampled pixels; area MAE is "
        "the mean absolute changed-area-fraction error per overlapping patch and is not hectares.",
        "Validation-selected thresholds are supplementary only; the fixed 0.5 threshold is primary.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    output = OUTPUT_ROOT
    per_run = output / "per_run"
    per_run.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_ROOT / "spatial_split_manifest.csv"
    manifest = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    region_rows = []
    threshold_rows = []
    efficiencies = {}

    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            run_dir = RUN_ROOT / f"{experiment}_seed{seed}"
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            set_global_seed(seed, deterministic_warn_only=True)
            model_name = summary["model"]
            model = build_model(
                model_name, device,
                online_use_gating=summary.get("online_use_gating", True),
                prior_only_gating=summary.get("prior_only_gating", False),
            )
            model.load_state_dict(torch.load(
                run_dir / "best_model.pth", map_location=device, weights_only=True))
            model.eval()
            prior_dir = ("spatial_prior_gwda_oof"
                         if model_name == "BiT_Online_Boundary" else None)
            control = summary.get("prior_control", "none")
            datasets = {
                split: CDDataset(
                    str(DATA_ROOT), split=split, transform=False,
                    prior_dir_name=prior_dir, manifest_path=str(manifest_path),
                    prior_control=control, control_seed=seed)
                for split in ("val", "test")
            }
            loaders = {
                split: DataLoader(dataset, batch_size=8, shuffle=False,
                                  num_workers=0, pin_memory=device.type == "cuda")
                for split, dataset in datasets.items()
            }
            if experiment not in efficiencies:
                efficiencies[experiment] = benchmark_efficiency(
                    model, model_name, datasets["test"], device)
            val = collect_predictions(model, model_name, datasets["val"], loaders["val"], device)
            threshold, curve = select_validation_threshold(val["probabilities"], val["labels"])
            test = collect_predictions(model, model_name, datasets["test"], loaders["test"], device)
            fixed = add_absolute_area_error(evaluate_threshold(
                test["probabilities"], test["labels"], 0.5, 2))
            selected = add_absolute_area_error(evaluate_threshold(
                test["probabilities"], test["labels"], threshold, 2))
            stored = summary["test_metrics"]
            f1_difference = fixed["f1"] - stored["F1"]
            iou_difference = fixed["iou"] - stored["IoU"]
            result = {
                "experiment": experiment, "seed": seed,
                "checkpoint": str((run_dir / "best_model.pth").relative_to(PROJECT)),
                "validation_selected_threshold": threshold,
                "pr_auc": float(average_precision_score(
                    test["labels"].reshape(-1), test["probabilities"].reshape(-1))),
                "calibration": calibration_metrics(test["probabilities"], test["labels"]),
                "fixed_0p5": fixed, "validation_threshold_test": selected,
                "stored_fixed_0p5": stored,
                "reproduction_check": {
                    "f1_difference": f1_difference, "iou_difference": iou_difference,
                    "tolerance": REPRODUCTION_TOLERANCE,
                    "passes_tolerance": abs(f1_difference) < REPRODUCTION_TOLERANCE
                    and abs(iou_difference) < REPRODUCTION_TOLERANCE,
                },
            }
            (per_run / f"{experiment}_seed{seed}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
            for row in curve:
                threshold_rows.append({"experiment": experiment, "seed": seed, **row})
            for row in region_metrics(test["probabilities"], test["labels"],
                                      test["filenames"], manifest, 0.5):
                region_rows.append({"experiment": experiment, "seed": seed, **row})
            print(f"[done] {experiment} seed {seed}", flush=True)
            del model, val, test
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = [flatten_result(result) for result in results]
    aggregates = aggregate(rows)
    tests = paired_tests(rows)
    bootstrap = spatial_block_bootstrap(region_rows)
    write_csv(output / "all_runs.csv", rows)
    write_csv(output / "aggregate_results.csv", aggregates)
    write_csv(output / "paired_significance_vs_COAST_ECAM.csv", tests)
    write_csv(output / "spatial_group_metrics.csv", region_rows)
    write_csv(output / "spatial_block_bootstrap_ci.csv", bootstrap)
    write_csv(output / "validation_threshold_curves.csv", threshold_rows)
    (output / "efficiency.json").write_text(
        json.dumps(efficiencies, indent=2), encoding="utf-8")
    payload = {"aggregate": aggregates, "paired_significance": tests,
               "spatial_block_bootstrap": bootstrap}
    (output / "aggregate_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (output / "summary.md").write_text(
        summary_markdown(aggregates, efficiencies), encoding="utf-8")
    failures = [result for result in results
                if not result["reproduction_check"]["passes_tolerance"]]
    print(f"[complete] runs={len(results)} reproduction_failures={len(failures)}")


if __name__ == "__main__":
    main()
