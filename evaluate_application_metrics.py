"""Evaluate application-facing metrics from existing best checkpoints.

The validation split is used only to select a decision threshold. The independent
test split is then evaluated once at both the fixed 0.5 threshold and the frozen
validation-selected threshold. Fixed-0.5 metrics remain the primary results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from dataset import CDDataset
from train import PRIOR_MODELS, build_model, set_global_seed


SEEDS = [42, 1337, 2025, 3407, 9001]
REPRODUCTION_TOLERANCE = 2e-4
DATASET_ROOT = Path(r"D:\yoyu\SA_Identification\dataset_patches_2020_2024")
RUNS = {
    "FCSiamDiff": ("FCSiamDiff", "revision_experiments_gwda", "FCSiamDiff_seed{seed}"),
    "SiameseNestedUNet_local": ("SNUNet", "revision_experiments_gwda", "SNUNet_seed{seed}"),
    "BiT": ("BiT", "revision_experiments_gwda", "BiT_seed{seed}"),
    "ChangeFormer": ("ChangeFormer", "revision_experiments_gwda", "ChangeFormer_seed{seed}"),
    "STeInFormer": ("STeInFormer", "recent_baseline_revision", "STeInFormer_seed{seed}"),
    "EdgeRefNet": ("EdgeRefNet", "recent_baseline_revision", "EdgeRefNet_seed{seed}"),
    "COAST": ("BiT_Online", "spg_optimization_revision", "COAST_SPGopt_seed{seed}"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/revision_2026/application_metrics"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--models", nargs="*", choices=list(RUNS),
                        default=list(RUNS))
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Evaluate only FCSiamDiff seed 42.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def forward_logits(model: torch.nn.Module, train_name: str,
                   image_a: torch.Tensor, image_b: torch.Tensor,
                   prior: torch.Tensor) -> torch.Tensor:
    output = (model(image_a, image_b, prior)
              if train_name in PRIOR_MODELS else model(image_a, image_b))
    if isinstance(output, (list, tuple)):
        output = output[-1]
    return output


@torch.inference_mode()
def collect_predictions(model: torch.nn.Module, train_name: str,
                        dataset: CDDataset, loader: DataLoader,
                        device: torch.device) -> dict:
    model.eval()
    probabilities = []
    labels = []
    filenames = []
    cursor = 0
    for image_a, image_b, label, prior in loader:
        batch = image_a.shape[0]
        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        prior = prior.to(device, non_blocking=True)
        logits = forward_logits(model, train_name, image_a, image_b, prior)
        probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
        labels.append(label.numpy().astype(np.uint8))
        filenames.extend(dataset.file_list[cursor:cursor + batch])
        cursor += batch
    return {
        "probabilities": np.concatenate(probabilities, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "filenames": filenames,
    }


def confusion(probabilities: np.ndarray, labels: np.ndarray,
              threshold: float) -> dict[str, int]:
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    return {
        "tp": int(np.logical_and(predicted, truth).sum()),
        "fp": int(np.logical_and(predicted, ~truth).sum()),
        "fn": int(np.logical_and(~predicted, truth).sum()),
        "tn": int(np.logical_and(~predicted, ~truth).sum()),
    }


def metrics_from_confusion(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    oa = (tp + tn) / (tp + fp + fn + tn + 1e-12)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "iou": iou, "oa": oa,
    }


def select_validation_threshold(probabilities: np.ndarray,
                                labels: np.ndarray) -> tuple[float, list[dict]]:
    curve = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        result = metrics_from_confusion(confusion(probabilities, labels, threshold))
        curve.append({"threshold": float(threshold), **result})
    best = max(curve, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5)))
    return best["threshold"], curve


def boundary_counts(predicted: np.ndarray, truth: np.ndarray,
                    tolerance: int) -> dict[str, int]:
    structure = np.ones((3, 3), dtype=bool)
    counts = {"pred_boundary": 0, "true_boundary": 0,
              "matched_pred_boundary": 0, "matched_true_boundary": 0}
    for pred_patch, true_patch in zip(predicted[:, 0], truth[:, 0]):
        pred_patch = pred_patch.astype(bool)
        true_patch = true_patch.astype(bool)
        pred_boundary = np.logical_xor(
            pred_patch, binary_erosion(pred_patch, structure=structure,
                                       border_value=0))
        true_boundary = np.logical_xor(
            true_patch, binary_erosion(true_patch, structure=structure,
                                       border_value=0))
        dilated_pred = binary_dilation(
            pred_boundary, structure=structure, iterations=tolerance)
        dilated_true = binary_dilation(
            true_boundary, structure=structure, iterations=tolerance)
        counts["pred_boundary"] += int(pred_boundary.sum())
        counts["true_boundary"] += int(true_boundary.sum())
        counts["matched_pred_boundary"] += int(
            np.logical_and(pred_boundary, dilated_true).sum())
        counts["matched_true_boundary"] += int(
            np.logical_and(true_boundary, dilated_pred).sum())
    return counts


def boundary_metrics(counts: dict[str, int]) -> dict[str, float]:
    precision = counts["matched_pred_boundary"] / max(counts["pred_boundary"], 1)
    recall = counts["matched_true_boundary"] / max(counts["true_boundary"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def area_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred_pixels = predicted.reshape(predicted.shape[0], -1).sum(axis=1)
    true_pixels = truth.reshape(truth.shape[0], -1).sum(axis=1)
    patch_pixels = predicted.shape[-2] * predicted.shape[-1]
    errors_pp = (pred_pixels - true_pixels) / patch_pixels * 100.0
    return {
        "patch_sampled_area_relative_bias": float(
            (pred_pixels.sum() - true_pixels.sum()) / max(true_pixels.sum(), 1)),
        "patch_area_fraction_mae_pp": float(np.mean(np.abs(errors_pp))),
        "patch_area_fraction_median_ae_pp": float(np.median(np.abs(errors_pp))),
        "patch_area_fraction_signed_error_pp": float(np.mean(errors_pp)),
    }


def calibration_metrics(probabilities: np.ndarray,
                        labels: np.ndarray) -> dict[str, float]:
    flat_prob = probabilities.reshape(-1).astype(np.float64)
    flat_labels = labels.reshape(-1).astype(np.float64)
    brier = float(np.mean((flat_prob - flat_labels) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = ((flat_prob >= lower) &
                  (flat_prob < upper if upper < 1.0 else flat_prob <= upper))
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(flat_prob[in_bin].mean()) - float(flat_labels[in_bin].mean()))
    return {"brier_score": brier, "ece_10bin": ece}


def region_metrics(probabilities: np.ndarray, labels: np.ndarray,
                   filenames: list[str], manifest: dict[str, dict[str, str]],
                   threshold: float) -> list[dict]:
    rows = []
    for group_type in ("region", "spatial_block"):
        groups = defaultdict(list)
        for index, filename in enumerate(filenames):
            metadata = manifest[filename]
            group_name = (metadata["region"] if group_type == "region" else
                          f"block_{metadata['block_x']}_{metadata['block_y']}")
            groups[group_name].append(index)
        for group_name, indices in sorted(groups.items()):
            prob = probabilities[indices]
            truth = labels[indices]
            predicted = prob >= threshold
            counts = confusion(prob, truth, threshold)
            rows.append({
                "group_type": group_type,
                "group_name": group_name,
                "n_patches": len(indices),
                **counts,
                **metrics_from_confusion(counts),
                **area_metrics(predicted, truth),
            })
    return rows


def evaluate_threshold(probabilities: np.ndarray, labels: np.ndarray,
                       threshold: float, boundary_tolerance: int) -> dict:
    predicted = probabilities >= threshold
    counts = confusion(probabilities, labels, threshold)
    boundaries = boundary_counts(predicted, labels, boundary_tolerance)
    return {
        "threshold": threshold,
        **counts,
        **metrics_from_confusion(counts),
        **boundary_metrics(boundaries),
        **area_metrics(predicted, labels),
    }


@torch.inference_mode()
def benchmark_efficiency(model: torch.nn.Module, train_name: str,
                         dataset: CDDataset, device: torch.device) -> dict:
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    image_a, image_b, _, prior = dataset[0]
    image_a = image_a.unsqueeze(0).to(device)
    image_b = image_b.unsqueeze(0).to(device)
    prior = prior.unsqueeze(0).to(device)
    model.eval()
    for _ in range(5):
        forward_logits(model, train_name, image_a, image_b, prior)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        forward_logits(model, train_name, image_a, image_b, prior)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
               if device.type == "cuda" else None)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "fp32_batch1_latency_ms_mean": float(np.mean(timings)),
        "fp32_batch1_latency_ms_sd": float(np.std(timings, ddof=1)),
        "fp32_batch1_throughput_images_s": 1000.0 / float(np.mean(timings)),
        "peak_allocated_memory_mb": peak_mb,
        "input_shape_each_date": list(image_a.shape),
        "warmup_iterations": 5,
        "timed_iterations": 20,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(result: dict) -> dict:
    row = {
        "model": result["model"], "seed": result["seed"],
        "validation_selected_threshold": result["validation_selected_threshold"],
        "pr_auc": result["threshold_free"]["pr_auc"],
        **result["calibration"],
        "stored_f1": result["stored_fixed_0p5"]["F1"],
        "recomputed_f1_difference": result["reproduction_check"]["f1_difference"],
    }
    for prefix, block in (("fixed_0p5", result["fixed_0p5"]),
                          ("val_threshold", result["validation_threshold_test"])):
        for key, value in block.items():
            if isinstance(value, (int, float)):
                row[f"{prefix}_{key}"] = value
    return row


def aggregate_rows(rows: list[dict]) -> list[dict]:
    output = []
    for model in RUNS:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        aggregate = {"model": model, "n": len(selected)}
        numeric_keys = [key for key, value in selected[0].items()
                        if key not in {"model", "seed"}
                        and isinstance(value, (int, float))]
        for key in numeric_keys:
            values = np.asarray([row[key] for row in selected], dtype=float)
            aggregate[f"{key}_mean"] = float(values.mean())
            aggregate[f"{key}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None)
        output.append(aggregate)
    return output


def aggregate_region_rows(rows: list[dict]) -> list[dict]:
    output = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["group_type"], row["group_name"])].append(row)
    for (model, group_type, group_name), selected in sorted(grouped.items()):
        result = {
            "model": model, "group_type": group_type,
            "group_name": group_name, "n_seeds": len(selected),
            "n_patches": int(float(selected[0]["n_patches"])),
        }
        for key in ("f1", "iou", "patch_sampled_area_relative_bias",
                    "patch_area_fraction_mae_pp"):
            values = np.asarray([float(row[key]) for row in selected])
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None)
        output.append(result)
    return output


def spatial_robustness_summary(rows: list[dict]) -> list[dict]:
    by_model_block = defaultdict(list)
    for row in rows:
        if row["group_type"] == "spatial_block":
            by_model_block[(row["model"], row["group_name"])].append(row)
    output = []
    for model in RUNS:
        positive_f1 = []
        negative_area_error = []
        for (row_model, _), selected in by_model_block.items():
            if row_model != model:
                continue
            truth_positive_pixels = int(float(selected[0]["tp"])) + int(
                float(selected[0]["fn"]))
            if truth_positive_pixels > 0:
                positive_f1.append(np.mean([
                    float(row["f1"]) for row in selected]))
            else:
                negative_area_error.append(np.mean([
                    float(row["patch_area_fraction_mae_pp"])
                    for row in selected]))
        if positive_f1:
            output.append({
                "model": model,
                "positive_test_blocks": len(positive_f1),
                "positive_block_macro_f1": float(np.mean(positive_f1)),
                "positive_block_f1_sd_across_blocks": (
                    float(np.std(positive_f1, ddof=1))
                    if len(positive_f1) > 1 else None),
                "worst_positive_block_f1": float(np.min(positive_f1)),
                "negative_test_blocks": len(negative_area_error),
                "worst_negative_block_area_mae_pp": (
                    float(np.max(negative_area_error))
                    if negative_area_error else None),
            })
    return output


def paired_significance(rows: list[dict], target: str = "COAST") -> list[dict]:
    by_model_seed = {(row["model"], int(row["seed"])): row for row in rows}
    specifications = {
        "pr_auc": True,
        "brier_score": False,
        "ece_10bin": False,
        "fixed_0p5_f1": True,
        "fixed_0p5_boundary_f1": True,
        "fixed_0p5_patch_area_fraction_mae_pp": False,
        "val_threshold_f1": True,
    }
    output = []
    for comparator in RUNS:
        if comparator == target:
            continue
        common = [seed for seed in SEEDS
                  if (target, seed) in by_model_seed
                  and (comparator, seed) in by_model_seed]
        if len(common) < 2:
            continue
        for metric, higher_is_better in specifications.items():
            target_values = np.asarray([
                float(by_model_seed[(target, seed)][metric]) for seed in common])
            other_values = np.asarray([
                float(by_model_seed[(comparator, seed)][metric]) for seed in common])
            raw_difference = target_values - other_values
            oriented = raw_difference if higher_is_better else -raw_difference
            try:
                wilcoxon_p = float(wilcoxon(raw_difference).pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            output.append({
                "target": target,
                "comparator": comparator,
                "metric": metric,
                "n": len(common),
                "higher_is_better": higher_is_better,
                "target_mean": float(target_values.mean()),
                "comparator_mean": float(other_values.mean()),
                "raw_target_minus_comparator": float(raw_difference.mean()),
                "oriented_advantage_positive_is_target_better": float(oriented.mean()),
                "paired_t_p": float(ttest_rel(target_values, other_values).pvalue),
                "wilcoxon_p": wilcoxon_p,
            })
    return output


def publication_table(aggregates: list[dict], efficiency: dict) -> list[dict]:
    rows = []
    for item in aggregates:
        model = item["model"]
        speed = efficiency.get(model, {})
        rows.append({
            "model": model,
            "f1_mean": item["fixed_0p5_f1_mean"],
            "f1_sd": item["fixed_0p5_f1_sd"],
            "pr_auc_mean": item["pr_auc_mean"],
            "pr_auc_sd": item["pr_auc_sd"],
            "boundary_f1_mean": item["fixed_0p5_boundary_f1_mean"],
            "boundary_f1_sd": item["fixed_0p5_boundary_f1_sd"],
            "patch_area_mae_pp_mean": item[
                "fixed_0p5_patch_area_fraction_mae_pp_mean"],
            "patch_area_mae_pp_sd": item[
                "fixed_0p5_patch_area_fraction_mae_pp_sd"],
            "brier_mean": item["brier_score_mean"],
            "brier_sd": item["brier_score_sd"],
            "ece_mean": item["ece_10bin_mean"],
            "ece_sd": item["ece_10bin_sd"],
            "val_selected_threshold_mean": item[
                "validation_selected_threshold_mean"],
            "val_threshold_test_f1_mean": item["val_threshold_f1_mean"],
            "parameters_million": (
                speed.get("total_parameters", math.nan) / 1e6),
            "latency_ms": speed.get("fp32_batch1_latency_ms_mean"),
            "peak_memory_mb": speed.get("peak_allocated_memory_mb"),
        })
    return rows


def summary_markdown(table: list[dict]) -> str:
    lines = [
        "# Application-facing metrics",
        "",
        "Fixed-threshold 0.5 results, mean +/- sample SD over five seeds.",
        "",
        "| Model | F1 | PR-AUC | Boundary F1 | Patch area MAE (pp) | Brier | Params (M) | Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['f1_mean']:.4f} +/- {row['f1_sd']:.4f} | "
            f"{row['pr_auc_mean']:.4f} +/- {row['pr_auc_sd']:.4f} | "
            f"{row['boundary_f1_mean']:.4f} +/- {row['boundary_f1_sd']:.4f} | "
            f"{row['patch_area_mae_pp_mean']:.3f} +/- {row['patch_area_mae_pp_sd']:.3f} | "
            f"{row['brier_mean']:.4f} +/- {row['brier_sd']:.4f} | "
            f"{row['parameters_million']:.2f} | {row['latency_ms']:.2f} |")
    lines.extend([
        "",
        "Interpretation: boundary F1 measures delineation quality; patch area MAE measures absolute error in changed-area fraction per overlapping test patch; Brier measures probability calibration (lower is better). Patch area values are not unique-area hectares.",
        "",
    ])
    return "\n".join(lines)


def protocol_markdown(boundary_tolerance: int) -> str:
    return f"""# Application-facing evaluation protocol

- All models are loaded from their validation-best checkpoints and evaluated in FP32.
- The fixed probability threshold of 0.5 is the primary comparison.
- PR-AUC is threshold-free and is computed over all independent-test pixels.
- A supplementary threshold is selected from 0.05 to 0.95 (step 0.01) using validation F1 only, then frozen before test evaluation.
- Boundary F1 uses a symmetric tolerance of {boundary_tolerance} pixels. At 10 m Sentinel-2 resolution this corresponds to approximately {boundary_tolerance * 10} m.
- Area metrics are patch-sampled area-fraction errors. They are not hectares and must not be interpreted as unique mapped area because neighbouring patches overlap.
- Per-region metrics aggregate pixels within the manifest's region field and are descriptive because region sample sizes differ.
- Efficiency uses FP32, batch size 1, two 8-channel 256 x 256 inputs, 5 warm-up iterations and 20 timed iterations on the same device.
- Results are reported as mean +/- sample SD across the same five random seeds.
- Recomputed legacy F1/IoU values must agree with stored values within an absolute tolerance of {REPRODUCTION_TOLERANCE}; this tolerance covers sub-threshold CUDA-kernel numerical variation and is far below the reported precision.
"""


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project / output_dir
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    models = ["FCSiamDiff"] if args.smoke else args.models
    seeds = [42] if args.smoke else args.seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = args.data_root / "spatial_split_manifest.csv"
    manifest = load_manifest(manifest_path)
    prior_dir = "spatial_prior_gwda_train_only"

    datasets = {
        split: CDDataset(
            str(args.data_root), split=split, transform=False,
            prior_dir_name=prior_dir, manifest_path=str(manifest_path))
        for split in ("val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        for split, dataset in datasets.items()
    }

    efficiency = {}
    region_rows = []
    curve_rows = []
    for display_name in models:
        train_name, root, pattern = RUNS[display_name]
        for seed in seeds:
            result_path = per_run_dir / f"{display_name}_seed{seed}.json"
            if result_path.exists() and not args.overwrite:
                print(f"[skip] {display_name} seed {seed}", flush=True)
                continue
            run_dir = project / root / pattern.format(seed=seed)
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8"))
            print(f"[evaluate] {display_name} seed {seed}", flush=True)
            # Match the numerical settings used by train.py validation exactly.
            # In particular this disables TF32 and CUDA benchmark kernels.
            set_global_seed(seed, deterministic_warn_only=True)
            model = build_model(train_name, device)
            model.load_state_dict(torch.load(
                run_dir / "best_model.pth", map_location=device,
                weights_only=True))
            model.eval()

            if display_name not in efficiency and not args.skip_efficiency:
                efficiency[display_name] = benchmark_efficiency(
                    model, train_name, datasets["test"], device)

            val = collect_predictions(
                model, train_name, datasets["val"], loaders["val"], device)
            selected_threshold, validation_curve = select_validation_threshold(
                val["probabilities"], val["labels"])
            test = collect_predictions(
                model, train_name, datasets["test"], loaders["test"], device)
            fixed = evaluate_threshold(
                test["probabilities"], test["labels"], 0.5,
                args.boundary_tolerance_px)
            selected = evaluate_threshold(
                test["probabilities"], test["labels"], selected_threshold,
                args.boundary_tolerance_px)
            pr_auc = float(average_precision_score(
                test["labels"].reshape(-1),
                test["probabilities"].reshape(-1)))
            stored = summary["test_metrics"]
            result = {
                "model": display_name,
                "training_model_name": train_name,
                "seed": seed,
                "checkpoint": str((run_dir / "best_model.pth").relative_to(project)),
                "device": str(device),
                "primary_threshold": 0.5,
                "validation_selected_threshold": selected_threshold,
                "threshold_free": {"pr_auc": pr_auc},
                "calibration": calibration_metrics(
                    test["probabilities"], test["labels"]),
                "fixed_0p5": fixed,
                "validation_threshold_test": selected,
                "stored_fixed_0p5": stored,
                "reproduction_check": {
                    "f1_difference": fixed["f1"] - stored["F1"],
                    "iou_difference": fixed["iou"] - stored["IoU"],
                    "tolerance": REPRODUCTION_TOLERANCE,
                    "passes_tolerance": (
                        abs(fixed["f1"] - stored["F1"])
                        < REPRODUCTION_TOLERANCE
                        and abs(fixed["iou"] - stored["IoU"])
                        < REPRODUCTION_TOLERANCE),
                },
            }
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            for row in validation_curve:
                curve_rows.append({
                    "model": display_name, "seed": seed,
                    "split": "validation", **row,
                })
            for row in region_metrics(
                    test["probabilities"], test["labels"],
                    test["filenames"], manifest, 0.5):
                region_rows.append({"model": display_name, "seed": seed, **row})

            del model, val, test
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result_files = sorted(per_run_dir.glob("*.json"))
    results = []
    for path in result_files:
        result = json.loads(path.read_text(encoding="utf-8"))
        check = result["reproduction_check"]
        check["tolerance"] = REPRODUCTION_TOLERANCE
        check["passes_tolerance"] = (
            abs(check["f1_difference"]) < REPRODUCTION_TOLERANCE
            and abs(check["iou_difference"]) < REPRODUCTION_TOLERANCE)
        check.pop("passes_1e-6", None)
        checkpoint = Path(result["checkpoint"])
        if checkpoint.is_absolute():
            try:
                result["checkpoint"] = str(checkpoint.relative_to(project))
            except ValueError:
                pass
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    rows = [flatten_result(result) for result in results]
    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "all_runs.csv", rows)
    write_csv(output_dir / "aggregate_results.csv", aggregates)
    write_csv(output_dir / "spatial_group_metrics.csv", region_rows)
    region_source = region_rows
    if not region_source and (output_dir / "spatial_group_metrics.csv").exists():
        with (output_dir / "spatial_group_metrics.csv").open(
                newline="", encoding="utf-8") as handle:
            region_source = list(csv.DictReader(handle))
    write_csv(
        output_dir / "spatial_group_aggregate.csv",
        aggregate_region_rows(region_source),
    )
    write_csv(
        output_dir / "spatial_robustness_summary.csv",
        spatial_robustness_summary(region_source),
    )
    write_csv(output_dir / "validation_threshold_curves.csv", curve_rows)
    (output_dir / "aggregate_results.json").write_text(
        json.dumps({"aggregate": aggregates}, indent=2), encoding="utf-8")
    if efficiency:
        (output_dir / "efficiency.json").write_text(
            json.dumps(efficiency, indent=2), encoding="utf-8")
    efficiency_source = efficiency
    if not efficiency_source and (output_dir / "efficiency.json").exists():
        efficiency_source = json.loads(
            (output_dir / "efficiency.json").read_text(encoding="utf-8"))
    compact_table = publication_table(aggregates, efficiency_source)
    write_csv(output_dir / "publication_metrics.csv", compact_table)
    write_csv(output_dir / "paired_significance_vs_COAST.csv",
              paired_significance(rows))
    (output_dir / "application_metrics_summary.md").write_text(
        summary_markdown(compact_table), encoding="utf-8")
    (output_dir / "evaluation_protocol.md").write_text(
        protocol_markdown(args.boundary_tolerance_px), encoding="utf-8")

    failures = [result for result in results
                if not result["reproduction_check"]["passes_tolerance"]]
    print(f"[done] {len(results)} runs; reproduction failures={len(failures)}")
    print(output_dir)


if __name__ == "__main__":
    main()
