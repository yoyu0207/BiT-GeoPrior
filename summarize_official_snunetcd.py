"""Prepare reproducible official SNUNet-CD versus COAST summaries."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon

from evaluate_application_metrics import forward_logits
from train import build_model


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "results" / "revision_2026" / "official_snunetcd"
SEEDS = [42, 1337, 2025, 3407, 9001]
METRICS = {
    "pr_auc": True,
    "brier_score": False,
    "ece_10bin": False,
    "fixed_0p5_f1": True,
    "fixed_0p5_iou": True,
    "fixed_0p5_boundary_f1": True,
    "fixed_0p5_patch_area_fraction_mae_pp": False,
    "fixed_0p5_patch_area_fraction_median_ae_pp": False,
    "fixed_0p5_patch_sampled_area_relative_bias": False,
    "fixed_0p5_patch_area_fraction_signed_error_pp": False,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparison(rows: list[dict[str, str]]) -> list[dict]:
    lookup = {(row["model"], int(row["seed"])): row for row in rows}
    rng = np.random.default_rng(42)
    output = []
    for metric, higher_is_better in METRICS.items():
        coast = np.asarray([
            float(lookup[("COAST", seed)][metric]) for seed in SEEDS])
        snunet = np.asarray([
            float(lookup[("SNUNetCDOfficial", seed)][metric]) for seed in SEEDS])
        difference = coast - snunet
        bootstrap = np.asarray([
            difference[rng.integers(0, len(SEEDS), len(SEEDS))].mean()
            for _ in range(20000)
        ])
        output.append({
            "metric": metric,
            "n_paired_seeds": len(SEEDS),
            "higher_is_better": higher_is_better,
            "coast_mean": float(coast.mean()),
            "coast_sd": float(coast.std(ddof=1)),
            "official_snunetcd_mean": float(snunet.mean()),
            "official_snunetcd_sd": float(snunet.std(ddof=1)),
            "coast_minus_official_snunetcd": float(difference.mean()),
            "bootstrap_95_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_95_ci_high": float(np.quantile(bootstrap, 0.975)),
            "paired_t_p": float(ttest_rel(coast, snunet).pvalue),
            "wilcoxon_p": float(wilcoxon(difference).pvalue),
        })
    for source, destination in (("paired_t_p", "paired_t_p_holm"),
                                ("wilcoxon_p", "wilcoxon_p_holm")):
        order = np.argsort([row[source] for row in output])
        adjusted = np.empty(len(output), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            value = output[index][source] * (len(output) - rank)
            running = max(running, value)
            adjusted[index] = min(running, 1.0)
        for row, value in zip(output, adjusted):
            row[destination] = float(value)
    return output


def profiler_flops(model_name: str) -> int:
    model = build_model(model_name, torch.device("cpu")).eval()
    image_a = torch.randn(1, 8, 256, 256)
    image_b = torch.randn(1, 8, 256, 256)
    prior = torch.randn(1, 1, 256, 256)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True) as profile:
        with torch.no_grad():
            forward_logits(model, model_name, image_a, image_b, prior)
    return int(sum(event.flops for event in profile.key_averages()))


def complexity_summary() -> dict:
    coast = json.loads((
        PROJECT / "results/revision_2026/application_metrics/efficiency.json"
    ).read_text(encoding="utf-8"))["COAST"]
    official = json.loads((
        PROJECT / "official_snunetcd_revision/efficiency_eval/efficiency.json"
    ).read_text(encoding="utf-8"))["SNUNetCDOfficial"]
    result = {
        "protocol": {
            "input_each_date": [1, 8, 256, 256],
            "latency_precision": "FP32",
            "latency_batch_size": 1,
            "flops_method": "sum of operations counted by torch.profiler(with_flops=True)",
            "flops_limitation": "operations unsupported by the profiler are not counted",
        },
        "COAST": {**coast, "profiler_counted_flops": profiler_flops("BiT_Online")},
        "SNUNetCDOfficial": {
            **official,
            "profiler_counted_flops": profiler_flops("SNUNetCDOfficial"),
        },
    }
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coast_dir = PROJECT / "results/revision_2026/application_metrics"
    official_dir = PROJECT / "official_snunetcd_revision/application_metrics"
    rows = read_csv(coast_dir / "all_runs.csv") + read_csv(
        official_dir / "all_runs.csv")
    comparisons = paired_comparison(rows)
    write_csv(OUTPUT / "paired_comparison_vs_coast.csv", comparisons)

    for name in (
        "aggregate_results.json",
        "all_runs.csv",
        "publication_metrics.csv",
        "spatial_group_aggregate.csv",
        "spatial_robustness_summary.csv",
        "evaluation_protocol.md",
    ):
        source = official_dir / name
        if source.exists():
            shutil.copy2(source, OUTPUT / name)
    shutil.copy2(
        PROJECT / "official_snunetcd_revision/aggregate_results.json",
        OUTPUT / "training_aggregate_results.json")

    complexity = complexity_summary()
    (OUTPUT / "complexity.json").write_text(
        json.dumps(complexity, indent=2), encoding="utf-8")

    by_metric = {row["metric"]: row for row in comparisons}
    robustness = read_csv(official_dir / "spatial_robustness_summary.csv")[0]
    coast_robustness = next(
        row for row in read_csv(coast_dir / "spatial_robustness_summary.csv")
        if row["model"] == "COAST")
    summary = f"""# Official SNUNet-CD versus COAST

All accuracy results use the same spatial split, five paired seeds, validation-best checkpoints, and a fixed test threshold of 0.5. Neither model uses pretrained weights.

| Metric | COAST | Official SNUNet-CD | COAST - SNUNet-CD | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
"""
    for metric in METRICS:
        row = by_metric[metric]
        summary += (
            f"| {metric} | {row['coast_mean']:.4f} +/- {row['coast_sd']:.4f} | "
            f"{row['official_snunetcd_mean']:.4f} +/- {row['official_snunetcd_sd']:.4f} | "
            f"{row['coast_minus_official_snunetcd']:.4f} | "
            f"{row['paired_t_p']:.4g} | {row['wilcoxon_p']:.4g} |\n")

    coast_complexity = complexity["COAST"]
    official_complexity = complexity["SNUNetCDOfficial"]
    summary += f"""
## Complexity

| Metric | COAST | Official SNUNet-CD |
|---|---:|---:|
| Parameters (M) | {coast_complexity['total_parameters'] / 1e6:.3f} | {official_complexity['total_parameters'] / 1e6:.3f} |
| Profiler-counted FLOPs (G) | {coast_complexity['profiler_counted_flops'] / 1e9:.3f} | {official_complexity['profiler_counted_flops'] / 1e9:.3f} |
| FP32 batch-1 latency (ms) | {coast_complexity['fp32_batch1_latency_ms_mean']:.2f} | {official_complexity['fp32_batch1_latency_ms_mean']:.2f} |
| Throughput (images/s) | {coast_complexity['fp32_batch1_throughput_images_s']:.2f} | {official_complexity['fp32_batch1_throughput_images_s']:.2f} |
| Peak allocated GPU memory (MB) | {coast_complexity['peak_allocated_memory_mb']:.1f} | {official_complexity['peak_allocated_memory_mb']:.1f} |

## Spatial-block robustness

| Metric | COAST | Official SNUNet-CD |
|---|---:|---:|
| Positive-block macro F1 | {float(coast_robustness['positive_block_macro_f1']):.4f} | {float(robustness['positive_block_macro_f1']):.4f} |
| Positive-block F1 SD across blocks | {float(coast_robustness['positive_block_f1_sd_across_blocks']):.4f} | {float(robustness['positive_block_f1_sd_across_blocks']):.4f} |
| Worst positive-block F1 | {float(coast_robustness['worst_positive_block_f1']):.4f} | {float(robustness['worst_positive_block_f1']):.4f} |

Spatial-block robustness is descriptive because the test set contains only three positive blocks. Profiler-counted FLOPs exclude operations unsupported by PyTorch's FLOP counter.
"""
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
