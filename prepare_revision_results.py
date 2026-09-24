"""Validate revision runs and export compact publication-ready result tables."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


SEEDS = [42, 1337, 2025, 3407, 9001]
MAIN_ORDER = [
    "FCSiamDiff",
    "SNUNet",
    "ChangeFormer",
    "BiT",
    "BiT_GWDA",
    "OEP_BiT",
    "COAST",
]
CONTROL_ORDER = [
    "BiT_GWDA",
    "BiT_GWDA_shuffled",
    "BiT_GWDA_random",
    "COAST",
    "COAST_shuffled",
    "COAST_random",
    "COAST_no_gating",
]
ALPHA_ORDER = ["COAST_alpha005_val", "COAST", "COAST_alpha020_val"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return ""
    if sd is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {sd:.4f}"


def publication_row(result: dict) -> dict:
    return {
        "experiment": result["experiment"],
        "n": result["n"],
        "validation_f1_mean_sd": format_metric(
            result.get("best_val_f1_mean"), result.get("best_val_f1_sd")
        ),
        "test_f1_mean_sd": format_metric(
            result.get("test_f1_mean"), result.get("test_f1_sd")
        ),
        "test_iou_mean_sd": format_metric(
            result.get("test_iou_mean"), result.get("test_iou_sd")
        ),
        "test_precision_mean_sd": format_metric(
            result.get("test_precision_mean"), result.get("test_precision_sd")
        ),
        "test_recall_mean_sd": format_metric(
            result.get("test_recall_mean"), result.get("test_recall_sd")
        ),
        "test_oa_mean_sd": format_metric(
            result.get("test_oa_mean"), result.get("test_oa_sd")
        ),
    }


def buffered_conflict(a: dict, b: dict, patch_size: int, buffer: int) -> bool:
    if a["region"] != b["region"]:
        return False
    return (
        max(a["x"] - buffer, b["x"])
        < min(a["x"] + patch_size + buffer, b["x"] + patch_size)
        and max(a["y"] - buffer, b["y"])
        < min(a["y"] + patch_size + buffer, b["y"] + patch_size)
    )


def count_cross_split_conflicts(manifest_path: Path, patch_size: int, buffer: int) -> int:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {
                **row,
                "x": int(row["x"]),
                "y": int(row["y"]),
            }
            for row in csv.DictReader(handle)
        ]
    retained = [row for row in rows if row["split"] in {"train", "val", "test"}]
    conflicts = 0
    for index, left in enumerate(retained):
        for right in retained[index + 1 :]:
            if left["split"] != right["split"] and buffered_conflict(
                left, right, patch_size, buffer
            ):
                conflicts += 1
    return conflicts


def main() -> None:
    project = Path(__file__).resolve().parent
    run_root = project / "revision_experiments_gwda"
    output = project / "results" / "revision_gwda"
    output.mkdir(parents=True, exist_ok=True)

    matrix = json.loads((run_root / "experiment_matrix.json").read_text(encoding="utf-8"))
    expected_experiments = [item["name"] for item in matrix["experiments"]]
    if matrix["seeds"] != SEEDS:
        raise AssertionError(f"Unexpected seeds: {matrix['seeds']}")
    if any(re.search(r"(^|_)GWR($|_)", name) for name in expected_experiments):
        raise AssertionError("A GWR experiment is present in the revision matrix")

    summaries: dict[str, list[dict]] = defaultdict(list)
    for experiment in expected_experiments:
        for seed in SEEDS:
            path = run_root / f"{experiment}_seed{seed}" / "summary.json"
            if not path.exists():
                raise AssertionError(f"Missing summary: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary["seed"] != seed:
                raise AssertionError(f"Seed mismatch in {path}")
            is_sensitivity = experiment.endswith("_val")
            if is_sensitivity:
                if summary.get("test_metrics") is not None:
                    raise AssertionError(f"Test metrics leaked into {experiment}")
                if summary.get("split_counts", {}).get("test") is not None:
                    raise AssertionError(f"Test data were loaded by {experiment}")
                if "--skip_test" not in summary.get("command", ""):
                    raise AssertionError(f"Missing --skip_test in {experiment}")
            elif not summary.get("test_metrics"):
                raise AssertionError(f"Missing test metrics in {experiment}")
            summaries[experiment].append(summary)

    if sum(map(len, summaries.values())) != 70:
        raise AssertionError("The revision matrix must contain exactly 70 completed runs")

    aggregate_payload = json.loads(
        (run_root / "aggregate_results.json").read_text(encoding="utf-8")
    )
    aggregates = {row["experiment"]: row for row in aggregate_payload["aggregate"]}
    if set(aggregates) != set(expected_experiments):
        raise AssertionError("Aggregate experiment set does not match the matrix")
    if any(row["n"] != 5 for row in aggregates.values()):
        raise AssertionError("Every aggregate must contain five seeds")

    split_summary_path = project / "splits" / "spatial_split_2048px_buffer256_seed42.summary.json"
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    split_manifest = project / "splits" / "spatial_split_2048px_buffer256_seed42.csv"
    conflicts = count_cross_split_conflicts(
        split_manifest,
        split_summary["patch_size_pixels"],
        split_summary["buffer_pixels"],
    )
    if conflicts:
        raise AssertionError(f"Found {conflicts} buffered cross-split conflicts")

    gwda_metadata_path = project / "splits" / "gwda_train_only_metadata.json"
    gwda = json.loads(gwda_metadata_path.read_text(encoding="utf-8"))
    if gwda.get("fit_split") != "train_only":
        raise AssertionError("GWDA teacher is not marked as train-only")

    write_csv(output / "publication_main_results.csv", [
        publication_row(aggregates[name]) for name in MAIN_ORDER
    ])
    write_csv(output / "publication_control_results.csv", [
        publication_row(aggregates[name]) for name in CONTROL_ORDER
    ])
    write_csv(output / "publication_alpha_sensitivity.csv", [
        {
            "alpha": {"COAST_alpha005_val": "0.05", "COAST": "0.10", "COAST_alpha020_val": "0.20"}[name],
            "selection_split": "validation",
            "n": aggregates[name]["n"],
            "validation_f1_mean_sd": format_metric(
                aggregates[name].get("best_val_f1_mean"),
                aggregates[name].get("best_val_f1_sd"),
            ),
            "test_metrics_used_for_selection": "no",
        }
        for name in ALPHA_ORDER
    ])

    significance_rows = []
    for comparator, metrics in aggregate_payload["paired_significance"].items():
        for metric, values in metrics.items():
            significance_rows.append({
                "comparison": f"COAST - {comparator}",
                "metric": metric.removeprefix("test_"),
                "mean_paired_difference": f"{values['coast_minus_comparator_mean']:.6f}",
                "bootstrap_95_ci_low": f"{values['paired_bootstrap_95_ci'][0]:.6f}",
                "bootstrap_95_ci_high": f"{values['paired_bootstrap_95_ci'][1]:.6f}",
                "paired_t_p": f"{values['paired_t_p']:.6f}",
                "wilcoxon_p": f"{values['wilcoxon_p']:.6f}",
            })
    write_csv(output / "publication_significance.csv", significance_rows)

    shutil.copy2(run_root / "all_runs.csv", output / "all_runs.csv")
    shutil.copy2(run_root / "aggregate_results.json", output / "aggregate_results.json")
    shutil.copy2(run_root / "experiment_matrix.json", output / "experiment_matrix.json")
    shutil.copy2(split_summary_path, output / "spatial_split_summary.json")
    shutil.copy2(gwda_metadata_path, output / "gwda_train_only_metadata.json")

    coast = aggregates["COAST"]
    report = f"""# Leakage-safe GWDA revision experiment audit

## Completion and reproducibility

- Completed runs: 70/70 (14 experiments x 5 seeds: {', '.join(map(str, SEEDS))}).
- Spatial split: {split_summary['counts']['train']} train, {split_summary['counts']['val']} validation, and {split_summary['counts']['test']} test patches; {split_summary['excluded_patch_count']} buffer patches excluded.
- Buffered cross-split conflicts: {conflicts}.
- GWDA teacher fit: training split only, using {gwda['training_source_file_count']} source patches and {gwda['unique_training_sample_count']} unique sampled pixels.
- Revision prior experiments: GWDA only; no GWR experiment is present.
- Alpha sensitivity: alpha = 0.05 and 0.20 used validation only; test metrics and test-loader counts are null. Alpha = 0.10 is the prespecified main COAST setting.

## Main result

Across five seeds, COAST achieved test F1 = {coast['test_f1_mean']:.4f} +/- {coast['test_f1_sd']:.4f} and IoU = {coast['test_iou_mean']:.4f} +/- {coast['test_iou_sd']:.4f}. Full baseline, control, sensitivity, and paired-comparison results are provided in the adjacent CSV and JSON files.

## Interpretation guardrails

- COAST and OEP-BiT were statistically indistinguishable in the paired five-seed tests.
- COAST exceeded the shuffled-prior control on mean F1 and IoU, but the paired t and Wilcoxon tests did not reach p < 0.05 with five seeds.
- The random-prior and no-gating controls had higher mean scores than COAST in this run. The manuscript must report this result directly and avoid claiming that the learned gate or GWDA map alone improves peak segmentation accuracy.
- SNUNet had the highest mean F1 and IoU among the listed main baselines. The revised claim should focus on scalable online-prior inference and comparable accuracy, not universal state-of-the-art accuracy.
"""
    (output / "reproducibility_report.md").write_text(report, encoding="utf-8")
    print(f"Validated 70 runs and wrote publication summaries to {output}")


if __name__ == "__main__":
    main()
