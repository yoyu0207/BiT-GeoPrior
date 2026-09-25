"""Run the leakage-safe revision experiment plan as a resumable state machine."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


SEEDS = [42, 1337, 2025, 3407, 9001]
SPG_EXPERIMENTS = [
    "OEP_BiT_SPGopt",
    "COAST_SPGopt",
    "COAST_shuffled_SPGopt",
    "COAST_random_SPGopt",
    "COAST_no_gating_SPGopt",
]
RECENT_MODELS = ["STeInFormer", "EdgeRefNet"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] $ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}"
        )


def assert_summary(path: Path, *, test_expected: bool) -> dict:
    if not path.exists():
        raise AssertionError(f"Missing summary: {path}")
    summary = load_json(path)
    expected = {"train": 313, "val": 59, "test": 71 if test_expected else None}
    if summary.get("split_counts") != expected:
        raise AssertionError(
            f"Unexpected split counts in {path}: {summary.get('split_counts')}"
        )
    if test_expected and not summary.get("test_metrics"):
        raise AssertionError(f"Missing independent-test metrics in {path}")
    if not test_expected and summary.get("test_metrics") is not None:
        raise AssertionError(f"Test blindness violated in {path}")
    return summary


def validate_data(data_root: Path) -> None:
    split = load_json(data_root / "spatial_split_manifest.summary.json")
    counts = dict(split.get("counts") or split.get("split_counts") or {})
    counts["excluded"] = split.get("excluded_patch_count")
    if counts != {"train": 313, "val": 59, "test": 71, "excluded": 49}:
        raise AssertionError(f"Unexpected spatial split summary: {counts}")
    prior = load_json(
        data_root / "spatial_prior_gwda_train_only" / "gwda_metadata.json")
    if prior.get("fit_split") != "train_only":
        raise AssertionError("GWDA was not fitted on train only")
    source_count = prior.get("training_source_file_count")
    if source_count != 313:
        raise AssertionError(f"Expected 313 GWDA training files, found {source_count}")


def validate_spg(output_root: Path) -> None:
    lr_selection = load_json(output_root / "spg_lr_selection.json")
    alpha_selection = load_json(output_root / "alpha_selection.json")
    if any(row.get("test_metrics_used") for row in lr_selection["candidates"]):
        raise AssertionError("SPG LR selection used test metrics")
    if any(row.get("test_metrics_used") for row in alpha_selection["candidates"]):
        raise AssertionError("Alpha selection used test metrics")
    for path in output_root.glob("SPG_lr_*_val_seed42/summary.json"):
        assert_summary(path, test_expected=False)
    for path in output_root.glob("alpha_*_val_seed42/summary.json"):
        assert_summary(path, test_expected=False)
    if len(list(output_root.glob("SPG_lr_*_val_seed42/summary.json"))) != 3:
        raise AssertionError("Expected three completed SPG LR pilots")
    if len(list(output_root.glob("alpha_*_val_seed42/summary.json"))) != 3:
        raise AssertionError("Expected three completed alpha pilots")
    for experiment in SPG_EXPERIMENTS:
        found = []
        for seed in SEEDS:
            path = output_root / f"{experiment}_seed{seed}" / "summary.json"
            assert_summary(path, test_expected=True)
            found.append(seed)
        if found != SEEDS:
            raise AssertionError(f"Incomplete seed set for {experiment}: {found}")
    rows = list(csv.DictReader((output_root / "all_runs.csv").open(encoding="utf-8")))
    if len(rows) != 25:
        raise AssertionError(f"Expected 25 formal SPG rows, found {len(rows)}")
    aggregate = load_json(output_root / "aggregate_results.json")
    if len(aggregate.get("aggregate", [])) != 5:
        raise AssertionError("SPG aggregate does not contain five experiments")


def validate_recent(output_root: Path) -> None:
    selection = load_json(output_root / "lr_selection.json")
    if set(selection.get("models", {})) != set(RECENT_MODELS):
        raise AssertionError("Recent-baseline LR selection has unexpected models")
    for model in RECENT_MODELS:
        candidates = selection["models"][model]["candidates"]
        if len(candidates) != 3 or any(row.get("test_metrics_used") for row in candidates):
            raise AssertionError(f"Invalid validation-only LR selection for {model}")
        pilot_paths = list(output_root.glob(f"{model}_lr_*_val_seed42/summary.json"))
        if len(pilot_paths) != 3:
            raise AssertionError(f"Expected three completed LR pilots for {model}")
        for path in pilot_paths:
            assert_summary(path, test_expected=False)
        for seed in SEEDS:
            assert_summary(
                output_root / f"{model}_seed{seed}" / "summary.json",
                test_expected=True,
            )
    rows = list(csv.DictReader((output_root / "all_runs.csv").open(encoding="utf-8")))
    if len(rows) != 10:
        raise AssertionError(f"Expected 10 formal recent-baseline rows, found {len(rows)}")
    aggregate = load_json(output_root / "aggregate_results.json")
    if len(aggregate.get("aggregate", [])) != 2:
        raise AssertionError("Recent-baseline aggregate does not contain two models")


def copy_results(project: Path, spg_root: Path, recent_root: Path,
                 include_spg: bool, include_recent: bool) -> Path:
    destination = project / "results" / "revision_2026"
    destination.mkdir(parents=True, exist_ok=True)
    selected = {
        "spg": [
            "spg_lr_selection.json", "alpha_selection.json",
            "experiment_matrix.json", "all_runs.csv", "aggregate_results.json",
        ],
        "recent_baselines": [
            "lr_selection.json", "experiment_matrix.json",
            "all_runs.csv", "aggregate_results.json",
        ],
    }
    roots = {"spg": spg_root, "recent_baselines": recent_root}
    enabled = {"spg": include_spg, "recent_baselines": include_recent}
    for group, names in selected.items():
        if not enabled[group]:
            continue
        target = destination / group
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(roots[group] / name, target / name)

    spg = (load_json(spg_root / "aggregate_results.json")["aggregate"]
           if include_spg else [])
    recent = (load_json(recent_root / "aggregate_results.json")["aggregate"]
              if include_recent else [])
    lines = [
        "# Leakage-safe revision experiments",
        "",
        "All hyperparameters were selected using validation data only. Final values "
        "are mean +/- sample SD over five fixed random seeds on the spatially "
        "independent test split.",
        "",
        "| Experiment | Year | Test F1 | Test IoU | n |",
        "|---|---:|---:|---:|---:|",
    ]
    years = {"STeInFormer": 2025, "EdgeRefNet": 2026}
    for row in spg + recent:
        name = row["experiment"]
        year = years.get(name, "-")
        lines.append(
            f"| {name} | {year} | "
            f"{row['test_f1_mean']:.4f} +/- {row['test_f1_sd']:.4f} | "
            f"{row['test_iou_mean']:.4f} +/- {row['test_iou_sd']:.4f} | "
            f"{row['n']} |"
        )
    (destination / "publication_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=project.parent / "dataset_patches_2020_2024")
    parser.add_argument("--spg-root", type=Path,
                        default=project / "spg_optimization_revision")
    parser.add_argument("--recent-root", type=Path,
                        default=project / "recent_baseline_revision")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    state_path = project / "revision_pipeline_status.json"
    log_path = project / "revision_pipeline.log"
    state = load_json(state_path) if state_path.exists() else {
        "created_at": now(), "steps": {}}

    steps = [
        {"name": "validate_data", "requires": [],
         "action": lambda: validate_data(args.data_root)},
        {"name": "prepare_sources", "requires": [],
         "action": lambda: run(
            [sys.executable, "prepare_recent_baseline_sources.py"], project, log_path)},
        {"name": "smoke_recent", "requires": ["prepare_sources"],
         "action": lambda: run(
            [sys.executable, "smoke_recent_baselines.py"], project, log_path)},
        {"name": "run_spg", "requires": ["validate_data"],
         "action": lambda: run([
            sys.executable, "run_spg_optimization_experiments.py",
            "--data_root", str(args.data_root),
            "--output_root", str(args.spg_root),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
        ], project, log_path)},
        {"name": "validate_spg", "requires": ["run_spg"],
         "action": lambda: validate_spg(args.spg_root)},
        {"name": "run_recent", "requires": ["validate_data", "smoke_recent"],
         "action": lambda: run([
            sys.executable, "run_recent_baselines.py",
            "--data_root", str(args.data_root),
            "--output_root", str(args.recent_root),
            "--spg_output_root", str(args.spg_root),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
        ], project, log_path)},
        {"name": "validate_recent", "requires": ["run_recent"],
         "action": lambda: validate_recent(args.recent_root)},
    ]

    for step in steps:
        name = step["name"]
        if state["steps"].get(name, {}).get("status") == "complete":
            continue
        blocked_by = [
            dependency for dependency in step["requires"]
            if state["steps"].get(dependency, {}).get("status") != "complete"
        ]
        if blocked_by:
            state["steps"][name] = {
                "status": "skipped",
                "skipped_at": now(),
                "reason": "prerequisite did not complete",
                "blocked_by": blocked_by,
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            continue
        state["steps"][name] = {"status": "running", "started_at": now()}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        try:
            step["action"]()
        except Exception as exc:
            state["steps"][name] = {
                "status": "failed",
                "failed_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "log": str(log_path),
            }
        else:
            state["steps"][name] = {
                "status": "complete", "completed_at": now()}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    include_spg = state["steps"].get("validate_spg", {}).get("status") == "complete"
    include_recent = (
        state["steps"].get("validate_recent", {}).get("status") == "complete")
    if include_spg or include_recent:
        try:
            copy_results(
                project, args.spg_root, args.recent_root,
                include_spg=include_spg, include_recent=include_recent)
            state["steps"]["prepare_publication_results"] = {
                "status": "complete", "completed_at": now()}
        except Exception as exc:
            state["steps"]["prepare_publication_results"] = {
                "status": "failed", "failed_at": now(),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "log": str(log_path),
            }
    else:
        state["steps"]["prepare_publication_results"] = {
            "status": "skipped", "skipped_at": now(),
            "reason": "no experiment branch passed validation",
            "blocked_by": ["validate_spg", "validate_recent"],
        }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if args.push:
        try:
            results_dir = project / "results" / "revision_2026"
            if results_dir.exists():
                run(["git", "add", "results/revision_2026"], project, log_path)
            staged = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=project).returncode
            if staged:
                run(["git", "commit", "-m", "Add revision experiment results"],
                    project, log_path)
                run(["git", "push"], project, log_path)
            state["steps"]["push_results"] = {
                "status": "complete", "completed_at": now()}
        except Exception as exc:
            state["steps"]["push_results"] = {
                "status": "failed", "failed_at": now(),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "log": str(log_path),
            }

    failed = [name for name, result in state["steps"].items()
              if result.get("status") == "failed"]
    state["status"] = "complete" if not failed else "partial_complete"
    state["failed_steps"] = failed
    state["completed_at"] = now()
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
