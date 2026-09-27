"""Run OOF-dependent controls after the main OOF GWDA experiment pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_oof_gwda_experiments import (
    SEEDS,
    command_for,
    load_summary,
    run_one,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--main_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--poll_seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.main_root = args.main_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "pipeline_status.json"
    write_json(status_path, {"status": "waiting", "stage": "main_oof_pipeline"})

    alpha_path = args.main_root / "alpha_selection.json"
    main_status_path = args.main_root / "pipeline_status.json"
    while True:
        main_status = (
            json.loads(main_status_path.read_text(encoding="utf-8"))
            if main_status_path.exists() else {}
        )
        if main_status.get("stage") == "completed" and alpha_path.exists():
            break
        if main_status.get("status") in {"failed", "blocked"}:
            raise RuntimeError(f"Main OOF pipeline did not complete: {main_status}")
        time.sleep(args.poll_seconds)

    alpha_info = json.loads(alpha_path.read_text(encoding="utf-8"))
    selected_alpha = float(alpha_info["selected_alpha"])
    controls = [
        ("COAST_shuffled_OOF", "shuffled", False),
        ("COAST_no_gating_OOF", None, True),
    ]
    reuse_independent = abs(selected_alpha - 0.2) < 1e-12
    if not reuse_independent:
        controls.extend([
            ("COAST_zero_OOF", "zero", False),
            ("COAST_constant_OOF", "constant", False),
            ("COAST_random_per_epoch_OOF", "random_per_epoch", False),
        ])

    failures = []
    write_json(status_path, {
        "status": "running",
        "stage": "controls",
        "selected_alpha": selected_alpha,
        "reusing_prior_independent_controls": reuse_independent,
        "controls": [item[0] for item in controls],
    })
    for experiment, prior_control, no_gating in controls:
        for seed in SEEDS:
            run_name = f"{experiment}_seed{seed}"
            command = command_for(
                args,
                "BiT_Online",
                run_name,
                seed,
                alpha=selected_alpha,
                prior_control=prior_control,
                no_gating=no_gating,
            )
            result = run_one(args, command, run_name)
            if result["status"] == "failed":
                failures.append(result)
            write_json(status_path, {
                "status": "running",
                "stage": "controls",
                "selected_alpha": selected_alpha,
                "last_run": result,
                "failures": failures,
            })

    summaries = {}
    for experiment, _, _ in controls:
        values = []
        for seed in SEEDS:
            summary = load_summary(args.output_root / f"{experiment}_seed{seed}" / "summary.json")
            if summary:
                values.append({
                    "seed": seed,
                    "best_val_f1": summary["best_val_f1"],
                    "test_metrics": summary.get("test_metrics"),
                })
        summaries[experiment] = values
    write_json(args.output_root / "control_results.json", {
        "selected_alpha": selected_alpha,
        "prior_independent_controls_reused": reuse_independent,
        "reuse_reason": (
            "zero, constant, and random-per-epoch overwrite the loaded prior and use the same alpha"
            if reuse_independent else None
        ),
        "runs": summaries,
        "failures": failures,
    })
    expected = len(controls) * len(SEEDS)
    completed = sum(len(value) for value in summaries.values())
    write_json(status_path, {
        "status": "completed" if completed == expected else "completed_with_failures",
        "stage": "completed",
        "selected_alpha": selected_alpha,
        "runs_completed": completed,
        "runs_expected": expected,
        "failures": failures,
    })


if __name__ == "__main__":
    main()
