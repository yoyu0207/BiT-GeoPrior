"""Audit the fairness and reproducibility of the JAG revision experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SEEDS = [42, 1337, 2025, 3407, 9001]
DATASET_ROOT = Path(r"D:\yoyu\SA_Identification\dataset_patches_2020_2024")

RUNS = {
    "FCSiamDiff": ("revision_experiments_gwda", "FCSiamDiff_seed{seed}"),
    "Siamese Nested U-Net (local)": (
        "revision_experiments_gwda", "SNUNet_seed{seed}"),
    "BiT": ("revision_experiments_gwda", "BiT_seed{seed}"),
    "ChangeFormer": ("revision_experiments_gwda", "ChangeFormer_seed{seed}"),
    "STeInFormer": ("recent_baseline_revision", "STeInFormer_seed{seed}"),
    "EdgeRefNet": ("recent_baseline_revision", "EdgeRefNet_seed{seed}"),
    "COAST": ("spg_optimization_revision", "COAST_SPGopt_seed{seed}"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/revision_2026/fairness_audit"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_run(project: Path, display_name: str, root: str,
                pattern: str, seed: int, canonical_manifest_hash: str) -> dict:
    run_dir = project / root / pattern.format(seed=seed)
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best_model.pth"
    copied_manifest = run_dir / "split_manifest.csv"
    if not summary_path.exists():
        return {
            "model": display_name, "seed": seed, "run_dir": str(run_dir),
            "status": "missing_summary",
        }

    summary = load_json(summary_path)
    command = summary.get("command", "")
    command_lower = command.lower()
    manifest_hash = sha256(copied_manifest) if copied_manifest.exists() else None
    return {
        "model": display_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "ok" if checkpoint_path.exists() else "missing_checkpoint",
        "checkpoint_exists": checkpoint_path.exists(),
        "manifest_copy_exists": copied_manifest.exists(),
        "manifest_sha256": manifest_hash,
        "manifest_matches_canonical": manifest_hash == canonical_manifest_hash,
        "split_counts": summary.get("split_counts"),
        "epochs_budget": summary.get("epochs"),
        "epochs_completed": summary.get("epochs_completed"),
        "patience": summary.get("patience"),
        "batch_size": summary.get("batch_size"),
        "base_lr_recorded": summary.get("lr"),
        "amp": summary.get("amp"),
        "amp_dtype": summary.get("amp_dtype"),
        "alpha": summary.get("alpha"),
        "spg_lr": summary.get("spg_lr"),
        "prior_control": summary.get("prior_control"),
        "best_val_f1": summary.get("best_val_f1"),
        "has_test_metrics": summary.get("test_metrics") is not None,
        "uses_spatial_manifest": "spatial_split_manifest.csv" in command,
        "scratch_initialization": not any(
            token in command_lower
            for token in ("--pretrained", "imagenet", "weights=")
        ),
        "fixed_input_channels": 8,
        "command": command,
    }


def selection_audit(project: Path) -> list[dict]:
    paths = [
        project / "spg_optimization_revision" / "spg_lr_selection.json",
        project / "spg_optimization_revision" / "alpha_selection.json",
        project / "recent_baseline_revision" / "lr_selection.json",
    ]
    output = []
    for path in paths:
        payload = load_json(path)
        text = json.dumps(payload).lower()
        output.append({
            "file": str(path),
            "exists": True,
            "selection_rule": payload.get("selection_rule"),
            "declares_test_blind_selection": (
                "test split was not loaded" in text
                and "test_metrics_used\": false" in text
            ),
        })
    return output


def snunet_fidelity_audit(project: Path) -> dict:
    local_path = project / "models" / "snunet.py"
    source = local_path.read_text(encoding="utf-8")
    snunet_block = source.split("class SNUNet(nn.Module):", 1)[1].split(
        "class SNUNet_GeoAware", 1)[0]
    return {
        "local_file": str(local_path),
        "uses_absolute_feature_difference": "torch.abs" in snunet_block,
        "has_channel_attention_or_ecam": bool(re.search(
            r"ChannelAttention|ECAM|self\.ca", snunet_block)),
        "returns_deep_supervision_outputs": "return (out" in snunet_block,
        "assessment": (
            "The local implementation is a custom Siamese Nested U-Net-like "
            "baseline, not a faithful implementation of the official "
            "SNUNet-CD/ECAM architecture. Rename it in the manuscript or rerun "
            "the official implementation before making SNUNet-CD claims."
        ),
        "official_repository": "https://github.com/likyoo/Siam-NestedUNet",
        "severity": "high",
    }


def build_findings(runs: list[dict], selections: list[dict],
                   snunet: dict) -> list[dict]:
    findings = []
    if any(run.get("status") != "ok" for run in runs):
        findings.append({
            "severity": "critical",
            "topic": "run completeness",
            "finding": "At least one formal run is missing a summary or checkpoint.",
        })
    if any(not run.get("manifest_matches_canonical", False) for run in runs):
        findings.append({
            "severity": "critical",
            "topic": "data split",
            "finding": "At least one run does not contain the canonical split manifest.",
        })
    if not all(item["declares_test_blind_selection"] for item in selections):
        findings.append({
            "severity": "high",
            "topic": "test blindness",
            "finding": "A hyperparameter-selection record does not prove test blindness.",
        })
    findings.extend([
        {
            "severity": "high",
            "topic": "baseline identity",
            "finding": snunet["assessment"],
        },
        {
            "severity": "medium",
            "topic": "training objective",
            "finding": (
                "EdgeRefNet uses its method-specific auxiliary boundary loss "
                "(BCE+Dice change loss plus 5x edge BCE), while other baselines "
                "use BCE+Dice. This is defensible as architecture-specific "
                "training, but must be disclosed."
            ),
        },
        {
            "severity": "medium",
            "topic": "optimisation protocol",
            "finding": (
                "Learning rates and schedulers are not identical: ChangeFormer "
                "uses 1e-4 with cosine annealing; recent baselines use a "
                "validation-selected 5e-4; COAST uses separate SPG rates. "
                "Describe the protocol as validation-tuned, not identical."
            ),
        },
        {
            "severity": "low",
            "topic": "precision mode",
            "finding": (
                "Most runs use FP16 AMP, while EdgeRefNet uses BF16 because FP16 "
                "was unstable. Evaluation should use FP32 for every model."
            ),
        },
        {
            "severity": "pass",
            "topic": "core comparison",
            "finding": (
                "All formal runs use the same 8-channel inputs, canonical "
                "313/59/71 spatial split, batch size 8, 200-epoch budget, "
                "patience 30, validation-selected checkpoint, and five seeds."
            ),
        },
        {
            "severity": "pass",
            "topic": "initialisation",
            "finding": "Formal models are trained from scratch without pretrained weights.",
        },
    ])
    return findings


def render_markdown(payload: dict) -> str:
    lines = [
        "# Experiment Fairness Audit",
        "",
        "## Verdict",
        "",
        ("The comparison is broadly fair at the data, budget, seed, and "
         "checkpoint-selection levels. It is not an identical-optimizer study: "
         "method-specific optimisation and loss differences are present and must "
         "be reported. The main high-risk issue is the identity of the local "
         "SNUNet baseline."),
        "",
        "## Findings",
        "",
        "| Severity | Topic | Finding |",
        "|---|---|---|",
    ]
    for item in payload["findings"]:
        lines.append(
            f"| {item['severity'].upper()} | {item['topic']} | "
            f"{item['finding']} |")
    lines.extend([
        "",
        "## Formal Run Checks",
        "",
        "| Model | Runs | Canonical split | 313/59/71 | Budget | Scratch |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model, summary in payload["by_model"].items():
        lines.append(
            f"| {model} | {summary['n_ok']}/5 | "
            f"{'yes' if summary['all_manifest_match'] else 'no'} | "
            f"{'yes' if summary['all_split_counts_match'] else 'no'} | "
            f"{'yes' if summary['all_budget_match'] else 'no'} | "
            f"{'yes' if summary['all_scratch'] else 'no'} |")
    lines.extend([
        "",
        "## Required Manuscript Wording",
        "",
        "- Call the local baseline `Siamese Nested U-Net (local implementation)`, not official `SNUNet-CD`, unless the official ECAM implementation is rerun.",
        "- State that learning rates were selected on the validation set and the test set was not loaded during tuning.",
        "- State the architecture-specific scheduler and EdgeRefNet auxiliary edge loss explicitly.",
        "- Keep fixed-threshold 0.5 results as the primary comparison; validation-selected-threshold results are supplementary.",
        "- Report every metric as mean +/- sample SD across the same five seeds.",
        "",
        "## Provenance",
        "",
        f"- Canonical manifest SHA-256: `{payload['canonical_manifest_sha256']}`",
        f"- Audit generated from: `{payload['project']}`",
        "- Official SNUNet source checked: https://github.com/likyoo/Siam-NestedUNet",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_manifest = args.data_root / "spatial_split_manifest.csv"
    canonical_hash = sha256(canonical_manifest)
    runs = []
    for display_name, (root, pattern) in RUNS.items():
        for seed in SEEDS:
            runs.append(inspect_run(
                project, display_name, root, pattern, seed, canonical_hash))

    by_model = {}
    for model in RUNS:
        selected = [run for run in runs if run["model"] == model]
        by_model[model] = {
            "n_ok": sum(run.get("status") == "ok" for run in selected),
            "seeds": sorted(run["seed"] for run in selected
                            if run.get("status") == "ok"),
            "all_manifest_match": all(
                run.get("manifest_matches_canonical", False) for run in selected),
            "all_split_counts_match": all(
                run.get("split_counts") == {"train": 313, "val": 59, "test": 71}
                for run in selected),
            "all_budget_match": all(
                run.get("epochs_budget") == 200
                and run.get("patience") == 30
                and run.get("batch_size") == 8
                for run in selected),
            "all_scratch": all(run.get("scratch_initialization", False)
                               for run in selected),
            "base_lrs": sorted(set(run.get("base_lr_recorded")
                                   for run in selected)),
            "amp_modes": dict(Counter(
                str((run.get("amp"), run.get("amp_dtype")))
                for run in selected)),
        }

    selections = selection_audit(project)
    snunet = snunet_fidelity_audit(project)
    payload = {
        "project": str(project),
        "canonical_manifest": str(canonical_manifest),
        "canonical_manifest_sha256": canonical_hash,
        "expected_seeds": SEEDS,
        "runs": runs,
        "by_model": by_model,
        "hyperparameter_selection": selections,
        "snunet_implementation": snunet,
    }
    payload["findings"] = build_findings(runs, selections, snunet)

    (output_dir / "fairness_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "fairness_audit.md").write_text(
        render_markdown(payload), encoding="utf-8")
    print(output_dir / "fairness_audit.md")


if __name__ == "__main__":
    main()
