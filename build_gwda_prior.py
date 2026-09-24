"""Build a leakage-safe GWDA teacher from training spatial blocks only.

The change labels in validation and test blocks are never read while fitting,
selecting the adaptive bandwidth, or calibrating the teacher. Spatial
coordinates affect only the geographic kernel; the discriminant variables are
the eight pre-eradication image channels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sample_stride", type=int, default=16)
    parser.add_argument("--prediction_stride", type=int, default=16)
    parser.add_argument("--candidate_neighbors", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--cv_folds", type=int, default=4)
    parser.add_argument("--cv_points_per_fold", type=int, default=2500)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--pixel_size_m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "region", "x", "y", "block_x", "block_y", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    return rows


def aligned_offsets(origin: int, size: int, stride: int) -> np.ndarray:
    first = (-origin) % stride
    return np.arange(first, size, stride, dtype=np.int32)


def collect_unique_training_samples(
    root: Path, rows: list[dict[str, str]], stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    samples: dict[tuple[str, int, int], tuple[np.ndarray, int, str]] = {}
    source_files = []
    for row in rows:
        if row["split"] != "train":
            continue
        name = row["filename"]
        source_files.append(name)
        x0, y0 = int(row["x"]), int(row["y"])
        image = np.load(root / "A" / name).astype(np.float32)[:8]
        label = np.load(root / "label" / name)
        ys = aligned_offsets(y0, label.shape[-2], stride)
        xs = aligned_offsets(x0, label.shape[-1], stride)
        group = f'{row["region"]}|{row["block_x"]}|{row["block_y"]}'
        for ly in ys:
            for lx in xs:
                key = (row["region"], x0 + int(lx), y0 + int(ly))
                value = (image[:, ly, lx], int(label[ly, lx] > 0.5), group)
                if key in samples:
                    old = samples[key]
                    if old[1] != value[1] or not np.allclose(old[0], value[0], atol=1e-6):
                        raise ValueError(f"Inconsistent overlapping source pixel: {key}")
                else:
                    samples[key] = value

    keys = sorted(samples)
    features = np.stack([samples[key][0] for key in keys]).astype(np.float64)
    labels = np.asarray([samples[key][1] for key in keys], dtype=np.uint8)
    coords = np.asarray([[key[1], key[2]] for key in keys], dtype=np.float64)
    groups = np.asarray([samples[key][2] for key in keys])
    return features, labels, coords, groups, sorted(source_files)


def standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def local_gwda_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_coords: np.ndarray,
    query_x: np.ndarray,
    query_coords: np.ndarray,
    neighbors: int,
    ridge: float,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(neighbors, len(train_x))
    if k < 16:
        raise ValueError("Too few GWDA fitting samples")
    tree = cKDTree(train_coords)
    probabilities = np.empty(len(query_x), dtype=np.float64)
    bandwidths = np.empty(len(query_x), dtype=np.float64)
    eye = np.eye(train_x.shape[1], dtype=np.float64)[None]

    for start in range(0, len(query_x), batch_size):
        stop = min(start + batch_size, len(query_x))
        distance, index = tree.query(query_coords[start:stop], k=k, workers=-1)
        if k == 1:
            distance, index = distance[:, None], index[:, None]
        bandwidth = np.maximum(distance[:, -1], 1.0)
        weight = np.exp(-0.5 * (distance / bandwidth[:, None]) ** 2)
        neighbor_x = train_x[index]
        neighbor_y = train_y[index]
        w1 = weight * neighbor_y
        w0 = weight * (1 - neighbor_y)
        s1 = np.maximum(w1.sum(axis=1), 1e-8)
        s0 = np.maximum(w0.sum(axis=1), 1e-8)
        mu1 = np.einsum("bk,bkd->bd", w1, neighbor_x) / s1[:, None]
        mu0 = np.einsum("bk,bkd->bd", w0, neighbor_x) / s0[:, None]
        d1 = neighbor_x - mu1[:, None, :]
        d0 = neighbor_x - mu0[:, None, :]
        cov = (
            np.einsum("bk,bki,bkj->bij", w1, d1, d1)
            + np.einsum("bk,bki,bkj->bij", w0, d0, d0)
        ) / np.maximum((s0 + s1 - 2.0)[:, None, None], 1.0)
        trace_scale = np.maximum(np.trace(cov, axis1=1, axis2=2) / cov.shape[1], 1e-6)
        cov = cov + ridge * trace_scale[:, None, None] * eye
        delta = mu1 - mu0
        try:
            direction = np.linalg.solve(cov, delta[..., None])[..., 0]
        except np.linalg.LinAlgError:
            direction = np.einsum("bij,bj->bi", np.linalg.pinv(cov), delta)
        midpoint = 0.5 * (mu1 + mu0)
        local_prior = np.log((s1 + 0.5) / (s0 + 0.5))
        score = np.einsum("bd,bd->b", query_x[start:stop] - midpoint, direction) + local_prior
        probabilities[start:stop] = sigmoid(score)
        bandwidths[start:stop] = bandwidth
    return probabilities, bandwidths


def metric_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fraction_true, fraction_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_mae": float(np.mean(np.abs(fraction_true - fraction_pred))),
    }


def stratified_limit(indices: np.ndarray, labels: np.ndarray, limit: int, rng) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    n_positive = min(len(positive), limit // 2)
    n_negative = min(len(negative), limit - n_positive)
    if n_positive + n_negative < limit:
        extra = limit - n_positive - n_negative
        if len(positive) - n_positive >= extra:
            n_positive += extra
        else:
            n_negative += extra
    chosen = np.concatenate([
        rng.choice(positive, n_positive, replace=False),
        rng.choice(negative, n_negative, replace=False),
    ])
    rng.shuffle(chosen)
    return chosen


def select_bandwidth_and_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    groups: np.ndarray,
    candidates: list[int],
    folds: int,
    points_per_fold: int,
    ridge: float,
    seed: int,
) -> tuple[int, IsotonicRegression, dict]:
    unique_groups = np.unique(groups)
    folds = min(folds, len(unique_groups))
    if folds < 2:
        raise ValueError("GWDA bandwidth selection needs at least two training blocks")
    splitter = GroupKFold(n_splits=folds)
    rng = np.random.default_rng(seed)
    fold_data = []
    for fit_index, holdout_index in splitter.split(x, y, groups):
        holdout_index = stratified_limit(holdout_index, y, points_per_fold, rng)
        mean, scale = standardize_fit(x[fit_index])
        fold_data.append((fit_index, holdout_index, mean, scale))

    scores = {}
    for neighbors in candidates:
        ys, ps, bandwidths = [], [], []
        for fit_index, holdout_index, mean, scale in fold_data:
            pred, bw = local_gwda_predict(
                (x[fit_index] - mean) / scale,
                y[fit_index],
                coords[fit_index],
                (x[holdout_index] - mean) / scale,
                coords[holdout_index],
                neighbors,
                ridge,
            )
            ys.append(y[holdout_index])
            ps.append(pred)
            bandwidths.append(bw)
        all_y, all_p = np.concatenate(ys), np.concatenate(ps)
        scores[str(neighbors)] = {
            **metric_dict(all_y, all_p),
            "median_adaptive_bandwidth_pixels": float(np.median(np.concatenate(bandwidths))),
        }

    selected = min(candidates, key=lambda value: scores[str(value)]["brier"])
    oof_y, oof_p = [], []
    for fit_index, holdout_index, mean, scale in fold_data:
        pred, _ = local_gwda_predict(
            (x[fit_index] - mean) / scale,
            y[fit_index],
            coords[fit_index],
            (x[holdout_index] - mean) / scale,
            coords[holdout_index],
            selected,
            ridge,
        )
        oof_y.append(y[holdout_index])
        oof_p.append(pred)
    oof_y, oof_p = np.concatenate(oof_y), np.concatenate(oof_p)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_p, oof_y)
    calibrated = calibrator.predict(oof_p)
    diagnostics = {
        "candidate_results": scores,
        "selected_neighbors": selected,
        "oof_points": int(len(oof_y)),
        "oof_uncalibrated": metric_dict(oof_y, oof_p),
        "oof_calibrated": metric_dict(oof_y, calibrated),
    }
    return selected, calibrator, diagnostics


def build_query_grid(
    root: Path, rows: list[dict[str, str]], stride: int
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[str, int, int]]]]:
    queries: dict[tuple[str, int, int], np.ndarray] = {}
    patch_keys: dict[str, list[tuple[str, int, int]]] = {}
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            continue
        name = row["filename"]
        image = np.load(root / "A" / name).astype(np.float32)[:8]
        x0, y0 = int(row["x"]), int(row["y"])
        ys = np.arange(0, image.shape[1], stride, dtype=np.int32)
        xs = np.arange(0, image.shape[2], stride, dtype=np.int32)
        keys = []
        for ly in ys:
            for lx in xs:
                key = (row["region"], x0 + int(lx), y0 + int(ly))
                queries.setdefault(key, image[:, ly, lx])
                keys.append(key)
        patch_keys[name] = keys
    keys = sorted(queries)
    features = np.stack([queries[key] for key in keys]).astype(np.float64)
    coords = np.asarray([[key[1], key[2]] for key in keys], dtype=np.float64)
    return features, coords, patch_keys


def bilinear_resize(values: np.ndarray, size: int = 256) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(values.astype(np.float32))[None, None]
    resized = functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def evaluate_saved_prior(root: Path, rows: list[dict[str, str]], output_dir: Path) -> dict:
    result = {}
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        ys, ps = [], []
        for row in rows:
            if row["split"] != split:
                continue
            y = np.load(root / "label" / row["filename"]).ravel()
            p = np.load(output_dir / row["filename"]).ravel()
            index = rng.choice(len(y), min(4096, len(y)), replace=False)
            ys.append(y[index])
            ps.append(p[index])
        result[split] = metric_dict(np.concatenate(ys), np.concatenate(ps))
    return result


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    manifest = (args.manifest or root / "spatial_split_manifest.csv").resolve()
    output_dir = (args.output_dir or root / "spatial_prior_gwda_train_only").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)

    x, y, coords, groups, source_files = collect_unique_training_samples(
        root, rows, args.sample_stride)
    selected, calibrator, cv = select_bandwidth_and_calibrator(
        x, y, coords, groups, args.candidate_neighbors, args.cv_folds,
        args.cv_points_per_fold, args.ridge, args.seed)
    mean, scale = standardize_fit(x)

    query_x, query_coords, patch_keys = build_query_grid(root, rows, args.prediction_stride)
    raw_probability, bandwidth = local_gwda_predict(
        (x - mean) / scale,
        y,
        coords,
        (query_x - mean) / scale,
        query_coords,
        selected,
        args.ridge,
    )
    probability = calibrator.predict(raw_probability)
    query_lookup = {
        (row["region"], int(row["x"]), int(row["y"])): value
        for row, value in zip(
            ({"region": key[0], "x": key[1], "y": key[2]} for key in sorted({key for keys in patch_keys.values() for key in keys})),
            probability,
        )
    }
    coarse_size = 256 // args.prediction_stride
    for name, keys in patch_keys.items():
        coarse = np.asarray([query_lookup[key] for key in keys], dtype=np.float32).reshape(coarse_size, coarse_size)
        np.save(output_dir / name, np.clip(bilinear_resize(coarse), 0.0, 1.0).astype(np.float32))

    source_digest = hashlib.sha256("\n".join(source_files).encode("utf-8")).hexdigest()
    metadata = {
        "method": "geographically_weighted_linear_discriminant_analysis",
        "fit_split": "train_only",
        "label_source": "change reference labels from training spatial blocks only",
        "feature_source": "eight pre-eradication image channels",
        "coordinate_role": "adaptive Gaussian geographic weights only",
        "manifest": str(manifest),
        "training_source_file_count": len(source_files),
        "training_source_files_sha256": source_digest,
        "unique_training_sample_count": int(len(y)),
        "positive_training_sample_count": int(y.sum()),
        "negative_training_sample_count": int((1 - y).sum()),
        "sample_stride_pixels": args.sample_stride,
        "prediction_stride_pixels": args.prediction_stride,
        "kernel": "adaptive Gaussian truncated to k nearest samples",
        "ridge": args.ridge,
        "standardization_mean": mean.tolist(),
        "standardization_scale": scale.tolist(),
        "pixel_size_m": args.pixel_size_m,
        "median_final_bandwidth_pixels": float(np.median(bandwidth)),
        "median_final_bandwidth_m": float(np.median(bandwidth) * args.pixel_size_m),
        "calibration": "isotonic regression fitted to spatial-block out-of-fold training predictions",
        "calibration_x": calibrator.X_thresholds_.tolist(),
        "calibration_y": calibrator.y_thresholds_.tolist(),
        "bandwidth_selection": cv,
        "teacher_metrics": evaluate_saved_prior(root, rows, output_dir),
        "seed": args.seed,
    }
    (output_dir / "gwda_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
