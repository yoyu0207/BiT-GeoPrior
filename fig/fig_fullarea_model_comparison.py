from __future__ import annotations

from pathlib import Path
import csv
import string

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage


ROOT = Path(r"D:/yoyu/SA_Identification")
FIG_DIR = ROOT / "project" / "fig"

T2_PATH = ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2024.tif"
GOEP_PATH = ROOT / "Results_rebuilt" / "2020_2024_GOEP" / "2020_2024_change_binary.tif"

MODEL_GROUPS = {
    "fig_models_ablation_comparison": [
        ("G-OEP-BiT", ROOT / "Results_rebuilt" / "2020_2024_GOEP" / "2020_2024_change_binary.tif"),
        ("BiT", ROOT / "Results_rebuilt" / "2020_2024_BiT" / "2020_2024_change_binary.tif"),
        ("BiT-GWR", ROOT / "Results_rebuilt" / "2020_2024_BiT_GWR" / "2020_2024_change_binary.tif"),
        ("BiT-GWDA", ROOT / "Results_rebuilt" / "2020_2024_BiT_GWDA" / "2020_2024_change_binary.tif"),
        ("BiT-Online", ROOT / "Results_rebuilt" / "2020_2024_BiT_Online" / "2020_2024_change_binary.tif"),
    ],
    "fig_models_cd_comparison": [
        ("G-OEP-BiT", ROOT / "Results_rebuilt" / "2020_2024_GOEP" / "2020_2024_change_binary.tif"),
        ("FC-SiamDiff", ROOT / "Results_rebuilt" / "2020_2024_FCSiamDiff" / "2020_2024_change_binary.tif"),
        ("SNUNet", ROOT / "Results_rebuilt" / "2020_2024_SNUNet" / "2020_2024_change_binary.tif"),
        ("ChangeFormer", ROOT / "Results_rebuilt" / "2020_2024_ChangeFormer" / "2020_2024_change_binary.tif"),
        ("BiT", ROOT / "Results_rebuilt" / "2020_2024_BiT" / "2020_2024_change_binary.tif"),
    ],
}

N_PATCHES = 4
MIN_SEPARATION = 2500.0
OUT_SIZE = 720


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    bands = []
    for i in range(3):
        band = arr[i].astype("float32")
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(valid, [2, 98])
        bands.append(np.clip((band - lo) / (hi - lo + 1e-6), 0, 1))
    return np.dstack(bands)


def add_panel_label(ax, text: str) -> None:
    ax.text(
        0.025,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
    )


def add_scale_bar(ax, length_px: float, label: str = "1 km") -> None:
    x0 = OUT_SIZE * 0.62
    x1 = x0 + length_px
    y = OUT_SIZE * 0.91
    ax.plot([x0, x1], [y, y], color="white", lw=2.8, solid_capstyle="butt")
    ax.plot([x0, x1], [y, y], color="black", lw=1.0, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2,
        y - OUT_SIZE * 0.04,
        label,
        ha="center",
        va="bottom",
        fontsize=7.3,
        fontweight="bold",
        color="white",
    )


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.92, 0.90),
        xytext=(0.92, 0.77),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", shrinkA=0, shrinkB=0),
        bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.78),
    )


def mask_boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    dilated = ndimage.binary_dilation(mask, iterations=iterations)
    eroded = ndimage.binary_erosion(mask, iterations=iterations)
    return dilated ^ eroded


def draw_overlay(ax, bg: np.ndarray, mask: np.ndarray) -> None:
    ax.imshow(bg)
    rgba = np.zeros((*mask.shape, 4), dtype="float32")
    rgba[..., 0] = 0.88
    rgba[..., 1] = 0.73
    rgba[..., 2] = 0.22
    rgba[..., 3] = np.where(mask, 0.52, 0.0)
    ax.imshow(rgba)

    edge = mask_boundary(mask)
    shadow_rgba = np.zeros((*mask.shape, 4), dtype="float32")
    shadow_rgba[..., 0] = 0.18
    shadow_rgba[..., 1] = 0.16
    shadow_rgba[..., 2] = 0.12
    shadow_rgba[..., 3] = np.where(edge, 0.42, 0.0)
    ax.imshow(shadow_rgba)

    edge_rgba = np.zeros((*mask.shape, 4), dtype="float32")
    edge_rgba[..., 0] = 0.99
    edge_rgba[..., 1] = 0.97
    edge_rgba[..., 2] = 0.91
    edge_rgba[..., 3] = np.where(edge, 0.82, 0.0)
    ax.imshow(edge_rgba)


def build_candidate_patches(mask: np.ndarray, width: int, height: int) -> list[dict]:
    labeled, n_labels = ndimage.label(mask)
    slices = ndimage.find_objects(labeled)
    candidates: list[dict] = []

    for label_idx in range(1, n_labels + 1):
        slc = slices[label_idx - 1]
        if slc is None:
            continue
        local = labeled[slc] == label_idx
        area = int(local.sum())
        if area < 20000:
            continue

        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        ys, xs = np.where(local)
        cx = x0 + float(xs.mean())
        cy = y0 + float(ys.mean())

        bbox_w = x1 - x0
        bbox_h = y1 - y0
        size = int(np.ceil(max(bbox_w, bbox_h) * 1.55))
        size = max(950, min(1800, size))
        half = size // 2

        win_x = int(round(cx)) - half
        win_y = int(round(cy)) - half
        win_x = max(0, min(win_x, width - size))
        win_y = max(0, min(win_y, height - size))

        candidates.append(
            {
                "label_id": label_idx,
                "area_px": area,
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "window_x": win_x,
                "window_y": win_y,
                "window_size": size,
            }
        )

    candidates.sort(key=lambda item: item["area_px"], reverse=True)
    return candidates


def select_patches(candidates: list[dict], n_patches: int = N_PATCHES) -> list[dict]:
    selected: list[dict] = []
    for cand in candidates:
        too_close = False
        for chosen in selected:
            dist = ((cand["centroid_x"] - chosen["centroid_x"]) ** 2 + (cand["centroid_y"] - chosen["centroid_y"]) ** 2) ** 0.5
            if dist < MIN_SEPARATION:
                too_close = True
                break
        if too_close:
            continue
        selected.append(cand)
        if len(selected) == n_patches:
            break
    return selected


def save_patch_metadata(patches: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = FIG_DIR / "comparison_patch_metadata.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_tag",
                "area_px",
                "centroid_x",
                "centroid_y",
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "window_x",
                "window_y",
                "window_size",
            ],
        )
        writer.writeheader()
        for idx, patch in enumerate(patches):
            row = {
                "row_tag": string.ascii_lowercase[idx],
                "area_px": patch["area_px"],
                "centroid_x": patch["centroid_x"],
                "centroid_y": patch["centroid_y"],
                "bbox_x0": patch["bbox_x0"],
                "bbox_y0": patch["bbox_y0"],
                "bbox_x1": patch["bbox_x1"],
                "bbox_y1": patch["bbox_y1"],
                "window_x": patch["window_x"],
                "window_y": patch["window_y"],
                "window_size": patch["window_size"],
            }
            writer.writerow(row)


def render_group(
    figure_name: str,
    model_specs: list[tuple[str, Path]],
    patches: list[dict],
    metres_per_pixel: float,
) -> None:
    n_rows = len(patches)
    n_cols = len(model_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.55 * n_cols, 1.7 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.035, "hspace": 0.001},
    )

    for ax, title in zip(axes[0], [name for name, _ in model_specs]):
        ax.set_title(title, fontsize=8.6, pad=5, fontweight="bold")

    with rasterio.open(T2_PATH) as t2_src:
        model_srcs = [rasterio.open(path) for _, path in model_specs]
        try:
            for row_idx, patch in enumerate(patches):
                window = Window(patch["window_x"], patch["window_y"], patch["window_size"], patch["window_size"])
                t2_rgb = t2_src.read(
                    [1, 2, 3],
                    window=window,
                    out_shape=(3, OUT_SIZE, OUT_SIZE),
                    resampling=Resampling.bilinear,
                )
                bg = stretch_rgb(t2_rgb)

                for col_idx, ((_, _), src) in enumerate(zip(model_specs, model_srcs)):
                    mask = (
                        src.read(
                            1,
                            window=window,
                            out_shape=(OUT_SIZE, OUT_SIZE),
                            resampling=Resampling.nearest,
                        )
                        == 1
                    )
                    ax = axes[row_idx, col_idx]
                    draw_overlay(ax, bg, mask)
                    add_panel_label(ax, f"{string.ascii_lowercase[row_idx]}{col_idx + 1}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.55)
                        spine.set_color("#303030")

                km_pixels = 1000.0 / (metres_per_pixel * patch["window_size"] / OUT_SIZE)
                add_scale_bar(axes[row_idx, -1], km_pixels, "1 km")
                if row_idx == 0:
                    add_north_arrow(axes[row_idx, -1])
        finally:
            for src in model_srcs:
                src.close()

    pred_patch = mpatches.Patch(
        facecolor=(0.88, 0.73, 0.22, 0.52),
        edgecolor="none",
        label="Predicted eradication area",
    )
    boundary_patch = mpatches.Patch(
        facecolor=(0.99, 0.97, 0.91, 0.82),
        edgecolor="none",
        label="Prediction boundary",
    )
    fig.legend(
        handles=[pred_patch, boundary_patch],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.4,
        columnspacing=1.8,
        fontsize=8.2,
    )

    fig.subplots_adjust(left=0.01, right=0.92, top=0.905, bottom=0.07)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{figure_name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{figure_name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{figure_name}.tif", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()

    for _, models in MODEL_GROUPS.items():
        for model_name, path in models:
            if not path.exists():
                raise FileNotFoundError(f"Missing raster for {model_name}: {path}")

    with rasterio.open(GOEP_PATH) as src:
        mask = src.read(1) == 1
        candidates = build_candidate_patches(mask, src.width, src.height)
        patches = select_patches(candidates, n_patches=N_PATCHES)
        if len(patches) < N_PATCHES:
            raise RuntimeError(f"Only selected {len(patches)} patches; expected {N_PATCHES}.")
        metres_per_pixel = abs(src.transform.a)

    save_patch_metadata(patches)

    for fig_name, models in MODEL_GROUPS.items():
        render_group(fig_name, models, patches, metres_per_pixel)


if __name__ == "__main__":
    main()
