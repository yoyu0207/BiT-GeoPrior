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

YEAR_SPECS = [
    (
        "2023",
        ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2023.tif",
        ROOT / "Results_rebuilt" / "2020_2023_GOEP" / "2020_2023_change_binary.tif",
    ),
    (
        "2024",
        ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2024.tif",
        ROOT / "Results_rebuilt" / "2020_2024_GOEP" / "2020_2024_change_binary.tif",
    ),
    (
        "2025",
        ROOT / "Dataset" / "images" / "Sentinel2_8ch_Stack_2025.tif",
        ROOT / "Results_rebuilt" / "2020_2025_GOEP" / "2020_2025__change_binary.tif",
    ),
]

N_PATCHES = 3
MIN_SEPARATION = 2200.0
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


def mask_boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    dilated = ndimage.binary_dilation(mask, iterations=iterations)
    eroded = ndimage.binary_erosion(mask, iterations=iterations)
    return dilated ^ eroded


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
        if area < 30000:
            continue

        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        ys, xs = np.where(local)
        cx = x0 + float(xs.mean())
        cy = y0 + float(ys.mean())

        bbox_w = x1 - x0
        bbox_h = y1 - y0
        size = int(np.ceil(max(bbox_w, bbox_h) * 1.55))
        size = max(1000, min(1850, size))
        half = size // 2

        win_x = int(round(cx)) - half
        win_y = int(round(cy)) - half
        win_x = max(0, min(win_x, width - size))
        win_y = max(0, min(win_y, height - size))

        candidates.append(
            {
                "area_px": area,
                "centroid_x": cx,
                "centroid_y": cy,
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
    out_csv = FIG_DIR / "transfer_patch_metadata.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_tag", "area_px", "centroid_x", "centroid_y", "window_x", "window_y", "window_size"],
        )
        writer.writeheader()
        for idx, patch in enumerate(patches):
            writer.writerow(
                {
                    "row_tag": string.ascii_lowercase[idx],
                    "area_px": patch["area_px"],
                    "centroid_x": patch["centroid_x"],
                    "centroid_y": patch["centroid_y"],
                    "window_x": patch["window_x"],
                    "window_y": patch["window_y"],
                    "window_size": patch["window_size"],
                }
            )


def main() -> None:
    setup_style()
    for _, img_path, mask_path in YEAR_SPECS:
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

    with rasterio.open(YEAR_SPECS[0][2]) as src_2023, rasterio.open(YEAR_SPECS[1][2]) as src_2024, rasterio.open(YEAR_SPECS[2][2]) as src_2025:
        union_mask = (src_2023.read(1) == 1) | (src_2024.read(1) == 1) | (src_2025.read(1) == 1)
        candidates = build_candidate_patches(union_mask, src_2024.width, src_2024.height)
        patches = select_patches(candidates, n_patches=N_PATCHES)
        if len(patches) < N_PATCHES:
            raise RuntimeError(f"Only selected {len(patches)} patches; expected {N_PATCHES}.")
        metres_per_pixel = abs(src_2024.transform.a)

    save_patch_metadata(patches)

    fig, axes = plt.subplots(
        N_PATCHES,
        len(YEAR_SPECS),
        figsize=(1.6 * len(YEAR_SPECS), 1.72 * N_PATCHES),
        squeeze=False,
        gridspec_kw={"wspace": -0.05, "hspace": 0.03},
    )

    for ax, (year, _, _) in zip(axes[0], YEAR_SPECS):
        ax.set_title(year, fontsize=8.8, pad=5, fontweight="bold")

    image_srcs = [rasterio.open(img) for _, img, _ in YEAR_SPECS]
    mask_srcs = [rasterio.open(msk) for _, _, msk in YEAR_SPECS]
    try:
        for row_idx, patch in enumerate(patches):
            window = Window(patch["window_x"], patch["window_y"], patch["window_size"], patch["window_size"])
            for col_idx, (year, _, _) in enumerate(YEAR_SPECS):
                img = image_srcs[col_idx].read(
                    [1, 2, 3],
                    window=window,
                    out_shape=(3, OUT_SIZE, OUT_SIZE),
                    resampling=Resampling.bilinear,
                )
                bg = stretch_rgb(img)
                mask = (
                    mask_srcs[col_idx].read(
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
        for src in image_srcs + mask_srcs:
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
        fontsize=8.1,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.075)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_transfer_spatial_comparison.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_transfer_spatial_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_transfer_spatial_comparison.tif", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
