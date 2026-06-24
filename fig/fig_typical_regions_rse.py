from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage


ROOT = Path(r"D:/yoyu/SA_Identification")
OUT_DIR = ROOT / "project" / "fig"

T1_PATH = ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2020.tif"
T2_PATH = ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2024.tif"
PRED_PATH = ROOT / "Results" / "Prediction_2020_2024_Governance0330.tif"
REF_PATH = ROOT / "Dataset" / "label" / "label001.tif"

REGIONS = [
    {
        "name": "Region I",
        "subtitle": "large irregular patch",
        "x": 12000,
        "y": 16400,
        "size": 1600,
    },
    {
        "name": "Region II",
        "subtitle": "linear tidal-flat edge",
        "x": 0,
        "y": 2100,
        "size": 1400,
    },
    {
        "name": "Region III",
        "subtitle": "fragmented tidal-channel landscape",
        "x": 15200,
        "y": 26800,
        "size": 1600,
    },
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def read_rgb(src: rasterio.DatasetReader, window: Window, out_size: int = 720) -> np.ndarray:
    return src.read(
        [1, 2, 3],
        window=window,
        out_shape=(3, out_size, out_size),
        resampling=Resampling.bilinear,
    ).astype("float32")


def stretch_pair(t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([t1, t2], axis=1)
    out = []
    for arr in (t1, t2):
        bands = []
        for i in range(3):
            valid = stacked[i][np.isfinite(stacked[i]) & (stacked[i] > 0)]
            if valid.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = np.percentile(valid, [2, 98])
            bands.append(np.clip((arr[i] - lo) / (hi - lo + 1e-6), 0, 1))
        out.append(np.dstack(bands))
    return out[0], out[1]


def mask_boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    dilated = ndimage.binary_dilation(mask, iterations=iterations)
    eroded = ndimage.binary_erosion(mask, iterations=iterations)
    return dilated ^ eroded


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.025,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="square,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
    )


def add_scale_bar(ax, metres_per_pixel: float, out_size: int, length_m: int = 2000) -> None:
    length_px = length_m / metres_per_pixel
    x0 = out_size * 0.09
    y0 = out_size * 0.91
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", lw=2.6, solid_capstyle="butt")
    ax.plot([x0, x0 + length_px], [y0, y0], color="black", lw=1.05, solid_capstyle="butt")
    ax.text(
        x0 + length_px / 2,
        y0 - out_size * 0.035,
        "2 km",
        color="white",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        path_effects=[],
    )


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.925, 0.91),
        xytext=(0.925, 0.785),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", shrinkA=0, shrinkB=0),
        bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.75),
    )


def draw_overlay(ax, base_rgb: np.ndarray, pred: np.ndarray) -> None:
    ax.imshow(base_rgb)
    red = np.zeros((*pred.shape, 4), dtype=float)
    red[..., 0] = 0.93
    red[..., 1] = 0.12
    red[..., 2] = 0.10
    red[..., 3] = np.where(pred, 0.46, 0.0)
    ax.imshow(red)
    edge = mask_boundary(pred)
    edge_rgba = np.zeros((*pred.shape, 4), dtype=float)
    edge_rgba[..., :3] = 1.0
    edge_rgba[..., 3] = np.where(edge, 0.85, 0.0)
    ax.imshow(edge_rgba)


def draw_agreement(ax, base_rgb: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> None:
    gray = np.mean(base_rgb, axis=2)
    ax.imshow(np.dstack([gray, gray, gray]), vmin=0, vmax=1)
    rgba = np.zeros((*pred.shape, 4), dtype=float)
    tp = pred & ref
    fp = pred & ~ref
    fn = ~pred & ref
    rgba[tp] = (0.78, 0.10, 0.10, 0.72)
    rgba[fp] = (0.95, 0.62, 0.18, 0.72)
    rgba[fn] = (0.14, 0.38, 0.86, 0.72)
    ax.imshow(rgba)


def plot() -> None:
    setup_style()
    out_size = 720
    panel_labels = [
        ["a1", "a2", "a3", "a4"],
        ["b1", "b2", "b3", "b4"],
        ["c1", "c2", "c3", "c4"],
    ]

    fig, axes = plt.subplots(3, 4, figsize=(7.15, 5.9))
    column_titles = [
        "T1 false-colour composite (2020)",
        "T2 false-colour composite (2024)",
        "Predicted eradication area",
        "Prediction-reference agreement",
    ]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=8.5, pad=5, fontweight="bold")

    with rasterio.open(T1_PATH) as t1_src, rasterio.open(T2_PATH) as t2_src, rasterio.open(
        PRED_PATH
    ) as pred_src, rasterio.open(REF_PATH) as ref_src:
        metres_per_pixel = abs(t1_src.transform.a) * REGIONS[0]["size"] / out_size

        for row, region in enumerate(REGIONS):
            size = region["size"]
            window = Window(region["x"], region["y"], size, size)
            t1_raw = read_rgb(t1_src, window, out_size=out_size)
            t2_raw = read_rgb(t2_src, window, out_size=out_size)
            t1_rgb, t2_rgb = stretch_pair(t1_raw, t2_raw)
            pred = (
                pred_src.read(
                    1,
                    window=window,
                    out_shape=(out_size, out_size),
                    resampling=Resampling.nearest,
                )
                == 1
            )
            ref = (
                ref_src.read(
                    1,
                    window=window,
                    out_shape=(out_size, out_size),
                    resampling=Resampling.nearest,
                )
                == 1
            )

            axes[row, 0].imshow(t1_rgb)
            axes[row, 1].imshow(t2_rgb)
            draw_overlay(axes[row, 2], t2_rgb, pred)
            draw_agreement(axes[row, 3], t2_rgb, pred, ref)

            axes[row, 0].text(
                -0.06,
                0.5,
                f"{region['name']}\n{region['subtitle']}",
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=8.1,
                fontweight="bold",
            )
            add_scale_bar(axes[row, 1], metres_per_pixel, out_size)
            if row == 0:
                add_north_arrow(axes[row, 1])

            for col in range(4):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                add_panel_label(ax, panel_labels[row][col])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.55)
                    spine.set_color("#303030")

    pred_patch = mpatches.Patch(facecolor=(0.93, 0.12, 0.10, 0.46), edgecolor="none", label="Prediction")
    tp_patch = mpatches.Patch(facecolor="#C71F1F", edgecolor="none", label="TP")
    fp_patch = mpatches.Patch(facecolor="#F29E2E", edgecolor="none", label="FP")
    fn_patch = mpatches.Patch(facecolor="#245FDA", edgecolor="none", label="FN")
    fig.legend(
        handles=[pred_patch, tp_patch, fp_patch, fn_patch],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.55, 0.013),
        handlelength=1.4,
        columnspacing=1.55,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.095, right=0.992, top=0.93, bottom=0.075, wspace=0.045, hspace=0.08)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "tif"):
        fig.savefig(OUT_DIR / f"fig_typical_regions_rse.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
