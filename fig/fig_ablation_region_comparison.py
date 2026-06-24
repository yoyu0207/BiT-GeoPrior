from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


ROOT = Path(r"D:/yoyu/SA_Identification")
OUT_DIR = ROOT / "project" / "fig"

T2_PATH = ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2024.tif"
REF_PATH = ROOT / "Dataset" / "label" / "label001.tif"

# Fill in the prediction rasters you want to compare.
# The current G-OEP-BiT result is pre-filled; add other model outputs here after inference.
MODEL_RASTERS = {
    "Reference": REF_PATH,
    # "BiT": ROOT / "Results" / "Prediction_2020_2024_BiT.tif",
    # "BiT-GWR": ROOT / "Results" / "Prediction_2020_2024_BiT_GWR.tif",
    # "BiT-GWDA": ROOT / "Results" / "Prediction_2020_2024_BiT_GWDA.tif",
    # "OEP-BiT": ROOT / "Results" / "Prediction_2020_2024_OEP_BiT.tif",
    "G-OEP-BiT": ROOT / "Results" / "Prediction_2020_2024_Governance0330.tif",
}

REGIONS = [
    ("Region I", "large irregular patch", "a", 12000, 16400, 1600),
    ("Region II", "linear tidal-flat edge", "b", 0, 2100, 1400),
    ("Region III", "fragmented tidal-channel landscape", "c", 15200, 26800, 1600),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    bands = []
    for i in range(3):
        valid = arr[i][np.isfinite(arr[i]) & (arr[i] > 0)]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        bands.append(np.clip((arr[i] - lo) / (hi - lo + 1e-6), 0, 1))
    return np.dstack(bands)


def add_panel_tag(ax, text: str) -> None:
    ax.text(
        0.03,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.84),
    )


def draw_overlay(ax, rgb: np.ndarray, mask: np.ndarray, is_reference: bool) -> None:
    ax.imshow(rgb)
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    if is_reference:
        rgba[..., 0] = 0.63
        rgba[..., 1] = 0.77
        rgba[..., 2] = 0.95
        rgba[..., 3] = np.where(mask, 0.55, 0.0)
    else:
        rgba[..., 0] = 0.90
        rgba[..., 1] = 0.20
        rgba[..., 2] = 0.16
        rgba[..., 3] = np.where(mask, 0.50, 0.0)
    ax.imshow(rgba)


def main() -> None:
    setup_style()
    valid_models = [(name, path) for name, path in MODEL_RASTERS.items() if Path(path).exists()]
    if len(valid_models) < 2:
        raise FileNotFoundError("Add at least one model prediction raster besides the reference.")

    n_rows = len(REGIONS)
    n_cols = 1 + len(valid_models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.25 * n_cols, 2.2 * n_rows), squeeze=False)

    with rasterio.open(T2_PATH) as t2_src:
        for col, title in enumerate(["T2 (2024)"] + [name for name, _ in valid_models]):
            axes[0, col].set_title(title, fontsize=8.6, fontweight="bold", pad=5)

        for row, (region, pattern, prefix, x, y, size) in enumerate(REGIONS):
            window = Window(x, y, size, size)
            t2 = t2_src.read([1, 2, 3], window=window, out_shape=(3, 640, 640), resampling=Resampling.bilinear)
            rgb = stretch_rgb(t2)

            axes[row, 0].imshow(rgb)
            axes[row, 0].text(
                -0.07,
                0.5,
                f"{region}\n{pattern}",
                transform=axes[row, 0].transAxes,
                rotation=90,
                ha="right",
                va="center",
                fontsize=8.1,
                fontweight="bold",
            )
            add_panel_tag(axes[row, 0], f"{prefix}0")

            for col, (name, path) in enumerate(valid_models, start=1):
                with rasterio.open(path) as src:
                    mask = (
                        src.read(
                            1,
                            window=window,
                            out_shape=(640, 640),
                            resampling=Resampling.nearest,
                        )
                        == 1
                    )
                draw_overlay(axes[row, col], rgb, mask, is_reference=(name == "Reference"))
                add_panel_tag(axes[row, col], f"{prefix}{col}")

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.55)
                    spine.set_color("#303030")

    fig.text(
        0.012,
        0.012,
        "Suggested use: keep this figure compact and compare only the strongest representative ablation variants in the main text.",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color="#51606E",
    )
    fig.subplots_adjust(left=0.09, right=0.992, top=0.93, bottom=0.07, wspace=0.03, hspace=0.08)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_ablation_region_comparison.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_ablation_region_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
