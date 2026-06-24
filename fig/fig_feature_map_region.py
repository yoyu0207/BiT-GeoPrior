from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TORCH_LIB = Path(r"D:\develop\miniconda3\envs\SA\Lib\site-packages\torch\lib")
if TORCH_LIB.is_dir():
    os.add_dll_directory(str(TORCH_LIB))

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
import torch


ROOT = Path(r"D:/yoyu/SA_Identification")
PROJECT_DIR = ROOT / "project"
FIG_DIR = PROJECT_DIR / "fig"

sys.path.insert(0, str(PROJECT_DIR))

from models.bit_online import BiT_Online


T1_PATH = ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2020.tif"
T2_PATH = ROOT / "Dataset" / "images" / "Sentinel2_Phenology_Stack_2024.tif"
CKPT_PATH = PROJECT_DIR / "checkpoints_BiT_Online_GWDA_0502_2252" / "best_model_GOEP.pth"

# A compact coastal subregion selected from the large-patch area (row a),
# chosen to visually match the user's screenshot as closely as possible.
WIN_X = 12350
WIN_Y = 16450
WIN_SIZE = 800


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def read_window(path: Path, x: int, y: int, size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(list(range(1, 9)), window=Window(x, y, size, size)).astype("float32")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    out = np.zeros_like(patch, dtype="float32")
    for c in range(patch.shape[0]):
        band = patch[c].astype("float32")
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            out[c] = 0.0
        else:
            out[c] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def false_colour(arr8: np.ndarray) -> np.ndarray:
    rgb = arr8[[0, 1, 2]].transpose(1, 2, 0)
    out = np.zeros_like(rgb, dtype="float32")
    for i in range(3):
        band = rgb[..., i]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(valid, [2, 98])
        out[..., i] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def robust_map(feat: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(feat, [2, 98])
    return np.clip((feat - lo) / (hi - lo + 1e-6), 0, 1)


def add_label(ax, text: str) -> None:
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8),
    )


def add_scale_bar(ax, size_px: int, label: str = "1 km") -> None:
    # Sentinel-2 at 10 m; 1 km = 100 px
    length_px = 100
    x0 = size_px * 0.67
    x1 = x0 + length_px
    y = size_px * 0.92
    ax.plot([x0, x1], [y, y], color="white", lw=2.6, solid_capstyle="butt")
    ax.plot([x0, x1], [y, y], color="black", lw=0.9, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2,
        y - size_px * 0.035,
        label,
        ha="center",
        va="bottom",
        fontsize=7.1,
        fontweight="bold",
        color="white",
    )


def load_model() -> BiT_Online:
    device = torch.device("cpu")
    model = BiT_Online(in_channels=8, num_classes=1).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def extract_maps(model: BiT_Online, t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = torch.from_numpy(normalize_patch(t1)).unsqueeze(0).float()
    x2 = torch.from_numpy(normalize_patch(t2)).unsqueeze(0).float()

    with torch.no_grad():
        prior = model.prior_encoder(x1)
        f1 = model.spg1(model._encode(x1), prior)
        f2 = model.spg2(model._encode(x2), prior)

    prior_map = prior.squeeze().cpu().numpy()
    fmap1 = torch.linalg.vector_norm(f1.squeeze(0), ord=2, dim=0).cpu().numpy()
    fmap2 = torch.linalg.vector_norm(f2.squeeze(0), ord=2, dim=0).cpu().numpy()
    return robust_map(prior_map), robust_map(fmap1), robust_map(fmap2)


def main() -> None:
    setup_style()

    t1 = read_window(T1_PATH, WIN_X, WIN_Y, WIN_SIZE)
    t2 = read_window(T2_PATH, WIN_X, WIN_Y, WIN_SIZE)

    rgb1 = false_colour(t1)
    rgb2 = false_colour(t2)

    model = load_model()
    prior_map, fmap1, fmap2 = extract_maps(model, t1, t2)

    out_npz = FIG_DIR / "feature_map_region_a_subwindow.npz"
    np.savez_compressed(
        out_npz,
        t1_false_colour=rgb1,
        t2_false_colour=rgb2,
        prior_map=prior_map,
        feature_map_2020=fmap1,
        feature_map_2024=fmap2,
        window=np.array([WIN_X, WIN_Y, WIN_SIZE], dtype=np.int32),
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.6), constrained_layout=True)

    axes[0, 0].imshow(rgb1)
    axes[0, 0].set_title("2020 false-colour composite", fontsize=10, fontweight="bold", pad=6)
    add_label(axes[0, 0], "a1")

    im0 = axes[0, 1].imshow(prior_map, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("Online prior map $P_{online}$", fontsize=10, fontweight="bold", pad=6)
    add_label(axes[0, 1], "a2")

    im1 = axes[0, 2].imshow(fmap1, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("2020 feature map", fontsize=10, fontweight="bold", pad=6)
    add_label(axes[0, 2], "a3")

    axes[1, 0].imshow(rgb2)
    axes[1, 0].set_title("2024 false-colour composite", fontsize=10, fontweight="bold", pad=6)
    add_label(axes[1, 0], "b1")
    add_scale_bar(axes[1, 0], WIN_SIZE)

    axes[1, 1].imshow(prior_map, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Online prior map $P_{online}$", fontsize=10, fontweight="bold", pad=6)
    add_label(axes[1, 1], "b2")

    im2 = axes[1, 2].imshow(fmap2, cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("2024 feature map", fontsize=10, fontweight="bold", pad=6)
    add_label(axes[1, 2], "b3")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    cbar_prior = fig.colorbar(im0, ax=axes[:, 1], fraction=0.03, pad=0.02)
    cbar_prior.set_label("$P_{online}$", fontsize=8.5)
    cbar_prior.ax.tick_params(labelsize=7.5)

    cbar_feat = fig.colorbar(im2, ax=axes[:, 2], fraction=0.03, pad=0.02)
    cbar_feat.set_label("Normalized feature intensity", fontsize=8.5)
    cbar_feat.ax.tick_params(labelsize=7.5)

    fig.suptitle("G-OEP-BiT feature response in a representative coastal subregion", fontsize=12, fontweight="bold", y=1.01)

    out_png = FIG_DIR / "fig_feature_map_region_a_subwindow.png"
    out_pdf = FIG_DIR / "fig_feature_map_region_a_subwindow.pdf"
    out_tif = FIG_DIR / "fig_feature_map_region_a_subwindow.tif"

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_tif, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
