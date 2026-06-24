from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(r"D:/yoyu/SA_Identification")
FIG_DIR = ROOT / "project" / "fig"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def add_box(ax, x, y, w, h, text, fc, ec="#3A3A3A", lw=1.0, fs=9, weight="normal", z=2):
    box = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, joinstyle="round", zorder=z)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, zorder=z + 1)
    return box


def add_arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.2, style="-|>", ms=12, cs="arc3"):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=cs,
        zorder=3,
    )
    ax.add_patch(arr)
    return arr


def add_note(ax, x, y, text, fs=8, color="#4A4A4A", ha="center", va="center", style="italic"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, style=style, zorder=4)


def main() -> None:
    setup_style()

    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # Title
    ax.text(
        50,
        57.2,
        "Detailed structure of the online ecological prior module in G-OEP-BiT",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # Stage bands
    add_box(ax, 3, 42.2, 30, 11.5, "Online ecological prior encoder", fc="#EFE7D7", lw=1.1, fs=12, weight="bold")
    add_box(ax, 36.5, 42.2, 23, 11.5, "Prior-to-feature coupling", fc="#E6ECF5", lw=1.1, fs=12, weight="bold")
    add_box(ax, 63, 42.2, 33.5, 11.5, "BiT feature modulation and decoding", fc="#E9F0E4", lw=1.1, fs=11.2, weight="bold")

    # Inputs
    add_box(ax, 5, 25.5, 12.5, 8, "T1 Sentinel-2 stack\n[B, 8, H, W]", fc="#F6F6F6", fs=10, weight="bold")
    add_note(ax, 11.25, 22.7, "B8 / B4 / B3 / B2 / NDVI / EVI / SAVI / GNDVI", fs=8.3)

    add_box(ax, 69, 10.5, 11.5, 7.4, "T1 image", fc="#F8F8F8", fs=10, weight="bold")
    add_box(ax, 83.5, 10.5, 11.5, 7.4, "T2 image", fc="#F8F8F8", fs=10, weight="bold")

    # Online prior encoder branch
    add_box(ax, 22, 27.2, 12, 6.5, "Eco-score layer\n1 x 1 Conv\n8 -> 1 + Sigmoid", fc="#F6DDB8", fs=9.5, weight="bold")
    add_box(ax, 38, 27.2, 12.3, 6.5, "Spatial diffusion-1\n3 x 3 Conv\n1 -> 16", fc="#F9E9C9", fs=9.2, weight="bold")
    add_box(ax, 53.7, 27.2, 12.3, 6.5, "Spatial diffusion-2\n3 x 3 Conv, d = 4\n16 -> 16", fc="#F9E9C9", fs=9.2, weight="bold")
    add_box(ax, 69.4, 27.2, 12.3, 6.5, "Spatial diffusion-3\n3 x 3 Conv, d = 8\n16 -> 1", fc="#F9E9C9", fs=9.2, weight="bold")
    add_box(ax, 85.1, 27.2, 10.2, 6.5, "Prior map\n[B, 1, H, W]\nSigmoid", fc="#F3CDA8", fs=9.6, weight="bold")

    add_note(ax, 44.15, 23.9, "BN + ReLU", fs=8.1)
    add_note(ax, 59.85, 23.9, "BN + ReLU", fs=8.1)
    add_note(ax, 59.8, 20.8, "Multi-scale receptive fields approximate local ecological diffusion", fs=8.3)

    # Training supervision
    add_box(ax, 84.3, 34.8, 11.2, 3.2, "GWDA posterior\n(teacher prior)", fc="#F7E6EE", fs=8.8, weight="bold")
    add_arrow(ax, 89.9, 34.6, 89.9, 31.4, color="#9B4D72", lw=1.0)
    add_note(ax, 92.2, 31.8, "distillation target\n(training only)", fs=8, color="#8C456A", ha="left")

    # Prior to SPG
    add_box(ax, 42.3, 10.2, 14.6, 8.0, "Resize prior to BiT\nfeature resolution\nbilinear interpolation", fc="#D7E3F4", fs=9.6, weight="bold")
    add_box(ax, 43.4, 2.0, 12.4, 5.8, "1 x 1 Conv\n1 -> 128\n+ Sigmoid", fc="#C9D8EE", fs=9.3, weight="bold")

    # BiT feature branches
    add_box(ax, 67.4, 20.8, 14.1, 8.2, "Shared BiT encoder\n+ projection +\nTransformer block", fc="#D8E6D3", fs=9.5, weight="bold")
    add_box(ax, 83.0, 20.8, 14.1, 8.2, "Shared BiT encoder\n+ projection +\nTransformer block", fc="#D8E6D3", fs=9.5, weight="bold")

    add_box(ax, 68.0, 2.0, 13.0, 5.8, "SPG on T1 feature\nF' = F + gamma(F * A)", fc="#BDD7B5", fs=9.1, weight="bold")
    add_box(ax, 83.8, 2.0, 13.0, 5.8, "SPG on T2 feature\nF' = F + gamma(F * A)", fc="#BDD7B5", fs=9.1, weight="bold")
    add_note(ax, 75.1, -0.4, "gamma initialized to 0 for identity behaviour at training start", fs=8, color="#51724E")

    add_box(ax, 75.7, 30.8, 14.3, 4.7, "Concatenate\nmodulated T1/T2 features", fc="#E5EFCF", fs=9.1, weight="bold")
    add_box(ax, 75.7, 36.3, 14.3, 4.9, "Upsample x16\n3 x 3 Conv 256 -> 64\n1 x 1 Conv -> output", fc="#DDE9C9", fs=8.7, weight="bold")
    add_box(ax, 76.7, 42.0, 12.3, 3.9, "Eradication\nprobability map", fc="#C9DEB3", fs=9.9, weight="bold")

    # Arrows main prior branch
    add_arrow(ax, 17.5, 29.5, 22.0, 30.4)
    add_arrow(ax, 34.0, 30.4, 38.0, 30.4)
    add_arrow(ax, 50.3, 30.4, 53.7, 30.4)
    add_arrow(ax, 66.0, 30.4, 69.4, 30.4)
    add_arrow(ax, 81.7, 30.4, 85.1, 30.4)

    # Prior down to coupling
    add_arrow(ax, 90.2, 27.0, 49.6, 18.4, cs="arc3,rad=0.06")
    add_arrow(ax, 49.6, 10.0, 49.6, 7.9)

    # T1/T2 to encoders
    add_arrow(ax, 74.7, 17.9, 74.5, 20.6)
    add_arrow(ax, 89.2, 17.9, 90.0, 20.6)

    # Encoders to SPG
    add_arrow(ax, 74.5, 20.6, 74.5, 8.0)
    add_arrow(ax, 90.0, 20.6, 90.0, 8.0)

    # Attention to both SPGs
    add_arrow(ax, 55.8, 4.9, 67.9, 4.9)
    add_arrow(ax, 55.8, 4.9, 83.8, 4.9, cs="arc3,rad=-0.02")
    add_note(ax, 61.7, 7.5, "channel-wise attention A", fs=8.1, color="#48607F")

    # To decoder / output
    add_arrow(ax, 74.5, 7.9, 81.0, 30.8, cs="arc3,rad=0.15")
    add_arrow(ax, 90.0, 7.9, 84.7, 30.8, cs="arc3,rad=-0.15")
    add_arrow(ax, 82.85, 35.6, 82.85, 36.1)
    add_arrow(ax, 82.85, 41.3, 82.85, 41.8)

    # Extra explanatory labels
    add_note(ax, 28.0, 35.8, "Step 1: learn ecological suitability from T1 channels", fs=8.4)
    add_note(ax, 61.5, 35.8, "Step 2: smooth and diffuse the prior in space", fs=8.4)
    add_note(ax, 49.5, 14.8, "prior aligned to H/16 x W/16", fs=8.2)
    add_note(ax, 82.8, 52.4, "inference uses only T1/T2 images; no external prior map is required", fs=8.0, color="#556B2F")

    out_png = FIG_DIR / "fig_online_prior_module_detailed.png"
    out_pdf = FIG_DIR / "fig_online_prior_module_detailed.pdf"
    out_tif = FIG_DIR / "fig_online_prior_module_detailed.tif"

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_tif, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
