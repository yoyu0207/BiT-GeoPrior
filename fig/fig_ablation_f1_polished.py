import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


ROOT = Path(r"D:/yoyu/SA_Identification/project")
OUT_DIR = ROOT / "fig"

LOGS = {
    "BiT": ROOT / "checkpoints" / "00checkpoints_BiT_0404_1736" / "training_log_bit.csv",
    "BiT-GWR": ROOT
    / "checkpoints"
    / "00checkpoints_BiT_GWR_0405_0009(有csv)"
    / "training_log_bit_gwr.csv",
    "BiT-GWDA": ROOT / "checkpoints_BiT_GWDA_0502_1440" / "training_log_gwda.csv",
    "OEP-BiT": ROOT / "checkpoints_BiT_Online_0502_1558" / "training_log_online.csv",
    "G-OEP-BiT": ROOT
    / "checkpoints_BiT_Online_GWDA_0502_2252"
    / "training_log_gwda_online.csv",
}

STYLES = {
    "BiT": {"color": "#7A7A7A", "ls": "-", "lw": 1.7, "marker": "o"},
    "BiT-GWR": {"color": "#FAB650", "ls": "-", "lw": 1.9, "marker": "s"},
    "BiT-GWDA": {"color": "#F07D5A", "ls": "-", "lw": 2.0, "marker": "^"},
    "OEP-BiT": {"color": "#5DA5BB", "ls": "-", "lw": 2.2, "marker": "D"},
    "G-OEP-BiT": {"color": "#1F55A7", "ls": "-", "lw": 2.4, "marker": "P"},
}


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return y
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def read_curves() -> dict[str, pd.DataFrame]:
    curves = {}
    for name, path in LOGS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["val_f1_smooth"] = smooth(df["val_f1"].to_numpy(float), window=5)
        curves[name] = df
    return curves


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def plot() -> None:
    setup_style()
    curves = read_curves()

    fig, ax = plt.subplots(figsize=(7.05, 3.95))
    ax.set_facecolor("white")

    for name, df in curves.items():
        st = STYLES[name]
        epoch = df["epoch"].to_numpy()
        f1 = df["val_f1"].to_numpy()
        f1_s = df["val_f1_smooth"].to_numpy()

        ax.plot(
            epoch,
            f1,
            color=st["color"],
            lw=0.65,
            alpha=0.16,
            zorder=1,
        )
        ax.plot(
            epoch,
            f1_s,
            color=st["color"],
            lw=st["lw"],
            ls=st["ls"],
            label=name,
            solid_capstyle="round",
            zorder=3,
        )

        best_idx = int(np.argmax(f1))
        ax.scatter(
            epoch[best_idx],
            f1[best_idx],
            s=28,
            marker=st["marker"],
            facecolor="white",
            edgecolor=st["color"],
            linewidth=1.1,
            zorder=5,
        )

    ax.axvspan(0, 25, color="#F2F2F2", zorder=0)
    ax.text(
        12.5,
        0.944,
        "rapid\nadaptation",
        ha="center",
        va="top",
        color="#666666",
        fontsize=7,
    )
    ax.axhline(0.929196, color="#9A9A9A", lw=0.85, ls=(0, (2.2, 2.2)), zorder=0)
    ax.text(
        3,
        0.9315,
        "BiT best F1",
        color="#666666",
        fontsize=7,
        ha="left",
        va="bottom",
    )

    ax.set_xlim(0, 200)
    ax.set_ylim(0.80, 0.948)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1-score")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#DADADA", lw=0.55, ls="-", zorder=0)
    ax.grid(axis="x", color="#ECECEC", lw=0.45, ls="-", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8, length=3.2)

    handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[name]["color"],
            lw=STYLES[name]["lw"],
            marker=STYLES[name]["marker"],
            markerfacecolor="white",
            markeredgewidth=1,
            markersize=4.6,
            label=name,
        )
        for name in LOGS
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#CFCFCF",
        borderpad=0.55,
        handlelength=2.3,
        columnspacing=1.25,
    )

    axins = inset_axes(ax, width="36%", height="37%", loc="center right", borderpad=1.2)
    for name, df in curves.items():
        st = STYLES[name]
        axins.plot(
            df["epoch"],
            df["val_f1_smooth"],
            color=st["color"],
            lw=st["lw"] * 0.86,
            solid_capstyle="round",
        )
    axins.set_xlim(150, 200)
    axins.set_ylim(0.918, 0.944)
    axins.xaxis.set_major_locator(mticker.MultipleLocator(25))
    axins.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    axins.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    axins.tick_params(labelsize=6.6, length=2.2, width=0.65, direction="out")
    axins.grid(axis="y", color="#E5E5E5", lw=0.45)
    for spine in axins.spines.values():
        spine.set_color("#BFBFBF")
        spine.set_linewidth(0.65)
    axins.set_title("late-stage convergence", fontsize=7, pad=2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#BFBFBF", lw=0.7)

    summary = []
    for name, df in curves.items():
        row = df.loc[df["val_f1"].idxmax()]
        summary.append((name, int(row["epoch"]), float(row["val_f1"])))
    summary_text = "Best F1: " + "  ".join(
        f"{name} {f1:.4f} (E{epoch})" for name, epoch, f1 in summary
    )
    fig.text(0.015, 0.012, summary_text, ha="left", va="bottom", fontsize=6.8, color="#4F4F4F")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.987, top=0.965, bottom=0.16)
    for suffix in ("png", "pdf", "tif"):
        fig.savefig(OUT_DIR / f"fig_ablation_f1_polished.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
