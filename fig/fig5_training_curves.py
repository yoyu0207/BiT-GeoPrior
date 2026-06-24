"""
fig5_training_curves.py
────────────────────────────────────────────────────────────────────────────
生成 Fig. 5：所有模型的 Val F1 训练曲线对比图
读取各模型的 training_log.csv 文件

使用方法：
  1. 把 LOG_PATHS 里的路径改成你实际的 checkpoints 文件夹名
  2. python fig5_training_curves.py
  3. 输出：fig5_training_curves.pdf / .png
────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# ── 字体设置 ──────────────────────────────────────────────────────────────
rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = 10
rcParams['axes.labelsize']  = 7
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 7

# ══════════════════════════════════════════════════════════════════════════
#  配置 — 修改为你实际的 CSV 路径
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"D:/yoyu/SA_Identification/project"

LOG_PATHS = {
    "FC-SiamDiff":  "checkpoints_FCSiamDiff_0404_2345/training_log_fc.csv",
    "SNUNet":       "checkpoints_SNUNet_0404_2117/training_log_snunet.csv",
    "SNUNet-GWR":   "checkpoints_SNUNet_GeoAware_0404_2218/training_log_snunet_gwr.csv",
    "ChangeFormer": "00checkpoints_ChangeFormer_0404_1906/training_log_changeformer.csv",
    "BiT":          "00checkpoints_BiT_0404_1736/training_log_bit.csv",
    "BiT-GWR":      "checkpoints_BiT_GWR_0405_0009/training_log_bit_gwr.csv",
}

# 线条样式（颜色/线型/粗细）
STYLES = {
    "FC-SiamDiff":  {"color": "#5EBD9D", "ls": "--",  "lw": 1.4, "zorder": 2},
    "SNUNet":       {"color": "#70C4F5", "ls": "-.",  "lw": 1.4, "zorder": 2},
    "SNUNet-GWR":   {"color": "#2058AC", "ls": ":",   "lw": 1.6, "zorder": 3},
    "ChangeFormer": {"color": "#F3D46D", "ls": "--",  "lw": 1.4, "zorder": 2},
    "BiT":          {"color": "#EC9E5E", "ls": "-",   "lw": 1.8, "zorder": 4},
    "BiT-GWR":      {"color": "#D85A30", "ls": "-",   "lw": 2.5, "zorder": 5},
}

OUT_PATH = "fig5_training_curves"
# ══════════════════════════════════════════════════════════════════════════


def smooth(values, window=5):
    """简单滑动平均平滑，减少小数据集导致的曲线噪声"""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def main():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    best_marks = {}   # 记录每条曲线最高点，用于标注

    for name, rel_path in LOG_PATHS.items():
        csv_path = os.path.join(PROJECT_DIR, rel_path)
        if not os.path.exists(csv_path):
            print(f"⚠ 找不到 {csv_path}，跳过 {name}")
            continue

        df   = pd.read_csv(csv_path)
        ep   = df["epoch"].values
        f1   = df["val_f1"].values
        f1_s = smooth(f1, window=3)   # 轻微平滑，保留趋势

        st = STYLES[name]
        ax.plot(ep, f1_s, label=name,
                color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], zorder=st["zorder"], alpha=0.95)

        # 标记最优点
        best_idx = int(np.argmax(f1))
        best_ep  = ep[best_idx]
        best_f1  = f1[best_idx]
        best_marks[name] = (best_ep, best_f1, st["color"])

    # 标注 BiT-GWR 和 BiT 的最优点
    for name in ["BiT", "BiT-GWR"]:
        if name not in best_marks:
            continue
        bep, bf1, bc = best_marks[name]
        ax.scatter(bep, bf1, color=bc, s=40, zorder=6, clip_on=False)
        # offset = 0.003 if name == "BiT" else -0.006
        # ax.annotate(
        #     f"{name}: {bf1:.4f}",
        #     xy=(bep, bf1), xytext=(bep - 18, bf1 + offset),
        #     fontsize=7.5, fontfamily="Times New Roman", color=bc,
        #     arrowprops=dict(arrowstyle="-", color=bc, lw=0.8)
        # )

    # 100 epoch 分界线（CNN 训练终止）
    ax.axvline(x=100, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(101, ax.get_ylim()[0] + 0.005, "CNN\nend",
            fontsize=7, fontfamily="Times New Roman",
            color="gray", va="bottom")

    ax.set_xlabel("Epoch", fontfamily="Times New Roman")
    ax.set_ylabel("Validation F1", fontfamily="Times New Roman")

    ax.set_xlim(0, 250)
    ax.set_ylim(0.40, 0.960)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(
        loc="lower right", frameon=True, framealpha=0.9,
        edgecolor="lightgray", ncol=2,
        prop={"family": "Times New Roman", "size": 8.5}
    )

    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT_PATH}.png", dpi=300, bbox_inches="tight")
    print(f"已保存 {OUT_PATH}.pdf / .png")


if __name__ == "__main__":
    main()
