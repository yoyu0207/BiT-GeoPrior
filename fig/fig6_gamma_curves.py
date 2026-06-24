"""
fig6_gamma_curves.py
────────────────────────────────────────────────────────────────────────────
生成 Fig. 6：BiT-GWR 和 SNUNet-GWR 的 γ 参数学习曲线
两个子图并排：(a) BiT-GWR，(b) SNUNet-GWR

使用方法：
  1. 把 LOG_PATHS 里的路径改成你实际的 checkpoints 文件夹名
  2. python fig6_gamma_curves.py
  3. 输出：fig6_gamma_curves.pdf / .png
────────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# ── 字体设置 ──────────────────────────────────────────────────────────────
rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = 10
rcParams['axes.labelsize']  = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

# ══════════════════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"D:/yoyu/SA_Identification/project"

LOG_PATHS = {
    "BiT-GWR":    "checkpoints_BiT_GWR_0405_0009/training_log_bit_gwr.csv",
    "SNUNet-GWR": "checkpoints_SNUNet_GeoAware_0404_2218/training_log_snunet_gwr.csv",
}

OUT_PATH = "fig6_gamma_curves"
# ══════════════════════════════════════════════════════════════════════════


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    configs = {
        "BiT-GWR": {
            "ax":     axes[0],
            "title":  "(a) BiT-GWR",
            "c1":     "#D85A30",    # γ₁ 颜色
            "c2":     "#F0997B",    # γ₂ 颜色
            "label":  r"$\gamma$",
            "ylabel": r"$\gamma$ value",
        },
        "SNUNet-GWR": {
            "ax":     axes[1],
            "title":  "(b) SNUNet-GWR",
            "c1":     "#1D9E75",
            "c2":     "#9FE1CB",
            "label":  r"$\gamma$",
            "ylabel": r"$\gamma$ value",
        },
    }

    for name, cfg in configs.items():
        csv_path = os.path.join(PROJECT_DIR, LOG_PATHS[name])
        if not os.path.exists(csv_path):
            print(f"⚠ 找不到 {csv_path}，跳过 {name}")
            continue

        df = pd.read_csv(csv_path)
        ep = df["epoch"].values
        g1 = df["gamma1"].values
        g2 = df["gamma2"].values

        ax = cfg["ax"]

        # γ₁ 实线，γ₂ 虚线
        ax.plot(ep, g1, color=cfg["c1"], lw=2.0, ls="-",
                label=r"$\gamma_1$ (spg1)", zorder=3)
        ax.plot(ep, g2, color=cfg["c2"], lw=2.0, ls="--",
                label=r"$\gamma_2$ (spg2)", zorder=3)

        # γ=0 参考线
        ax.axhline(y=0, color="gray", lw=0.8, ls=":", alpha=0.7)

        # 标注最终值
        # 标注最终值 - 调整为斜上方
        final_ep = ep[-1]
        # 获取当前 y 轴的范围，用于计算比例偏移，防止偏移量过大或过小
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        for i, (gval, gc, glabel) in enumerate([(g1[-1], cfg["c1"], "$\gamma_1$"), (g2[-1], cfg["c2"], "$\gamma_2$")]):
            # 设置偏移：gamma1 向上偏，gamma2 如果离得近可以向下偏或者也向上偏
            # 这里统一设置在点位的 左上方
            y_offset = y_range * 0.06  # 向上偏移 y 轴总长度的 12%
            x_offset = ep[-1] * 0.05   # 向左偏移 x 轴总长度的 15%
            
            # 如果是第二个标注且两个值靠得太近，可以微调 y_offset 避免重叠
            if i == 1 and abs(g1[-1] - g2[-1]) < y_range * 0.1:
                y_offset = -y_range * 0.12 # 向下偏
            
            ax.annotate(
                f"{glabel} = {gval:.4f}",
                xy=(final_ep, gval),
                xytext=(final_ep - x_offset, gval + y_offset), # 文字坐标：x减少，y增加
                fontsize=8, 
                fontfamily="Times New Roman", 
                color=gc,
                ha='right', # 文字右对齐，这样指向更自然
                va='center',
                arrowprops=dict(
                    arrowstyle="-", # 改成带箭头的
                    color=gc, 
                    lw=0.8,
                    # connectionstyle="arc3,rad=0.1" # 带一点点弧度更好看，不需要弧度可以删掉
                )
            )

        ax.set_title(cfg["title"], fontfamily="Times New Roman", pad=5)
        ax.set_xlabel("Epoch", fontfamily="Times New Roman")
        ax.set_ylabel(cfg["ylabel"], fontfamily="Times New Roman")
        ax.set_xlim(0, ep[-1] + 5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        leg = ax.legend(
            loc="best", frameon=True, framealpha=0.9,
            edgecolor="lightgray",
            prop={"family": "Times New Roman", "size": 8.5}
        )

        # 给 SNUNet-GWR 子图加阴影标注负值区域
        if name == "SNUNet-GWR":
            ymin = ax.get_ylim()[0]
            ax.fill_between(ep, ymin, 0, alpha=0.06, color="red",
                            label="_nolegend_")
            ax.text(ep[len(ep)//3], ymin * 0.6,
                    "Inhibitory\nmodulation (γ < 0)",
                    fontsize=7, fontfamily="Times New Roman",
                    color="red", alpha=0.7, ha="center")

    # 共同图注（在图下方）
    note = (
        "Note: γ = 0 initialisation (dashed grey line) ensures BiT-GWR is "
        "equivalent to BiT baseline at training onset. "
        "Positive γ → constructive prior amplification; "
        "Negative γ → inhibitory prior suppression."
    )
    fig.text(0.5, -0.04, note, ha="center", fontsize=7.5,
             fontfamily="Times New Roman", color="#444444",
             wrap=True)

    plt.savefig(f"{OUT_PATH}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT_PATH}.png", dpi=300, bbox_inches="tight")
    print(f"已保存 {OUT_PATH}.pdf / .png")


if __name__ == "__main__":
    main()
