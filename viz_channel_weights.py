"""
viz_channel_weights.py — EcologicalPriorEncoder 通道权重可视化
======================================================================
加载训练好的 BiT_Online 权重，提取 eco_score 层的 8 通道权重，
画成柱状图，直观展示各波段/指数对 Spartina 生态适宜性的贡献度。

用法：
    python viz_channel_weights.py --pth checkpoints_BiT_Online_0502_1558/best_model.pth  --out fig7_channel_weights.png
======================================================================
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

import numpy as np
from models.bit_online import BiT_Online

rcParams['font.family']      = 'Times New Roman'
rcParams['axes.labelsize']   = 10
rcParams['xtick.labelsize']  = 9
rcParams['ytick.labelsize']  = 9

# 8 通道名称（与训练时输入顺序严格对应）
CHANNEL_NAMES = ['B8\n(NIR)', 'B4\n(Red)', 'B3\n(Green)', 'B2\n(Blue)',
                 'NDVI', 'EVI', 'SAVI', 'GNDVI']

# 各通道对应颜色（光谱波段用蓝色系，植被指数用绿色系）
COLORS = ['#BFDFD2', '#51999F', '#4198AC', '#7BC0CD',
          '#DBCB92', '#ECB66C', '#EA9E58', '#ED8D5A']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, required=True,
                        help='best_model.pth 路径')
    parser.add_argument('--out', type=str,
                        default='fig7_channel_weights.png')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cpu')

    # 加载模型
    model = BiT_Online(in_channels=8, num_classes=1)
    state = torch.load(args.pth, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 提取通道权重
    weights = model.prior_encoder.get_channel_weights().numpy()  # [8]

    # 归一化到 [-1, 1] 方便对比正负贡献
    abs_max = np.abs(weights).max()
    weights_norm = weights / (abs_max + 1e-8)

    print("=== eco_score 通道权重 ===")
    for name, w, wn in zip(CHANNEL_NAMES, weights, weights_norm):
        bar = '█' * int(abs(wn) * 20)
        sign = '+' if w >= 0 else '-'
        print(f"  {name.replace(chr(10),' '):12s}  {sign}{abs(w):.5f}  {bar}")

    # ── 图 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={'width_ratios': [2, 1]})
    fig.subplots_adjust(wspace=0.35)

    # 左图：原始权重柱状图
    ax = axes[0]
    x  = np.arange(len(CHANNEL_NAMES))
    bars = ax.bar(x, weights, color=COLORS, edgecolor='white',
                  linewidth=0.6, zorder=3)

    # 正负分界线
    ax.axhline(0, color='#888780', lw=0.8, ls='--', zorder=2)

    # 数值标注
    for bar, w in zip(bars, weights):
        va     = 'bottom' if w >= 0 else 'top'
        offset = 0.0003 if w >= 0 else -0.0003
        ax.text(bar.get_x() + bar.get_width() / 2,
                w + offset, f'{w:.4f}',
                ha='center', va=va, fontsize=7.5,
                fontfamily='Times New Roman', color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, fontfamily='Times New Roman')
    ax.set_ylabel('Learned weight', fontfamily='Times New Roman')
    ax.set_title('(a) eco_score channel weights',
                 fontfamily='Times New Roman', pad=6)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 右图：绝对值大小排序（贡献度排名）
    ax2  = axes[1]
    idx  = np.argsort(np.abs(weights))[::-1]
    names_sorted  = [CHANNEL_NAMES[i].replace('\n', ' ') for i in idx]
    weights_sorted = np.abs(weights)[idx]
    colors_sorted  = [COLORS[i] for i in idx]

    ax2.barh(range(len(names_sorted)), weights_sorted,
             color=colors_sorted, edgecolor='white',
             linewidth=0.6, zorder=3)
    ax2.set_yticks(range(len(names_sorted)))
    ax2.set_yticklabels(names_sorted, fontfamily='Times New Roman')
    ax2.set_xlabel('|Weight| (contribution rank)',
                   fontfamily='Times New Roman')
    ax2.set_title('(b) Contribution ranking',
                  fontfamily='Times New Roman', pad=6)
    ax2.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.4, zorder=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 图注
    fig.text(
        0.5, -0.04,
        "Fig. 7.  Learned channel weights of the EcologicalPriorEncoder eco_score layer. "
        "(a) Raw weights — positive values indicate spectral features positively correlated "
        "with Spartina ecological suitability; negative values indicate inhibitory channels. "
        "(b) Absolute weight ranking — channels with higher |weight| contribute more to the "
        "prior map estimation.",
        ha='center', fontsize=8, fontfamily='Times New Roman',
        color='#444444', style='italic', wrap=True
    )

    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.savefig(args.out.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"\n已保存：{args.out}")


if __name__ == '__main__':
    main()
