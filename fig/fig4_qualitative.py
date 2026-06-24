"""
fig4_qualitative.py
────────────────────────────────────────────────────────────────────────────
生成 Fig. 4：6 个典型场景的定性对比图
每行 6 列：T1 / T2 / GT / SNUNet / BiT / BiT-GWR
红色 = 漏检（FN），绿色 = 误检（FP），白色 = TP，黑色 = TN

使用方法：
  1. 在 SCENE_LIST 里填入你想展示的 6 个 patch stem（文件名不含 .npy）
     前 3 个是治理场景，后 3 个是再入侵场景
  2. 填入各模型的 best_model.pth 路径
  3. python fig4_qualitative.py
  4. 输出：fig4_qualitative.pdf（和 .png）
────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ── 字体设置 ──────────────────────────────────────────────────────────────
rcParams['font.family']      = 'Times New Roman'
rcParams['axes.titlesize']   = 14
rcParams['axes.labelsize']   = 13
rcParams['xtick.labelsize']  = 12
rcParams['ytick.labelsize']  = 12

# ══════════════════════════════════════════════════════════════════════════
#  配置（按实际情况修改）
# ══════════════════════════════════════════════════════════════════════════
DATA_ROOT = r"D:/yoyu/SA_Identification/dataset_patches_2020_2024"

# 填入 6 个 patch stem，前 3 治理，后 3 再入侵
SCENE_LIST = [
    "EstuariesA_768_128",   # 治理场景 1
    "EstuariesA_12800_17024",   # 治理场景 1
    "EstuariesA_12288_16896",     # 治理场景 2
    "EstuariesA_16640_27776",   # 再入侵场景 3
]

MODEL_PTNS = {
    "SNUNet":    r"D:/yoyu/SA_Identification/project/checkpoints_SNUNet_0404_2117/best_model.pth",
    "BiT":       r"D:/yoyu/SA_Identification/project/00checkpoints_BiT_0404_1736/best_model.pth",
    "BiT-GWR":   r"D:/yoyu/SA_Identification/project/00checkpoints_BiT_GWR_0404_1635/best_model.pth",
}

OUT_PATH  = "fig4_qualitative"
THRESHOLD = 0.5

# RGB 合成波段索引（在 8 通道里：B8=0, B4=1, B3=2, B2=3）
# False-colour: NIR-Red-Green = 通道 0,1,2
RGB_BANDS = [0, 1, 2]

PRIOR_MODELS = {"BiT_GWR", "SNUNet_GeoAware"}
# ══════════════════════════════════════════════════════════════════════════

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ── 模型加载 ──────────────────────────────────────────────────────────────
def load_model(name, pth_path, device):
    from models.snunet import SNUNet
    from models.bit import BiT
    from models.bit_gwr import BiT_GWR

    if name == "SNUNet":
        m = SNUNet(in_channels=8, num_classes=1)
    elif name == "BiT":
        m = BiT(in_channels=8, num_classes=1)
    elif name == "BiT-GWR":
        m = BiT_GWR(in_channels=8, num_classes=1)
    else:
        raise ValueError(f"Unknown model: {name}")

    state = torch.load(pth_path, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m


def infer_patch(model, name, imgA, imgB, prior, device):
    """单 patch 推理，返回二值预测图 [H, W]"""
    with torch.no_grad():
        a = torch.tensor(imgA[None], dtype=torch.float32).to(device)
        b = torch.tensor(imgB[None], dtype=torch.float32).to(device)
        p = torch.tensor(prior[None, None], dtype=torch.float32).to(device)

        if name in ("BiT-GWR",):
            out = model(a, b, p)
        else:
            out = model(a, b)

        if isinstance(out, list):
            out = out[-1]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    return (prob > THRESHOLD).astype(np.uint8)


# ── 影像预处理 ────────────────────────────────────────────────────────────
def norm_img(arr):
    """归一化到 [0,1] 供显示"""
    mn, mx = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - mn) / (mx - mn + 1e-6), 0, 1)


def make_rgb(patch):
    """从 8 通道 patch 取 false-colour RGB"""
    r = norm_img(patch[RGB_BANDS[0]])
    g = norm_img(patch[RGB_BANDS[1]])
    b = norm_img(patch[RGB_BANDS[2]])
    return np.stack([r, g, b], axis=-1)


def make_error_map(pred, gt):
    """
    返回 RGB 误差图：
      白  = TP（正确检测到变化）
      黑  = TN（正确无变化）
      红  = FN（漏检）
      绿  = FP（误检）
    """
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    tp = (pred == 1) & (gt == 1)
    tn = (pred == 0) & (gt == 0)
    fn = (pred == 0) & (gt == 1)   # 漏检 → 红
    fp = (pred == 1) & (gt == 0)   # 误检 → 绿

    rgb[tp] = [1.0, 1.0, 1.0]      # 白
    rgb[tn] = [0.0, 0.0, 0.0]      # 黑
    rgb[fn] = [1.0, 0.0, 0.0]      # 红
    rgb[fp] = [0.0, 0.8, 0.0]      # 绿
    return rgb


def make_gt_display(gt):
    """GT 显示：白=变化，黑=无变化"""
    rgb = np.stack([gt.astype(float)] * 3, axis=-1)
    return rgb


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print("加载模型...")
    models = {}
    for name, pth in MODEL_PTNS.items():
        if not os.path.exists(pth):
            print(f"  ⚠ 找不到 {pth}，跳过")
            continue
        models[name] = load_model(name, pth, device)
        print(f"  {name} 加载完成")

    n_scenes  = len(SCENE_LIST)
    col_names = ["T1 (false-colour)", "T2 (false-colour)", "Ground truth"] + list(models.keys())
    n_cols    = len(col_names)

    fig, axes = plt.subplots(
        n_scenes, n_cols,
        figsize=(n_cols * 2.2, n_scenes * 2.2),
        squeeze=False
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.04,
                        top=0.96, bottom=0.06, left=0.06, right=0.98)

    # 列标题
    for j, cname in enumerate(col_names):
        axes[0, j].set_title(cname, fontsize=9, fontfamily="Times New Roman", pad=4)

    for i, stem in enumerate(SCENE_LIST):
        print(f"场景 {i+1}/{n_scenes}: {stem}")

        # 读取数据
        imgA  = np.load(os.path.join(DATA_ROOT, "A",     f"{stem}.npy")).astype(np.float32)
        imgB  = np.load(os.path.join(DATA_ROOT, "B",     f"{stem}.npy")).astype(np.float32)
        label = np.load(os.path.join(DATA_ROOT, "label", f"{stem}.npy")).squeeze().astype(np.uint8)

        # GWR 先验（如果有）
        prior_path = os.path.join(DATA_ROOT, "spatial_prior", f"{stem}.npy")
        if os.path.exists(prior_path):
            prior = np.load(prior_path).astype(np.float32).squeeze()
        else:
            prior = np.zeros((256, 256), dtype=np.float32)

        # 行标签（治理 / 再入侵）
        row_label = "Eradication" if i < 3 else "Re-invasion"
        axes[i, 0].set_ylabel(f"({chr(97+i)}) {row_label}",
                               fontsize=8, fontfamily="Times New Roman",
                               rotation=90, va="center", labelpad=6)

        col = 0
        # T1 影像
        axes[i, col].imshow(make_rgb(imgA), interpolation="nearest")
        axes[i, col].axis("off"); col += 1

        # T2 影像
        axes[i, col].imshow(make_rgb(imgB), interpolation="nearest")
        axes[i, col].axis("off"); col += 1

        # Ground truth
        axes[i, col].imshow(make_gt_display(label), interpolation="nearest")
        axes[i, col].axis("off"); col += 1

        # 各模型预测误差图
        for mname, model in models.items():
            pred = infer_patch(model, mname, imgA, imgB, prior, device)
            axes[i, col].imshow(make_error_map(pred, label), interpolation="nearest")
            axes[i, col].axis("off"); col += 1

    # 图例
    legend_patches = [
        mpatches.Patch(color="white",        label="TP (correct change)",     ec="gray", lw=0.5),
        mpatches.Patch(color="black",        label="TN (correct no-change)"),
        mpatches.Patch(color="red",          label="FN (missed change)"),
        mpatches.Patch(facecolor=(0, 0.8, 0), label="FP (spurious change)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=4,
        fontsize=7.5, frameon=True,
        prop={"family": "Times New Roman", "size": 7.5},
        bbox_to_anchor=(0.52, 0.01)
    )

    plt.savefig(f"{OUT_PATH}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT_PATH}.png", dpi=300, bbox_inches="tight")
    print(f"已保存 {OUT_PATH}.pdf / .png")


if __name__ == "__main__":
    main()
