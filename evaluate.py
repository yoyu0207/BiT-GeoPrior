"""
evaluate.py  ── 加载已有 pth，在验证集上重新计算指标
用法:
  python evaluate.py --model BiT_GWR --pth "D:/yoyu/SA_Identification/project/checkpoints_BiT_GWR_0404_1635/best_model.pth"
  python evaluate.py --model BiT     --pth "checkpoints_BiT_xxx/best_model.pth"
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CDDataset
from models.snunet import SNUNet, SNUNet_GeoAware
from models.FC_Siam_diff import FCSiamDiff_Aligned
from models.bit import BiT
from models.bit_gwr import BiT_GWR
from models.changeformer import ChangeFormer
from utils import MetricTracker

PRIOR_MODELS = {'SNUNet_GeoAware', 'BiT_GWR'}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['SNUNet_GeoAware', 'SNUNet', 'FCSiamDiff',
                                 'BiT', 'ChangeFormer', 'BiT_GWR'])
    parser.add_argument('--pth', type=str, required=True,
                        help='best_model.pth 的完整路径')
    parser.add_argument('--data_root', type=str,
                        default=r"D:/yoyu/SA_Identification/dataset_patches_2020_2024")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--split_ratio', type=float, default=0.85)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 数据集（与训练时相同的划分）──────────────────────────────────────
    val_ds = CDDataset(args.data_root, split='val',
                       split_ratio=args.split_ratio, transform=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"验证集样本数: {len(val_ds)}")

    # ── 模型 ──────────────────────────────────────────────────────────────
    if args.model == 'SNUNet_GeoAware':
        model = SNUNet_GeoAware(in_channels=8, num_classes=1)
    elif args.model == 'SNUNet':
        model = SNUNet(in_channels=8, num_classes=1)
    elif args.model == 'FCSiamDiff':
        model = FCSiamDiff_Aligned(in_channels=8, num_classes=1, base_c=32)
    elif args.model == 'BiT':
        model = BiT(in_channels=8, num_classes=1)
    elif args.model == 'BiT_GWR':
        model = BiT_GWR(in_channels=8, num_classes=1)
    elif args.model == 'ChangeFormer':
        model = ChangeFormer(in_channels=8, num_classes=1)

    # 加载权重
    state_dict = torch.load(args.pth, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"权重加载成功: {args.pth}")

    # BiT_GWR 的 gamma 值
    if hasattr(model, 'spg1') and hasattr(model, 'spg2'):
        print(f"gamma1={model.spg1.gamma.item():.6f}  "
              f"gamma2={model.spg2.gamma.item():.6f}")

    # ── 推理 ──────────────────────────────────────────────────────────────
    tracker = MetricTracker()
    tracker.reset()

    with torch.no_grad():
        for imgA, imgB, label, spatial_prior in tqdm(val_dl, desc="Evaluating"):
            imgA          = imgA.to(device)
            imgB          = imgB.to(device)
            label         = label.to(device)
            spatial_prior = spatial_prior.to(device)

            if args.model in PRIOR_MODELS:
                output = model(imgA, imgB, spatial_prior)
            else:
                output = model(imgA, imgB)

            if isinstance(output, list):
                output = output[-1]

            tracker.update(output, label)

    # ── 输出指标 ──────────────────────────────────────────────────────────
    metrics = tracker.get_metrics()
    print("\n" + "=" * 45)
    print(f"  Model : {args.model}")
    print(f"  Pth   : {os.path.basename(args.pth)}")
    print("-" * 45)
    print(f"  IoU       : {metrics['IoU']:.4f}")
    print(f"  F1        : {metrics['F1']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    main()