"""
train.py — Spartina Change Detection Trainer
======================================================================
支持模型：
  基线（无先验）  : SNUNet | FCSiamDiff | BiT | ChangeFormer
  输出端先验注入  : SNUNet_GeoAware
  特征层静态先验  : BiT_GWR | BiT_GWDA
  特征层在线先验  : BiT_Online

BiT_Online 使用说明
----------------------------------------------------------------------
BiT_Online 是纯在线先验模型，先验完全由网络从 T1 影像实时估计，
不读取任何静态先验文件。在论文消融实验中对应两个条目：

  条目 1  BiT_GWR_Online
          将 data_root 下 spatial_prior/ 替换为 GWR 先验图后训练
          （模型本身不使用该文件，但目录名作为实验记录依据）
          运行命令：
            python train.py --model BiT_Online --prior_tag GWR_Online

  条目 2  BiT_GWDA_Online
          将 data_root 下 spatial_prior/ 替换为 GWDA 先验图后训练
          运行命令：
            python train.py --model BiT_Online --prior_tag GWDA_Online

  --prior_tag 仅影响 checkpoints 文件夹命名，不影响训练逻辑。

======================================================================
用法示例：
  python train.py --model BiT_GWR   --lr 6e-5 --epochs 200
  python train.py --model BiT_GWDA  --lr 6e-5 --epochs 200
  python train.py --model BiT_Online --prior_tag GWR_Online  --lr 6e-5 --epochs 200
  python train.py --model BiT_Online --prior_tag GWDA_Online --lr 6e-5 --epochs 200
  python train.py --model ChangeFormer --epochs 200
======================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset               import CDDataset
from models.snunet         import SNUNet, SNUNet_GeoAware
from models.FC_Siam_diff   import FCSiamDiff_Aligned
from models.bit            import BiT
from models.changeformer   import ChangeFormer
from models.bit_gwr        import BiT_GWR
from models.bit_gwda       import BiT_GWDA
from models.bit_online     import BiT_Online
from losses                import BCEHybridLoss
from utils                 import MetricTracker

# ── 需要把 spatial_prior 传入 forward() 的模型 ─────────────────────────
# BiT_Online 虽然在此集合中，但其 forward() 会忽略 spatial_prior 参数
PRIOR_MODELS = {
    'SNUNet_GeoAware',
    'BiT_GWR',
    'BiT_GWDA',
    'BiT_Online',
}

ALL_MODELS = [
    'SNUNet', 'SNUNet_GeoAware',
    'FCSiamDiff',
    'BiT', 'ChangeFormer',
    'BiT_GWR', 'BiT_GWDA',
    'BiT_Online',
]


# ──────────────────────────────────────────────────────────────────────
#  命令行参数
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Change Detection Trainer")
    parser.add_argument('--model',      type=str, required=True,
                        choices=ALL_MODELS)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--data_root',  type=str,
                        default=r"D:/yoyu/SA_Identification/"
                                r"dataset_patches_2020_2024")
    parser.add_argument(
        '--prior_dir', type=str, default=None,
        help="明确指定先验文件夹名称，例如 spatial_prior_gwr 或 spatial_prior_gwda。"
             "不填则自动查找。"
    )
    parser.add_argument(
        '--prior_tag', type=str, default='',
        help="仅用于 BiT_Online 的 checkpoint 文件夹命名，不影响训练逻辑。"
    )
    parser.add_argument(
        '--alpha', type=float, default=0.0,
        help="GWDA 知识蒸馏损失权重（仅 BiT_Online 有效）。"
             "0.0 = 不使用蒸馏（默认）；建议从 0.1 开始试。"
             "需要同时指定 --prior_dir spatial_prior_gwda。"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  模型工厂
# ──────────────────────────────────────────────────────────────────────
def build_model(name: str, device) -> torch.nn.Module:
    kw = dict(in_channels=8, num_classes=1)
    mapping = {
        'SNUNet':          lambda: SNUNet(**kw),
        'SNUNet_GeoAware': lambda: SNUNet_GeoAware(**kw),
        'FCSiamDiff':      lambda: FCSiamDiff_Aligned(
                               in_channels=8, num_classes=1, base_c=32),
        'BiT':             lambda: BiT(**kw),
        'ChangeFormer':    lambda: ChangeFormer(**kw),
        'BiT_GWR':         lambda: BiT_GWR(**kw),
        'BiT_GWDA':        lambda: BiT_GWDA(**kw),
        'BiT_Online':      lambda: BiT_Online(**kw),
    }
    return mapping[name]().to(device)


# ──────────────────────────────────────────────────────────────────────
#  优化器工厂
#
#  学习率分配：
#    Backbone / Transformer      →  命令行 --lr
#    在线先验编码器 prior_encoder →  5e-5
#    SPG 门控参数 spg1 / spg2    →  1e-5（保守更新）
# ──────────────────────────────────────────────────────────────────────
def _ids(*modules) -> set:
    return {id(p) for m in modules for p in m.parameters()}


def build_optimizer(model, name: str, lr: float) -> optim.Optimizer:

    # 静态先验模型：仅 SPG 差异化学习率
    if name in ('BiT_GWR', 'BiT_GWDA', 'SNUNet_GeoAware'):
        excl = _ids(model.spg1, model.spg2)
        base = [p for p in model.parameters() if id(p) not in excl]
        return optim.AdamW([
            {'params': base,                            'lr': lr  },
            {'params': list(model.spg1.parameters()),   'lr': 1e-5},
            {'params': list(model.spg2.parameters()),   'lr': 1e-5},
        ], weight_decay=1e-3)

    # 在线先验模型：prior_encoder + SPG 差异化学习率
    if name == 'BiT_Online':
        excl = _ids(model.prior_encoder, model.spg1, model.spg2)
        base = [p for p in model.parameters() if id(p) not in excl]
        return optim.AdamW([
            {'params': base,                                    'lr': lr  },
            {'params': list(model.prior_encoder.parameters()),  'lr': 5e-5},
            {'params': list(model.spg1.parameters()),           'lr': 1e-5},
            {'params': list(model.spg2.parameters()),           'lr': 1e-5},
        ], weight_decay=1e-3)

    # ChangeFormer：原论文建议较大初始 lr
    if name == 'ChangeFormer':
        return optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # 其余（SNUNet / FCSiamDiff / BiT）
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)


# ──────────────────────────────────────────────────────────────────────
#  调度器工厂
# ──────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, name: str, epochs: int):
    if name == 'ChangeFormer':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5)


# ──────────────────────────────────────────────────────────────────────
#  单 epoch 训练 / 验证
# ──────────────────────────────────────────────────────────────────────
def train_one_epoch(model, name, loader, criterion, optimizer, device,
                    alpha: float = 0.0):
    """
    alpha > 0 且模型为 BiT_Online 时，启用 GWDA 知识蒸馏：
      L_total = L_cd + alpha * L_distill
      L_distill = MSE(prior_online, prior_gwda)
    prior_gwda 来自 dataset 返回的第四个元素（需指定 --prior_dir spatial_prior_gwda）。
    """
    model.train()
    total      = 0.0
    use_distil = (alpha > 0.0 and name == 'BiT_Online')
    bar        = tqdm(loader, desc="Train", leave=False)

    for imgA, imgB, label, prior in bar:
        imgA  = imgA.to(device)
        imgB  = imgB.to(device)
        label = label.to(device)
        prior = prior.to(device)

        optimizer.zero_grad()
        out  = model(imgA, imgB, prior) if name in PRIOR_MODELS \
               else model(imgA, imgB)
        loss = sum(criterion(o, label) for o in out) \
               if isinstance(out, list) else criterion(out, label)

        # ── GWDA 知识蒸馏损失 ─────────────────────────────────────────
        if use_distil:
            # prior_encoder 的输出就是在线先验图
            prior_online = model.prior_encoder(imgA)    # [B, 1, H, W]
            # prior 是 dataset 读进来的 GWDA 先验图，作为蒸馏目标
            loss_distil  = torch.nn.functional.mse_loss(
                prior_online, prior)
            loss = loss + alpha * loss_distil

        loss.backward()
        optimizer.step()

        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total / len(loader)


@torch.no_grad()
def validate(model, name, loader, device, tracker):
    model.eval()
    tracker.reset()

    for imgA, imgB, label, prior in tqdm(loader, desc="Val  ", leave=False):
        imgA  = imgA.to(device)
        imgB  = imgB.to(device)
        label = label.to(device)
        prior = prior.to(device)

        out = model(imgA, imgB, prior) if name in PRIOR_MODELS \
              else model(imgA, imgB)
        if isinstance(out, list):
            out = out[-1]
        tracker.update(out, label)

    return tracker.get_metrics()


def get_gamma(model):
    """读取 SPG 门控参数；无 SPG 的模型返回 0.0。"""
    if hasattr(model, 'spg1') and hasattr(model, 'spg2'):
        return model.spg1.gamma.item(), model.spg2.gamma.item()
    return 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint 文件夹命名
    # BiT_Online 时附加 prior_tag 以区分 GWR_Online / GWDA_Online
    tag       = f"_{args.prior_tag}" if args.prior_tag else ""
    timestamp = time.strftime("%m%d_%H%M")
    save_dir  = f"checkpoints_{args.model}{tag}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Model      : {args.model}{tag}")
    print(f"  Device     : {device}")
    print(f"  LR / Epochs: {args.lr:.1e}  /  {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Save dir   : {save_dir}")
    print("=" * 60)

    # 数据集
    train_ds = CDDataset(args.data_root, split='train',
                         split_ratio=0.85, transform=True,
                         prior_dir_name=args.prior_dir)
    val_ds   = CDDataset(args.data_root, split='val',
                         split_ratio=0.85, transform=False,
                         prior_dir_name=args.prior_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True)

    # 构建模型 / 优化器 / 调度器
    model     = build_model(args.model, device)
    optimizer = build_optimizer(model, args.model, args.lr)
    scheduler = build_scheduler(optimizer, args.model, args.epochs)
    criterion = BCEHybridLoss()
    tracker   = MetricTracker()

    # CSV 日志
    log_path = os.path.join(save_dir, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss',
            'val_iou', 'val_f1', 'val_prec', 'val_rec',
            'lr', 'gamma1', 'gamma2',
        ])

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model, args.model, train_loader, criterion, optimizer, device,
            alpha=args.alpha)
        metrics    = validate(
            model, args.model, val_loader, device, tracker)
        scheduler.step()

        lr             = scheduler.get_last_lr()[0]
        gamma1, gamma2 = get_gamma(model)

        g = f"  g={gamma1:.4f}/{gamma2:.4f}" if (gamma1 or gamma2) else ""
        print(f"[{epoch:3d}/{args.epochs}]  "
              f"loss={train_loss:.4f}  "
              f"F1={metrics['F1']:.4f}  IoU={metrics['IoU']:.4f}  "
              f"P={metrics['Precision']:.4f}  R={metrics['Recall']:.4f}  "
              f"lr={lr:.1e}{g}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{metrics['IoU']:.6f}",
                f"{metrics['F1']:.6f}",
                f"{metrics['Precision']:.6f}",
                f"{metrics['Recall']:.6f}",
                f"{lr:.2e}",
                f"{gamma1:.6f}",
                f"{gamma2:.6f}",
            ])

        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model.pth"))
            print(f"          ↑ best saved  (F1={best_f1:.4f})")

        if epoch % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"epoch_{epoch}.pth"))

    # 训练结束汇总
    summary = {
        'model':      args.model + tag,
        'best_f1':    round(best_f1, 6),
        'epochs':     args.epochs,
        'lr':         args.lr,
        'batch_size': args.batch_size,
        'data_root':  args.data_root,
        'prior_tag':  args.prior_tag,
        'alpha':      args.alpha,
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Done : {args.model}{tag}  |  Best F1 = {best_f1:.4f}")
    print(f"  CSV  : {log_path}")
    print(f"  JSON : {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()