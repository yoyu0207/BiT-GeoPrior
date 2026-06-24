"""
bit_online.py — BiT + 纯在线生态先验（EcologicalPriorEncoder）+ SPG

先验完全由网络从 T1 影像实时估计，不依赖任何预计算的静态先验文件。
运行时 spatial_prior 参数传 None 即可，dataset.py 返回全零占位不影响结果。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用场景说明（在 train.py 的注释中也有对应标注）：

  BiT_Online + GWR 先验图目录   →  对应论文中 "BiT_GWR_Online" 条目
                                    消融：在线先验 vs 静态 GWR 先验

  BiT_Online + GWDA 先验图目录  →  对应论文中 "BiT_GWDA_Online" 条目
                                    消融：在线先验 vs 静态 GWDA 先验

两次训练使用同一份代码，区别只在于 data_root 下
spatial_prior/ 文件夹里放的是 GWR 还是 GWDA 先验图。
由于模型是纯在线的，spatial_prior 文件夹有没有都不影响前向推理，
仅在消融分析叙事中区分两个对比组。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from models.transformer_block  import TransformerBlock
from models.SPGmodule           import SpatialPriorGate
from models.ecological_prior    import EcologicalPriorEncoder


class BiT_Online(nn.Module):

    def __init__(self, in_channels: int = 8, num_classes: int = 1):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────
        base = resnet18(weights=None)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3,
        )

        # ── Transformer ───────────────────────────────────────────────
        self.embed_dim   = 128
        self.project     = nn.Conv2d(256, self.embed_dim, kernel_size=1)
        self.transformer = TransformerBlock(
            dim=self.embed_dim, heads=8, mlp_dim=256)

        # ── 在线生态先验编码器（从 T1 实时估计先验）─────────────────
        self.prior_encoder = EcologicalPriorEncoder(
            in_channels=in_channels)

        # ── SPG ───────────────────────────────────────────────────────
        self.spg1 = SpatialPriorGate(self.embed_dim)
        self.spg2 = SpatialPriorGate(self.embed_dim)

        # ── Decoder ──────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(self.embed_dim * 2, 64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def _encode(self, x):
        f = self.backbone(x)
        b, _, h, w = f.shape
        f = self.project(f).flatten(2).transpose(1, 2)
        f = self.transformer(f)
        return f.transpose(1, 2).reshape(b, self.embed_dim, h, w)

    def forward(self, x1, x2, spatial_prior=None):
        """
        Args:
            x1 / x2:       双时相影像  [B, C, H, W]
            spatial_prior: 忽略，保留参数接口与其他模型一致
                           （dataset.py 返回全零占位，不影响推理）
        """
        prior = self.prior_encoder(x1)          # [B, 1, H, W]
        f1    = self.spg1(self._encode(x1), prior)
        f2    = self.spg2(self._encode(x2), prior)
        return self.decoder(torch.cat([f1, f2], dim=1))
