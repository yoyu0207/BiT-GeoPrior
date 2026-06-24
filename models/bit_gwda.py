"""
bit_gwda.py — BiT + 静态 GWDA 先验 + SPG
与 bit_gwr.py 结构完全相同，区别仅在于运行时传入的
spatial_prior 文件使用 GWDA 后验概率而非 GWR local-R²。

先验图来源：gwda_cell1.py 生成的 uncertainty_weight 列
            = P(Spartina=1 | X, location)，值域 [0,1]
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from models.transformer_block import TransformerBlock
from models.SPGmodule          import SpatialPriorGate


class BiT_GWDA(nn.Module):

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

    def forward(self, x1, x2, spatial_prior):
        """
        spatial_prior: GWDA 后验概率切片 [B, 1, H, W]
                       来源：gwda_cell1.py → 栅格化 → 切片
        """
        f1 = self.spg1(self._encode(x1), spatial_prior)
        f2 = self.spg2(self._encode(x2), spatial_prior)
        return self.decoder(torch.cat([f1, f2], dim=1))
