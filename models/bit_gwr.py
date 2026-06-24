import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.SPGmodule import SpatialPriorGate


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.atn   = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.atn(x, x, x)
        x = self.norm1(attn_out + x)
        x = self.norm2(self.mlp(x) + x)
        return x


class BiT_GWR(nn.Module):
    """
    BiT + GWR 空间先验注入版本。

    相比原始 BiT 唯一的结构差异：
      在 Transformer 输出 reshape 回空间特征图之后，
      用两个 SpatialPriorGate（spg1, spg2）分别对
      T1 特征图 f1 和 T2 特征图 f2 做先验调制，
      再送入 decoder。

    - spg1 / spg2 的 gamma 均初始化为 0，
      训练初期行为与原始 BiT 完全一致
    - forward 接口与 SNUNet_GeoAware 相同：
        output = model(imgA, imgB, spatial_prior)
      在 train.py 中只需把 BiT_GWR 加入 PRIOR_MODELS 集合即可
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 1,
                 base_c: int = 64):
        super().__init__()

        # ── Backbone（与 bit.py 完全相同）──────────────────────────────
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,   # 1/4,  64ch
            resnet.layer2,   # 1/8,  128ch
            resnet.layer3,   # 1/16, 256ch
        )

        # ── Transformer（与 bit.py 完全相同）──────────────────────────
        self.embed_dim  = 128
        self.project    = nn.Conv2d(256, self.embed_dim, kernel_size=1)
        self.transformer = TransformerBlock(
            dim=self.embed_dim, heads=8, dim_head=64, mlp_dim=256
        )

        # ── GWR 先验注入模块 ───────────────────────────────────────────
        # 注入位置：Transformer 输出 reshape 回 [B, 128, H, W] 之后
        # 用两个独立的 SPG 分别调制 T1 / T2 特征图
        # （共享一个 SPG 也可以，但独立参数给模型更多自由度）
        self.spg1 = SpatialPriorGate(in_channels=self.embed_dim)
        self.spg2 = SpatialPriorGate(in_channels=self.embed_dim)

        # ── Decoder（与 bit.py 完全相同）──────────────────────────────
        self.upsample   = nn.Upsample(scale_factor=16, mode='bilinear',
                                       align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                spatial_prior: torch.Tensor) -> torch.Tensor:
        """
        x1, x2        : [B, C, H, W]   双时相影像
        spatial_prior : [B, 1, H, W]   GWR 空间先验（值域 [0,1]）
        return        : [B, num_classes, H, W]
        """
        # Backbone
        f1 = self.backbone(x1)          # [B, 256, h, w]，h=w=16
        f2 = self.backbone(x2)
        b, c, h, w = f1.shape

        # Project & Flatten
        f1 = self.project(f1).flatten(2).transpose(1, 2)   # [B, L, 128]
        f2 = self.project(f2).flatten(2).transpose(1, 2)

        # Transformer
        f1 = self.transformer(f1)
        f2 = self.transformer(f2)

        # Reshape 回空间特征图
        f1 = f1.transpose(1, 2).reshape(b, self.embed_dim, h, w)   # [B,128,h,w]
        f2 = f2.transpose(1, 2).reshape(b, self.embed_dim, h, w)

        # ── GWR 先验注入（SPG 内部自动对齐尺寸）──────────────────────
        # spatial_prior: [B,1,256,256] → SPG 内插值到 [B,1,h,w]=[B,1,16,16]
        f1 = self.spg1(f1, spatial_prior)
        f2 = self.spg2(f2, spatial_prior)

        # Decoder
        out = torch.cat([f1, f2], dim=1)    # [B, 256, h, w]
        out = self.upsample(out)             # [B, 256, 256, 256]
        return self.final_conv(out)          # [B, 1, 256, 256]
