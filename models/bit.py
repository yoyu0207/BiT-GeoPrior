import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.atn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.atn(x, x, x)
        x = self.norm1(attn_out + x)
        x = self.norm2(self.mlp(x) + x)
        return x

class BiT(nn.Module):
    def __init__(self, in_channels=8, num_classes=1, base_c=64):
        super().__init__()
        
        # 1. Backbone: 使用 ResNet18 作为特征提取器
        resnet = models.resnet18(pretrained=False)
        # 修改第一层卷积以适配 8 通道输入
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, # 1/4 size
            resnet.layer2, # 1/8 size
            resnet.layer3  # 1/16 size, 256 channels
        )

        # 2. Transformer 部分
        # 将特征投影到 embedding 维度 (例如 128)
        self.embed_dim = 128
        self.project = nn.Conv2d(256, self.embed_dim, kernel_size=1)
        
        # Transformer Encoder 层 (处理语义建模)
        self.transformer = TransformerBlock(dim=self.embed_dim, heads=8, dim_head=64, mlp_dim=256)

        # 3. Decoder: 还原尺寸
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x1, x2):
        # --- Encoder ---
        f1 = self.backbone(x1) # [B, 256, H/16, W/16]
        f2 = self.backbone(x2)
        
        # Project & Flatten
        b, c, h, w = f1.shape
        f1 = self.project(f1).flatten(2).transpose(1, 2) # [B, L, 128]
        f2 = self.project(f2).flatten(2).transpose(1, 2)
        
        # --- Transformer Processing ---
        f1 = self.transformer(f1)
        f2 = self.transformer(f2)
        
        # --- Reshape back ---
        f1 = f1.transpose(1, 2).reshape(b, self.embed_dim, h, w)
        f2 = f2.transpose(1, 2).reshape(b, self.embed_dim, h, w)
        
        # --- Difference & Upsample ---
        # BiT 常用做法是将两个特征 concat 然后还原
        out = torch.cat([f1, f2], dim=1)
        out = self.upsample(out) # 回到 256x256
        return self.final_conv(out)