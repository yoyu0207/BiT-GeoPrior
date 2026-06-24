"""
ecological_prior.py — EcologicalPriorEncoder
从 T1 影像在线估计空间先验图，替代预计算的静态先验文件。

设计两步：
  Step 1  生态响应打分（1×1 Conv）
          对 8 通道输入（B8/B4/B3/B2/NDVI/EVI/SAVI/GNDVI）做可学习
          加权组合，自主学习各光谱/指数对 Spartina 适宜性的贡献权重。
          训练后可通过 get_channel_weights() 读取权重作可解释性分析。

  Step 2  空间扩散平滑（多尺度空洞卷积）
          感受野分别对应 30 m / 90 m / 170 m，模拟潮沟网络驱动的
          生态传播特性，输出空间一致的先验概率图。

输入：T1 影像  [B, C, H, W]（8 通道）
输出：先验图   [B, 1, H, W]，值域 (0, 1)
"""

import torch
import torch.nn as nn


class EcologicalPriorEncoder(nn.Module):

    def __init__(self, in_channels: int = 8):
        super().__init__()

        # Step 1 — 生态响应打分
        self.eco_score = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 2 — 空间扩散平滑（三尺度空洞卷积）
        self.spatial_diffusion = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding=4,
                      dilation=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=8,
                      dilation=8, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eco   = self.eco_score(x)
        prior = self.spatial_diffusion(eco)
        return prior                                # [B, 1, H, W]

    def get_channel_weights(self) -> torch.Tensor:
        """返回 eco_score 的 8 通道权重，用于可解释性分析。"""
        return self.eco_score[0].weight.squeeze().detach()
