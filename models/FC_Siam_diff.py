import torch
import torch.nn as nn
from models.snunet import ConvBlock # 确保引用你 SNUNet 里定义的同一个 Block

class FCSiamDiff_Aligned(nn.Module):
    def __init__(self, in_channels=8, num_classes=1, base_c=32):
        super().__init__()
        
        # 1. Encoder (与你的 SNUNet Encoder 完全对齐)
        self.conv1 = ConvBlock(in_channels, base_c)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(base_c, base_c*2)
        self.conv3 = ConvBlock(base_c*2, base_c*4)
        self.conv4 = ConvBlock(base_c*4, base_c*8)
        self.conv5 = ConvBlock(base_c*8, base_c*16)

        # 2. Decoder (标准的 U-Net 结构，非 Nested 结构)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 注意：这里的输入通道是 cat([当前层差值, 上层上采样])
        # 例如 d4: 16*base_c (差值) + 8*base_c (上采样) -> 并不，是差值特征与上采样特征拼接
        self.dec5 = ConvBlock(base_c*16 + base_c*8, base_c*8)
        self.dec4 = ConvBlock(base_c*8 + base_c*4, base_c*4)
        self.dec3 = ConvBlock(base_c*4 + base_c*2, base_c*2)
        self.dec2 = ConvBlock(base_c*2 + base_c, base_c)
        
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        # --- Siamese Encoder ---
        f1_1 = self.conv1(x1)
        f1_2 = self.conv2(self.pool(f1_1))
        f1_3 = self.conv3(self.pool(f1_2))
        f1_4 = self.conv4(self.pool(f1_3))
        f1_5 = self.conv5(self.pool(f1_4))

        f2_1 = self.conv1(x2)
        f2_2 = self.conv2(self.pool(f2_1))
        f2_3 = self.conv3(self.pool(f2_2))
        f2_4 = self.conv4(self.pool(f2_3))
        f2_5 = self.conv5(self.pool(f2_4))

        # --- 特征差分 (与你的 SNUNet 逻辑一致) ---
        diff1 = torch.abs(f1_1 - f2_1)
        diff2 = torch.abs(f1_2 - f2_2)
        diff3 = torch.abs(f1_3 - f2_3)
        diff4 = torch.abs(f1_4 - f2_4)
        diff5 = torch.abs(f1_5 - f2_5)

        # --- Standard Decoder (Skip Connection) ---
        # 这里没有 Nested 连接，只有标准的 U-Net 跳跃连接
        d4 = self.dec5(torch.cat([diff4, self.up(diff5)], 1))
        d3 = self.dec4(torch.cat([diff3, self.up(d4)], 1))
        d2 = self.dec3(torch.cat([diff2, self.up(d3)], 1))
        d1 = self.dec2(torch.cat([diff1, self.up(d2)], 1))

        return self.out_conv(d1)