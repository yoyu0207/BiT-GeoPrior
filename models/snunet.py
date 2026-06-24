import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from models.SPGmodule import SpatialPriorGate

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SNUNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=1, base_c=32):
        super().__init__()
        
        # Encoder 
        self.conv1 = ConvBlock(in_channels, base_c)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(base_c, base_c*2)
        self.conv3 = ConvBlock(base_c*2, base_c*4)
        self.conv4 = ConvBlock(base_c*4, base_c*8)
        self.conv5 = ConvBlock(base_c*8, base_c*16)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        # L2
        self.conv1_2 = ConvBlock(base_c*2 + base_c, base_c)
        self.conv2_2 = ConvBlock(base_c*4 + base_c*2, base_c*2)
        self.conv3_2 = ConvBlock(base_c*8 + base_c*4, base_c*4)
        self.conv4_2 = ConvBlock(base_c*16 + base_c*8, base_c*8)

        # L3
        self.conv1_3 = ConvBlock(base_c*2 + base_c*2, base_c)
        self.conv2_3 = ConvBlock(base_c*4 + base_c*4, base_c*2)
        self.conv3_3 = ConvBlock(base_c*8 + base_c*8, base_c*4)

        # L4
        self.conv1_4 = ConvBlock(base_c*2 + base_c*3, base_c)
        self.conv2_4 = ConvBlock(base_c*4 + base_c*6, base_c*2)
        
        # L5 (Output)
        self.conv1_5 = ConvBlock(base_c*2 + base_c*4, base_c)

        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        # 1. Encoding (Siamese)
        f1 = self._encoder(x1) # [x1_1, x1_2, x1_3, x1_4, x1_5]
        f2 = self._encoder(x2)
        
        # 2. Difference Calculation (特征差分)
        # 取绝对值差，用于捕捉变化
        diffs = [torch.abs(f1[i] - f2[i]) for i in range(5)]
        
        # 3. Decoding (Nested)
        x1_1, x2_1, x3_1, x4_1, x5_1 = diffs

        # Level 2
        x1_2 = self.conv1_2(torch.cat([x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_1, self.up(x3_1)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_1, self.up(x4_1)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_1, self.up(x5_1)], 1))

        # Level 3
        x1_3 = self.conv1_3(torch.cat([x1_1, x1_2, self.up(x2_2)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_1, x2_2, self.up(x3_2)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_1, x3_2, self.up(x4_2)], 1))

        # Level 4
        x1_4 = self.conv1_4(torch.cat([x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_1, x2_2, x2_3, self.up(x3_3)], 1))

        # Level 5
        x1_5 = self.conv1_5(torch.cat([x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))

        return self.out_conv(x1_5)

    def _encoder(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        return [x1, x2, x3, x4, x5]
    


# ==========================================
# 3. 融合后的 SNUNet
# ==========================================
class SNUNet_GeoAware(nn.Module):
    def __init__(self, in_channels=8, num_classes=1, base_c=32):
        super().__init__()
        
        # Encoder 
        self.conv1 = ConvBlock(in_channels, base_c)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(base_c, base_c*2)
        self.conv3 = ConvBlock(base_c*2, base_c*4)
        self.conv4 = ConvBlock(base_c*4, base_c*8)
        self.conv5 = ConvBlock(base_c*8, base_c*16)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ================================
        # 【新增】：声明两层地学先验门控
        # 第一层：原分辨率，通道数 base_c
        # 第二层：1/2 分辨率，通道数 base_c*2
        # ================================
        self.spg1 = SpatialPriorGate(in_channels=base_c)
        self.spg2 = SpatialPriorGate(in_channels=base_c*2)

        # Decoder (保持你的密集连接结构完全不变)
        # L2
        self.conv1_2 = ConvBlock(base_c*2 + base_c, base_c)
        self.conv2_2 = ConvBlock(base_c*4 + base_c*2, base_c*2)
        self.conv3_2 = ConvBlock(base_c*8 + base_c*4, base_c*4)
        self.conv4_2 = ConvBlock(base_c*16 + base_c*8, base_c*8)

        # L3
        self.conv1_3 = ConvBlock(base_c*2 + base_c*2, base_c)
        self.conv2_3 = ConvBlock(base_c*4 + base_c*4, base_c*2)
        self.conv3_3 = ConvBlock(base_c*8 + base_c*8, base_c*4)

        # L4
        self.conv1_4 = ConvBlock(base_c*2 + base_c*3, base_c)
        self.conv2_4 = ConvBlock(base_c*4 + base_c*6, base_c*2)
        
        # L5 (Output)
        self.conv1_5 = ConvBlock(base_c*2 + base_c*4, base_c)

        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    # ================================
    # 【修改】：forward 函数增加了 spatial_prior 输入
    # ================================
    def forward(self, x1, x2, spatial_prior):
        # 1. Encoding (Siamese)
        f1 = self._encoder(x1) # [x1_1, x1_2, x1_3, x1_4, x1_5]
        f2 = self._encoder(x2)
        
        # 2. Difference Calculation (特征差分)
        # diffs 包含了 5 个尺度的差异特征
        diffs = [torch.abs(f1[i] - f2[i]) for i in range(5)]
        x1_1, x2_1, x3_1, x4_1, x5_1 = diffs

        # ================================
        # 【核心注入】：用地学先验调制浅层的差异特征！
        # 为什么要在这里调制？因为浅层最容易在复杂区域产生碎片和虚警。
        # ================================
        x1_1_geo = self.spg1(x1_1, spatial_prior)
        x2_1_geo = self.spg2(x2_1, spatial_prior)
        
        # 剩下的 x3_1, x4_1, x5_1 属于深层语义特征，分辨率太低，不需要调制了
        
        # 3. Decoding (Nested)
        # 注意：把原来的 x1_1, x2_1 替换为你调制后的 x1_1_geo, x2_1_geo

        # Level 2
        x1_2 = self.conv1_2(torch.cat([x1_1_geo, self.up(x2_1_geo)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_1_geo, self.up(x3_1)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_1, self.up(x4_1)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_1, self.up(x5_1)], 1))

        # Level 3
        x1_3 = self.conv1_3(torch.cat([x1_1_geo, x1_2, self.up(x2_2)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_1_geo, x2_2, self.up(x3_2)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_1, x3_2, self.up(x4_2)], 1))

        # Level 4
        x1_4 = self.conv1_4(torch.cat([x1_1_geo, x1_2, x1_3, self.up(x2_3)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_1_geo, x2_2, x2_3, self.up(x3_3)], 1))

        # Level 5
        x1_5 = self.conv1_5(torch.cat([x1_1_geo, x1_2, x1_3, x1_4, self.up(x2_4)], 1))

        # Output
        return self.out_conv(x1_5)

    def _encoder(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        return [x1, x2, x3, x4, x5]