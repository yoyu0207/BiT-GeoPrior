import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=256, embed_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ChangeFormer(nn.Module):
    def __init__(self, in_channels=8, num_classes=1, embed_dim=128):
        super().__init__()
        
        # 1. Encoder: 简化版的分层 Transformer (4个阶段)
        # Stage 1: 1/4 size
        self.patch_embed1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3)
        self.norm1 = nn.LayerNorm(32)
        
        # Stage 2: 1/8 size
        self.patch_embed2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(64)
        
        # Stage 3: 1/16 size
        self.patch_embed3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.LayerNorm(128)
        
        # Stage 4: 1/32 size
        self.patch_embed4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.LayerNorm(256)

        # 2. MLP Decoder (将 4 个阶段的特征融合)
        self.mlp1 = MLP(input_dim=32,  embed_dim=embed_dim)
        self.mlp2 = MLP(input_dim=64,  embed_dim=embed_dim)
        self.mlp3 = MLP(input_dim=128, embed_dim=embed_dim)
        self.mlp4 = MLP(input_dim=256, embed_dim=embed_dim)
        
        # 3. Final Fusion & Head
        # 输入是 2个时相 * 4个阶段融合后的特征
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def _forward_encoder(self, x):
        # 模拟分层提取特征
        c1 = self.patch_embed1(x)
        c2 = self.patch_embed2(c1)
        c3 = self.patch_embed3(c2)
        c4 = self.patch_embed4(c3)
        return c1, c2, c3, c4

    def forward(self, x1, x2):
        size = x1.size()[2:]
        
        # --- Siamese Encoder ---
        out1 = self._forward_encoder(x1)
        out2 = self._forward_encoder(x2)
        
        # --- Feature Difference & MLP Fusion ---
        # 对每一个尺度求差值并投影到相同维度
        def get_mlp_out(o1, o2, mlp_layer, target_size):
            diff = torch.abs(o1 - o2)
            b, c, h, w = diff.shape
            # 投影后还原为 2D 特征图
            out = mlp_layer(diff).transpose(1, 2).reshape(b, -1, h, w)
            return F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)

        target_size = out1[0].size()[2:] # 以第一层 (1/4) 为基准
        x1_f = get_mlp_out(out1[0], out2[0], self.mlp1, target_size)
        x2_f = get_mlp_out(out1[1], out2[1], self.mlp2, target_size)
        x3_f = get_mlp_out(out1[2], out2[2], self.mlp3, target_size)
        x4_f = get_mlp_out(out1[3], out2[3], self.mlp4, target_size)
        
        # --- Final Concatenation & Prediction ---
        out = torch.cat([x1_f, x2_f, x3_f, x4_f], dim=1)
        out = self.linear_fuse(out)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True) # 还原回 256x256
        
        return self.final_conv(out)