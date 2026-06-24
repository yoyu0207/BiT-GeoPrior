import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 BCEWithLogitsLoss，它是最稳定的二分类损失
        # pos_weight=2.0 稍微侧重于正样本(变化区域)，防止背景太多导致模型躺平
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0])) 

    def forward(self, inputs, targets):
        # 确保 pos_weight 在正确的设备上 (GPU/CPU)
        if self.bce.pos_weight.device != inputs.device:
            self.bce.pos_weight = self.bce.pos_weight.to(inputs.device)
            
        loss = self.bce(inputs, targets)
        return loss


class IPW_BCELoss(nn.Module):
    def __init__(self, clip_max=10.0):
        super(IPW_BCELoss, self).__init__()
        self.clip_max = clip_max # 防止权重过大导致梯度爆炸
        self.bce = nn.BCELoss(reduction='none') # 注意这里一定要选 none

    def forward(self, pred, target, prob_map):
        """
        pred: SNUNet输出的预测图 (B, 1, H, W)
        target: 真实的治理事件标签 (B, 1, H, W)
        prob_map: 提前算好的该区域样本出现概率图 (B, 1, H, W)
        """
        # 1. 计算逆概率权重
        weight_map = 1.0 / (prob_map + 1e-5)
        weight_map = torch.clamp(weight_map, max=self.clip_max) 
        
        # 2. 计算基础Loss
        base_loss = self.bce(pred, target)
        
        # 3. 施加空间权重
        weighted_loss = base_loss * weight_map
        
        return weighted_loss.mean()
    

class EdgeAwareBCELoss(nn.Module):
    def __init__(self):
        super(EdgeAwareBCELoss, self).__init__()
        # 退回使用最原始的 BCELoss
        self.bce = nn.BCELoss(reduction='none') 

    def forward(self, pred, target, weight_map):
        """
        pred: SNUNet 已经经过 Sigmoid 的概率图 (B, 1, H, W)
        """
        # 【核心杀招：防止 CUDA 报错】
        # 强制把所有预测值限制在 0.000001 到 0.999999 之间
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        
        # 1. 计算基础 Loss
        base_loss = self.bce(pred, target)
        
        # 2. 乘上我们的空间先验权重
        weighted_loss = base_loss * weight_map
        
        # 3. 求平均
        return weighted_loss.mean()

# 结合边缘权重和全局 Dice Loss 的混合损失函数
class EdgeAwareHybridLoss(nn.Module):
    # 【修改】：调低 BCE 的话语权，让 Dice 先主导大局
    def __init__(self, weight_bce=0.5, weight_dice=1.0, dampen_factor=0.2):
        super(EdgeAwareHybridLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none') 
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        # dampen_factor 是"衰减系数"
        # 比如你原来算出的权重最大是 6.0，(6.0 - 1.0) * 0.2 + 1.0 = 2.0
        # 这样就把 6 倍的惩罚温和地降到了 2 倍！
        self.dampen_factor = dampen_factor 

    def forward(self, pred, target, weight_map):
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)

        # ==========================================
        # 0. 动态驯服过激的空间权重
        # ==========================================
        # 确保基础权重是 1.0，只对大于 1.0 的惩罚部分进行衰减
        adjusted_weight = 1.0 + (weight_map - 1.0) * self.dampen_factor

        # ==========================================
        # 1. 空间感知加权 BCE Loss
        # ==========================================
        base_bce = self.bce(pred, target)
        
        # 使用温和化后的权重进行加权平均
        weighted_bce = torch.sum(base_bce * adjusted_weight) / (torch.sum(adjusted_weight) + 1e-8)

        # ==========================================
        # 2. 全局 Dice Loss (极度重要：维持 Precision 的核心)
        # ==========================================
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - ((2. * intersection + smooth) / 
                           (pred_flat.sum() + target_flat.sum() + smooth))

        # ==========================================
        # 3. 最终损失融合
        # ==========================================
        total_loss = self.weight_bce * weighted_bce + self.weight_dice * dice_loss
        
        return total_loss

class BCEHybridLoss(nn.Module):
    """
    真正的混合损失函数 (BCE + Dice Loss)
    BCE 负责像素级别的稳定分类，Dice 负责全局形状的贴合和 F1 的直接提升。
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEHybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # 恢复公平的权重，因为模型内部已经有了地理先验门控的辅助
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, inputs, targets):
        # 1. 计算 BCE Loss
        bce_loss = self.bce(inputs, targets)

        # 2. 计算 Dice Loss
        # 先将 logits 通过 sigmoid 转为 0~1 的概率
        probs = torch.sigmoid(inputs)
        
        # 展平 tensor 以计算交集和并集
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        # 加 1e-6 防止除以 0
        dice_loss = 1.0 - (2. * intersection + 1e-6) / (union + 1e-6)

        # 3. 混合返回
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss