import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def update(self, inputs, targets):
        """
        累加混淆矩阵 (TP, TN, FP, FN)
        inputs: 模型的原始输出 logits
        targets: 0 或 1 的标签
        """
        with torch.no_grad():
            # 1. Sigmoid 激活 (变成 0-1 概率)
            probs = torch.sigmoid(inputs)
            
            # 2. 阈值分割 (大于0.5算变化)
            preds = (probs > 0.5).long()
            targets = targets.long()
            
            # 3. 计算并累加像素数
            self.tp += (preds * targets).sum().item()
            self.tn += ((1 - preds) * (1 - targets)).sum().item()
            self.fp += (preds * (1 - targets)).sum().item()
            self.fn += ((1 - preds) * targets).sum().item()
        
    def get_metrics(self):
        epsilon = 1e-7 # 防止除以0
        
        # 计算全局指标
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        
        # 总体准确率
        total_pixels = self.tp + self.tn + self.fp + self.fn
        oa = (self.tp + self.tn) / (total_pixels + epsilon)
        
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "OA": oa
        }